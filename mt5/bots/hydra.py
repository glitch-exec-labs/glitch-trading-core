"""
HYDRA v1.1 — Prop Firm Challenge Killer
Account: 143447 (GetLeveraged-Trade)

Multi-strategy hybrid bot optimised for prop firm challenge:
  Working Capital: $100,000
  Profit Target: 6% ($6,000)
  Max Daily Loss: 3% ($3,000)
  Max Trailing Loss: 6% ($6,000)

Dual Mode Architecture:
  STANDARD MODE (H1 primary, M15 confirm):
    1. PRICE_ACTION — Pin Bar + Engulfing at S/R (from Cobra) — Trending, 2:1 R:R
    2. BREAKOUT    — 4/5 condition strict breakout (from Anaconda) — Trending, 2:1 R:R
    3. EMA_PULLBACK — EMA-20 bounce in trend (from Viper) — M15 confirm, 1.5:1 R:R
    4. BB_FADE     — Bollinger Band mean reversion (from Mamba) — Ranging, 1.5:1 R:R

  M1 SCALPING MODE (M1 primary, M5 confirm) — via --profile propfirm:
    - EMA_PULLBACK + BB_FADE only (Price Action + Breakout disabled — noise on M1)
    - Regime detection on M5 data (M1 ADX too noisy)
    - Fixed-point TP/trailing/breakeven (ATR-based too volatile on M1)
    - M5 trend gate (blocks signals fighting M5 EMA trend)
    - Viper alignment gate (ML-validated direction proxy)
    - 10s strategy loop with M1 candle dedup

Safety: PropFirmGuard with 3-tier risk mode + daily halt + trailing DD halt
Exits: Fixed-point or ATR trailing + RSI extreme exit + Friday flatten
"""
import sys
import os
import json
import time
import threading
import logging
import argparse
import math
import socket
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pro_modules'))

import requests as http_requests
import MetaTrader5 as mt5
import numpy as np

from ultra_fast_indicators import (
    ema_numba, rsi_numba, atr_numba, adx_numba,
    bollinger_numba, macd_numba
)
from shared_data_collector import SharedDataCollector
from trade_logger import TradeDecisionLogger
from news_guard import should_skip_trade

try:
    import cupy as cp
    _gpu_probe = cp.asarray([1.0, 2.0], dtype=cp.float64)
    cp.asnumpy(_gpu_probe * 2.0)
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {}
ACCOUNT_NUMBER = None
PROFILE_NAME = None
BOT_NAME = 'hydra'
API_KEY = os.environ.get("HYDRA_API_KEY", "")

# Thread-safe state
state_lock = threading.Lock()
last_entry_time = {}
bot_stop = threading.Event()
_strategy_thread = None
_trailing_thread = None

# Per-symbol entry lock — prevents duplicate positions within a process
_symbol_entry_locks: dict = {}
_symbol_entry_locks_meta = threading.Lock()

# Per-symbol last-processed candle timestamp — prevents duplicate scans on same M1 bar
_last_candle_ts: dict = {}
_last_candle_lock = threading.Lock()

def _get_symbol_entry_lock(symbol: str) -> threading.Lock:
    with _symbol_entry_locks_meta:
        if symbol not in _symbol_entry_locks:
            _symbol_entry_locks[symbol] = threading.Lock()
        return _symbol_entry_locks[symbol]

# Process-singleton lock — prevents two hydra.py processes running simultaneously
_instance_lock_sock = None

def _acquire_instance_lock(port: int = 18060) -> None:
    """Bind an internal socket so a second process can't start.
    Uses port 18060 (Flask is on 8060; this is a private lock port)."""
    global _instance_lock_sock
    try:
        _instance_lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _instance_lock_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        _instance_lock_sock.bind(('127.0.0.1', port))
        _instance_lock_sock.listen(1)
    except OSError:
        print(
            f"\n[HYDRA] FATAL: Another Hydra instance is already running "
            f"(lock port {port} is taken). Exiting.\n"
        )
        sys.exit(1)

# Loss tracking
_consecutive_losses = {}
_consecutive_losses_date = {}
_symbol_session_losses = {}
_symbol_daily_loss_count = {}
_session_loss_lock = threading.Lock()
_last_loss_time = {}

# PropFirmGuard state
_prop_firm_state = {
    'peak_equity': 0.0,
    'initial_balance': 100000.0,
    'daily_start_equity': 0.0,
    'daily_start_date': None,
    'last_daily_reset': None,
    'halted': False,
    'halt_reason': '',
    'risk_mode': 'normal',  # normal, conservative, critical
    'target_reached': False,
}
_prop_lock = threading.Lock()

def get_trading_day_start() -> datetime:
    """Return the most recent prop-firm reset boundary: 20:00 UTC."""
    now = datetime.now(timezone.utc)
    reset_time = now.replace(hour=20, minute=0, second=0, microsecond=0)
    if now < reset_time:
        reset_time -= timedelta(days=1)
    return reset_time

def get_trading_day_key() -> str:
    """Stable string key for the current trading day (resets at 20:00 UTC)."""
    return get_trading_day_start().strftime('%Y-%m-%d')

def parse_reset_timestamp(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None

def should_reset_daily(last_reset_time):
    if last_reset_time is None:
        return True
    return last_reset_time < get_trading_day_start()

# Flask
app = Flask(__name__)

# ============================================================================
# AUTH
# ============================================================================
@app.before_request
def check_auth():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

# ============================================================================
# LOGGING
# ============================================================================
logger = None

def init_logger():
    global logger
    logger = logging.getLogger(f"HYDRA-{ACCOUNT_NUMBER}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"hydra_{ACCOUNT_NUMBER}.log", encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    logger.info('[GPU] CuPy loaded - GPU acceleration enabled' if GPU_AVAILABLE else '[GPU] CuPy not available - using CPU')

# ============================================================================
# SINGLETONS
# ============================================================================
_trade_logger = None
_ml_dc = None
_tracked_positions = {}
positions_lock = threading.Lock()
_pending_close_checks = {}
pending_close_lock = threading.Lock()

def get_trade_logger(): return _trade_logger
def set_trade_logger(tl):
    global _trade_logger
    _trade_logger = tl

def get_ml_dc(): return _ml_dc
def set_ml_dc(dc):
    global _ml_dc
    _ml_dc = dc

# ============================================================================
# MT5 TIMEFRAME MAP
# ============================================================================
TF_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def mt5_rates_to_numpy(rates):
    if hasattr(rates, 'dtype') and rates.dtype.names:
        return np.column_stack([
            rates['time'].astype(np.float64),
            rates['open'].astype(np.float64),
            rates['high'].astype(np.float64),
            rates['low'].astype(np.float64),
            rates['close'].astype(np.float64),
            rates['tick_volume'].astype(np.float64),
        ])
    return np.array(rates, dtype=np.float64)

def safe_last(val):
    if val is None:
        raise ValueError("safe_last: received None")
    if isinstance(val, np.ndarray):
        if len(val) == 0:
            raise ValueError("safe_last: empty array")
        return float(val[-1])
    return float(val)

def normalize_pos_type(t):
    if t in ('BUY', 'buy', 0): return 'BUY'
    if t in ('SELL', 'sell', 1): return 'SELL'
    return None

def get_symbol_digits(symbol):
    try:
        info = mt5.symbol_info(symbol)
        return info.digits if info else 5
    except Exception:
        return 5

def get_spread_points(symbol):
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return round(tick.ask - tick.bid, get_symbol_digits(symbol))
        return None
    except Exception:
        return None

def get_account_info():
    try:
        info = mt5.account_info()
        if info:
            return {
                'balance': info.balance,
                'equity': info.equity,
                'margin': info.margin,
                'free_margin': info.margin_free,
                'profit': info.profit,
            }
        return None
    except Exception:
        return None

def determine_exit_reason(entry_price, close_price, sl, tp, symbol):
    try:
        sym_info = mt5.symbol_info(symbol)
        tolerance = (sym_info.point * 5) if sym_info else 0.0001
        if tp and abs(close_price - tp) <= tolerance:
            return 'TP_HIT'
        elif sl and abs(close_price - sl) <= tolerance:
            if abs(sl - entry_price) <= tolerance:
                return 'BREAKEVEN'
            else:
                return 'SL_HIT'
        else:
            return 'TRAILING_SL'
    except Exception:
        return 'UNKNOWN'

def serialize_result(result):
    if result is None:
        return None
    if isinstance(result, (bool, int, float, str, dict, list)):
        return result
    data = {}
    for attr in ('retcode', 'order', 'deal', 'volume', 'price', 'bid', 'ask', 'comment', 'request_id'):
        if hasattr(result, attr):
            data[attr] = getattr(result, attr)
    return data or str(result)

# ============================================================================
# ML DATA LOGGING
# ============================================================================
def build_ml_row(symbol, signal, trigger, confidence, atr, executed, rates_np=None):
    bar_open = bar_high = bar_low = bar_close = None
    if rates_np is not None and len(rates_np) >= 1:
        bar_open  = float(rates_np[-1, 1])
        bar_high  = float(rates_np[-1, 2])
        bar_low   = float(rates_np[-1, 3])
        bar_close = float(rates_np[-1, 4])
    spread_val = get_spread_points(symbol)
    return {
        'timestamp':     datetime.now(timezone.utc).isoformat(),
        'symbol':        symbol,
        'bot':           BOT_NAME,
        'account':       ACCOUNT_NUMBER,
        'timeframe':     CONFIG.get('timeframe_primary', 'H1'),
        'signal':        signal,
        'signal_type':   signal,
        'trigger':       trigger,
        'confidence':    confidence,
        'atr':           atr,
        'executed':      executed,
        'entry_price':   None,
        'sl_price':      None,
        'sl':            None,
        'tp_price':      None,
        'tp':            None,
        'volume_lots':   None,
        'ticket':        None,
        'exit_price':    None,
        'exit_reason':   None,
        'profit':        None,
        'pnl':           None,
        'outcome':       None,
        'duration_minutes': None,
        'account_balance': None,
        'account_equity':  None,
        'spread':       spread_val,
        'spread_points': spread_val,
        'bar_open':      bar_open,
        'bar_high':      bar_high,
        'bar_low':       bar_low,
        'bar_close':     bar_close,
        'regime':        None,
        'adx':           None,
        'rsi':           None,
        'ema_fast_val':  None,
        'ema_slow_val':  None,
        'bb_upper':      None,
        'bb_lower':      None,
        'bb_mid':        None,
        'risk_mode':     None,
        'hold_reason':   None,
    }

def log_trade_outcome(symbol, ticket, entry_price, sl, tp, volume,
                      close_price, profit, exit_reason, duration_minutes,
                      atr=0, account_balance=None, account_equity=None):
    try:
        outcome = 'WIN' if profit > 0.01 else ('LOSS' if profit < -0.01 else 'BREAKEVEN')
        dc = get_ml_dc()
        if dc:
            row = build_ml_row(symbol, 'TRADE_CLOSED', 'OUTCOME', 0, atr, 0)
            row.update({
                'entry_price': entry_price, 'sl_price': sl, 'sl': sl,
                'tp_price': tp, 'tp': tp, 'volume_lots': volume,
                'ticket': ticket, 'exit_price': close_price,
                'exit_reason': exit_reason, 'profit': profit, 'pnl': profit,
                'outcome': outcome, 'duration_minutes': round(duration_minutes, 1),
                'account_balance': account_balance, 'account_equity': account_equity,
            })
            dc.log_signal(row)
        logger.info(f"[OUTCOME] {symbol} #{ticket}: {outcome} ${profit:.2f} via {exit_reason}")
    except Exception as e:
        logger.warning(f"Failed to log trade outcome: {e}")


def find_closing_deal(ticket, lookback_hours=168):
    """Return the most recent closing deal for a tracked position."""
    out_entries = {mt5.DEAL_ENTRY_OUT}
    inout_entry = getattr(mt5, 'DEAL_ENTRY_INOUT', None)
    if inout_entry is not None:
        out_entries.add(inout_entry)
    ticket_int = int(ticket)

    def pick_candidate(deals):
        if not deals:
            return None
        candidates = [
            deal for deal in deals
            if getattr(deal, 'position_id', None) == ticket_int
            and getattr(deal, 'entry', None) in out_entries
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: (getattr(d, 'time_msc', 0), getattr(d, 'time', 0), getattr(d, 'ticket', 0)))

    deal = pick_candidate(mt5.history_deals_get(position=ticket_int))
    if deal:
        return deal

    now_naive = datetime.now()
    start_time = now_naive - timedelta(hours=lookback_hours)
    return pick_candidate(mt5.history_deals_get(start_time, now_naive))

def note_pending_close(ticket):
    now_ts = time.time()
    with pending_close_lock:
        pending = _pending_close_checks.get(ticket)
        if pending is None:
            pending = {'first_seen': now_ts, 'last_log': 0.0}
            _pending_close_checks[ticket] = pending
        age_seconds = now_ts - pending['first_seen']
        should_log = (now_ts - pending['last_log']) >= 60.0
        if should_log:
            pending['last_log'] = now_ts
    return age_seconds, should_log

def clear_pending_close(ticket):
    with pending_close_lock:
        _pending_close_checks.pop(ticket, None)

# ============================================================================
# LOSS / COOLDOWN TRACKING
# ============================================================================
def record_trade_result(symbol, profit):
    today = get_trading_day_key()
    with _session_loss_lock:
        if profit < 0:
            _consecutive_losses[symbol] = _consecutive_losses.get(symbol, 0) + 1
            _last_loss_time[symbol] = time.time()
            count = _symbol_daily_loss_count.get(symbol, {})
            if count.get('date') != today:
                count = {'date': today, 'count': 0}
            count['count'] += 1
            _symbol_daily_loss_count[symbol] = count
        else:
            _consecutive_losses[symbol] = 0
        _consecutive_losses_date[symbol] = today

        state = _symbol_session_losses.get(symbol, {})
        if state.get('date') != today:
            state = {'date': today, 'total_loss': 0.0, 'trade_count': 0}
        state['total_loss'] = round(state['total_loss'] + profit, 2)
        state['trade_count'] += 1
        _symbol_session_losses[symbol] = state

        if profit < 0:
            logger.info(
                f"[LOSS-TRACK] {symbol}: streak={_consecutive_losses[symbol]}, "
                f"session P&L={state['total_loss']:.2f}, losses_today={_symbol_daily_loss_count.get(symbol, {}).get('count', 0)}"
            )

def get_adaptive_cooldown(symbol, base_cooldown, cfg):
    with _session_loss_lock:
        losses = _consecutive_losses.get(symbol, 0)
    if losses == 0:
        return base_cooldown
    multiplier = cfg.get('loss_cooldown_multiplier', 2.0)
    max_cd = cfg.get('max_loss_cooldown', 3600)
    adapted = base_cooldown * (multiplier ** min(losses, 5))
    return min(adapted, max_cd)

# ============================================================================
# PROP FIRM GUARD — THE SAFETY LAYER
# ============================================================================
def prop_firm_guard_update():
    """Update PropFirmGuard state. Called every strategy cycle."""
    # Compute state inside the lock, but execute MT5 close calls outside it.
    close_reason = None

    with _prop_lock:
        acc = get_account_info()
        if not acc:
            return
        equity = acc['equity']
        pf = CONFIG.get('prop_firm', {})
        today = get_trading_day_key()
        reset_boundary = get_trading_day_start()
        last_daily_reset = parse_reset_timestamp(_prop_firm_state.get('last_daily_reset'))

        # Reset daily tracking at 20:00 UTC each day
        if should_reset_daily(last_daily_reset):
            _prop_firm_state['daily_start_equity'] = equity
            _prop_firm_state['daily_start_date'] = today
            _prop_firm_state['last_daily_reset'] = reset_boundary.isoformat()
            _prop_firm_state['halted'] = False
            _prop_firm_state['halt_reason'] = ''
            logger.info(f"[DAILY RESET] Daily P&L reset at 20:00 UTC -- start equity: ${equity:.2f}")

        # Track peak equity for trailing drawdown
        if equity > _prop_firm_state['peak_equity']:
            _prop_firm_state['peak_equity'] = equity
            logger.info(f"[PROP] New peak equity: ${equity:.2f}")

        # Calculate daily P&L
        daily_pnl = equity - _prop_firm_state['daily_start_equity']
        daily_pnl_pct = (daily_pnl / _prop_firm_state['initial_balance']) * 100

        # Calculate trailing drawdown from peak
        dd_from_peak = equity - _prop_firm_state['peak_equity']
        dd_from_peak_pct = (dd_from_peak / _prop_firm_state['initial_balance']) * 100

        # Check profit target reached
        total_profit_pct = ((equity - _prop_firm_state['initial_balance']) / _prop_firm_state['initial_balance']) * 100
        if total_profit_pct >= pf.get('profit_target_pct', 6.0):
            _prop_firm_state['target_reached'] = True
            logger.info(f"[PROP] TARGET REACHED! Total profit: {total_profit_pct:.2f}%")

        # === HALT CONDITIONS ===
        halt_pct = pf.get('daily_loss_halt_pct', 2.5)
        if daily_pnl_pct <= -halt_pct:
            _prop_firm_state['halted'] = True
            _prop_firm_state['halt_reason'] = f'daily_loss_{daily_pnl_pct:.2f}%'
            logger.warning(f"[PROP] HALT — Daily loss {daily_pnl_pct:.2f}% exceeds -{halt_pct}%")
            close_reason = "PROP_DAILY_HALT"

        dd_halt_pct = pf.get('trailing_dd_halt_pct', 5.5)
        if close_reason is None and dd_from_peak_pct <= -dd_halt_pct:
            _prop_firm_state['halted'] = True
            _prop_firm_state['halt_reason'] = f'trailing_dd_{dd_from_peak_pct:.2f}%'
            logger.warning(f"[PROP] HALT — Trailing DD {dd_from_peak_pct:.2f}% exceeds -{dd_halt_pct}%")
            close_reason = "PROP_DD_HALT"

        # === RISK MODE ===
        daily_bank_pct = pf.get('daily_profit_bank_pct', 2.0)
        if close_reason is None and daily_pnl_pct >= daily_bank_pct:
            _prop_firm_state['halted'] = True
            _prop_firm_state['halt_reason'] = f'daily_profit_banked_{daily_pnl_pct:.2f}%'
            logger.info(f"[PROP] BANK — Daily profit {daily_pnl_pct:.2f}% hit bank threshold, stopping for the day")

        dd_critical_pct = pf.get('trailing_dd_critical_pct', 5.0)
        dd_warning_pct = pf.get('trailing_dd_warning_pct', 4.0)
        daily_warn_pct = pf.get('daily_loss_warning_pct', 1.5)
        daily_conservative_pct = pf.get('daily_profit_conservative_pct', 1.5)

        if close_reason is None and not _prop_firm_state['halted']:
            if dd_from_peak_pct <= -dd_critical_pct or daily_pnl_pct <= -halt_pct + 0.5:
                _prop_firm_state['risk_mode'] = 'critical'
            elif dd_from_peak_pct <= -dd_warning_pct or daily_pnl_pct <= -daily_warn_pct:
                _prop_firm_state['risk_mode'] = 'conservative'
            elif daily_pnl_pct >= daily_conservative_pct:
                _prop_firm_state['risk_mode'] = 'conservative'
            else:
                _prop_firm_state['risk_mode'] = 'normal'

        logger.info(
            f"[PROP] equity=${equity:.2f} daily={daily_pnl_pct:+.2f}% "
            f"dd={dd_from_peak_pct:.2f}% mode={_prop_firm_state['risk_mode']} "
            f"peak=${_prop_firm_state['peak_equity']:.2f}"
        )

    # Execute close OUTSIDE the lock so MT5 calls don't hold _prop_lock
    if close_reason:
        _close_all_positions(close_reason)

def prop_firm_can_trade(symbol):
    """Check if PropFirmGuard allows trading. Returns (bool, reason)."""
    with _prop_lock:
        if _prop_firm_state['halted']:
            return False, f"halted: {_prop_firm_state['halt_reason']}"
        if _prop_firm_state['target_reached']:
            return False, "profit_target_reached"

    pf = CONFIG.get('prop_firm', {})

    # Check loss cooloff
    with _session_loss_lock:
        last_loss = _last_loss_time.get(symbol, 0)
        cooloff = pf.get('loss_cooloff_seconds', 300)
        if time.time() - last_loss < cooloff and last_loss > 0:
            return False, f"loss_cooloff ({int(cooloff - (time.time() - last_loss))}s remaining)"

        # Check max losses per symbol per day
        today = get_trading_day_key()
        daily_losses = _symbol_daily_loss_count.get(symbol, {})
        max_losses = pf.get('max_losses_per_symbol_per_day', 2)
        if daily_losses.get('date') == today and daily_losses.get('count', 0) >= max_losses:
            return False, f"max_losses_per_symbol_per_day ({daily_losses['count']}/{max_losses})"

    # Check position limits
    positions = mt5.positions_get()
    if positions is None:
        positions = []

    total_positions = len(positions)
    max_total = pf.get('max_total_positions', 3)
    if total_positions >= max_total:
        return False, f"max_positions ({total_positions}/{max_total})"

    symbol_positions = [p for p in positions if p.symbol == symbol]
    max_per_symbol = pf.get('max_positions_per_symbol', 1)
    if len(symbol_positions) >= max_per_symbol:
        return False, f"max_per_symbol ({len(symbol_positions)}/{max_per_symbol})"

    return True, "OK"

def get_risk_percent():
    """Return base risk % per trade (1.0%).
    Drawdown-based scaling is applied smoothly inside calculate_lot_size()
    so the stepped mode reduction is no longer needed here."""
    return 1.0

def get_max_positions():
    """Get max simultaneous positions based on mode."""
    with _prop_lock:
        mode = _prop_firm_state['risk_mode']
    if mode == 'critical':
        return 1
    elif mode == 'conservative':
        return 2
    else:
        return 3

# ============================================================================
# INDICATORS
# ============================================================================
def EMA(prices, period):
    return EMA_GPU(prices, period)

def RSI(prices, period=14):
    return RSI_GPU(prices, period)

def ATR(highs, lows, closes, period=14):
    return ATR_GPU(highs, lows, closes, period)

def EMA_CPU(prices, period):
    return ema_numba(prices, period)

def RSI_CPU(prices, period=14):
    return rsi_numba(prices, period)

def ATR_CPU(highs, lows, closes, period=14):
    return atr_numba(highs, lows, closes, period)

def EMA_GPU(data, period):
    """GPU-accelerated EMA using CuPy with CPU fallback."""
    if not GPU_AVAILABLE or len(data) < period:
        return EMA_CPU(data, period)
    d = cp.asarray(data, dtype=cp.float64)
    alpha = 2.0 / (period + 1)
    result = cp.zeros_like(d)
    result[0] = d[0]
    for i in range(1, len(d)):
        result[i] = alpha * d[i] + (1 - alpha) * result[i - 1]
    return cp.asnumpy(result)

def ATR_GPU(highs, lows, closes, period=14):
    """GPU-accelerated ATR with CPU fallback."""
    if not GPU_AVAILABLE or len(closes) < period + 1:
        return ATR_CPU(highs, lows, closes, period)
    h = cp.asarray(highs, dtype=cp.float64)
    l = cp.asarray(lows, dtype=cp.float64)
    c = cp.asarray(closes, dtype=cp.float64)
    tr = cp.maximum(h[1:] - l[1:], cp.maximum(cp.abs(h[1:] - c[:-1]), cp.abs(l[1:] - c[:-1])))
    atr_vals = cp.zeros(len(tr), dtype=cp.float64)
    atr_vals[:period] = cp.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr_vals[i] = alpha * tr[i] + (1 - alpha) * atr_vals[i - 1]
    return float(cp.asnumpy(atr_vals[-1]))

def RSI_GPU(closes, period=14):
    """GPU-accelerated RSI with CPU fallback."""
    if not GPU_AVAILABLE or len(closes) < period + 1:
        return RSI_CPU(closes, period)
    d = cp.asarray(closes, dtype=cp.float64)
    deltas = cp.diff(d)
    gains = cp.where(deltas > 0, deltas, 0)
    losses = cp.where(deltas < 0, -deltas, 0)
    avg_gain = cp.mean(gains[:period])
    avg_loss = cp.mean(losses[:period])
    rsi_vals = cp.zeros(len(deltas), dtype=cp.float64)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        rsi_vals[i] = 100.0 - (100.0 / (1.0 + rs))
    return cp.asnumpy(rsi_vals)

def ADX(highs, lows, closes, period=14):
    return adx_numba(highs, lows, closes, period)

def BB(closes, period=20, num_std=2.0):
    return bollinger_numba(closes, period, num_std)

def ATR_series(highs, lows, closes, period=14):
    n = len(closes)
    if n < 2: return np.array([0.0])
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr_arr = np.empty(n)
    atr_arr[:period-1] = np.nan
    atr_arr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr_arr[i] = (atr_arr[i-1] * (period - 1) + tr[i]) / period
    return atr_arr

# ============================================================================
# SUPPORT / RESISTANCE DETECTION (from Cobra)
# ============================================================================
def find_support_resistance(highs, lows, closes, lookback=50, tolerance_atr_mult=0.5, curr_atr=1.0):
    supports = []
    resistances = []
    tolerance = curr_atr * tolerance_atr_mult
    start = max(2, len(highs) - lookback)
    for i in range(start, len(highs) - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            resistances.append(float(highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            supports.append(float(lows[i]))
    supports = _cluster_levels(supports, tolerance)
    resistances = _cluster_levels(resistances, tolerance)
    return supports, resistances

def _cluster_levels(levels, tolerance):
    if not levels: return []
    levels.sort()
    clusters = []
    cluster = [levels[0]]
    for i in range(1, len(levels)):
        if abs(levels[i] - np.mean(cluster)) < tolerance:
            cluster.append(levels[i])
        else:
            clusters.append(sum(cluster) / len(cluster))
            cluster = [levels[i]]
    clusters.append(sum(cluster) / len(cluster))
    return clusters

def is_near_level(price, levels, tolerance):
    for level in levels:
        if abs(price - level) <= tolerance:
            return True, level
    return False, None

# ============================================================================
# SESSION / TIME FILTER
# ============================================================================
def is_in_session(symbol, cfg):
    """Check if current time is within trading session for this symbol."""
    time_cfg = cfg.get('time_filter', {})
    if not time_cfg.get('enabled', True):
        return True

    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()

    # No trading on weekends
    if weekday >= 5:
        return False

    # Friday cutoff
    if weekday == 4:
        cutoff = time_cfg.get('friday_cutoff_hour', 18)
        if hour >= cutoff:
            return False

    # Check sessions (supports multiple sessions like USDJPY)
    sessions = time_cfg.get('sessions', [])
    if not sessions:
        return True

    for sess in sessions:
        start = sess.get('start_hour', 0)
        end = sess.get('end_hour', 24)
        if start <= hour < end:
            return True

    return False

def should_friday_flatten():
    """Check if it's Friday and past flatten hour."""
    now = datetime.now(timezone.utc)
    if now.weekday() != 4:
        return False
    pf = CONFIG.get('prop_firm', {})
    flatten_hour = pf.get('friday_flatten_hour', 19)
    return now.hour >= flatten_hour

# ============================================================================
# TRIGGER 1: PRICE ACTION (from Cobra) — H1
# Pin Bar + Engulfing at S/R with EMA trend agreement
# ============================================================================
def check_price_action(rates_np, cfg):
    """
    Check for price action patterns: Pin Bar and Engulfing at S/R.
    Uses completed bar (index -2) for decision.
    Returns: (signal_dict, indicators_dict) or (None, {})
    """
    if len(rates_np) < 55:
        return None, {}, ['insufficient_data']

    closes = rates_np[:, 4]
    highs = rates_np[:, 2]
    lows = rates_np[:, 3]
    opens = rates_np[:, 1]

    curr_atr = ATR(highs, lows, closes, cfg.get('atr_period', 14))
    if curr_atr <= 0:
        return None, {}, ['atr_zero']

    ema_vals = EMA(closes, cfg.get('ema_period', 50))
    supports, resistances = find_support_resistance(
        highs, lows, closes,
        lookback=cfg.get('sr_lookback', 50),
        tolerance_atr_mult=cfg.get('sr_tolerance_atr', 0.5),
        curr_atr=curr_atr
    )

    hold_reasons = []

    # --- Pin Bar Check ---
    pin_signal, pin_ind = _check_pin_bar(opens, highs, lows, closes, ema_vals,
                                          supports, resistances, curr_atr, cfg)
    if pin_signal:
        return pin_signal, pin_ind, []

    # --- Engulfing Check ---
    eng_signal, eng_ind = _check_engulfing(opens, highs, lows, closes, ema_vals, curr_atr, cfg)
    if eng_signal:
        return eng_signal, eng_ind, []

    hold_reasons.append('no_price_action_pattern')
    return None, {}, hold_reasons

def _check_pin_bar(opens, highs, lows, closes, ema_vals, supports, resistances, curr_atr, cfg):
    min_wick_atr = cfg.get('pin_wick_atr_mult', 0.8)
    body_ratio = cfg.get('pin_body_ratio', 2.0)
    tolerance = curr_atr * cfg.get('sr_tolerance_atr', 0.5)

    if len(closes) < 3:
        return None, {}

    o = float(opens[-2]); h = float(highs[-2]); l = float(lows[-2]); c = float(closes[-2])
    curr_ema = float(ema_vals[-2])
    body = max(abs(c - o), 0.0000001)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Bullish pin bar
    if (lower_wick > curr_atr * min_wick_atr and
        lower_wick > body * body_ratio and
        upper_wick < body):
        at_support, level = is_near_level(l, supports, tolerance)
        if at_support and c > curr_ema:
            return {
                'trigger': 'PIN_BAR', 'direction': 'BUY', 'confidence': 0.80,
                'rr_target': 2.0,
                'reason': f'Bullish pin at support {level:.5f}'
            }, {'ema_50': curr_ema, 'nearest_sr_level': level, 'pattern_type': 'BULLISH_PIN'}

    # Bearish pin bar
    if (upper_wick > curr_atr * min_wick_atr and
        upper_wick > body * body_ratio and
        lower_wick < body):
        at_resistance, level = is_near_level(h, resistances, tolerance)
        if at_resistance and c < curr_ema:
            return {
                'trigger': 'PIN_BAR', 'direction': 'SELL', 'confidence': 0.80,
                'rr_target': 2.0,
                'reason': f'Bearish pin at resistance {level:.5f}'
            }, {'ema_50': curr_ema, 'nearest_sr_level': level, 'pattern_type': 'BEARISH_PIN'}

    return None, {}

def _check_engulfing(opens, highs, lows, closes, ema_vals, curr_atr, cfg):
    min_size_atr = cfg.get('engulf_min_atr_mult', 0.5)
    if len(closes) < 4:
        return None, {}

    prev_o = float(opens[-3]); prev_c = float(closes[-3])
    curr_o = float(opens[-2]); curr_c = float(closes[-2])
    curr_ema = float(ema_vals[-2])
    curr_body = abs(curr_c - curr_o)

    if curr_body < curr_atr * min_size_atr:
        return None, {}

    # Bullish engulfing
    if (prev_c < prev_o and curr_c > curr_o and
        curr_o <= prev_c and curr_c > prev_o and curr_c > curr_ema):
        return {
            'trigger': 'ENGULFING', 'direction': 'BUY', 'confidence': 0.75,
            'rr_target': 2.0,
            'reason': f'Bullish engulfing, body={curr_body:.5f}'
        }, {'ema_50': curr_ema, 'pattern_type': 'BULLISH_ENGULF'}

    # Bearish engulfing
    if (prev_c > prev_o and curr_c < curr_o and
        curr_o >= prev_c and curr_c < prev_o and curr_c < curr_ema):
        return {
            'trigger': 'ENGULFING', 'direction': 'SELL', 'confidence': 0.75,
            'rr_target': 2.0,
            'reason': f'Bearish engulfing, body={curr_body:.5f}'
        }, {'ema_50': curr_ema, 'pattern_type': 'BEARISH_ENGULF'}

    return None, {}

# ============================================================================
# TRIGGER 2: BREAKOUT (from Anaconda) — H1, 4/5 conditions
# ============================================================================
def check_breakout(rates_np, cfg):
    """
    Strict breakout: 4 of 5 conditions must be true.
    1. EMA trend direction (fast > slow or fast < slow)
    2. EMA separation (minimum ATR distance)
    3. Price position (close above/below both EMAs)
    4. N-bar breakout (new high/low with ATR buffer)
    5. RSI in healthy zone (40-65 buy, 35-60 sell)
    """
    if len(rates_np) < 55:
        return None, {}, ['insufficient_data']

    closes = rates_np[:, 4]
    highs = rates_np[:, 2]
    lows = rates_np[:, 3]

    ema_fast_period = cfg.get('ema_fast', 20)
    ema_slow_period = cfg.get('ema_slow', 50)
    ema_fast = EMA(closes, ema_fast_period)
    ema_slow = EMA(closes, ema_slow_period)
    curr_atr = ATR(highs, lows, closes, cfg.get('atr_period', 14))
    rsi_val = RSI(closes, cfg.get('rsi_period', 14))

    if curr_atr <= 0:
        return None, {}, ['atr_zero']

    lookback = cfg.get('breakout_lookback', 20)
    buffer_mult = cfg.get('breakout_buffer_mult', 0.2)
    required_conditions = cfg.get('breakout_conditions_required', 4)

    fast_val = float(ema_fast[-2])
    slow_val = float(ema_slow[-2])
    close_val = float(closes[-2])
    separation = abs(fast_val - slow_val) / curr_atr

    n_bar_high = float(np.max(highs[-lookback-2:-2]))
    n_bar_low = float(np.min(lows[-lookback-2:-2]))

    hold_reasons = []

    # Check BUY conditions
    buy_conditions = 0
    if fast_val > slow_val: buy_conditions += 1
    else: hold_reasons.append('ema_not_bullish')
    if separation > 0.3: buy_conditions += 1
    else: hold_reasons.append('ema_separation_low')
    if close_val > fast_val and close_val > slow_val: buy_conditions += 1
    else: hold_reasons.append('price_below_emas')
    if close_val > n_bar_high + curr_atr * buffer_mult: buy_conditions += 1
    else: hold_reasons.append('no_breakout_high')
    if 40 <= rsi_val <= 65: buy_conditions += 1
    else: hold_reasons.append(f'rsi_out_of_zone_{rsi_val:.0f}')

    if buy_conditions >= required_conditions:
        return {
            'trigger': 'BREAKOUT', 'direction': 'BUY', 'confidence': 0.70 + (buy_conditions - 4) * 0.10,
            'rr_target': 2.0,
            'reason': f'Breakout BUY {buy_conditions}/5 conditions, close>{n_bar_high:.5f}'
        }, {
            'ema_fast_val': fast_val, 'ema_slow_val': slow_val,
            'ema_separation': separation, 'rsi': rsi_val,
            'n_bar_high': n_bar_high, 'conditions_met': buy_conditions,
        }, []

    # Check SELL conditions
    hold_reasons = []
    sell_conditions = 0
    if fast_val < slow_val: sell_conditions += 1
    else: hold_reasons.append('ema_not_bearish')
    if separation > 0.3: sell_conditions += 1
    else: hold_reasons.append('ema_separation_low')
    if close_val < fast_val and close_val < slow_val: sell_conditions += 1
    else: hold_reasons.append('price_above_emas')
    if close_val < n_bar_low - curr_atr * buffer_mult: sell_conditions += 1
    else: hold_reasons.append('no_breakout_low')
    if 35 <= rsi_val <= 60: sell_conditions += 1
    else: hold_reasons.append(f'rsi_out_of_zone_{rsi_val:.0f}')

    if sell_conditions >= required_conditions:
        return {
            'trigger': 'BREAKOUT', 'direction': 'SELL', 'confidence': 0.70 + (sell_conditions - 4) * 0.10,
            'rr_target': 2.0,
            'reason': f'Breakout SELL {sell_conditions}/5 conditions, close<{n_bar_low:.5f}'
        }, {
            'ema_fast_val': fast_val, 'ema_slow_val': slow_val,
            'ema_separation': separation, 'rsi': rsi_val,
            'n_bar_low': n_bar_low, 'conditions_met': sell_conditions,
        }, []

    return None, {}, hold_reasons

# ============================================================================
# TRIGGER 3: EMA PULLBACK (from Viper) — M15 confirmation with H1 direction
# ============================================================================
def check_ema_pullback(rates_h1, rates_m15, cfg):
    """
    Price pulls back to EMA-20 on M15, bounces in H1 trend direction.
    H1 determines direction, M15 gives entry timing.
    """
    if len(rates_h1) < 55 or len(rates_m15) < 25:
        return None, {}, ['insufficient_data']

    # H1 trend direction
    h1_closes = rates_h1[:, 4]
    h1_ema50 = EMA(h1_closes, 50)
    h1_close = float(h1_closes[-2])
    h1_ema_val = float(h1_ema50[-2])

    if h1_close > h1_ema_val:
        h1_trend = 'BUY'
    elif h1_close < h1_ema_val:
        h1_trend = 'SELL'
    else:
        return None, {}, ['h1_trend_neutral']

    # M15 pullback to EMA-20
    m15_closes = rates_m15[:, 4]
    m15_highs = rates_m15[:, 2]
    m15_lows = rates_m15[:, 3]
    m15_ema20 = EMA(m15_closes, 20)
    m15_atr = ATR(m15_highs, m15_lows, m15_closes, 14)
    m15_rsi = RSI(m15_closes, 14)

    if m15_atr <= 0:
        return None, {}, ['m15_atr_zero']

    m15_close = float(m15_closes[-2])
    m15_ema_val = float(m15_ema20[-2])
    pullback_dist = abs(m15_close - m15_ema_val) / m15_atr
    pullback_max = cfg.get('pullback_atr_mult', 0.7)

    hold_reasons = []

    if pullback_dist > pullback_max:
        hold_reasons.append(f'pullback_too_far_{pullback_dist:.2f}')
        return None, {}, hold_reasons

    # BUY: H1 bullish, M15 price near EMA-20 from below/at, bouncing up
    if h1_trend == 'BUY':
        rsi_min = cfg.get('pullback_rsi_buy_min', 45)
        if (m15_close >= m15_ema_val * 0.998 and  # near or above EMA
            m15_rsi >= rsi_min and m15_rsi <= 65):
            return {
                'trigger': 'EMA_PULLBACK', 'direction': 'BUY', 'confidence': 0.70,
                'rr_target': 1.5,
                'reason': f'EMA pullback BUY, dist={pullback_dist:.2f}ATR, RSI={m15_rsi:.0f}'
            }, {
                'h1_trend': h1_trend, 'ema_20': m15_ema_val,
                'pullback_distance_atr': pullback_dist, 'rsi': m15_rsi,
            }, []
        hold_reasons.append('pullback_buy_conditions_not_met')

    # SELL: H1 bearish, M15 price near EMA-20 from above/at, bouncing down
    elif h1_trend == 'SELL':
        rsi_max = cfg.get('pullback_rsi_sell_max', 55)
        if (m15_close <= m15_ema_val * 1.002 and
            m15_rsi <= rsi_max and m15_rsi >= 35):
            return {
                'trigger': 'EMA_PULLBACK', 'direction': 'SELL', 'confidence': 0.70,
                'rr_target': 1.5,
                'reason': f'EMA pullback SELL, dist={pullback_dist:.2f}ATR, RSI={m15_rsi:.0f}'
            }, {
                'h1_trend': h1_trend, 'ema_20': m15_ema_val,
                'pullback_distance_atr': pullback_dist, 'rsi': m15_rsi,
            }, []
        hold_reasons.append('pullback_sell_conditions_not_met')

    return None, {}, hold_reasons

# ============================================================================
# TRIGGER 4: BB FADE (from Mamba) — M15, Ranging markets only
# ============================================================================
def check_bb_fade(rates_m15, cfg):
    """
    Bollinger Band mean reversion in ranging markets.
    Buy at lower band + RSI oversold, Sell at upper band + RSI overbought.
    """
    if len(rates_m15) < 25:
        return None, {}, ['insufficient_data']

    closes = rates_m15[:, 4]
    highs = rates_m15[:, 2]
    lows = rates_m15[:, 3]

    bb_period = cfg.get('bb_period', 20)
    bb_std = cfg.get('bb_std', 2.0)
    bb_upper, bb_mid, bb_lower = BB(closes, bb_period, bb_std)
    rsi_val = RSI(closes, cfg.get('rsi_period', 14))
    curr_atr = ATR(highs, lows, closes, 14)

    if curr_atr <= 0:
        return None, {}, ['atr_zero']

    close_val = float(closes[-2])
    rsi_oversold = cfg.get('bb_rsi_oversold', 35)
    rsi_overbought = cfg.get('bb_rsi_overbought', 65)

    hold_reasons = []

    # BUY: price at/below lower BB + RSI oversold
    if close_val <= bb_lower and rsi_val <= rsi_oversold:
        return {
            'trigger': 'BB_FADE', 'direction': 'BUY', 'confidence': 0.65,
            'rr_target': 1.5,
            'tp_price': bb_mid,  # Target is middle band
            'reason': f'BB fade BUY, price={close_val:.5f} <= lower={bb_lower:.5f}, RSI={rsi_val:.0f}'
        }, {
            'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_mid': bb_mid,
            'rsi': rsi_val,
        }, []

    # SELL: price at/above upper BB + RSI overbought
    if close_val >= bb_upper and rsi_val >= rsi_overbought:
        return {
            'trigger': 'BB_FADE', 'direction': 'SELL', 'confidence': 0.65,
            'rr_target': 1.5,
            'tp_price': bb_mid,
            'reason': f'BB fade SELL, price={close_val:.5f} >= upper={bb_upper:.5f}, RSI={rsi_val:.0f}'
        }, {
            'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_mid': bb_mid,
            'rsi': rsi_val,
        }, []

    hold_reasons.append('bb_fade_no_signal')
    return None, {}, hold_reasons

# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================
def detect_regime(rates_np, cfg):
    """Detect trending vs ranging using ADX + EMA alignment."""
    if len(rates_np) < 55:
        return 'unknown', 0, 'unknown'

    closes = rates_np[:, 4]
    highs = rates_np[:, 2]
    lows = rates_np[:, 3]

    adx_val, plus_di, minus_di = ADX(highs, lows, closes, cfg.get('adx_period', 14))
    ema_fast = EMA(closes, cfg.get('ema_fast', 20))
    ema_slow = EMA(closes, cfg.get('ema_slow', 50))

    fast_val = float(ema_fast[-2])
    slow_val = float(ema_slow[-2])
    close_val = float(closes[-2])

    min_adx = cfg.get('min_adx_trending', 25)

    if adx_val >= min_adx:
        if close_val > fast_val > slow_val:
            return 'trending', adx_val, 'bullish'
        elif close_val < fast_val < slow_val:
            return 'trending', adx_val, 'bearish'
        else:
            return 'trending', adx_val, 'mixed'
    else:
        return 'ranging', adx_val, 'sideways'

# ============================================================================
# POSITION SIZING
# ============================================================================
def calculate_lot_size(symbol, sl_distance, cfg):
    """Calculate position size based on risk % and SL distance."""
    acc = get_account_info()
    if not acc or sl_distance <= 0:
        return 0.0

    risk_pct = get_risk_percent()
    # Override with symbol config if lower
    symbol_risk = cfg.get('risk_percent', 1.0)
    risk_pct = min(risk_pct, symbol_risk)

    risk_amount = acc['equity'] * (risk_pct / 100.0)

    sym_info = mt5.symbol_info(symbol)
    if not sym_info:
        return 0.0

    contract_size = sym_info.trade_contract_size
    tick_value = sym_info.trade_tick_value
    tick_size = sym_info.trade_tick_size

    if tick_size <= 0 or tick_value <= 0:
        return 0.0

    # lots = risk_amount / (sl_distance / tick_size * tick_value)
    ticks_in_sl = sl_distance / tick_size
    risk_per_lot = ticks_in_sl * tick_value

    if risk_per_lot <= 0:
        return 0.0

    lots = risk_amount / risk_per_lot

    # === SMOOTH DRAWDOWN SCALING ===
    # Scale lots down linearly as we consume the daily/trailing loss budgets.
    # daily_scale    = 1 - (daily_loss_used%  / daily_halt%)
    # trailing_scale = 1 - (trailing_dd_used% / trailing_dd_halt%)
    # smooth_mult    = min(daily_scale, trailing_scale)  — range [0.01, 1.0]
    pf_cfg = CONFIG.get('prop_firm', {})
    daily_halt_pct    = float(pf_cfg.get('daily_loss_halt_pct', 2.5))
    trailing_halt_pct = float(pf_cfg.get('trailing_dd_halt_pct', 5.5))
    with _prop_lock:
        pf_mode     = _prop_firm_state.get('risk_mode', 'normal')
        daily_start = _prop_firm_state.get('daily_start_equity', acc['equity'])
        peak        = _prop_firm_state.get('peak_equity', acc['equity'])
        initial     = _prop_firm_state.get('initial_balance', acc['equity'])
    daily_scale    = 1.0
    trailing_scale = 1.0
    if pf_mode == 'halted':
        smooth_mult = 0.01
    elif initial > 0:
        daily_loss_pct  = max(0.0, (daily_start - acc['equity']) / initial * 100)
        daily_scale     = max(0.01, 1.0 - daily_loss_pct  / daily_halt_pct)    if daily_halt_pct    > 0 else 1.0
        trailing_dd_pct = max(0.0, (peak        - acc['equity']) / initial * 100)
        trailing_scale  = max(0.01, 1.0 - trailing_dd_pct / trailing_halt_pct) if trailing_halt_pct > 0 else 1.0
        smooth_mult = round(min(daily_scale, trailing_scale), 3)
    else:
        smooth_mult = 1.0
    if smooth_mult < 1.0:
        logger.info(f"[LOT-SCALE] {symbol} smooth_mult={smooth_mult:.3f} "
                    f"(daily_scale={daily_scale:.3f}, trail_scale={trailing_scale:.3f})")
    lots = lots * smooth_mult

    # Clamp to min/max
    min_lot = cfg.get('min_lot', 0.01)
    max_lot = sym_info.volume_max if getattr(sym_info, 'volume_max', 0) > 0 else cfg.get('max_lot', 1.0)
    lots = max(min_lot, min(lots, max_lot))

    # Round to lot step
    lot_step = sym_info.volume_step
    if lot_step > 0:
        lots = round(lots / lot_step) * lot_step
        lots = round(lots, 2)

    return lots

# ============================================================================
# ORDER EXECUTION
# ============================================================================
def execute_trade(symbol, direction, sl, tp, volume, trigger, reason, cfg):
    """Execute a market order via MT5."""
    try:
        sym_info = mt5.symbol_info(symbol)
        if not sym_info or not sym_info.visible:
            mt5.symbol_select(symbol, True)
            sym_info = mt5.symbol_info(symbol)
            if not sym_info:
                logger.error(f"[EXEC] Symbol {symbol} not available")
                return None

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"[EXEC] No tick data for {symbol}")
            return None

        if direction == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        digits = sym_info.digits
        sl = round(sl, digits)
        tp = round(tp, digits)
        price = round(price, digits)

        request_obj = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": ACCOUNT_NUMBER,
            "comment": f"HYDRA_{trigger}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request_obj)
        if result is None:
            logger.error(f"[EXEC] order_send returned None for {symbol}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Try FOK filling
            request_obj["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(request_obj)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"[EXEC] Order failed for {symbol}: {result.retcode if result else 'None'} - {result.comment if result else ''}")
                return None

        ticket = result.order
        actual_price = result.price if result.price else price

        logger.info(
            f"[TRADE] {direction} {symbol} | {volume} lots @ {actual_price} | "
            f"SL={sl} TP={tp} | trigger={trigger} | ticket={ticket}"
        )

        # Track position
        with positions_lock:
            _tracked_positions[ticket] = {
                'symbol': symbol, 'entry_price': actual_price,
                'sl': sl, 'tp': tp, 'volume': volume,
                'open_time': time.time(), 'type': direction,
                'trigger': trigger, 'atr': 0,
                'breakeven_moved': False,
                'trail_phase': 0,
            }

        # Update last entry time
        with state_lock:
            last_entry_time[symbol] = time.time()

        # Log to ML
        acc = get_account_info()
        dc = get_ml_dc()
        if dc:
            row = build_ml_row(symbol, direction, trigger, 0, 0, 1, None)
            row.update({
                'entry_price': actual_price, 'sl_price': sl, 'sl': sl,
                'tp_price': tp, 'tp': tp, 'volume_lots': volume,
                'ticket': ticket,
                'account_balance': acc['balance'] if acc else None,
                'account_equity': acc['equity'] if acc else None,
                'risk_mode': _prop_firm_state.get('risk_mode', 'normal'),
            })
            dc.log_signal(row)

        return ticket

    except Exception as e:
        logger.error(f"[EXEC] Exception executing {direction} {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ============================================================================
# CLOSE ALL POSITIONS (emergency)
# ============================================================================
def _close_all_positions(reason="MANUAL"):
    """Close all open positions. Used by PropFirmGuard halt."""
    try:
        positions = mt5.positions_get()
        if not positions:
            return
        for pos in positions:
            if pos.magic != ACCOUNT_NUMBER:
                continue
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue
            if pos.type == mt5.ORDER_TYPE_BUY:
                close_price = tick.bid
                close_type = mt5.ORDER_TYPE_SELL
            else:
                close_price = tick.ask
                close_type = mt5.ORDER_TYPE_BUY

            close_req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": 30,
                "magic": ACCOUNT_NUMBER,
                "comment": f"HYDRA_CLOSE_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(close_req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[CLOSE] Closed {pos.symbol} #{pos.ticket} — {reason}")
                record_trade_result(pos.symbol, pos.profit)
            else:
                # Try FOK
                close_req["type_filling"] = mt5.ORDER_FILLING_FOK
                result = mt5.order_send(close_req)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[CLOSE] Closed {pos.symbol} #{pos.ticket} — {reason} (FOK)")
                    record_trade_result(pos.symbol, pos.profit)
                else:
                    logger.error(f"[CLOSE] Failed to close {pos.symbol} #{pos.ticket}")
    except Exception as e:
        logger.error(f"[CLOSE] Error closing all: {e}")

# ============================================================================
# TRAILING STOP MANAGER (3-phase tightening)
# ============================================================================
def trailing_loop():
    """Run every 30 seconds — manage breakeven + trailing + RSI exit + Friday flatten."""
    while not bot_stop.is_set():
        try:
            if should_friday_flatten():
                _close_all_positions("FRIDAY_FLATTEN")
                bot_stop.wait(300)
                continue

            positions = mt5.positions_get()
            if not positions:
                bot_stop.wait(30)
                continue

            for pos in positions:
                if pos.magic != ACCOUNT_NUMBER:
                    continue
                _manage_position(pos)

        except Exception as e:
            logger.error(f"[TRAIL] Error: {e}")

        bot_stop.wait(CONFIG.get('trailing_interval', 30))

def _manage_position(pos):
    """Manage a single open position: breakeven, trailing, RSI exit."""
    try:
        symbol = pos.symbol
        ticket = pos.ticket
        entry_price = pos.price_open
        current_sl = pos.sl
        current_tp = pos.tp
        volume = pos.volume
        pos_type = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return

        current_price = tick.bid if pos_type == 'BUY' else tick.ask

        # Get symbol config
        sym_cfg = CONFIG.get('symbols', {}).get(symbol, {})
        if not sym_cfg:
            # Try without suffix variations
            for s, c in CONFIG.get('symbols', {}).items():
                if s in symbol or symbol in s:
                    sym_cfg = c
                    break
            if not sym_cfg:
                return

        # Get tracked position data
        with positions_lock:
            tracked = _tracked_positions.get(ticket, {})

        # Calculate ATR for this symbol
        tf_map_val = TF_MAP.get(CONFIG.get('timeframe_primary', 'H1'), mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf_map_val, 0, 100)
        if rates is None or len(rates) < 20:
            return
        rates_np = mt5_rates_to_numpy(rates)
        curr_atr = ATR(rates_np[:, 2], rates_np[:, 3], rates_np[:, 4], 14)
        if curr_atr <= 0:
            return

        # Calculate distance from entry in ATR units
        if pos_type == 'BUY':
            distance_atr = (current_price - entry_price) / curr_atr
        else:
            distance_atr = (entry_price - current_price) / curr_atr

        new_sl = current_sl
        digits = get_symbol_digits(symbol)

        # Check for fixed-point mode (preferred for M1 scalping)
        sym_info_trail = mt5.symbol_info(symbol)
        point = sym_info_trail.point if sym_info_trail else 0.0
        fixed_be_pts = sym_cfg.get('fixed_breakeven_points', 0)
        fixed_trail_pts = sym_cfg.get('fixed_trail_points', 0)
        use_fixed_mode = (fixed_be_pts > 0 or fixed_trail_pts > 0) and point > 0

        if use_fixed_mode:
            # === FIXED-POINT MODE (M1 scalping) ===
            if pos_type == 'BUY':
                distance_pts = (current_price - entry_price) / point
            else:
                distance_pts = (entry_price - current_price) / point

            # Fixed breakeven
            if fixed_be_pts > 0 and distance_pts >= fixed_be_pts and not tracked.get('breakeven_moved', False):
                be_offset = point * max(fixed_be_pts * 0.1, 10)  # small profit lock
                if pos_type == 'BUY':
                    new_sl = round(entry_price + be_offset, digits)
                else:
                    new_sl = round(entry_price - be_offset, digits)
                with positions_lock:
                    if ticket in _tracked_positions:
                        _tracked_positions[ticket]['breakeven_moved'] = True
                logger.info(f"[TRAIL] {symbol} #{ticket}: FIXED BREAKEVEN at {distance_pts:.0f}pts, SL -> {new_sl}")

            # Fixed trailing
            if fixed_trail_pts > 0 and distance_pts >= fixed_be_pts:
                trail_dist = fixed_trail_pts * point
                if pos_type == 'BUY':
                    trail_sl = round(current_price - trail_dist, digits)
                    new_sl = max(new_sl, trail_sl) if new_sl else trail_sl
                else:
                    trail_sl = round(current_price + trail_dist, digits)
                    new_sl = min(new_sl, trail_sl) if new_sl else trail_sl
                with positions_lock:
                    if ticket in _tracked_positions:
                        _tracked_positions[ticket]['trail_phase'] = 1

        else:
            # === ATR-BASED MODE (standard H1/M15/M30/H4) ===

            # --- Phase 0: Breakeven ---
            be_mult = sym_cfg.get('breakeven_atr_mult', 0.8)
            if distance_atr >= be_mult and not tracked.get('breakeven_moved', False):
                if pos_type == 'BUY':
                    new_sl = round(entry_price + curr_atr * 0.1, digits)
                else:
                    new_sl = round(entry_price - curr_atr * 0.1, digits)
                with positions_lock:
                    if ticket in _tracked_positions:
                        _tracked_positions[ticket]['breakeven_moved'] = True
                logger.info(f"[TRAIL] {symbol} #{ticket}: BREAKEVEN SL -> {new_sl}")

            # --- Phase 1: Trail at 1.2x ATR behind ---
            phase1_trigger = sym_cfg.get('trail_phase1_atr', 1.5)
            phase1_dist = sym_cfg.get('trail_phase1_distance', 1.2)
            if distance_atr >= phase1_trigger:
                if pos_type == 'BUY':
                    trail_sl = round(current_price - curr_atr * phase1_dist, digits)
                    new_sl = max(new_sl, trail_sl) if new_sl else trail_sl
                else:
                    trail_sl = round(current_price + curr_atr * phase1_dist, digits)
                    new_sl = min(new_sl, trail_sl) if new_sl else trail_sl
                with positions_lock:
                    if ticket in _tracked_positions:
                        _tracked_positions[ticket]['trail_phase'] = 1

            # --- Phase 2: Tighten to 0.8x ATR behind ---
            phase2_trigger = sym_cfg.get('trail_phase2_atr', 2.5)
            phase2_dist = sym_cfg.get('trail_phase2_distance', 0.8)
            if distance_atr >= phase2_trigger:
                if pos_type == 'BUY':
                    trail_sl = round(current_price - curr_atr * phase2_dist, digits)
                    new_sl = max(new_sl, trail_sl) if new_sl else trail_sl
                else:
                    trail_sl = round(current_price + curr_atr * phase2_dist, digits)
                    new_sl = min(new_sl, trail_sl) if new_sl else trail_sl
                with positions_lock:
                    if ticket in _tracked_positions:
                        _tracked_positions[ticket]['trail_phase'] = 2

        # --- RSI Exit: Close if in profit and RSI extreme ---
        rsi_exit_high = sym_cfg.get('rsi_exit_high', 75)
        rsi_exit_low = sym_cfg.get('rsi_exit_low', 25)
        closes = rates_np[:, 4]
        rsi_val = RSI(closes, 14)

        profit = pos.profit
        if profit > 0:
            if (pos_type == 'BUY' and rsi_val >= rsi_exit_high) or \
               (pos_type == 'SELL' and rsi_val <= rsi_exit_low):
                logger.info(f"[TRAIL] RSI EXIT {symbol} #{ticket}: RSI={rsi_val:.0f}, profit=${profit:.2f}")
                _close_position(pos, f"RSI_EXIT_{rsi_val:.0f}")
                return

        # --- Modify SL if changed ---
        if new_sl and new_sl != current_sl:
            # Safety: never move SL further from entry (only tighter)
            if pos_type == 'BUY' and (current_sl is None or current_sl == 0 or new_sl > current_sl):
                _modify_sl(ticket, symbol, new_sl, current_tp, volume)
            elif pos_type == 'SELL' and (current_sl is None or current_sl == 0 or new_sl < current_sl):
                _modify_sl(ticket, symbol, new_sl, current_tp, volume)

    except Exception as e:
        logger.error(f"[TRAIL] Error managing {pos.symbol} #{pos.ticket}: {e}")

def _modify_sl(ticket, symbol, new_sl, tp, volume):
    """Modify SL on an open position."""
    try:
        request_obj = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": tp,
        }
        result = mt5.order_send(request_obj)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[TRAIL] Modified {symbol} #{ticket} SL -> {new_sl}")
        else:
            logger.warning(f"[TRAIL] Failed to modify {symbol} #{ticket}: {result.retcode if result else 'None'}")
    except Exception as e:
        logger.error(f"[TRAIL] Modify error: {e}")

def _close_position(pos, reason):
    """Close a specific position."""
    try:
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return
        if pos.type == mt5.ORDER_TYPE_BUY:
            close_price = tick.bid
            close_type = mt5.ORDER_TYPE_SELL
        else:
            close_price = tick.ask
            close_type = mt5.ORDER_TYPE_BUY

        close_req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": 30,
            "magic": ACCOUNT_NUMBER,
            "comment": f"HYDRA_{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        duration_min = (time.time() - pos.time) / 60.0 if pos.time else 0
        result = mt5.order_send(close_req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[CLOSE] {pos.symbol} #{pos.ticket} — {reason}")
            log_trade_outcome(
                pos.symbol, pos.ticket, pos.price_open, pos.sl, pos.tp,
                pos.volume, close_price, pos.profit, reason, duration_min
            )
            record_trade_result(pos.symbol, pos.profit)
        else:
            # Try FOK
            close_req["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(close_req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[CLOSE] {pos.symbol} #{pos.ticket} — {reason} (FOK)")
                log_trade_outcome(
                    pos.symbol, pos.ticket, pos.price_open, pos.sl, pos.tp,
                    pos.volume, close_price, pos.profit, reason, duration_min
                )
                record_trade_result(pos.symbol, pos.profit)
    except Exception as e:
        logger.error(f"[CLOSE] Error closing {pos.symbol}: {e}")

# ============================================================================
# CLOSED TRADE DETECTOR — Pick up SL/TP fills
# ============================================================================
def check_closed_trades():
    """Detect positions that were closed by SL/TP since last check."""
    # Step 1: identify closed tickets with minimal lock time
    with positions_lock:
        if not _tracked_positions:
            return

    positions = mt5.positions_get()
    current_tickets = set()
    if positions:
        for p in positions:
            if p.magic == ACCOUNT_NUMBER:
                current_tickets.add(p.ticket)

    with positions_lock:
        closed = [(t, _tracked_positions[t]) for t in list(_tracked_positions) if t not in current_tickets]

    # Step 2: process closed tickets outside the lock
    for ticket, tracked in closed:
        symbol = tracked.get('symbol', 'UNKNOWN')
        try:
            close_deal = find_closing_deal(ticket)
            if close_deal:
                profit = close_deal.profit + close_deal.commission + close_deal.swap
                close_price = close_deal.price
                entry_price = tracked.get('entry_price', 0)
                sl = tracked.get('sl', 0)
                tp = tracked.get('tp', 0)
                volume = tracked.get('volume', 0)
                open_time = tracked.get('open_time', time.time())
                duration_min = (time.time() - open_time) / 60.0

                exit_reason = determine_exit_reason(entry_price, close_price, sl, tp, symbol)
                acc = get_account_info()
                log_trade_outcome(
                    symbol, ticket, entry_price, sl, tp, volume,
                    close_price, profit, exit_reason, duration_min,
                    atr=tracked.get('atr', 0),
                    account_balance=acc['balance'] if acc else None,
                    account_equity=acc['equity'] if acc else None,
                )
                record_trade_result(symbol, profit)
                with positions_lock:
                    _tracked_positions.pop(ticket, None)
                clear_pending_close(str(ticket))
            else:
                age_seconds, should_log = note_pending_close(str(ticket))
                if should_log:
                    logger.info(
                        f"[CLOSED] {symbol} #{ticket} is closed but broker history is still pending "
                        f"({age_seconds:.0f}s). Will retry until realized P&L is available."
                    )
                if age_seconds >= 6 * 3600:
                    logger.warning(
                        f"[CLOSED] {symbol} #{ticket} close history is still missing after "
                        f"{age_seconds / 60:.0f} minutes - dropping tracked position without realized P&L."
                    )
                    with positions_lock:
                        _tracked_positions.pop(ticket, None)
                    clear_pending_close(str(ticket))
        except Exception as e:
            logger.warning(f"[CLOSED] Error processing closed #{ticket}: {e}")

# ============================================================================
# M5 TREND GATE — blocks M1 signals that fight the M5 EMA trend
# ============================================================================
def check_m5_trend_gate(signal_direction, m5_np, cfg):
    """
    Check EMA 20/50 on M5 data.  Returns (passes: bool, m5_trend: str).
    Uses the already-fetched confirm-timeframe bars (m15_np in _scan_symbol,
    which is M5 when timeframe_confirm='M5' is set in the propfirm config).
    """
    if m5_np is None or len(m5_np) < 55:
        return True, 'insufficient_data'   # fail open
    try:
        closes = m5_np[:, 4]
        fast_p = cfg.get('ema_fast', 20)
        slow_p = cfg.get('ema_slow', 50)
        fast_ema = EMA(closes, fast_p)
        slow_ema = EMA(closes, slow_p)
        m5_trend = 'BUY' if float(fast_ema[-1]) > float(slow_ema[-1]) else 'SELL'
        passes = (m5_trend == signal_direction)
        return passes, m5_trend
    except Exception as e:
        logger.debug(f"[M5-GATE] EMA calc error: {e}")
        return True, 'error'   # fail open


# ============================================================================
# VIPER ALIGNMENT GATE — uses Viper's ML-filtered positions as a trend proxy
# ============================================================================
def check_viper_alignment(symbol, direction, cfg):
    """
    Query an alignment bot's /positions endpoint for the given symbol.
    Uses per-symbol bot URLs so different symbols can be validated by the
    most appropriate bot (e.g. XAUUSD → Viper M5, EURUSD → Cobra H1).

    Config layout (viper_gate in config):
        {
            "enabled": true,
            "fail_open": true,
            "timeout": 2.0,
            "symbols": {
                "XAUUSD": {"url": "http://127.0.0.1:8059", "api_key": ""},
                "EURUSD": {"url": "http://127.0.0.1:8050", "api_key": ""}
            }
        }
    Falls back to top-level "url"/"api_key" if no per-symbol entry exists.

    Returns True if:
      - Target bot is unreachable (fail_open=True)
      - No symbol entry in config (no bot configured for this symbol — allow)
      - Bot has no open position on this symbol (neutral — allow)
      - Bot holds a position in the SAME direction (agree — allow)
    Returns False if:
      - Bot holds a position in the OPPOSITE direction (disagree — block)
    """
    fail_open = cfg.get('fail_open', True)
    timeout   = cfg.get('timeout', 2.0)

    # Resolve per-symbol config; fall back to top-level url/api_key
    sym_cfg  = cfg.get('symbols', {}).get(symbol, {})
    url      = sym_cfg.get('url') or cfg.get('url', '')
    api_key  = sym_cfg.get('api_key', cfg.get('api_key', ''))

    if not url:
        # No bot configured for this symbol — nothing to check, allow
        logger.debug(f"[ALIGN-GATE] {symbol}: no alignment bot configured — pass")
        return True

    # Identify which bot we're querying for cleaner log labels
    port_labels = {
        '8059': 'Viper/M5', '8050': 'Cobra/H1', '8055': 'Taipan/M30',
        '8056': 'Mamba/M15', '8051': 'Anaconda/H4',
    }
    port = url.rsplit(':', 1)[-1].rstrip('/')
    bot_label = port_labels.get(port, url)

    try:
        headers = {'X-API-Key': api_key} if api_key else {}
        resp = http_requests.get(f"{url}/positions", headers=headers, timeout=timeout)
        if resp.status_code != 200:
            logger.debug(f"[ALIGN-GATE] {symbol} via {bot_label}: HTTP {resp.status_code} — fail_open={fail_open}")
            return fail_open

        data = resp.json()
        positions = data if isinstance(data, list) else data.get('positions', [])
        sym_positions = [p for p in positions if p.get('symbol') == symbol]

        if not sym_positions:
            return True  # Bot has no view on this symbol — allow

        for pos in sym_positions:
            pos_type = str(pos.get('type', '')).upper()
            if pos_type in ('BUY', '0') and direction == 'SELL':
                logger.info(f"[ALIGN-GATE] {symbol}: {bot_label} is LONG, our signal is SELL — blocked")
                return False
            if pos_type in ('SELL', '1') and direction == 'BUY':
                logger.info(f"[ALIGN-GATE] {symbol}: {bot_label} is SHORT, our signal is BUY — blocked")
                return False

        logger.debug(f"[ALIGN-GATE] {symbol}: {bot_label} agrees with {direction} — pass")
        return True

    except Exception as e:
        logger.debug(f"[ALIGN-GATE] {symbol}: Could not reach {bot_label} ({e}) — fail_open={fail_open}")
        return fail_open


# ============================================================================
# MAIN STRATEGY LOOP
# ============================================================================
def strategy_loop():
    """Main strategy loop — runs every 60 seconds."""
    logger.info("[STRATEGY] Hydra strategy loop started")

    while not bot_stop.is_set():
        try:
            # 1. Update PropFirmGuard
            prop_firm_guard_update()

            # Check if halted
            with _prop_lock:
                if _prop_firm_state['halted']:
                    logger.info(f"[STRATEGY] Halted: {_prop_firm_state['halt_reason']}")
                    check_closed_trades()
                    bot_stop.wait(60)
                    continue

            # 2. Check closed trades
            check_closed_trades()

            # 3. Scan each symbol
            symbols = CONFIG.get('symbols', {})
            # Sort by priority
            sorted_symbols = sorted(symbols.items(), key=lambda x: x[1].get('priority', 99))

            for symbol, sym_cfg in sorted_symbols:
                if not sym_cfg.get('enabled', True):
                    continue

                try:
                    _scan_symbol(symbol, sym_cfg)
                except Exception as e:
                    logger.error(f"[STRATEGY] Error scanning {symbol}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"[STRATEGY] Loop error: {e}")
            import traceback
            logger.error(traceback.format_exc())

        bot_stop.wait(CONFIG.get('strategy_interval', 60))

def _scan_symbol(symbol, cfg):
    """Scan a single symbol for trade setups."""

    # Check session
    if not is_in_session(symbol, cfg):
        dc = get_ml_dc()
        if dc:
            row = build_ml_row(symbol, 'OUTSIDE_HOURS', 'NONE', 0, 0, 0)
            dc.log_signal(row)
        return

    # Check PropFirmGuard
    can_trade, reason = prop_firm_can_trade(symbol)
    if not can_trade:
        dc = get_ml_dc()
        if dc:
            row = build_ml_row(symbol, 'BLOCKED', 'PROP_GUARD', 0, 0, 0)
            row['hold_reason'] = reason
            dc.log_signal(row)
        logger.debug(f"[STRATEGY] {symbol} blocked: {reason}")
        return

    # Check cooldown
    with state_lock:
        last_time = last_entry_time.get(symbol, 0)
    cooldown = get_adaptive_cooldown(symbol, CONFIG.get('entry_cooldown_seconds', 300), cfg)
    if time.time() - last_time < cooldown:
        return

    # Spread filter
    spread = get_spread_points(symbol)
    if spread is None:
        return

    # M1 candle dedup — skip if we already processed this candle (10s loop on 60s bars)
    h1_tf = TF_MAP.get(CONFIG.get('timeframe_primary', 'H1'), mt5.TIMEFRAME_H1)
    if CONFIG.get('timeframe_primary', 'H1') == 'M1':
        peek_rates = mt5.copy_rates_from_pos(symbol, h1_tf, 0, 3)
        if peek_rates is not None and len(peek_rates) >= 2:
            latest_closed_ts = int(peek_rates[-2]['time'])
            with _last_candle_lock:
                if _last_candle_ts.get(symbol) == latest_closed_ts:
                    return  # already scanned this candle
                _last_candle_ts[symbol] = latest_closed_ts

    # Fetch primary timeframe data (M1 in scalp mode, H1 in standard mode)
    h1_rates = mt5.copy_rates_from_pos(symbol, h1_tf, 0, 200)
    if h1_rates is None or len(h1_rates) < 55:
        logger.warning(f"[STRATEGY] {symbol}: insufficient primary TF data")
        return
    h1_np = mt5_rates_to_numpy(h1_rates)

    # Fetch confirm timeframe data (M5 in scalp mode, M15 in standard mode)
    m15_tf = TF_MAP.get(CONFIG.get('timeframe_confirm', 'M15'), mt5.TIMEFRAME_M15)
    m15_rates = mt5.copy_rates_from_pos(symbol, m15_tf, 0, 200)
    if m15_rates is None or len(m15_rates) < 25:
        logger.warning(f"[STRATEGY] {symbol}: insufficient confirm TF data")
        return
    m15_np = mt5_rates_to_numpy(m15_rates)

    # Calculate H1 ATR for spread filter
    h1_atr = ATR(h1_np[:, 2], h1_np[:, 3], h1_np[:, 4], cfg.get('atr_period', 14))
    max_spread_mult = cfg.get('max_spread_atr_mult', 0.3)
    if h1_atr > 0 and spread > h1_atr * max_spread_mult:
        dc = get_ml_dc()
        if dc:
            row = build_ml_row(symbol, 'HOLD', 'SPREAD_FILTER', 0, h1_atr, 0, h1_np)
            row['hold_reason'] = f'spread_{spread}_vs_atr_{h1_atr:.5f}'
            dc.log_signal(row)
        return

    # Detect market regime
    # For M1 scalping: use M5 data (m15_np) for regime detection — M1 ADX is too noisy
    is_m1_mode = CONFIG.get('timeframe_primary', 'H1') == 'M1'
    regime_data = m15_np if (is_m1_mode and m15_np is not None and len(m15_np) >= 55) else h1_np
    regime, adx_val, direction = detect_regime(regime_data, cfg)
    logger.info(f"[SCAN] {symbol}: regime={regime}, ADX={adx_val:.1f}, direction={direction}, m1_mode={is_m1_mode}")

    signal = None
    indicators = {}
    hold_reasons = []

    if is_m1_mode:
        # === M1 SCALPING MODE ===
        # Only use BB_FADE (works on M5 statistical range) and EMA_PULLBACK (M5 confirm)
        # Price Action and Breakout are noise generators on M1 — disabled

        if regime == 'trending':
            # EMA Pullback only — uses M5 for entry timing
            signal, indicators, hold_reasons = check_ema_pullback(h1_np, m15_np, cfg)

        elif regime == 'ranging':
            # BB Fade on M5 data — best fit for M1 scalping
            signal, indicators, hold_reasons = check_bb_fade(m15_np, cfg)

        # If no signal from regime-gated triggers, try BB_FADE regardless of regime
        # (on M1, regime flips too fast to be a reliable gate)
        if signal is None and regime == 'trending':
            bb_signal, bb_ind, bb_reasons = check_bb_fade(m15_np, cfg)
            if bb_signal:
                signal, indicators, hold_reasons = bb_signal, bb_ind, bb_reasons
                logger.info(f"[SCAN] {symbol}: BB_FADE fired despite trending regime (M1 regime override)")

    else:
        # === STANDARD MODE (H1/M15/M30/H4) ===
        # === TRENDING: Check Price Action, Breakout, EMA Pullback ===
        if regime == 'trending':
            # Trigger 1: Price Action (Cobra style)
            signal, indicators, hold_reasons = check_price_action(h1_np, cfg)

            # Trigger 2: Breakout (Anaconda style) — only if no price action signal
            if signal is None:
                signal, indicators, hold_reasons = check_breakout(h1_np, cfg)

            # Trigger 3: EMA Pullback (Viper style) — M15 confirm
            if signal is None:
                signal, indicators, hold_reasons = check_ema_pullback(h1_np, m15_np, cfg)

        # === RANGING: Check BB Fade ===
        elif regime == 'ranging':
            # Trigger 4: BB Fade (Mamba style) — M15
            signal, indicators, hold_reasons = check_bb_fade(m15_np, cfg)

    # === LOG HOLD ===
    if signal is None:
        dc = get_ml_dc()
        if dc:
            row = build_ml_row(symbol, 'HOLD', 'NONE', 0, h1_atr, 0, h1_np)
            row['regime'] = regime
            row['adx'] = adx_val
            row['hold_reason'] = ';'.join(hold_reasons[:5]) if hold_reasons else 'no_trigger'
            with _prop_lock:
                row['risk_mode'] = _prop_firm_state['risk_mode']
            dc.log_signal(row)
        return

    # === GATE 0: NEWS BLACKOUT ===
    if CONFIG.get('news_blackout_enabled', True) and should_skip_trade(symbol):
        logger.info(f"[NEWS] {symbol}: news blackout active — skipping entry")
        return

    # === GATE 1: M5 TREND GATE ===
    # m15_np is the confirm-timeframe data — equals M5 bars when timeframe_confirm='M5'
    # (set in propfirm config for M1 scalping mode).
    m5_gate_cfg = CONFIG.get('m5_trend_gate', {})
    if m5_gate_cfg.get('enabled', False):
        passes, m5_trend = check_m5_trend_gate(signal['direction'], m15_np, m5_gate_cfg)
        if not passes:
            logger.info(
                f"[M5-GATE] {symbol}: M5 trend={m5_trend} opposes M1 signal "
                f"{signal['direction']} via {signal['trigger']} — suppressed"
            )
            dc = get_ml_dc()
            if dc:
                row = build_ml_row(symbol, 'BLOCKED', 'M5_GATE', signal['confidence'], h1_atr, 0, h1_np)
                row['hold_reason'] = f'm5_trend_{m5_trend}_vs_signal_{signal["direction"]}'
                row['regime'] = regime
                dc.log_signal(row)
            return
        logger.info(f"[M5-GATE] {symbol}: M5 trend={m5_trend} aligns — pass")

    # === GATE 2: VIPER ALIGNMENT GATE (ML-filtered M5 positions as trend proxy) ===
    viper_cfg = CONFIG.get('viper_gate', {})
    if viper_cfg.get('enabled', False):
        if not check_viper_alignment(symbol, signal['direction'], viper_cfg):
            dc = get_ml_dc()
            if dc:
                row = build_ml_row(symbol, 'BLOCKED', 'VIPER_GATE', signal['confidence'], h1_atr, 0, h1_np)
                row['hold_reason'] = f'viper_opposes_{signal["direction"]}'
                row['regime'] = regime
                dc.log_signal(row)
            return

    # === SIGNAL FIRED — Execute ===
    # Acquire per-symbol lock to prevent two threads/processes racing into the
    # same symbol simultaneously.  We re-validate position count inside the lock
    # so a duplicate from a second concurrent scan is caught and discarded.
    entry_lock = _get_symbol_entry_lock(symbol)
    if not entry_lock.acquire(blocking=False):
        logger.debug(f"[STRATEGY] {symbol}: entry lock busy — skipping duplicate signal")
        return

    try:
        # Re-check position count now that we hold the lock
        can_still_trade, guard_reason = prop_firm_can_trade(symbol)
        if not can_still_trade:
            logger.debug(f"[STRATEGY] {symbol}: re-check blocked post-lock: {guard_reason}")
            return

        trigger = signal['trigger']
        direction = signal['direction']
        confidence = signal['confidence']
        rr_target = signal.get('rr_target', 1.5)

        logger.info(
            f"[SIGNAL] {symbol} {direction} via {trigger} | "
            f"confidence={confidence:.2f} | R:R={rr_target} | "
            f"reason={signal['reason']}"
        )

        # Calculate SL and TP
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return

        entry_price = tick.ask if direction == 'BUY' else tick.bid
        digits = get_symbol_digits(symbol)
        sl_mult = cfg.get('atr_sl_multiplier', 1.5)

        if trigger == 'BB_FADE':
            # For BB fade, SL beyond the band + ATR buffer
            m15_atr = ATR(m15_np[:, 2], m15_np[:, 3], m15_np[:, 4], 14)
            if direction == 'BUY':
                sl = round(entry_price - m15_atr * sl_mult, digits)
                tp = round(signal.get('tp_price', entry_price + m15_atr * rr_target * sl_mult), digits)
            else:
                sl = round(entry_price + m15_atr * sl_mult, digits)
                tp = round(signal.get('tp_price', entry_price - m15_atr * rr_target * sl_mult), digits)
            sl_distance = abs(entry_price - sl)
        elif trigger == 'EMA_PULLBACK':
            # M15 ATR for tighter stop
            m15_atr = ATR(m15_np[:, 2], m15_np[:, 3], m15_np[:, 4], 14)
            if direction == 'BUY':
                sl = round(entry_price - m15_atr * sl_mult, digits)
                tp = round(entry_price + m15_atr * sl_mult * rr_target, digits)
            else:
                sl = round(entry_price + m15_atr * sl_mult, digits)
                tp = round(entry_price - m15_atr * sl_mult * rr_target, digits)
            sl_distance = abs(entry_price - sl)
        else:
            # H1 ATR for Price Action and Breakout
            if direction == 'BUY':
                sl = round(entry_price - h1_atr * sl_mult, digits)
                tp = round(entry_price + h1_atr * sl_mult * rr_target, digits)
            else:
                sl = round(entry_price + h1_atr * sl_mult, digits)
                tp = round(entry_price - h1_atr * sl_mult * rr_target, digits)
            sl_distance = abs(entry_price - sl)

        # Enforce config min_stop_points (critical for M1 where ATR is tiny)
        sym_info = mt5.symbol_info(symbol)
        min_stop_pts = cfg.get('min_stop_points', 0)
        if min_stop_pts > 0 and sym_info:
            min_cfg_dist = min_stop_pts * sym_info.point
            if sl_distance < min_cfg_dist:
                logger.info(f"[STRATEGY] {symbol}: SL distance {sl_distance:.5f} < min_stop_points {min_cfg_dist:.5f}, enforcing minimum")
                sl_distance = min_cfg_dist
                if direction == 'BUY':
                    sl = round(entry_price - sl_distance, digits)
                else:
                    sl = round(entry_price + sl_distance, digits)

        # Enforce fixed_tp_points (hard TP target in points)
        fixed_tp_pts = cfg.get('fixed_tp_points', 0)
        if fixed_tp_pts > 0 and sym_info:
            tp_dist = fixed_tp_pts * sym_info.point
            if direction == 'BUY':
                tp = round(entry_price + tp_dist, digits)
            else:
                tp = round(entry_price - tp_dist, digits)
            logger.info(f"[STRATEGY] {symbol}: Fixed TP at {fixed_tp_pts} points = {tp}")

        # Enforce broker minimum stop distance
        if sym_info:
            min_stop_dist = sym_info.trade_stops_level * sym_info.point
            if min_stop_dist > 0:
                if abs(entry_price - sl) < min_stop_dist:
                    if direction == 'BUY':
                        sl = round(entry_price - min_stop_dist, digits)
                    else:
                        sl = round(entry_price + min_stop_dist, digits)
                    sl_distance = abs(entry_price - sl)
                    logger.debug(f"[STRATEGY] {symbol}: SL adjusted to broker min stop distance ({min_stop_dist:.5f})")
                if abs(entry_price - tp) < min_stop_dist:
                    if direction == 'BUY':
                        tp = round(entry_price + min_stop_dist, digits)
                    else:
                        tp = round(entry_price - min_stop_dist, digits)
                    logger.debug(f"[STRATEGY] {symbol}: TP adjusted to broker min stop distance ({min_stop_dist:.5f})")

        # Recalculate lot size after potential SL adjustment
        volume = calculate_lot_size(symbol, sl_distance, cfg)
        if volume <= 0:
            logger.warning(f"[STRATEGY] {symbol}: volume calculation returned 0")
            return

        # Final execute
        ticket = execute_trade(symbol, direction, sl, tp, volume, trigger, signal['reason'], cfg)
        if ticket:
            # Update cooldown timestamp while still holding entry lock
            with state_lock:
                last_entry_time[symbol] = time.time()

            # Store ATR for trailing reference
            with positions_lock:
                if ticket in _tracked_positions:
                    atr_val = h1_atr if trigger not in ('BB_FADE', 'EMA_PULLBACK') else ATR(m15_np[:, 2], m15_np[:, 3], m15_np[:, 4], 14)
                    _tracked_positions[ticket]['atr'] = atr_val

            # Log executed signal
            dc = get_ml_dc()
            if dc:
                row = build_ml_row(symbol, direction, trigger, confidence, h1_atr, 1, h1_np)
                row.update({
                    'entry_price': entry_price, 'sl_price': sl, 'sl': sl,
                    'tp_price': tp, 'tp': tp, 'volume_lots': volume,
                    'ticket': ticket, 'regime': regime, 'adx': adx_val,
                })
                with _prop_lock:
                    row['risk_mode'] = _prop_firm_state['risk_mode']
                row.update(indicators)
                dc.log_signal(row)

    finally:
        entry_lock.release()

# ============================================================================
# FLASK API ENDPOINTS
# ============================================================================
@app.route('/status', methods=['GET'])
def status():
    acc = get_account_info()
    with _prop_lock:
        prop_state = dict(_prop_firm_state)
    return jsonify({
        'bot': BOT_NAME,
        'account': ACCOUNT_NUMBER,
        'status': 'halted' if prop_state['halted'] else 'running',
        'risk_mode': prop_state['risk_mode'],
        'prop_firm': prop_state,
        'account_info': acc,
        'tracked_positions': len(_tracked_positions),
    })

@app.route('/positions', methods=['GET'])
def get_positions():
    positions = mt5.positions_get()
    if not positions:
        return jsonify({'positions': [], 'count': 0})
    pos_list = []
    for p in positions:
        if p.magic == ACCOUNT_NUMBER:
            pos_list.append({
                'ticket': p.ticket, 'symbol': p.symbol,
                'type': 'BUY' if p.type == 0 else 'SELL',
                'volume': p.volume, 'price_open': p.price_open,
                'sl': p.sl, 'tp': p.tp, 'profit': p.profit,
                'swap': getattr(p, 'swap', 0.0),
                'commission': getattr(p, 'commission', 0.0),
            })
    return jsonify({'positions': pos_list, 'count': len(pos_list)})

@app.route('/account', methods=['GET'])
def account():
    acc = get_account_info()
    return jsonify(acc or {'error': 'no account info'})

@app.route('/start', methods=['POST'])
def start():
    global _strategy_thread, _trailing_thread
    with _prop_lock:
        _prop_firm_state['halted'] = False
        _prop_firm_state['halt_reason'] = ''
    bot_stop.clear()
    if _strategy_thread is None or not _strategy_thread.is_alive():
        _strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
        _strategy_thread.start()
    if _trailing_thread is None or not _trailing_thread.is_alive():
        _trailing_thread = threading.Thread(target=trailing_loop, daemon=True)
        _trailing_thread.start()
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop():
    bot_stop.set()
    return jsonify({'status': 'stopped'})

@app.route('/close_all', methods=['POST'])
def close_all():
    _close_all_positions("API_CLOSE_ALL")
    return jsonify({'status': 'closed_all'})

@app.route('/analyze/<symbol>', methods=['GET'])
def analyze(symbol):
    cfg = CONFIG.get('symbols', {}).get(symbol, {})
    if not cfg:
        return jsonify({'error': f'Symbol {symbol} not configured'}), 404

    h1_tf = TF_MAP.get(CONFIG.get('timeframe_primary', 'H1'), mt5.TIMEFRAME_H1)
    h1_rates = mt5.copy_rates_from_pos(symbol, h1_tf, 0, 200)
    if h1_rates is None or len(h1_rates) < 55:
        return jsonify({'error': 'Insufficient H1 data'}), 500
    h1_np = mt5_rates_to_numpy(h1_rates)

    m15_tf = TF_MAP.get(CONFIG.get('timeframe_confirm', 'M15'), mt5.TIMEFRAME_M15)
    m15_rates = mt5.copy_rates_from_pos(symbol, m15_tf, 0, 200)
    m15_np = mt5_rates_to_numpy(m15_rates) if m15_rates is not None and len(m15_rates) >= 25 else None

    regime, adx_val, direction = detect_regime(h1_np, cfg)
    h1_atr = ATR(h1_np[:, 2], h1_np[:, 3], h1_np[:, 4], 14)
    rsi_val = RSI(h1_np[:, 4], 14)
    spread = get_spread_points(symbol)

    analysis = {
        'symbol': symbol,
        'regime': regime,
        'adx': adx_val,
        'direction': direction,
        'h1_atr': h1_atr,
        'rsi': rsi_val,
        'spread': spread,
        'in_session': is_in_session(symbol, cfg),
        'risk_mode': _prop_firm_state.get('risk_mode', 'normal'),
    }

    # Check all triggers
    if regime == 'trending':
        pa_sig, _, _ = check_price_action(h1_np, cfg)
        bo_sig, _, _ = check_breakout(h1_np, cfg)
        pb_sig = None
        if m15_np is not None:
            pb_sig, _, _ = check_ema_pullback(h1_np, m15_np, cfg)
        analysis['triggers'] = {
            'price_action': pa_sig,
            'breakout': bo_sig,
            'ema_pullback': pb_sig,
        }
    elif regime == 'ranging' and m15_np is not None:
        bb_sig, _, _ = check_bb_fade(m15_np, cfg)
        analysis['triggers'] = {'bb_fade': bb_sig}

    return jsonify(analysis)

@app.route('/prop_firm', methods=['GET'])
def prop_firm_status():
    acc = get_account_info()
    with _prop_lock:
        state = dict(_prop_firm_state)
    if acc:
        state['current_equity'] = acc['equity']
        state['current_balance'] = acc['balance']
        daily_pnl = acc['equity'] - state.get('daily_start_equity', acc['equity'])
        state['daily_pnl'] = daily_pnl
        state['daily_pnl_pct'] = (daily_pnl / state.get('initial_balance', 100000)) * 100
        dd = acc['equity'] - state.get('peak_equity', acc['equity'])
        state['trailing_dd'] = dd
        state['trailing_dd_pct'] = (dd / state.get('initial_balance', 100000)) * 100
        total_profit = acc['equity'] - state.get('initial_balance', 100000)
        state['total_profit'] = total_profit
        state['total_profit_pct'] = (total_profit / state.get('initial_balance', 100000)) * 100
        state['target_amount'] = state.get('initial_balance', 100000) * 1.06
        state['remaining_to_target'] = state['target_amount'] - acc['equity']
    return jsonify(state)

# ============================================================================
# INITIALIZATION & MAIN
# ============================================================================
def load_config(config_path, profile_name=None):
    global CONFIG, ACCOUNT_NUMBER, PROFILE_NAME
    with open(config_path, 'r') as f:
        raw = json.load(f)
    if profile_name and profile_name in raw:
        PROFILE_NAME = profile_name
    else:
        PROFILE_NAME = list(raw.keys())[0]
    CONFIG = raw[PROFILE_NAME]
    ACCOUNT_NUMBER = CONFIG['account_number']

def init_mt5():
    """Initialize MT5 connection. Uses the already-logged-in terminal session."""
    mt5_path = CONFIG.get('mt5_path', '')

    if not mt5.initialize(path=mt5_path):
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    acc = mt5.account_info()
    if acc:
        logger.info(f"Connected: account={acc.login}, balance=${acc.balance:.2f}, equity=${acc.equity:.2f}")

        # Recover already-realised P&L for the current trading day so a restart
        # doesn't wipe the loss counter and let the bot trade through the daily limit.
        day_start_dt = get_trading_day_start()
        now_dt = datetime.now(timezone.utc)
        try:
            deals = mt5.history_deals_get(day_start_dt, now_dt)
            realized_today = 0.0
            if deals:
                for d in deals:
                    # entry=0 in/out=1 deals carry the realised P&L
                    if d.entry == 1:
                        realized_today += d.profit + d.commission + d.swap
                logger.info(f"[PROP] Realized P&L since {day_start_dt.strftime('%H:%M UTC')}: ${realized_today:+.2f} ({len(deals)} deals)")
            else:
                logger.info(f"[PROP] No deals found since {day_start_dt.strftime('%H:%M UTC')} — assuming clean start")
        except Exception as e:
            realized_today = 0.0
            logger.warning(f"[PROP] Could not fetch deal history on startup: {e}")

        # daily_start_equity = current equity MINUS what was already made/lost today
        # e.g. equity=$97,495  realized=-$2,504  →  start=$100,000
        recovered_start_equity = acc.equity - realized_today

        initial_cap = CONFIG.get('prop_firm', {}).get('initial_capital', acc.balance)

        # Peak equity must be at least initial_capital — the prop firm tracks
        # trailing DD from the highest equity ever reached, which started at
        # initial_capital.  On restart we reconstruct peak from deal history:
        # peak = max(initial_capital, highest equity seen today).
        # Since we know today's starting equity and the max intra-day P&L,
        # the conservative safe floor is max(initial_capital, current equity).
        # If the account ever went ABOVE initial_capital on a prior day, the
        # prop firm's own tracker holds the true peak — we use initial_capital
        # as our floor so trailing DD is never under-counted.
        peak_equity = max(initial_cap, acc.equity)

        with _prop_lock:
            _prop_firm_state['initial_balance'] = initial_cap
            _prop_firm_state['peak_equity'] = peak_equity
            _prop_firm_state['daily_start_equity'] = recovered_start_equity
            _prop_firm_state['daily_start_date'] = get_trading_day_key()
            _prop_firm_state['last_daily_reset'] = day_start_dt.isoformat()

        logger.info(f"[PROP] Daily start equity recovered: ${recovered_start_equity:.2f} "
                    f"(current ${acc.equity:.2f}, realized today ${realized_today:+.2f})")
        logger.info(f"[PROP] Peak equity set to: ${peak_equity:.2f} "
                    f"(initial_capital=${initial_cap:.2f}, current=${acc.equity:.2f})")
    else:
        logger.error("Could not get account info")
        sys.exit(1)

    # Enable symbols (only those marked enabled in config)
    for symbol, sym_cfg in CONFIG.get('symbols', {}).items():
        if isinstance(sym_cfg, dict) and not sym_cfg.get('enabled', True):
            logger.info(f"Skipped symbol: {symbol} (disabled)")
            continue
        mt5.symbol_select(symbol, True)
        logger.info(f"Enabled symbol: {symbol}")

def main():
    global _strategy_thread, _trailing_thread

    # Prevent two processes from running simultaneously
    _acquire_instance_lock()

    parser = argparse.ArgumentParser(description='HYDRA Prop Firm Bot')
    parser.add_argument('--config', default=None, help='Config file path (overrides --profile)')
    parser.add_argument('--profile', default=None, help='Profile name (e.g. propfirm). Loads hydra_config_<profile>.json')
    parser.add_argument('--port', type=int, default=None, help='Override Flask port')
    args = parser.parse_args()

    # Resolve config file: explicit --config beats --profile beats default
    if args.config:
        config_file = args.config
    elif args.profile:
        config_file = f'hydra_config_{args.profile}.json'
    else:
        config_file = 'hydra_config.json'

    # Load config first so ACCOUNT_NUMBER is set before logger filename is created
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    load_config(config_path, profile_name=args.profile)

    init_logger()

    logger.info("=" * 70)
    logger.info("  HYDRA v1 — Prop Firm Challenge Killer")
    logger.info("  Multi-Strategy Hybrid Bot")
    logger.info("=" * 70)
    logger.info(f"Loaded config profile: {PROFILE_NAME}, account: {ACCOUNT_NUMBER}")

    # Init MT5
    init_mt5()

    # Init ML data collector
    dc = SharedDataCollector(BOT_NAME, ACCOUNT_NUMBER)
    set_ml_dc(dc)

    # Init trade logger
    tl = TradeDecisionLogger(BOT_NAME)
    set_trade_logger(tl)

    # Start strategy thread
    _strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
    _strategy_thread.start()

    # Start trailing thread
    _trailing_thread = threading.Thread(target=trailing_loop, daemon=True)
    _trailing_thread.start()

    # Start Flask
    port = args.port or CONFIG.get('server_port', 8060)
    logger.info(f"Flask API starting on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == '__main__':
    main()
