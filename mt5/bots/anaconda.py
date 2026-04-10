"""
ANACONDA v1 — Strict Breakout Bot (H4)
Account B: 24480388, Port 8051

ExMachina-inspired 5-condition breakout system.
Every condition must align — trades rarely but with high conviction.

CONDITIONS (ALL must be true):
  1. EMA trend direction (fast > slow or fast < slow)
  2. Trend strength — EMAs separated by minimum ATR distance
  3. Price position — close above/below both EMAs
  4. N-bar breakout — new high/low with ATR buffer
  5. RSI in healthy zone (40-65 buy, 35-60 sell — not extreme)

Exits: Fixed 2:1 R:R TP + ATR trailing stop + breakeven at 1x ATR
Timeframe: H4 (~2-5 trades per week)
Risk: 1% per trade, 2x ATR stop, max 1 position per symbol
"""
import sys
import os
import json
import time
import threading
import logging
import argparse
import math
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pro_modules'))

import requests as http_requests
import MetaTrader5 as mt5
import numpy as np
from mt5_broker import MT5Broker
from ultra_fast_indicators import ema_numba, rsi_numba, atr_numba
from risk_manager_ultra import RiskManagerUltra
from trade_logger import TradeDecisionLogger
from shared_data_collector import SharedDataCollector
from portfolio_risk_guard import PortfolioRiskGuard
from news_guard import should_skip_trade
from prop_firm_guard import PropFirmGuard

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
BOT_NAME = 'anaconda'
API_KEY = os.environ.get("ANACONDA_API_KEY", "")

state_lock = threading.Lock()
last_entry_time = {}
bot_stop = threading.Event()
_strategy_thread = None
_trailing_thread = None

app = Flask(__name__)

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
    logger = logging.getLogger(f"ANACONDA-{ACCOUNT_NUMBER}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"anaconda_{ACCOUNT_NUMBER}.log", encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('[GPU] CuPy loaded - GPU acceleration enabled' if GPU_AVAILABLE else '[GPU] CuPy not available - using CPU')

# ============================================================================
# SINGLETONS
# ============================================================================
_broker = None
_risk_manager = None
_trade_logger = None

def get_daily_reset_boundary(now=None):
    now_utc = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    reset_time = now_utc.replace(hour=20, minute=0, second=0, microsecond=0)
    if now_utc < reset_time:
        reset_time -= timedelta(days=1)
    return reset_time

def get_broker(): return _broker
def set_broker(b):
    global _broker
    _broker = b

def get_risk_manager(): return _risk_manager
def set_risk_manager(rm):
    global _risk_manager
    _risk_manager = rm

def get_trade_logger(): return _trade_logger
def set_trade_logger(tl):
    global _trade_logger
    _trade_logger = tl


_portfolio_guard = None
def get_portfolio_guard(): return _portfolio_guard
def set_portfolio_guard(pg):
    global _portfolio_guard
    _portfolio_guard = pg

_prop_guard = None
def get_prop_guard(): return _prop_guard
def set_prop_guard(pg):
    global _prop_guard
    _prop_guard = pg

def notify_portfolio_trade_open(ticket, symbol, side, volume):
    pg = get_portfolio_guard()
    if pg and ticket is not None:
        try:
            pg.on_trade_open(int(ticket), symbol, side, float(volume))
        except Exception as e:
            logger.warning(f"Portfolio open update failed: {e}")

def notify_portfolio_trade_close(ticket, profit, duration_seconds, symbol):
    pg = get_portfolio_guard()
    if pg and ticket is not None:
        try:
            pg.on_trade_close(int(ticket), float(profit), float(duration_seconds), symbol)
        except Exception as e:
            logger.warning(f"Portfolio close update failed: {e}")

_ml_dc = None
def get_ml_dc(): return _ml_dc
def set_ml_dc(dc):
    global _ml_dc
    _ml_dc = dc

# Position tracking for ML outcome logging
_tracked_positions = {}  # ticket -> {symbol, entry_price, sl, tp, volume, open_time}
positions_lock = threading.Lock()
_pending_close_checks = {}
pending_close_lock = threading.Lock()

# ============================================================================
# ML HELPERS
# ============================================================================
def build_ml_row(symbol, signal, trigger, confidence, atr, executed, rates_np=None):
    """Build standardized ML logging row with all fields."""
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
        'timeframe':     CONFIG.get('timeframe', 'H4'),
        'signal':        signal,
        'signal_type':   signal,
        'trigger':       trigger,
        'confidence':    confidence,
        'atr':           atr,
        'executed':      executed,
        # Entry details (filled for EXECUTED/OUTCOME)
        'entry_price':   None,
        'sl_price':      None,
        'sl':            None,
        'tp_price':      None,
        'tp':            None,
        'volume_lots':   None,
        'ticket':        None,
        # Exit details (filled for OUTCOME)
        'exit_price':    None,
        'exit_reason':   None,
        'profit':        None,
        'pnl':           None,
        'outcome':       None,
        'duration_minutes': None,
        # Account context (filled for EXECUTED/OUTCOME)
        'account_balance': None,
        'account_equity':  None,
        # Market context
        'spread':       spread_val,
        'spread_points': spread_val,
        'bar_open':      bar_open,
        'bar_high':      bar_high,
        'bar_low':       bar_low,
        'bar_close':     bar_close,
        # Strategy-specific indicators (None = not applicable)
        # Anaconda-specific
        'ema_fast':              None,
        'ema_slow':              None,
        'ema_separation':        None,
        'rsi':                   None,
        'n_bar_high':            None,
        'n_bar_low':             None,
        'conditions_met':        None,
        # Cobra-specific
        'ema_50':                None,
        'nearest_sr_level':      None,
        'sr_distance_atr':       None,
        'pattern_type':          None,
        'pin_wick_size':         None,
        'pin_body_ratio':        None,
        'engulf_size_atr':       None,
        'inside_bar_vol_ratio':  None,
        'pattern_detected':      None,
        'sr_nearby':             None,
        # Viper-specific
        'ema_20':                None,
        'momentum_fast_ema':     None,
        'momentum_slow_ema':     None,
        'breakout_high':         None,
        'breakout_low':          None,
        'pullback_distance_atr': None,
        'volume_ratio':          None,
        'h1_trend':              None,
        # Mamba-specific
        'bb_upper':              None,
        'bb_lower':              None,
        'bb_mid':                None,
        'price_position_in_bb':  None,
        'adx':                   None,
        # Hold reason
        'hold_reason':           None,
    }

def get_spread_points(symbol):
    """Get current spread in price points."""
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return round(tick.ask - tick.bid, get_symbol_digits(symbol))
        return None
    except Exception:
        return None

def get_account_balance_equity():
    """Get current account balance and equity."""
    try:
        info = mt5.account_info()
        if info:
            return info.balance, info.equity
        return None, None
    except Exception:
        return None, None

def determine_exit_reason(entry_price, close_price, sl, tp, symbol):
    """Determine why a position closed based on exit price vs TP/SL."""
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

def log_trade_outcome(symbol, ticket, entry_price, sl, tp, volume, close_price, profit, exit_reason, duration_minutes, atr=0, account_balance=None, account_equity=None):
    """Log closed trade outcome to ML data collector and risk manager."""
    try:
        outcome = 'WIN' if profit > 0 else ('LOSS' if profit < 0 else 'BREAKEVEN')
        dc = get_ml_dc()
        if dc:
            row = build_ml_row(symbol, 'TRADE_CLOSED', 'OUTCOME', 0, atr, 0)
            row.update({
                'entry_price':      entry_price,
                'sl_price':         sl,
                'sl':               sl,
                'tp_price':         tp,
                'tp':               tp,
                'volume_lots':      volume,
                'ticket':           ticket,
                'exit_price':       close_price,
                'exit_reason':      exit_reason,
                'profit':           profit,
                'pnl':              profit,
                'outcome':          outcome,
                'duration_minutes': duration_minutes,
                'account_balance':  account_balance,
                'account_equity':   account_equity,
                'spread_points':    get_spread_points(symbol),
            })
            dc.log_signal(row)

        rm = get_risk_manager()
        if rm:
            rm.on_trade_close(int(ticket), float(profit), float(duration_minutes) * 60.0, symbol)
        notify_portfolio_trade_close(ticket, profit, float(duration_minutes) * 60.0, symbol)

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

TF_MAP = {
    'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1,
}

# ============================================================================
# UTILITY
# ============================================================================
def mt5_rates_to_numpy(rates):
    if hasattr(rates, 'dtype') and rates.dtype.names:
        return np.column_stack([
            rates['time'].astype(np.float64), rates['open'].astype(np.float64),
            rates['high'].astype(np.float64), rates['low'].astype(np.float64),
            rates['close'].astype(np.float64), rates['tick_volume'].astype(np.float64),
        ])
    return np.array(rates, dtype=np.float64)

def safe_last(val):
    if val is None:
        raise ValueError("safe_last: received None indicator value")
    if isinstance(val, np.ndarray):
        if len(val) == 0:
            raise ValueError("safe_last: received empty indicator array")
        return float(val[-1])
    return float(val)

def normalize_pos_type(t):
    if t in ('BUY', 'buy', 0): return 'BUY'
    if t in ('SELL', 'sell', 1): return 'SELL'
    return None

def get_symbol_digits(symbol):
    """Get broker's decimal places for a symbol (e.g., 5 for EURUSD, 2 for XAUUSD, 3 for USDJPY)."""
    try:
        info = mt5.symbol_info(symbol)
        return info.digits if info else 5
    except Exception:
        return 5

def serialize_result(result):
    """Convert MT5 result objects into JSON-safe data for Flask responses."""
    if result is None:
        return None
    if isinstance(result, (bool, int, float, str, dict, list)):
        return result
    data = {}
    for attr in ('retcode', 'order', 'deal', 'volume', 'price', 'bid', 'ask', 'comment', 'request_id'):
        if hasattr(result, attr):
            data[attr] = getattr(result, attr)
    return data or str(result)

WEBHOOK_URL = "https://dashboard.glitchexecutor.com/api/trades/webhook"

def send_webhook(payload):
    if not CONFIG.get('webhook_enabled', False):
        return False
    try:
        payload.update({"bot": "anaconda", "account": ACCOUNT_NUMBER,
            "timestamp": datetime.now(timezone.utc).isoformat()})
        resp = http_requests.post(WEBHOOK_URL, json=payload, timeout=2)
        data = resp.json()
        if data.get('status') == 'ok':
            return True
        logger.warning(f"[WEBHOOK] Unexpected response: {data}")
        return False
    except Exception as e:
        logger.debug(f"[WEBHOOK] Failed: {e}")
        return False

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
# 5-CONDITION BREAKOUT SIGNAL
# ============================================================================
def check_signal(rates_np, cfg):
    """
    ExMachina-style strict breakout. ALL 5 conditions must be true.

    Returns: (signal_dict or None, atr_value, indicators_dict)
      indicators_dict always contains anaconda-specific fields.
    """
    closes = rates_np[:, 4]
    highs = rates_np[:, 2]
    lows = rates_np[:, 3]

    # Config
    ema_fast_period = cfg.get('ema_fast_period', 50)
    ema_slow_period = cfg.get('ema_slow_period', 200)
    atr_period = cfg.get('atr_period', 14)
    breakout_lookback = cfg.get('breakout_lookback', 20)
    breakout_buffer_mult = cfg.get('breakout_buffer_mult', 0.3)
    trend_strength_mult = cfg.get('trend_strength_mult', 0.3)
    rsi_period = cfg.get('rsi_period', 10)
    rsi_buy_min = cfg.get('rsi_buy_min', 40)
    rsi_buy_max = cfg.get('rsi_buy_max', 65)
    rsi_sell_min = cfg.get('rsi_sell_min', 35)
    rsi_sell_max = cfg.get('rsi_sell_max', 60)

    min_bars = max(ema_slow_period, breakout_lookback, atr_period) + 10
    if len(closes) < min_bars:
        return None, 0.0, {}

    # Compute indicators
    ema_fast = EMA(closes, ema_fast_period)
    ema_slow = EMA(closes, ema_slow_period)
    rsi_vals = RSI(closes, rsi_period)
    curr_atr = safe_last(ATR(highs, lows, closes, atr_period))

    if curr_atr <= 0:
        return None, 0.0, {}

    # Values on last completed bar
    ema_f = float(ema_fast[-2])
    ema_s = float(ema_slow[-2])
    curr_rsi = float(rsi_vals[-2]) if isinstance(rsi_vals, np.ndarray) else float(rsi_vals)
    c1 = float(closes[-2])  # Last completed bar close
    c2 = float(closes[-3])  # Bar before that

    # -- CONDITION 1: EMA TREND DIRECTION --
    bull_trend = ema_f > ema_s
    bear_trend = ema_f < ema_s

    # -- CONDITION 2: TREND STRENGTH (EMA separation) --
    ema_separation = abs(ema_f - ema_s)
    min_separation = curr_atr * trend_strength_mult
    trend_strong = ema_separation >= min_separation

    # -- CONDITION 3: PRICE POSITION --
    above_both = c1 > ema_f and c1 > ema_s
    below_both = c1 < ema_f and c1 < ema_s

    # -- CONDITION 4: N-BAR BREAKOUT --
    lookback_highs = highs[-(breakout_lookback+3):-3]
    lookback_lows = lows[-(breakout_lookback+3):-3]
    n_bar_high = float(np.max(lookback_highs))
    n_bar_low = float(np.min(lookback_lows))
    buf = curr_atr * breakout_buffer_mult

    bull_breakout = c1 > (n_bar_high + buf) and c2 <= (n_bar_high + buf)
    bear_breakout = c1 < (n_bar_low - buf) and c2 >= (n_bar_low - buf)

    # -- CONDITION 5: RSI HEALTHY ZONE --
    rsi_buy_ok = rsi_buy_min <= curr_rsi <= rsi_buy_max
    rsi_sell_ok = rsi_sell_min <= curr_rsi <= rsi_sell_max

    # Count conditions met (for HOLD logging)
    conditions_met = sum([
        bull_trend or bear_trend,
        trend_strong,
        above_both or below_both,
        bull_breakout or bear_breakout,
        rsi_buy_ok or rsi_sell_ok
    ])

    indicators = {
        'ema_fast':       ema_f,
        'ema_slow':       ema_s,
        'ema_separation': ema_separation,
        'rsi':            curr_rsi,
        'n_bar_high':     n_bar_high,
        'n_bar_low':      n_bar_low,
        'conditions_met': conditions_met,
    }

    # -- COMBINE ALL 5 --
    if bull_trend and trend_strong and above_both and bull_breakout and rsi_buy_ok:
        return {
            'trigger': 'BREAKOUT_5C',
            'direction': 'BUY',
            'confidence': 0.90,
            'reason': (f'5/5 BUY: EMA trend, sep={ema_separation:.5f}, '
                      f'above EMAs, broke {breakout_lookback}-bar high {n_bar_high:.5f}, RSI={curr_rsi:.1f}')
        }, curr_atr, indicators

    if bear_trend and trend_strong and below_both and bear_breakout and rsi_sell_ok:
        return {
            'trigger': 'BREAKOUT_5C',
            'direction': 'SELL',
            'confidence': 0.90,
            'reason': (f'5/5 SELL: EMA trend, sep={ema_separation:.5f}, '
                      f'below EMAs, broke {breakout_lookback}-bar low {n_bar_low:.5f}, RSI={curr_rsi:.1f}')
        }, curr_atr, indicators

    # FALLBACK: EMA pullback entry for demo-mode trade collection.
    pullback_ema_fast = float(ema_fast[-1])
    pullback_ema_slow = float(ema_slow[-1])
    close_now = float(closes[-1])
    close_prev = float(closes[-2])
    pullback_rsi = float(rsi_vals[-1]) if isinstance(rsi_vals, np.ndarray) else float(rsi_vals)

    if pullback_ema_fast > pullback_ema_slow:
        if close_prev <= pullback_ema_fast * 1.002 and close_now > pullback_ema_fast:
            if 40 < pullback_rsi < 70:
                indicators['hold_reason'] = 'ema_pullback_buy'
                return {
                    'trigger': 'EMA_PULLBACK',
                    'direction': 'BUY',
                    'confidence': 0.72,
                    'reason': (
                        f'EMA pullback BUY: close {close_now:.5f} reclaimed EMA{ema_fast_period} '
                        f'{pullback_ema_fast:.5f}, trend intact, RSI={pullback_rsi:.1f}'
                    )
                }, curr_atr, indicators
    elif pullback_ema_fast < pullback_ema_slow:
        if close_prev >= pullback_ema_fast * 0.998 and close_now < pullback_ema_fast:
            if 30 < pullback_rsi < 60:
                indicators['hold_reason'] = 'ema_pullback_sell'
                return {
                    'trigger': 'EMA_PULLBACK',
                    'direction': 'SELL',
                    'confidence': 0.72,
                    'reason': (
                        f'EMA pullback SELL: close {close_now:.5f} rejected EMA{ema_fast_period} '
                        f'{pullback_ema_fast:.5f}, trend intact, RSI={pullback_rsi:.1f}'
                    )
                }, curr_atr, indicators

    # Build hold_reason string from which conditions failed
    failed = []
    if not (bull_trend or bear_trend): failed.append('no_ema_trend')
    if not trend_strong: failed.append('weak_separation')
    if not (above_both or below_both): failed.append('price_mixed_ema')
    if not (bull_breakout or bear_breakout): failed.append('no_breakout')
    if not (rsi_buy_ok or rsi_sell_ok): failed.append('rsi_out_of_zone')
    indicators['hold_reason'] = ','.join(failed) if failed else 'unknown'

    return None, curr_atr, indicators

# ============================================================================
# TRADING UTILITIES
# ============================================================================
def is_within_trading_hours(cfg):
    tf = cfg.get('time_filter', {})
    if not tf.get('enabled', False): return True
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5: return False
    friday_cutoff = tf.get('friday_cutoff_hour', 0)
    if friday_cutoff > 0 and now.weekday() == 4 and now.hour >= friday_cutoff:
        return False
    return tf.get('start_hour', 0) <= now.hour < tf.get('end_hour', 24)

def calculate_sl(direction, entry_price, atr_value, cfg):
    mult = cfg.get('atr_sl_multiplier', 2.0)
    offset = atr_value * mult
    min_stop_points = float(cfg.get('min_stop_points', 0) or 0)
    if min_stop_points > 0:
        try:
            broker = get_broker()
            sym_info = broker.get_symbol_info(cfg.get('_symbol_name')) if broker and cfg.get('_symbol_name') else None
            point = float(sym_info.get('point', 0)) if sym_info else 0.0
            if point > 0:
                offset = max(offset, point * min_stop_points)
        except Exception:
            pass
    return entry_price - offset if direction == 'buy' else entry_price + offset

def calculate_tp(direction, entry_price, atr_value, cfg):
    """TP at fixed R:R ratio based on ATR-derived SL distance.
    INTENTIONAL: TP is always set on every new H4 order (never 0.0).
    Any existing position showing tp=0.0 is a legacy position opened before
    this code was deployed. Primary exits are trailing stop and Friday flatten;
    TP acts as a hard cap safety net."""
    sl = calculate_sl(direction, entry_price, atr_value, cfg)
    sl_distance = abs(entry_price - sl)
    rr = cfg.get('min_reward_risk', 2.0)
    if direction == 'buy':
        return entry_price + (sl_distance * rr)
    else:
        return entry_price - (sl_distance * rr)

def calculate_position_size(symbol, cfg, sl_price, entry_price):
    broker = get_broker()
    account = broker.get_account_info()
    if account is None: return cfg.get('min_lot', 0.01)
    balance = account.get('balance', 1000)
    risk_pct = cfg.get('risk_percent', 1.0)
    risk_amount = balance * (risk_pct / 100)
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0: return cfg.get('min_lot', 0.01)
    try:
        sym_info = broker.get_symbol_info(symbol)
        if sym_info is None: return cfg.get('min_lot', 0.01)
        tick_value = sym_info.get('tick_value', 1)
        tick_size = sym_info.get('tick_size', 0.00001)
        if tick_size == 0 or tick_value == 0: return cfg.get('min_lot', 0.01)
        ticks_at_risk = sl_distance / tick_size
        dollar_risk = ticks_at_risk * tick_value  # dollar risk per 1 standard lot
        lots = risk_amount / dollar_risk if dollar_risk > 0 else cfg.get('min_lot', 0.01)
    except Exception as e:
        logger.warning(f"Position sizing error {symbol}: {e}")
        lots = cfg.get('min_lot', 0.01)
        dollar_risk = 0  # flag: can't compute actual risk
    min_lot = cfg.get('min_lot', 0.01)
    broker_max_lot = float(sym_info.get('volume_max') or 0.0)
    max_lot = broker_max_lot if broker_max_lot > 0 else max(float(lots), min_lot)
    raw_lots = lots
    clamped = round(max(min_lot, min(lots, max_lot)), 2)
    # Warn when min/max lot clamping causes effective risk to deviate significantly
    if dollar_risk > 0:
        actual_risk = clamped * dollar_risk
        if actual_risk > risk_amount * 1.5:
            logger.warning(
                f"[SIZING] {symbol}: target {risk_pct}% (${risk_amount:.2f}) "
                f"but actual ${actual_risk:.2f} ({actual_risk / balance * 100:.1f}%) "
                f"— raw lots {raw_lots:.4f} clamped to min_lot {clamped:.2f}. "
                f"Consider lowering risk_percent for {symbol}."
            )
        elif clamped == max_lot and actual_risk < risk_amount * 0.5:
            logger.warning(
                f"[SIZING] {symbol}: target {risk_pct}% (${risk_amount:.2f}) "
                f"but actual ${actual_risk:.2f} ({actual_risk / balance * 100:.1f}%) "
                f"— raw lots {raw_lots:.4f} clamped to max_lot {clamped:.2f}."
            )
    return clamped

# ============================================================================
# STRATEGY LOOP
# ============================================================================
def strategy_loop():
    broker = get_broker()
    timeframe = TF_MAP.get(CONFIG.get('timeframe', 'H4'), mt5.TIMEFRAME_H4)
    cooldown = CONFIG.get('entry_cooldown_seconds', 14400)  # 4 hours on H4
    iteration = 0
    _mt5_fail_count = 0
    MT5_RECONNECT_THRESHOLD = 3

    while not bot_stop.is_set():
        # PropFirmGuard update
        pfg = get_prop_guard()
        if pfg:
            broker_acc = broker.get_account_info() if broker else None
            if broker_acc:
                pfg.update(broker_acc.get('equity', 0), broker_acc.get('balance', 0))
                if pfg.should_flatten_friday():
                    positions = broker.get_positions() if broker else []
                    for pos in positions:
                        try:
                            broker.close_position(pos['ticket'])
                            logger.info(f"[PROP] Friday flatten: closed {pos['symbol']} #{pos['ticket']}")
                        except Exception as e:
                            logger.warning(f"[PROP] Friday flatten failed #{pos.get('ticket')}: {e}")

        symbols_cfg = CONFIG.get('symbols')
        if not symbols_cfg:
            logger.error("CONFIG missing 'symbols' key — check config file")
            time.sleep(300)
            continue
        for symbol, cfg in symbols_cfg.items():
            if not cfg.get('enabled', False): continue
            try:
                if not is_within_trading_hours(cfg):
                    if iteration % 10 == 0:
                        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
                        if rates is not None and len(rates) >= 50:
                            rates_np = mt5_rates_to_numpy(rates)
                            atr_value = safe_last(ATR(rates_np[:, 2], rates_np[:, 3], rates_np[:, 4], cfg.get('atr_period', 14)))
                            dc = get_ml_dc()
                            if dc:
                                try:
                                    row = build_ml_row(symbol, 'OUTSIDE_HOURS', 'NONE', 0, atr_value, 0, rates_np)
                                    row['spread_points'] = get_spread_points(symbol)
                                    dc.log_signal(row)
                                except Exception:
                                    pass
                            logger.info(f"[{symbol}] Outside trading hours (ML logged)")
                    continue
                with state_lock:
                    if time.time() - last_entry_time.get(symbol, 0) < cooldown: continue

                all_positions = broker.get_positions()
                pg = get_portfolio_guard()
                if pg:
                    try:
                        pg.sync_account_positions(all_positions)
                    except Exception as e:
                        if iteration % 60 == 0:
                            logger.warning(f"[PORTFOLIO-RISK] Sync failed: {e}")
                sym_positions = [p for p in all_positions if p['symbol'] == symbol]
                if len(sym_positions) >= cfg.get('max_positions', 1): continue

                rm = get_risk_manager()
                if rm:
                    can_trade, reason = rm.can_trade(symbol=symbol)
                    if not can_trade:
                        if iteration % 60 == 0: logger.warning(f"[RISK] Blocked: {reason}")
                        continue

                # Margin level gate
                min_margin = CONFIG.get('min_margin_level', 200)
                if min_margin > 0:
                    account_info = broker.get_account_info()
                    if account_info:
                        margin_level = account_info.get('margin_level', 9999)
                        if margin_level > 0 and margin_level < min_margin:
                            if iteration % 20 == 0:
                                logger.warning(
                                    f"[MARGIN] {symbol}: Margin level {margin_level:.0f}% < {min_margin}% — skipping entry"
                                )
                            continue

                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 300)
                if rates is None or len(rates) < 220:
                    _mt5_fail_count += 1
                    if _mt5_fail_count >= MT5_RECONNECT_THRESHOLD:
                        logger.warning(f"[{BOT_NAME}] MT5 {_mt5_fail_count} consecutive failures — reinitialising")
                        try:
                            mt5.shutdown()
                            mt5.initialize()
                            _mt5_fail_count = 0
                        except Exception as reinit_err:
                            logger.error(f"[{BOT_NAME}] MT5 reinit failed: {reinit_err}")
                    continue
                _mt5_fail_count = 0
                rates_np = mt5_rates_to_numpy(rates)

                signal, atr_value, indicators = check_signal(rates_np, cfg)

                if atr_value is None or math.isnan(atr_value) or atr_value <= 0:
                    logger.warning(f"[{BOT_NAME}] ATR invalid ({atr_value}) for {symbol} — skipping bar")
                    continue

                if signal:
                    logger.info(f">> {symbol}: {signal['direction']} via {signal['trigger']} -- {signal['reason']} ATR={atr_value:.5f}")
                elif iteration % 5 == 0:
                    logger.info(f"{symbol}: HOLD (conditions not met) ATR={atr_value:.5f}")

                # Log signal for ML (every check including HOLD)
                dc = get_ml_dc()
                if dc:
                    try:
                        row = build_ml_row(
                            symbol,
                            signal['direction'] if signal else 'HOLD',
                            signal['trigger'] if signal else 'NONE',
                            signal['confidence'] if signal else 0,
                            atr_value,
                            0,
                            rates_np
                        )
                        row.update({
                            'ema_fast':       indicators.get('ema_fast'),
                            'ema_slow':       indicators.get('ema_slow'),
                            'ema_separation': indicators.get('ema_separation'),
                            'rsi':            indicators.get('rsi'),
                            'n_bar_high':     indicators.get('n_bar_high'),
                            'n_bar_low':      indicators.get('n_bar_low'),
                            'conditions_met': indicators.get('conditions_met'),
                            'hold_reason':    indicators.get('hold_reason') if not signal else None,
                            'spread_points':  get_spread_points(symbol),
                        })
                        dc.log_signal(row)
                    except Exception:
                        pass

                if signal:
                    if CONFIG.get('news_blackout_enabled', True) and should_skip_trade(symbol):
                        logger.info(f"[{BOT_NAME}] News blackout active for {symbol} — skipping entry")
                        continue
                    direction = signal['direction'].lower()
                    pg = get_portfolio_guard()
                    if pg:
                        allowed, p_reason = pg.can_open(symbol, direction.upper())
                        if not allowed:
                            if iteration % 10 == 0:
                                logger.warning(f"[PORTFOLIO-RISK] Blocked {symbol} {direction.upper()}: {p_reason}")
                            continue
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None: continue
                    spread = tick.ask - tick.bid
                    if atr_value > 0 and spread > atr_value * 0.3:
                        logger.info(f"{symbol}: Spread too wide")
                        continue

                    entry_price = tick.ask if direction == 'buy' else tick.bid
                    cfg_runtime = dict(cfg)
                    cfg_runtime['_symbol_name'] = symbol
                    sl = calculate_sl(direction, entry_price, atr_value, cfg_runtime)
                    # INTENTIONAL: anaconda trades 4H trends and holds for the week.
                    # TP is deliberately not set — the trailing stop is the sole profit exit.
                    # Friday flatten closes any position still open at week end.
                    tp = 0.0
                    volume = calculate_position_size(symbol, cfg, sl, entry_price)

                    # Pre-flight SL check — must be on correct side of entry
                    if direction == "buy" and sl >= entry_price:
                        logger.error(f"[{BOT_NAME}] SL {sl} >= entry {entry_price} on BUY — aborting trade")
                        continue
                    if direction == "sell" and sl <= entry_price:
                        logger.error(f"[{BOT_NAME}] SL {sl} <= entry {entry_price} on SELL — aborting trade")
                        continue

                    # PropFirmGuard check
                    pfg = get_prop_guard()
                    if pfg:
                        positions = broker.get_positions() if broker else []
                        pf_allowed, pf_reason, pf_mult = pfg.can_trade(symbol, positions)
                        if not pf_allowed:
                            logger.info(f"[PROP] Blocked {symbol} {direction.upper()}: {pf_reason}")
                            if pf_reason == 'halted' or pf_reason.startswith('halted'):
                                for pos in positions:
                                    try:
                                        broker.close_position(pos['ticket'])
                                        logger.warning(f"[PROP] HALT close: {pos['symbol']} #{pos['ticket']}")
                                    except Exception:
                                        pass
                            continue
                        if pf_mult < 1.0:
                            volume = round(volume * pf_mult, 2)
                            volume = max(volume, cfg.get('min_lot', 0.01))

                    result = broker.open_position(symbol, direction, volume, sl, tp)
                    if result:
                        with state_lock:
                            last_entry_time[symbol] = time.time()
                        logger.info(f"[OK] {direction.upper()} {symbol} @ {entry_price:.5f} vol={volume} SL={sl:.5f} TP={tp:.5f} conf=0.90")
                        tl = get_trade_logger()
                        if tl:
                            try:
                                tl.log_entry(symbol=symbol, direction=direction, price=entry_price, size=volume,
                                    decision_factors={"trigger": "BREAKOUT_5C", "confidence": 0.90,
                                        "reason": signal['reason'], "atr": atr_value, "account": ACCOUNT_NUMBER},
                                    market_context={"entry_price": entry_price, "sl": sl, "tp": tp})
                            except Exception as e:
                                logger.warning(f"Trade log failed: {e}")

                        # Extract ticket from result
                        ticket_num = None
                        try:
                            ticket_num = result.order
                        except Exception:
                            pass

                        rm = get_risk_manager()
                        if rm and ticket_num is not None:
                            try:
                                rm.on_trade_open(int(ticket_num), symbol, direction.upper(), float(volume))
                            except Exception as e:
                                logger.warning(f"Risk open update failed: {e}")
                        notify_portfolio_trade_open(ticket_num, symbol, direction.upper(), float(volume))
                        bal, eqt = get_account_balance_equity()
                        spread_pts = get_spread_points(symbol)

                        # ML data — executed trade
                        dc = get_ml_dc()
                        if dc:
                            try:
                                row = build_ml_row(
                                    symbol,
                                    f"{signal['direction']}_EXECUTED",
                                    signal['trigger'],
                                    0.90,
                                    atr_value,
                                    1,
                                    rates_np
                                )
                                row.update({
                                    'entry_price':     entry_price,
                                    'sl_price':        sl,
                                    'tp_price':        tp,
                                    'volume_lots':     volume,
                                    'ticket':          ticket_num,
                                    'account_balance': bal,
                                    'account_equity':  eqt,
                                    'spread_points':   spread_pts,
                                    'ema_fast':        indicators.get('ema_fast'),
                                    'ema_slow':        indicators.get('ema_slow'),
                                    'ema_separation':  indicators.get('ema_separation'),
                                    'rsi':             indicators.get('rsi'),
                                    'n_bar_high':      indicators.get('n_bar_high'),
                                    'n_bar_low':       indicators.get('n_bar_low'),
                                    'conditions_met':  indicators.get('conditions_met'),
                                })
                                dc.log_signal(row)
                            except Exception:
                                pass
                        send_webhook({"event": "trade", "symbol": symbol, "direction": direction.upper(), "trigger": "BREAKOUT_5C", "entry_price": entry_price, "sl": sl, "tp": tp, "volume": volume, "strategy": "anaconda_breakout_5c", "timeframe": CONFIG.get('timeframe', 'H4'), "confidence": signal.get("confidence"), "rsi": indicators.get("rsi"), "atr": atr_value, "h1_trend": None, "adx": None})
                    else:
                        logger.warning(f"XX {symbol}: Order rejected")
                        send_webhook({"event": "rejected", "symbol": symbol, "direction": direction.upper(), "trigger": signal.get("trigger"), "entry_price": entry_price, "rsi": indicators.get("rsi")})
            except Exception as e:
                logger.error(f"{symbol} error: {e}")

        iteration += 1
        if iteration % 5 == 0:
            logger.info(f"[HEARTBEAT] ANACONDA loop -- iteration {iteration}")
        send_webhook({"event": "heartbeat", "iteration": iteration})
        time.sleep(CONFIG.get('strategy_interval', 300))  # 5 min on H4

# ============================================================================
# TRAILING STOP + BREAKEVEN
# ============================================================================
def trailing_loop():
    broker = get_broker()
    timeframe = TF_MAP.get(CONFIG.get('timeframe', 'H4'), mt5.TIMEFRAME_H4)
    trail_iter = 0
    global _tracked_positions

    while not bot_stop.is_set():
        try:
            positions = broker.get_positions()
            trail_iter += 1
            if trail_iter % 20 == 0:
                logger.info(f"[HEARTBEAT] Trailing -- {len(positions)} positions")

            # Track positions and detect closures for ML outcome logging
            current_tickets = {pos['ticket'] for pos in positions}

            # Add new positions to tracking
            for pos in positions:
                ticket = pos['ticket']
                with positions_lock:
                    if ticket not in _tracked_positions:
                        _tracked_positions[ticket] = {
                            'symbol': pos['symbol'],
                            'entry_price': float(pos.get('price_open', pos.get('price', 0))),
                            'sl': float(pos.get('sl') or 0.0),
                            'tp': float(pos.get('tp') or 0.0),
                            'volume': float(pos.get('volume', 0.01)),
                            'open_time': datetime.fromtimestamp(pos.get('time', time.time()), timezone.utc),
                            'type': pos.get('type', '')
                        }

            # Detect closed positions (tickets in tracking but not in current)
            with positions_lock:
                closed_tickets = set(_tracked_positions.keys()) - current_tickets
            for ticket in closed_tickets:
                try:
                    with positions_lock:
                        entry = _tracked_positions[ticket]
                    symbol = entry['symbol']
                    deal = find_closing_deal(ticket)
                    if deal:
                        close_price = deal.price
                        profit = deal.profit + getattr(deal, 'commission', 0.0) + getattr(deal, 'swap', 0.0)
                        close_time = datetime.fromtimestamp(deal.time, timezone.utc)
                        duration = (close_time - entry['open_time']).total_seconds() / 60
                        exit_reason = determine_exit_reason(
                            entry['entry_price'], close_price, entry['sl'], entry['tp'], symbol
                        )
                        rates_out = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50)
                        curr_atr_out = 0
                        if rates_out is not None and len(rates_out) >= 20:
                            rn = mt5_rates_to_numpy(rates_out)
                            curr_atr_out = safe_last(ATR(rn[:, 2], rn[:, 3], rn[:, 4], 14))
                        bal, eqt = get_account_balance_equity()
                        log_trade_outcome(
                            symbol, ticket, entry['entry_price'], entry['sl'], entry['tp'],
                            entry['volume'], close_price, profit, exit_reason, duration,
                            atr=curr_atr_out, account_balance=bal, account_equity=eqt
                        )
                        with positions_lock:
                            _tracked_positions.pop(ticket, None)
                        clear_pending_close(str(ticket))
                    else:
                        age_seconds, should_log = note_pending_close(str(ticket))
                        if should_log:
                            logger.info(
                                f"[OUTCOME] {symbol} #{ticket} is closed but broker history is still pending "
                                f"({age_seconds:.0f}s). Will retry until realized P&L is available."
                            )
                        if age_seconds >= 6 * 3600:
                            logger.warning(
                                f"[OUTCOME] {symbol} #{ticket} close history is still missing after "
                                f"{age_seconds / 60:.0f} minutes - dropping tracked position without realized P&L."
                            )
                            with positions_lock:
                                _tracked_positions.pop(ticket, None)
                            clear_pending_close(str(ticket))
                except Exception as e:
                    logger.warning(f"Failed to log outcome for closed ticket {ticket}: {e}", exc_info=True)

            now = datetime.now(timezone.utc)

            for pos in positions:
                symbol = pos['symbol']
                cfg = CONFIG['symbols'].get(symbol)
                if cfg is None or not cfg.get('use_trailing', True): continue

                # Friday flatten — close positions before weekend gap
                if now.weekday() == 4:
                    tf = cfg.get('time_filter', {})
                    flatten_hour = tf.get('friday_flatten_hour', 0)
                    if flatten_hour > 0 and now.hour >= flatten_hour:
                        ticket = pos.get('ticket', 0)
                        logger.info(f"[FRIDAY-FLATTEN] Closing {symbol} #{ticket} before weekend")

                        # Get ATR for ML row
                        rates_ff = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50)
                        curr_atr_ff = 0
                        rates_np_ff = None
                        if rates_ff is not None and len(rates_ff) >= 20:
                            rates_np_ff = mt5_rates_to_numpy(rates_ff)
                            curr_atr_ff = safe_last(ATR(rates_np_ff[:, 2], rates_np_ff[:, 3], rates_np_ff[:, 4], 14))

                        # Log ML outcome inline for Friday flatten
                        with positions_lock:
                            tracked = _tracked_positions.get(ticket)
                        if tracked:
                            try:
                                pos_type_ff = normalize_pos_type(pos.get('type', ''))
                                tick_ff = mt5.symbol_info_tick(symbol)
                                if tick_ff:
                                    close_px = tick_ff.bid if pos_type_ff == 'BUY' else tick_ff.ask
                                    duration_ff = (now - tracked['open_time']).total_seconds() / 60
                                    unrealized = pos.get('profit', 0.0)
                                    bal, eqt = get_account_balance_equity()
                                    row = build_ml_row(symbol, 'TRADE_CLOSED', 'OUTCOME', 0, curr_atr_ff, 0, rates_np_ff)
                                    row.update({
                                        'entry_price': tracked['entry_price'],
                                        'sl_price':    tracked['sl'],
                                        'tp_price':    tracked['tp'],
                                        'volume_lots': tracked['volume'],
                                        'ticket':      ticket,
                                        'exit_price':  close_px,
                                        'exit_reason': 'FRIDAY_FLATTEN',
                                        'profit':      unrealized,
                                        'outcome':     'WIN' if unrealized > 0 else ('LOSS' if unrealized < 0 else 'BREAKEVEN'),
                                        'duration_minutes': duration_ff,
                                        'account_balance': bal,
                                        'account_equity':  eqt,
                                        'spread_points': get_spread_points(symbol),
                                    })
                                    dc = get_ml_dc()
                                    if dc: dc.log_signal(row)
                                    rm = get_risk_manager()
                                    if rm:
                                        try:
                                            rm.on_trade_close(int(ticket), float(unrealized), float(duration_ff) * 60.0, symbol)
                                            notify_portfolio_trade_close(ticket, unrealized, float(duration_ff) * 60.0, symbol)
                                        except Exception as e:
                                            logger.warning(f"Risk close update failed: {e}")
                                    with positions_lock:
                                        _tracked_positions.pop(ticket, None)
                            except Exception:
                                pass

                        broker.close_position(ticket)
                        send_webhook({"event": "friday_flatten", "symbol": symbol, "ticket": ticket})
                        continue

                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
                if rates is None or len(rates) < 50: continue
                rates_np = mt5_rates_to_numpy(rates)
                curr_atr = safe_last(ATR(rates_np[:, 2], rates_np[:, 3], rates_np[:, 4], 14))

                pos_type = normalize_pos_type(pos.get('type', ''))
                ticket = pos.get('ticket', 0)
                if pos_type is None: continue

                tick = mt5.symbol_info_tick(symbol)
                if tick is None: continue

                current_sl = float(pos.get('sl') or 0.0)
                entry_price = float(pos.get('price_open', pos.get('price', 0)))                # Breakeven
                be_mult = cfg.get('breakeven_atr_mult', 1.0)
                be_offset = curr_atr * 0.1
                digits = get_symbol_digits(symbol)
                curr_rsi = safe_last(RSI(rates_np[:, 4], 14))
                if entry_price > 0:
                    if pos_type == 'BUY':
                        if tick.bid >= entry_price + (curr_atr * be_mult) and current_sl < entry_price:
                            be_sl = entry_price + be_offset
                            broker.modify_position(ticket, sl=round(be_sl, digits))
                            logger.info(f"[BE] {symbol} #{ticket}: SL -> {be_sl:.{digits}f}")
                            send_webhook({"event": "breakeven", "symbol": symbol, "ticket": ticket, "direction": "BUY", "old_sl": current_sl, "new_sl": round(be_sl, digits), "rsi": curr_rsi})
                            continue
                    elif pos_type == 'SELL':
                        if tick.ask <= entry_price - (curr_atr * be_mult) and (current_sl > entry_price or current_sl == 0):
                            be_sl = entry_price - be_offset
                            broker.modify_position(ticket, sl=round(be_sl, digits))
                            logger.info(f"[BE] {symbol} #{ticket}: SL -> {be_sl:.{digits}f}")
                            send_webhook({"event": "breakeven", "symbol": symbol, "ticket": ticket, "direction": "SELL", "old_sl": current_sl, "new_sl": round(be_sl, digits), "rsi": curr_rsi})
                            continue

                trail_start_mult = cfg.get('trail_start_atr_mult', be_mult)
                if entry_price > 0:
                    if pos_type == 'BUY' and tick.bid < entry_price + (curr_atr * trail_start_mult):
                        continue
                    elif pos_type == 'SELL' and tick.ask > entry_price - (curr_atr * trail_start_mult):
                        continue

                # Trailing
                trail_dist = curr_atr * cfg.get('atr_trail_multiplier', 1.5)
                if pos_type == 'BUY':
                    new_sl = tick.bid - trail_dist
                    if new_sl > current_sl and current_sl > 0:
                        broker.modify_position(ticket, sl=round(new_sl, digits))
                        logger.info(f"[TRAIL] {symbol} #{ticket}: SL {current_sl:.{digits}f} -> {new_sl:.{digits}f}")
                        send_webhook({"event": "trail_update", "symbol": symbol, "ticket": ticket, "direction": "BUY", "old_sl": current_sl, "new_sl": round(new_sl, digits), "rsi": curr_rsi})
                elif pos_type == 'SELL':
                    new_sl = tick.ask + trail_dist
                    if (new_sl < current_sl or current_sl == 0) and new_sl > 0:
                        broker.modify_position(ticket, sl=round(new_sl, digits))
                        logger.info(f"[TRAIL] {symbol} #{ticket}: SL {current_sl:.{digits}f} -> {new_sl:.{digits}f}")
                        send_webhook({"event": "trail_update", "symbol": symbol, "ticket": ticket, "direction": "SELL", "old_sl": current_sl, "new_sl": round(new_sl, digits), "rsi": curr_rsi})
        except Exception as e:
            logger.error(f"[TRAIL] Error: {e}")
        time.sleep(60)  # Check every 60s on H4

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/status', methods=['GET'])
def get_status():
    broker = get_broker()
    account = broker.get_account_info() if broker else None
    positions = broker.get_positions() if broker else []
    pg = get_portfolio_guard()
    portfolio = pg.get_snapshot() if pg else None
    return jsonify({'bot': 'anaconda-v1', 'profile': PROFILE_NAME, 'account': ACCOUNT_NUMBER,
        'timeframe': CONFIG.get('timeframe', 'H4'), 'balance': account.get('balance', 0) if account else 0,
        'equity': account.get('equity', 0) if account else 0, 'positions_count': len(positions),
        'portfolio_risk': portfolio, 'timestamp': datetime.now(timezone.utc).isoformat()})

@app.route('/prop_firm', methods=['GET'])
def get_prop_firm():
    pfg = get_prop_guard()
    if not pfg:
        return jsonify({'enabled': False})
    return jsonify(pfg.get_state())

@app.route('/positions', methods=['GET'])
def get_positions():
    broker = get_broker()
    return jsonify({'positions': broker.get_positions() if broker else []})

@app.route('/account', methods=['GET'])
def get_account():
    broker = get_broker()
    return jsonify(broker.get_account_info() or {} if broker else {})

@app.route('/buy', methods=['POST'])
def buy():
    broker = get_broker()
    if not broker:
        return jsonify({'success': False, 'error': 'Broker not initialized'}), 503
    data = request.get_json() or {}
    result = broker.open_position(data.get('symbol'), 'buy', data.get('volume', 0.01), data.get('sl'), data.get('tp'))
    return jsonify({'success': bool(result), 'result': serialize_result(result)})

@app.route('/sell', methods=['POST'])
def sell():
    broker = get_broker()
    if not broker:
        return jsonify({'success': False, 'error': 'Broker not initialized'}), 503
    data = request.get_json() or {}
    result = broker.open_position(data.get('symbol'), 'sell', data.get('volume', 0.01), data.get('sl'), data.get('tp'))
    return jsonify({'success': bool(result), 'result': serialize_result(result)})

@app.route('/close', methods=['POST'])
def close():
    broker = get_broker()
    if not broker:
        return jsonify({'success': False, 'error': 'Broker not initialized'}), 503
    data = request.get_json() or {}
    ticket = data.get('ticket')

    # Get position details before closing for ML logging
    with positions_lock:
        entry = _tracked_positions.get(ticket)
    result = broker.close_position(ticket)

    # Log ML outcome for manual close
    if result and entry:
        try:
            symbol = entry['symbol']
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                pos_type = normalize_pos_type(entry.get('type', ''))
                close_price = tick.bid if pos_type == 'BUY' else tick.ask
                duration = (datetime.now(timezone.utc) - entry['open_time']).total_seconds() / 60
                time.sleep(0.5)
                deals = mt5.history_deals_get(datetime.now(timezone.utc) - timedelta(hours=1), datetime.now(timezone.utc))
                profit = 0.0
                if deals:
                    for deal in deals:
                        if deal.position_id == ticket and deal.entry == 1:
                            profit = deal.profit
                            break
                tf_close = TF_MAP.get(CONFIG.get('timeframe', 'H4'), mt5.TIMEFRAME_H4)
                rates_close = mt5.copy_rates_from_pos(symbol, tf_close, 0, 50)
                atr_close = 0
                if rates_close is not None and len(rates_close) >= 20:
                    rn_close = mt5_rates_to_numpy(rates_close)
                    atr_close = safe_last(ATR(rn_close[:, 2], rn_close[:, 3], rn_close[:, 4], 14))
                bal, eqt = get_account_balance_equity()
                log_trade_outcome(symbol, ticket, entry['entry_price'], entry['sl'], entry['tp'],
                    entry['volume'], close_price, profit, 'MANUAL', duration,
                    atr=atr_close, account_balance=bal, account_equity=eqt)
                with positions_lock:
                    _tracked_positions.pop(ticket, None)
        except Exception as e:
            logger.warning(f"Failed to log manual close outcome: {e}")

    return jsonify({'success': bool(result), 'result': serialize_result(result)})

@app.route('/modify', methods=['POST'])
def modify():
    broker = get_broker()
    if not broker:
        return jsonify({'success': False, 'error': 'Broker not initialized'}), 503
    data = request.get_json() or {}
    result = broker.modify_position(data.get('ticket'), sl=data.get('sl'), tp=data.get('tp'))
    return jsonify({'success': bool(result), 'result': serialize_result(result)})

@app.route('/analyze/<symbol>', methods=['GET'])
def analyze(symbol):
    timeframe = TF_MAP.get(CONFIG.get('timeframe', 'H4'), mt5.TIMEFRAME_H4)
    cfg = CONFIG['symbols'].get(symbol, {})
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 300)
    if rates is None or len(rates) < 220:
        return jsonify({'error': 'No data'}), 400
    rates_np = mt5_rates_to_numpy(rates)
    signal, atr_val, indicators = check_signal(rates_np, cfg)
    return jsonify({
        'symbol': symbol, 'strategy': 'anaconda_breakout_5c',
        'signal': signal['direction'] if signal else 'HOLD',
        'trigger': signal['trigger'] if signal else None,
        'atr': atr_val,
        'confidence': signal['confidence'] if signal else 0,
        'reason': signal['reason'] if signal else 'Conditions not met',
        'timeframe': CONFIG.get('timeframe', 'H4'),
    })

@app.route('/start', methods=['POST'])
def start():
    global _strategy_thread, _trailing_thread
    with state_lock:
        if _strategy_thread and _strategy_thread.is_alive():
            return jsonify({'success': False, 'message': 'Already running'}), 409
        bot_stop.clear()
        _strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
        _trailing_thread = threading.Thread(target=trailing_loop, daemon=True)
        _strategy_thread.start()
        _trailing_thread.start()
    return jsonify({'success': True, 'message': 'Started'})

@app.route('/stop', methods=['POST'])
def stop():
    bot_stop.set()
    return jsonify({'success': True, 'message': 'Stopped'})

# ============================================================================
# INIT
# ============================================================================
def load_config(profile):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'anaconda_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found")
    return profiles[profile]

def initialize():
    global CONFIG, ACCOUNT_NUMBER, PROFILE_NAME
    parser = argparse.ArgumentParser(description='ANACONDA v1 -- Strict Breakout Bot')
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    PROFILE_NAME = args.profile
    CONFIG = load_config(PROFILE_NAME)
    ACCOUNT_NUMBER = CONFIG['account_number']
    init_logger()
    logger.info("=" * 60)
    logger.info(f"ANACONDA v1 Starting -- Profile: {PROFILE_NAME}")
    logger.info(f"Account: {ACCOUNT_NUMBER}, Timeframe: {CONFIG.get('timeframe')}")
    logger.info(f"Strategy: 5-Condition Strict Breakout (ExMachina-inspired)")
    logger.info("=" * 60)
    password = os.environ.get('MT5_PASSWORD', '')
    if not password:
        logger.error("MT5_PASSWORD not set.")
        return False
    try:
        broker = MT5Broker(account=ACCOUNT_NUMBER, password=password,
            server=CONFIG['server'], mt5_path=CONFIG.get('mt5_path'))
        set_broker(broker)
        logger.info("Broker connected")
    except Exception as e:
        logger.error(f"Broker failed: {e}")
        return False
    try:
        rc = CONFIG.get('risk', {})
        rm = RiskManagerUltra(daily_loss_limit=rc.get('daily_loss_limit', -30),
            max_consecutive_losses=rc.get('max_consecutive_losses', 3),
            max_drawdown_percent=rc.get('max_drawdown_pct', 8))
        set_risk_manager(rm)
        logger.info("Risk manager ready")
    except Exception as e:
        logger.error(f"Risk manager failed: {e}")
        return False

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        portfolio_cfg = dict(CONFIG.get('portfolio_risk', {}))
        if 'global_daily_loss_limit' in CONFIG:
            portfolio_cfg['global_daily_loss_limit'] = CONFIG['global_daily_loss_limit']
        pg = PortfolioRiskGuard(
            bot_name=BOT_NAME,
            account_number=ACCOUNT_NUMBER,
            config_path=os.path.join(base_dir, 'portfolio_risk_config.json'),
            db_path=os.path.join(base_dir, 'portfolio_risk_guard.db'),
            config=portfolio_cfg,
        )
        set_portfolio_guard(pg)
        logger.info(f"Portfolio risk guard ready (enabled={pg.enabled})")
    except Exception as e:
        logger.warning(f"Portfolio risk guard init failed (non-fatal): {e}")
    # PropFirmGuard
    if CONFIG.get('prop_firm', {}).get('enabled', False):
        try:
            state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      f"prop_firm_state_{ACCOUNT_NUMBER}.json")
            pfg = PropFirmGuard(CONFIG, logger, state_file=state_file)
            set_prop_guard(pfg)
            logger.info(f"[PROP] PropFirmGuard initialized: capital=${pfg.initial_capital:.0f}, "
                        f"daily_halt={pfg.daily_loss_halt_pct}%, dd_halt={pfg.trailing_dd_halt_pct}%")
        except Exception as e:
            logger.warning(f"PropFirmGuard init failed (non-fatal): {e}")
    # Recover daily P&L from MT5 history so restart mid-day doesn't reset to $0
    pfg = get_prop_guard()
    if pfg:
        try:
            acc = mt5.account_info()
            if acc:
                now_utc = datetime.now(timezone.utc)
                day_start_utc = get_daily_reset_boundary(now_utc)
                deals = mt5.history_deals_get(day_start_utc, now_utc)
                realized_today = sum(
                    d.profit + d.commission + d.swap
                    for d in (deals or []) if d.entry == 1
                )
                pfg.recover_from_history(realized_today, acc.equity)
        except Exception as e:
            logger.warning(f"[PROP] Daily P&L history recovery failed (non-fatal): {e}")
    try:
        tl = TradeDecisionLogger(f"anaconda_v1_{PROFILE_NAME}")
        set_trade_logger(tl)
    except Exception as e:
        logger.warning(f"Trade logger failed (non-fatal): {e}")
    # Initialize ML data collector
    try:
        dc = SharedDataCollector('anaconda', ACCOUNT_NUMBER)
        set_ml_dc(dc)
        logger.info(f"ML data collector initialized — saving to {dc.data_dir}")
    except Exception as e:
        logger.warning(f"Data collector init failed (non-fatal): {e}")
    # Pre-populate tracked positions so closures are logged after restart
    try:
        existing = get_broker().get_positions()
        for pos in existing:
            _tracked_positions[pos['ticket']] = {
                'symbol': pos['symbol'],
                'entry_price': float(pos.get('price_open', pos.get('price', 0))),
                'sl': float(pos.get('sl') or 0.0),
                'tp': float(pos.get('tp') or 0.0),
                'volume': float(pos.get('volume', 0.01)),
                'open_time': datetime.fromtimestamp(pos.get('time', 0), timezone.utc) if pos.get('time') else datetime.now(timezone.utc),
                'type': pos.get('type', '')
            }
        logger.info(f"Loaded {len(existing)} existing positions into tracking")
    except Exception as e:
        logger.warning(f"Failed to load existing positions: {e}")
    return True

def sync_open_positions():
    """At startup, push a 'sync' event to the dashboard for every position already
    open in MT5 so the dashboard can reconcile state after a bot restart.
    Safe to call on every restart — server-side upsert means no duplicates."""
    broker = get_broker()
    if broker is None:
        return
    try:
        positions = broker.get_positions()
    except Exception as e:
        logger.warning(f"[SYNC] Failed to fetch open positions: {e}")
        return
    if not positions:
        logger.info("[SYNC] No open positions to sync")
        return
    for pos in positions:
        try:
            send_webhook({
                "event":       "sync",
                "symbol":      pos.get("symbol"),
                "direction":   pos.get("type"),       # already "BUY" or "SELL"
                "entry_price": pos.get("price_open"),
                "sl":          pos.get("sl"),
                "tp":          pos.get("tp", 0.0),    # 0.0 = no TP set in MT5
                "volume":      pos.get("volume"),
                "ticket":      pos.get("ticket"),
            })
            logger.info(f"[SYNC] {pos.get('symbol')} #{pos.get('ticket')} synced to dashboard")
        except Exception as e:
            logger.warning(f"[SYNC] Failed for ticket {pos.get('ticket')}: {e}")


def main():
    global _strategy_thread, _trailing_thread
    if not initialize(): sys.exit(1)
    sync_open_positions()
    _strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
    _trailing_thread = threading.Thread(target=trailing_loop, daemon=True)
    _strategy_thread.start()
    _trailing_thread.start()
    port = CONFIG.get('server_port', 8051)
    logger.info(f"Flask API on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)

if __name__ == '__main__':
    main()




