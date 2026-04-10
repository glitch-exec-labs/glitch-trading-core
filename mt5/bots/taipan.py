"""
TAIPAN v1 - Session Breakout Bot (M30)
Account E: 24849108, Port 8055

Trades Asian session range breakouts at London/NY open.
Time-based edge - orthogonal to pattern/indicator strategies.

Strategy:
  1. Calculate Asian session range (configurable per symbol, default 00:00-06:00 UTC)
  2. At kill zone open (default 07:00-10:00 UTC), trade breakout of range
  3. H4 EMA trend filter gates direction (only trade with macro trend)
  4. Volume confirmation on breakout candle
  5. Range size filter (min/max ATR multiples - skip tiny or blown-out ranges)

Exits: ATR trailing stop + breakeven at 1x ATR
Timeframe: M30 (~1 trade per symbol per day, max 5 trades/day)
Risk: 1% per trade, SL at opposite side of range, TP at 1.5x range width
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
from oracle_guard import request_oracle_approval
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
BOT_NAME = 'taipan'
API_KEY = os.environ.get("TAIPAN_API_KEY", "")

MT5_RECONNECT_THRESHOLD = 5
_mt5_fail_count = 0

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
    logger = logging.getLogger(f"TAIPAN-{ACCOUNT_NUMBER}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"taipan_{ACCOUNT_NUMBER}.log", encoding='utf-8')
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

_ml_dc = None
def get_ml_dc(): return _ml_dc
def set_ml_dc(dc):
    global _ml_dc
    _ml_dc = dc

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

# Position tracking for ML outcome logging
_tracked_positions = {}  # ticket -> {symbol, entry_price, sl, tp, volume, open_time}
_pending_close_checks = {}
pending_close_lock = threading.Lock()

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
        return 'TRAILING_SL'

def log_trade_outcome(symbol, ticket, entry_price, sl, tp, volume, close_price, profit, exit_reason, duration_minutes):
    """Log closed trade outcome to ML data collector."""
    try:
        dc = get_ml_dc()
        if not dc:
            return
        now = datetime.now(timezone.utc)
        outcome = 'WIN' if profit > 0 else ('LOSS' if profit < 0 else 'BREAKEVEN')
        # Fetch current bars to build the full 63-column ML row
        row = None
        try:
            timeframe = TF_MAP.get(CONFIG.get('timeframe', 'M30'), mt5.TIMEFRAME_M30)
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
            if rates is not None and len(rates) >= 50:
                rates_np = mt5_rates_to_numpy(rates)
                curr_atr = float(safe_last(ATR(rates_np[:, 2], rates_np[:, 3], rates_np[:, 4], 14)))
                synthetic_signal = {'direction': 'TRADE_CLOSED', 'trigger': 'OUTCOME', 'reason': exit_reason}
                row = _build_ml_row(
                    symbol, now, rates_np, synthetic_signal,
                    curr_atr, None, None, None, None, 'CLOSED',
                    executed=0, entry_price=entry_price, sl=sl, tp=tp,
                    volume=volume, ticket=ticket, exit_price=close_price,
                    exit_reason=exit_reason, profit=profit, outcome=outcome,
                    duration_minutes=duration_minutes
                )
        except Exception:
            pass
        if row is None:
            # Fallback if MT5 bars unavailable at close time
            row = {
                'timestamp': now.isoformat(),
                'symbol': symbol, 'bot': 'taipan', 'account': ACCOUNT_NUMBER,
                'timeframe': CONFIG.get('timeframe', 'M30'),
                'signal': 'TRADE_CLOSED', 'trigger': 'OUTCOME', 'confidence': 0,
                'atr': 0, 'executed': 0,
                'entry_price': entry_price, 'sl_price': sl, 'tp_price': tp,
                'volume_lots': volume, 'ticket': ticket,
                'exit_price': close_price, 'exit_reason': exit_reason,
                'profit': profit, 'outcome': outcome,
                'duration_minutes': duration_minutes,
                'asian_high': None, 'asian_low': None,
                'range_width': None, 'h4_trend': None, 'session_phase': 'CLOSED'
            }
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

TF_MAP = {
    'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
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
    if isinstance(val, np.ndarray): return float(val[-1])
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

WEBHOOK_URL = "https://dashboard.glitchexecutor.com/api/trades/webhook"

def send_webhook(payload):
    """Post event to dashboard webhook. Accepts {"status":"ok"} and {"status":"ok","duplicate":true} as success."""
    if not CONFIG.get('webhook_enabled', False):
        return False
    try:
        payload.update({"bot": "taipan", "account": ACCOUNT_NUMBER,
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

def ATR(highs, lows, closes, period=14):
    return ATR_GPU(highs, lows, closes, period)

def RSI(prices, period=14):
    return RSI_GPU(prices, period)

def EMA_CPU(prices, period):
    return ema_numba(prices, period)

def ATR_CPU(highs, lows, closes, period=14):
    return atr_numba(highs, lows, closes, period)

def RSI_CPU(prices, period=14):
    return rsi_numba(prices, period)

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

def bollinger_bands(closes, period=20):
    if len(closes) < period:
        return None, None, None
    sma = float(np.mean(closes[-period:]))
    std = float(np.std(closes[-period:], ddof=1))
    return round(sma + 2 * std, 6), round(sma - 2 * std, 6), round(sma, 6)

def calculate_adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period * 2 + 2:
        return None
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        h_diff = float(highs[i] - highs[i - 1])
        l_diff = float(lows[i - 1] - lows[i])
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0
        tr[i] = max(float(highs[i] - lows[i]),
                    abs(float(highs[i] - closes[i - 1])),
                    abs(float(lows[i] - closes[i - 1])))
    atr_s = float(np.sum(tr[1:period + 1]))
    p_dm  = float(np.sum(plus_dm[1:period + 1]))
    m_dm  = float(np.sum(minus_dm[1:period + 1]))
    dx_vals = []
    for i in range(period + 1, n):
        atr_s = atr_s - atr_s / period + tr[i]
        p_dm  = p_dm  - p_dm  / period + plus_dm[i]
        m_dm  = m_dm  - m_dm  / period + minus_dm[i]
        p_di = 100 * p_dm / atr_s if atr_s else 0
        m_di = 100 * m_dm / atr_s if atr_s else 0
        dx = 100 * abs(p_di - m_di) / (p_di + m_di) if (p_di + m_di) else 0
        dx_vals.append(dx)
    if len(dx_vals) < period:
        return None
    adx = float(np.mean(dx_vals[:period]))
    for i in range(period, len(dx_vals)):
        adx = (adx * (period - 1) + dx_vals[i]) / period
    return round(adx, 2)

def _build_ml_row(symbol, now, rates_np, signal, curr_atr, asian_high, asian_low,
                  range_width, h4_trend, session_phase, executed=0,
                  entry_price=None, sl=None, tp=None, volume=None, ticket=None,
                  exit_price=None, exit_reason=None, profit=None, outcome=None,
                  duration_minutes=None, account_balance=None, account_equity=None,
                  spread=None, spread_points=None):
    """Build a full 63-column ML row matching the shared schema."""
    closes  = rates_np[:, 4]
    highs   = rates_np[:, 2]
    lows    = rates_np[:, 3]
    volumes = rates_np[:, 5]
    try: bar_open  = round(float(rates_np[-1, 1]), 6)
    except: bar_open = None
    try: bar_high  = round(float(highs[-1]), 6)
    except: bar_high = None
    try: bar_low   = round(float(lows[-1]), 6)
    except: bar_low = None
    try: bar_close = round(float(closes[-1]), 6)
    except: bar_close = None
    try: ema_fast_val = round(float(EMA(closes, 8)[-1]), 6)
    except: ema_fast_val = None
    try: ema_slow_val = round(float(EMA(closes, 20)[-1]), 6)
    except: ema_slow_val = None
    try: ema_50_val = round(float(EMA(closes, 50)[-1]), 6)
    except: ema_50_val = None
    try: ema_sep = round(float(ema_fast_val - ema_slow_val), 6) if ema_fast_val and ema_slow_val else None
    except: ema_sep = None
    try: rsi_val = round(float(RSI(closes, 14)[-1]), 4)
    except: rsi_val = None
    try: bb_u, bb_l, bb_m = bollinger_bands(closes, 20)
    except: bb_u = bb_l = bb_m = None
    try: price_bb = round((bar_close - bb_l) / (bb_u - bb_l), 4) if bb_u and bb_l and bb_u != bb_l else None
    except: price_bb = None
    try: adx_val = calculate_adx(highs, lows, closes, 14)
    except: adx_val = None
    try:
        avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        vol_ratio = round(float(volumes[-1]) / avg_vol, 4) if avg_vol > 0 else None
    except: vol_ratio = None
    sig_dir   = signal['direction'] if signal else 'HOLD'
    sig_type  = signal['trigger']   if signal else 'NONE'
    sig_trig  = signal['trigger']   if signal else 'NONE'
    hold_rsn  = signal.get('reason', session_phase) if signal else session_phase
    return {
        'timestamp':              now.isoformat(),
        'symbol':                 symbol,
        'bot':                    'taipan',
        'account':                ACCOUNT_NUMBER,
        'timeframe':              CONFIG.get('timeframe', 'M30'),
        'signal':                 f"{sig_dir}_EXECUTED" if executed else sig_dir,
        'signal_type':            sig_type,
        'trigger':                sig_trig,
        'confidence':             0.80 if signal else 0,
        'atr':                    curr_atr,
        'executed':               executed,
        'entry_price':            entry_price,
        'sl_price':               sl,
        'sl':                     sl,
        'tp_price':               tp,
        'tp':                     tp,
        'volume_lots':            volume,
        'ticket':                 ticket,
        'exit_price':             exit_price,
        'exit_reason':            exit_reason,
        'profit':                 profit,
        'pnl':                    profit,
        'outcome':                outcome,
        'duration_minutes':       duration_minutes,
        'account_balance':        account_balance,
        'account_equity':         account_equity,
        'spread':                 spread,
        'spread_points':          spread_points,
        'bar_open':               bar_open,
        'bar_high':               bar_high,
        'bar_low':                bar_low,
        'bar_close':              bar_close,
        'ema_fast':               ema_fast_val,
        'ema_slow':               ema_slow_val,
        'ema_separation':         ema_sep,
        'rsi':                    rsi_val,
        'n_bar_high':             asian_high,
        'n_bar_low':              asian_low,
        'conditions_met':         None,
        'ema_50':                 ema_50_val,
        'nearest_sr_level':       None,
        'sr_distance_atr':        None,
        'pattern_type':           'SESSION_BREAKOUT' if signal else 'NONE',
        'pin_wick_size':          None,
        'pin_body_ratio':         None,
        'engulf_size_atr':        None,
        'inside_bar_vol_ratio':   None,
        'pattern_detected':       bool(signal),
        'sr_nearby':              None,
        'ema_20':                 ema_slow_val,
        'momentum_fast_ema':      None,
        'momentum_slow_ema':      None,
        'breakout_high':          asian_high,
        'breakout_low':           asian_low,
        'pullback_distance_atr':  round(range_width / curr_atr, 4) if range_width and curr_atr else None,
        'volume_ratio':           vol_ratio,
        'h1_trend':               h4_trend,
        'bb_upper':               bb_u,
        'bb_lower':               bb_l,
        'bb_mid':                 bb_m,
        'price_position_in_bb':   price_bb,
        'adx':                    adx_val,
        'hold_reason':            hold_rsn,
    }

# ============================================================================
# H4 TREND FILTER
# ============================================================================
def get_h4_trend(symbol, cfg):
    """
    Fetch H4 data and determine trend direction using EMA.
    Returns: 'BUY', 'SELL', or 'BOTH' (transition zone).
    """
    h4_ema_period = cfg.get('h4_ema_period', 50)
    try:
        h4_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, h4_ema_period + 20)
        if h4_rates is None or len(h4_rates) < h4_ema_period + 5:
            return 'BOTH'
        h4_closes = np.array([r[4] for r in h4_rates], dtype=np.float64)
        h4_highs = np.array([r[2] for r in h4_rates], dtype=np.float64)
        h4_lows = np.array([r[3] for r in h4_rates], dtype=np.float64)
        h4_ema = EMA(h4_closes, h4_ema_period)
        h4_atr = safe_last(ATR(h4_highs, h4_lows, h4_closes, 14))
        h4_close = float(h4_closes[-1])
        h4_ema_val = float(h4_ema[-1])
        transition = h4_atr * cfg.get('h4_transition_mult', 0.5)
        if h4_close > h4_ema_val + transition:
            return 'BUY'
        elif h4_close < h4_ema_val - transition:
            return 'SELL'
        else:
            return 'BOTH'
    except Exception:
        return 'BOTH'

# ============================================================================
# SESSION RANGE CALCULATION
# ============================================================================
def is_bar_in_session(bar_time_utc, start_hour, end_hour):
    """
    Check if a bar's hour falls within session window.
    Handles cross-midnight ranges (e.g., start=21, end=5).
    """
    hour = bar_time_utc.hour
    if start_hour <= end_hour:
        # Normal range: e.g., 0-6
        return start_hour <= hour < end_hour
    else:
        # Cross-midnight: e.g., 21-5 means 21,22,23,0,1,2,3,4
        return hour >= start_hour or hour < end_hour

def calculate_asian_range(rates_np, cfg, now_utc):
    """
    Calculate the Asian session range from M30 bars.

    Scans bars that fall within [asian_start_hour, asian_end_hour) UTC today.
    For cross-midnight ranges (BTC: 21:00-05:00), includes bars from yesterday evening.

    Returns: (asian_high, asian_low, range_width, curr_atr) or (None, None, None, atr)
    """
    asian_start = cfg.get('asian_start_hour', 0)
    asian_end = cfg.get('asian_end_hour', 6)
    atr_period = cfg.get('atr_period', 14)

    highs = rates_np[:, 2]
    lows = rates_np[:, 3]
    closes = rates_np[:, 4]
    times = rates_np[:, 0]

    curr_atr = safe_last(ATR(highs, lows, closes, atr_period))
    if curr_atr <= 0:
        return None, None, None, 0.0

    # Find bars in Asian session
    today = now_utc.date()
    asian_highs = []
    asian_lows = []

    for i in range(len(times)):
        bar_dt = datetime.fromtimestamp(int(times[i]), tz=timezone.utc)
        bar_date = bar_dt.date()

        # For normal ranges (0-6): only today's bars
        # For cross-midnight (21-5): yesterday's 21+ and today's 0-5
        if asian_start <= asian_end:
            # Normal: bars from today within [start, end)
            if bar_date == today and is_bar_in_session(bar_dt, asian_start, asian_end):
                asian_highs.append(float(highs[i]))
                asian_lows.append(float(lows[i]))
        else:
            # Cross-midnight: yesterday's start..24 + today's 0..end
            yesterday = today - timedelta(days=1)
            if bar_date == yesterday and bar_dt.hour >= asian_start:
                asian_highs.append(float(highs[i]))
                asian_lows.append(float(lows[i]))
            elif bar_date == today and bar_dt.hour < asian_end:
                asian_highs.append(float(highs[i]))
                asian_lows.append(float(lows[i]))

    # Need minimum bars for valid range (at least 6 of 12 possible M30 bars in 6 hours)
    min_bars = cfg.get('min_asian_bars', 6)
    if len(asian_highs) < min_bars:
        return None, None, None, curr_atr

    asian_high = max(asian_highs)
    asian_low = min(asian_lows)
    range_width = asian_high - asian_low

    # Range size filters
    min_range = curr_atr * cfg.get('min_range_atr_mult', 0.5)
    max_range = curr_atr * cfg.get('max_range_atr_mult', 3.0)

    if range_width < min_range:
        return None, None, None, curr_atr  # Too tight - noise
    if range_width > max_range:
        return None, None, None, curr_atr  # Too wide - already moved

    return asian_high, asian_low, range_width, curr_atr

# ============================================================================
# BREAKOUT SIGNAL DETECTION
# ============================================================================
def check_session_breakout(rates_np, asian_high, asian_low, range_width, h4_trend, curr_atr, cfg):
    """
    Check if price has broken out of Asian range during kill zone.

    Conditions (ALL must be true):
    1. Completed M30 bar closes above asian_high (buy) or below asian_low (sell)
    2. Close exceeds boundary by breakout_buffer_mult * ATR
    3. H4 trend filter agrees with direction
    4. Volume confirmation (breakout bar vol > breakout_vol_mult * avg volume)

    Returns: signal_dict or None
    """
    closes = rates_np[:, 4]
    volumes = rates_np[:, 5]

    breakout_buffer = curr_atr * cfg.get('breakout_buffer_mult', 0.1)
    vol_mult = cfg.get('breakout_vol_mult', 1.2)

    if len(closes) < 50:
        return None

    # Use completed bar (-2)
    c1 = float(closes[-2])
    vol_1 = float(volumes[-2])

    # Volume confirmation
    avg_vol = float(np.mean(volumes[-50:])) if len(volumes) >= 50 else float(volumes[-1])
    vol_ok = vol_1 > avg_vol * vol_mult

    # Bullish breakout: close above Asian high + buffer
    if c1 > asian_high + breakout_buffer:
        if h4_trend in ('BUY', 'BOTH'):
            if vol_ok:
                return {
                    'trigger': 'SESSION_BREAKOUT',
                    'direction': 'BUY',
                    'confidence': 0.80,
                    'reason': (f'Bullish breakout: close {c1:.5f} > Asian high {asian_high:.5f} '
                              f'+ buf {breakout_buffer:.5f}, vol {vol_1/avg_vol:.1f}x, H4 trend={h4_trend}')
                }

    # Bearish breakout: close below Asian low - buffer
    if c1 < asian_low - breakout_buffer:
        if h4_trend in ('SELL', 'BOTH'):
            if vol_ok:
                return {
                    'trigger': 'SESSION_BREAKOUT',
                    'direction': 'SELL',
                    'confidence': 0.80,
                    'reason': (f'Bearish breakout: close {c1:.5f} < Asian low {asian_low:.5f} '
                              f'- buf {breakout_buffer:.5f}, vol {vol_1/avg_vol:.1f}x, H4 trend={h4_trend}')
                }

    # FALLBACK: EMA trend pullback during the kill zone when breakout has not fired.
    ema_fast = EMA(closes, cfg.get('ema_fast_period', 8))
    ema_slow = EMA(closes, cfg.get('ema_slow_period', 20))
    rsi_vals = RSI(closes, cfg.get('rsi_period', 14))

    ema_fast_val = float(ema_fast[-1])
    ema_slow_val = float(ema_slow[-1])
    close_now = float(closes[-1])
    close_prev = float(closes[-2])
    curr_rsi = float(rsi_vals[-1]) if isinstance(rsi_vals, np.ndarray) else float(rsi_vals)

    if h4_trend in ('BUY', 'BOTH') and ema_fast_val > ema_slow_val:
        if close_prev <= ema_fast_val * 1.003 and close_now > ema_fast_val:
            if 35 < curr_rsi < 75:
                return {
                    'trigger': 'EMA_PULLBACK',
                    'direction': 'BUY',
                    'confidence': 0.72,
                    'reason': (
                        f'Bullish EMA pullback: close {close_now:.5f} reclaimed EMA fast {ema_fast_val:.5f}, '
                        f'prev close {close_prev:.5f}, RSI={curr_rsi:.1f}, H4 trend={h4_trend}'
                    )
                }

    if h4_trend in ('SELL', 'BOTH') and ema_fast_val < ema_slow_val:
        if close_prev >= ema_fast_val * 0.997 and close_now < ema_fast_val:
            if 25 < curr_rsi < 65:
                return {
                    'trigger': 'EMA_PULLBACK',
                    'direction': 'SELL',
                    'confidence': 0.72,
                    'reason': (
                        f'Bearish EMA pullback: close {close_now:.5f} rejected EMA fast {ema_fast_val:.5f}, '
                        f'prev close {close_prev:.5f}, RSI={curr_rsi:.1f}, H4 trend={h4_trend}'
                    )
                }

    return None

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

def calculate_sl(direction, asian_high, asian_low, atr_value, cfg):
    """SL at opposite side of Asian range + configured ATR buffer."""
    sl_mult = cfg.get('atr_sl_multiplier', cfg.get('sl_buffer_atr_mult', 0.3))
    buf = atr_value * sl_mult
    if direction == 'buy':
        return asian_low - buf
    else:
        return asian_high + buf

def calculate_tp(direction, entry_price, range_width, cfg):
    """TP at configured multiple of Asian range width."""
    mult = cfg.get('tp_range_mult', 1.5)
    if direction == 'buy':
        return entry_price + (range_width * mult)
    else:
        return entry_price - (range_width * mult)

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
        dollar_risk = ticks_at_risk * tick_value
        lots = risk_amount / dollar_risk if dollar_risk > 0 else cfg.get('min_lot', 0.01)
    except Exception as e:
        logger.warning(f"Position sizing error {symbol}: {e}")
        lots = cfg.get('min_lot', 0.01)
    min_lot = cfg.get('min_lot', 0.01)
    broker_max_lot = float(sym_info.get('volume_max') or 0.0)
    max_lot = broker_max_lot if broker_max_lot > 0 else max(float(lots), min_lot)
    return round(max(min_lot, min(lots, max_lot)), 2)

# ============================================================================
# ADAPTIVE COOLDOWN + PER-SYMBOL SESSION LIMITS
# ============================================================================
_consecutive_losses = {}
_consecutive_losses_date = {}
_symbol_session_losses = {}
_session_loss_lock = threading.Lock()

def get_daily_reset_boundary(now=None):
    now_utc = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    reset_time = now_utc.replace(hour=20, minute=0, second=0, microsecond=0)
    if now_utc < reset_time:
        reset_time -= timedelta(days=1)
    return reset_time

def get_daily_reset_key(now=None):
    return get_daily_reset_boundary(now).strftime('%Y-%m-%d')

def record_trade_result(symbol, profit):
    """Track consecutive losses and daily per-symbol P&L."""
    with _session_loss_lock:
        today = get_daily_reset_key()
        if profit < 0:
            _consecutive_losses[symbol] = _consecutive_losses.get(symbol, 0) + 1
            _consecutive_losses_date[symbol] = today
            logger.info(f"[LOSS] {symbol} loss #{_consecutive_losses[symbol]}: ${profit:.2f}")
        else:
            if symbol in _consecutive_losses:
                logger.info(f"[WIN] {symbol} streak reset (was {_consecutive_losses[symbol]} losses)")
            _consecutive_losses[symbol] = 0
            _consecutive_losses_date[symbol] = today
        today = get_daily_reset_key()
        key = symbol
        if key not in _symbol_session_losses or _symbol_session_losses[key].get('date') != today:
            _symbol_session_losses[key] = {'date': today, 'total_loss': 0.0, 'trade_count': 0}
        _symbol_session_losses[key]['total_loss'] += profit
        _symbol_session_losses[key]['trade_count'] += 1

def get_adaptive_cooldown(symbol, base_cooldown, cfg):
    """Exponential backoff: cooldown doubles after each consecutive loss."""
    with _session_loss_lock:
        losses = _consecutive_losses.get(symbol, 0)
    if losses == 0:
        return base_cooldown
    multiplier = cfg.get('loss_cooldown_multiplier', 2.0)
    max_cd = cfg.get('max_loss_cooldown', 3600)
    cd = base_cooldown * (multiplier ** min(losses, 5))
    return min(cd, max_cd)

def is_symbol_session_limit_hit(symbol, cfg):
    """Check if symbol has hit daily loss % or consecutive loss limit."""
    max_consec = cfg.get('max_symbol_consecutive_losses', 3)
    today = get_daily_reset_key()
    with _session_loss_lock:
        if _consecutive_losses_date.get(symbol) != today:
            _consecutive_losses[symbol] = 0
        consec = _consecutive_losses.get(symbol, 0)
        if consec >= max_consec:
            return True, f"{symbol} hit {consec} consecutive losses (limit {max_consec})"
    max_loss_pct = cfg.get('max_symbol_daily_loss_pct', 5)
    broker = get_broker()
    account_info = broker.get_account_info() if broker else None
    curr_balance = account_info.get('balance', 150) if account_info else 150
    max_loss_dollar = -(curr_balance * max_loss_pct / 100)
    today = get_daily_reset_key()
    with _session_loss_lock:
        session = _symbol_session_losses.get(symbol, {})
        if session.get('date') == today and session.get('total_loss', 0) <= max_loss_dollar:
            return True, f"{symbol} daily loss ${session['total_loss']:.2f} exceeds limit ${max_loss_dollar:.2f} ({max_loss_pct}% of ${curr_balance:.2f})"
    return False, ""

# ============================================================================
# STRATEGY LOOP
# ============================================================================
def strategy_loop():
    global _mt5_fail_count
    broker = get_broker()
    default_tf_str = CONFIG.get('timeframe', 'M30')
    timeframe = TF_MAP.get(default_tf_str, mt5.TIMEFRAME_M30)
    iteration = 0

    # Daily state per symbol: tracks Asian range + whether we already traded today
    daily_state = {}  # symbol -> {date, asian_high, asian_low, range_width, traded_today, atr}

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
        now = datetime.now(timezone.utc)
        today = now.date()

        for symbol, cfg in CONFIG['symbols'].items():
            if not cfg.get('enabled', False): continue
            sym_tf = TF_MAP.get(cfg.get('timeframe', default_tf_str), timeframe)
            try:
                if not is_within_trading_hours(cfg):
                    if iteration % 10 == 0:
                        asian_start = cfg.get('asian_start_hour', 0)
                        asian_end = cfg.get('asian_end_hour', 6)
                        kill_start = cfg.get('kill_zone_start_hour', 7)
                        kill_end = cfg.get('kill_zone_end_hour', 10)
                        if now.weekday() >= 5:
                            session_phase = 'WEEKEND'
                        elif asian_start <= asian_end:
                            in_asian = asian_start <= now.hour < asian_end
                            in_transition = asian_end <= now.hour < kill_start
                        else:
                            in_asian = now.hour >= asian_start or now.hour < asian_end
                            in_transition = (not in_asian and now.hour < kill_start)
                        if now.weekday() < 5:
                            in_kill_zone = kill_start <= now.hour < kill_end
                            past_kill_zone = now.hour >= kill_end
                            if in_asian:
                                session_phase = 'ASIAN_RANGE'
                            elif in_transition:
                                session_phase = 'TRANSITION'
                            elif in_kill_zone:
                                session_phase = 'KILL_ZONE'
                            elif past_kill_zone:
                                session_phase = 'PAST_KILL_ZONE'
                            else:
                                session_phase = 'PRE_ASIAN'
                        logger.info(f"[{symbol}] HOLD phase={session_phase} reason=outside_time_filter utc_hour={now.hour}")
                    continue

                # Initialize / reset daily state
                state = daily_state.get(symbol)
                if state is None or state.get('date') != today:
                    daily_state[symbol] = {
                        'date': today, 'asian_high': None, 'asian_low': None,
                        'range_width': None, 'traded_today': False, 'atr': 0.0,
                        'traded_kz1': False, 'traded_kz2': False
                    }
                    state = daily_state[symbol]

                asian_start = cfg.get('asian_start_hour', 0)
                asian_end = cfg.get('asian_end_hour', 6)
                kill_start = cfg.get('kill_zone_start_hour', 7)
                kill_end = cfg.get('kill_zone_end_hour', 10)
                kz2_start = cfg.get('kill_zone_2_start_hour', None)
                kz2_end = cfg.get('kill_zone_2_end_hour', None)
                _has_dual_kz = kz2_start is not None and kz2_end is not None

                # Skip if already traded this session (dual-KZ aware)
                if _has_dual_kz:
                    _in_kz2_now = kz2_start <= now.hour < kz2_end
                    if _in_kz2_now:
                        if state.get('traded_kz2', False):
                            continue
                    else:
                        if state.get('traded_kz1', False):
                            continue
                else:
                    if state['traded_today']:
                        continue

                # Determine current session phase
                # Handle cross-midnight Asian range
                if asian_start <= asian_end:
                    in_asian = asian_start <= now.hour < asian_end
                else:
                    in_asian = now.hour >= asian_start or now.hour < asian_end

                in_transition = asian_end <= now.hour < kill_start if asian_start <= asian_end else (
                    not in_asian and now.hour < kill_start
                )
                in_kz1 = kill_start <= now.hour < kill_end
                in_kz2 = _has_dual_kz and kz2_start <= now.hour < kz2_end
                in_kill_zone = in_kz1 or in_kz2
                past_kill_zone = (now.hour >= kz2_end) if _has_dual_kz else (now.hour >= kill_end)

                # Fetch rates for range calculation
                rates = mt5.copy_rates_from_pos(symbol, sym_tf, 0, 200)
                if rates is None or len(rates) < 50:
                    _mt5_fail_count += 1
                    if _mt5_fail_count >= MT5_RECONNECT_THRESHOLD:
                        logger.warning(f"[taipan] MT5 {_mt5_fail_count} consecutive failures - reinitialising")
                        try:
                            mt5.shutdown()
                            mt5.initialize()
                            _mt5_fail_count = 0
                        except Exception as reinit_err:
                            logger.error(f"[taipan] MT5 reinit failed: {reinit_err}")
                    continue
                _mt5_fail_count = 0
                rates_np = mt5_rates_to_numpy(rates)

                # ---- PHASE 1: ASIAN SESSION - calculate/update range ----
                if in_asian:
                    asian_high, asian_low, range_width, atr = calculate_asian_range(rates_np, cfg, now)
                    if asian_high is not None:
                        state['asian_high'] = asian_high
                        state['asian_low'] = asian_low
                        state['range_width'] = range_width
                        state['atr'] = atr
                    if iteration % 5 == 0:
                        if asian_high:
                            logger.info(f"[{symbol}] HOLD phase=ASIAN reason=building_range "
                                       f"range={asian_high:.5f}-{asian_low:.5f} "
                                       f"width={range_width:.5f} ATR={atr:.5f}")
                        else:
                            logger.info(f"[{symbol}] HOLD phase=ASIAN reason=range_not_yet_valid")

                    # Log ML data
                    dc = get_ml_dc()
                    if dc and iteration % 5 == 0:
                        try:
                            dc.log_signal(_build_ml_row(
                                symbol, now, rates_np, None,
                                state['atr'], state['asian_high'], state['asian_low'],
                                state['range_width'], None, 'ASIAN'
                            ))
                        except Exception:
                            pass
                    continue

                # ---- PHASE 2: TRANSITION - finalize range ----
                if in_transition:
                    if state['asian_high'] is None:
                        asian_high, asian_low, range_width, atr = calculate_asian_range(rates_np, cfg, now)
                        if asian_high is not None:
                            state['asian_high'] = asian_high
                            state['asian_low'] = asian_low
                            state['range_width'] = range_width
                            state['atr'] = atr
                            logger.info(f"[{symbol}] HOLD phase=TRANSITION reason=range_finalized "
                                       f"range={asian_high:.5f}-{asian_low:.5f} "
                                       f"width={range_width:.5f}")
                        else:
                            logger.info(f"[{symbol}] HOLD phase=TRANSITION reason=no_valid_range_skip_today")
                    continue

                # ---- PHASE 3: KILL ZONE - look for breakout ----
                if in_kill_zone:
                    if state['asian_high'] is None:
                        # Try one last time to compute range
                        asian_high, asian_low, range_width, atr = calculate_asian_range(rates_np, cfg, now)
                        if asian_high is not None:
                            state['asian_high'] = asian_high
                            state['asian_low'] = asian_low
                            state['range_width'] = range_width
                            state['atr'] = atr
                        else:
                            continue

                    asian_high = state['asian_high']
                    asian_low = state['asian_low']
                    range_width = state['range_width']
                    curr_atr = state['atr']

                    # Recompute ATR from current bars
                    curr_atr = safe_last(ATR(rates_np[:, 2], rates_np[:, 3], rates_np[:, 4],
                                            cfg.get('atr_period', 14)))

                    atr_value = curr_atr
                    if math.isnan(atr_value):
                        logger.warning(f"{symbol}: ATR is NaN - skipping")
                        continue

                    # Position count (for ML context and execution gate)
                    all_positions = broker.get_positions()
                    pg = get_portfolio_guard()
                    if pg:
                        try:
                            pg.sync_account_positions(all_positions)
                        except Exception as e:
                            if iteration % 60 == 0:
                                logger.warning(f"[PORTFOLIO-RISK] Sync failed: {e}")
                    sym_positions = [p for p in all_positions if p['symbol'] == symbol]
                    max_pos = cfg.get('max_positions', 1)
                    _at_position_limit = len(sym_positions) >= max_pos

                    # H4 trend filter
                    h4_trend = get_h4_trend(symbol, cfg)

                    # Check breakout -- always, so ML data is logged regardless of risk/balance state
                    signal = check_session_breakout(rates_np, asian_high, asian_low,
                                                   range_width, h4_trend, curr_atr, cfg)

                    # Log ML data (always -- before entry gates)
                    dc = get_ml_dc()
                    if dc:
                        try:
                            dc.log_signal(_build_ml_row(
                                symbol, now, rates_np, signal,
                                curr_atr, asian_high, asian_low,
                                range_width, h4_trend, 'KILL_ZONE'
                            ))
                        except Exception:
                            pass

                    if signal:
                        pos_tag = f" [MONITORING {len(sym_positions)}/{max_pos}]" if _at_position_limit else ""
                        logger.info(f">> {symbol}: {signal['direction']} via {signal['trigger']} -- {signal['reason']} ATR={curr_atr:.5f} H4={h4_trend} range=[{asian_low:.5f}-{asian_high:.5f}] width={range_width:.5f}{pos_tag}")

                    # --- ENTRY GATES: signal computed and logged, now check if we can act ---

                    # Risk gates
                    rm = get_risk_manager()
                    if rm:
                        can_trade, reason = rm.can_trade()
                        if not can_trade:
                            if iteration % 60 == 0: logger.warning(f"[RISK] Blocked: {reason}")
                            continue

                    # Margin gate
                    min_margin = CONFIG.get('min_margin_level', 200)
                    if min_margin > 0:
                        account_info = broker.get_account_info()
                        if account_info:
                            margin_level = account_info.get('margin_level', 9999)
                            if margin_level > 0 and margin_level < min_margin:
                                if iteration % 20 == 0:
                                    logger.warning(f"[MARGIN] {symbol}: {margin_level:.0f}% < {min_margin}%")
                                continue

                    # Per-symbol session limit
                    sym_blocked, sym_reason = is_symbol_session_limit_hit(symbol, cfg)
                    if sym_blocked:
                        logger.info(f"[SESSION_LIMIT] {sym_reason} - skipping")
                        continue

                    if signal:
                        if _at_position_limit:
                            continue
                        if CONFIG.get('news_blackout_enabled', True) and should_skip_trade(symbol):
                            logger.info(f"[taipan] News blackout active for {symbol} - skipping entry")
                            continue

                        direction = signal['direction'].lower()
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is None: continue

                        # Spread filter
                        spread = tick.ask - tick.bid
                        if curr_atr > 0 and spread > curr_atr * 0.3:
                            logger.info(f"{symbol}: Spread too wide ({spread:.5f} > {curr_atr * 0.3:.5f})")
                            entry_price = tick.ask if direction == 'buy' else tick.bid
                            send_webhook({"event": "rejected", "symbol": symbol,
                                "direction": direction, "entry_price": entry_price,
                                "reason": "spread_too_wide"})
                            continue

                        entry_price = tick.ask if direction == 'buy' else tick.bid
                        sl = calculate_sl(direction, asian_high, asian_low, curr_atr, cfg)
                        min_stop_points = float(cfg.get('min_stop_points', 0) or 0)
                        if min_stop_points > 0:
                            try:
                                sym_info = broker.get_symbol_info(symbol)
                                point = float(sym_info.get('point', 0)) if sym_info else 0.0
                                if point > 0:
                                    min_sl_distance = point * min_stop_points
                                    sl_distance = abs(entry_price - sl)
                                    if sl_distance < min_sl_distance:
                                        sl_distance = min_sl_distance
                                        sl = entry_price - sl_distance if direction == 'buy' else entry_price + sl_distance
                            except Exception:
                                pass
                        tp = calculate_tp(direction, entry_price, range_width, cfg)
                        volume = calculate_position_size(symbol, cfg, sl, entry_price)

                        # Pre-flight SL check
                        if direction == 'buy' and sl >= entry_price:
                            logger.warning(f"{symbol}: BUY SL {sl} >= entry {entry_price} - aborting")
                            continue
                        if direction == 'sell' and sl <= entry_price:
                            logger.warning(f"{symbol}: SELL SL {sl} <= entry {entry_price} - aborting")
                            continue

                        oracle_cfg = CONFIG.get('oracle_guard', {})
                        if oracle_cfg.get('enabled', True):
                            oracle_allowed, oracle_reason, _ = request_oracle_approval(
                                CONFIG, 'taipan', PROFILE_NAME, symbol, direction, volume,
                                entry_price=entry_price, sl=sl, tp=tp
                            )
                            if not oracle_allowed:
                                logger.warning(f"[ORACLE] Blocked {symbol} {direction.upper()}: {oracle_reason}")
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
                            state['traded_today'] = True
                            if _has_dual_kz and in_kz2:
                                state['traded_kz2'] = True
                            else:
                                state['traded_kz1'] = True
                            with state_lock:
                                last_entry_time[symbol] = time.time()
                            logger.info(f"[OK] {direction.upper()} {symbol} @ {entry_price:.5f} "
                                       f"vol={volume} SL={sl:.5f} TP={tp:.5f} "
                                       f"range=[{asian_low:.5f}-{asian_high:.5f}]")

                            tl = get_trade_logger()
                            if tl:
                                try:
                                    tl.log_entry(symbol=symbol, direction=direction, price=entry_price, size=volume,
                                        decision_factors={"trigger": "SESSION_BREAKOUT", "confidence": 0.80,
                                            "reason": signal['reason'], "atr": curr_atr, "account": ACCOUNT_NUMBER,
                                            "asian_high": asian_high, "asian_low": asian_low,
                                            "range_width": range_width, "h4_trend": h4_trend},
                                        market_context={"entry_price": entry_price, "sl": sl, "tp": tp})
                                except Exception as e:
                                    logger.warning(f"Trade log failed: {e}")

                            # Extract ticket from result
                            ticket_num = None
                            try:
                                ticket_num = result.order
                            except Exception:
                                pass

                            notify_portfolio_trade_open(ticket_num, symbol, direction.upper(), float(volume))

                            # ML data - executed trade
                            dc = get_ml_dc()
                            if dc:
                                try:
                                    dc.log_signal(_build_ml_row(
                                        symbol, now, rates_np, signal,
                                        curr_atr, asian_high, asian_low,
                                        range_width, h4_trend, 'KILL_ZONE',
                                        executed=1, entry_price=entry_price,
                                        sl=sl, tp=tp, volume=volume,
                                        ticket=ticket_num
                                    ))
                                except Exception:
                                    pass

                            send_webhook({"event": "trade", "symbol": symbol, "direction": direction,
                                "trigger": "SESSION_BREAKOUT", "entry_price": entry_price, "sl": sl, "tp": tp,
                                "volume": volume, "strategy": "taipan_session_breakout",
                                "timeframe": CONFIG.get('timeframe', 'M30'),
                                "asian_high": asian_high, "asian_low": asian_low,
                                "range_width": range_width})
                        else:
                            logger.warning(f"XX {symbol}: Order rejected")
                            send_webhook({"event": "rejected", "symbol": symbol,
                                "direction": direction, "entry_price": entry_price,
                                "reason": "broker_rejected"})
                    elif iteration % 5 == 0:
                        logger.info(f"[{symbol}] HOLD phase=KILL_ZONE reason=no_breakout "
                                   f"range={asian_low:.5f}-{asian_high:.5f} H4={h4_trend} ATR={curr_atr:.5f}")
                    continue

                # ---- PHASE 4: PAST KILL ZONE - no new entries ----
                if past_kill_zone:
                    if iteration % 20 == 0:
                        logger.info(f"[{symbol}] HOLD phase=PAST_KILL_ZONE reason=trailing_only")
                    continue

            except Exception as e:
                logger.error(f"{symbol} error: {e}")

        iteration += 1
        if iteration % 5 == 0:
            logger.info(f"[HEARTBEAT] TAIPAN loop -- iteration {iteration}")
        send_webhook({"event": "heartbeat", "iteration": iteration})
        time.sleep(CONFIG.get('strategy_interval', 60))

# ============================================================================
# TRAILING STOP + BREAKEVEN
# ============================================================================
def trailing_loop():
    broker = get_broker()
    default_tf_str = CONFIG.get('timeframe', 'M30')
    timeframe = TF_MAP.get(default_tf_str, mt5.TIMEFRAME_M30)
    trail_iter = 0
    global _tracked_positions

    while not bot_stop.is_set():
        try:
            positions = broker.get_positions()
            trail_iter += 1
            if trail_iter % 20 == 0:
                logger.info(f"[HEARTBEAT] Trailing -- {len(positions)} positions")

            # Track positions and detect closures
            current_tickets = {pos['ticket'] for pos in positions}

            for pos in positions:
                ticket = pos['ticket']
                if ticket not in _tracked_positions:
                    _tracked_positions[ticket] = {
                        'symbol': pos['symbol'],
                        'entry_price': float(pos.get('price_open', pos.get('price', 0))),
                        'sl': float(pos.get('sl') or 0.0),
                        'tp': float(pos.get('tp') or 0.0),
                        'volume': float(pos.get('volume', 0.01)),
                        'open_time': datetime.now(timezone.utc),
                        'type': pos.get('type', '')
                    }

            # Detect closed positions
            closed_tickets = set(_tracked_positions.keys()) - current_tickets
            for ticket in closed_tickets:
                try:
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
                        log_trade_outcome(
                            symbol, ticket, entry['entry_price'], entry['sl'], entry['tp'],
                            entry['volume'], close_price, profit, exit_reason, duration
                        )
                        record_trade_result(symbol, profit)
                        notify_portfolio_trade_close(ticket, profit, duration * 60.0, symbol)
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
                            _tracked_positions.pop(ticket, None)
                            clear_pending_close(str(ticket))
                except Exception as e:
                    logger.warning(f"Failed to log outcome for closed ticket {ticket}: {e}", exc_info=True)

            now = datetime.now(timezone.utc)

            for pos in positions:
                symbol = pos['symbol']
                cfg = CONFIG['symbols'].get(symbol)
                if cfg is None or not cfg.get('use_trailing', True): continue
                sym_tf = TF_MAP.get(cfg.get('timeframe', default_tf_str), timeframe)

                # Friday flatten
                if now.weekday() == 4:
                    tf = cfg.get('time_filter', {})
                    flatten_hour = tf.get('friday_flatten_hour', 0)
                    if flatten_hour > 0 and now.hour >= flatten_hour:
                        ticket = pos.get('ticket', 0)
                        logger.info(f"[FRIDAY-FLATTEN] Closing {symbol} #{ticket} before weekend")
                        unrealized = pos.get('profit', 0.0)
                        broker.close_position(ticket)
                        record_trade_result(symbol, unrealized)
                        send_webhook({"event": "friday_flatten", "ticket": ticket, "symbol": symbol})
                        continue

                rates = mt5.copy_rates_from_pos(symbol, sym_tf, 0, 200)
                if rates is None or len(rates) < 50: continue
                rates_np = mt5_rates_to_numpy(rates)
                curr_atr = safe_last(ATR(rates_np[:, 2], rates_np[:, 3], rates_np[:, 4], 14))

                pos_type = normalize_pos_type(pos.get('type', ''))
                ticket = pos.get('ticket', 0)
                if pos_type is None: continue

                tick = mt5.symbol_info_tick(symbol)
                if tick is None: continue

                current_sl = float(pos.get('sl') or 0.0)
                entry_price = float(pos.get('price_open', pos.get('price', 0)))
                digits = get_symbol_digits(symbol)

                # Breakeven
                be_mult = cfg.get('breakeven_atr_mult', 1.0)
                trail_start_mult = cfg.get('trail_start_atr_mult', be_mult)
                be_offset = curr_atr * 0.1
                if entry_price > 0:
                    if pos_type == 'BUY':
                        if tick.bid >= entry_price + (curr_atr * be_mult) and current_sl < entry_price:
                            be_sl = entry_price + be_offset
                            broker.modify_position(ticket, sl=round(be_sl, digits))
                            logger.info(f"[BE] {symbol} #{ticket}: SL -> {be_sl:.{digits}f}")
                            send_webhook({"event": "breakeven", "symbol": symbol,
                                "direction": pos_type.lower(), "old_sl": current_sl,
                                "new_sl": round(be_sl, digits), "ticket": ticket})
                            continue
                    elif pos_type == 'SELL':
                        if tick.ask <= entry_price - (curr_atr * be_mult) and (current_sl > entry_price or current_sl == 0):
                            be_sl = entry_price - be_offset
                            broker.modify_position(ticket, sl=round(be_sl, digits))
                            logger.info(f"[BE] {symbol} #{ticket}: SL -> {be_sl:.{digits}f}")
                            send_webhook({"event": "breakeven", "symbol": symbol,
                                "direction": pos_type.lower(), "old_sl": current_sl,
                                "new_sl": round(be_sl, digits), "ticket": ticket})
                            continue

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
                        send_webhook({"event": "trail_update", "symbol": symbol,
                            "direction": pos_type.lower(), "old_sl": current_sl,
                            "new_sl": round(new_sl, digits), "ticket": ticket})
                elif pos_type == 'SELL':
                    new_sl = tick.ask + trail_dist
                    if (new_sl < current_sl or current_sl == 0) and new_sl > 0:
                        broker.modify_position(ticket, sl=round(new_sl, digits))
                        logger.info(f"[TRAIL] {symbol} #{ticket}: SL {current_sl:.{digits}f} -> {new_sl:.{digits}f}")
                        send_webhook({"event": "trail_update", "symbol": symbol,
                            "direction": pos_type.lower(), "old_sl": current_sl,
                            "new_sl": round(new_sl, digits), "ticket": ticket})
        except Exception as e:
            logger.error(f"[TRAIL] Error: {e}")
        time.sleep(60)

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/status', methods=['GET'])
def get_status():
    broker = get_broker()
    account = broker.get_account_info() if broker else None
    positions = broker.get_positions() if broker else []
    return jsonify({'bot': 'taipan-v1', 'profile': PROFILE_NAME, 'account': ACCOUNT_NUMBER,
        'timeframe': CONFIG.get('timeframe', 'M30'), 'balance': account.get('balance', 0) if account else 0,
        'equity': account.get('equity', 0) if account else 0, 'positions_count': len(positions),
        'timestamp': datetime.now(timezone.utc).isoformat()})

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
    entry = _tracked_positions.get(ticket)
    result = broker.close_position(ticket)
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
                log_trade_outcome(symbol, ticket, entry['entry_price'], entry['sl'], entry['tp'],
                    entry['volume'], close_price, profit, 'MANUAL', duration)
                notify_portfolio_trade_close(ticket, profit, duration * 60.0, symbol)
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
    timeframe = TF_MAP.get(CONFIG.get('timeframe', 'M30'), mt5.TIMEFRAME_M30)
    cfg = CONFIG['symbols'].get(symbol, {})
    now = datetime.now(timezone.utc)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
    if rates is None or len(rates) < 50:
        return jsonify({'error': 'No data'}), 400
    rates_np = mt5_rates_to_numpy(rates)
    asian_high, asian_low, range_width, curr_atr = calculate_asian_range(rates_np, cfg, now)
    h4_trend = get_h4_trend(symbol, cfg)
    signal = None
    if asian_high is not None:
        signal = check_session_breakout(rates_np, asian_high, asian_low, range_width, h4_trend, curr_atr, cfg)
    return jsonify({
        'symbol': symbol, 'strategy': 'taipan_session_breakout',
        'signal': signal['direction'] if signal else 'HOLD',
        'trigger': signal['trigger'] if signal else None,
        'atr': curr_atr,
        'confidence': signal['confidence'] if signal else 0,
        'reason': signal['reason'] if signal else 'No breakout',
        'timeframe': CONFIG.get('timeframe', 'M30'),
        'asian_high': asian_high, 'asian_low': asian_low,
        'range_width': range_width, 'h4_trend': h4_trend,
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

@app.route('/reload', methods=['POST'])
def reload_config_endpoint():
    global CONFIG
    try:
        new_config = load_config(PROFILE_NAME)
        CONFIG.update(new_config)
        logger.info(f"[RELOAD] Config reloaded for profile {PROFILE_NAME}")
        return jsonify({
            'success': True,
            'message': f'Config reloaded for {PROFILE_NAME}',
            'symbols': list(CONFIG.get('symbols', {}).keys()),
        })
    except Exception as e:
        logger.error(f"[RELOAD] Failed: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# INIT
# ============================================================================
def load_config(profile):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'taipan_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found")
    return profiles[profile]

def initialize():
    global CONFIG, ACCOUNT_NUMBER, PROFILE_NAME
    parser = argparse.ArgumentParser(description='TAIPAN v1 -- Session Breakout Bot')
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    PROFILE_NAME = args.profile
    CONFIG = load_config(PROFILE_NAME)
    ACCOUNT_NUMBER = CONFIG['account_number']
    init_logger()
    logger.info("=" * 60)
    logger.info(f"TAIPAN v1 Starting -- Profile: {PROFILE_NAME}")
    logger.info(f"Account: {ACCOUNT_NUMBER}, Timeframe: {CONFIG.get('timeframe')}")
    logger.info(f"Strategy: Asian Session Range Breakout at London/NY Kill Zone")
    logger.info("=" * 60)
    password = os.environ.get('MT5_PASSWORD', '')
    if not password:
        logger.error("MT5_PASSWORD not set.")
        return False
    try:
        broker = MT5Broker(account=ACCOUNT_NUMBER, password=password,
            server=CONFIG['server'], mt5_path=CONFIG.get('mt5_path'), owner_tag=f"{BOT_NAME}:{PROFILE_NAME}")
        set_broker(broker)
        logger.info("Broker connected")
    except Exception as e:
        logger.error(f"Broker failed: {e}")
        return False
    try:
        rc = CONFIG.get('risk', {})
        account_info = broker.get_account_info() or {}
        curr_balance = account_info.get('balance', 0.0)
        curr_equity = account_info.get('equity', 0.0)
        capital_base = max(abs(curr_balance), abs(curr_equity), 150.0)
        daily_pct = rc.get('daily_loss_limit_pct', 13)
        daily_dollar = -(capital_base * daily_pct / 100)
        rm = RiskManagerUltra(
            daily_loss_limit=daily_dollar,
            max_consecutive_losses=rc.get('max_consecutive_losses', 3),
            max_drawdown_percent=rc.get('max_drawdown_pct', 8),
            state_file=f"risk_state_ultra_{ACCOUNT_NUMBER}.json",
        )
        set_risk_manager(rm)
        logger.info(f"Risk manager ready: daily limit ${daily_dollar:.2f} ({daily_pct}% of base ${capital_base:.2f})")
    except Exception as e:
        logger.error(f"Risk manager failed: {e}")
        return False
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        portfolio_cfg = dict(CONFIG.get('portfolio_risk', {}))
        if 'global_daily_loss_limit' in CONFIG:
            portfolio_cfg['global_daily_loss_limit'] = CONFIG['global_daily_loss_limit']
        pg = PortfolioRiskGuard(
            bot_name='taipan',
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
        tl = TradeDecisionLogger(f"taipan_v1_{PROFILE_NAME}")
        set_trade_logger(tl)
    except Exception as e:
        logger.warning(f"Trade logger failed (non-fatal): {e}")
    try:
        dc = SharedDataCollector('taipan', ACCOUNT_NUMBER)
        set_ml_dc(dc)
        logger.info(f"ML data collector initialized - saving to {dc.data_dir}")
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
    Safe to call on every restart - server-side upsert means no duplicates."""
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
                "direction":   pos.get("type"),
                "entry_price": pos.get("price_open"),
                "sl":          pos.get("sl"),
                "tp":          pos.get("tp", 0.0),
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
    port = CONFIG.get('server_port', 8055)
    logger.info(f"Flask API on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)

if __name__ == '__main__':
    main()
