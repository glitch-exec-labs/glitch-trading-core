"""
MAMBA v1 -- Mean-Reversion Range Bot (M15)
Fills the gap when Hawk/Cobra/Viper sit idle in ranging markets.

Regime filter: ADX < threshold = ranging = TRADE, ADX >= threshold = trending = SIT OUT

Triggers:
  1. BB Lower Fade: price at/below lower Bollinger Band + RSI oversold + ADX ranging
  2. BB Upper Fade: price at/above upper Bollinger Band + RSI overbought + ADX ranging

Exits: TP at Bollinger midline (SMA), SL beyond band + ATR buffer, breakeven at 1x ATR
Timeframe: M15 (~5-15 trades per day in ranging conditions)
Risk: 1% per trade, max 1 position per symbol
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
BOT_NAME = 'mamba'
API_KEY = os.environ.get("MAMBA_API_KEY", "")

state_lock = threading.Lock()
last_entry_time = {}
bot_stop = threading.Event()
_strategy_thread = None
_trailing_thread = None

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
    logger = logging.getLogger(f"MAMBA-{ACCOUNT_NUMBER}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"mamba_{ACCOUNT_NUMBER}.log", encoding='utf-8')
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


_portfolio_guard = None
def get_portfolio_guard(): return _portfolio_guard
def set_portfolio_guard(pg):
    global _portfolio_guard
    _portfolio_guard = pg

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

_prop_guard = None
def get_prop_guard(): return _prop_guard
def set_prop_guard(pg):
    global _prop_guard
    _prop_guard = pg

_ml_dc = None
_tracked_positions = {}  # ticket -> {symbol, entry_price, sl, tp, volume, open_time, type}
positions_lock = threading.Lock()
_pending_close_checks = {}
pending_close_lock = threading.Lock()
def get_ml_dc(): return _ml_dc
def set_ml_dc(dc):
    global _ml_dc
    _ml_dc = dc

# ============================================================================
# MT5 TIMEFRAME MAP
# ============================================================================
TF_MAP = {
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
}

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
        'timeframe':     CONFIG.get('timeframe', 'M15'),
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
        # Hawk-specific
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

# ============================================================================
# UTILITY
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

def determine_exit_reason(entry_price, close_price, sl, tp, symbol):
    try:
        sym_info = mt5.symbol_info(symbol)
        tolerance = (sym_info.point * 5) if sym_info else 0.0001
        if tp and abs(close_price - tp) <= tolerance:
            exit_reason = 'TP_HIT'
        elif sl and abs(close_price - sl) <= tolerance:
            if abs(sl - entry_price) <= tolerance:
                exit_reason = 'BREAKEVEN'
            else:
                exit_reason = 'SL_HIT'
        else:
            exit_reason = 'TRAILING_SL'
    except Exception:
        return 'UNKNOWN'
    return exit_reason

def log_trade_outcome(symbol, ticket, entry_price, sl, tp, volume, close_price, profit, exit_reason, duration_minutes, atr=0, account_balance=None, account_equity=None):
    try:
        outcome = 'WIN' if profit > 0.01 else ('LOSS' if profit < -0.01 else 'BREAKEVEN')
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
        payload.update({"bot": "mamba", "account": ACCOUNT_NUMBER,
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

def BollingerBands(closes, period=20, std_mult=2.0):
    """Calculate Bollinger Bands: upper, mid (SMA), lower."""
    n = len(closes)
    if n < period:
        mid = float(closes[-1])
        return mid + 1, mid, mid - 1  # Fallback

    recent = closes[-period:]
    mid = float(np.mean(recent))
    std = float(np.std(recent))

    upper = mid + std * std_mult
    lower = mid - std * std_mult

    return upper, mid, lower

def calc_adx(highs, lows, closes, period=14):
    """
    Calculate ADX (Average Directional Index) using Wilder's smoothing.
    Returns current ADX value (0-100). Higher = stronger trend.
    """
    n = len(closes)
    if n < period * 2 + 2:
        return 0.0

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = float(highs[i] - highs[i - 1])
        down_move = float(lows[i - 1] - lows[i])
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    tr = np.zeros(n)
    tr[0] = float(highs[0] - lows[0])
    for i in range(1, n):
        tr[i] = max(
            float(highs[i] - lows[i]),
            abs(float(highs[i] - closes[i - 1])),
            abs(float(lows[i] - closes[i - 1]))
        )

    atr_s = np.zeros(n)
    pdm_s = np.zeros(n)
    mdm_s = np.zeros(n)

    atr_s[period] = np.sum(tr[1:period + 1])
    pdm_s[period] = np.sum(plus_dm[1:period + 1])
    mdm_s[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - (atr_s[i - 1] / period) + tr[i]
        pdm_s[i] = pdm_s[i - 1] - (pdm_s[i - 1] / period) + plus_dm[i]
        mdm_s[i] = mdm_s[i - 1] - (mdm_s[i - 1] / period) + minus_dm[i]

    dx = np.zeros(n)
    for i in range(period, n):
        if atr_s[i] > 0:
            pdi = 100 * pdm_s[i] / atr_s[i]
            mdi = 100 * mdm_s[i] / atr_s[i]
        else:
            pdi = mdi = 0
        di_sum = pdi + mdi
        if di_sum > 0:
            dx[i] = 100 * abs(pdi - mdi) / di_sum

    adx = np.zeros(n)
    start = 2 * period
    if start >= n:
        return 0.0
    adx[start] = float(np.mean(dx[period + 1:start + 1]))
    for i in range(start + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return float(adx[-1])

# ============================================================================
# SIGNAL DETECTION
# ============================================================================
def check_bb_fade(rates_np, cfg):
    """
    Mean-reversion signal: fade Bollinger Band extremes when market is ranging.

    Returns: (signal_dict or None, atr_value, adx_value, bb_upper, bb_mid, bb_lower, rsi_value)
    """
    closes = rates_np[:, 4]
    highs = rates_np[:, 2]
    lows = rates_np[:, 3]

    bb_period = cfg.get('bb_period', 20)
    bb_std = cfg.get('bb_std_mult', 2.0)
    rsi_period = cfg.get('rsi_period', 14)
    atr_period = cfg.get('atr_period', 14)
    adx_period = cfg.get('adx_period', 14)
    adx_threshold = cfg.get('adx_threshold', 25)
    rsi_oversold = cfg.get('rsi_oversold', 30)
    rsi_overbought = cfg.get('rsi_overbought', 70)

    min_bars = max(bb_period, adx_period * 2 + 5, rsi_period) + 10
    if len(closes) < min_bars:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    curr_atr = safe_last(ATR(highs, lows, closes, atr_period))
    if curr_atr <= 0:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    adx = calc_adx(highs, lows, closes, adx_period)
    upper, mid, lower = BollingerBands(closes, bb_period, bb_std)

    rsi_vals = RSI(closes, rsi_period)
    curr_rsi = float(rsi_vals[-2]) if isinstance(rsi_vals, np.ndarray) else float(rsi_vals)

    price = float(closes[-2])

    # Price position in BB (0=lower band, 0.5=midline, 1=upper band)
    bb_range = upper - lower
    price_pos_in_bb = (price - lower) / bb_range if bb_range > 0 else 0.5

    if adx >= adx_threshold:
        return None, curr_atr, adx, upper, mid, lower, curr_rsi  # Trending -- sit out

    # bb_entry_pct: how far inside the band counts as "near the band".
    # 0.0  = price must be at or beyond the band (original strict behaviour).
    # 0.15 = price within the outer 15% of the band range also qualifies.
    # E.g. bb_range=$5, entry_pct=0.15 → triggers within $0.75 of either band.
    #
    # Two-tier RSI confirmation:
    #   Exact band touch  (BB_LOWER_FADE / BB_UPPER_FADE):  strict RSI ≤ rsi_oversold / ≥ rsi_overbought
    #   Near-band approach(BB_LOWER_APPROACH / BB_UPPER_APP):loose RSI < 50 for buy, > 50 for sell
    #   This avoids the anti-correlation problem in low-ADX markets where extreme RSI
    #   and extreme price rarely coincide.
    bb_entry_pct = cfg.get('bb_entry_pct', 0.0)
    proximity     = bb_range * bb_entry_pct
    near_lower    = price <= lower + proximity
    near_upper    = price >= upper - proximity

    if near_lower:
        at_band   = price <= lower
        rsi_ok    = (curr_rsi <= rsi_oversold) if at_band else (curr_rsi < 50)
        if rsi_ok:
            trigger = 'BB_LOWER_FADE' if at_band else 'BB_LOWER_APPROACH'
            conf    = 0.70 if at_band else 0.60
            return {
                'trigger': trigger,
                'direction': 'BUY',
                'confidence': conf,
                'reason': f'Price {price:.5f} near/below lower BB {lower:.5f} ({price_pos_in_bb*100:.0f}%), RSI={curr_rsi:.1f}, ADX={adx:.1f}'
            }, curr_atr, adx, upper, mid, lower, curr_rsi

    if near_upper:
        at_band   = price >= upper
        rsi_ok    = (curr_rsi >= rsi_overbought) if at_band else (curr_rsi > 50)
        if rsi_ok:
            trigger = 'BB_UPPER_FADE' if at_band else 'BB_UPPER_APPROACH'
            conf    = 0.70 if at_band else 0.60
            return {
                'trigger': trigger,
                'direction': 'SELL',
                'confidence': conf,
                'reason': f'Price {price:.5f} near/above upper BB {upper:.5f} ({price_pos_in_bb*100:.0f}%), RSI={curr_rsi:.1f}, ADX={adx:.1f}'
            }, curr_atr, adx, upper, mid, lower, curr_rsi

    return None, curr_atr, adx, upper, mid, lower, curr_rsi

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
    """SL at fixed ATR distance (tighter for mean-reversion)."""
    mult = cfg.get('atr_sl_multiplier', 1.0)
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

def calculate_tp_midline(direction, bb_mid):
    """TP at Bollinger Band midline (SMA). The core mean-reversion exit."""
    return bb_mid

def calculate_position_size(symbol, cfg, sl_price, entry_price):
    broker = get_broker()
    if broker is None:
        logger.warning("calculate_position_size: broker not initialized")
        return None
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
                f"-- raw lots {raw_lots:.4f} clamped to min_lot {clamped:.2f}. "
                f"Consider lowering risk_percent for {symbol}."
            )
        elif clamped == max_lot and actual_risk < risk_amount * 0.5:
            logger.warning(
                f"[SIZING] {symbol}: target {risk_pct}% (${risk_amount:.2f}) "
                f"but actual ${actual_risk:.2f} ({actual_risk / balance * 100:.1f}%) "
                f"-- raw lots {raw_lots:.4f} clamped to max_lot {clamped:.2f}."
            )
    return clamped

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
    broker = get_broker()
    default_tf_str = CONFIG.get('timeframe', 'M15')
    timeframe = TF_MAP.get(default_tf_str, mt5.TIMEFRAME_M15)
    cooldown = CONFIG.get('entry_cooldown_seconds', 900)
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
            logger.error("CONFIG missing 'symbols' key -- check config file")
            time.sleep(30)
            continue
        for symbol, cfg in symbols_cfg.items():
            if not cfg.get('enabled', False):
                continue
            sym_tf = TF_MAP.get(cfg.get('timeframe', default_tf_str), timeframe)

            try:
                # 1. Trading hours
                if not is_within_trading_hours(cfg):
                    if iteration % 10 == 0:
                        rates = mt5.copy_rates_from_pos(symbol, sym_tf, 0, 200)
                        if rates is not None and len(rates) >= 50:
                            rates_np = mt5_rates_to_numpy(rates)
                            signal, atr_value, adx_value, bb_upper, bb_mid, bb_lower, curr_rsi = check_bb_fade(rates_np, cfg)
                            adx_str = f"ADX={adx_value:.1f}"
                            rsi_str = f"RSI={curr_rsi:.1f}" if curr_rsi is not None else "RSI=?"
                            price_now = float(rates_np[-2, 4]) if len(rates_np) >= 2 else 0
                            if signal:
                                logger.info(f"[{symbol}] Outside hours decision: {signal['direction']} via {signal['trigger']} - {signal['reason']} ATR={atr_value:.5f} {rsi_str} {adx_str}")
                                hold_reason = None
                            else:
                                adx_threshold = cfg.get('adx_threshold', 25)
                                rsi_oversold = cfg.get('rsi_oversold', 30)
                                rsi_overbought = cfg.get('rsi_overbought', 70)
                                if adx_value >= adx_threshold:
                                    hold_reason = 'adx_too_high'
                                else:
                                    price_at_lower = price_now <= bb_lower
                                    price_at_upper = price_now >= bb_upper
                                    rsi_oversold_ok = curr_rsi <= rsi_oversold
                                    rsi_overbought_ok = curr_rsi >= rsi_overbought
                                    failed = []
                                    if not (price_at_lower or price_at_upper):
                                        failed.append('price_not_at_band')
                                    if not (rsi_oversold_ok or rsi_overbought_ok):
                                        failed.append('rsi_not_extreme')
                                    hold_reason = ','.join(failed) if failed else 'conditions_not_met'
                                logger.info(f"[{symbol}] Outside hours HOLD: {hold_reason} ATR={atr_value:.5f} {rsi_str} {adx_str}")
                            dc = get_ml_dc()
                            if dc:
                                try:
                                    row = build_ml_row(symbol, 'OUTSIDE_HOURS', 'NONE', 0, atr_value, 0, rates_np)
                                    row['adx'] = adx_value
                                    row['hold_reason'] = hold_reason
                                    row['spread_points'] = get_spread_points(symbol)
                                    dc.log_signal(row)
                                except Exception:
                                    pass
                    continue

                # 3. Position count (tracked for ML context and execution gate)
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

                # 5. Market data (always -- ML signal generation continues regardless of risk/balance state)
                rates = mt5.copy_rates_from_pos(symbol, sym_tf, 0, 200)
                if rates is None or len(rates) < 60:
                    _mt5_fail_count += 1
                    if _mt5_fail_count >= MT5_RECONNECT_THRESHOLD:
                        logger.warning(f"[{BOT_NAME}] MT5 {_mt5_fail_count} consecutive failures -- reinitialising")
                        try:
                            mt5.shutdown()
                            mt5.initialize()
                            _mt5_fail_count = 0
                        except Exception as reinit_err:
                            logger.error(f"[{BOT_NAME}] MT5 reinit failed: {reinit_err}")
                    continue
                _mt5_fail_count = 0
                rates_np = mt5_rates_to_numpy(rates)

                # 6. Signal check
                signal, atr_value, adx_value, bb_upper, bb_mid, bb_lower, curr_rsi = check_bb_fade(rates_np, cfg)

                if atr_value is None or math.isnan(atr_value) or atr_value <= 0:
                    logger.warning(f"[{BOT_NAME}] ATR invalid ({atr_value}) for {symbol} -- skipping bar")
                    continue

                # Compute price position in BB for logging
                bb_range = bb_upper - bb_lower
                price_now = float(rates_np[-2, 4]) if len(rates_np) >= 2 else 0
                price_pos_in_bb = (price_now - bb_lower) / bb_range if bb_range > 0 else 0.5

                # 7. Log
                adx_str = f"ADX={adx_value:.1f}"
                rsi_str = f"RSI={curr_rsi:.1f}" if curr_rsi is not None else "RSI=?"
                bb_pct = f"BB={price_pos_in_bb:.0%}"
                if signal:
                    pos_tag = f" [MONITORING {len(sym_positions)}/{max_pos}]" if _at_position_limit else ""
                    logger.info(f">> {symbol}: {signal['direction']} via {signal['trigger']} -- {signal['reason']} ATR={atr_value:.5f} {rsi_str} {adx_str} {bb_pct}{pos_tag}")
                elif iteration % 10 == 0:
                    regime = "RANGING" if adx_value < cfg.get('adx_threshold', 25) else "TRENDING (idle)"
                    logger.info(f"{symbol}: HOLD ({regime}) ATR={atr_value:.5f} {rsi_str} {adx_str} {bb_pct}")

                # ML data -- every check including HOLD
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
                        # Compute detailed hold_reason from which conditions failed
                        hold_reason = None
                        if not signal:
                            adx_threshold = cfg.get('adx_threshold', 25)
                            rsi_oversold = cfg.get('rsi_oversold', 30)
                            rsi_overbought = cfg.get('rsi_overbought', 70)
                            if adx_value >= adx_threshold:
                                hold_reason = 'adx_too_high'
                            else:
                                # Check which sub-conditions failed
                                price_at_lower = price_now <= bb_lower
                                price_at_upper = price_now >= bb_upper
                                rsi_oversold_ok = curr_rsi <= rsi_oversold
                                rsi_overbought_ok = curr_rsi >= rsi_overbought
                                failed = []
                                if not (price_at_lower or price_at_upper):
                                    failed.append('price_not_at_band')
                                if not (rsi_oversold_ok or rsi_overbought_ok):
                                    failed.append('rsi_not_extreme')
                                hold_reason = ','.join(failed) if failed else 'conditions_not_met'
                        row.update({
                            'bb_upper':           bb_upper,
                            'bb_lower':           bb_lower,
                            'bb_mid':             bb_mid,
                            'price_position_in_bb': price_pos_in_bb,
                            'adx':                adx_value,
                            'rsi':                curr_rsi,
                            'hold_reason':        hold_reason,
                            'spread_points':      get_spread_points(symbol),
                        })
                        dc.log_signal(row)
                    except Exception:
                        pass

                # --- ENTRY GATES: signal computed and logged, now check if we can act ---

                # 2. Cooldown
                adaptive_cd = get_adaptive_cooldown(symbol, cooldown, cfg)
                with state_lock:
                    if time.time() - last_entry_time.get(symbol, 0) < adaptive_cd:
                        continue

                # 4. Risk gate
                rm = get_risk_manager()
                if rm:
                    can_trade, reason = rm.can_trade(symbol=symbol)
                    if not can_trade:
                        if iteration % 60 == 0:
                            logger.warning(f"[RISK] Blocked: {reason}")
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
                                    f"[MARGIN] {symbol}: Margin level {margin_level:.0f}% < {min_margin}% -- skipping entry"
                                )
                            continue

                # Per-symbol session limit
                sym_blocked, sym_reason = is_symbol_session_limit_hit(symbol, cfg)
                if sym_blocked:
                    logger.info(f"[SESSION_LIMIT] {sym_reason} -- skipping")
                    continue

                # 8. Execute
                if signal:
                    if _at_position_limit:
                        continue
                    if CONFIG.get('news_blackout_enabled', True) and should_skip_trade(symbol):
                        logger.info(f"[{BOT_NAME}] News blackout active for {symbol} -- skipping entry")
                        continue
                    adx_threshold = cfg.get('adx_threshold', 25)
                    if adx_value >= adx_threshold:
                        if iteration % 10 == 0:
                            logger.info(f"{symbol}: ADX {adx_value:.1f} >= {adx_threshold} (trend regime) - entry blocked")
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
                    if tick is None:
                        continue

                    spread = tick.ask - tick.bid
                    max_spread_mult = cfg.get('max_spread_atr_mult', CONFIG.get('max_spread_atr_mult', 0.30))
                    if atr_value > 0 and spread > atr_value * max_spread_mult:
                        logger.info(
                            f"{symbol}: Spread {spread:.5f} > {max_spread_mult:.2f} * ATR {atr_value:.5f}, skipping"
                        )
                        continue

                    entry_price = tick.ask if direction == 'buy' else tick.bid
                    cfg_runtime = dict(cfg)
                    cfg_runtime['_symbol_name'] = symbol
                    sl = calculate_sl(direction, entry_price, atr_value, cfg_runtime)
                    tp = calculate_tp_midline(direction, bb_mid)

                    # Sanity: TP must be on the right side
                    if direction == 'buy' and tp <= entry_price:
                        logger.info(f"{symbol}: BB midline {tp:.5f} <= entry {entry_price:.5f}, skipping")
                        continue
                    if direction == 'sell' and tp >= entry_price:
                        logger.info(f"{symbol}: BB midline {tp:.5f} >= entry {entry_price:.5f}, skipping")
                        continue

                    # Min R:R check
                    sl_dist = abs(entry_price - sl)
                    tp_dist = abs(tp - entry_price)
                    min_rr = cfg.get('min_reward_risk', 1.0)
                    if sl_dist > 0 and tp_dist / sl_dist < min_rr:
                        logger.info(f"{symbol}: R:R {tp_dist/sl_dist:.2f} < {min_rr}, skipping")
                        continue

                    volume = calculate_position_size(symbol, cfg, sl, entry_price)

                    # Pre-flight SL check -- must be on correct side of entry
                    if direction == "buy" and sl >= entry_price:
                        logger.error(f"[{BOT_NAME}] SL {sl} >= entry {entry_price} on BUY -- aborting trade")
                        continue
                    if direction == "sell" and sl <= entry_price:
                        logger.error(f"[{BOT_NAME}] SL {sl} <= entry {entry_price} on SELL -- aborting trade")
                        continue

                    oracle_cfg = CONFIG.get('oracle_guard', {})
                    if oracle_cfg.get('enabled', True):
                        oracle_allowed, oracle_reason, _ = request_oracle_approval(
                            CONFIG, BOT_NAME, PROFILE_NAME, symbol, direction, volume,
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
                        with state_lock:
                            last_entry_time[symbol] = time.time()

                        logger.info(
                            f"[OK] {direction.upper()} {symbol} via {signal['trigger']} @ {entry_price:.5f} "
                            f"vol={volume} SL={sl:.5f} TP={tp:.5f} (BB mid) {adx_str}"
                        )

                        # Trade log
                        tl = get_trade_logger()
                        if tl:
                            try:
                                tl.log_entry(
                                    symbol=symbol, direction=direction,
                                    price=entry_price, size=volume,
                                    decision_factors={
                                        "trigger": signal['trigger'],
                                        "confidence": signal['confidence'],
                                        "reason": signal['reason'],
                                        "atr": atr_value,
                                        "adx": adx_value,
                                        "bb_mid": bb_mid,
                                        "timeframe": CONFIG.get('timeframe', 'M15'),
                                        "account": ACCOUNT_NUMBER,
                                    },
                                    market_context={
                                        "entry_price": entry_price,
                                        "sl": sl, "tp": tp,
                                    }
                                )
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

                        # ML data -- executed trade
                        dc = get_ml_dc()
                        if dc:
                            try:
                                row = build_ml_row(
                                    symbol,
                                    f"{signal['direction']}_EXECUTED",
                                    signal['trigger'],
                                    signal['confidence'],
                                    atr_value,
                                    1,
                                    rates_np
                                )
                                row.update({
                                    'entry_price':          entry_price,
                                    'sl_price':             sl,
                                    'tp_price':             tp,
                                    'volume_lots':          volume,
                                    'ticket':               ticket_num,
                                    'account_balance':      bal,
                                    'account_equity':       eqt,
                                    'spread_points':        spread_pts,
                                    'bb_upper':             bb_upper,
                                    'bb_lower':             bb_lower,
                                    'bb_mid':               bb_mid,
                                    'price_position_in_bb': price_pos_in_bb,
                                    'adx':                  adx_value,
                                    'rsi':                  curr_rsi,
                                })
                                dc.log_signal(row)
                            except Exception:
                                pass

                        send_webhook({"event": "trade", "symbol": symbol, "direction": direction.upper(), "trigger": signal['trigger'], "strategy": f"mamba_{signal['trigger'].lower()}", "entry_price": entry_price, "sl": sl, "tp": tp, "volume": volume, "timeframe": CONFIG.get('timeframe', 'M15'), "confidence": signal.get("confidence"), "rsi": curr_rsi, "atr": atr_value, "h1_trend": None, "adx": adx_value})
                    else:
                        logger.warning(f"XX {symbol}: Order rejected")
                        send_webhook({"event": "rejected", "symbol": symbol, "direction": direction.upper(), "trigger": signal['trigger'], "entry_price": entry_price, "rsi": curr_rsi})
            except Exception as e:
                logger.error(f"{symbol} error: {e}")

        iteration += 1
        if iteration % 10 == 0:
            logger.info(f"[HEARTBEAT] MAMBA loop -- iteration {iteration}")
        send_webhook({"event": "heartbeat", "iteration": iteration})
        time.sleep(CONFIG.get('strategy_interval', 30))

# ============================================================================
# TRAILING / BREAKEVEN / FRIDAY FLATTEN
# ============================================================================
def trailing_loop():
    """
    Mamba doesn't trail -- TP is fixed at BB midline.
    This loop handles: breakeven moves + Friday flatten only.
    """
    broker = get_broker()
    trail_iter = 0

    while not bot_stop.is_set():
        try:
            positions = broker.get_positions()
            trail_iter += 1
            if trail_iter % 30 == 0:
                logger.info(f"[HEARTBEAT] Mamba monitor -- {len(positions)} positions")

            now = datetime.now(timezone.utc)

            # Track positions and detect closures for ML outcome logging
            current_tickets = {pos.get('ticket', 0) for pos in positions}

            # Add new positions to tracking
            for pos in positions:
                ticket = pos.get('ticket', 0)
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
                        close_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)
                        duration = (close_time - entry['open_time']).total_seconds() / 60
                        exit_reason = determine_exit_reason(
                            entry['entry_price'], close_price, entry['sl'], entry['tp'], symbol
                        )
                        sym_cfg_out = CONFIG.get('symbols', {}).get(symbol, {})
                        sym_tf_out = TF_MAP.get(sym_cfg_out.get('timeframe', CONFIG.get('timeframe', 'M15')), mt5.TIMEFRAME_M15)
                        rates_out = mt5.copy_rates_from_pos(symbol, sym_tf_out, 0, 50)
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
                        record_trade_result(symbol, profit)
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

            for pos in positions:
                symbol = pos['symbol']
                cfg = CONFIG['symbols'].get(symbol)
                if cfg is None:
                    continue

                # Friday flatten
                if now.weekday() == 4:
                    tf = cfg.get('time_filter', {})
                    flatten_hour = tf.get('friday_flatten_hour', 0)
                    if flatten_hour > 0 and now.hour >= flatten_hour:
                        ticket = pos.get('ticket', 0)
                        logger.info(f"[FRIDAY-FLATTEN] Closing {symbol} #{ticket} before weekend")

                        # Get ATR for ML row
                        sym_cfg_ff = CONFIG.get('symbols', {}).get(symbol, {})
                        sym_tf_ff = TF_MAP.get(sym_cfg_ff.get('timeframe', CONFIG.get('timeframe', 'M15')), mt5.TIMEFRAME_M15)
                        rates_ff = mt5.copy_rates_from_pos(symbol, sym_tf_ff, 0, 50)
                        curr_atr_ff = 0
                        rates_np_ff = None
                        if rates_ff is not None and len(rates_ff) >= 20:
                            rates_np_ff = mt5_rates_to_numpy(rates_ff)
                            curr_atr_ff = safe_last(ATR(rates_np_ff[:, 2], rates_np_ff[:, 3], rates_np_ff[:, 4], 14))

                        # Log ML outcome inline for Friday flatten (before closing)
                        with positions_lock:
                            entry = _tracked_positions.get(ticket)
                        if entry:
                            try:
                                pos_type_ff = normalize_pos_type(pos.get('type', ''))
                                tick_ff = mt5.symbol_info_tick(symbol)
                                if tick_ff:
                                    close_px = tick_ff.bid if pos_type_ff == 'BUY' else tick_ff.ask
                                    duration_ff = (now - entry['open_time']).total_seconds() / 60
                                    unrealized = pos.get('profit', 0.0)
                                    bal, eqt = get_account_balance_equity()
                                    row = build_ml_row(symbol, 'TRADE_CLOSED', 'OUTCOME', 0, curr_atr_ff, 0, rates_np_ff)
                                    row.update({
                                        'entry_price': entry['entry_price'],
                                        'sl_price':    entry['sl'],
                                        'tp_price':    entry['tp'],
                                        'volume_lots': entry['volume'],
                                        'ticket':      ticket,
                                        'exit_price':  close_px,
                                        'exit_reason': 'FRIDAY_FLATTEN',
                                        'profit':      unrealized,
                                        'outcome':     'WIN' if unrealized > 0 else ('LOSS' if unrealized < 0 else 'BREAKEVEN'),
                                        'duration_minutes': round(duration_ff, 1),
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
                                    record_trade_result(symbol, unrealized)
                                    logger.info(f"[OUTCOME] {symbol} #{ticket}: FRIDAY_FLATTEN P&L={unrealized:.2f}")
                            except Exception:
                                pass

                        broker.close_position(ticket)
                        send_webhook({"event": "friday_flatten", "symbol": symbol, "ticket": ticket})
                        continue

                # Breakeven logic
                be_mult = cfg.get('breakeven_atr_mult', 1.0)
                if be_mult <= 0:
                    continue

                sym_cfg_trail = CONFIG.get('symbols', {}).get(symbol, {})
                sym_tf_trail = TF_MAP.get(sym_cfg_trail.get('timeframe', CONFIG.get('timeframe', 'M15')), mt5.TIMEFRAME_M15)
                rates = mt5.copy_rates_from_pos(symbol, sym_tf_trail, 0, 50)
                if rates is None or len(rates) < 20:
                    continue
                rates_np = mt5_rates_to_numpy(rates)
                curr_atr = safe_last(ATR(rates_np[:, 2], rates_np[:, 3], rates_np[:, 4], 14))

                pos_type = normalize_pos_type(pos.get('type', ''))
                ticket = pos.get('ticket', 0)
                if pos_type is None:
                    continue

                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue

                current_sl = float(pos.get('sl') or 0.0)
                entry_price = float(pos.get('price_open', pos.get('price', 0)))
                digits = get_symbol_digits(symbol)
                be_offset = curr_atr * 0.1
                trail_start_mult = cfg.get('trail_start_atr_mult', be_mult)
                trail_dist = curr_atr * cfg.get('atr_trail_multiplier', 1.0)
                curr_rsi = safe_last(RSI(rates_np[:, 4], 14))

                if entry_price > 0:
                    if pos_type == 'BUY':
                        if tick.bid >= entry_price + (curr_atr * be_mult) and current_sl < entry_price:
                            be_sl = entry_price + be_offset
                            broker.modify_position(ticket, sl=round(be_sl, digits))
                            logger.info(f"[BE] {symbol} #{ticket}: SL -> {be_sl:.{digits}f}")
                            send_webhook({"event": "breakeven", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(be_sl, digits), "rsi": curr_rsi})
                            continue
                    elif pos_type == 'SELL':
                        if tick.ask <= entry_price - (curr_atr * be_mult) and (current_sl > entry_price or current_sl == 0):
                            be_sl = entry_price - be_offset
                            broker.modify_position(ticket, sl=round(be_sl, digits))
                            logger.info(f"[BE] {symbol} #{ticket}: SL -> {be_sl:.{digits}f}")
                            send_webhook({"event": "breakeven", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(be_sl, digits), "rsi": curr_rsi})
                            continue

                if entry_price > 0:
                    if pos_type == 'BUY' and tick.bid < entry_price + (curr_atr * trail_start_mult):
                        continue
                    elif pos_type == 'SELL' and tick.ask > entry_price - (curr_atr * trail_start_mult):
                        continue

                if pos_type == 'BUY':
                    new_sl = tick.bid - trail_dist
                    if new_sl > current_sl and current_sl > 0:
                        broker.modify_position(ticket, sl=round(new_sl, digits))
                        logger.info(f"[TRAIL] {symbol} #{ticket}: SL {current_sl:.{digits}f} -> {new_sl:.{digits}f}")
                        send_webhook({"event": "trail_update", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(new_sl, digits), "rsi": curr_rsi})
                elif pos_type == 'SELL':
                    new_sl = tick.ask + trail_dist
                    if (new_sl < current_sl or current_sl == 0) and new_sl > 0:
                        broker.modify_position(ticket, sl=round(new_sl, digits))
                        logger.info(f"[TRAIL] {symbol} #{ticket}: SL {current_sl:.{digits}f} -> {new_sl:.{digits}f}")
                        send_webhook({"event": "trail_update", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(new_sl, digits), "rsi": curr_rsi})
        except Exception as e:
            logger.error(f"[MONITOR] Error: {e}")
        time.sleep(30)

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
    return jsonify({
        'bot': 'mamba-v1', 'profile': PROFILE_NAME, 'account': ACCOUNT_NUMBER,
        'timeframe': CONFIG.get('timeframe', 'M15'),
        'balance': account.get('balance', 0) if account else 0,
        'equity': account.get('equity', 0) if account else 0,
        'positions_count': len(positions),
        'portfolio_risk': portfolio,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

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

    # Get position details before closing for ML logging (use .get, not .pop)
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
                deals = mt5.history_deals_get(datetime.now(timezone.utc) - timedelta(minutes=5), datetime.now(timezone.utc))
                profit = 0.0
                if deals:
                    for deal in deals:
                        if deal.position_id == ticket and deal.entry == 1:
                            profit = deal.profit
                            break
                tf_close = TF_MAP.get(CONFIG.get('timeframe', 'M15'), mt5.TIMEFRAME_M15)
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
    timeframe = TF_MAP.get(CONFIG.get('timeframe', 'M15'), mt5.TIMEFRAME_M15)
    cfg = CONFIG['symbols'].get(symbol, {})
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
    if rates is None or len(rates) < 60:
        return jsonify({'error': 'No data'}), 400
    rates_np = mt5_rates_to_numpy(rates)
    signal, atr_val, adx_val, bb_upper, bb_mid, bb_lower, curr_rsi = check_bb_fade(rates_np, cfg)
    price_now = float(rates_np[-2, 4]) if len(rates_np) >= 2 else 0
    bb_range = bb_upper - bb_lower
    bb_pct = round((price_now - bb_lower) / bb_range * 100, 1) if bb_range > 0 else 50
    bb_entry_pct = cfg.get('bb_entry_pct', 0.0)
    proximity = bb_range * bb_entry_pct
    return jsonify({
        'symbol': symbol, 'strategy': 'mamba_bb_fade',
        'signal': signal['direction'] if signal else 'HOLD',
        'trigger': signal['trigger'] if signal else None,
        'atr': atr_val, 'adx': adx_val,
        'rsi': curr_rsi,
        'bb_upper': round(bb_upper, 5), 'bb_mid': round(bb_mid, 5), 'bb_lower': round(bb_lower, 5),
        'bb_pct': bb_pct,
        'price': price_now,
        'bb_entry_pct': bb_entry_pct,
        'near_upper': price_now >= bb_upper - proximity,
        'near_lower': price_now <= bb_lower + proximity,
        'rsi_overbought': cfg.get('rsi_overbought', 70),
        'rsi_oversold': cfg.get('rsi_oversold', 30),
        'confidence': signal['confidence'] if signal else 0,
        'reason': signal['reason'] if signal else 'No signal',
        'regime': 'RANGING' if adx_val < cfg.get('adx_threshold', 25) else 'TRENDING',
        'timeframe': CONFIG.get('timeframe', 'M15'),
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
    config_path = os.path.join(base_dir, 'mamba_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found")
    return profiles[profile]

def initialize():
    global CONFIG, ACCOUNT_NUMBER, PROFILE_NAME
    parser = argparse.ArgumentParser(description='MAMBA v1 -- Mean-Reversion Range Bot')
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    PROFILE_NAME = args.profile
    CONFIG = load_config(PROFILE_NAME)
    ACCOUNT_NUMBER = CONFIG['account_number']
    init_logger()
    logger.info("=" * 60)
    logger.info(f"MAMBA v1 Starting -- Profile: {PROFILE_NAME}")
    logger.info(f"Account: {ACCOUNT_NUMBER}, Timeframe: {CONFIG.get('timeframe')}")
    logger.info(f"Strategy: Mean-Reversion BB Fade (ADX range filter)")
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
            max_consecutive_losses=rc.get('max_consecutive_losses', 4),
            max_drawdown_percent=rc.get('max_drawdown_pct', 10),
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
        tl = TradeDecisionLogger(f"mamba_v1_{PROFILE_NAME}")
        set_trade_logger(tl)
    except Exception as e:
        logger.warning(f"Trade logger failed (non-fatal): {e}")
    # Initialize ML data collector
    try:
        dc = SharedDataCollector('mamba', ACCOUNT_NUMBER)
        set_ml_dc(dc)
        logger.info(f"ML data collector initialized -- saving to {dc.data_dir}")
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
    Safe to call on every restart -- server-side upsert means no duplicates."""
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
    port = CONFIG.get('server_port', 8056)
    logger.info(f"Flask API on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)

if __name__ == '__main__':
    main()








