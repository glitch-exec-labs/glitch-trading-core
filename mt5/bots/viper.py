"""
VIPER v3 — Aggressive Momentum Scalper (M5)
3 trend-aligned entry triggers — ANY ONE fires = trade.

Trigger 1: N-Bar Breakout (20-bar high/low break + volume + RSI confirmation)
Trigger 2: EMA Pullback (price pulls back to 20 EMA, bounces in trend direction)
Trigger 3: Momentum Continuation (8/20 EMA cross + volume + RSI)

All triggers are trend-following and gated by H1 trend filter.
H1 50-EMA direction determines allowed trade direction on M5.

Exits: Fixed 1.5:1 R:R TP + RSI extreme exit + ATR trailing stop
Timeframe: M5 (288 candles per day)
Risk: 2% per trade, 1.5x ATR stop, max 2 positions per symbol

Changes from v2:
  - Replaced counter-trend Reversal with EMA Pullback (trend-aligned)
  - Replaced FVG with Momentum Continuation / EMA Cross (M5-appropriate)
  - Fixed Breakout: completed bar, ATR buffer, 2-bar window, vol 1.2x
  - Added H1 trend filter (prevents counter-trend scalps)
  - Fixed TP: BB-based -> fixed 1.5:1 R:R (no more wrong-side TP)
  - Spread filter widened: 25% -> 40% ATR (appropriate for M5)
  - Flask API hardened: null-checks, race condition fix
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
BOT_NAME = 'viper'
API_KEY = os.environ.get("VIPER_API_KEY", "")

# Thread-safe state
state_lock = threading.Lock()
last_entry_time = {}
bot_stop = threading.Event()
_strategy_thread = None
_trailing_thread = None

# Loss tracking for adaptive cooldown & per-symbol session limits
_consecutive_losses = {}        # symbol -> consecutive loss count
_consecutive_losses_date = {}   # symbol -> date the counter was last updated
_symbol_session_losses = {}     # symbol -> {date, total_loss, trade_count}
_session_loss_lock = threading.Lock()

def get_daily_reset_boundary(now=None):
    now_utc = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    reset_time = now_utc.replace(hour=20, minute=0, second=0, microsecond=0)
    if now_utc < reset_time:
        reset_time -= timedelta(days=1)
    return reset_time

def get_daily_reset_key(now=None):
    return get_daily_reset_boundary(now).strftime('%Y-%m-%d')

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
    logger = logging.getLogger(f"VIPER3-{ACCOUNT_NUMBER}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"viper3_{ACCOUNT_NUMBER}.log", encoding='utf-8')
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
_tracked_positions = {}  # ticket -> {symbol, entry_price, sl, tp, volume, open_time, type}
positions_lock = threading.Lock()
_pending_close_checks = {}
pending_close_lock = threading.Lock()
def get_ml_dc(): return _ml_dc
def set_ml_dc(dc):
    global _ml_dc
    _ml_dc = dc

# ── Signal quality ML model (loaded once at startup) ──────────────────────────
_signal_quality_model = None  # {'model': XGB, 'feature_cols': [...], 'threshold': float}

def load_signal_quality_model():
    """Load the viper signal quality XGBoost model if available."""
    global _signal_quality_model
    try:
        import pickle
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'viper_signal_quality.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                _signal_quality_model = pickle.load(f)
            # Model trained on GPU — set to CPU for single-row real-time inference
            _signal_quality_model['model'].get_booster().set_param('device', 'cpu')
            threshold = _signal_quality_model.get('threshold', 0.5)
            samples = _signal_quality_model.get('samples', 0)
            auc = _signal_quality_model.get('auc', 0)
            logger.info(f"[ML-FILTER] Loaded viper signal quality model | samples={samples} AUC={auc:.3f} threshold={threshold:.2f}")
        else:
            logger.info("[ML-FILTER] No signal quality model found — all signals pass (train with ml_trainer.py)")
    except Exception as e:
        logger.warning(f"[ML-FILTER] Could not load signal quality model: {e}")

# ============================================================================
# MT5 TIMEFRAME MAP
# ============================================================================
TF_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
}

# ============================================================================
# ML HELPERS
# ============================================================================
H1_TREND_NUM = {"UP": 1, "BULLISH": 1, "BUY": 1, "DOWN": -1, "BEARISH": -1, "SELL": -1, "BOTH": 0}

def score_signal_quality(signal, indicators, atr_value, rates_np, symbol, threshold_override=None) -> tuple:
    """
    Score a signal using the trained signal quality model.
    Returns (quality_score: float, passes_filter: bool).
    If no model loaded, always returns (1.0, True).
    threshold_override: if set, uses this value instead of the model's trained threshold.
    """
    if _signal_quality_model is None:
        return 1.0, True
    try:
        model = _signal_quality_model['model']
        feature_cols = _signal_quality_model['feature_cols']
        threshold = threshold_override if threshold_override is not None else _signal_quality_model.get('threshold', 0.5)

        closes = rates_np[:, 4]
        highs  = rates_np[:, 2]
        lows   = rates_np[:, 3]

        direction = signal.get('direction', 'BUY')
        is_buy = 1 if direction == 'BUY' else 0

        adx_val = indicators.get('adx') or 0
        rsi_val = indicators.get('rsi') or 0
        volume_ratio_val = indicators.get('volume_ratio') or 0
        pullback_dist = indicators.get('pullback_distance_atr') or 0
        h1_val = H1_TREND_NUM.get(indicators.get('h1_trend', 'BOTH'), 0)

        # Spread points (already computed in the calling loop)
        spread_pts = get_spread_points(symbol) if callable(get_spread_points) else 0

        # Bollinger band position from last bar
        bb_upper = float(rates_np[-1, 2]) if len(rates_np) > 0 else 0
        bb_lower = float(rates_np[-1, 3]) if len(rates_np) > 0 else 0
        curr_close = float(rates_np[-1, 4]) if len(rates_np) > 0 else 0
        bb_range = bb_upper - bb_lower
        price_pos_bb = (curr_close - bb_lower) / bb_range if bb_range > 0 else 0.5

        # Build feature map matching training columns
        feat_map = {
            'confidence':            signal.get('confidence', 0),
            'rsi':                   rsi_val,
            'ema_separation':        indicators.get('ema_separation', 0) or 0,
            'adx':                   adx_val,
            'price_position_in_bb':  price_pos_bb,
            'volume_ratio':          volume_ratio_val,
            'sr_distance_atr':       indicators.get('sr_distance_atr', 0) or 0,
            'spread_points':         spread_pts,
            'conditions_met':        indicators.get('conditions_met', 0) or 0,
            'pin_wick_size':         indicators.get('pin_wick_size', 0) or 0,
            'pin_body_ratio':        indicators.get('pin_body_ratio', 0) or 0,
            'engulf_size_atr':       indicators.get('engulf_size_atr', 0) or 0,
            'pullback_distance_atr': pullback_dist,
            'pattern_detected':      int(bool(indicators.get('pattern_detected', False))),
            'sr_nearby':             int(bool(indicators.get('sr_nearby', False))),
            'atr':                   atr_value,
            'h1_trend_num':          h1_val,
            'is_buy':                is_buy,
        }
        row = [feat_map.get(c, 0) for c in feature_cols]
        import numpy as _np
        X = _np.array(row, dtype=float).reshape(1, -1)
        score = float(model.predict_proba(X)[0][1])
        return score, score >= threshold
    except Exception as e:
        logger.warning(f"[ML-FILTER] Scoring error: {e}")
        return 1.0, True


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
        'timeframe':     CONFIG.get('timeframe', 'M5'),
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

def get_signal_rsi(rsi_vals):
    """
    Return RSI value for the current decision candle.
    Supports both scalar RSI (numba-optimized path) and RSI series.
    """
    try:
        if isinstance(rsi_vals, np.ndarray):
            if len(rsi_vals) >= 1:
                return float(rsi_vals[-1])
            return None
        if rsi_vals is None:
            return None
        return float(rsi_vals)
    except Exception:
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
    """Log trade outcome to ML data collector and risk manager."""
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

        logger.info(f"[OUTCOME] {symbol} #{ticket}: {outcome} {profit:.2f} via {exit_reason}")
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
        payload.update({"bot": "viper3", "account": ACCOUNT_NUMBER,
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


def calculate_adx(highs, lows, closes, period=14):
    """Calculate Average Directional Index — chop vs trend filter.
    ADX < 20 = ranging/choppy (avoid), ADX > 25 = trending (trade)."""
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
    plus_s = float(np.sum(plus_dm[1:period + 1]))
    minus_s = float(np.sum(minus_dm[1:period + 1]))

    dx_vals = []
    for i in range(period + 1, n):
        atr_s = atr_s - (atr_s / period) + tr[i]
        plus_s = plus_s - (plus_s / period) + plus_dm[i]
        minus_s = minus_s - (minus_s / period) + minus_dm[i]

        if atr_s > 0:
            pdi = 100.0 * plus_s / atr_s
            mdi = 100.0 * minus_s / atr_s
        else:
            pdi = mdi = 0.0

        di_sum = pdi + mdi
        dx = (100.0 * abs(pdi - mdi) / di_sum) if di_sum > 0 else 0.0
        dx_vals.append(dx)

    if len(dx_vals) < period:
        return None

    adx = float(np.mean(dx_vals[:period]))
    for i in range(period, len(dx_vals)):
        adx = (adx * (period - 1) + dx_vals[i]) / period

    return round(adx, 2)


def record_trade_result(symbol, profit):
    """Track consecutive losses and daily symbol P&L for adaptive cooldown."""
    today = get_daily_reset_key()
    with _session_loss_lock:
        if profit < 0:
            _consecutive_losses[symbol] = _consecutive_losses.get(symbol, 0) + 1
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
                f"session P&L={state['total_loss']:.2f} ({state['trade_count']} trades)"
            )


def get_adaptive_cooldown(symbol, base_cooldown, cfg):
    """Exponential backoff: after each consecutive loss, double the cooldown."""
    with _session_loss_lock:
        losses = _consecutive_losses.get(symbol, 0)
    if losses == 0:
        return base_cooldown
    multiplier = cfg.get('loss_cooldown_multiplier', 2.0)
    max_cd = cfg.get('max_loss_cooldown', 3600)
    adapted = base_cooldown * (multiplier ** min(losses, 5))
    return min(adapted, max_cd)


def is_symbol_session_limit_hit(symbol, cfg):
    """Check if per-symbol daily loss limit (% of balance) or consecutive loss limit reached."""
    today = get_daily_reset_key()
    max_consec = cfg.get('max_symbol_consecutive_losses', 3)

    # Percentage-based: compute dollar limit from current balance
    sym_loss_pct = cfg.get('max_symbol_daily_loss_pct', 5)
    broker = get_broker()
    account = broker.get_account_info() if broker else None
    balance = account.get('balance', 150) if account else 150
    max_loss = -(balance * sym_loss_pct / 100)  # e.g. $150 * 5% = -$7.50

    with _session_loss_lock:
        state = _symbol_session_losses.get(symbol, {})
        if state.get('date') == today and state.get('total_loss', 0.0) <= max_loss:
            return True, f"daily loss {state['total_loss']:.2f} hit {sym_loss_pct}% limit (${max_loss:.2f})"
        if _consecutive_losses_date.get(symbol) != today:
            _consecutive_losses[symbol] = 0
        consec = _consecutive_losses.get(symbol, 0)
        if consec >= max_consec:
            return True, f"{consec} consecutive losses (limit {max_consec})"
    return False, ""


# ============================================================================
# H1 TREND FILTER
# ============================================================================
def get_h1_trend(symbol, cfg):
    """
    Fetch H1 data and determine trend direction using 50-period EMA.
    Returns: 'BUY', 'SELL', or 'BOTH' (transition zone).
    """
    h1_ema_period = cfg.get('h1_ema_period', 50)
    try:
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, h1_ema_period + 20)
        if h1_rates is None or len(h1_rates) < h1_ema_period + 5:
            return 'BOTH'
        h1_closes = np.array([r[4] for r in h1_rates], dtype=np.float64)
        h1_highs = np.array([r[2] for r in h1_rates], dtype=np.float64)
        h1_lows = np.array([r[3] for r in h1_rates], dtype=np.float64)
        h1_ema = EMA(h1_closes, h1_ema_period)
        h1_atr = safe_last(ATR(h1_highs, h1_lows, h1_closes, 14))
        h1_close = float(h1_closes[-1])
        h1_ema_val = float(h1_ema[-1])
        transition = h1_atr * cfg.get('h1_transition_mult', 0.5)
        if h1_close > h1_ema_val + transition:
            return 'BUY'
        elif h1_close < h1_ema_val - transition:
            return 'SELL'
        else:
            return 'BOTH'
    except Exception:
        return 'BOTH'

# ============================================================================
# TRIGGER 1: N-BAR BREAKOUT (FIXED)
# ============================================================================
def check_breakout(closes, highs, lows, volumes, ema_vals, rsi_vals, curr_atr, cfg):
    """
    Price breaks above N-bar high (buy) or below N-bar low (sell).
    Uses COMPLETED bar (-2), 0.2 ATR buffer, 2-bar break window, vol 1.2x.
    RSI must be > 50 for buys, < 50 for sells.

    Returns: (signal_dict or None, indicators_dict)
    """
    lookback = cfg.get('breakout_lookback', 20)
    vol_mult = cfg.get('breakout_vol_mult', 1.2)
    buf_mult = cfg.get('breakout_buffer_mult', 0.2)

    if len(closes) < lookback + 5:
        return None, {}
    curr_rsi = get_signal_rsi(rsi_vals)
    if curr_rsi is None:
        return None, {}

    c1 = float(closes[-2])
    c2 = float(closes[-3])
    curr_ema = float(ema_vals[-2])
    buf = curr_atr * buf_mult

    recent_highs = highs[-(lookback+3):-3]
    recent_lows = lows[-(lookback+3):-3]
    n_bar_high = float(np.max(recent_highs))
    n_bar_low = float(np.min(recent_lows))

    avg_vol = float(np.mean(volumes[-50:])) if len(volumes) >= 50 else float(np.mean(volumes[:-1])) if len(volumes) > 1 else float(volumes[-1])
    vol_1 = float(volumes[-2])
    vol_2 = float(volumes[-3])
    vol_ok = max(vol_1, vol_2) > avg_vol * vol_mult
    vol_ratio = max(vol_1, vol_2) / avg_vol if avg_vol > 0 else 1.0

    indicators = {
        'ema_20': curr_ema,
        'rsi': curr_rsi,
        'breakout_high': n_bar_high,
        'breakout_low': n_bar_low,
        'volume_ratio': vol_ratio,
        'pullback_distance_atr': None,
        'momentum_fast_ema': None,
        'momentum_slow_ema': None,
    }

    bullish_break = (
        c1 > (n_bar_high + buf) and
        c1 > curr_ema and
        vol_ok and
        curr_rsi > 50
    )

    bearish_break = (
        c1 < (n_bar_low - buf) and
        c1 < curr_ema and
        vol_ok and
        curr_rsi < 50
    )

    if bullish_break:
        return {
            'trigger': 'BREAKOUT',
            'direction': 'BUY',
            'confidence': 0.85,
            'reason': f'Broke {lookback}-bar high {n_bar_high:.5f} (buf {buf:.5f}), vol {max(vol_1,vol_2)/avg_vol:.1f}x, RSI {curr_rsi:.0f}'
        }, indicators
    elif bearish_break:
        return {
            'trigger': 'BREAKOUT',
            'direction': 'SELL',
            'confidence': 0.85,
            'reason': f'Broke {lookback}-bar low {n_bar_low:.5f} (buf {buf:.5f}), vol {max(vol_1,vol_2)/avg_vol:.1f}x, RSI {curr_rsi:.0f}'
        }, indicators
    return None, indicators

# ============================================================================
# TRIGGER 2: EMA PULLBACK (replaces Reversal)
# ============================================================================
def check_ema_pullback(opens, closes, highs, lows, ema_vals, rsi_vals, curr_atr, cfg):
    """
    Price pulls back to 20 EMA in trend direction, then bounces.
    Returns: (signal_dict or None, indicators_dict)
    """
    pullback_dist = cfg.get('pullback_atr_mult', 0.7)
    rsi_buy_min = cfg.get('pullback_rsi_buy_min', 45)
    rsi_sell_max = cfg.get('pullback_rsi_sell_max', 55)

    if len(closes) < 10:
        return None, {}
    rsi1 = get_signal_rsi(rsi_vals)
    if rsi1 is None:
        return None, {}

    c1 = float(closes[-1])
    c2 = float(closes[-2])
    c3 = float(closes[-3])
    low1 = float(lows[-1])
    high1 = float(highs[-1])
    ema1 = float(ema_vals[-1])
    ema3 = float(ema_vals[-3])
    max_dist = curr_atr * pullback_dist
    pullback_dist_atr = min(abs(low1 - ema1), abs(high1 - ema1)) / curr_atr if curr_atr > 0 else 0.0

    indicators = {
        'ema_20': ema1,
        'rsi': rsi1,
        'pullback_distance_atr': pullback_dist_atr,
        'breakout_high': None,
        'breakout_low': None,
        'volume_ratio': None,
        'momentum_fast_ema': None,
        'momentum_slow_ema': None,
    }

    body = abs(float(closes[-1]) - float(opens[-1]))
    candle_range = float(highs[-1]) - float(lows[-1])
    if candle_range > 0:
        body_ratio = body / candle_range
        indicators['body_ratio'] = body_ratio
        if body_ratio < 0.4:
            return None, indicators

    bullish_pullback = (
        c3 > ema3 and
        low1 <= ema1 + max_dist and
        low1 >= ema1 - max_dist and
        c1 > ema1 and
        c1 > c2 and
        rsi1 > rsi_buy_min
    )

    bearish_pullback = (
        c3 < ema3 and
        high1 >= ema1 - max_dist and
        high1 <= ema1 + max_dist and
        c1 < ema1 and
        c1 < c2 and
        rsi1 < rsi_sell_max
    )

    if bullish_pullback:
        return {
            'trigger': 'EMA_PULLBACK',
            'direction': 'BUY',
            'confidence': 0.80,
            'reason': f'Bullish EMA pullback: low {low1:.5f} near EMA {ema1:.5f}, bounce close {c1:.5f}, RSI {rsi1:.0f}'
        }, indicators
    elif bearish_pullback:
        return {
            'trigger': 'EMA_PULLBACK',
            'direction': 'SELL',
            'confidence': 0.80,
            'reason': f'Bearish EMA pullback: high {high1:.5f} near EMA {ema1:.5f}, rejection close {c1:.5f}, RSI {rsi1:.0f}'
        }, indicators
    return None, indicators

# ============================================================================
# TRIGGER 2: MOMENTUM CONTINUATION / EMA CROSS
# ============================================================================
def check_momentum(opens, closes, highs, lows, volumes, rsi_vals, curr_atr, cfg):
    """
    Fast EMA crosses above/below slow EMA on the current forming candle.
    Returns: (signal_dict or None, indicators_dict)
    """
    fast_period = cfg.get('ema_fast', cfg.get('momentum_fast_ema', 3))
    slow_period = cfg.get('ema_slow', cfg.get('momentum_slow_ema', 8))
    vol_mult = cfg.get('momentum_vol_mult', 1.2)

    if len(closes) < slow_period + 10:
        return None, {}
    rsi1 = get_signal_rsi(rsi_vals)
    if rsi1 is None:
        return None, {}

    ema_fast = EMA_GPU(closes, fast_period)
    ema_slow = EMA_GPU(closes, slow_period)

    fast_now = float(ema_fast[-1])
    slow_now = float(ema_slow[-1])
    c1 = float(closes[-1])
    bullish_cross = bearish_cross = False
    for offset in [1, 2, 3]:
        if offset + 1 >= len(ema_fast):
            continue
        f_curr = float(ema_fast[-offset])
        s_curr = float(ema_slow[-offset])
        f_prev = float(ema_fast[-(offset + 1)])
        s_prev = float(ema_slow[-(offset + 1)])
        if f_prev <= s_prev and f_curr > s_curr:
            bullish_cross = True
            break
        if f_prev >= s_prev and f_curr < s_curr:
            bearish_cross = True
            break

    avg_vol = float(np.mean(volumes[-51:-1])) if len(volumes) >= 51 else float(np.mean(volumes[:-1])) if len(volumes) > 1 else float(volumes[-1])
    recent_vol = float(volumes[-1])
    vol_ok = recent_vol > avg_vol * vol_mult
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

    indicators = {
        'ema_20': None,
        'momentum_fast_ema': fast_now,
        'momentum_slow_ema': slow_now,
        'rsi': rsi1,
        'volume_ratio': vol_ratio,
        'breakout_high': None,
        'breakout_low': None,
        'pullback_distance_atr': None,
    }

    body = abs(float(closes[-1]) - float(opens[-1]))
    candle_range = float(highs[-1]) - float(lows[-1])
    if candle_range > 0:
        body_ratio = body / candle_range
        indicators['body_ratio'] = body_ratio
        if body_ratio < 0.4:
            return None, indicators

    if bullish_cross and fast_now > slow_now and c1 > fast_now and c1 > slow_now and vol_ok and rsi1 > 50:
        return {
            'trigger': 'MOMENTUM',
            'direction': 'BUY',
            'confidence': 0.75,
            'reason': f'Bullish EMA cross ({fast_period}/{slow_period}), vol {vol_ratio:.1f}x, RSI {rsi1:.0f}'
        }, indicators
    elif bearish_cross and fast_now < slow_now and c1 < fast_now and c1 < slow_now and vol_ok and rsi1 < 50:
        return {
            'trigger': 'MOMENTUM',
            'direction': 'SELL',
            'confidence': 0.75,
            'reason': f'Bearish EMA cross ({fast_period}/{slow_period}), vol {vol_ratio:.1f}x, RSI {rsi1:.0f}'
        }, indicators
    return None, indicators

# ============================================================================
# MASTER SIGNAL: King Cobra momentum + EMA pullback
# ============================================================================
def check_all_signals(rates_np, cfg, h1_trend='BOTH'):
    """
    Runs the King Cobra trend-following signal stack.
    Priority: Momentum (0.75) > EMA Pullback (0.80)

    Returns: (signal_dict or None, atr_value, curr_rsi, trigger_name, indicators_dict)
    """
    opens = rates_np[:, 1]
    closes = rates_np[:, 4]
    highs = rates_np[:, 2]
    lows = rates_np[:, 3]
    volumes = rates_np[:, 5]

    rsi_period = cfg.get('rsi_period', 14)
    atr_period = cfg.get('atr_period', 14)
    slow_period = max(
        cfg.get('ema_slow', cfg.get('momentum_slow_ema', 8)),
        20,
    )

    min_bars = max(slow_period, rsi_period, atr_period, 30) + 10
    if len(closes) < min_bars:
        return None, 0.0, None, None, {}

    ema_pullback_vals = EMA_GPU(closes, 20)
    rsi_vals = RSI_GPU(closes, rsi_period)
    curr_atr = safe_last(ATR_GPU(highs, lows, closes, atr_period))
    curr_rsi = get_signal_rsi(rsi_vals)

    if curr_atr <= 0 or curr_rsi is None:
        return None, 0.0, curr_rsi, None, {}

    # ADX chop filter — avoid trading in directionless/ranging markets
    adx_period = cfg.get('adx_period', 14)
    min_adx = cfg.get('min_adx', 20)
    curr_adx = calculate_adx(highs, lows, closes, adx_period)

    # Base indicators dict for HOLD case
    base_indicators = {
        'ema_20': float(ema_pullback_vals[-1]) if len(ema_pullback_vals) >= 1 else None,
        'rsi': curr_rsi,
        'adx': curr_adx,
        'h1_trend': h1_trend,
        'breakout_high': None,
        'breakout_low': None,
        'pullback_distance_atr': None,
        'volume_ratio': None,
        'momentum_fast_ema': None,
        'momentum_slow_ema': None,
    }

    # Block all triggers when market is choppy
    if curr_adx is not None and curr_adx < min_adx:
        base_indicators['hold_reason'] = f'adx_too_low({curr_adx:.1f}<{min_adx})'
        return None, curr_atr, curr_rsi, None, base_indicators

    # RSI dead zone filter — 50-55 has 19.3% win rate (ML-derived, 2433 samples)
    # RSI in this band means no momentum in either direction; skip all entries
    rsi_dead_low  = cfg.get('rsi_dead_zone_low', 50)
    rsi_dead_high = cfg.get('rsi_dead_zone_high', 55)
    if curr_rsi is not None and rsi_dead_low < curr_rsi < rsi_dead_high:
        base_indicators['hold_reason'] = f'rsi_dead_zone({curr_rsi:.1f})'
        return None, curr_atr, curr_rsi, None, base_indicators

    def direction_allowed(signal):
        if signal is None:
            return False
        if h1_trend == 'BOTH':
            return True
        return signal['direction'] == h1_trend

    # 1. MOMENTUM (King Cobra priority)
    signal, indicators = check_momentum(opens, closes, highs, lows, volumes, rsi_vals, curr_atr, cfg)
    if direction_allowed(signal):
        indicators['h1_trend'] = h1_trend
        indicators['adx'] = curr_adx
        return signal, curr_atr, curr_rsi, signal['trigger'], indicators

    # 2. EMA PULLBACK
    signal, indicators = check_ema_pullback(opens, closes, highs, lows, ema_pullback_vals, rsi_vals, curr_atr, cfg)
    if direction_allowed(signal):
        indicators['h1_trend'] = h1_trend
        indicators['adx'] = curr_adx
        return signal, curr_atr, curr_rsi, signal['trigger'], indicators

    return None, curr_atr, curr_rsi, None, base_indicators


def build_viper_hold_reason(rates_np, cfg, h1_trend, curr_atr):
    """Explain why the King Cobra-style signal stack stayed in HOLD."""
    try:
        opens = rates_np[:, 1]
        closes = rates_np[:, 4]
        highs = rates_np[:, 2]
        lows = rates_np[:, 3]
        volumes = rates_np[:, 5]

        ema_vals = EMA_GPU(closes, 20)
        rsi_vals = RSI_GPU(closes, cfg.get('rsi_period', 14))

        p_sig, p_ind = check_ema_pullback(opens, closes, highs, lows, ema_vals, rsi_vals, curr_atr, cfg)
        m_sig, m_ind = check_momentum(opens, closes, highs, lows, volumes, rsi_vals, curr_atr, cfg)

        if h1_trend != 'BOTH':
            if p_sig and p_sig.get('direction') != h1_trend:
                return 'pullback_h1_trend_filtered'
            if m_sig and m_sig.get('direction') != h1_trend:
                return 'momentum_h1_trend_filtered'

        pdist = p_ind.get('pullback_distance_atr')
        if pdist is None:
            pullback_reason = 'pullback_not_ready'
        elif float(pdist) > cfg.get('pullback_atr_mult', 0.5):
            pullback_reason = 'pullback_not_near_ema'
        elif (p_ind.get('body_ratio') or 0) < 0.4:
            pullback_reason = 'pullback_weak_candle'
        elif p_ind.get('rsi') is not None and not (
            float(p_ind.get('rsi')) > cfg.get('pullback_rsi_buy_min', 45)
            or float(p_ind.get('rsi')) < cfg.get('pullback_rsi_sell_max', 55)
        ):
            pullback_reason = 'pullback_rsi_not_in_range'
        else:
            pullback_reason = 'pullback_structure_not_confirmed'

        vratio = m_ind.get('volume_ratio')
        if vratio is None:
            momentum_reason = 'momentum_not_ready'
        elif float(vratio) < cfg.get('momentum_vol_mult', 1.2):
            momentum_reason = 'momentum_volume_low'
        elif (m_ind.get('body_ratio') or 0) < 0.4:
            momentum_reason = 'momentum_weak_candle'
        else:
            fast = m_ind.get('momentum_fast_ema')
            slow = m_ind.get('momentum_slow_ema')
            if fast is not None and slow is not None and curr_atr > 0 and abs(float(fast) - float(slow)) < curr_atr * 0.05:
                momentum_reason = 'momentum_no_cross'
            else:
                momentum_reason = 'momentum_rsi_or_price_filter'

        return f"{pullback_reason};{momentum_reason}"
    except Exception:
        return 'no_trigger'
# ============================================================================
# TRADING HOURS
# ============================================================================
def is_within_trading_hours(cfg):
    tf = cfg.get('time_filter', {})
    if not tf.get('enabled', False):
        return True
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    friday_cutoff = tf.get('friday_cutoff_hour', 0)
    if friday_cutoff > 0 and now.weekday() == 4 and now.hour >= friday_cutoff:
        return False
    return tf.get('start_hour', 0) <= now.hour < tf.get('end_hour', 24)

# ============================================================================
# POSITION SIZING & SL/TP
# ============================================================================
def calculate_sl(direction, entry_price, atr_value, cfg):
    mult = cfg.get('atr_sl_multiplier', 1.5)
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

def calculate_tp(direction, entry_price, sl_price, cfg):
    """Return 0.0 when fixed TP is disabled so trailing stop is the only profit exit."""
    # fixed_tp_points mode: exact point-distance ceiling (overrides R:R)
    fixed_tp_pts = cfg.get('fixed_tp_points', 0) or 0
    if fixed_tp_pts > 0:
        try:
            broker = get_broker()
            sym_info = broker.get_symbol_info(cfg.get('_symbol_name')) if broker and cfg.get('_symbol_name') else None
            point = float(sym_info.get('point', 0)) if sym_info else 0.0
            if point > 0:
                dist = point * fixed_tp_pts
                return (entry_price + dist) if direction == 'buy' else (entry_price - dist)
        except Exception:
            pass
    if not cfg.get('use_fixed_tp', True):
        return 0.0
    rr = cfg.get('min_reward_risk', 1.5)
    sl_distance = abs(entry_price - sl_price)
    if direction == 'buy':
        return entry_price + (sl_distance * rr)
    else:
        return entry_price - (sl_distance * rr)
def calculate_position_size(symbol, cfg, sl_price, entry_price):
    broker = get_broker()
    account = broker.get_account_info()
    if account is None:
        return cfg.get('min_lot', 0.01)

    balance = account.get('balance', 1000)
    risk_pct = cfg.get('risk_percent', 2.0)
    risk_amount = balance * (risk_pct / 100)

    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        return cfg.get('min_lot', 0.01)

    try:
        sym_info = broker.get_symbol_info(symbol)
        if sym_info is None:
            return cfg.get('min_lot', 0.01)
        tick_value = sym_info.get('tick_value', 1)
        tick_size = sym_info.get('tick_size', 0.00001)
        if tick_size == 0 or tick_value == 0:
            return cfg.get('min_lot', 0.01)
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
    default_tf_str = CONFIG.get('timeframe', 'M5')
    timeframe = TF_MAP.get(default_tf_str, mt5.TIMEFRAME_M5)
    cooldown = CONFIG.get('entry_cooldown_seconds', 300)  # 5 min on M5
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
            time.sleep(60)
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
                            h1_trend = get_h1_trend(symbol, cfg)
                            signal, atr_value, curr_rsi, trigger, indicators = check_all_signals(rates_np, cfg, h1_trend)
                            adx_val = indicators.get('adx')
                            rsi_str = f"RSI={curr_rsi:.1f}" if curr_rsi else "RSI=?"
                            h1_str = f"H1={h1_trend}"
                            adx_str = f"ADX={adx_val:.1f}" if adx_val else "ADX=?"
                            if signal:
                                ml_str = ""
                                if _signal_quality_model is not None:
                                    ml_score, ml_passes = score_signal_quality(signal, indicators, atr_value, rates_np, symbol)
                                    ml_str = f" ML={ml_score:.3f} {'PASS' if ml_passes else 'BLOCK'}"
                                logger.info(f"[{symbol}] Outside hours decision: {signal['direction']} via {trigger} - {signal['reason']} ATR={atr_value:.5f} {rsi_str} {h1_str} {adx_str}{ml_str}")
                            else:
                                hold_reason = build_viper_hold_reason(rates_np, cfg, h1_trend, atr_value)
                                logger.info(f"[{symbol}] Outside hours HOLD: {hold_reason} ATR={atr_value:.5f} {rsi_str} {h1_str} {adx_str}")
                            dc = get_ml_dc()
                            if dc:
                                try:
                                    row = build_ml_row(symbol, 'OUTSIDE_HOURS', 'NONE', 0, atr_value, 0, rates_np)
                                    row['rsi'] = curr_rsi if curr_rsi else 0
                                    row['adx'] = adx_val
                                    row['h1_trend'] = h1_trend
                                    row['hold_reason'] = None if signal else hold_reason
                                    row['spread_points'] = get_spread_points(symbol)
                                    dc.log_signal(row)
                                except Exception:
                                    pass
                    continue

                # 3. Position limit + price gap check
                all_positions = broker.get_positions()
                pg = get_portfolio_guard()
                if pg:
                    try:
                        pg.sync_account_positions(all_positions)
                    except Exception as e:
                        if iteration % 60 == 0:
                            logger.warning(f"[PORTFOLIO-RISK] Sync failed: {e}")
                sym_positions = [p for p in all_positions if p['symbol'] == symbol]
                max_pos = cfg.get('max_positions', 2)

                # 5. H1 trend filter — always needed for signal computation
                h1_trend = get_h1_trend(symbol, cfg)

                # 6. Market data — always fetch regardless of position count
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

                # 7. Compute signal — always, even when position is full
                signal, atr_value, curr_rsi, trigger, indicators = check_all_signals(rates_np, cfg, h1_trend)

                if atr_value is None or math.isnan(atr_value) or atr_value <= 0:
                    logger.warning(f"[{BOT_NAME}] ATR invalid ({atr_value}) for {symbol} -- skipping bar")
                    continue

                # 8. Log — always, gives full visibility while in a position
                rsi_str = f"RSI={curr_rsi:.1f}" if curr_rsi else "RSI=?"
                h1_str = f"H1={h1_trend}"
                adx_val = indicators.get('adx')
                adx_str = f"ADX={adx_val:.1f}" if adx_val else "ADX=?"
                if signal:
                    pos_tag = f" [MONITORING {len(sym_positions)}/{max_pos}]" if len(sym_positions) >= max_pos else ""
                    logger.info(f">> {symbol}: {signal['direction']} via {trigger} -- {signal['reason']} ATR={atr_value:.5f} {rsi_str} {h1_str} {adx_str}{pos_tag}")
                elif iteration % 10 == 0:
                    hold_reason = indicators.get('hold_reason') or 'no_trigger'
                    logger.info(f"{symbol}: HOLD ({hold_reason}) ATR={atr_value:.5f} {rsi_str} {h1_str} {adx_str}")

                # ML data — every check including HOLD and in-position bars
                dc = get_ml_dc()
                if dc:
                    try:
                        row = build_ml_row(
                            symbol,
                            signal['direction'] if signal else 'HOLD',
                            trigger if trigger else 'NONE',
                            signal['confidence'] if signal else 0,
                            atr_value,
                            0,
                            rates_np
                        )
                        viper_hold_reason = None
                        if not signal:
                            viper_hold_reason = build_viper_hold_reason(rates_np, cfg, h1_trend, atr_value)
                        row.update({
                            'ema_20':                indicators.get('ema_20'),
                            'rsi':                   indicators.get('rsi', curr_rsi),
                            'adx':                   indicators.get('adx'),
                            'h1_trend':              h1_trend,
                            'breakout_high':         indicators.get('breakout_high'),
                            'breakout_low':          indicators.get('breakout_low'),
                            'pullback_distance_atr': indicators.get('pullback_distance_atr'),
                            'volume_ratio':          indicators.get('volume_ratio'),
                            'momentum_fast_ema':     indicators.get('momentum_fast_ema'),
                            'momentum_slow_ema':     indicators.get('momentum_slow_ema'),
                            'hold_reason':           viper_hold_reason,
                            'spread_points':        get_spread_points(symbol),
                        })
                        dc.log_signal(row)
                    except Exception:
                        pass

                # --- ENTRY GATES: signal computed and logged, now check if we can act ---

                # 2. Adaptive cooldown (doubles after each consecutive loss)
                # Per-symbol cooldown override (falls back to global)
                sym_cooldown = cfg.get('entry_cooldown_seconds', cooldown)
                adaptive_cd = get_adaptive_cooldown(symbol, sym_cooldown, cfg)
                with state_lock:
                    if time.time() - last_entry_time.get(symbol, 0) < adaptive_cd:
                        continue

                # No signal — nothing to enter
                if not signal:
                    continue

                # Max positions reached — signal is noted/logged above but no new entry
                if len(sym_positions) >= max_pos:
                    continue

                # 3b. Enforce minimum price gap to existing position
                if len(sym_positions) > 0:
                    min_gap_pct = cfg.get('min_position_gap_pct', 0.5)
                    tick_check = mt5.symbol_info_tick(symbol)
                    if tick_check:
                        curr_price = (tick_check.ask + tick_check.bid) / 2
                        too_close = False
                        closest_gap_pct = 999.0
                        for existing_pos in sym_positions:
                            existing_entry = existing_pos.get('price_open', existing_pos.get('price', 0))
                            if existing_entry > 0:
                                gap_pct = abs(curr_price - existing_entry) / existing_entry * 100
                                closest_gap_pct = min(closest_gap_pct, gap_pct)
                                if gap_pct < min_gap_pct:
                                    too_close = True
                                    break
                        if too_close:
                            if iteration % 10 == 0:
                                logger.info(f"{symbol}: Price too close to existing position ({closest_gap_pct:.2f}% < {min_gap_pct}%), skipping")
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

                # 4b. Per-symbol session loss limit
                sym_blocked, sym_reason = is_symbol_session_limit_hit(symbol, cfg)
                if sym_blocked:
                    if iteration % 60 == 0:
                        logger.warning(f"[SYMBOL-LIMIT] {symbol}: {sym_reason} -- paused for session")
                    continue

                # 8c. Existing position not too deeply underwater before adding
                if len(sym_positions) > 0:
                    if trigger == 'EMA_PULLBACK':
                        max_underwater_atr = cfg.get('pullback_add_max_underwater_atr', 1.0)
                    else:
                        max_underwater_atr = cfg.get('add_position_max_underwater_atr', 0.5)
                    tick_check2 = mt5.symbol_info_tick(symbol)
                    if tick_check2:
                        blocked = False
                        for ep in sym_positions:
                            ep_type = normalize_pos_type(ep.get('type', ''))
                            ep_entry = float(ep.get('price_open', ep.get('price', 0)))
                            if ep_type == 'BUY':
                                unrealized = tick_check2.bid - ep_entry
                            elif ep_type == 'SELL':
                                unrealized = ep_entry - tick_check2.ask
                            else:
                                continue
                            if unrealized < -(atr_value * max_underwater_atr):
                                logger.info(
                                    f"{symbol}: Existing position too underwater "
                                    f"({unrealized:.5f} < -{atr_value * max_underwater_atr:.5f}), "
                                    f"skipping 2nd entry via {trigger}"
                                )
                                blocked = True
                                break
                        if blocked:
                            continue

                # 8d. ML signal quality filter
                # config: ml_filter.enabled (default True) and ml_filter.threshold_override (optional)
                # Set threshold_override to a lower value (e.g. 0.10) when model needs retraining
                # Set enabled: false to bypass the filter entirely while retraining
                ml_cfg = CONFIG.get('ml_filter', {})
                ml_enabled = ml_cfg.get('enabled', True)
                if signal and _signal_quality_model is not None and ml_enabled:
                    threshold_override = ml_cfg.get('threshold_override', None)
                    ml_score, ml_passes = score_signal_quality(
                        signal, indicators, atr_value, rates_np, symbol,
                        threshold_override=threshold_override
                    )
                    eff_threshold = threshold_override if threshold_override is not None else _signal_quality_model.get('threshold', 0.5)
                    logger.info(
                        f"[ML-FILTER] {symbol} {signal['direction']} via {trigger} — "
                        f"quality={ml_score:.3f} (threshold={eff_threshold:.2f}) "
                        f"{'PASS' if ml_passes else 'BLOCK'}"
                    )
                    if not ml_passes:
                        signal = None  # suppress trade
                elif signal and _signal_quality_model is not None and not ml_enabled:
                    logger.debug(f"[ML-FILTER] disabled in config — {symbol} signal passes without scoring")

                # 9. Execute
                if signal:
                    if should_skip_trade(symbol):
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
                    if tick is None:
                        continue

                    spread = tick.ask - tick.bid
                    max_spread_mult = cfg.get('max_spread_atr_mult', 0.30)
                    if atr_value > 0 and spread > atr_value * max_spread_mult:
                        logger.info(
                            f"{symbol}: Spread {spread:.5f} too wide vs ATR {atr_value:.5f} "
                            f"(limit={max_spread_mult:.2f}x ATR)"
                        )
                        continue

                    # Tighten low-liquidity entries for scalping quality.
                    min_entry_vol_ratio = cfg.get('entry_min_volume_ratio', 1.15)
                    if min_entry_vol_ratio > 0:
                        avg_vol = float(np.mean(rates_np[-51:-1, 5])) if len(rates_np) >= 51 else float(np.mean(rates_np[:-1, 5])) if len(rates_np) > 1 else float(rates_np[-1, 5])
                        recent_vol = float(rates_np[-1, 5])
                        vol_ratio_now = (recent_vol / avg_vol) if avg_vol > 0 else 0.0
                        if vol_ratio_now < min_entry_vol_ratio:
                            logger.info(
                                f"{symbol}: Volume ratio {vol_ratio_now:.2f} below entry floor "
                                f"{min_entry_vol_ratio:.2f} via {trigger}"
                            )
                            continue

                    entry_price = tick.ask if direction == 'buy' else tick.bid
                    cfg_runtime = dict(cfg)
                    cfg_runtime['_symbol_name'] = symbol
                    sl = calculate_sl(direction, entry_price, atr_value, cfg_runtime)
                    tp = calculate_tp(direction, entry_price, sl, cfg_runtime)

                    volume = calculate_position_size(symbol, cfg, sl, entry_price)

                    # Pre-flight SL check — must be on correct side of entry
                    if direction == "buy" and sl >= entry_price:
                        logger.error(f"[{BOT_NAME}] SL {sl} >= entry {entry_price} on BUY — aborting trade")
                        continue
                    if direction == "sell" and sl <= entry_price:
                        logger.error(f"[{BOT_NAME}] SL {sl} <= entry {entry_price} on SELL — aborting trade")
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
                            f"[OK] {direction.upper()} {symbol} via {trigger} @ {entry_price:.5f} "
                            f"vol={volume} SL={sl:.5f} TP={tp:.5f} conf={signal['confidence']} {h1_str}"
                        )

                        # Trade log
                        tl = get_trade_logger()
                        if tl:
                            try:
                                tl.log_entry(
                                    symbol=symbol, direction=direction,
                                    price=entry_price, size=volume,
                                    decision_factors={
                                        "trigger": trigger,
                                        "signal": signal['direction'],
                                        "confidence": signal['confidence'],
                                        "reason": signal['reason'],
                                        "rsi": curr_rsi,
                                        "atr": atr_value,
                                        "h1_trend": h1_trend,
                                        "timeframe": CONFIG.get('timeframe', 'M5'),
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

                        # ML data — executed trade
                        dc = get_ml_dc()
                        if dc:
                            try:
                                row = build_ml_row(
                                    symbol,
                                    f"{signal['direction']}_EXECUTED",
                                    trigger,
                                    signal['confidence'],
                                    atr_value,
                                    1,
                                    rates_np
                                )
                                row.update({
                                    'entry_price':           entry_price,
                                    'sl_price':              sl,
                                    'tp_price':              tp,
                                    'volume_lots':           volume,
                                    'ticket':                ticket_num,
                                    'account_balance':       bal,
                                    'account_equity':        eqt,
                                    'spread_points':         spread_pts,
                                    'ema_20':                indicators.get('ema_20'),
                                    'rsi':                   indicators.get('rsi', curr_rsi),
                                    'adx':                   indicators.get('adx'),
                                    'h1_trend':              h1_trend,
                                    'breakout_high':         indicators.get('breakout_high'),
                                    'breakout_low':          indicators.get('breakout_low'),
                                    'pullback_distance_atr': indicators.get('pullback_distance_atr'),
                                    'volume_ratio':          indicators.get('volume_ratio'),
                                    'momentum_fast_ema':     indicators.get('momentum_fast_ema'),
                                    'momentum_slow_ema':     indicators.get('momentum_slow_ema'),
                                })
                                dc.log_signal(row)
                            except Exception:
                                pass

                        send_webhook({
                            "event": "trade",
                            "symbol": symbol,
                            "direction": direction.upper(),
                            "trigger": trigger,
                            "entry_price": entry_price,
                            "sl": sl, "tp": tp,
                            "volume": volume,
                            "confidence": signal['confidence'],
                            "rsi": curr_rsi,
                            "atr": atr_value,
                            "h1_trend": h1_trend,
                            "strategy": f"viper3_{trigger.lower()}",
                            "timeframe": CONFIG.get('timeframe', 'M5'),
                        })
                    else:
                        logger.warning(f"XX {symbol}: Order rejected by broker")
                        send_webhook({"event": "rejected", "symbol": symbol, "direction": direction.upper(), "trigger": trigger, "entry_price": entry_price, "rsi": curr_rsi})

            except Exception as e:
                logger.error(f"{symbol} error: {e}")

        iteration += 1
        if iteration % 30 == 0:
            logger.info(f"[HEARTBEAT] Strategy loop -- iteration {iteration}")
        send_webhook({"event": "heartbeat", "iteration": iteration})

        time.sleep(CONFIG.get('strategy_interval', 10))  # 10s on M5

# ============================================================================
# TRAILING STOP LOOP (with RSI-based early exit)
# ============================================================================
def trailing_loop():
    broker = get_broker()
    default_tf_str = CONFIG.get('timeframe', 'M5')
    timeframe = TF_MAP.get(default_tf_str, mt5.TIMEFRAME_M5)
    trail_iter = 0

    while not bot_stop.is_set():
        try:
            positions = broker.get_positions()
            trail_iter += 1
            if trail_iter % 60 == 0:
                logger.info(f"[HEARTBEAT] Trailing loop — {len(positions)} positions")

            now = datetime.now(timezone.utc)

            # Track positions and detect closures for ML outcome logging
            current_tickets = {pos.get('ticket', 0) for pos in positions}

            # Add new positions to tracking
            with positions_lock:
                for pos in positions:
                    ticket = pos.get('ticket', 0)
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
                tickets = set(_tracked_positions.keys())
            closed_tickets = tickets - current_tickets
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
                if cfg is None or not cfg.get('use_trailing', True):
                    continue
                sym_tf = TF_MAP.get(cfg.get('timeframe', default_tf_str), timeframe)

                # Friday flatten — close positions before weekend gap
                if now.weekday() == 4:
                    tf = cfg.get('time_filter', {})
                    flatten_hour = tf.get('friday_flatten_hour', 0)
                    if flatten_hour > 0 and now.hour >= flatten_hour:
                        ticket = pos.get('ticket', 0)
                        logger.info(f"[FRIDAY-FLATTEN] Closing {symbol} #{ticket} before weekend")

                        # Get ATR for ML row
                        rates_ff = mt5.copy_rates_from_pos(symbol, sym_tf, 0, 50)
                        curr_atr_ff = 0
                        rates_np_ff = None
                        if rates_ff is not None and len(rates_ff) >= 20:
                            rates_np_ff = mt5_rates_to_numpy(rates_ff)
                            curr_atr_ff = safe_last(ATR(rates_np_ff[:, 2], rates_np_ff[:, 3], rates_np_ff[:, 4], 14))

                        # Log ML outcome inline for Friday flatten (before closing)
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
                                    record_trade_result(symbol, unrealized)
                                    _tracked_positions.pop(ticket, None)
                                    logger.info(f"[OUTCOME] {symbol} #{ticket}: FRIDAY_FLATTEN P&L={unrealized:.2f}")
                            except Exception:
                                pass

                        broker.close_position(ticket)
                        send_webhook({"event": "friday_flatten", "symbol": symbol, "ticket": ticket})
                        continue

                rates = mt5.copy_rates_from_pos(symbol, sym_tf, 0, 200)
                if rates is None or len(rates) < 50:
                    continue
                rates_np = mt5_rates_to_numpy(rates)
                closes = rates_np[:, 4]
                curr_atr = safe_last(ATR(rates_np[:, 2], rates_np[:, 3], closes, 14))

                # RSI-based early exit
                curr_rsi = safe_last(RSI(closes, 14))
                pos_type = normalize_pos_type(pos.get('type', ''))
                ticket = pos.get('ticket', 0)

                if pos_type is None:
                    continue

                rsi_exit_high = cfg.get('rsi_exit_high', 78)
                rsi_exit_low = cfg.get('rsi_exit_low', 22)

                # Profit gate + minimum hold time for RSI exit
                # Prevents RSI from killing trades immediately after entry
                _rsi_entry = _tracked_positions.get(ticket)
                _rsi_trade_age = (now - _rsi_entry['open_time']).total_seconds() if _rsi_entry else 0
                _rsi_unrealized = pos.get('profit', 0.0)
                _rsi_min_hold = 3 * 300  # 3 bars x 300s (M5) = 15 minutes

                # RSI exit gate: only exit if trade is in profit AND held for 3+ bars (15 min on M5)
                _rsi_can_exit = _rsi_trade_age >= _rsi_min_hold and _rsi_unrealized > 0

                if pos_type == 'BUY' and curr_rsi >= rsi_exit_high:
                    if not _rsi_can_exit:
                        logger.info(
                            f"[MR-SKIP] {symbol} #{ticket}: RSI={curr_rsi:.1f} >= {rsi_exit_high} "
                            f"but hold={_rsi_trade_age:.0f}s P&L={_rsi_unrealized:.2f} — skipping RSI exit"
                        )
                    else:
                        logger.info(f"[MR-EXIT] {symbol} #{ticket}: RSI={curr_rsi:.1f} >= {rsi_exit_high}")
                        # Log ML outcome inline for RSI exit
                        entry = _tracked_positions.get(ticket)
                        if entry:
                            try:
                                tick_rsi = mt5.symbol_info_tick(symbol)
                                if tick_rsi:
                                    close_px_rsi = tick_rsi.bid
                                    duration_rsi = (now - entry['open_time']).total_seconds() / 60
                                    unrealized_rsi = pos.get('profit', 0.0)
                                    bal, eqt = get_account_balance_equity()
                                    row = build_ml_row(symbol, 'TRADE_CLOSED', 'OUTCOME', 0, curr_atr, 0, rates_np)
                                    row.update({
                                        'entry_price': entry['entry_price'],
                                        'sl_price':    entry['sl'],
                                        'tp_price':    entry['tp'],
                                        'volume_lots': entry['volume'],
                                        'ticket':      ticket,
                                        'exit_price':  close_px_rsi,
                                        'exit_reason': 'RSI_EXIT',
                                        'profit':      unrealized_rsi,
                                        'outcome':     'WIN' if unrealized_rsi > 0 else ('LOSS' if unrealized_rsi < 0 else 'BREAKEVEN'),
                                        'duration_minutes': round(duration_rsi, 1),
                                        'account_balance': bal,
                                        'account_equity':  eqt,
                                        'spread_points': get_spread_points(symbol),
                                        'rsi':         curr_rsi,
                                    })
                                    dc = get_ml_dc()
                                    if dc: dc.log_signal(row)
                                    rm = get_risk_manager()
                                    if rm:
                                        try:
                                            rm.on_trade_close(int(ticket), float(unrealized_rsi), float(duration_rsi) * 60.0, symbol)
                                            notify_portfolio_trade_close(ticket, unrealized_rsi, float(duration_rsi) * 60.0, symbol)
                                        except Exception as e:
                                            logger.warning(f"Risk close update failed: {e}")
                                    record_trade_result(symbol, unrealized_rsi)
                                    _tracked_positions.pop(ticket, None)
                            except Exception:
                                pass
                        broker.close_position(ticket)
                        send_webhook({"event": "mr_exit", "symbol": symbol, "ticket": ticket, "exit_rsi": curr_rsi})
                        continue
                elif pos_type == 'SELL' and curr_rsi <= rsi_exit_low:
                    if not _rsi_can_exit:
                        logger.info(
                            f"[MR-SKIP] {symbol} #{ticket}: RSI={curr_rsi:.1f} <= {rsi_exit_low} "
                            f"but hold={_rsi_trade_age:.0f}s P&L={_rsi_unrealized:.2f} — skipping RSI exit"
                        )
                    else:
                        logger.info(f"[MR-EXIT] {symbol} #{ticket}: RSI={curr_rsi:.1f} <= {rsi_exit_low}")
                        # Log ML outcome inline for RSI exit
                        entry = _tracked_positions.get(ticket)
                        if entry:
                            try:
                                tick_rsi = mt5.symbol_info_tick(symbol)
                                if tick_rsi:
                                    close_px_rsi = tick_rsi.ask
                                    duration_rsi = (now - entry['open_time']).total_seconds() / 60
                                    unrealized_rsi = pos.get('profit', 0.0)
                                    bal, eqt = get_account_balance_equity()
                                    row = build_ml_row(symbol, 'TRADE_CLOSED', 'OUTCOME', 0, curr_atr, 0, rates_np)
                                    row.update({
                                        'entry_price': entry['entry_price'],
                                        'sl_price':    entry['sl'],
                                        'tp_price':    entry['tp'],
                                        'volume_lots': entry['volume'],
                                        'ticket':      ticket,
                                        'exit_price':  close_px_rsi,
                                        'exit_reason': 'RSI_EXIT',
                                        'profit':      unrealized_rsi,
                                        'outcome':     'WIN' if unrealized_rsi > 0 else ('LOSS' if unrealized_rsi < 0 else 'BREAKEVEN'),
                                        'duration_minutes': round(duration_rsi, 1),
                                        'account_balance': bal,
                                        'account_equity':  eqt,
                                        'spread_points': get_spread_points(symbol),
                                        'rsi':         curr_rsi,
                                    })
                                    dc = get_ml_dc()
                                    if dc: dc.log_signal(row)
                                    rm = get_risk_manager()
                                    if rm:
                                        try:
                                            rm.on_trade_close(int(ticket), float(unrealized_rsi), float(duration_rsi) * 60.0, symbol)
                                            notify_portfolio_trade_close(ticket, unrealized_rsi, float(duration_rsi) * 60.0, symbol)
                                        except Exception as e:
                                            logger.warning(f"Risk close update failed: {e}")
                                    record_trade_result(symbol, unrealized_rsi)
                                    _tracked_positions.pop(ticket, None)
                            except Exception:
                                pass
                        broker.close_position(ticket)
                        send_webhook({"event": "mr_exit", "symbol": symbol, "ticket": ticket, "exit_rsi": curr_rsi})
                        continue

                # ATR trailing stop (or fixed_trail_points override)
                fixed_trail_pts = cfg.get('fixed_trail_points', 0) or 0
                if fixed_trail_pts > 0:
                    sym_info_t = None
                    try:
                        sym_info_t = get_broker().get_symbol_info(symbol)
                    except Exception:
                        pass
                    pt_t = float(sym_info_t.get('point', 0)) if sym_info_t else 0.0
                    trail_dist = pt_t * fixed_trail_pts if pt_t > 0 else curr_atr * cfg.get('atr_trail_multiplier', 1.0)
                else:
                    trail_mult = cfg.get('atr_trail_multiplier', 1.0)
                    trail_dist = curr_atr * trail_mult

                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue

                current_sl = float(pos.get('sl') or 0.0)
                entry_price = float(pos.get('price_open', pos.get('price', 0)))
                digits = get_symbol_digits(symbol)

                # BREAKEVEN MOVE (fixed_breakeven_points or ATR-based)
                be_trigger_mult = cfg.get('breakeven_atr_mult', 1.0)
                trail_start_mult = cfg.get('trail_start_atr_mult', be_trigger_mult)
                be_offset = curr_atr * 0.1

                fixed_be_pts = cfg.get('fixed_breakeven_points', 0) or 0
                if fixed_be_pts > 0:
                    sym_info_be = None
                    try:
                        sym_info_be = get_broker().get_symbol_info(symbol)
                    except Exception:
                        pass
                    pt_be = float(sym_info_be.get('point', 0)) if sym_info_be else 0.0
                    if pt_be > 0:
                        be_trigger_dist = pt_be * fixed_be_pts
                        be_offset = pt_be  # 1-point buffer above entry
                        if entry_price > 0:
                            if pos_type == 'BUY':
                                current_price = tick.bid
                                be_trigger = entry_price + be_trigger_dist
                                be_sl = entry_price + be_offset
                                if current_price >= be_trigger and current_sl < entry_price:
                                    broker.modify_position(ticket, sl=round(be_sl, digits))
                                    logger.info(f"[BE] {symbol} #{ticket}: Fixed-BE triggered @ {fixed_be_pts}pts — SL moved to {be_sl:.{digits}f}")
                                    send_webhook({"event": "breakeven", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(be_sl, digits), "rsi": curr_rsi})
                                    continue
                            elif pos_type == 'SELL':
                                current_price = tick.ask
                                be_trigger = entry_price - be_trigger_dist
                                be_sl = entry_price - be_offset
                                if current_price <= be_trigger and (current_sl > entry_price or current_sl == 0):
                                    broker.modify_position(ticket, sl=round(be_sl, digits))
                                    logger.info(f"[BE] {symbol} #{ticket}: Fixed-BE triggered @ {fixed_be_pts}pts — SL moved to {be_sl:.{digits}f}")
                                    send_webhook({"event": "breakeven", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(be_sl, digits), "rsi": curr_rsi})
                                    continue

                if entry_price > 0 and not fixed_be_pts:
                    if pos_type == 'BUY':
                        current_price = tick.bid
                        be_trigger = entry_price + (curr_atr * be_trigger_mult)
                        be_sl = entry_price + be_offset
                        if current_price >= be_trigger and current_sl < entry_price:
                            broker.modify_position(ticket, sl=round(be_sl, digits))
                            logger.info(f"[BE] {symbol} #{ticket}: Breakeven triggered — SL moved to {be_sl:.{digits}f} (entry was {entry_price:.{digits}f})")
                            send_webhook({"event": "breakeven", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(be_sl, digits), "rsi": curr_rsi})
                            continue

                    elif pos_type == 'SELL':
                        current_price = tick.ask
                        be_trigger = entry_price - (curr_atr * be_trigger_mult)
                        be_sl = entry_price - be_offset
                        if current_price <= be_trigger and (current_sl > entry_price or current_sl == 0):
                            broker.modify_position(ticket, sl=round(be_sl, digits))
                            logger.info(f"[BE] {symbol} #{ticket}: Breakeven triggered — SL moved to {be_sl:.{digits}f} (entry was {entry_price:.{digits}f})")
                            send_webhook({"event": "breakeven", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(be_sl, digits), "rsi": curr_rsi})
                            continue

                if entry_price > 0:
                    if pos_type == 'BUY' and tick.bid < entry_price + (curr_atr * trail_start_mult):
                        continue
                    elif pos_type == 'SELL' and tick.ask > entry_price - (curr_atr * trail_start_mult):
                        continue

                if pos_type == 'BUY':
                    current_price = tick.bid
                    new_sl = current_price - trail_dist
                    if new_sl > current_sl and current_sl > 0:
                        broker.modify_position(ticket, sl=round(new_sl, digits))
                        logger.info(f"[TRAIL] {symbol} #{ticket}: SL {current_sl:.{digits}f} -> {new_sl:.{digits}f}")
                        send_webhook({"event": "trail_update", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(new_sl, digits), "rsi": curr_rsi})

                elif pos_type == 'SELL':
                    current_price = tick.ask
                    new_sl = current_price + trail_dist
                    if (new_sl < current_sl or current_sl == 0) and new_sl > 0:
                        broker.modify_position(ticket, sl=round(new_sl, digits))
                        logger.info(f"[TRAIL] {symbol} #{ticket}: SL {current_sl:.{digits}f} -> {new_sl:.{digits}f}")
                        send_webhook({"event": "trail_update", "symbol": symbol, "ticket": ticket, "direction": pos_type, "old_sl": current_sl, "new_sl": round(new_sl, digits), "rsi": curr_rsi})

        except Exception as e:
            logger.error(f"[TRAIL] Error: {e}")

        time.sleep(10)  # 10s on M5

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
        'bot': 'viper-v3',
        'profile': PROFILE_NAME,
        'account': ACCOUNT_NUMBER,
        'timeframe': CONFIG.get('timeframe', 'M5'),
        'balance': account.get('balance', 0) if account else 0,
        'equity': account.get('equity', 0) if account else 0,
        'positions_count': len(positions),
        'portfolio_risk': portfolio,
        'timestamp': datetime.now(timezone.utc).isoformat(),
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
    return jsonify((broker.get_account_info() or {}) if broker else {})

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
                tf_close = TF_MAP.get(CONFIG.get('timeframe', 'M5'), mt5.TIMEFRAME_M5)
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
    return jsonify({'success': True, 'message': 'Strategy and trailing started'})

@app.route('/stop', methods=['POST'])
def stop():
    bot_stop.set()
    return jsonify({'success': True, 'message': 'Strategy stopped'})

@app.route('/analyze/<symbol>', methods=['GET'])
def analyze(symbol):
    timeframe = TF_MAP.get(CONFIG.get('timeframe', 'M5'), mt5.TIMEFRAME_M5)
    cfg = CONFIG['symbols'].get(symbol, {})
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
    if rates is None or len(rates) < 60:
        return jsonify({'error': 'No data'}), 400
    rates_np = mt5_rates_to_numpy(rates)
    h1_trend = get_h1_trend(symbol, cfg)
    signal, atr_val, rsi, trigger, indicators = check_all_signals(rates_np, cfg, h1_trend)
    return jsonify({
        'symbol': symbol,
        'strategy': 'viper3_king_cobra_auto',
        'signal': signal['direction'] if signal else 'HOLD',
        'trigger': trigger,
        'confidence': signal['confidence'] if signal else 0,
        'reason': signal['reason'] if signal else 'No trigger fired',
        'atr': atr_val, 'rsi': rsi,
        'h1_trend': h1_trend,
        'timeframe': CONFIG.get('timeframe', 'M5'),
    })

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
    config_path = os.path.join(base_dir, 'viper_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found in viper_config.json")
    return profiles[profile]

def initialize():
    global CONFIG, ACCOUNT_NUMBER, PROFILE_NAME

    parser = argparse.ArgumentParser(description='VIPER v3 — King Cobra Auto Strategy')
    parser.add_argument('--profile', required=True, help='Profile name from viper_config.json')
    args = parser.parse_args()

    PROFILE_NAME = args.profile
    CONFIG = load_config(PROFILE_NAME)
    ACCOUNT_NUMBER = CONFIG['account_number']

    init_logger()
    logger.info("=" * 60)
    logger.info(f"VIPER v3 Starting — Profile: {PROFILE_NAME}")
    logger.info(f"Account: {ACCOUNT_NUMBER}, Timeframe: {CONFIG.get('timeframe')}")
    logger.info(f"Strategy: King Cobra Auto (Momentum + EMA Pullback)")
    logger.info(f"Triggers: current-candle 3/8 EMA momentum + EMA20 pullback + H1 trend filter")
    logger.info("=" * 60)

    password = os.environ.get('MT5_PASSWORD', '')
    if not password:
        logger.error("MT5_PASSWORD not set. Exiting.")
        return False

    try:
        broker = MT5Broker(
            account=ACCOUNT_NUMBER, password=password,
            server=CONFIG['server'], mt5_path=CONFIG.get('mt5_path'),
            owner_tag=f"{BOT_NAME}:{PROFILE_NAME}",
        )
        set_broker(broker)
        logger.info("Broker connected")
    except Exception as e:
        logger.error(f"Broker init failed: {e}")
        return False

    try:
        rc = CONFIG.get('risk', {})
        # Compute daily loss limit from % of current balance (scales with account size)
        account_info = broker.get_account_info() or {}
        curr_balance = account_info.get('balance', 0.0)
        curr_equity = account_info.get('equity', 0.0)
        capital_base = max(abs(curr_balance), abs(curr_equity), 150.0)
        daily_pct = rc.get('daily_loss_limit_pct', 13)
        daily_dollar = -(capital_base * daily_pct / 100)
        rm = RiskManagerUltra(
            daily_loss_limit=daily_dollar,
            max_consecutive_losses=rc.get('max_consecutive_losses', 3),
            max_drawdown_percent=rc.get('max_drawdown_pct', 12),
            state_file=f"risk_state_ultra_{ACCOUNT_NUMBER}.json",
        )
        set_risk_manager(rm)
        logger.info(f"Risk manager ready: daily limit ${daily_dollar:.2f} ({daily_pct}% of base ${capital_base:.2f})")
    except Exception as e:
        logger.error(f"Risk manager failed: {e}")
        return False

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pg = PortfolioRiskGuard(
            bot_name=BOT_NAME,
            account_number=ACCOUNT_NUMBER,
            config_path=os.path.join(base_dir, 'portfolio_risk_config.json'),
            db_path=os.path.join(base_dir, 'portfolio_risk_guard.db'),
        )
        set_portfolio_guard(pg)
        logger.info("PortfolioRiskGuard initialized")
    except Exception as e:
        logger.warning(f"PortfolioRiskGuard init failed: {e}")

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
        tl = TradeDecisionLogger(f"viper_v3_{PROFILE_NAME}")
        set_trade_logger(tl)
        logger.info("Trade logger ready")
    except Exception as e:
        logger.warning(f"Trade logger init failed (non-fatal): {e}")

    # Initialize ML data collector
    try:
        dc = SharedDataCollector('viper', ACCOUNT_NUMBER)
        set_ml_dc(dc)
        logger.info(f"ML data collector initialized — saving to {dc.data_dir}")
    except Exception as e:
        logger.warning(f"Data collector init failed (non-fatal): {e}")

    # Load signal quality ML filter model
    load_signal_quality_model()

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

    if not initialize():
        sys.exit(1)

    sync_open_positions()

    _strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
    _trailing_thread = threading.Thread(target=trailing_loop, daemon=True)
    _strategy_thread.start()
    _trailing_thread.start()

    port = CONFIG.get('server_port', 8059)
    logger.info(f"Strategy + trailing threads started")
    logger.info(f"Flask API on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)

if __name__ == '__main__':
    main()










