"""
INDIAN KING COBRA v2.0 — Auto-Execution Scalping Bot
=====================================================
M5 Scanner | MOMENTUM + EMA_PULLBACK | 4/5-Check System
Auto-executes when all checks pass. ATR trailing stops.
News layer checks every 2h to avoid blind entries.

Symbols: BTCUSD-ECN, XAUUSD-ECN, GBPJPY-ECN, EURUSD-ECN, USDJPY-ECN, USOUSD-ECN
Account: VT Markets Raw ECN (CAD base) | Raw spread + $6/lot commission

Check System:
  1. Trigger fires (MOMENTUM or EMA_PULLBACK)
  2. H1 trend confirms direction
  3. ADX >= 25 (momentum present)
  4. RSI between 30-70 (not overbought/oversold)
  5. ML agrees (XAU only — required)
     ML soft (BTC — reduce lot if disagrees)
  6. News not strongly against direction

Usage: python king_cobra_auto.py
       python king_cobra_auto.py --config king_cobra_pepperstone.json
"""
import sys
import os
import json
import time
import csv
import threading
import logging
import argparse
from datetime import datetime, timezone, timedelta
import urllib.request
import urllib.error
import ssl
import traceback

# Keep console writes alive even when Windows stdout is using a narrow code page.
for stream_name in ('stdout', 'stderr'):
    stream = getattr(sys, stream_name, None)
    reconfigure = getattr(stream, 'reconfigure', None)
    if callable(reconfigure):
        try:
            reconfigure(errors='replace')
        except Exception:
            pass

# SSL context for news/API calls. This environment cannot verify the issuer chain,
# so use an unverified context for public API reads and outbound notifications.
_ssl_ctx = ssl._create_unverified_context()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pro_modules'))

import MetaTrader5 as mt5
import numpy as np

from ultra_fast_indicators import ema_numba, rsi_numba, atr_numba, adx_numba

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(level=logging.WARNING, format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('king_cobra')
logger.setLevel(logging.INFO)

default_cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
cuda_path = os.environ.get('CUDA_PATH')
if not cuda_path and os.path.isdir(default_cuda_path):
    cuda_path = default_cuda_path
    os.environ['CUDA_PATH'] = cuda_path
if cuda_path:
    for extra_dir in (os.path.join(cuda_path, 'bin'), os.path.join(cuda_path, 'bin', 'x64')):
        if os.path.isdir(extra_dir):
            if extra_dir not in os.environ.get('PATH', ''):
                os.environ['PATH'] = extra_dir + os.pathsep + os.environ.get('PATH', '')
            add_dll_directory = getattr(os, 'add_dll_directory', None)
            if callable(add_dll_directory):
                try:
                    add_dll_directory(extra_dir)
                except Exception:
                    pass

try:
    import cupy as cp
    _gpu_probe = cp.asarray([1.0, 2.0], dtype=cp.float64)
    cp.asnumpy(_gpu_probe * 2.0)
    GPU_AVAILABLE = True
    logger.info('[GPU] CuPy loaded - GPU acceleration enabled')
except Exception:
    cp = None
    GPU_AVAILABLE = False
    logger.info('[GPU] CuPy not available - using CPU')

try:
    import xgboost as xgb
    XGBOOST_GPU_AVAILABLE = bool(xgb.build_info().get('USE_CUDA'))
except Exception:
    XGBOOST_GPU_AVAILABLE = False

# File logging — logs everything to daily log files
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), exist_ok=True)
_log_date = datetime.now().strftime('%Y-%m-%d')
_file_handler = logging.FileHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'king_cobra_{_log_date}.log'),
    encoding='utf-8'
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(_file_handler)

# ============================================================================
# GLOBALS
# ============================================================================
CONFIG = {}
BOT_NAME = 'indian_king_cobra_v2'
MAGIC_NUMBER = 888888

# Per-symbol cooldown timers  {symbol: datetime_utc_when_cooldown_expires}
cooldowns = {}
cooldown_lock = threading.Lock()

# News sentiment cache  {symbol_group: {sentiment, headlines, updated_at}}
news_cache = {}
news_lock = threading.Lock()

# Daily P&L tracking
daily_pnl = {'date': None, 'realized': 0.0, 'closed_tickets': set()}
pnl_lock = threading.Lock()

# Shutdown flag
shutdown_event = threading.Event()

# Signal dedup — track last signaled candle per symbol {symbol: candle_timestamp}
_last_signal_candle = {}
signal_dedup_lock = threading.Lock()

# Closed-position reconciliation — broker history can lag briefly after a position disappears.
_pending_close_checks = {}
pending_close_lock = threading.Lock()

# Circuit breaker — consecutive loss tracking
consecutive_losses = 0
circuit_breaker_until = None
circuit_breaker_lock = threading.Lock()

# Persistent state
STATE_FILE = 'king_cobra_state.json'
state = {
    'trades': [],               # Full trade history [{symbol, direction, entry, exit, pnl, trigger, time_open, time_close}]
    'daily_stats': {},          # {"2026-03-20": {realized, trades, wins, losses}}
    'cooldowns': {},            # {symbol: iso_timestamp}
    'high_water_equity': 0.0,   # Highest equity seen
    'starting_balance': 0.0,    # Balance when bot first started
    'total_realized': 0.0,      # Cumulative realized P&L
    'total_trades': 0,
    'total_wins': 0,
    'total_losses': 0,
    'news_cache': {},           # Preserved across restarts
    'last_news_update': None,
    'known_positions': {},      # {ticket: {symbol, direction, entry, sl, tp, trigger, time_open}}
}
state_lock = threading.RLock()  # RLock allows same thread to re-enter (prevents deadlock)


# ============================================================================
# PERSISTENCE — Save/Load state to JSON file
# ============================================================================
def load_state():
    """Load persistent state from disk. Called once at startup."""
    global state
    state_path = os.path.join(os.path.dirname(__file__), STATE_FILE)
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                saved = json.load(f)
            # Merge saved into state (keeps defaults for new keys)
            for key in state:
                if key in saved:
                    state[key] = saved[key]
            logger.info(f'[STATE] Loaded: {state["total_trades"]} trades, ${state["total_realized"]:+.2f} total P&L, HWM ${state["high_water_equity"]:,.2f}')
        except Exception as e:
            logger.warning(f'[STATE] Could not load state: {e} — starting fresh')
    else:
        logger.info('[STATE] No state file found — starting fresh')


def save_state():
    """Save persistent state to disk. Called after every trade event."""
    state_path = os.path.join(os.path.dirname(__file__), STATE_FILE)
    with state_lock:
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f'[STATE] Save failed: {e}')


def record_trade_open(ticket, symbol, direction, entry, sl, tp, trigger, lot):
    """Record a new trade entry in persistent state."""
    with state_lock:
        state['known_positions'][str(ticket)] = {
            'ticket': ticket,
            'symbol': symbol,
            'direction': direction,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'trigger': trigger,
            'lot': lot,
            'time_open': datetime.now(timezone.utc).isoformat(),
        }
    save_state()


def record_trade_close(ticket, symbol, direction, entry, exit_price, pnl, reason):
    """Record a trade exit in persistent state."""
    now = datetime.now(timezone.utc)
    today = now.strftime('%Y-%m-%d')

    with state_lock:
        # Remove from known positions
        open_info = state['known_positions'].pop(str(ticket), {})
        time_open = open_info.get('time_open', now.isoformat())
        trigger = open_info.get('trigger', 'unknown')
        lot = open_info.get('lot', 0)

        # Add to trade history
        trade = {
            'ticket': ticket,
            'symbol': symbol,
            'direction': direction,
            'trigger': trigger,
            'lot': lot,
            'entry': entry,
            'exit': exit_price,
            'pnl': round(pnl, 2),
            'reason': reason,
            'time_open': time_open,
            'time_close': now.isoformat(),
        }
        state['trades'].append(trade)

        # Update cumulative stats
        state['total_realized'] = round(state['total_realized'] + pnl, 2)
        state['total_trades'] += 1
        if pnl >= 0:
            state['total_wins'] += 1
        else:
            state['total_losses'] += 1

        # Update daily stats
        if today not in state['daily_stats']:
            state['daily_stats'][today] = {'realized': 0, 'trades': 0, 'wins': 0, 'losses': 0}
        day = state['daily_stats'][today]
        day['realized'] = round(day['realized'] + pnl, 2)
        day['trades'] += 1
        if pnl >= 0:
            day['wins'] += 1
        else:
            day['losses'] += 1

    save_state()

    # Update circuit breaker
    update_circuit_breaker(pnl >= 0)


def update_high_water(equity):
    """Track highest equity ever seen."""
    with state_lock:
        if equity > state['high_water_equity']:
            state['high_water_equity'] = equity
            save_state()


def restore_cooldowns():
    """Restore cooldown timers from saved state."""
    global cooldowns
    saved = state.get('cooldowns', {})
    now = datetime.now(timezone.utc)
    restored = 0
    with cooldown_lock:
        for symbol, expires_str in saved.items():
            try:
                expires = datetime.fromisoformat(expires_str)
                if expires > now:
                    cooldowns[symbol] = expires
                    restored += 1
            except Exception:
                pass
    if restored:
        logger.info(f'[STATE] Restored {restored} active cooldowns')


def restore_news_cache():
    """Restore news cache from saved state."""
    global news_cache
    saved_news = state.get('news_cache', {})
    saved_time = state.get('last_news_update')
    if saved_news and saved_time:
        try:
            last_update = datetime.fromisoformat(saved_time)
            age_hours = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
            if age_hours < 12:  # Only restore if less than 12h old
                with news_lock:
                    news_cache = saved_news
                logger.info(f'[STATE] Restored news cache ({age_hours:.1f}h old)')
            else:
                logger.info(f'[STATE] News cache too old ({age_hours:.1f}h), will refresh')
        except Exception:
            pass


def find_closing_deal(ticket_str, lookback_hours=168):
    """Find the latest closing deal for a tracked position."""
    out_entries = {mt5.DEAL_ENTRY_OUT}
    inout_entry = getattr(mt5, 'DEAL_ENTRY_INOUT', None)
    if inout_entry is not None:
        out_entries.add(inout_entry)

    def pick_candidate(deals):
        if not deals:
            return None

        exact_magic = []
        fallback = []
        for deal in deals:
            if str(getattr(deal, 'position_id', '')) != ticket_str:
                continue
            if getattr(deal, 'entry', None) not in out_entries:
                continue
            fallback.append(deal)
            if getattr(deal, 'magic', MAGIC_NUMBER) == MAGIC_NUMBER:
                exact_magic.append(deal)

        candidates = exact_magic or fallback
        if not candidates:
            return None

        return max(candidates, key=lambda d: (getattr(d, 'time_msc', 0), getattr(d, 'time', 0), getattr(d, 'ticket', 0)))

    try:
        position_id = int(ticket_str)
    except (TypeError, ValueError):
        position_id = None

    # Prefer direct position lookup. Range-based history can miss closes on
    # some brokers/terminal timezones even when the position history exists.
    if position_id is not None:
        deal = pick_candidate(mt5.history_deals_get(position=position_id))
        if deal:
            return deal

    now_naive = datetime.now()
    start_time = now_naive - timedelta(hours=lookback_hours)
    deal = pick_candidate(mt5.history_deals_get(start_time, now_naive))
    if deal:
        return deal

    return None


def classify_close_reason(info, deal, pnl):
    """Best-effort close reason using MT5 deal metadata first, then SL/TP proximity."""
    if abs(pnl) < 0.01:
        return 'breakeven'

    deal_reason = getattr(deal, 'reason', None)
    if deal_reason == getattr(mt5, 'DEAL_REASON_TP', object()):
        return 'tp_hit'
    if deal_reason == getattr(mt5, 'DEAL_REASON_SL', object()):
        return 'sl_hit'
    if deal_reason == getattr(mt5, 'DEAL_REASON_SO', object()):
        return 'stopout'
    if deal_reason == getattr(mt5, 'DEAL_REASON_CLIENT', object()):
        return 'manual_close'

    exit_price = float(getattr(deal, 'price', 0.0) or 0.0)
    entry = float(info.get('entry', exit_price) or exit_price)
    sl = float(info.get('sl', 0.0) or 0.0)
    tp = float(info.get('tp', 0.0) or 0.0)
    point = CONFIG.get('symbols', {}).get(info.get('symbol', ''), {}).get('point_value', 0.01)
    tolerance = max(point * 50, max(abs(entry - sl), abs(tp - entry), point) * 0.05)

    if tp and abs(exit_price - tp) <= tolerance:
        return 'tp_hit'
    if sl and abs(exit_price - sl) <= tolerance:
        return 'sl_hit'
    if pnl > 0:
        return 'profit_close'
    return 'loss_close'


def note_pending_close(ticket_str):
    """Track delayed broker history so we retry without spamming logs."""
    now_ts = time.time()
    with pending_close_lock:
        pending = _pending_close_checks.get(ticket_str)
        if pending is None:
            pending = {'first_seen': now_ts, 'last_log': 0.0}
            _pending_close_checks[ticket_str] = pending
        age_seconds = now_ts - pending['first_seen']
        should_log = (now_ts - pending['last_log']) >= 60.0
        if should_log:
            pending['last_log'] = now_ts
    return age_seconds, should_log


def clear_pending_close(ticket_str):
    with pending_close_lock:
        _pending_close_checks.pop(ticket_str, None)


def finalize_closed_position(ticket_str, info, deal, source_tag):
    """Record, notify, and cool down a position once its closing deal is visible."""
    pnl = deal.profit + deal.swap + deal.commission
    reason = classify_close_reason(info, deal, pnl)
    logger.info(
        f'[{source_tag}] {info["symbol"]} #{ticket_str} closed: {reason.upper()} | '
        f'Entry {fmt_price(info["entry"])} -> Exit {fmt_price(deal.price)} | P&L {pnl:+.2f} CAD'
    )

    record_trade_close(int(ticket_str), info['symbol'], info['direction'], info['entry'], deal.price, pnl, reason)
    notify_exit(info['symbol'], info['direction'], info['entry'], deal.price, pnl, reason.upper())
    set_cooldown(info['symbol'])
    clear_pending_close(ticket_str)
    return pnl, reason


def reconcile_positions():
    """On startup, check MT5 for open positions we might not know about.
    Also detect positions that closed while bot was offline."""
    positions = mt5.positions_get()
    our_positions = [p for p in (positions or []) if p.magic == MAGIC_NUMBER]

    # Find positions in MT5 that we don't have in state
    known = state.get('known_positions', {})
    for pos in our_positions:
        ticket_str = str(pos.ticket)
        if ticket_str not in known:
            direction = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'
            logger.info(f'[STATE] Found untracked position: {direction} {pos.symbol} #{pos.ticket} @ {pos.price_open}')
            record_trade_open(pos.ticket, pos.symbol, direction, pos.price_open,
                            pos.sl, pos.tp, 'pre-restart', pos.volume)

    # Find positions in state that are no longer in MT5 (closed while offline)
    mt5_tickets = {str(p.ticket) for p in our_positions}
    closed_offline = []
    for ticket_str, info in list(known.items()):
        if ticket_str not in mt5_tickets:
            closed_offline.append((ticket_str, info))

    # Try to find close details from deal history
    if closed_offline:
        for ticket_str, info in closed_offline:
            deal = find_closing_deal(ticket_str)
            if deal:
                finalize_closed_position(ticket_str, info, deal, 'STATE')
            else:
                logger.info(f'[STATE] {info["symbol"]} #{ticket_str} is closed, waiting for broker history to report the exit deal')

    active = len(our_positions)
    if active > 0:
        logger.info(f'[STATE] {active} active position(s) from previous session — trailing monitor will pick them up')
    return our_positions


def print_lifetime_stats():
    """Print cumulative bot performance since first run."""
    with state_lock:
        total = state['total_trades']
        wins = state['total_wins']
        losses = state['total_losses']
        realized = state['total_realized']
        hwm = state['high_water_equity']
        start_bal = state['starting_balance']

    if total == 0:
        print('  No trade history yet')
        return

    win_rate = (wins / total * 100) if total > 0 else 0
    avg_pnl = realized / total if total > 0 else 0

    print(f'\n  ╔══════════════════════════════════════════╗')
    print(f'  ║   LIFETIME STATS                         ║')
    print(f'  ╠══════════════════════════════════════════╣')
    print(f'  ║  Total Trades:  {total:<25}║')
    print(f'  ║  Wins/Losses:   {wins}W / {losses}L ({win_rate:.1f}%)        ║')
    print(f'  ║  Total P&L:     {"${:+,.2f}".format(realized):<25}║')
    print(f'  ║  Avg P&L/Trade: {"${:+,.2f}".format(avg_pnl):<25}║')
    print(f'  ║  High Water:    {"${:,.2f}".format(hwm):<25}║')
    if start_bal > 0:
        roi = (realized / start_bal) * 100
        print(f'  ║  ROI:           {roi:+.1f}%{" "*20}║')
    print(f'  ╚══════════════════════════════════════════╝')

    # Last 5 trades
    trades = state.get('trades', [])
    if trades:
        print(f'\n  Last {min(5, len(trades))} trades:')
        for t in trades[-5:]:
            emoji = '✅' if t['pnl'] >= 0 else '❌'
            print(f'    {emoji} {t["direction"]} {t["symbol"]} | {t["trigger"]} | ${t["pnl"]:+.2f} | {t.get("time_close", "")[:16]}')

    # Today's stats
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    day = state.get('daily_stats', {}).get(today)
    if day:
        print(f'\n  Today ({today}): {day["trades"]} trades | {day["wins"]}W/{day["losses"]}L | ${day["realized"]:+.2f}')
    print()


# ============================================================================
# CONFIG
# ============================================================================
def load_config(path):
    global CONFIG, MAGIC_NUMBER
    with open(path) as f:
        CONFIG = json.load(f)
    MAGIC_NUMBER = CONFIG.get('magic_number', 888888)
    return CONFIG

# ============================================================================
# TELEGRAM NOTIFICATIONS
# ============================================================================
def send_telegram(title, message):
    """Send push notification via Telegram."""
    wh = CONFIG.get('webhook', {})
    if not wh.get('enabled', False):
        return
    tg_token = wh.get('telegram_bot_token', '')
    tg_chat = wh.get('telegram_chat_id', '')
    if not tg_token or not tg_chat:
        return

    def _send():
        try:
            text = f"*{title}*\n\n{message}"
            tg_url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            payload = json.dumps({
                "chat_id": tg_chat,
                "text": text,
                "parse_mode": "Markdown"
            }).encode('utf-8')
            req = urllib.request.Request(tg_url, data=payload,
                headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=5, context=_ssl_ctx)
        except Exception as e:
            logger.debug(f'Telegram error: {e}')

    threading.Thread(target=_send, daemon=True).start()


def notify_entry(symbol, direction, trigger, entry, sl, tp, lot, risk_pct, indicators, ml_score, news):
    """Send Telegram notification for trade entry."""
    arrow = '🟢 BUY' if direction == 'BUY' else '🔴 SELL'
    sl_pips = abs(entry - sl)
    tp_pips = abs(tp - entry)

    ml_line = ''
    if ml_score:
        agrees = '✅' if ml_score['action'] == direction else '❌'
        ml_line = f"\nML: {ml_score['action']} ({ml_score['confidence']:.0%}) {agrees}"

    news_line = ''
    if news:
        news_line = f"\nNews: {news.get('sentiment', 'N/A')} ({news.get('summary', '')[:60]})"

    title = f"{arrow} ENTRY — {symbol} — {trigger}"
    message = (
        f"Entry: {fmt_price(entry)}\n"
        f"SL: {fmt_price(sl)}\n"
        f"TP: {fmt_price(tp)}\n"
        f"Lot: {lot} | Risk: {risk_pct:.1f}%\n"
        f"─────────────\n"
        f"RSI: {indicators.get('rsi', 0):.1f} | ADX: {indicators.get('adx', 0):.1f} | H1: {indicators.get('h1_trend', '?')}"
        f"{ml_line}{news_line}"
    )
    send_telegram(title, message)


def notify_exit(symbol, direction, entry, close_price, pnl, reason):
    """Send Telegram notification for trade exit."""
    arrow = '💰' if pnl > 0 else '🔻'
    title = f"{arrow} EXIT — {symbol} — {reason}"
    message = (
        f"Direction: {direction}\n"
        f"Entry: {fmt_price(entry)} → Exit: {fmt_price(close_price)}\n"
        f"P&L: {'+'if pnl>0 else ''}{pnl:.2f} CAD"
    )
    send_telegram(title, message)


def notify_news_update(news_data):
    """Send Telegram summary of news sentiment."""
    lines = []
    for group, data in news_data.items():
        emoji = '🟢' if data['sentiment'] == 'BULLISH' else '🔴' if data['sentiment'] == 'BEARISH' else '⚪'
        lines.append(f"{emoji} {group}: {data['sentiment']}")
        if data.get('headlines'):
            for h in data['headlines'][:2]:
                lines.append(f"  • {h[:70]}")
    if lines:
        send_telegram('📰 News Update', '\n'.join(lines))


# ============================================================================
# INDICATOR WRAPPERS
# ============================================================================
def EMA(prices, period):
    return ema_numba(prices, period)

def RSI(prices, period=14):
    return rsi_numba(prices, period)

def ATR(highs, lows, closes, period=14):
    return atr_numba(highs, lows, closes, period)

def ADX(highs, lows, closes, period=14):
    return adx_numba(highs, lows, closes, period)

def EMA_GPU(data, period):
    """GPU-accelerated EMA using CuPy with CPU fallback."""
    if not GPU_AVAILABLE or len(data) < period:
        return EMA(data, period)
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
        return ATR(highs, lows, closes, period)
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
        return RSI(closes, period)
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

def safe_last(val):
    if val is None:
        raise ValueError("safe_last: received None")
    if isinstance(val, np.ndarray):
        if len(val) == 0:
            raise ValueError("safe_last: empty array")
        return float(val[-1])
    return float(val)

def get_signal_rsi(rsi_vals):
    """Return RSI value for the current decision candle."""
    try:
        if isinstance(rsi_vals, np.ndarray):
            if len(rsi_vals) >= 1:
                return float(rsi_vals[-1])
            return None
        return float(rsi_vals) if rsi_vals is not None else None
    except Exception:
        return None


# ============================================================================
# FORMATTING
# ============================================================================
def fmt_pts(symbol, price_diff):
    pv = CONFIG.get('symbols', {}).get(symbol, {}).get('point_value', 0.01)
    return int(round(price_diff / pv))

def fmt_price(price):
    if price > 10000:
        return f'{price:,.2f}'
    elif price > 100:
        return f'{price:,.2f}'
    elif price > 10:
        return f'{price:.3f}'
    else:
        return f'{price:.5f}'


# ============================================================================
# H1 TREND FILTER
# ============================================================================
def get_h1_trend(symbol, cfg):
    """Fetch H1 data and determine trend using EMA."""
    h1_ema_period = cfg.get('h1_ema_period', 20)
    try:
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, h1_ema_period + 20)
        if h1_rates is None or len(h1_rates) < h1_ema_period + 5:
            return 'BOTH'
        h1_closes = np.array([r[4] for r in h1_rates], dtype=np.float64)
        h1_highs = np.array([r[2] for r in h1_rates], dtype=np.float64)
        h1_lows = np.array([r[3] for r in h1_rates], dtype=np.float64)
        h1_ema = EMA_GPU(h1_closes, h1_ema_period)
        h1_atr = safe_last(ATR_GPU(h1_highs, h1_lows, h1_closes, 14))
        h1_close = float(h1_closes[-1])
        h1_ema_val = float(h1_ema[-1])
        transition = h1_atr * cfg.get('h1_transition_mult', 0.3)
        if h1_close > h1_ema_val + transition:
            return 'BUY'
        elif h1_close < h1_ema_val - transition:
            return 'SELL'
        else:
            return 'BOTH'
    except Exception:
        return 'BOTH'


# ============================================================================
# TRIGGER: EMA_PULLBACK
# ============================================================================
def check_ema_pullback(opens, closes, highs, lows, ema_vals, rsi_vals, curr_atr, cfg):
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

    indicators = {'ema_20': ema1, 'rsi': rsi1}

    body = abs(float(closes[-1]) - float(opens[-1]))
    candle_range = float(highs[-1]) - float(lows[-1])
    if candle_range > 0:
        body_ratio = body / candle_range
        indicators['body_ratio'] = body_ratio
        if body_ratio < 0.4:
            return None, indicators

    bullish = (c3 > ema3 and low1 <= ema1 + max_dist and low1 >= ema1 - max_dist and
               c1 > ema1 and c1 > c2 and rsi1 > rsi_buy_min)
    bearish = (c3 < ema3 and high1 >= ema1 - max_dist and high1 <= ema1 + max_dist and
               c1 < ema1 and c1 < c2 and rsi1 < rsi_sell_max)

    if bullish:
        return {'trigger': 'EMA_PULLBACK', 'direction': 'BUY', 'confidence': 0.80,
                'reason': f'Bullish pullback near EMA {ema1:.5g}, RSI {rsi1:.0f}'}, indicators
    elif bearish:
        return {'trigger': 'EMA_PULLBACK', 'direction': 'SELL', 'confidence': 0.80,
                'reason': f'Bearish pullback near EMA {ema1:.5g}, RSI {rsi1:.0f}'}, indicators
    return None, indicators


# ============================================================================
# TRIGGER: MOMENTUM
# ============================================================================
def check_momentum(opens, closes, highs, lows, volumes, rsi_vals, curr_atr, cfg):
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

    indicators = {'ema_fast': fast_now, 'ema_slow': slow_now, 'momentum_fast_ema': fast_now,
                  'momentum_slow_ema': slow_now, 'rsi': rsi1, 'volume_ratio': vol_ratio}

    # Candle body confirmation — reject doji/spinning tops.
    body = abs(float(closes[-1]) - float(opens[-1]))
    candle_range = float(highs[-1]) - float(lows[-1])
    if candle_range > 0:
        body_ratio = body / candle_range
        indicators['body_ratio'] = body_ratio
        if body_ratio < 0.4:
            return None, indicators

    if bullish_cross and fast_now > slow_now and c1 > fast_now and c1 > slow_now and vol_ok and rsi1 > 50:
        return {'trigger': 'MOMENTUM', 'direction': 'BUY', 'confidence': 0.75,
                'reason': f'Bullish EMA cross, vol {vol_ratio:.1f}x, RSI {rsi1:.0f}'}, indicators
    elif bearish_cross and fast_now < slow_now and c1 < fast_now and c1 < slow_now and vol_ok and rsi1 < 50:
        return {'trigger': 'MOMENTUM', 'direction': 'SELL', 'confidence': 0.75,
                'reason': f'Bearish EMA cross, vol {vol_ratio:.1f}x, RSI {rsi1:.0f}'}, indicators
    return None, indicators


# ============================================================================
# ML MODELS
# ============================================================================
ML_MODELS = {}


def canonical_symbol(symbol):
    """Normalize broker-specific suffixes so aliases resolve consistently."""
    return str(symbol or '').replace('.s', '').replace('-STD', '').replace('-ECN', '')

def enable_xgboost_gpu_predictor(model, label):
    """Switch fitted XGBoost models to GPU prediction when supported."""
    if not XGBOOST_GPU_AVAILABLE or model is None:
        return
    try:
        if hasattr(model, 'set_params'):
            model.set_params(device='cuda')
        booster = model.get_booster() if hasattr(model, 'get_booster') else model
        if hasattr(booster, 'set_param'):
            booster.set_param({'device': 'cuda'})
            try:
                booster.set_param({'predictor': 'gpu_predictor'})
            except Exception:
                pass
        logger.info(f'  [ML] Enabled GPU predictor for {label}')
    except Exception as e:
        logger.info(f'  [ML] GPU predictor unavailable for {label}: {e}')

def load_ml_models():
    global ML_MODELS
    try:
        from ml_engine import BTCMLModel, XAUMLModel
        btc_model = BTCMLModel(model_path='models/btc_xgb_model.pkl')
        if btc_model.is_trained:
            enable_xgboost_gpu_predictor(getattr(btc_model, 'model', None), 'BTC')
            ML_MODELS['BTCUSD'] = btc_model
            ML_MODELS[canonical_symbol('BTCUSD')] = btc_model
            logger.info('  [ML] Loaded BTC model')
        xau_model = XAUMLModel(model_path='models/xau_xgb_model.pkl')
        if xau_model.is_trained:
            enable_xgboost_gpu_predictor(getattr(xau_model, 'model', None), 'XAU')
            ML_MODELS['XAUUSD'] = xau_model
            ML_MODELS['XAUUSD-ECN'] = xau_model
            ML_MODELS[canonical_symbol('XAUUSD-ECN')] = xau_model
            logger.info('  [ML] Loaded XAU model')
    except Exception as e:
        logger.info(f'  [ML] Models not loaded: {e}')

def get_ml_score(symbol, rates):
    base = canonical_symbol(symbol)
    model = ML_MODELS.get(symbol) or ML_MODELS.get(base)
    if model is None:
        return None
    try:
        rate_array = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rates], dtype=np.float64)
        result = model.predict(rate_array)
        return {'action': result['action'], 'confidence': result['confidence'], 'probs': result['probabilities']}
    except Exception:
        return None


# ============================================================================
# NEWS LAYER — Checks headlines every 2 hours, classifies sentiment
# ============================================================================
NEWS_GROUPS = {
    'CRYPTO': {'symbols': ['BTCUSD'], 'keywords': 'bitcoin crypto BTC cryptocurrency'},
    'GOLD': {'symbols': ['XAUUSD-ECN'], 'keywords': 'gold XAUUSD precious metals safe haven'},
    'FOREX': {'symbols': ['GBPJPY-ECN', 'EURUSD-ECN', 'USDJPY-ECN'], 'keywords': 'forex USD EUR GBP JPY interest rate central bank fed ecb boj'},
    'OIL': {'symbols': ['USOUSD-ECN'], 'keywords': 'crude oil WTI OPEC petroleum energy'},
}

# Map symbols to their news group
SYMBOL_NEWS_GROUP = {}
for group, info in NEWS_GROUPS.items():
    for sym in info['symbols']:
        SYMBOL_NEWS_GROUP[sym] = group
        SYMBOL_NEWS_GROUP[canonical_symbol(sym)] = group


def fetch_news_headlines(keywords, max_results=10):
    """Fetch recent news headlines using free NewsData.io API."""
    api_key = os.environ.get('NEWSDATA_API_KEY', CONFIG.get('newsdata_api_key', ''))
    if not api_key:
        logger.warning('[NEWS] No API key found — check newsdata_api_key in config')
        return []
    logger.info(f'[NEWS] Fetching with key: {api_key[:8]}... | query: {keywords[:30]}')
    try:
        # Use first 3 keywords for query
        query = '+'.join(keywords.split()[:3])
        request_size = max(1, min(int(max_results), 10))
        url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={query}&language=en&size={request_size}"
        req = urllib.request.Request(url, headers={'User-Agent': 'KingCobra/2.0'})
        resp = urllib.request.urlopen(req, timeout=10, context=_ssl_ctx)
        data = json.loads(resp.read().decode())
        headlines = []
        for article in data.get('results', [])[:max_results]:
            title = article.get('title', '')
            desc = article.get('description', '') or ''
            headlines.append(f"{title}. {desc[:100]}")
        return headlines
    except Exception as e:
        logger.warning(f'[NEWS] Fetch error: {e}')
        return []


def classify_news_sentiment(headlines, group_name):
    """Classify news sentiment using OpenAI API (lightweight, no library needed)."""
    api_key = os.environ.get('OPENAI_API_KEY', CONFIG.get('openai_api_key', ''))
    if not api_key or not headlines:
        return {'sentiment': 'NEUTRAL', 'summary': 'No news data', 'headlines': headlines}

    try:
        combined = '\n'.join([f"- {h}" for h in headlines[:8]])
        prompt = f"""Analyze these {group_name} market news headlines and classify overall sentiment for TRADING:

{combined}

Respond in EXACTLY this JSON format:
{{"sentiment": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence reason"}}

Rules:
- BULLISH = prices likely to rise (positive earnings, rate cuts, supply shortage)
- BEARISH = prices likely to fall (bad data, rate hikes, oversupply, geopolitical risk)
- NEUTRAL = mixed signals or no clear direction
- Be conservative — only say BULLISH/BEARISH if sentiment is clearly one-sided"""

        payload = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 150,
        }).encode('utf-8')

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
        )
        resp = urllib.request.urlopen(req, timeout=15, context=_ssl_ctx)
        result = json.loads(resp.read().decode())
        content = result['choices'][0]['message']['content'].strip()

        # Parse JSON from response
        # Handle markdown code blocks
        if '```' in content:
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
        parsed = json.loads(content)

        return {
            'sentiment': parsed.get('sentiment', 'NEUTRAL').upper(),
            'confidence': parsed.get('confidence', 0.5),
            'summary': parsed.get('summary', ''),
            'headlines': headlines,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.debug(f'News classification error for {group_name}: {e}')
        return {'sentiment': 'NEUTRAL', 'summary': f'Classification failed: {e}', 'headlines': headlines}


def update_news_sentiment():
    """Fetch and classify news — uses ONE API call for all markets, then classifies per group."""
    global news_cache
    logger.info('[NEWS] Updating news sentiment...')

    # Single broad query to save API credits (1 credit instead of 4)
    all_headlines = fetch_news_headlines('bitcoin gold forex oil crude USD EUR GBP interest rate OPEC', max_results=20)
    if not all_headlines:
        logger.info('[NEWS] No headlines fetched')
        return

    new_cache = {}
    for group_name, group_info in NEWS_GROUPS.items():
        # Filter headlines relevant to this group using keywords
        keywords_lower = group_info['keywords'].lower().split()
        relevant = [h for h in all_headlines if any(kw in h.lower() for kw in keywords_lower)]

        if relevant:
            result = classify_news_sentiment(relevant, group_name)
        else:
            result = {'sentiment': 'NEUTRAL', 'summary': 'No relevant headlines', 'headlines': [],
                      'updated_at': datetime.now(timezone.utc).isoformat()}
        new_cache[group_name] = result
        logger.info(f'  [NEWS] {group_name}: {result["sentiment"]} — {result.get("summary", "")[:60]}')
        time.sleep(0.5)  # Rate limit between OpenAI calls

    with news_lock:
        news_cache = new_cache

    # Persist news cache
    with state_lock:
        state['news_cache'] = new_cache
        state['last_news_update'] = datetime.now(timezone.utc).isoformat()
    save_state()

    notify_news_update(new_cache)
    logger.info('[NEWS] Update complete')


def news_update_loop():
    """Background thread: update news every N hours (default 4h, saves API credits)."""
    # Initial fetch on startup
    try:
        update_news_sentiment()
    except Exception as e:
        logger.error(f'[NEWS] Initial fetch error: {e}')

    interval_hours = CONFIG.get('news_interval_hours', 4)
    cycles = int(interval_hours * 120)  # 120 half-minutes per hour

    while not shutdown_event.is_set():
        for _ in range(cycles):
            if shutdown_event.is_set():
                return
            time.sleep(30)
        try:
            update_news_sentiment()
        except Exception as e:
            logger.error(f'[NEWS] Update error: {e}')


def get_news_for_symbol(symbol):
    """Get cached news sentiment for a symbol."""
    group = SYMBOL_NEWS_GROUP.get(symbol) or SYMBOL_NEWS_GROUP.get(canonical_symbol(symbol))
    if not group:
        return None
    with news_lock:
        return news_cache.get(group)


def news_allows_trade(symbol, direction):
    """Check if news sentiment allows this trade direction.
    Returns: (allowed: bool, risk_modifier: float, reason: str)
    - allowed=True, modifier=1.0 = full green light
    - allowed=True, modifier=0.5 = reduce lot size
    - allowed=False = skip trade
    """
    news = get_news_for_symbol(symbol)
    if news is None:
        return True, 1.0, 'No news data'

    sentiment = news.get('sentiment', 'NEUTRAL')
    confidence = news.get('confidence', 0.5)

    if sentiment == 'NEUTRAL':
        return True, 1.0, 'News neutral'

    # News agrees with direction
    if (sentiment == 'BULLISH' and direction == 'BUY') or \
       (sentiment == 'BEARISH' and direction == 'SELL'):
        return True, 1.0, f'News {sentiment} — confirms {direction}'

    # News opposes direction
    if confidence >= 0.8:
        return False, 0.0, f'NEWS BLOCK: Strong {sentiment} opposes {direction}'
    elif confidence >= 0.6:
        return True, 0.5, f'News {sentiment} (conf {confidence:.0%}) — reducing lot'
    else:
        return True, 0.8, f'News slightly {sentiment} — minor reduction'


# ============================================================================
# RISK MANAGEMENT — Lot Sizing
# ============================================================================
def calculate_lot_size(symbol, sl_distance, risk_modifier=1.0):
    """Calculate lot size based on risk % and SL distance.
    Uses MT5 tick_value for automatic currency conversion.
    Accounts for commission ($6/lot round turn on VT Markets raw spread).
    """
    risk_pct = CONFIG.get('risk_percent', 2.0) * risk_modifier

    acc = mt5.account_info()
    if not acc:
        logger.error('Cannot get account info for lot sizing')
        return 0.0

    risk_amount = acc.balance * risk_pct / 100.0
    sym_info = mt5.symbol_info(symbol)
    if not sym_info:
        logger.error(f'Cannot get symbol info for {symbol}')
        return 0.0

    tick_size = sym_info.trade_tick_size
    tick_value = sym_info.trade_tick_value  # In account currency per tick per lot
    if tick_size <= 0 or tick_value <= 0:
        logger.error(f'{symbol}: Invalid tick_size={tick_size} or tick_value={tick_value}')
        return 0.0

    sl_ticks = sl_distance / tick_size
    if sl_ticks <= 0:
        return 0.0

    # First pass: calculate lot without commission
    lot = risk_amount / (sl_ticks * tick_value)

    # Deduct commission from risk budget and recalculate
    # Commission is per lot per round turn (open+close)
    commission_per_lot = CONFIG.get('commission_per_lot_round', 0.0)
    if commission_per_lot > 0 and lot > 0:
        total_commission = lot * commission_per_lot
        adjusted_risk = risk_amount - total_commission
        if adjusted_risk > 0:
            lot = adjusted_risk / (sl_ticks * tick_value)
        else:
            logger.warning(f'{symbol}: Commission ${total_commission:.2f} exceeds risk budget ${risk_amount:.2f}')
            return 0.0

    # Clamp to broker limits and round to volume step
    vol_min = sym_info.volume_min
    vol_max = sym_info.volume_max
    vol_step = sym_info.volume_step

    if vol_step > 0:
        lot = round(lot / vol_step) * vol_step
    lot = max(vol_min, min(vol_max, lot))
    lot = round(lot, 6)  # Avoid floating point artifacts

    return lot


# ============================================================================
# DAILY P&L TRACKING
# ============================================================================
def get_daily_pnl():
    """Calculate today's realized + unrealized P&L."""
    now = datetime.now(timezone.utc)
    today = now.date()

    with pnl_lock:
        if daily_pnl['date'] != today:
            daily_pnl['date'] = today
            daily_pnl['realized'] = 0.0
            daily_pnl['closed_tickets'] = set()

    # Get today's closed deals (our magic number only)
    # MT5 on Windows needs naive datetimes (no tzinfo)
    from_time = datetime(today.year, today.month, today.day)
    now_naive = datetime.now()
    deals = mt5.history_deals_get(from_time, now_naive)
    realized = 0.0
    if deals:
        for deal in deals:
            if deal.magic == MAGIC_NUMBER and deal.entry == mt5.DEAL_ENTRY_OUT:
                realized += deal.profit + deal.swap + deal.commission

    # Unrealized from open positions
    unrealized = 0.0
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            if pos.magic == MAGIC_NUMBER:
                unrealized += pos.profit + pos.swap

    return realized + unrealized


def check_daily_limit():
    """Check if daily loss limit has been breached."""
    limit_pct = CONFIG.get('daily_loss_limit_pct', 5.0)
    acc = mt5.account_info()
    if not acc:
        return False

    daily = get_daily_pnl()
    limit = acc.balance * limit_pct / 100.0
    if daily < -limit:
        logger.warning(f'[RISK] Daily loss ${daily:.2f} exceeds limit -${limit:.2f} — NO NEW TRADES')
        return True
    return False


# ============================================================================
# ORDER EXECUTION
# ============================================================================
def get_our_positions():
    """Get all open positions with our magic number."""
    positions = mt5.positions_get()
    if positions is None:
        return []
    return [p for p in positions if p.magic == MAGIC_NUMBER]


def has_position(symbol):
    """Check if we already have ANY position on this symbol (ours or manual).
    Also checks all positions (not just our magic) to prevent hedge traps."""
    # Check ALL positions on this symbol (not just our magic number)
    all_positions = mt5.positions_get(symbol=symbol)
    if all_positions and len(all_positions) > 0:
        return True
    # Also check our tracked positions in state (covers race condition where
    # MT5 hasn't registered the position yet)
    with state_lock:
        for ticket, info in state.get('known_positions', {}).items():
            if info.get('symbol') == symbol:
                return True
    return False


def count_positions():
    """Count our open positions."""
    return len(get_our_positions())


def set_cooldown(symbol):
    """Set cooldown timer for a symbol after trade close."""
    minutes = CONFIG.get('cooldown_minutes', 30)
    expires = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    with cooldown_lock:
        cooldowns[symbol] = expires
    # Persist cooldowns
    with state_lock:
        state['cooldowns'][symbol] = expires.isoformat()
    save_state()


def is_on_cooldown(symbol):
    """Check if symbol is on cooldown."""
    with cooldown_lock:
        expires = cooldowns.get(symbol)
        if expires and datetime.now(timezone.utc) < expires:
            return True
    return False


def execute_trade(symbol, direction, entry, sl, tp, trigger, lot, indicators, ml_score, news_data):
    """Place a market order with SL and TP."""
    sym_info = mt5.symbol_info(symbol)
    if not sym_info:
        logger.error(f'Cannot execute: {symbol} not found')
        return False

    # Get live price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f'Cannot get tick for {symbol}')
        return False

    if direction == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    # Spread check — don't enter if spread is too wide (> 30% of SL distance)
    spread_price = abs(tick.ask - tick.bid)
    sl_dist = abs(entry - sl)
    if sl_dist > 0 and spread_price / sl_dist > 0.30:
        logger.warning(f'[EXEC] Spread too wide for {symbol}: {spread_price:.5g} > 30% of SL {sl_dist:.5g}')
        return False

    # Adjust SL/TP relative to actual fill price (not signal price)
    tp_dist = abs(entry - tp)
    if direction == 'BUY':
        actual_sl = price - sl_dist
        actual_tp = price + tp_dist
    else:
        actual_sl = price + sl_dist
        actual_tp = price - tp_dist

    # Enforce broker minimum stop distance
    min_stop_pts = sym_info.trade_stops_level  # In points
    min_stop_price = min_stop_pts * sym_info.point
    if min_stop_price > 0:
        if abs(price - actual_sl) < min_stop_price:
            # Push SL out to minimum distance
            if direction == 'BUY':
                actual_sl = price - min_stop_price
            else:
                actual_sl = price + min_stop_price
            logger.info(f'[EXEC] {symbol}: SL adjusted to min stop level {min_stop_pts} pts')
        if abs(price - actual_tp) < min_stop_price:
            if direction == 'BUY':
                actual_tp = price + min_stop_price
            else:
                actual_tp = price - min_stop_price

    # Round to symbol digits
    digits = sym_info.digits
    actual_sl = round(actual_sl, digits)
    actual_tp = round(actual_tp, digits)
    price = round(price, digits)

    # Build order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": actual_sl,
        "tp": actual_tp,
        "deviation": 30,  # Slippage tolerance in points
        "magic": MAGIC_NUMBER,
        "comment": f"KC_{trigger}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Try IOC → FOK → RETURN filling modes (broker-dependent)
    result = mt5.order_send(request)
    if result is None:
        logger.error(f'[EXEC] order_send returned None for {symbol}')
        return False

    if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)
        if result is None or result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            if result is None:
                logger.error(f'[EXEC] All fill modes failed for {symbol}')
                return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f'[EXEC] Order failed for {symbol}: {result.retcode} — {result.comment}')
        return False

    actual_risk_pct = CONFIG.get('risk_percent', 2.0)
    # Some brokers return 0 for result.price — use the requested price as fallback
    fill_price = result.price if result.price > 0 else price
    logger.info(f'[EXEC] ✅ {direction} {symbol} | Lot: {lot} | Entry: {fill_price} | SL: {actual_sl} | TP: {actual_tp}')

    # Record in persistent state
    record_trade_open(result.order, symbol, direction, fill_price, actual_sl, actual_tp, trigger, lot)

    # Set cooldown immediately to prevent opposite trade on next scan
    set_cooldown(symbol)

    # Notify via Telegram
    notify_entry(symbol, direction, trigger, fill_price, actual_sl, actual_tp, lot,
                 actual_risk_pct, indicators, ml_score, news_data)
    return True


def close_position(position, reason='manual'):
    """Close a specific position."""
    sym_info = mt5.symbol_info(position.symbol)
    if not sym_info:
        return False

    tick = mt5.symbol_info_tick(position.symbol)
    if not tick:
        return False

    if position.type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": 30,
        "magic": MAGIC_NUMBER,
        "comment": f"KC_close_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        return False
    if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)
        if result is None or result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            if result is None:
                return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        pnl = position.profit + position.swap
        direction = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
        logger.info(f'[CLOSE] {position.symbol} | {reason} | P&L: {pnl:+.2f} CAD')

        # Record in persistent state
        record_trade_close(position.ticket, position.symbol, direction,
                          position.price_open, price, pnl, reason)

        notify_exit(position.symbol, direction, position.price_open, price, pnl, reason)
        set_cooldown(position.symbol)
        return True
    return False


def modify_sl(position, new_sl):
    """Modify SL for an existing position (trailing stop)."""
    sym_info = mt5.symbol_info(position.symbol)
    if not sym_info:
        return False
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": round(new_sl, sym_info.digits),
        "tp": position.tp,
        "magic": MAGIC_NUMBER,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    if result:
        logger.debug(f'[TRAIL] SL modify failed {position.symbol}: {result.retcode} — {result.comment}')
    return False


# ============================================================================
# POSITION MONITOR — Trailing Stops
# ============================================================================
def check_closed_positions():
    """Detect positions that closed via SL/TP (not by our close_position function).
    Record them in state and log ML training data."""
    with state_lock:
        known = dict(state.get('known_positions', {}))

    if not known:
        return

    # Get current MT5 positions
    mt5_positions = get_our_positions()
    mt5_tickets = {str(p.ticket) for p in mt5_positions}

    for ticket_str, info in known.items():
        if ticket_str not in mt5_tickets:
            # Position closed! Find the deal
            deal = find_closing_deal(ticket_str)
            if deal:
                pnl, _reason = finalize_closed_position(ticket_str, info, deal, 'MONITOR')

                # Log ML training data
                outcome = 'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'BREAKEVEN'
                rates = mt5.copy_rates_from_pos(info['symbol'], mt5.TIMEFRAME_M5, 0, 200)
                if rates is not None and len(rates) >= 60:
                    log_ml_training_data(info['symbol'], rates, info['direction'],
                                       info['entry'], info.get('sl', 0), info.get('tp', 0),
                                       pnl, outcome)
            else:
                age_seconds, should_log = note_pending_close(ticket_str)
                if should_log:
                    logger.info(
                        f'[MONITOR] {info["symbol"]} #{ticket_str} is closed but broker history is still pending '
                        f'({age_seconds:.0f}s). Will retry until exit P&L is available.'
                    )

                if age_seconds >= 6 * 3600:
                    logger.warning(
                        f'[MONITOR] {info["symbol"]} #{ticket_str} close history is still missing after '
                        f'{age_seconds / 60:.0f} minutes — removing from active state without realized P&L.'
                    )
                    with state_lock:
                        state['known_positions'].pop(ticket_str, None)
                    save_state()
                    clear_pending_close(ticket_str)


def position_monitor():
    """Background thread: manage trailing stops + detect SL/TP closes."""
    logger.info('[MONITOR] Position monitor started')

    while not shutdown_event.is_set():
        try:
            # Check for positions closed by SL/TP
            check_closed_positions()

            positions = get_our_positions()
            for pos in positions:
                symbol = pos.symbol
                cfg = CONFIG.get('symbols', {}).get(symbol)
                if not cfg:
                    continue

                # Get current ATR for dynamic trail distances
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 50)
                if rates is None or len(rates) < 20:
                    continue

                closes = np.array([r[4] for r in rates], dtype=np.float64)
                highs_arr = np.array([r[2] for r in rates], dtype=np.float64)
                lows_arr = np.array([r[3] for r in rates], dtype=np.float64)
                curr_atr = safe_last(ATR_GPU(highs_arr, lows_arr, closes, cfg.get('atr_period', 14)))

                breakeven_dist = curr_atr * cfg.get('breakeven_atr_mult', 1.0)
                trail_start_dist = curr_atr * cfg.get('trail_start_atr_mult', 1.5)
                trail_distance = curr_atr * cfg.get('trail_distance_atr_mult', 1.2)

                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    continue

                entry = pos.price_open
                current_sl = pos.sl
                is_buy = pos.type == mt5.ORDER_TYPE_BUY

                if is_buy:
                    current_price = tick.bid
                    pnl_dist = current_price - entry
                else:
                    current_price = tick.ask
                    pnl_dist = entry - current_price

                # Stage 1: Move to breakeven at 1.0x ATR profit
                if pnl_dist >= breakeven_dist:
                    be_sl = entry + (curr_atr * 0.1) if is_buy else entry - (curr_atr * 0.1)
                    # Only move SL forward, never backward
                    if is_buy and current_sl < be_sl:
                        if modify_sl(pos, be_sl):
                            logger.info(f'[TRAIL] {symbol} → Breakeven SL: {fmt_price(be_sl)}')
                    elif not is_buy and (current_sl > be_sl or current_sl == 0):
                        if modify_sl(pos, be_sl):
                            logger.info(f'[TRAIL] {symbol} → Breakeven SL: {fmt_price(be_sl)}')

                # Stage 2: Start trailing at trail_start ATR
                if pnl_dist >= trail_start_dist:
                    if is_buy:
                        ideal_sl = current_price - trail_distance
                        if ideal_sl > current_sl:
                            if modify_sl(pos, ideal_sl):
                                locked = fmt_pts(symbol, ideal_sl - entry)
                                logger.info(f'[TRAIL] {symbol} ↑ SL: {fmt_price(ideal_sl)} (lock +{locked:,} pts)')
                    else:
                        ideal_sl = current_price + trail_distance
                        if ideal_sl < current_sl or current_sl == 0:
                            if modify_sl(pos, ideal_sl):
                                locked = fmt_pts(symbol, entry - ideal_sl)
                                logger.info(f'[TRAIL] {symbol} ↓ SL: {fmt_price(ideal_sl)} (lock +{locked:,} pts)')

        except Exception as e:
            logger.error(f'[MONITOR] Error: {e}')

        # Check every 5 seconds
        for _ in range(10):
            if shutdown_event.is_set():
                return
            time.sleep(0.5)


# ============================================================================
# WEEKEND FILTER
# ============================================================================
def is_weekend():
    """Check if markets are closed (Saturday or Sunday UTC).
    Forex/Metals: closed from Friday 22:00 UTC to Sunday 22:00 UTC.
    Crypto (BTC): often 24/7 but many brokers close weekends.
    We skip all trading on Saturday and most of Sunday to be safe.
    """
    now = datetime.now(timezone.utc)
    day = now.weekday()  # 0=Mon, 5=Sat, 6=Sun
    hour = now.hour
    # Saturday all day = closed
    if day == 5:
        return True
    # Sunday before 22:00 UTC = closed (most brokers open at 22:00 Sun)
    if day == 6 and hour < 22:
        return True
    # Friday after 22:00 UTC = closing (some brokers already closed)
    if day == 4 and hour >= 22:
        return True
    return False


def normalize_symbol(symbol):
    """Strip broker-specific suffixes so symbol rules can match consistently."""
    return canonical_symbol(symbol)


def is_weekend_closed_for_symbol(symbol):
    """BTC keeps scanning on weekends; other configured markets stay paused."""
    if not is_weekend():
        return False
    return normalize_symbol(symbol) != 'BTCUSD'


# ============================================================================
# CIRCUIT BREAKER — Pause after consecutive losses
# ============================================================================
def check_circuit_breaker():
    """Check if circuit breaker is active (3 consecutive losses → pause 1 hour)."""
    global circuit_breaker_until
    with circuit_breaker_lock:
        if circuit_breaker_until is not None:
            if datetime.now(timezone.utc) < circuit_breaker_until:
                remaining = (circuit_breaker_until - datetime.now(timezone.utc)).total_seconds() / 60
                return True, f'Circuit breaker active — {remaining:.0f}min remaining ({consecutive_losses} consecutive losses)'
            else:
                # Breaker expired, reset
                circuit_breaker_until = None
                return False, ''
        return False, ''


def update_circuit_breaker(is_win):
    """Update consecutive loss counter after a trade closes."""
    global consecutive_losses, circuit_breaker_until
    max_consecutive = CONFIG.get('circuit_breaker_losses', 3)
    pause_minutes = CONFIG.get('circuit_breaker_pause_minutes', 60)

    with circuit_breaker_lock:
        if is_win:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            logger.warning(f'[CIRCUIT] Consecutive losses: {consecutive_losses}/{max_consecutive}')
            if consecutive_losses >= max_consecutive:
                circuit_breaker_until = datetime.now(timezone.utc) + timedelta(minutes=pause_minutes)
                logger.warning(f'[CIRCUIT] ⚠️ {consecutive_losses} consecutive losses — PAUSING for {pause_minutes} minutes')
                send_telegram(
                    '⚠️ CIRCUIT BREAKER',
                    f'{consecutive_losses} consecutive losses\nTrading paused for {pause_minutes} minutes\nBot will resume automatically'
                )


# ============================================================================
# MT5 RECONNECTION
# ============================================================================
def ensure_mt5_connected():
    """Check MT5 connection and reconnect if needed. Returns True if connected."""
    try:
        # Quick check: try to get account info
        acc = mt5.account_info()
        if acc is not None:
            return True
    except Exception:
        pass

    logger.warning('[MT5] Connection lost — attempting reconnect...')
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            mt5.shutdown()
            time.sleep(2)
            mt5_path = CONFIG.get('mt5_path', '')
            if mt5_path:
                ok = mt5.initialize(path=mt5_path)
            else:
                ok = mt5.initialize()

            if ok:
                acc = mt5.account_info()
                acc_num = CONFIG.get('account_number')
                # Only login if not already on correct account
                if not acc or acc.login != acc_num:
                    password = CONFIG.get('password', '')
                    server = CONFIG.get('server', '')
                    if acc_num and password:
                        mt5.login(acc_num, password, server)

                acc = mt5.account_info()
                if acc:
                    logger.info(f'[MT5] Reconnected on attempt {attempt} — Account {acc.login} | Equity: ${acc.equity:,.2f}')
                    send_telegram('🔄 MT5 Reconnected', f'Attempt {attempt}\nEquity: ${acc.equity:,.2f}')
                    # Re-enable symbols
                    for symbol in CONFIG.get('symbols', {}):
                        mt5.symbol_select(symbol, True)
                    return True
        except Exception as e:
            logger.error(f'[MT5] Reconnect attempt {attempt} failed: {e}')

        time.sleep(5 * attempt)  # Backoff: 5s, 10s, 15s

    logger.error('[MT5] All reconnect attempts failed')
    send_telegram('🔴 MT5 DISCONNECTED', f'Failed to reconnect after {max_retries} attempts\nBot is blind — positions have SL/TP set')
    return False


# ============================================================================
# SIGNAL DEDUP — Only signal once per M5 candle per symbol
# ============================================================================
def reserve_signal_candle(dedup_key, candle_time):
    """Reserve the current candle so we do not re-signal on the same bar."""
    with signal_dedup_lock:
        if _last_signal_candle.get(dedup_key, 0) == candle_time:
            return False
        _last_signal_candle[dedup_key] = candle_time
        return True


# ============================================================================
# END-OF-DAY SUMMARY
# ============================================================================
def send_daily_summary():
    """Send end-of-day Telegram summary."""
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    with state_lock:
        day = state.get('daily_stats', {}).get(today)
        total = state['total_trades']
        total_pnl = state['total_realized']

    if day and day['trades'] > 0:
        wr = (day['wins'] / day['trades'] * 100) if day['trades'] > 0 else 0
        emoji = '🟢' if day['realized'] >= 0 else '🔴'
        send_telegram(
            f'📊 Daily Summary — {today}',
            f"{emoji} P&L: ${day['realized']:+.2f} CAD\n"
            f"Trades: {day['trades']} ({day['wins']}W / {day['losses']}L | {wr:.0f}%)\n"
            f"─────────────\n"
            f"Lifetime: {total} trades | ${total_pnl:+,.2f} CAD"
        )


# ============================================================================
# SESSION FILTER
# ============================================================================
def is_in_session(symbol, cfg):
    """Check if current time is within the symbol's trading session."""
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour + now_utc.minute / 60.0
    start = cfg.get('session_start_utc', 0)
    end = cfg.get('session_end_utc', 24)
    if start < end:
        return start <= hour < end
    else:  # Wraps around midnight
        return hour >= start or hour < end


# ============================================================================
# SIGNAL CSV LOGGING
# ============================================================================
def log_signal(symbol, signal, indicators, entry, sl, tp, ml_score, executed, lot):
    """Log signal to daily CSV for analysis."""
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    log_dir = os.path.join('ml_data', 'auto_trades')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'signals_{date_str}.csv')

    row = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': symbol,
        'direction': signal.get('direction', ''),
        'trigger': signal.get('trigger', ''),
        'confidence': signal.get('confidence', 0),
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'lot': lot,
        'executed': executed,
        'rsi': indicators.get('rsi', ''),
        'adx': indicators.get('adx', ''),
        'h1_trend': indicators.get('h1_trend', ''),
        'ema_fast': indicators.get('ema_fast', ''),
        'ema_slow': indicators.get('ema_slow', ''),
        'atr_pts': indicators.get('atr_pts', ''),
        'spread': indicators.get('spread', ''),
        'volume_ratio': indicators.get('volume_ratio', ''),
        'ml_action': ml_score['action'] if ml_score else '',
        'ml_confidence': ml_score['confidence'] if ml_score else '',
        'news_sentiment': indicators.get('news_sentiment', ''),
    }

    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_ml_training_data(symbol, rates, direction, entry, sl, tp, pnl, outcome):
    """Log completed trade data for future ML training.
    Each row = one completed trade with full indicator snapshot + P&L outcome.
    This is the gold mine for training better models later.
    """
    log_dir = os.path.join('ml_data', 'training')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{symbol}_training_data.csv')

    try:
        cfg = CONFIG.get('symbols', {}).get(symbol, {})
        opens = np.array([r[1] for r in rates], dtype=np.float64)
        closes = np.array([r[4] for r in rates], dtype=np.float64)
        highs = np.array([r[2] for r in rates], dtype=np.float64)
        lows = np.array([r[3] for r in rates], dtype=np.float64)
        volumes = np.array([r[5] for r in rates], dtype=np.float64)

        ema8 = float(EMA_GPU(closes, 8)[-1])
        ema20 = float(EMA_GPU(closes, 20)[-1])
        ema50 = float(EMA_GPU(closes, 50)[-1]) if len(closes) >= 55 else 0
        rsi = float(get_signal_rsi(RSI_GPU(closes, 14)) or 0) if len(closes) >= 20 else 0
        atr = float(safe_last(ATR_GPU(highs, lows, closes, 14)))
        adx_val, _, _ = ADX(highs, lows, closes, 14)

        avg_vol = float(np.mean(volumes[-50:])) if len(volumes) >= 50 else float(np.mean(volumes))
        vol_ratio = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1.0

        row = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'direction': direction,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'pnl': round(pnl, 4),
            'outcome': outcome,  # WIN, LOSS, BREAKEVEN
            'close': float(closes[-1]),
            'ema8': round(ema8, 6),
            'ema20': round(ema20, 6),
            'ema50': round(ema50, 6),
            'rsi': round(rsi, 2),
            'atr': round(atr, 6),
            'adx': round(adx_val, 2),
            'volume_ratio': round(vol_ratio, 2),
            'ema8_ema20_diff': round((ema8 - ema20) / atr, 4) if atr > 0 else 0,
            'close_ema20_diff': round((float(closes[-1]) - ema20) / atr, 4) if atr > 0 else 0,
            'h1_trend': get_h1_trend(symbol, cfg),
            'spread': mt5.symbol_info(symbol).spread if mt5.symbol_info(symbol) else 0,
            'bar_range': round(float(highs[-1] - lows[-1]) / atr, 4) if atr > 0 else 0,
            'bar_body': round(abs(float(closes[-1] - opens[-1])) / atr, 4) if atr > 0 else 0,
        }

        file_exists = os.path.exists(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        logger.debug(f'[ML_DATA] Error logging training data: {e}')


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================
def print_scan(symbol, status, detail, rsi, atr_pts, adx, h1_trend, spread, positions=0):
    ts = datetime.now().strftime('%H:%M:%S')
    pos_tag = f' | POS:{positions}' if positions > 0 else ''
    print(f'[{ts}] {symbol:<12} | {status} | {detail} | RSI {rsi:.0f} | ATR {atr_pts:,} | ADX {adx:.0f} | H1: {h1_trend} | Spr: {spread}{pos_tag}')


def print_signal_box(symbol, direction, trigger, confidence, entry, sl, tp,
                     indicators, ml_score, news_data, lot, executed):
    pv = CONFIG.get('symbols', {}).get(symbol, {}).get('point_value', 0.01)
    sl_pts = fmt_pts(symbol, abs(entry - sl))
    tp_pts = fmt_pts(symbol, abs(tp - entry))
    atr_pts = indicators.get('atr_pts', 0)

    ml_line = ''
    if ml_score:
        agrees = 'AGREES ✅' if ml_score['action'] == direction else 'DISAGREES ❌'
        ml_line = f'\n  ML: {ml_score["action"]} ({ml_score["confidence"]:.0%}) — {agrees}'

    news_line = ''
    if news_data:
        ns = news_data.get('sentiment', 'N/A')
        ns_emoji = '🟢' if ns == 'BULLISH' else '🔴' if ns == 'BEARISH' else '⚪'
        news_line = f'\n  News: {ns_emoji} {ns} — {news_data.get("summary", "")[:50]}'

    exec_tag = f'✅ EXECUTED (lot {lot})' if executed else '⛔ SKIPPED'

    print(f'''
{'='*62}
  {direction} SIGNAL — {symbol} — {trigger} — {exec_tag}
{'='*62}
  Entry:    {fmt_price(entry)}
  SL:       {fmt_price(sl)}  (-{sl_pts:,} pts)
  TP:       {fmt_price(tp)}  (+{tp_pts:,} pts | 3:1 R:R)
  Lot:      {lot}
  {'-'*58}
  RSI: {indicators.get("rsi", 0):.1f} | ATR: {atr_pts:,} pts | ADX: {indicators.get("adx", 0):.1f} | H1: {indicators.get("h1_trend", "?")}{ml_line}{news_line}
{'='*62}''')


# ============================================================================
# MAIN SCANNING LOGIC
# ============================================================================
def scan_symbol(symbol, cfg):
    """Scan one symbol. If all checks pass, auto-execute."""
    try:
        # Weekend check — allow BTC to keep scanning when other markets are shut.
        if is_weekend_closed_for_symbol(symbol):
            return

        # Session check
        if not is_in_session(symbol, cfg):
            return

        # Circuit breaker check
        breaker_active, breaker_msg = check_circuit_breaker()
        if breaker_active:
            return

        # Cooldown check
        if is_on_cooldown(symbol):
            return

        # Already have position on this symbol?
        if has_position(symbol):
            return

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 200)
        if rates is None or len(rates) < 60:
            return

        opens = np.array([r[1] for r in rates], dtype=np.float64)
        closes = np.array([r[4] for r in rates], dtype=np.float64)
        highs = np.array([r[2] for r in rates], dtype=np.float64)
        lows = np.array([r[3] for r in rates], dtype=np.float64)
        volumes = np.array([r[5] for r in rates], dtype=np.float64)

        # Compute indicators
        ema_pullback_vals = EMA_GPU(closes, 20)
        ema_slow = EMA_GPU(closes, cfg.get('ema_slow', 8))
        ema_fast_vals = EMA_GPU(closes, cfg.get('ema_fast', 3))
        rsi_vals = RSI_GPU(closes, cfg.get('rsi_period', 14))
        curr_atr = safe_last(ATR_GPU(highs, lows, closes, cfg.get('atr_period', 14)))
        adx_val, plus_di, minus_di = ADX(highs, lows, closes, cfg.get('adx_period', 14))
        curr_rsi = get_signal_rsi(rsi_vals) or 50.0

        sym_info = mt5.symbol_info(symbol)
        spread = sym_info.spread if sym_info else 0
        current_price = float(closes[-1])
        pv = cfg.get('point_value', 0.01)
        atr_pts = int(round(curr_atr / pv))

        # ── Spread check (pre-filter before expensive checks) ──
        max_spread = cfg.get('max_spread_points', 0)
        if max_spread > 0 and spread > max_spread:
            h1_trend = get_h1_trend(symbol, cfg) if CONFIG.get('h1_trend_filter', True) else 'BOTH'
            print_scan(symbol, 'HOLD', f'Spread {spread} > max {max_spread}', curr_rsi, atr_pts, adx_val, h1_trend, spread)
            return

        # ── CHECK 2: H1 trend ──
        h1_trend = get_h1_trend(symbol, cfg) if CONFIG.get('h1_trend_filter', True) else 'BOTH'

        # ── CHECK 3: ADX filter ──
        min_adx = max(cfg.get('min_adx', 25), CONFIG.get('global_min_adx', 25))
        if adx_val < min_adx:
            print_scan(symbol, 'HOLD', f'Choppy (ADX {adx_val:.0f} < {min_adx})', curr_rsi, atr_pts, adx_val, h1_trend, spread)
            return

        # ── CHECK 1: Trigger ──
        signal = None
        indicators = {}

        # MOMENTUM first (higher profit factor)
        sig, ind = check_momentum(opens, closes, highs, lows, volumes, rsi_vals, curr_atr, cfg)
        if sig and (h1_trend == 'BOTH' or h1_trend == sig['direction']):
            signal = sig
            indicators = ind

        if signal is None:
            sig, ind = check_ema_pullback(opens, closes, highs, lows, ema_pullback_vals, rsi_vals, curr_atr, cfg)
            if sig and (h1_trend == 'BOTH' or h1_trend == sig['direction']):
                signal = sig
                indicators = ind

        if signal is None:
            # No trigger — print scan status
            ema20 = float(ema_pullback_vals[-1])
            ema_fast_curr = float(ema_fast_vals[-1])
            ema_slow_curr = float(ema_slow[-1])
            if abs(current_price - ema20) / curr_atr < 0.7:
                detail = f'Pullback zone near {fmt_price(ema20)}'
            elif abs(ema_fast_curr - ema_slow_curr) < curr_atr * 0.3:
                detail = f'EMAs converging'
            else:
                detail = f'Waiting'
            print_scan(symbol, 'HOLD', detail, curr_rsi, atr_pts, adx_val, h1_trend, spread)
            return

        direction = signal['direction']

        # ── CHECK 4: RSI filter ──
        rsi_min = CONFIG.get('global_rsi_min', 30)
        rsi_max = CONFIG.get('global_rsi_max', 70)
        if curr_rsi < rsi_min or curr_rsi > rsi_max:
            print_scan(symbol, 'SKIP', f'RSI {curr_rsi:.0f} out of range [{rsi_min}-{rsi_max}]',
                       curr_rsi, atr_pts, adx_val, h1_trend, spread)
            return

        # ── CHECK 5: ML filter (symbol-specific) ──
        ml_mode = cfg.get('ml_mode', 'none')
        ml_score = get_ml_score(symbol, rates)
        risk_modifier = 1.0

        if ml_mode == 'required':
            # XAU: ML must agree
            if ml_score is None:
                print_scan(symbol, 'SKIP', 'ML model not available (required)', curr_rsi, atr_pts, adx_val, h1_trend, spread)
                return
            if ml_score['action'] != direction:
                print_scan(symbol, 'SKIP', f'ML {ml_score["action"]} disagrees (required)',
                           curr_rsi, atr_pts, adx_val, h1_trend, spread)
                return
        elif ml_mode == 'soft':
            # BTC: ML disagree → reduce lot to 1%
            if ml_score and ml_score['action'] != direction and ml_score['action'] != 'HOLD':
                risk_modifier *= 0.5
                logger.info(f'[ML] {symbol}: ML {ml_score["action"]} disagrees — reducing risk to {CONFIG.get("risk_percent", 2.0) * 0.5:.1f}%')

        # ── CHECK 6: News filter ──
        news_data = get_news_for_symbol(symbol)
        news_allowed, news_modifier, news_reason = news_allows_trade(symbol, direction)
        if not news_allowed:
            print_scan(symbol, 'SKIP', news_reason, curr_rsi, atr_pts, adx_val, h1_trend, spread)
            log_signal(symbol, signal, indicators, current_price, 0, 0, ml_score, False, 0)
            return
        risk_modifier *= news_modifier

        # ── Max positions check ──
        max_pos = CONFIG.get('max_positions', 3)
        if count_positions() >= max_pos:
            print_scan(symbol, 'SKIP', f'Max {max_pos} positions reached', curr_rsi, atr_pts, adx_val, h1_trend, spread)
            return

        # ── Daily loss limit check ──
        if check_daily_limit():
            return

        # ═══ ALL CHECKS PASSED — CALCULATE & EXECUTE ═══
        sl_dist = curr_atr * cfg.get('sl_atr_mult', 1.0)
        tp_dist = curr_atr * cfg.get('tp_atr_mult', 3.0)

        # Enforce minimum stop distance (config-defined per symbol)
        min_stop_pts = cfg.get('min_stop_points', 0)
        if min_stop_pts > 0:
            min_stop_price = min_stop_pts * pv
            if sl_dist < min_stop_price:
                sl_dist = min_stop_price
                tp_dist = sl_dist * 3.0  # Maintain 3:1 R:R

        if direction == 'BUY':
            sl = current_price - sl_dist
            tp = current_price + tp_dist
        else:
            sl = current_price + sl_dist
            tp = current_price - tp_dist

        # Calculate lot size
        lot = calculate_lot_size(symbol, sl_dist, risk_modifier)
        if lot <= 0:
            logger.error(f'[EXEC] Lot calculation returned 0 for {symbol}')
            return

        candle_time = int(rates[-1]['time'])
        dedup_key = f'{symbol}_{direction}'
        if not reserve_signal_candle(dedup_key, candle_time):
            return

        # Build full indicators dict
        indicators.update({
            'atr_pts': atr_pts,
            'adx': adx_val,
            'h1_trend': h1_trend,
            'spread': spread,
            'ema_fast': float(ema_fast_vals[-1]),
            'ema_slow': float(ema_slow[-1]),
            'news_sentiment': news_data.get('sentiment', 'N/A') if news_data else 'N/A',
        })
        if 'rsi' not in indicators or indicators['rsi'] is None:
            indicators['rsi'] = curr_rsi
        if 'volume_ratio' not in indicators:
            indicators['volume_ratio'] = 0

        # Execute the trade
        executed = execute_trade(symbol, direction, current_price, sl, tp,
                                signal['trigger'], lot, indicators, ml_score, news_data)

        # Print signal box
        print_signal_box(symbol, direction, signal['trigger'], signal['confidence'],
                         current_price, sl, tp, indicators, ml_score, news_data,
                         lot, executed)

        # Log to CSV
        log_signal(symbol, signal, indicators, current_price, sl, tp, ml_score, executed, lot)

    except Exception as e:
        logger.error(f'[SCAN] Error scanning {symbol}: {e}')
        traceback.print_exc()


# ============================================================================
# STATUS DISPLAY
# ============================================================================
def print_status():
    """Print current positions and daily P&L summary."""
    positions = get_our_positions()
    daily = get_daily_pnl()
    acc = mt5.account_info()

    ts = datetime.now().strftime('%H:%M:%S')
    bal = acc.balance if acc else 0
    eq = acc.equity if acc else 0

    # Update high water mark
    if eq > 0:
        update_high_water(eq)

    if positions:
        pos_lines = []
        for p in positions:
            direction = 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL'
            pnl = p.profit + p.swap
            pos_lines.append(f'  {direction} {p.symbol} | Lot: {p.volume} | Entry: {fmt_price(p.price_open)} | SL: {fmt_price(p.sl)} | P&L: {pnl:+.2f}')
        positions_str = '\n'.join(pos_lines)
    else:
        positions_str = '  No open positions'

    print(f'\n{"─"*62}')
    print(f'  [{ts}] Balance: ${bal:,.2f} CAD | Equity: ${eq:,.2f} | Daily P&L: {daily:+.2f}')
    print(f'  Open positions ({len(positions)}/{CONFIG.get("max_positions", 3)}):')
    print(positions_str)

    # Lifetime stats summary line
    with state_lock:
        total = state['total_trades']
        wins = state['total_wins']
        realized = state['total_realized']
    if total > 0:
        wr = (wins / total * 100)
        print(f'  Lifetime: {total} trades | {wins}W ({wr:.0f}%) | ${realized:+,.2f} total P&L')

    # News status
    with news_lock:
        if news_cache:
            news_parts = []
            for group, data in news_cache.items():
                emoji = '🟢' if data['sentiment'] == 'BULLISH' else '🔴' if data['sentiment'] == 'BEARISH' else '⚪'
                news_parts.append(f'{emoji}{group}')
            print(f'  News: {" | ".join(news_parts)}')

    print(f'{"─"*62}\n')


# ============================================================================
# KEYBOARD INPUT (background thread for manual commands)
# ============================================================================
def keyboard_listener():
    """Listen for user commands."""
    while not shutdown_event.is_set():
        try:
            cmd = input().strip().lower()

            if cmd == 'q':
                print('\n  Shutting down...')
                shutdown_event.set()
                time.sleep(1)
                os._exit(0)

            elif cmd == 'status' or cmd == 's':
                print_status()

            elif cmd == 'news' or cmd == 'n':
                with news_lock:
                    if not news_cache:
                        print('  No news data yet')
                    else:
                        for group, data in news_cache.items():
                            emoji = '🟢' if data['sentiment'] == 'BULLISH' else '🔴' if data['sentiment'] == 'BEARISH' else '⚪'
                            print(f'  {emoji} {group}: {data["sentiment"]} — {data.get("summary", "")[:60]}')
                            for h in data.get('headlines', [])[:3]:
                                print(f'    • {h[:80]}')
                            print()

            elif cmd == 'refresh' or cmd == 'r':
                print('  Refreshing news...')
                threading.Thread(target=update_news_sentiment, daemon=True).start()

            elif cmd.startswith('close '):
                parts = cmd.split()
                if len(parts) >= 2:
                    sym = parts[1].upper()
                    positions = get_our_positions()
                    closed = False
                    for p in positions:
                        if sym in p.symbol or p.symbol in sym:
                            close_position(p, 'manual')
                            closed = True
                    if not closed:
                        print(f'  No position found for {sym}')

            elif cmd == 'closeall':
                positions = get_our_positions()
                for p in positions:
                    close_position(p, 'manual_closeall')
                print(f'  Closed {len(positions)} positions')

            elif cmd == 'stats':
                print_lifetime_stats()

            elif cmd == 'help' or cmd == 'h':
                print('\n  Commands:')
                print('    s / status     — Show positions & daily P&L')
                print('    stats          — Show lifetime performance stats')
                print('    n / news       — Show current news sentiment')
                print('    r / refresh    — Force refresh news')
                print('    close <symbol> — Close position on symbol')
                print('    closeall       — Close all positions')
                print('    q              — Quit\n')

            else:
                print('  Type "help" for commands')

        except EOFError:
            break
        except Exception as e:
            if not shutdown_event.is_set():
                print(f'  Input error: {e}')


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='INDIAN KING COBRA v2.0 — Auto-Execution Scalping Bot')
    parser.add_argument('--config', default='king_cobra_vt_raw.json', help='Config file path')
    args = parser.parse_args()

    load_config(args.config)

    # Load .env file if present
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

    symbols_list = [s for s, c in CONFIG.get('symbols', {}).items() if c.get('enabled', True)]

    # Banner
    print()
    print('  ╔════════════════════════════════════════════════════╗')
    print('  ║   INDIAN KING COBRA v2.0 — Auto Scalping Bot     ║')
    print('  ║   M5 | MOMENTUM + EMA_PULLBACK | 6-Check System  ║')
    print('  ║   Auto-Execute | ATR Trailing | News Layer        ║')
    print('  ╚════════════════════════════════════════════════════╝')
    print(f'  Account: {CONFIG.get("account_number")} | Server: {CONFIG.get("server")}')
    print(f'  Risk: {CONFIG.get("risk_percent", 2)}% per trade | Max: {CONFIG.get("max_positions", 3)} positions')
    print(f'  Symbols: {", ".join(symbols_list)}')
    print()

    # Connect to MT5
    mt5_path = CONFIG.get('mt5_path', '')
    if mt5_path:
        if not mt5.initialize(path=mt5_path):
            print(f'  MT5 init failed: {mt5.last_error()}')
            sys.exit(1)
    else:
        if not mt5.initialize():
            print(f'  MT5 init failed: {mt5.last_error()}')
            sys.exit(1)

    # Only login if not already on the correct account
    acc_num = CONFIG.get('account_number')
    acc_check = mt5.account_info()
    if acc_check and acc_check.login == acc_num:
        print(f'  Already logged in to account {acc_num} — skipping login')
    else:
        password = CONFIG.get('password', '')
        server = CONFIG.get('server', '')
        if acc_num and password:
            if not mt5.login(acc_num, password, server):
                print(f'  MT5 login failed: {mt5.last_error()}')

    acc = mt5.account_info()
    if acc:
        print(f'  Connected: Account {acc.login} | Balance: ${acc.balance:,.2f} {acc.currency} | Leverage: 1:{acc.leverage}')
        # Set starting balance if first run
        with state_lock:
            if state['starting_balance'] == 0:
                state['starting_balance'] = acc.balance
            update_high_water(acc.equity)
    else:
        print('  Warning: Could not get account info')

    # Load ML models (skip if not available — forex pairs don't use ML anyway)
    print('  Loading ML models...', flush=True)
    try:
        load_ml_models()
    except Exception as e:
        print(f'  ML load error (non-fatal): {e}', flush=True)
    print('  ML models done', flush=True)

    # Enable symbols
    print('  Enabling symbols...', flush=True)
    for symbol in CONFIG.get('symbols', {}):
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        if info:
            print(f'  ✅ {symbol} | point={info.point} | tick_val={info.trade_tick_value} | tick_size={info.trade_tick_size} | spread={info.spread}', flush=True)
        else:
            print(f'  ❌ {symbol} — not found on broker', flush=True)

    # ── PERSISTENCE: Restore state ──
    print('  Loading state...', flush=True)
    load_state()
    restore_cooldowns()
    restore_news_cache()

    # Reconcile: pick up positions from before restart, detect offline closes
    print('  Reconciling positions...', flush=True)
    reconcile_positions()

    # Show lifetime stats on startup
    print_lifetime_stats()

    symbols = {s: c for s, c in CONFIG.get('symbols', {}).items() if c.get('enabled', True)}
    interval = CONFIG.get('scan_interval_seconds', 10)

    print()
    print(f'  Scanning {len(symbols)} symbols every {interval}s')
    print(f'  Checks: Trigger → H1 → ADX≥25 → RSI[30-70] → ML → News')
    print(f'  Commands: status | stats | news | refresh | close <sym> | closeall | help | q')
    print(f'  {"─"*58}')
    print()

    # Start background threads
    # 1. Position monitor (trailing stops)
    monitor_thread = threading.Thread(target=position_monitor, daemon=True)
    monitor_thread.start()

    # 2. News update loop
    news_thread = threading.Thread(target=news_update_loop, daemon=True)
    news_thread.start()

    # 3. Keyboard listener
    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    # Print status every 5 minutes
    last_status = time.time()
    last_summary_date = None
    weekend_logged = False

    # Main scanning loop
    while not shutdown_event.is_set():
        try:
            # ── Weekend check ──
            if is_weekend():
                weekend_symbols = [symbol for symbol in symbols if not is_weekend_closed_for_symbol(symbol)]
                if weekend_symbols:
                    if not weekend_logged:
                        logger.info(f'[MAIN] Weekend mode — scanning only: {", ".join(weekend_symbols)}')
                        weekend_logged = True
                else:
                    if not weekend_logged:
                        logger.info('[MAIN] Market closed (weekend) — sleeping...')
                        weekend_logged = True
                    time.sleep(60)  # Check every minute during weekend
                    continue
            else:
                if weekend_logged:
                    logger.info('[MAIN] Market open — resuming scanning')
                    weekend_logged = False

            # ── MT5 connection check (every loop) ──
            if not ensure_mt5_connected():
                logger.error('[MAIN] MT5 not connected — waiting 30s before retry')
                time.sleep(30)
                continue

            # ── Circuit breaker status ──
            breaker_active, breaker_msg = check_circuit_breaker()
            if breaker_active:
                ts = datetime.now().strftime('%H:%M:%S')
                print(f'[{ts}] ⚠️ {breaker_msg}')
                time.sleep(30)  # Check less frequently during breaker
                continue

            # ── Scan all symbols ──
            for symbol, cfg in symbols.items():
                scan_symbol(symbol, cfg)

            # ── Print status every 5 minutes ──
            if time.time() - last_status > 300:
                print_status()
                last_status = time.time()

            # ── End-of-day summary at 21:00 UTC ──
            now_utc = datetime.now(timezone.utc)
            if now_utc.hour == 21 and now_utc.minute < 1:
                today_str = now_utc.strftime('%Y-%m-%d')
                if last_summary_date != today_str:
                    send_daily_summary()
                    last_summary_date = today_str

            time.sleep(interval)

        except KeyboardInterrupt:
            print('\n  Shutting down...')
            shutdown_event.set()
            # Send final summary before exit
            send_daily_summary()
            positions = get_our_positions()
            if positions:
                print(f'  {len(positions)} positions still open (not auto-closing)')
            mt5.shutdown()
            break
        except Exception as e:
            logger.error(f'[MAIN] Loop error: {e}')
            traceback.print_exc()
            time.sleep(5)


if __name__ == '__main__':
    main()
