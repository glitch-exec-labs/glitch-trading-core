"""
ORACLE v1 — Coordinator / Risk Overlay
Port 8070

NOT a trading bot. Monitors all 5 snake bots via their Flask APIs.
Detects conflicts, tracks aggregate risk, monitors currency correlation.

Features:
  1. Position aggregation — unified view of all bot positions
  2. Conflict detection — opposing positions on same symbol across bots
  3. Aggregate risk — total exposure, total lots, position count warnings
  4. Currency correlation — net exposure per currency (prevents 3x USD short)
  5. Global kill switch — stop all bots simultaneously
  6. Unified dashboard — single endpoint for all bot statuses

No MT5 connection. No account. Pure monitoring and coordination.
"""
import os
import json
import time
import threading
import logging
import argparse
from datetime import datetime, timezone
from flask import Flask, request, jsonify

import requests as http_requests

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {}
API_KEY = os.environ.get("ORACLE_API_KEY", "")

state_lock = threading.Lock()
stop_event = threading.Event()
_monitor_thread = None

# Latest state from monitoring loop
_latest_state = {
    'timestamp': None,
    'statuses': {},
    'positions': {},
    'conflicts': [],
    'correlation': {'exposure': {}, 'warnings': []},
    'risk': {'total_positions': 0, 'total_lots': 0.0, 'warnings': []}
}

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
    logger = logging.getLogger("ORACLE")
    logger.setLevel(logging.INFO)
    log_file = CONFIG.get('log_file', 'oracle.log')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ============================================================================
# UTILITY
# ============================================================================
def normalize_pos_type(t):
    if t in ('BUY', 'buy', 0): return 'BUY'
    if t in ('SELL', 'sell', 1): return 'SELL'
    return None


def build_simulated_position(bot_name, request_data):
    return {
        'symbol': request_data['symbol'],
        'type': request_data['direction'],
        'volume': float(request_data.get('volume', 0.0)),
        'ticket': 'SIMULATED',
        'bot': bot_name,
        'entry_price': request_data.get('entry_price'),
        'sl': request_data.get('sl'),
        'tp': request_data.get('tp'),
    }


def clone_positions(all_positions):
    return {
        bot_name: [dict(pos) for pos in positions]
        for bot_name, positions in all_positions.items()
    }


def get_bot_account(bot_name):
    return CONFIG.get('bots', {}).get(bot_name, {}).get('account')


def get_unique_accounts():
    accounts = []
    for bot_cfg in CONFIG.get('bots', {}).values():
        account = bot_cfg.get('account')
        if account and account not in accounts:
            accounts.append(account)
    return accounts


def get_account_starting_equity(account):
    if account is None:
        return 0.0
    equities = [
        float(bot_cfg.get('starting_equity', 0.0) or 0.0)
        for bot_cfg in CONFIG.get('bots', {}).values()
        if bot_cfg.get('account') == account
    ]
    return max(equities) if equities else 0.0


def get_max_lots_threshold(account=None):
    thresholds = CONFIG.get('risk_thresholds', {})
    per_100k = thresholds.get('max_total_lots_per_100k')

    if per_100k is None:
        base_limit = float(thresholds.get('max_total_lots', 2.0))
        return base_limit if base_limit > 0 else 0.0

    per_100k = float(per_100k)
    if per_100k <= 0:
        return 0.0
    if account is not None:
        starting_equity = get_account_starting_equity(account)
        if starting_equity > 0:
            return round((starting_equity / 100000.0) * per_100k, 2)
        return per_100k

    total_threshold = 0.0
    for acct in get_unique_accounts():
        total_threshold += get_max_lots_threshold(acct)
    return round(total_threshold, 2)


def get_conflict_peers(bot_name):
    conflict_scope = CONFIG.get('conflict_scope', 'cross_account')
    if conflict_scope != 'tiered':
        return set(CONFIG.get('bots', {}).keys())

    for tier in CONFIG.get('conflict_tiers', []):
        if bot_name in tier:
            return set(tier)

    # Bots outside configured tiers trade independently.
    return {bot_name}


def filter_positions_by_bots(all_positions, allowed_bots):
    allowed = set(allowed_bots)
    return {
        bot_name: [dict(pos) for pos in positions]
        for bot_name, positions in all_positions.items()
        if bot_name in allowed
    }


def _detect_conflicts_for_positions(all_positions):
    by_symbol = {}

    for bot_name, positions in all_positions.items():
        for pos in positions:
            symbol = pos.get('symbol', '')
            direction = normalize_pos_type(pos.get('type', ''))
            if not direction:
                continue
            by_symbol.setdefault(symbol, []).append({
                'bot': bot_name,
                'direction': direction,
                'ticket': pos.get('ticket'),
                'volume': float(pos.get('volume', 0)),
            })

    conflicts = []
    for symbol, entries in by_symbol.items():
        buys = [e for e in entries if e['direction'] == 'BUY']
        sells = [e for e in entries if e['direction'] == 'SELL']
        if buys and sells:
            long_volume = sum(e['volume'] for e in buys)
            short_volume = sum(e['volume'] for e in sells)
            conflicts.append({
                'symbol': symbol,
                'long_bots': [e['bot'] for e in buys],
                'short_bots': [e['bot'] for e in sells],
                'long_volume': round(long_volume, 2),
                'short_volume': round(short_volume, 2),
                'net_direction': 'LONG' if long_volume > short_volume else 'SHORT',
                'net_volume': round(abs(long_volume - short_volume), 2),
                'severity': 'HIGH'
            })

    return conflicts


def _check_correlation_for_positions(all_positions):
    currency_map = CONFIG.get('currency_map', {})
    exposure = {}

    for bot_name, positions in all_positions.items():
        for pos in positions:
            symbol = pos.get('symbol', '')
            direction = normalize_pos_type(pos.get('type', ''))
            if not direction:
                continue

            pair = currency_map.get(symbol)
            if not pair or len(pair) != 2:
                continue

            base, quote = pair[0], pair[1]

            if direction == 'BUY':
                exposure[base] = exposure.get(base, 0) + 1
                exposure[quote] = exposure.get(quote, 0) - 1
            elif direction == 'SELL':
                exposure[base] = exposure.get(base, 0) - 1
                exposure[quote] = exposure.get(quote, 0) + 1

    max_conc = CONFIG.get('risk_thresholds', {}).get('max_currency_concentration', 3)
    warnings = []
    for currency, net in exposure.items():
        if abs(net) >= max_conc:
            side = 'LONG' if net > 0 else 'SHORT'
            warnings.append({
                'currency': currency,
                'net_exposure': net,
                'direction': side,
                'severity': 'HIGH' if abs(net) > max_conc else 'MEDIUM'
            })

    return exposure, warnings


def get_opposite_positions(all_positions, symbol, direction):
    opposite = 'SELL' if direction == 'BUY' else 'BUY'
    matches = []
    for bot_name, positions in all_positions.items():
        for pos in positions:
            if pos.get('symbol') != symbol:
                continue
            pos_direction = normalize_pos_type(pos.get('type'))
            if pos_direction != opposite:
                continue
            matches.append({
                'bot': bot_name,
                'ticket': pos.get('ticket'),
                'direction': pos_direction,
                'volume': float(pos.get('volume', 0.0)),
            })
    return matches


def evaluate_open_request(bot_name, request_data):
    symbol = request_data['symbol']
    direction = request_data['direction']
    requesting_account = get_bot_account(bot_name)
    conflict_peers = get_conflict_peers(bot_name)

    all_statuses, all_positions = poll_all_bots()
    scoped_positions = filter_positions_by_bots(all_positions, conflict_peers)
    base_exposure, _ = check_correlation(scoped_positions)
    simulated_positions = clone_positions(all_positions)
    simulated_positions.setdefault(bot_name, []).append(build_simulated_position(bot_name, request_data))
    simulated_scoped_positions = filter_positions_by_bots(simulated_positions, conflict_peers)
    simulated_exposure, _ = check_correlation(simulated_scoped_positions)
    simulated_risk = check_aggregate_risk(
        all_statuses,
        simulated_positions,
        account_filter=requesting_account,
    )

    opposite_positions = get_opposite_positions(scoped_positions, symbol, direction)
    if opposite_positions:
        conflict_scope = CONFIG.get('conflict_scope', 'cross_account')
        if conflict_scope == 'same_account_only':
            # Only block if the opposing bot is on the same MT5 account
            blocking = [
                op for op in opposite_positions
                if get_bot_account(op['bot']) == requesting_account
            ]
        elif conflict_scope == 'tiered':
            # Only block if the requesting bot and opposing bot share a conflict tier.
            # Bots in different tiers (e.g. H1 vs M5) can hold opposite positions.
            # Bots in the same tier (e.g. M5 vs M1 scalpers) must stay aligned.
            blocking = [op for op in opposite_positions if op['bot'] in conflict_peers]
        else:
            # cross_account: block all opposite positions regardless of bot/account
            blocking = opposite_positions
        if blocking:
            return {
                'allowed': False,
                'reason': f'opposite_position_exists_on_{symbol}',
                'blocked_by': 'SYMBOL_CONFLICT',
                'opposite_positions': blocking,
                'symbol': symbol,
                'direction': direction,
            }

    thresholds = CONFIG.get('risk_thresholds', {})
    max_positions = int(thresholds.get('max_total_positions', 15))
    if simulated_risk['total_positions'] > max_positions:
        return {
            'allowed': False,
            'reason': f'total_positions_limit_exceeded:{simulated_risk["total_positions"]}>{max_positions}',
            'blocked_by': 'TOTAL_POSITIONS',
            'symbol': symbol,
            'direction': direction,
        }

    max_lots = float(simulated_risk.get('max_total_lots_threshold', get_max_lots_threshold(requesting_account)))
    if max_lots > 0 and simulated_risk['total_lots'] > max_lots:
        return {
            'allowed': False,
            'reason': f'total_lots_limit_exceeded:{simulated_risk["total_lots"]}>{max_lots}',
            'blocked_by': 'TOTAL_LOTS',
            'symbol': symbol,
            'direction': direction,
            'account': requesting_account,
        }

    max_per_symbol = int(thresholds.get('max_positions_per_symbol', 3))
    simulated_symbol_count = int(simulated_risk['symbol_counts'].get(symbol, 0))
    if simulated_symbol_count > max_per_symbol:
        return {
            'allowed': False,
            'reason': f'symbol_stacking_limit_exceeded:{simulated_symbol_count}>{max_per_symbol}',
            'blocked_by': 'SYMBOL_STACKING',
            'symbol': symbol,
            'direction': direction,
        }

    pair = CONFIG.get('currency_map', {}).get(symbol, [])
    currency_warnings = []
    max_conc = int(thresholds.get('max_currency_concentration', 3))
    for currency in pair:
        before = abs(int(base_exposure.get(currency, 0)))
        after = abs(int(simulated_exposure.get(currency, 0)))
        if after >= max_conc and after > before:
            currency_warnings.append({
                'currency': currency,
                'before': before,
                'after': after,
                'threshold': max_conc,
            })
    if currency_warnings:
        return {
            'allowed': False,
            'reason': f'currency_concentration_limit_reached_for_{symbol}',
            'blocked_by': 'CURRENCY_CONCENTRATION',
            'currency_warnings': currency_warnings,
            'symbol': symbol,
            'direction': direction,
        }

    return {
        'allowed': True,
        'reason': 'approved',
        'blocked_by': None,
        'symbol': symbol,
        'direction': direction,
        'risk': {
            'total_positions': simulated_risk['total_positions'],
            'total_lots': simulated_risk['total_lots'],
            'symbol_count': simulated_symbol_count,
        },
    }

# ============================================================================
# BOT POLLING
# ============================================================================
def poll_bot(name, bot_cfg):
    """
    Poll a bot's /status and /positions endpoints.
    Returns: (status_dict, positions_list)
    Raises on failure.
    """
    headers = {}
    api_key = bot_cfg.get('api_key', '')
    if api_key:
        headers['X-API-Key'] = api_key

    url = bot_cfg['url']
    timeout = CONFIG.get('poll_timeout', 5)

    status_resp = http_requests.get(f"{url}/status", headers=headers, timeout=timeout)
    status = status_resp.json()

    pos_resp = http_requests.get(f"{url}/positions", headers=headers, timeout=timeout)
    positions = pos_resp.json().get('positions', [])

    return status, positions

def poll_all_bots():
    """
    Poll all configured bots. Returns (all_statuses, all_positions) dicts.
    Offline bots get status='OFFLINE' and empty positions.
    """
    bots = CONFIG.get('bots', {})
    all_statuses = {}
    all_positions = {}

    for bot_name, bot_cfg in bots.items():
        try:
            status, positions = poll_bot(bot_name, bot_cfg)
            all_statuses[bot_name] = status
            all_positions[bot_name] = positions
        except Exception as e:
            all_statuses[bot_name] = {
                'status': 'OFFLINE',
                'error': str(e),
                'account': bot_cfg.get('account'),
                'timeframe': bot_cfg.get('timeframe')
            }
            all_positions[bot_name] = []

    return all_statuses, all_positions

# ============================================================================
# CONFLICT DETECTION
# ============================================================================
def detect_conflicts(all_positions):
    """
    Find opposing positions on the same symbol across different bots.
    E.g., Anaconda LONG EURUSD + Mamba SHORT EURUSD = conflict.
    Returns: list of conflict dicts.
    """
    if CONFIG.get('conflict_scope', 'cross_account') != 'tiered':
        return _detect_conflicts_for_positions(all_positions)

    conflicts = []
    seen = set()
    for tier in CONFIG.get('conflict_tiers', []):
        tier_key = tuple(sorted(tier))
        if tier_key in seen:
            continue
        seen.add(tier_key)
        tier_positions = filter_positions_by_bots(all_positions, tier)
        conflicts.extend(_detect_conflicts_for_positions(tier_positions))
    return conflicts

# ============================================================================
# CURRENCY CORRELATION
# ============================================================================
def check_correlation(all_positions):
    """
    Count net directional exposure per currency across all bots.

    EURUSD BUY = long EUR, short USD
    USDJPY SELL = short USD, long JPY
    XAUUSD BUY = long XAU, short USD

    3 positions all short USD = 3x the same bet = WARNING.

    Returns: (exposure_dict, warnings_list)
    """
    if CONFIG.get('conflict_scope', 'cross_account') != 'tiered':
        return _check_correlation_for_positions(all_positions)

    merged_exposure = {}
    warnings = []
    seen = set()
    for tier in CONFIG.get('conflict_tiers', []):
        tier_key = tuple(sorted(tier))
        if tier_key in seen:
            continue
        seen.add(tier_key)
        tier_positions = filter_positions_by_bots(all_positions, tier)
        tier_exposure, tier_warnings = _check_correlation_for_positions(tier_positions)
        for currency, net in tier_exposure.items():
            current = merged_exposure.get(currency)
            if current is None or abs(net) > abs(current):
                merged_exposure[currency] = net
        if tier_warnings:
            tier_label = '/'.join(sorted(tier))
            for warning in tier_warnings:
                warning = dict(warning)
                warning['tier'] = tier_label
                warnings.append(warning)

    return merged_exposure, warnings

# ============================================================================
# AGGREGATE RISK
# ============================================================================
def check_aggregate_risk(all_statuses, all_positions, account_filter=None):
    """
    Check total position count, total lots, and per-symbol stacking.
    """
    thresholds = CONFIG.get('risk_thresholds', {})

    scoped_positions = {
        bot_name: positions
        for bot_name, positions in all_positions.items()
        if account_filter is None or get_bot_account(bot_name) == account_filter
    }
    scoped_statuses = {
        bot_name: status
        for bot_name, status in all_statuses.items()
        if account_filter is None or get_bot_account(bot_name) == account_filter
    }

    total_positions = sum(len(p) for p in scoped_positions.values())
    total_lots = sum(
        sum(float(pos.get('volume', 0)) for pos in positions)
        for positions in scoped_positions.values()
    )

    # Per-symbol position count
    symbol_counts = {}
    for bot_name, positions in scoped_positions.items():
        for pos in positions:
            sym = pos.get('symbol', '')
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

    warnings = []

    max_positions = thresholds.get('max_total_positions', 15)
    if total_positions > max_positions:
        warnings.append({
            'type': 'TOTAL_POSITIONS',
            'value': total_positions,
            'threshold': max_positions,
            'severity': 'HIGH'
        })

    max_lots = get_max_lots_threshold(account_filter)
    if max_lots > 0 and total_lots > max_lots:
        warnings.append({
            'type': 'TOTAL_LOTS',
            'value': round(total_lots, 2),
            'threshold': max_lots,
            'severity': 'HIGH'
        })

    max_per_symbol = thresholds.get('max_positions_per_symbol', 3)
    for sym, count in symbol_counts.items():
        if count > max_per_symbol:
            warnings.append({
                'type': 'SYMBOL_STACKING',
                'symbol': sym,
                'value': count,
                'threshold': max_per_symbol,
                'severity': 'MEDIUM'
            })

    # Aggregate account stats (dedupe shared accounts so one account is not counted 5x)
    total_balance = 0
    total_equity = 0
    online_bots = 0
    seen_accounts = set()
    for bot_name, status in scoped_statuses.items():
        if status.get('status') == 'OFFLINE':
            continue
        online_bots += 1
        account_id = status.get('account')
        if account_id in seen_accounts:
            continue
        seen_accounts.add(account_id)
        total_balance += status.get('balance', 0)
        total_equity += status.get('equity', 0)

    return {
        'total_positions': total_positions,
        'total_lots': round(total_lots, 2),
        'symbol_counts': symbol_counts,
        'total_balance': round(total_balance, 2),
        'total_equity': round(total_equity, 2),
        'online_bots': online_bots,
        'total_bots': len(scoped_statuses),
        'max_total_lots_threshold': max_lots,
        'account_filter': account_filter,
        'warnings': warnings
    }

# ============================================================================
# MONITORING LOOP
# ============================================================================
def monitor_loop():
    """
    Main monitoring loop. Polls all bots, runs analysis, stores results.
    """
    global _latest_state
    iteration = 0

    while not stop_event.is_set():
        try:
            all_statuses, all_positions = poll_all_bots()

            # Run analysis
            conflicts = detect_conflicts(all_positions)
            exposure, correlation_warnings = check_correlation(all_positions)
            risk_summary = check_aggregate_risk(all_statuses, all_positions)

            # Log warnings
            for c in conflicts:
                logger.warning(
                    f"[CONFLICT] {c['symbol']}: LONG by {c['long_bots']} ({c['long_volume']} lots) "
                    f"vs SHORT by {c['short_bots']} ({c['short_volume']} lots)"
                )
            for w in correlation_warnings:
                logger.warning(
                    f"[CORRELATION] {w['currency']} net {w['direction']} x{abs(w['net_exposure'])}"
                )
            for w in risk_summary['warnings']:
                logger.warning(
                    f"[RISK] {w['type']}: {w.get('value')} (threshold: {w.get('threshold')})"
                )

            # Store latest state
            with state_lock:
                _latest_state = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'statuses': all_statuses,
                    'positions': all_positions,
                    'conflicts': conflicts,
                    'correlation': {
                        'exposure': exposure,
                        'warnings': correlation_warnings
                    },
                    'risk': risk_summary
                }

            iteration += 1
            if iteration % 10 == 0:
                online = risk_summary['online_bots']
                total = risk_summary['total_bots']
                logger.info(
                    f"[HEARTBEAT] Iteration {iteration} | "
                    f"Bots: {online}/{total} online | "
                    f"Positions: {risk_summary['total_positions']} | "
                    f"Lots: {risk_summary['total_lots']} | "
                    f"Conflicts: {len(conflicts)} | "
                    f"Correlation warnings: {len(correlation_warnings)}"
                )

        except Exception as e:
            logger.error(f"[MONITOR] Error: {e}")

        time.sleep(CONFIG.get('poll_interval', 30))

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/can_open', methods=['POST'])
def can_open():
    data = request.get_json() or {}
    bot_name = str(data.get('bot', '')).strip().lower()
    symbol = str(data.get('symbol', '')).strip()
    direction = normalize_pos_type(data.get('direction'))

    if not bot_name:
        return jsonify({'allowed': False, 'reason': 'missing_bot'}), 400
    if bot_name not in CONFIG.get('bots', {}):
        return jsonify({'allowed': False, 'reason': f'unknown_bot:{bot_name}'}), 400
    if not symbol:
        return jsonify({'allowed': False, 'reason': 'missing_symbol'}), 400
    if direction not in ('BUY', 'SELL'):
        return jsonify({'allowed': False, 'reason': 'invalid_direction'}), 400

    try:
        volume = float(data.get('volume', 0.0))
    except Exception:
        return jsonify({'allowed': False, 'reason': 'invalid_volume'}), 400
    if volume <= 0:
        return jsonify({'allowed': False, 'reason': 'invalid_volume'}), 400

    request_data = {
        'symbol': symbol,
        'direction': direction,
        'volume': volume,
        'entry_price': data.get('entry_price'),
        'sl': data.get('sl'),
        'tp': data.get('tp'),
    }

    try:
        decision = evaluate_open_request(bot_name, request_data)
        if decision['allowed']:
            logger.info(f"[APPROVE] {bot_name} {direction} {symbol} vol={volume}")
        else:
            logger.warning(f"[BLOCK] {bot_name} {direction} {symbol} vol={volume} -- {decision['reason']}")
        return jsonify(decision)
    except Exception as e:
        logger.error(f"[CAN_OPEN] Error evaluating {bot_name} {direction} {symbol}: {e}")
        return jsonify({'allowed': False, 'reason': f'oracle_error:{e}'}), 500

@app.route('/status', methods=['GET'])
def get_status():
    with state_lock:
        state = _latest_state.copy()
    return jsonify({
        'bot': 'oracle-v1',
        'timestamp': state['timestamp'],
        'online_bots': state['risk'].get('online_bots', 0),
        'total_bots': state['risk'].get('total_bots', 0),
        'total_positions': state['risk'].get('total_positions', 0),
        'total_lots': state['risk'].get('total_lots', 0),
        'conflicts': len(state['conflicts']),
        'correlation_warnings': len(state['correlation'].get('warnings', [])),
        'risk_warnings': len(state['risk'].get('warnings', []))
    })

@app.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Full unified dashboard — all bots, positions, risks, conflicts."""
    with state_lock:
        state = _latest_state.copy()
    return jsonify(state)

@app.route('/conflicts', methods=['GET'])
def get_conflicts():
    with state_lock:
        return jsonify({'conflicts': _latest_state.get('conflicts', [])})

@app.route('/correlation', methods=['GET'])
def get_correlation():
    with state_lock:
        return jsonify(_latest_state.get('correlation', {}))

@app.route('/risk', methods=['GET'])
def get_risk():
    with state_lock:
        return jsonify(_latest_state.get('risk', {}))

@app.route('/positions', methods=['GET'])
def get_all_positions():
    """All positions across all bots."""
    with state_lock:
        all_pos = _latest_state.get('positions', {})

    # Flatten into a single list with bot name attached
    flat = []
    for bot_name, positions in all_pos.items():
        for pos in positions:
            entry = dict(pos)
            entry['bot'] = bot_name
            flat.append(entry)

    return jsonify({'positions': flat, 'count': len(flat)})

@app.route('/bots', methods=['GET'])
def get_bots():
    """Status of each individual bot."""
    with state_lock:
        statuses = _latest_state.get('statuses', {})
    return jsonify({'bots': statuses})

@app.route('/kill', methods=['POST'])
def kill_all():
    """Emergency stop — POST /stop to ALL bots."""
    bots = CONFIG.get('bots', {})
    results = {}

    for bot_name, bot_cfg in bots.items():
        try:
            headers = {}
            api_key = bot_cfg.get('api_key', '')
            if api_key:
                headers['X-API-Key'] = api_key
            resp = http_requests.post(
                f"{bot_cfg['url']}/stop",
                headers=headers,
                timeout=5
            )
            results[bot_name] = {'success': True, 'status': resp.status_code}
            logger.warning(f"[KILL] Stopped {bot_name}: {resp.status_code}")
        except Exception as e:
            results[bot_name] = {'success': False, 'error': str(e)}
            logger.error(f"[KILL] Failed to stop {bot_name}: {e}")

    return jsonify({'action': 'kill_all', 'results': results})

@app.route('/stop/<bot_name>', methods=['POST'])
def stop_bot(bot_name):
    """Stop a specific bot."""
    bots = CONFIG.get('bots', {})
    bot_cfg = bots.get(bot_name)
    if not bot_cfg:
        return jsonify({'error': f'Bot {bot_name} not found'}), 404

    try:
        headers = {}
        api_key = bot_cfg.get('api_key', '')
        if api_key:
            headers['X-API-Key'] = api_key
        resp = http_requests.post(
            f"{bot_cfg['url']}/stop",
            headers=headers,
            timeout=5
        )
        logger.warning(f"[STOP] Stopped {bot_name}: {resp.status_code}")
        return jsonify({'success': True, 'bot': bot_name, 'status': resp.status_code})
    except Exception as e:
        logger.error(f"[STOP] Failed to stop {bot_name}: {e}")
        return jsonify({'success': False, 'bot': bot_name, 'error': str(e)}), 500

@app.route('/start/<bot_name>', methods=['POST'])
def start_bot(bot_name):
    """Start a specific bot."""
    bots = CONFIG.get('bots', {})
    bot_cfg = bots.get(bot_name)
    if not bot_cfg:
        return jsonify({'error': f'Bot {bot_name} not found'}), 404

    try:
        headers = {}
        api_key = bot_cfg.get('api_key', '')
        if api_key:
            headers['X-API-Key'] = api_key
        resp = http_requests.post(
            f"{bot_cfg['url']}/start",
            headers=headers,
            timeout=5
        )
        logger.info(f"[START] Started {bot_name}: {resp.status_code}")
        return jsonify({'success': True, 'bot': bot_name, 'status': resp.status_code})
    except Exception as e:
        logger.error(f"[START] Failed to start {bot_name}: {e}")
        return jsonify({'success': False, 'bot': bot_name, 'error': str(e)}), 500

# ============================================================================
# INIT
# ============================================================================
def load_config(config_path=None):
    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'oracle_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    global CONFIG, _monitor_thread

    parser = argparse.ArgumentParser(description='ORACLE v1 -- Coordinator / Risk Overlay')
    parser.add_argument('--config', default=None, help='Path to oracle_config.json')
    args = parser.parse_args()

    CONFIG = load_config(args.config)
    init_logger()

    bot_names = list(CONFIG.get('bots', {}).keys())
    logger.info("=" * 60)
    logger.info("ORACLE v1 Starting -- Coordinator / Risk Overlay")
    logger.info(f"Monitoring {len(bot_names)} bots: {', '.join(bot_names)}")
    logger.info(f"Poll interval: {CONFIG.get('poll_interval', 30)}s")
    logger.info(f"Risk thresholds: {CONFIG.get('risk_thresholds', {})}")
    logger.info("=" * 60)

    # Start monitoring thread
    _monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    _monitor_thread.start()
    logger.info("Monitoring loop started")

    port = CONFIG.get('server_port', 8070)
    logger.info(f"Flask API on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)

if __name__ == '__main__':
    main()
