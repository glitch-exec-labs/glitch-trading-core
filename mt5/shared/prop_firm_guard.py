"""
PropFirmGuard — Prop Firm Safety Layer (Standalone Module)

Monitors equity, daily P&L, and trailing drawdown to protect prop firm accounts.
Can be imported by any bot.

Features:
  1. Daily loss halt — stops trading at configurable % loss
  2. Trailing drawdown halt — stops at % from peak equity
  3. Daily profit banking — stops after good day to lock gains
  4. Risk mode scaling — reduces position size as risk increases
  5. Profit target detection — flags when target reached
  6. Friday flatten — closes all positions before weekend

Usage:
    from prop_firm_guard import PropFirmGuard

    guard = PropFirmGuard(config, logger)
    guard.update(equity, balance)
    allowed, risk_mode, multiplier = guard.can_trade()
"""
import time
import threading
import json
import os
from datetime import datetime, timezone, timedelta


class PropFirmGuard:
    def __init__(self, config, logger, state_file=None):
        """
        config: dict with prop_firm section:
            initial_capital: float (e.g. 150000)
            profit_target_pct: float (e.g. 6.0)
            max_daily_loss_pct: float (e.g. 3.0)
            max_trailing_loss_pct: float (e.g. 6.0)
            daily_loss_halt_pct: float (e.g. 2.5) — halt threshold (with buffer)
            trailing_dd_halt_pct: float (e.g. 5.5) — halt threshold (with buffer)
            daily_loss_warning_pct: float (e.g. 1.5)
            daily_profit_bank_pct: float (e.g. 2.0)
            daily_profit_conservative_pct: float (e.g. 1.5)
            trailing_dd_warning_pct: float (e.g. 4.0)
            trailing_dd_critical_pct: float (e.g. 5.0)
            friday_flatten_hour: int (e.g. 19) — UTC hour
        logger: logging.Logger instance
        state_file: optional path for persisting state across restarts
        """
        pf = config.get('prop_firm', {})
        self.initial_capital = float(pf.get('initial_capital', 150000))
        self.profit_target_pct = float(pf.get('profit_target_pct', 6.0))
        self.daily_loss_halt_pct = float(pf.get('daily_loss_halt_pct', 2.5))
        self.trailing_dd_halt_pct = float(pf.get('trailing_dd_halt_pct', 5.5))
        self.daily_loss_warning_pct = float(pf.get('daily_loss_warning_pct', 1.5))
        self.daily_profit_bank_pct = float(pf.get('daily_profit_bank_pct', 2.0))
        self.daily_profit_conservative_pct = float(pf.get('daily_profit_conservative_pct', 1.5))
        self.trailing_dd_warning_pct = float(pf.get('trailing_dd_warning_pct', 4.0))
        self.trailing_dd_critical_pct = float(pf.get('trailing_dd_critical_pct', 5.0))
        self.friday_flatten_hour = int(pf.get('friday_flatten_hour', 19))
        self.max_total_positions = int(pf.get('max_total_positions', 5))
        self.max_positions_per_symbol = int(pf.get('max_positions_per_symbol', 1))
        self.loss_cooloff_seconds = int(pf.get('loss_cooloff_seconds', 300))
        self.max_losses_per_symbol_per_day = int(pf.get('max_losses_per_symbol_per_day', 3))

        self.logger = logger
        self.state_file = state_file
        self._lock = threading.Lock()

        self._state = {
            'peak_equity': self.initial_capital,
            'initial_balance': self.initial_capital,
            'daily_start_equity': 0.0,
            'daily_start_date': None,
            'last_daily_reset': None,
            'current_equity': self.initial_capital,
            'halted': False,
            'halt_reason': '',
            'risk_mode': 'normal',
            'target_reached': False,
        }

        # Per-symbol loss tracking
        self._last_loss_time = {}
        self._symbol_daily_loss_count = {}

        # Try to load persisted state
        self._load_state()

    def _parse_halt_reason_pct(self):
        reason = str(self._state.get('halt_reason', '') or '')
        if '_' not in reason or not reason.endswith('%'):
            return None
        try:
            return float(reason.rsplit('_', 1)[1].rstrip('%'))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # STATE PERSISTENCE
    # ------------------------------------------------------------------
    def _load_state(self):
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, 'r') as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                for key in self._state:
                    if key in saved:
                        self._state[key] = saved[key]
                self.logger.info(f"[PROP] Loaded state from {self.state_file}: "
                                 f"peak=${self._state['peak_equity']:.2f}, "
                                 f"mode={self._state['risk_mode']}")
        except Exception as e:
            self.logger.warning(f"[PROP] Failed to load state: {e}")

    def _save_state(self):
        if not self.state_file:
            return
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self._state, f, indent=2)
        except Exception:
            pass

    def get_current_reset_boundary(self, now=None):
        now_utc = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
        reset_time = now_utc.replace(hour=20, minute=0, second=0, microsecond=0)
        if now_utc < reset_time:
            reset_time -= timedelta(days=1)
        return reset_time

    def get_trading_day_key(self, now=None):
        return self.get_current_reset_boundary(now).date().isoformat()

    def _parse_reset_timestamp(self):
        raw = self._state.get('last_daily_reset')
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw)
                return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            except Exception:
                pass

        legacy_day = self._state.get('daily_start_date')
        if isinstance(legacy_day, str) and legacy_day:
            try:
                return datetime.fromisoformat(legacy_day).replace(hour=20, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            except Exception:
                pass

        return datetime.min.replace(tzinfo=timezone.utc)

    def should_reset_daily(self, last_reset_time, now=None):
        return last_reset_time < self.get_current_reset_boundary(now)

    # ------------------------------------------------------------------
    # UPDATE — call every strategy loop iteration
    # ------------------------------------------------------------------
    def update(self, equity, balance):
        """
        Update guard state with current account equity/balance.
        Returns the current risk_mode string.
        """
        with self._lock:
            if equity is None or balance is None or equity <= 0 or balance <= 0:
                self.logger.warning(
                    f"[PROP] Skipping guard update due to invalid account snapshot "
                    f"(equity={equity}, balance={balance})"
                )
                return self._state.get('risk_mode', 'normal')

            now = datetime.now(timezone.utc)
            today = self.get_trading_day_key(now)
            reset_boundary = self.get_current_reset_boundary(now)
            last_daily_reset = self._parse_reset_timestamp()

            # Always track latest equity for smooth sizing
            self._state['current_equity'] = equity

            # Reset daily tracking at 20:00 UTC
            if self.should_reset_daily(last_daily_reset, now):
                self._state['daily_start_equity'] = equity
                self._state['daily_start_date'] = today
                self._state['last_daily_reset'] = reset_boundary.isoformat()
                self._state['halted'] = False
                self._state['halt_reason'] = ''
                self._symbol_daily_loss_count.clear()
                self.logger.info(f"[DAILY RESET] Daily P&L reset at 20:00 UTC -- start equity: ${equity:.2f}")
            elif not self._state.get('daily_start_date'):
                self._state['daily_start_date'] = today
                self._state['last_daily_reset'] = reset_boundary.isoformat()

            if self._state.get('daily_start_equity', 0) <= 0:
                self._state['daily_start_equity'] = equity
                self._state['daily_start_date'] = today
                self._state['last_daily_reset'] = reset_boundary.isoformat()
                self.logger.warning(
                    f"[PROP] Repaired invalid daily_start_equity using current equity ${equity:.2f}"
                )

            # Track peak equity for trailing drawdown
            if equity > self._state['peak_equity']:
                self._state['peak_equity'] = equity
                self.logger.info(f"[PROP] New peak equity: ${equity:.2f}")

            # Calculate daily P&L
            daily_pnl = equity - self._state['daily_start_equity']
            daily_pnl_pct = (daily_pnl / self._state['initial_balance']) * 100

            # Calculate trailing drawdown from peak
            dd_from_peak = equity - self._state['peak_equity']
            dd_from_peak_pct = (dd_from_peak / self._state['initial_balance']) * 100

            # Check profit target
            total_profit_pct = ((equity - self._state['initial_balance']) / self._state['initial_balance']) * 100
            if total_profit_pct >= self.profit_target_pct:
                self._state['target_reached'] = True
                self.logger.info(f"[PROP] TARGET REACHED! Total profit: {total_profit_pct:.2f}%")

            reason = str(self._state.get('halt_reason', '') or '')
            reason_pct = self._parse_halt_reason_pct()
            stale_daily_halt = (
                self._state.get('halted')
                and reason.startswith('daily_loss_')
                and reason_pct is not None
                and reason_pct <= -25.0
                and daily_pnl_pct > -self.daily_loss_warning_pct
                and dd_from_peak_pct > -self.trailing_dd_warning_pct
            )
            stale_dd_halt = (
                self._state.get('halted')
                and reason.startswith('trailing_dd_')
                and reason_pct is not None
                and reason_pct <= -25.0
                and daily_pnl_pct > -self.daily_loss_warning_pct
                and dd_from_peak_pct > -self.trailing_dd_warning_pct
            )
            if stale_daily_halt or stale_dd_halt:
                self.logger.warning(
                    f"[PROP] Clearing stale halt state '{reason}' "
                    f"(daily={daily_pnl_pct:+.2f}%, dd={dd_from_peak_pct:.2f}%)"
                )
                self._state['halted'] = False
                self._state['halt_reason'] = ''

            # === HALT CONDITIONS ===
            if daily_pnl_pct <= -self.daily_loss_halt_pct:
                self._state['halted'] = True
                self._state['halt_reason'] = f'daily_loss_{daily_pnl_pct:.2f}%'
                self.logger.warning(f"[PROP] HALT -- Daily loss {daily_pnl_pct:.2f}% exceeds -{self.daily_loss_halt_pct}%")
                self._save_state()
                return 'halted'

            if dd_from_peak_pct <= -self.trailing_dd_halt_pct:
                self._state['halted'] = True
                self._state['halt_reason'] = f'trailing_dd_{dd_from_peak_pct:.2f}%'
                self.logger.warning(f"[PROP] HALT -- Trailing DD {dd_from_peak_pct:.2f}% exceeds -{self.trailing_dd_halt_pct}%")
                self._save_state()
                return 'halted'

            # === DAILY PROFIT BANKING ===
            if daily_pnl_pct >= self.daily_profit_bank_pct:
                self._state['halted'] = True
                self._state['halt_reason'] = f'daily_profit_banked_{daily_pnl_pct:.2f}%'
                self.logger.info(f"[PROP] BANK -- Daily profit {daily_pnl_pct:.2f}% hit bank threshold")
                self._save_state()
                return 'halted'

            # === RISK MODE ===
            if dd_from_peak_pct <= -self.trailing_dd_critical_pct or daily_pnl_pct <= -(self.daily_loss_halt_pct - 0.5):
                self._state['risk_mode'] = 'critical'
            elif dd_from_peak_pct <= -self.trailing_dd_warning_pct or daily_pnl_pct <= -self.daily_loss_warning_pct:
                self._state['risk_mode'] = 'conservative'
            elif daily_pnl_pct >= self.daily_profit_conservative_pct:
                self._state['risk_mode'] = 'conservative'
            else:
                self._state['risk_mode'] = 'normal'

            self.logger.info(
                f"[PROP] equity=${equity:.2f} daily={daily_pnl_pct:+.2f}% "
                f"dd={dd_from_peak_pct:.2f}% mode={self._state['risk_mode']} "
                f"peak=${self._state['peak_equity']:.2f}"
            )

            self._save_state()
            return self._state['risk_mode']

    # ------------------------------------------------------------------
    # CAN TRADE — check before any entry
    # ------------------------------------------------------------------
    def can_trade(self, symbol=None, current_positions=None):
        """
        Check if trading is allowed.
        Returns: (allowed: bool, risk_mode: str, risk_multiplier: float)

        current_positions: list of open position objects (optional, for position limit checks)
        """
        with self._lock:
            if self._state['halted']:
                return False, 'halted', 0.0
            if self._state['target_reached']:
                return False, 'target_reached', 0.0
            mode = self._state['risk_mode']

        # Position limit checks
        if current_positions is not None and symbol:
            total = len(current_positions)
            if total >= self.max_total_positions:
                return False, f'max_positions_{total}', 0.0

            sym_count = sum(1 for p in current_positions if getattr(p, 'symbol', '') == symbol)
            if sym_count >= self.max_positions_per_symbol:
                return False, f'max_per_symbol_{sym_count}', 0.0

        # Per-symbol loss cooloff
        if symbol:
            last_loss = self._last_loss_time.get(symbol, 0)
            if last_loss > 0 and time.time() - last_loss < self.loss_cooloff_seconds:
                remaining = int(self.loss_cooloff_seconds - (time.time() - last_loss))
                return False, f'loss_cooloff_{remaining}s', 0.0

            today = self.get_trading_day_key()
            daily = self._symbol_daily_loss_count.get(symbol, {})
            if daily.get('date') == today and daily.get('count', 0) >= self.max_losses_per_symbol_per_day:
                return False, f'max_daily_losses_{symbol}', 0.0

        # Risk multiplier based on mode
        multiplier = self.get_risk_multiplier()
        return True, mode, multiplier

    # ------------------------------------------------------------------
    # RISK MULTIPLIER — smooth continuous scaling
    # ------------------------------------------------------------------
    def get_risk_multiplier(self):
        """
        Returns a smooth position sizing multiplier (0.01 – 1.0).

        Scales linearly based on how much of each loss budget has been consumed:

            daily_scale    = 1 - (daily_loss_used%  / daily_halt%)
            trailing_scale = 1 - (trailing_dd_used% / trailing_dd_halt%)
            multiplier     = min(daily_scale, trailing_scale)

        Examples (daily_halt=2.5%, trailing_halt=5.5%):
            Account at peak, no daily loss   → 1.00  (full size)
            Down 1% daily, 2% trailing DD    → min(0.60, 0.64) = 0.60
            Down 2% daily, 3% trailing DD    → min(0.20, 0.45) = 0.20
            Down 2.5% daily (at halt)        → 0.01  (near-zero, halted next tick)

        Never returns exactly 0.0 so min_lot is still respected by callers.
        """
        with self._lock:
            mode       = self._state['risk_mode']
            equity     = self._state.get('current_equity', self.initial_capital)
            daily_start = self._state.get('daily_start_equity', equity)
            peak       = self._state.get('peak_equity', equity)
            initial    = self._state.get('initial_balance', self.initial_capital)

        if mode == 'halted':
            return 0.01

        if initial <= 0:
            return 1.0

        # Daily loss scale
        daily_loss_pct = max(0.0, (daily_start - equity) / initial * 100)
        daily_scale = max(0.01, 1.0 - daily_loss_pct / self.daily_loss_halt_pct) \
                      if self.daily_loss_halt_pct > 0 else 1.0

        # Trailing drawdown scale
        trailing_dd_pct = max(0.0, (peak - equity) / initial * 100)
        trailing_scale = max(0.01, 1.0 - trailing_dd_pct / self.trailing_dd_halt_pct) \
                         if self.trailing_dd_halt_pct > 0 else 1.0

        multiplier = min(daily_scale, trailing_scale)
        return round(multiplier, 3)

    def get_risk_mode(self):
        """Get current risk mode string."""
        with self._lock:
            return self._state['risk_mode']

    # ------------------------------------------------------------------
    # LOSS RECORDING
    # ------------------------------------------------------------------
    def record_loss(self, symbol):
        """Call this when a trade closes at a loss."""
        self._last_loss_time[symbol] = time.time()
        today = self.get_trading_day_key()
        daily = self._symbol_daily_loss_count.get(symbol, {})
        if daily.get('date') != today:
            daily = {'date': today, 'count': 0}
        daily['count'] += 1
        self._symbol_daily_loss_count[symbol] = daily

    # ------------------------------------------------------------------
    # DAILY P&L RECOVERY AFTER RESTART
    # ------------------------------------------------------------------
    def recover_from_history(self, realized_today_pnl, current_equity):
        """
        Recover daily_start_equity from MT5 deal history after a bot restart.

        Call once after MT5 connects, before the strategy loop starts.
        No-op if state_file already has the current 20:00 UTC reset window.

        Args:
            realized_today_pnl: sum of (profit + commission + swap) for all
                                 closed deals since the latest 20:00 UTC reset.
            current_equity:     current account equity from MT5.

        Returns True if recovery was applied, False if not needed.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            today = self.get_trading_day_key(now)
            reset_boundary = self.get_current_reset_boundary(now)
            last_daily_reset = self._parse_reset_timestamp()
            if self._state['daily_start_date'] == today and not self.should_reset_daily(last_daily_reset, now):
                # State file is current — daily_start_equity already correct
                self.logger.info(
                    f"[PROP] Recovery skipped — state file is current for {today} "
                    f"(daily_start=${self._state['daily_start_equity']:.2f})"
                )
                return False

            # State is stale or missing — reconstruct from MT5 history
            recovered_start = current_equity - realized_today_pnl
            self._state['daily_start_equity'] = recovered_start
            self._state['daily_start_date'] = today
            self._state['last_daily_reset'] = reset_boundary.isoformat()
            self._state['halted'] = False
            self._state['halt_reason'] = ''
            self._symbol_daily_loss_count.clear()

            # Also update peak_equity if current equity is higher than stored peak
            if current_equity > self._state['peak_equity']:
                self._state['peak_equity'] = current_equity

            self.logger.info(
                f"[PROP] Daily P&L recovered from MT5 history since {reset_boundary.strftime('%Y-%m-%d %H:%M UTC')}: "
                f"daily_start=${recovered_start:.2f} "
                f"(equity=${current_equity:.2f}, realized_today={realized_today_pnl:+.2f})"
            )
            self._save_state()
            return True

    # ------------------------------------------------------------------
    # FRIDAY FLATTEN CHECK
    # ------------------------------------------------------------------
    def should_flatten_friday(self):
        """Returns True if it's Friday and past the flatten hour (UTC)."""
        now = datetime.now(timezone.utc)
        return now.weekday() == 4 and now.hour >= self.friday_flatten_hour

    # ------------------------------------------------------------------
    # STATE FOR API / DASHBOARD
    # ------------------------------------------------------------------
    def get_state(self):
        """Return state dict for Flask API /prop_firm endpoint."""
        with self._lock:
            state = dict(self._state)
        now = datetime.now(timezone.utc)
        equity = state.get('daily_start_equity', 0)
        state['timestamp'] = now.isoformat()
        mult = self.get_risk_multiplier()
        state['risk_multiplier'] = mult
        state['position_size_pct'] = f"{mult * 100:.0f}%"
        state['friday_flatten'] = self.should_flatten_friday()
        state['config'] = {
            'initial_capital': self.initial_capital,
            'profit_target_pct': self.profit_target_pct,
            'daily_loss_halt_pct': self.daily_loss_halt_pct,
            'trailing_dd_halt_pct': self.trailing_dd_halt_pct,
        }
        return state
