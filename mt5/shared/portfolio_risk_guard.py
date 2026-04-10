"""
Shared cross-bot portfolio risk guard.
Uses SQLite for safe multi-process coordination across bot services.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional, Tuple


DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "global_daily_loss_limit": -250.0,
    "max_total_positions": 8,
    "max_positions_per_bot": 4,
    "max_positions_per_symbol": 3,
    "max_same_direction_per_symbol": 2,
    "cooldown_minutes_after_breach": 30,
    "closed_trades_retention_days": 30,
}


class PortfolioRiskGuard:
    def __init__(
        self,
        bot_name: str,
        account_number: int,
        config_path: Optional[str] = None,
        db_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.bot_name = bot_name
        self.account_number = int(account_number)
        self._lock = threading.Lock()

        merged = dict(DEFAULT_CONFIG)

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    if isinstance(loaded.get("portfolio_risk"), dict):
                        merged.update(loaded["portfolio_risk"])
                    else:
                        merged.update(loaded)
            except Exception:
                # Non-fatal: keep defaults.
                pass

        if config:
            merged.update(config)

        self.config = merged
        self.enabled = bool(self.config.get("enabled", True))

        if db_path:
            resolved_db = db_path
        else:
            resolved_db = self.config.get("db_path", "portfolio_risk_guard.db")

        if not os.path.isabs(resolved_db):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resolved_db = os.path.join(base, resolved_db)

        self.db_path = resolved_db
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS open_positions (
                    pos_key TEXT PRIMARY KEY,
                    ticket INTEGER NOT NULL,
                    bot TEXT NOT NULL,
                    account INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    volume REAL,
                    entry_price REAL,
                    opened_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    closed_at TEXT NOT NULL,
                    closed_date TEXT NOT NULL,
                    bot TEXT NOT NULL,
                    account INTEGER NOT NULL,
                    ticket INTEGER,
                    symbol TEXT,
                    profit REAL NOT NULL,
                    duration_seconds REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_closed_date ON closed_trades(closed_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_open_symbol_side ON open_positions(symbol, side)"
            )
            conn.commit()

    def _meta_key(self, key: str) -> str:
        """Namespace meta keys per account so breach state is never shared across accounts."""
        return f"{self.account_number}:{key}"

    def _meta_get(self, conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (self._meta_key(key),)).fetchone()
        return row[0] if row else default

    def _meta_set(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (self._meta_key(key), value),
        )

    def _normalize_side(self, side: Any) -> str:
        if side in (0, "0", "BUY", "buy", "LONG", "long"):
            return "BUY"
        if side in (1, "1", "SELL", "sell", "SHORT", "short"):
            return "SELL"
        s = str(side or "").upper()
        if "BUY" in s or "LONG" in s:
            return "BUY"
        if "SELL" in s or "SHORT" in s:
            return "SELL"
        return "UNKNOWN"

    def _position_key(self, account: int, ticket: Any) -> str:
        return f"{int(account)}:{int(ticket)}"

    def _cleanup(self, conn: sqlite3.Connection) -> None:
        keep_days = int(self.config.get("closed_trades_retention_days", 30))
        cutoff = (self._now() - timedelta(days=keep_days)).date().isoformat()
        conn.execute("DELETE FROM closed_trades WHERE closed_date < ?", (cutoff,))

    def _global_daily_loss_limit(self) -> float:
        try:
            return float(
                self.config.get("global_daily_loss_limit", DEFAULT_CONFIG["global_daily_loss_limit"])
            )
        except Exception:
            return float(DEFAULT_CONFIG["global_daily_loss_limit"])

    def sync_account_positions(self, positions: Iterable[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        now_iso = self._now().isoformat()
        with self._lock, self._connect() as conn:
            current_keys = set()
            for pos in positions or []:
                ticket = pos.get("ticket")
                if ticket in (None, 0, "0"):
                    continue
                try:
                    ticket_int = int(ticket)
                except Exception:
                    continue

                key = self._position_key(self.account_number, ticket_int)
                current_keys.add(key)

                opened_ts = pos.get("time")
                if opened_ts:
                    try:
                        opened_at = datetime.fromtimestamp(float(opened_ts), timezone.utc).isoformat()
                    except Exception:
                        opened_at = now_iso
                else:
                    opened_at = now_iso

                conn.execute(
                    """
                    INSERT INTO open_positions
                    (pos_key, ticket, bot, account, symbol, side, volume, entry_price, opened_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pos_key) DO UPDATE SET
                      bot=excluded.bot,
                      account=excluded.account,
                      symbol=excluded.symbol,
                      side=excluded.side,
                      volume=excluded.volume,
                      entry_price=excluded.entry_price,
                      opened_at=excluded.opened_at,
                      updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        ticket_int,
                        self.bot_name,
                        self.account_number,
                        str(pos.get("symbol", "")),
                        self._normalize_side(pos.get("type")),
                        float(pos.get("volume", 0.0) or 0.0),
                        float(pos.get("price_open", pos.get("price", 0.0)) or 0.0),
                        opened_at,
                        now_iso,
                    ),
                )

            rows = conn.execute(
                "SELECT pos_key FROM open_positions WHERE account=?",
                (self.account_number,),
            ).fetchall()
            existing = {r[0] for r in rows}
            stale = existing - current_keys
            if stale:
                conn.executemany("DELETE FROM open_positions WHERE pos_key=?", [(k,) for k in stale])

            self._cleanup(conn)
            conn.commit()

    def can_open(self, symbol: str, direction: str) -> Tuple[bool, str]:
        if not self.enabled:
            return True, "portfolio_guard_disabled"

        symbol = str(symbol)
        side = self._normalize_side(direction)
        now = self._now()
        today = now.date().isoformat()

        with self._lock, self._connect() as conn:
            self._cleanup(conn)

            daily_pnl_row = conn.execute(
                "SELECT COALESCE(SUM(profit), 0.0) AS pnl FROM closed_trades WHERE closed_date=? AND account=?",
                (today, self.account_number),
            ).fetchone()
            daily_pnl = float(daily_pnl_row["pnl"] if daily_pnl_row else 0.0)
            loss_limit = self._global_daily_loss_limit()

            breach_until_raw = self._meta_get(conn, "breach_until", None)
            if breach_until_raw:
                try:
                    breach_until = datetime.fromisoformat(breach_until_raw)
                    if breach_until.tzinfo is None:
                        breach_until = breach_until.replace(tzinfo=timezone.utc)
                    if now < breach_until:
                        reason = self._meta_get(conn, "breach_reason", "cooldown_active") or "cooldown_active"
                        # If the limit was relaxed for demo use, clear any stale daily-loss breach immediately.
                        if reason.startswith("global_daily_loss_limit_hit") and daily_pnl > loss_limit:
                            conn.execute("DELETE FROM meta WHERE key=?", (self._meta_key("breach_until"),))
                            conn.execute("DELETE FROM meta WHERE key=?", (self._meta_key("breach_reason"),))
                            conn.commit()
                        else:
                            return False, f"{reason} until {breach_until.isoformat()}"
                except Exception:
                    pass

            if daily_pnl <= loss_limit:
                cooldown = int(self.config.get("cooldown_minutes_after_breach", 30))
                until = now + timedelta(minutes=max(0, cooldown))
                reason = f"global_daily_loss_limit_hit ({daily_pnl:.2f} <= {loss_limit:.2f})"
                self._meta_set(conn, "breach_until", until.isoformat())
                self._meta_set(conn, "breach_reason", reason)
                conn.commit()
                return False, reason

            total_open = int(conn.execute(
                "SELECT COUNT(*) FROM open_positions WHERE account=?",
                (self.account_number,),
            ).fetchone()[0])
            max_total = int(self.config.get("max_total_positions", 8))
            if max_total > 0 and total_open >= max_total:
                return False, f"max_total_positions reached ({total_open}/{max_total})"

            bot_open = int(
                conn.execute(
                    "SELECT COUNT(*) FROM open_positions WHERE bot=? AND account=?",
                    (self.bot_name, self.account_number),
                ).fetchone()[0]
            )
            max_bot = int(self.config.get("max_positions_per_bot", 4))
            if max_bot > 0 and bot_open >= max_bot:
                return False, f"max_positions_per_bot reached ({bot_open}/{max_bot})"

            symbol_open = int(
                conn.execute(
                    "SELECT COUNT(*) FROM open_positions WHERE symbol=? AND account=?",
                    (symbol, self.account_number),
                ).fetchone()[0]
            )
            max_symbol = int(self.config.get("max_positions_per_symbol", 3))
            if max_symbol > 0 and symbol_open >= max_symbol:
                return False, f"max_positions_per_symbol reached ({symbol_open}/{max_symbol})"

            if side in ("BUY", "SELL"):
                same_side = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM open_positions WHERE symbol=? AND side=? AND account=?",
                        (symbol, side, self.account_number),
                    ).fetchone()[0]
                )
                max_same_side = int(self.config.get("max_same_direction_per_symbol", 2))
                if max_same_side > 0 and same_side >= max_same_side:
                    return False, f"max_same_direction_per_symbol reached ({same_side}/{max_same_side})"

            return True, "OK"

    def on_trade_open(self, ticket: int, symbol: str, side: str, volume: float) -> None:
        if not self.enabled:
            return
        if ticket in (None, 0):
            return
        now_iso = self._now().isoformat()
        key = self._position_key(self.account_number, ticket)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO open_positions
                (pos_key, ticket, bot, account, symbol, side, volume, entry_price, opened_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pos_key) DO UPDATE SET
                  symbol=excluded.symbol,
                  side=excluded.side,
                  volume=excluded.volume,
                  updated_at=excluded.updated_at
                """,
                (
                    key,
                    int(ticket),
                    self.bot_name,
                    self.account_number,
                    str(symbol),
                    self._normalize_side(side),
                    float(volume or 0.0),
                    0.0,
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()

    def on_trade_close(self, ticket: int, profit: float, duration_seconds: float = 0.0, symbol: Optional[str] = None) -> None:
        if not self.enabled:
            return
        now = self._now()
        now_iso = now.isoformat()
        key = self._position_key(self.account_number, ticket)
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM open_positions WHERE pos_key=?", (key,))
            conn.execute(
                """
                INSERT INTO closed_trades
                (closed_at, closed_date, bot, account, ticket, symbol, profit, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    now.date().isoformat(),
                    self.bot_name,
                    self.account_number,
                    int(ticket) if ticket not in (None, 0) else None,
                    symbol,
                    float(profit or 0.0),
                    float(duration_seconds or 0.0),
                ),
            )
            self._cleanup(conn)
            conn.commit()

    def get_snapshot(self) -> Dict[str, Any]:
        now = self._now()
        today = now.date().isoformat()
        with self._lock, self._connect() as conn:
            total_open = int(conn.execute(
                "SELECT COUNT(*) FROM open_positions WHERE account=?",
                (self.account_number,),
            ).fetchone()[0])
            bot_open = int(
                conn.execute(
                    "SELECT COUNT(*) FROM open_positions WHERE bot=? AND account=?",
                    (self.bot_name, self.account_number),
                ).fetchone()[0]
            )
            daily_pnl = float(
                conn.execute(
                    "SELECT COALESCE(SUM(profit), 0.0) FROM closed_trades WHERE closed_date=? AND account=?",
                    (today, self.account_number),
                ).fetchone()[0]
            )
            breach_until = self._meta_get(conn, "breach_until", None)
            breach_reason = self._meta_get(conn, "breach_reason", None)

        cooling_down = False
        if breach_until:
            try:
                dt = datetime.fromisoformat(breach_until)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                cooling_down = now < dt
            except Exception:
                cooling_down = False

        return {
            "enabled": self.enabled,
            "bot": self.bot_name,
            "account": self.account_number,
            "daily_pnl": daily_pnl,
            "global_daily_loss_limit": self._global_daily_loss_limit(),
            "open_positions_total": total_open,
            "open_positions_this_bot": bot_open,
            "max_total_positions": int(self.config.get("max_total_positions", 8)),
            "max_positions_per_bot": int(self.config.get("max_positions_per_bot", 4)),
            "cooling_down": cooling_down,
            "breach_until": breach_until,
            "breach_reason": breach_reason,
            "timestamp": now.isoformat(),
        }
