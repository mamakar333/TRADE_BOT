"""Append-only audit ledger for live (real-money) trades.

This is deliberately NOT the source of truth for cash or positions -- the
Kalshi account itself is, queried fresh via `data.get_available_balance_dollars`
and `data.get_real_positions` every cycle (see live_engine.py). This ledger
exists only to: (1) give the operator/dashboard a readable history of what
the bot actually did and why, (2) let PerformanceGovernor compute trailing
win rate from LIVE results specifically -- never mixed with paper-trading
results, which run under completely different (much larger, unvalidated)
sizing, and (3) let the daily kill-switch sum today's realized P&L.

Separate database file from paper_trading.db on purpose: a bug in one must
never be able to silently misreport or corrupt the other.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "live_trading.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('YES', 'NO')),
    quantity INTEGER NOT NULL,
    entry_price_pct REAL NOT NULL,
    exit_price_pct REAL,
    entry_fee REAL NOT NULL,
    exit_fee REAL,
    realized_pnl REAL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    strategy_name TEXT NOT NULL,
    open_order_id TEXT,
    close_order_id TEXT,
    open_client_order_id TEXT,
    close_client_order_id TEXT,
    close_reason TEXT,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed'))
);
CREATE INDEX IF NOT EXISTS idx_live_trades_ticker_status ON live_trades (ticker, status);
CREATE INDEX IF NOT EXISTS idx_live_trades_strategy ON live_trades (strategy_name, closed_at);

CREATE TABLE IF NOT EXISTS live_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT NOT NULL
);

-- Single-row table: the OnlineLogisticLearner's current weights (see
-- online_learner.py). Persisted here so learning survives a bot restart --
-- unlike per-ticker price history, which intentionally resets each time.
CREATE TABLE IF NOT EXISTS learner_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Single-row table: live, user-controlled risk toggles that must be visible
-- across processes (dashboard.service and api.service both WRITE this, the
-- live trading process READS it every cycle) without a restart or redeploy
-- -- same "shared sqlite file is the cross-process source of truth" pattern
-- as learner_state above. Added 2026-08-05 for the "small bets only" cap
-- (see LiveExecutionEngine._act's SMALL_BETS_ONLY_CAP_DOLLARS). Also holds
-- btc_dip_enabled (added 2026-08-07, see DataDrivenCryptoStrategy's
-- btc_dip_* fields) -- same shared-toggle need, so it lives in the same
-- single-row table rather than a second one.
CREATE TABLE IF NOT EXISTS risk_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    small_bets_only INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LiveTrade:
    id: int
    ticker: str
    side: str
    quantity: int
    entry_price_pct: float
    exit_price_pct: float | None
    entry_fee: float
    exit_fee: float | None
    realized_pnl: float | None
    opened_at: str
    closed_at: str | None
    strategy_name: str
    open_order_id: str | None
    close_order_id: str | None
    open_client_order_id: str | None
    close_client_order_id: str | None
    close_reason: str | None
    status: str
    features_json: str | None = None
    entry_kind: str | None = None


class LiveLedger:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        # Added 2026-08-03 for the online learner -- ALTER TABLE ADD COLUMN
        # rather than dropping/recreating, so existing trade history (and
        # any live process's open connection) is never disturbed.
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(live_trades)").fetchall()}
        if "features_json" not in cols:
            self._conn.execute("ALTER TABLE live_trades ADD COLUMN features_json TEXT")
        # Added 2026-08-04: tags which entry path opened a position (e.g.
        # "scalp", "gamble") so exit logic can apply different rules per
        # entry type -- see strategy.py's _evaluate_special_exit. Kept
        # separate from features_json on purpose: that blob feeds the online
        # learner's dot-product directly (see OnlineLogisticLearner.update),
        # and a non-numeric string value there would break it.
        if "entry_kind" not in cols:
            self._conn.execute("ALTER TABLE live_trades ADD COLUMN entry_kind TEXT")
        # Added 2026-08-07: risk_settings already exists on the production
        # DB from the small_bets_only rollout, so CREATE TABLE IF NOT
        # EXISTS alone won't add this column -- same ALTER-in-place pattern
        # as above.
        risk_cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(risk_settings)").fetchall()}
        if "btc_dip_enabled" not in risk_cols:
            self._conn.execute("ALTER TABLE risk_settings ADD COLUMN btc_dip_enabled INTEGER NOT NULL DEFAULT 0")
        # Added 2026-08-12 for the MLB game-winner strategy (see
        # trade_bot/mlb_strategy.py) -- same shared risk_settings table
        # rather than a new one, same reasoning as btc_dip_enabled above.
        # Trade history itself lives fully isolated in mlb_trading.db (see
        # trade_bot/mlb_ledger.py); only this cross-process toggle lives
        # here, reusing the one mechanism already wired end-to-end.
        if "mlb_trading_enabled" not in risk_cols:
            self._conn.execute("ALTER TABLE risk_settings ADD COLUMN mlb_trading_enabled INTEGER NOT NULL DEFAULT 0")

    def close(self) -> None:
        self._conn.close()

    def record_open(
        self,
        ticker: str,
        side: str,
        quantity: int,
        entry_price_pct: float,
        entry_fee: float,
        strategy_name: str,
        order_id: str | None,
        client_order_id: str | None,
        features_json: str | None = None,
        entry_kind: str | None = None,
    ) -> LiveTrade:
        cur = self._conn.execute(
            """INSERT INTO live_trades
               (ticker, side, quantity, entry_price_pct, entry_fee, opened_at,
                strategy_name, open_order_id, open_client_order_id, status, features_json, entry_kind)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
            (ticker, side, quantity, entry_price_pct, entry_fee, _now(), strategy_name, order_id, client_order_id,
             features_json, entry_kind),
        )
        self._conn.commit()
        return self.get_open_trade(ticker)

    def get_open_trade(self, ticker: str) -> LiveTrade | None:
        row = self._conn.execute(
            "SELECT * FROM live_trades WHERE ticker = ? AND status = 'open' ORDER BY id DESC LIMIT 1", (ticker,)
        ).fetchone()
        return LiveTrade(**dict(row)) if row else None

    def get_all_open_trades(self) -> list[LiveTrade]:
        rows = self._conn.execute("SELECT * FROM live_trades WHERE status = 'open'").fetchall()
        return [LiveTrade(**dict(r)) for r in rows]

    def get_last_trade_for_ticker(self, ticker: str) -> LiveTrade | None:
        """Most recent trade for a ticker regardless of open/closed status --
        unlike get_open_trade, which returns None the moment a position
        closes. Added 2026-08-05 for interval_tracker.py, which needs to
        know "did the bot ever trade this interval, and how did it turn
        out" even long after the position that traded it has closed."""
        row = self._conn.execute(
            "SELECT * FROM live_trades WHERE ticker = ? ORDER BY id DESC LIMIT 1", (ticker,)
        ).fetchone()
        return LiveTrade(**dict(row)) if row else None

    def record_close(
        self,
        trade_id: int,
        exit_price_pct: float,
        exit_fee: float,
        realized_pnl: float,
        close_reason: str,
        order_id: str | None,
        client_order_id: str | None,
    ) -> None:
        self._conn.execute(
            """UPDATE live_trades
               SET exit_price_pct = ?, exit_fee = ?, realized_pnl = ?, closed_at = ?,
                   close_reason = ?, close_order_id = ?, close_client_order_id = ?, status = 'closed'
               WHERE id = ?""",
            (exit_price_pct, exit_fee, realized_pnl, _now(), close_reason, order_id, client_order_id, trade_id),
        )
        self._conn.commit()

    def get_closed_trades(self, limit: int = 500, offset: int = 0) -> list[LiveTrade]:
        rows = self._conn.execute(
            "SELECT * FROM live_trades WHERE status = 'closed' ORDER BY closed_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [LiveTrade(**dict(r)) for r in rows]

    def get_total_realized_pnl(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM live_trades WHERE status = 'closed'"
        ).fetchone()
        return float(row["total"])

    def get_recent_events(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM live_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_recent_realized_pnls(self, strategy_name: str, limit: int) -> list[float]:
        rows = self._conn.execute(
            """SELECT realized_pnl FROM
               (SELECT realized_pnl, closed_at FROM live_trades
                WHERE strategy_name = ? AND status = 'closed'
                ORDER BY closed_at DESC LIMIT ?)
               ORDER BY closed_at ASC""",
            (strategy_name, limit),
        ).fetchall()
        return [r["realized_pnl"] for r in rows]

    def get_recent_realized_pnls_within_hours(self, strategy_name: str, limit: int, hours: float) -> list[float]:
        """Same as get_recent_realized_pnls, but only trades closed within
        the last `hours` count toward the trailing window. Without this, a
        strategy that PerformanceGovernor pauses can get permanently stuck:
        pausing blocks new entries, no new entries means no new closed
        trades, and no new closed trades means the trailing window can never
        shift away from the old data that triggered the pause in the first
        place. Diagnosed 2026-08-03 -- the governor paused at 05:20 UTC on a
        pre-fix losing streak and was still blocking every entry 8+ hours
        later with zero new trades since, because nothing could ever refresh
        its window. Time-bounding the window means stale data eventually
        ages out and the gate re-opens (to "not enough data yet", full size)
        on its own, rather than staying hostage to trades made under
        parameters that may no longer even be in effect."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self._conn.execute(
            """SELECT realized_pnl FROM
               (SELECT realized_pnl, closed_at FROM live_trades
                WHERE strategy_name = ? AND status = 'closed' AND closed_at >= ?
                ORDER BY closed_at DESC LIMIT ?)
               ORDER BY closed_at ASC""",
            (strategy_name, cutoff, limit),
        ).fetchall()
        return [r["realized_pnl"] for r in rows]

    def get_realized_pnl_since(self, iso_timestamp: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM live_trades "
            "WHERE status = 'closed' AND closed_at >= ?",
            (iso_timestamp,),
        ).fetchone()
        return float(row["total"])

    def log_event(self, event: str, detail: str) -> None:
        self._conn.execute(
            "INSERT INTO live_events (timestamp, event, detail) VALUES (?, ?, ?)", (_now(), event, detail)
        )
        self._conn.commit()

    def save_learner_state(self, state_json: str) -> None:
        self._conn.execute(
            "INSERT INTO learner_state (id, state_json, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at",
            (state_json, _now()),
        )
        self._conn.commit()

    def load_learner_state(self) -> str | None:
        row = self._conn.execute("SELECT state_json FROM learner_state WHERE id = 1").fetchone()
        return row["state_json"] if row else None

    def get_small_bets_only(self) -> bool:
        """Live, cross-process toggle checked every cycle by
        LiveExecutionEngine._act -- False (normal sizing) until a human
        turns it on from the dashboard or app. See risk_settings' schema
        comment above."""
        row = self._conn.execute("SELECT small_bets_only FROM risk_settings WHERE id = 1").fetchone()
        return bool(row["small_bets_only"]) if row else False

    def set_small_bets_only(self, enabled: bool) -> None:
        self._conn.execute(
            "INSERT INTO risk_settings (id, small_bets_only, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET small_bets_only = excluded.small_bets_only, updated_at = excluded.updated_at",
            (1 if enabled else 0, _now()),
        )
        self._conn.commit()
        self.log_event("risk_settings_changed", f"small_bets_only={'on' if enabled else 'off'}")

    def get_btc_dip_enabled(self) -> bool:
        """Live, cross-process toggle for the BTC15M dip-buy mechanism
        (see DataDrivenCryptoStrategy._evaluate_btc_dip_entry) -- checked
        fresh every cycle by LiveExecutionEngine.run_once(), same pattern
        as get_small_bets_only above. False (off) until a human turns it
        on from the dashboard or app."""
        row = self._conn.execute("SELECT btc_dip_enabled FROM risk_settings WHERE id = 1").fetchone()
        return bool(row["btc_dip_enabled"]) if row else False

    def set_btc_dip_enabled(self, enabled: bool) -> None:
        self._conn.execute(
            "INSERT INTO risk_settings (id, btc_dip_enabled, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET btc_dip_enabled = excluded.btc_dip_enabled, updated_at = excluded.updated_at",
            (1 if enabled else 0, _now()),
        )
        self._conn.commit()
        self.log_event("risk_settings_changed", f"btc_dip_enabled={'on' if enabled else 'off'}")

    def get_mlb_trading_enabled(self) -> bool:
        """Live, cross-process toggle for the MLB game-winner strategy (see
        trade_bot/mlb_strategy.py) -- checked fresh every cycle by the MLB
        LiveExecutionEngine instance (run_mlb_trading.py), same pattern as
        get_btc_dip_enabled above. False (off) until a human turns it on
        from the dashboard or app."""
        row = self._conn.execute("SELECT mlb_trading_enabled FROM risk_settings WHERE id = 1").fetchone()
        return bool(row["mlb_trading_enabled"]) if row else False

    def set_mlb_trading_enabled(self, enabled: bool) -> None:
        self._conn.execute(
            "INSERT INTO risk_settings (id, mlb_trading_enabled, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET mlb_trading_enabled = excluded.mlb_trading_enabled, "
            "updated_at = excluded.updated_at",
            (1 if enabled else 0, _now()),
        )
        self._conn.commit()
        self.log_event("risk_settings_changed", f"mlb_trading_enabled={'on' if enabled else 'off'}")
