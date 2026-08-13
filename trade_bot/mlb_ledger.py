"""Append-only audit ledger for MLB (real-money) trades. Mirrors
trade_bot/live_ledger.py's LiveLedger structure and interface closely --
own isolated mlb_trading.db, own mlb_trades/mlb_events/learner_state
tables, same "the exchange account itself is the source of truth for cash/
positions, this is audit history + governor/learner support" contract.

Separate database file from live_trading.db/paper_trading.db/btc_price_history.db
on purpose -- same isolation rationale as every other db in this repo: a
bug in the crypto ledger must never be able to touch MLB trade history or
vice versa, and this owns its OWN OnlineLogisticLearner state (an entirely
different feature schema from crypto's, see trade_bot/mlb_features.py).

The one thing NOT duplicated here: cross-process risk toggles
(mlb_trading_enabled, small_bets_only) still live in live_trading.db's
risk_settings table (see live_ledger.py) -- that's the one already-working
cross-process toggle mechanism (dashboard/api write it, the trading
process reads it fresh every cycle), and duplicating it into a second db
would just be two sources of truth for the same on/off switches. This
class holds an internal LiveLedger purely to delegate those reads.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .live_ledger import LiveLedger

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "mlb_trading.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS mlb_trades (
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
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    features_json TEXT,
    entry_kind TEXT
);
CREATE INDEX IF NOT EXISTS idx_mlb_trades_ticker_status ON mlb_trades (ticker, status);
CREATE INDEX IF NOT EXISTS idx_mlb_trades_strategy ON mlb_trades (strategy_name, closed_at);

CREATE TABLE IF NOT EXISTS mlb_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learner_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MlbTrade:
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


class MlbLedger:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, toggle_source: LiveLedger | None = None):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        # See module docstring -- toggles are NOT duplicated here.
        self._toggle_source = toggle_source or LiveLedger()

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
    ) -> MlbTrade:
        self._conn.execute(
            """INSERT INTO mlb_trades
               (ticker, side, quantity, entry_price_pct, entry_fee, opened_at,
                strategy_name, open_order_id, open_client_order_id, status, features_json, entry_kind)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
            (ticker, side, quantity, entry_price_pct, entry_fee, _now(), strategy_name, order_id, client_order_id,
             features_json, entry_kind),
        )
        self._conn.commit()
        return self.get_open_trade(ticker)

    def get_open_trade(self, ticker: str) -> MlbTrade | None:
        row = self._conn.execute(
            "SELECT * FROM mlb_trades WHERE ticker = ? AND status = 'open' ORDER BY id DESC LIMIT 1", (ticker,)
        ).fetchone()
        return MlbTrade(**dict(row)) if row else None

    def get_all_open_trades(self) -> list[MlbTrade]:
        rows = self._conn.execute("SELECT * FROM mlb_trades WHERE status = 'open'").fetchall()
        return [MlbTrade(**dict(r)) for r in rows]

    def get_last_trade_for_ticker(self, ticker: str) -> MlbTrade | None:
        row = self._conn.execute(
            "SELECT * FROM mlb_trades WHERE ticker = ? ORDER BY id DESC LIMIT 1", (ticker,)
        ).fetchone()
        return MlbTrade(**dict(row)) if row else None

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
            """UPDATE mlb_trades
               SET exit_price_pct = ?, exit_fee = ?, realized_pnl = ?, closed_at = ?,
                   close_reason = ?, close_order_id = ?, close_client_order_id = ?, status = 'closed'
               WHERE id = ?""",
            (exit_price_pct, exit_fee, realized_pnl, _now(), close_reason, order_id, client_order_id, trade_id),
        )
        self._conn.commit()

    def get_closed_trades(self, limit: int = 500, offset: int = 0) -> list[MlbTrade]:
        rows = self._conn.execute(
            "SELECT * FROM mlb_trades WHERE status = 'closed' ORDER BY closed_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [MlbTrade(**dict(r)) for r in rows]

    def get_total_realized_pnl(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM mlb_trades WHERE status = 'closed'"
        ).fetchone()
        return float(row["total"])

    def get_recent_events(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM mlb_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_recent_realized_pnls_within_hours(self, strategy_name: str, limit: int, hours: float) -> list[float]:
        """Same time-bounded trailing-window pattern as LiveLedger's --
        see that class's docstring for why an unbounded window can leave
        PerformanceGovernor permanently stuck paused."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self._conn.execute(
            """SELECT realized_pnl FROM
               (SELECT realized_pnl, closed_at FROM mlb_trades
                WHERE strategy_name = ? AND status = 'closed' AND closed_at >= ?
                ORDER BY closed_at DESC LIMIT ?)
               ORDER BY closed_at ASC""",
            (strategy_name, cutoff, limit),
        ).fetchall()
        return [r["realized_pnl"] for r in rows]

    def get_realized_pnl_since(self, iso_timestamp: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM mlb_trades WHERE status = 'closed' AND closed_at >= ?",
            (iso_timestamp,),
        ).fetchone()
        return float(row["total"])

    def log_event(self, event: str, detail: str) -> None:
        self._conn.execute(
            "INSERT INTO mlb_events (timestamp, event, detail) VALUES (?, ?, ?)", (_now(), event, detail)
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

    # -- delegated cross-process toggles (see module docstring) -----------

    def get_small_bets_only(self) -> bool:
        return self._toggle_source.get_small_bets_only()

    def get_mlb_trading_enabled(self) -> bool:
        return self._toggle_source.get_mlb_trading_enabled()
