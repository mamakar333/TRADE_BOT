"""SQLite-backed paper-trading portfolio: cash, open positions, closed trades.

All money fields are stored as REAL dollars (not integer cents) because
Kalshi's own fee accounting goes to $0.0001 precision (see fees.py) --
sub-cent float error here is immaterial for a paper account, and integer
cents would just force fee rounding elsewhere.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "paper_trading.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    starting_balance REAL NOT NULL,
    cash_balance REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('YES', 'NO')),
    quantity INTEGER NOT NULL,
    entry_price_pct REAL NOT NULL,
    entry_fee REAL NOT NULL,
    opened_at TEXT NOT NULL,
    strategy_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS closed_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price_pct REAL NOT NULL,
    exit_price_pct REAL NOT NULL,
    entry_fee REAL NOT NULL,
    exit_fee REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    close_reason TEXT NOT NULL
);
-- entry_kind/features_json added via migration below, same columns as
-- live_ledger.py's live_trades -- kept identical on purpose so paper and
-- live trade history can be compared/analyzed with the same code once
-- there's enough paper data to be worth it (see docs/ALGORITHM.md).

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cash_balance REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_equity REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    yes_bid_pct REAL,
    yes_ask_pct REAL,
    last_price_pct REAL,
    volume REAL
);
CREATE INDEX IF NOT EXISTS idx_price_history_ticker_ts ON price_history (ticker, timestamp);

CREATE TABLE IF NOT EXISTS reset_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cash_before REAL NOT NULL,
    reset_to REAL NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Position:
    id: int
    ticker: str
    side: str  # "YES" | "NO"
    quantity: int
    entry_price_pct: float
    entry_fee: float
    opened_at: str
    strategy_name: str
    # Tags which entry path opened this position (e.g. "scalp", "gamble") --
    # None means the strategy's normal/standard entry logic. Lets exit logic
    # apply different rules per entry type (see strategy.py's
    # _evaluate_special_exit).
    entry_kind: str | None = None
    # Entry-time feature snapshot (see online_learner.py's build_features) --
    # same field live_ledger.py stores, added here 2026-08-04 so paper and
    # live trades are directly comparable later, not just paper's own P&L.
    features_json: str | None = None

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price_pct / 100 + self.entry_fee


@dataclass
class ClosedTrade:
    id: int
    ticker: str
    side: str
    quantity: int
    entry_price_pct: float
    exit_price_pct: float
    entry_fee: float
    exit_fee: float
    realized_pnl: float
    opened_at: str
    closed_at: str
    strategy_name: str
    close_reason: str
    entry_kind: str | None = None
    features_json: str | None = None


class PaperPortfolio:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, starting_balance: float = 1000.0):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL: the execution engine and the Streamlit dashboard are separate processes
        # reading/writing this same file concurrently; WAL lets dashboard reads proceed
        # without blocking on (or being blocked by) the engine's writes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate_schema()
        self._conn.commit()
        self._init_account(starting_balance)

    def _migrate_schema(self) -> None:
        # Added 2026-08-04 for ledger parity with live_ledger.py -- ALTER
        # TABLE ADD COLUMN rather than dropping/recreating, same reasoning
        # as live_ledger.py's own migration (never disturb existing rows or
        # an already-open connection from the running paper bot).
        for table in ("positions", "closed_trades"):
            cols = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if "entry_kind" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN entry_kind TEXT")
            if "features_json" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN features_json TEXT")

    def _init_account(self, starting_balance: float) -> None:
        row = self._conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO account (id, starting_balance, cash_balance, created_at) VALUES (1, ?, ?, ?)",
                (starting_balance, starting_balance, _now()),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- account --------------------------------------------------------

    def get_cash_balance(self) -> float:
        return self._conn.execute("SELECT cash_balance FROM account WHERE id = 1").fetchone()["cash_balance"]

    def get_starting_balance(self) -> float:
        return self._conn.execute("SELECT starting_balance FROM account WHERE id = 1").fetchone()["starting_balance"]

    def _adjust_cash(self, delta: float) -> None:
        self._conn.execute("UPDATE account SET cash_balance = cash_balance + ? WHERE id = 1", (delta,))

    # -- positions --------------------------------------------------------

    def get_open_position(self, ticker: str) -> Position | None:
        row = self._conn.execute("SELECT * FROM positions WHERE ticker = ?", (ticker,)).fetchone()
        return Position(**dict(row)) if row else None

    def get_all_open_positions(self) -> list[Position]:
        rows = self._conn.execute("SELECT * FROM positions ORDER BY opened_at").fetchall()
        return [Position(**dict(r)) for r in rows]

    def get_total_exposure(self) -> float:
        """Sum of cost basis across all open positions."""
        return sum(p.cost_basis for p in self.get_all_open_positions())

    def open_position(
        self,
        ticker: str,
        side: str,
        quantity: int,
        entry_price_pct: float,
        entry_fee: float,
        strategy_name: str,
        features_json: str | None = None,
        entry_kind: str | None = None,
    ) -> Position:
        if self.get_open_position(ticker) is not None:
            raise ValueError(f"already have an open position in {ticker}")
        cost = quantity * entry_price_pct / 100 + entry_fee
        if cost > self.get_cash_balance() + 1e-9:
            raise ValueError(f"insufficient cash: need {cost:.4f}, have {self.get_cash_balance():.4f}")

        opened_at = _now()
        cur = self._conn.execute(
            "INSERT INTO positions (ticker, side, quantity, entry_price_pct, entry_fee, opened_at, strategy_name, "
            "features_json, entry_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, side, quantity, entry_price_pct, entry_fee, opened_at, strategy_name, features_json, entry_kind),
        )
        self._adjust_cash(-cost)
        self._conn.commit()
        return Position(
            cur.lastrowid, ticker, side, quantity, entry_price_pct, entry_fee, opened_at, strategy_name,
            entry_kind=entry_kind, features_json=features_json,
        )

    def close_position(self, ticker: str, exit_price_pct: float, exit_fee: float, close_reason: str) -> ClosedTrade:
        position = self.get_open_position(ticker)
        if position is None:
            raise ValueError(f"no open position in {ticker}")

        proceeds = position.quantity * exit_price_pct / 100 - exit_fee
        realized_pnl = proceeds - position.cost_basis
        closed_at = _now()

        cur = self._conn.execute(
            "INSERT INTO closed_trades (ticker, side, quantity, entry_price_pct, exit_price_pct, entry_fee, "
            "exit_fee, realized_pnl, opened_at, closed_at, strategy_name, close_reason, features_json, entry_kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                position.ticker,
                position.side,
                position.quantity,
                position.entry_price_pct,
                exit_price_pct,
                position.entry_fee,
                exit_fee,
                realized_pnl,
                position.opened_at,
                closed_at,
                position.strategy_name,
                close_reason,
                position.features_json,
                position.entry_kind,
            ),
        )
        self._conn.execute("DELETE FROM positions WHERE id = ?", (position.id,))
        self._adjust_cash(proceeds)
        self._conn.commit()

        return ClosedTrade(
            cur.lastrowid,
            position.ticker,
            position.side,
            position.quantity,
            position.entry_price_pct,
            exit_price_pct,
            position.entry_fee,
            exit_fee,
            realized_pnl,
            position.opened_at,
            closed_at,
            position.strategy_name,
            close_reason,
            entry_kind=position.entry_kind,
            features_json=position.features_json,
        )

    # -- closed trades / history --------------------------------------------------------

    def get_closed_trades(self, limit: int = 500, offset: int = 0) -> list[ClosedTrade]:
        rows = self._conn.execute(
            "SELECT * FROM closed_trades ORDER BY closed_at DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [ClosedTrade(**dict(r)) for r in rows]

    def get_realized_pnl_since(self, since_iso: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM closed_trades WHERE closed_at >= ?", (since_iso,)
        ).fetchone()
        return row["total"]

    def get_total_realized_pnl(self) -> float:
        row = self._conn.execute("SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM closed_trades").fetchone()
        return float(row["total"])

    # -- equity curve --------------------------------------------------------

    def record_equity_snapshot(self, positions_value: float) -> None:
        cash = self.get_cash_balance()
        self._conn.execute(
            "INSERT INTO equity_snapshots (timestamp, cash_balance, positions_value, total_equity) "
            "VALUES (?, ?, ?, ?)",
            (_now(), cash, positions_value, cash + positions_value),
        )
        self._conn.commit()

    def get_equity_curve(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM equity_snapshots ORDER BY timestamp").fetchall()
        return [dict(r) for r in rows]

    # -- price history (for sparklines / detail-page charts) --------------------------------------------------------

    def record_price_tick(
        self,
        ticker: str,
        yes_bid_pct: float | None,
        yes_ask_pct: float | None,
        last_price_pct: float | None,
        volume: float | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO price_history (ticker, timestamp, yes_bid_pct, yes_ask_pct, last_price_pct, volume) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ticker, _now(), yes_bid_pct, yes_ask_pct, last_price_pct, volume),
        )
        self._conn.commit()

    def get_price_history(self, ticker: str, limit: int = 500) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM price_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?", (ticker, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def prune_price_history(self, keep_per_ticker: int = 2000) -> None:
        """Call occasionally from the engine loop -- unbounded growth over a
        24/7 loop would otherwise be the one thing that isn't self-limiting."""
        self._conn.execute(
            """
            DELETE FROM price_history WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY ticker ORDER BY timestamp DESC
                    ) AS rn
                    FROM price_history
                ) WHERE rn > ?
            )
            """,
            (keep_per_ticker,),
        )
        self._conn.commit()

    # -- bust reset --------------------------------------------------------

    def reset_if_busted(self) -> bool:
        """If cash has hit zero (or gone negative), top back up to the
        starting balance so a 24/7 loop can keep learning instead of
        stalling out. Returns True if a reset happened."""
        cash = self.get_cash_balance()
        if cash > 0:
            return False
        starting = self.get_starting_balance()
        self._conn.execute(
            "INSERT INTO reset_events (timestamp, cash_before, reset_to) VALUES (?, ?, ?)",
            (_now(), cash, starting),
        )
        self._adjust_cash(starting - cash)
        self._conn.commit()
        return True

    def get_reset_events(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM reset_events ORDER BY timestamp").fetchall()
        return [dict(r) for r in rows]
