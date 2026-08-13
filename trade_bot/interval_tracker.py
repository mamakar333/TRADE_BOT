"""Ground-truth tracking for every 15-minute BTC/ETH/BNB market interval this
bot evaluates -- whether or not it ever placed a trade on it.

Added 2026-08-05 per explicit user request: the live ledger only has ~200
trades to learn from. Every 15-minute rotating-single ticker this bot looks
at, traded or not, carries a real signal (the order-book-imbalance lean, and
DataDrivenCryptoStrategy's predicted win probability where available) and a
real eventual outcome (Kalshi's own settlement result). Recording both for
EVERY interval, not just the ones that became a trade, gives a much larger
dataset to check the algorithm's actual directional accuracy against --
this is the table behind the "final decision vs actual result" comparison
the user asked for.

Best-effort and fully isolated from trading, same contract as
notifications.py/push.py: every public method catches its own errors and
never raises into live_engine.py's run_once() loop. A bug in this file must
never be able to delay, skip, or corrupt a real trading decision.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .client import KalshiAPIError, KalshiClient
from .data import get_market
from .strategy import MarketSnapshot, StrategyDecision

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "live_trading.db"

# Scoped to exactly the three rotating single-strike 15-minute series the
# user is asking about. Each ticker in these series IS one interval (a fresh
# ticker every 15 minutes, never reused), so the ticker itself is the
# interval key -- no separate time-bucketing needed.
TRACKED_SERIES_PREFIXES: dict[str, str] = {
    "KXBTC15M-": "BTC",
    "KXETH15M-": "ETH",
    "KXBNB15M-": "BNB",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS interval_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    asset TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    close_time TEXT,
    last_yes_bid_pct REAL,
    last_yes_ask_pct REAL,
    last_imbalance REAL,
    bot_lean TEXT,
    last_predicted_probability REAL,
    last_signal TEXT,
    bot_traded INTEGER NOT NULL DEFAULT 0,
    trade_side TEXT,
    trade_realized_pnl REAL,
    result TEXT,
    settled_at TEXT,
    lean_correct INTEGER,
    trade_correct INTEGER
);
CREATE INDEX IF NOT EXISTS idx_interval_outcomes_asset_settled
    ON interval_outcomes (asset, settled_at);
CREATE INDEX IF NOT EXISTS idx_interval_outcomes_pending
    ON interval_outcomes (result, close_time);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tracked_asset(ticker: str) -> str | None:
    for prefix, asset in TRACKED_SERIES_PREFIXES.items():
        if ticker.startswith(prefix):
            return asset
    return None


@dataclass
class IntervalOutcome:
    id: int
    ticker: str
    asset: str
    first_seen_at: str
    last_seen_at: str
    close_time: str | None
    last_yes_bid_pct: float | None
    last_yes_ask_pct: float | None
    last_imbalance: float | None
    bot_lean: str | None
    last_predicted_probability: float | None
    last_signal: str | None
    bot_traded: int
    trade_side: str | None
    trade_realized_pnl: float | None
    result: str | None
    settled_at: str | None
    lean_correct: int | None
    trade_correct: int | None


class IntervalTracker:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._logger = logging.getLogger(__name__)

    def close(self) -> None:
        self._conn.close()

    def record(
        self,
        snapshot: MarketSnapshot,
        decision: StrategyDecision,
        trade_side: str | None = None,
        trade_realized_pnl: float | None = None,
    ) -> None:
        """Upsert the latest evaluation for one ticker. Call once per cycle
        for every tracked ticker, regardless of what signal fired.
        `bot_lean`/`last_predicted_probability` freeze on the FIRST non-null
        observation and never get overwritten after that -- NOT the most
        recent one. `trade_side`/`trade_realized_pnl` should be the ticker's
        most recent ledger trade (open or closed), if any -- pass None when
        the bot never traded this ticker. Never raises.

        Diagnosed live 2026-08-05: this originally kept overwriting the lean
        with the latest orderbook imbalance every cycle, on the theory that
        the row right before a market goes terminal would hold the bot's
        "final" read. In practice that made lean_correct nearly tautological
        -- 21/21 settled intervals came back "correct" -- because these
        short-duration threshold contracts' order books mechanically
        converge toward the true outcome in their last seconds as
        uncertainty resolves, regardless of whether the bot's own signal had
        any real predictive value earlier on. Freezing on first observation
        instead captures what the bot could actually have ACTED on, which is
        the only version of this comparison that means anything -- it's also
        exactly how `trade_side` already behaved (set once, at entry, never
        changed), so this just makes `bot_lean` consistent with it.
        """
        asset = tracked_asset(snapshot.ticker)
        if asset is None:
            return
        try:
            self._record(snapshot, decision, asset, trade_side, trade_realized_pnl)
        except Exception:
            self._logger.exception(
                "interval_tracker.record failed for %s -- ignoring, trading unaffected", snapshot.ticker
            )

    def _record(
        self,
        snapshot: MarketSnapshot,
        decision: StrategyDecision,
        asset: str,
        trade_side: str | None,
        trade_realized_pnl: float | None,
    ) -> None:
        row = self._conn.execute(
            "SELECT * FROM interval_outcomes WHERE ticker = ?", (snapshot.ticker,)
        ).fetchone()
        existing = dict(row) if row else None

        imbalance = snapshot.orderbook_imbalance
        lean = existing["bot_lean"] if existing else None
        if lean is None and imbalance is not None and imbalance != 0:
            lean = "YES" if imbalance > 0 else "NO"

        last_prob = existing["last_predicted_probability"] if existing else None
        if last_prob is None and decision.predicted_probability is not None:
            last_prob = decision.predicted_probability

        bot_traded = 1 if (trade_side is not None or (existing and existing["bot_traded"])) else 0
        final_trade_side = trade_side or (existing["trade_side"] if existing else None)
        final_trade_pnl = (
            trade_realized_pnl
            if trade_realized_pnl is not None
            else (existing["trade_realized_pnl"] if existing else None)
        )

        result = (snapshot.result or "").upper() or None
        if result is None and existing:
            result = existing["result"]
        settled_at = existing["settled_at"] if existing else None
        if result is not None and settled_at is None:
            settled_at = _now()

        lean_correct = None
        trade_correct = None
        if result is not None:
            if lean is not None:
                lean_correct = 1 if lean == result else 0
            if final_trade_side is not None:
                trade_correct = 1 if final_trade_side == result else 0

        now = _now()
        first_seen = existing["first_seen_at"] if existing else now
        close_time = (
            snapshot.close_time.isoformat()
            if snapshot.close_time
            else (existing["close_time"] if existing else None)
        )

        self._conn.execute(
            """INSERT INTO interval_outcomes (
                   ticker, asset, first_seen_at, last_seen_at, close_time,
                   last_yes_bid_pct, last_yes_ask_pct, last_imbalance, bot_lean,
                   last_predicted_probability, last_signal, bot_traded, trade_side,
                   trade_realized_pnl, result, settled_at, lean_correct, trade_correct
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                   last_seen_at = excluded.last_seen_at,
                   close_time = excluded.close_time,
                   last_yes_bid_pct = excluded.last_yes_bid_pct,
                   last_yes_ask_pct = excluded.last_yes_ask_pct,
                   last_imbalance = excluded.last_imbalance,
                   bot_lean = excluded.bot_lean,
                   last_predicted_probability = excluded.last_predicted_probability,
                   last_signal = excluded.last_signal,
                   bot_traded = excluded.bot_traded,
                   trade_side = excluded.trade_side,
                   trade_realized_pnl = excluded.trade_realized_pnl,
                   result = excluded.result,
                   settled_at = excluded.settled_at,
                   lean_correct = excluded.lean_correct,
                   trade_correct = excluded.trade_correct
            """,
            (
                snapshot.ticker, asset, first_seen, now, close_time,
                snapshot.yes_bid_pct, snapshot.yes_ask_pct, imbalance, lean,
                last_prob, decision.signal.value, bot_traded, final_trade_side,
                final_trade_pnl, result, settled_at, lean_correct, trade_correct,
            ),
        )
        self._conn.commit()

    def reconcile_pending(self, client: KalshiClient, limit: int = 5) -> None:
        """A ticker can rotate off the watchlist (and out of every open
        position/ledger row) before Kalshi finishes finalizing it, so
        run_once()'s normal per-cycle `record()` calls may never see its
        settlement. This directly re-fetches a small batch of still-
        unsettled, already-past-close tickers each cycle so every interval
        eventually gets a result -- same "exchange is authoritative,
        reconcile against it directly" pattern live_engine.py already uses
        for real positions. Bounded to `limit` per call and only considers
        tickers at least 2 minutes past their close_time (settlement isn't
        instant) so this never turns into an unbounded API burst. Never
        raises."""
        try:
            self._reconcile_pending(client, limit)
        except Exception:
            self._logger.exception("interval_tracker.reconcile_pending failed -- ignoring, trading unaffected")

    def _reconcile_pending(self, client: KalshiClient, limit: int) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        rows = self._conn.execute(
            """SELECT ticker FROM interval_outcomes
               WHERE result IS NULL AND close_time IS NOT NULL AND close_time < ?
               ORDER BY close_time ASC LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        for row in rows:
            ticker = row["ticker"]
            try:
                m = get_market(client, ticker)
            except KalshiAPIError:
                continue
            result = (m.get("result") or "").upper() or None
            if result is None:
                continue
            existing = self._conn.execute(
                "SELECT bot_lean, trade_side FROM interval_outcomes WHERE ticker = ?", (ticker,)
            ).fetchone()
            lean_correct = None if not existing["bot_lean"] else (1 if existing["bot_lean"] == result else 0)
            trade_correct = None if not existing["trade_side"] else (1 if existing["trade_side"] == result else 0)
            self._conn.execute(
                """UPDATE interval_outcomes
                   SET result = ?, settled_at = ?, lean_correct = ?, trade_correct = ?
                   WHERE ticker = ?""",
                (result, _now(), lean_correct, trade_correct, ticker),
            )
            self._conn.commit()

    def get_recent(self, limit: int = 100, asset: str | None = None) -> list[IntervalOutcome]:
        if asset:
            rows = self._conn.execute(
                "SELECT * FROM interval_outcomes WHERE asset = ? ORDER BY last_seen_at DESC LIMIT ?",
                (asset, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM interval_outcomes ORDER BY last_seen_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [IntervalOutcome(**dict(r)) for r in rows]

    def get_summary_stats(self) -> dict:
        """Aggregate accuracy across all settled intervals: `lean_*` covers
        every evaluated interval (traded or not) using the EARLIEST observed
        order-book-imbalance lean for that ticker (see record()'s docstring
        for why it must be the earliest, not the latest); `trade_*` covers
        only the subset that actually became a real trade. Comparing the two
        is the point -- if lean accuracy is meaningfully higher than trade
        accuracy, the entry/stop logic is throwing away edge the raw signal
        already had; if they're close, the signal itself is the thing to
        improve. (Historical rows recorded before 2026-08-05's freeze-on-
        first-observation fix have an inflated, near-tautological
        lean_correct and should be discounted -- they'll age out as new,
        correctly-recorded intervals accumulate.)"""
        row = self._conn.execute(
            """SELECT
                   COUNT(*) AS settled_count,
                   SUM(CASE WHEN lean_correct IS NOT NULL THEN 1 ELSE 0 END) AS lean_evaluated_count,
                   SUM(CASE WHEN lean_correct = 1 THEN 1 ELSE 0 END) AS lean_correct_count,
                   SUM(CASE WHEN bot_traded = 1 THEN 1 ELSE 0 END) AS traded_count,
                   SUM(CASE WHEN bot_traded = 1 AND trade_correct = 1 THEN 1 ELSE 0 END) AS trade_correct_count
               FROM interval_outcomes WHERE result IS NOT NULL"""
        ).fetchone()
        return dict(row)
