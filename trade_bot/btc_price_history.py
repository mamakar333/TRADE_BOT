"""Continuous BTC/USD price collection, independent of Kalshi entirely.

Added 2026-08-10 per explicit user request: graph real BTC price movement
over multiple timeframes (5min through 1 month) to look for patterns. This
is the FIRST external (non-Kalshi) data dependency in this codebase --
deliberate, and confirmed with the user first: Kalshi's own API exposes no
live index/spot price, only a `floor_strike` snapshot captured once when
each 15-min market opens (see docs/ALGORITHM.md-era investigation). Pulls
from Coinbase's free, unauthenticated public spot-price endpoint instead --
not literally "Kalshi's number", but the same underlying asset, close
enough for spotting patterns, and Coinbase prices are one of the real
exchange feeds CF Benchmarks' index (which Kalshi actually settles against)
is itself built from.

Separate database file from live_trading.db/paper_trading.db on purpose:
this is high-frequency (~86,400 rows/day at 1-second resolution) time-
series data, conceptually unrelated to trading, and must never contend for
the same WAL file real trading decisions depend on.
"""
from __future__ import annotations

import logging
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "btc_price_history.db"
COINBASE_SPOT_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"

SCHEMA = """
CREATE TABLE IF NOT EXISTS btc_price_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    price_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_btc_price_ticks_timestamp ON btc_price_ticks (timestamp);
"""

# Every graph window this feature supports, and the real elapsed-time
# duration each one needs before it has enough data to be worth showing --
# see BtcPriceHistory.get_availability. Order matters: this is also the
# canonical listing order for both UIs.
WINDOWS: dict[str, timedelta] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "3h": timedelta(hours=3),
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "1w": timedelta(weeks=1),
    "1mo": timedelta(days=30),
}

# Roughly how many points to return per window, regardless of how many raw
# ticks actually fall in it -- keeps every chart equally cheap to transfer
# and render whether it's a 5-minute or 1-month view. Real bucket width is
# computed per request as window_seconds / this.
TARGET_POINTS = 360


def fetch_btc_price_usd(client: httpx.Client | None = None) -> float | None:
    """One-shot fetch, best-effort. Returns None (never raises) on any
    network/parse failure -- the collector loop must survive Coinbase being
    slow, down, or rate-limiting, same contract as every other outbound
    call in this codebase (see notifications.py, client.py's retry note)."""
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=5.0)
    try:
        resp = client.get(COINBASE_SPOT_URL)
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
        logger.warning("BTC price fetch failed (will retry next tick): %s", e)
        return None
    finally:
        if owns_client:
            client.close()


class BtcPriceHistory:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record_tick(self, price_usd: float, timestamp: datetime | None = None) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        self._conn.execute(
            "INSERT INTO btc_price_ticks (timestamp, price_usd) VALUES (?, ?)", (ts, price_usd)
        )
        self._conn.commit()

    def get_earliest_timestamp(self) -> datetime | None:
        row = self._conn.execute("SELECT MIN(timestamp) AS ts FROM btc_price_ticks").fetchone()
        if row is None or row["ts"] is None:
            return None
        return datetime.fromisoformat(row["ts"])

    def get_availability(self) -> dict[str, bool]:
        """A window is "available" once we've been collecting for at least
        that long -- there's no point showing a flat/misleading "1 month"
        chart built from 3 days of real data. Matches the user's explicit
        "enable these tabs when the data for that time is available"."""
        earliest = self.get_earliest_timestamp()
        if earliest is None:
            return {key: False for key in WINDOWS}
        elapsed = datetime.now(timezone.utc) - earliest
        return {key: elapsed >= duration for key, duration in WINDOWS.items()}

    def get_history(self, window: str) -> list[dict]:
        """Downsampled (bucket-averaged) points for the given window, most-
        recent-covering. Bucketing at query time rather than maintaining
        separate rollup tables -- simplest correct thing for a dataset that
        won't even reach its largest supported window (1mo, ~2.6M raw rows)
        for weeks; worth revisiting if/when that actually gets slow."""
        if window not in WINDOWS:
            raise ValueError(f"unknown window {window!r}, must be one of {list(WINDOWS)}")
        duration = WINDOWS[window]
        cutoff = (datetime.now(timezone.utc) - duration).isoformat()
        bucket_seconds = max(1, int(duration.total_seconds() / TARGET_POINTS))
        rows = self._conn.execute(
            """
            SELECT
                (CAST(strftime('%s', timestamp) AS INTEGER) / ?) * ? AS bucket_epoch,
                AVG(price_usd) AS avg_price,
                MAX(timestamp) AS last_timestamp
            FROM btc_price_ticks
            WHERE timestamp >= ?
            GROUP BY bucket_epoch
            ORDER BY bucket_epoch ASC
            """,
            (bucket_seconds, bucket_seconds, cutoff),
        ).fetchall()
        return [{"t": r["last_timestamp"], "price": r["avg_price"]} for r in rows]

    def get_recent_volatility_pct(self, window_minutes: float = 5.0, min_ticks: int = 30) -> float | None:
        """Realized volatility of the true underlying BTC/USD price over the
        trailing window, as the population stdev of consecutive tick-to-tick
        %-returns. Feeds data_driven_strategy.py's entry model as a cleaner
        signal than the pre-existing `_recent_volatility` (which measures
        noise in the Kalshi CONTRACT's own mid-price over its last 10
        snapshots -- a bounded, low-resolution, once-per-poll-interval proxy)
        -- this is the actual asset moving, sampled every second. Returns
        None below min_ticks so an undersized window (collector just
        started, or a gap) reads as "no signal" rather than a noisy stdev of
        a handful of points."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
        rows = self._conn.execute(
            "SELECT price_usd FROM btc_price_ticks WHERE timestamp >= ? ORDER BY timestamp ASC",
            (cutoff,),
        ).fetchall()
        prices = [r["price_usd"] for r in rows]
        if len(prices) < min_ticks:
            return None
        returns = [(b - a) / a * 100 for a, b in zip(prices, prices[1:]) if a]
        if not returns:
            return None
        return statistics.pstdev(returns)

    def prune_older_than(self, days: float) -> int:
        """Not called anywhere yet (the 1-month window is the longest
        supported, and this dataset is brand new) -- available for when
        retention actually needs bounding. Deletes raw ticks older than
        `days`, returns the number of rows removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.execute("DELETE FROM btc_price_ticks WHERE timestamp < ?", (cutoff,))
        self._conn.commit()
        return cur.rowcount
