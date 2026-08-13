"""One-time (well, re-runnable -- upserts) backfill of starting-pitcher
stats into mlb_data.db's pitcher_stats_snapshot table. See that module's
docstring for why this is monthly-granularity, not true per-game as-of.

    uv run python backfill_pitcher_stats.py

For every distinct (pitcher_id, month) pair that appears as a probable
starter in `games`, fetches real stats accumulated through the end of the
PRIOR month and stores them. A pitcher's first-ever month in the dataset is
skipped (no prior data exists to snapshot -- correctly reads as "no signal
yet" via get_pitcher_stats_asof, same convention as every other optional
feature in this pipeline).

SEASON_START matters here beyond just the collector's own backfill start --
it's the beginning of the accumulation window for EVERY snapshot (a
pitcher's May stats-through-April still only accumulate from opening day
forward, not from some arbitrary earlier date).
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import httpx

from trade_bot.mlb_data import MlbDataStore
from trade_bot.mlb_stats_client import fetch_pitcher_stats_byrange

SEASON_START = "2026-03-25"
SLEEP_SECONDS = 0.4


def _prior_month_end(month_first_day: str) -> str:
    d = date.fromisoformat(month_first_day)
    return (d - timedelta(days=1)).isoformat()


def main() -> None:
    store = MlbDataStore()
    games = store.get_completed_games()
    print(f"scanning {len(games)} completed games for probable pitchers...")

    pairs: set[tuple[int, str]] = set()
    for g in games:
        month = g["game_date"][:7] + "-01"
        if g["home_probable_pitcher_id"]:
            pairs.add((g["home_probable_pitcher_id"], month))
        if g["away_probable_pitcher_id"]:
            pairs.add((g["away_probable_pitcher_id"], month))

    # Skip any month equal to the season-start month or earlier -- no prior
    # accumulation window exists yet for those.
    season_start_month = SEASON_START[:7] + "-01"
    pairs = {(pid, month) for pid, month in pairs if month > season_start_month}
    print(f"{len(pairs)} distinct (pitcher, month) snapshots to fetch")

    fetched = skipped = 0
    with httpx.Client(timeout=15.0) as client:
        for i, (pitcher_id, month) in enumerate(sorted(pairs, key=lambda p: p[1])):
            end_date = _prior_month_end(month)
            stat = fetch_pitcher_stats_byrange(client, pitcher_id, SEASON_START, end_date)
            if stat is None:
                skipped += 1
            else:
                store.record_pitcher_stats_snapshot(pitcher_id, month, stat)
                fetched += 1
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(pairs)} processed ({fetched} fetched, {skipped} skipped)")
            time.sleep(SLEEP_SECONDS)

    print(f"\ndone: {fetched} snapshots recorded, {skipped} had no data in range")
    store.close()


if __name__ == "__main__":
    main()
