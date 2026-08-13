"""MLB schedule collector -- pulls from the free MLB Stats API
(trade_bot/mlb_stats_client.py) and stores into mlb_data.db
(trade_bot/mlb_data.py), independent of Kalshi and both trading bots.

Two modes:

    uv run python run_mlb_data_collector.py --backfill --start-date 2026-03-25
        One-shot: walks every date from --start-date through today, pulling
        that day's completed/scheduled games (final scores, probable
        pitchers). Run once at setup to seed the current season. Team form
        (win%, last-10, home/away splits, run differential) is NOT synced
        separately -- it's computed on demand from these games rows, see
        MlbDataStore.get_team_form_asof and mlb_data.py's module docstring
        for why.

    uv run python run_mlb_data_collector.py
        Always-on loop (deploy/mlb-data-collector.service): every
        POLL_INTERVAL_SECONDS, re-pulls today's + the next few days' games
        (catches newly-final games, updated probable pitchers, newly
        scheduled games). Games change slowly (a handful of times a day at
        most) so this polls far less aggressively than the BTC price
        collector.

Regular-season games only (gameType == "R") -- spring training/postseason/
All-Star games are excluded so the training set in fit_mlb_model.py reflects
one consistent competitive incentive structure.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import date, datetime, timedelta, timezone

import httpx

from trade_bot.mlb_data import MlbDataStore
from trade_bot.mlb_stats_client import fetch_schedule, fetch_teams

POLL_INTERVAL_SECONDS = 3600.0  # games change slowly; hourly is plenty
LOOKAHEAD_DAYS = 5
# A fast backfill pace (0.3s) triggered what looks like temporary rate-
# limiting from this API against this project's deploy server IP -- even a
# previously-fine endpoint started 406ing broadly afterward, recovering only
# for the last couple of dates in that run. 2s/request is far more
# conservative; mlb_stats_client._get also retries-with-backoff on 406 as a
# second layer.
BACKFILL_SLEEP_SECONDS = 2.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class _StopRequested(BaseException):
    """Same pattern as run_btc_price_collector.py's -- subclasses
    BaseException so a per-cycle try/except can't accidentally swallow a
    real shutdown request."""


def _handle_sigterm(signum, frame) -> None:
    raise _StopRequested()


def _sync_teams(client: httpx.Client, store: MlbDataStore) -> None:
    for t in fetch_teams(client):
        store.upsert_team(t["id"], t.get("name", ""), t.get("abbreviation", ""))


def _sync_date(client: httpx.Client, store: MlbDataStore, date_str: str) -> int:
    """Pulls one date's games (regular season only). Returns games_upserted.
    Team form is computed on demand from `games` (see MlbDataStore.
    get_team_form_asof), not synced separately -- see mlb_data.py's module
    docstring for why the Stats API's historical /standings?date= endpoint
    is avoided entirely here."""
    games = [g for g in fetch_schedule(client, date_str) if g.get("gameType") == "R"]
    for g in games:
        store.upsert_game(g)
    return len(games)


def run_backfill(start_date: str) -> None:
    store = MlbDataStore()
    with httpx.Client(timeout=15.0) as client:
        print("Syncing team list...")
        _sync_teams(client, store)

        start = date.fromisoformat(start_date)
        today = datetime.now(timezone.utc).date()
        d = start
        total_games, total_days = 0, 0
        while d <= today:
            date_str = d.isoformat()
            n_games = _sync_date(client, store, date_str)
            total_games += n_games
            total_days += 1
            if n_games:
                print(f"  {date_str}: {n_games} games")
            d += timedelta(days=1)
            time.sleep(BACKFILL_SLEEP_SECONDS)

        print(
            f"\nBackfill complete: {total_days} days walked, {total_games} regular-season games upserted, "
            f"{store.completed_game_count()} now marked Final in the db."
        )
    store.close()


def run_loop() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    store = MlbDataStore()
    print(f"Collecting MLB schedule every {POLL_INTERVAL_SECONDS:.0f}s.")
    print(f"Writing to {store.db_path}")

    try:
        with httpx.Client(timeout=15.0) as client:
            _sync_teams(client, store)
            while True:
                cycle_start = time.monotonic()
                today = datetime.now(timezone.utc).date()
                try:
                    for offset in range(LOOKAHEAD_DAYS):
                        d = today + timedelta(days=offset)
                        _sync_date(client, store, d.isoformat())
                    logger.info("Synced today + next %d days of MLB schedule.", LOOKAHEAD_DAYS - 1)
                except Exception as e:
                    logger.warning("MLB data sync cycle failed (will retry next cycle): %s", e)
                elapsed = time.monotonic() - cycle_start
                time.sleep(max(0.0, POLL_INTERVAL_SECONDS - elapsed))
    except (KeyboardInterrupt, _StopRequested):
        print("\nStopped.")
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", action="store_true", help="One-shot historical backfill, then exit.")
    parser.add_argument("--start-date", default="2026-03-25", help="Backfill start date (YYYY-MM-DD).")
    args = parser.parse_args()

    if args.backfill:
        run_backfill(args.start_date)
    else:
        run_loop()


if __name__ == "__main__":
    main()
