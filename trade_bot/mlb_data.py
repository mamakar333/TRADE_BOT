"""MLB team/game/standings data store, built from the free MLB Stats API
(trade_bot/mlb_stats_client.py). Separate database file from every other db
in this repo (live_trading.db/paper_trading.db/btc_price_history.db) --
same isolation rationale as btc_price_history.py's docstring: this is a
distinct data domain and must never contend for a WAL file real trading
decisions elsewhere depend on.

Schema deliberately does NOT include a per-game team/pitcher box-score log,
and deliberately does NOT use the Stats API's /standings?date= endpoint for
historical team form, despite that endpoint returning correct as-of-date
data when tested from a residential/dev network. Confirmed directly against
the real deploy server: /standings without a `date` param 200s fine from
there, but the SAME endpoint WITH a historical `date` param reliably 406s
from that server's IP across every date tried -- some CDN/WAF rule on that
specific query shape, not fixable with headers, and it would break both the
one-time backfill AND every future cycle of the always-on collector (which
runs on that same server). Team form (win%, last-10, home/away splits, run
differential) is instead computed directly from `games` via SQL, strictly
before a cutoff date -- see get_team_form_asof below. This only needs the
plain /schedule endpoint, confirmed reliable from the deploy server, and
removes the dependency on the endpoint that isn't. Head-to-head is derived
from `games` the same way (two teams' completed meetings before a given
date).

Starting-pitcher stats (added 2026-08-13, see pitcher_stats_snapshot below):
true per-game as-of-date precision (fetching stats freshly for every
individual start) would be ~3,600+ API calls for a full-season backfill --
too many given this API's demonstrated rate-limiting. Instead, snapshots
are taken at MONTHLY granularity: a pitcher's real, accurate stats
(fetched via the `byDateRange` endpoint, not the leakier plain `season`
one) accumulated through the end of the prior month, reused for every
start that pitcher makes within the following month. This cuts ~3,600
calls down to ~1,200 (one per distinct pitcher/month pair actually
observed) while keeping the anti-leakage guarantee intact -- a game in
August only ever sees stats accumulated through end of July, never
anything from August itself. Coarser than true per-game precision (a
pitcher's form 25 days into a month reads the same as day 1), an explicit,
disclosed tradeoff, not a hidden one.

All read helpers that exist to build TRAINING features are as-of-only --
they filter strictly on a `before_date` cutoff so a historical game's
features can never see data from its own date or later. This is the
anti-leakage guarantee fit_mlb_model.py depends on.
"""
from __future__ import annotations

import sqlite3
from datetime import date as date_type
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "mlb_data.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    abbreviation TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    game_pk INTEGER PRIMARY KEY,
    game_date TEXT NOT NULL,
    game_date_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    winner_team_id INTEGER,
    home_probable_pitcher_id INTEGER,
    home_probable_pitcher_name TEXT,
    away_probable_pitcher_id INTEGER,
    away_probable_pitcher_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games (game_date);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games (home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games (away_team_id, game_date);

-- One row per (pitcher, calendar month) -- real accumulated stats through
-- the end of the PRIOR month (see module docstring for why monthly, not
-- per-game). `as_of_date` is always the first of the month this snapshot
-- applies to; get_pitcher_stats_asof looks up by (pitcher_id, that month).
CREATE TABLE IF NOT EXISTS pitcher_stats_snapshot (
    pitcher_id INTEGER NOT NULL,
    as_of_month TEXT NOT NULL,
    era REAL,
    whip REAL,
    starts INTEGER,
    innings_pitched REAL,
    PRIMARY KEY (pitcher_id, as_of_month)
);
"""


class MlbDataStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- writes -----------------------------------------------------------

    def upsert_team(self, team_id: int, name: str, abbreviation: str) -> None:
        self._conn.execute(
            "INSERT INTO teams (team_id, name, abbreviation) VALUES (?, ?, ?) "
            "ON CONFLICT(team_id) DO UPDATE SET name = excluded.name, abbreviation = excluded.abbreviation",
            (team_id, name, abbreviation),
        )
        self._conn.commit()

    def upsert_game(self, game: dict[str, Any]) -> None:
        """`game` is one raw game dict from mlb_stats_client.fetch_schedule."""
        teams = game.get("teams", {})
        home, away = teams.get("home", {}), teams.get("away", {})
        home_team, away_team = home.get("team", {}), away.get("team", {})
        status = game.get("status", {}).get("abstractGameState", "Unknown")

        home_score = home.get("score")
        away_score = away.get("score")
        winner_team_id = None
        if status == "Final":
            if home.get("isWinner"):
                winner_team_id = home_team.get("id")
            elif away.get("isWinner"):
                winner_team_id = away_team.get("id")

        home_pitcher = home.get("probablePitcher") or {}
        away_pitcher = away.get("probablePitcher") or {}

        self._conn.execute(
            """
            INSERT INTO games (
                game_pk, game_date, game_date_utc, status, home_team_id, away_team_id,
                home_score, away_score, winner_team_id,
                home_probable_pitcher_id, home_probable_pitcher_name,
                away_probable_pitcher_id, away_probable_pitcher_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_pk) DO UPDATE SET
                status = excluded.status, home_score = excluded.home_score,
                away_score = excluded.away_score, winner_team_id = excluded.winner_team_id,
                home_probable_pitcher_id = excluded.home_probable_pitcher_id,
                home_probable_pitcher_name = excluded.home_probable_pitcher_name,
                away_probable_pitcher_id = excluded.away_probable_pitcher_id,
                away_probable_pitcher_name = excluded.away_probable_pitcher_name
            """,
            (
                game["gamePk"], game.get("officialDate"), game.get("gameDate"), status,
                home_team.get("id"), away_team.get("id"), home_score, away_score, winner_team_id,
                home_pitcher.get("id"), home_pitcher.get("fullName"),
                away_pitcher.get("id"), away_pitcher.get("fullName"),
            ),
        )
        self._conn.commit()

    def record_pitcher_stats_snapshot(self, pitcher_id: int, as_of_month: str, stat: dict[str, Any]) -> None:
        """`as_of_month` is the first-of-month (YYYY-MM-01) this snapshot
        applies to; `stat` is the raw dict from
        mlb_stats_client.fetch_pitcher_stats_byrange (era/whip come back as
        strings from the API, e.g. "3.76")."""
        def _float_or_none(v: Any) -> float | None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        self._conn.execute(
            """
            INSERT INTO pitcher_stats_snapshot (pitcher_id, as_of_month, era, whip, starts, innings_pitched)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(pitcher_id, as_of_month) DO UPDATE SET
                era = excluded.era, whip = excluded.whip,
                starts = excluded.starts, innings_pitched = excluded.innings_pitched
            """,
            (
                pitcher_id, as_of_month,
                _float_or_none(stat.get("era")), _float_or_none(stat.get("whip")),
                stat.get("gamesStarted"), _float_or_none(stat.get("inningsPitched")),
            ),
        )
        self._conn.commit()

    def get_pitcher_stats_asof(self, pitcher_id: int, before_date: str) -> sqlite3.Row | None:
        """Most recent monthly snapshot strictly before `before_date`'s
        month (i.e. through the end of the prior month) -- see module
        docstring for the monthly-granularity tradeoff. None if this
        pitcher has no snapshot yet (e.g. a rookie's first month, or the
        backfill hasn't reached them) -- callers treat that as
        no-signal-yet, same convention as every other optional feature
        here."""
        month = before_date[:7] + "-01"
        return self._conn.execute(
            "SELECT * FROM pitcher_stats_snapshot WHERE pitcher_id = ? AND as_of_month < ? "
            "ORDER BY as_of_month DESC LIMIT 1",
            (pitcher_id, month),
        ).fetchone()

    # -- as-of reads (anti-leakage: strictly before_date) ------------------

    def get_team_form_asof(self, team_id: int, before_date: str, recent_n: int = 10) -> dict[str, Any]:
        """Team form strictly before `before_date` (YYYY-MM-DD), computed
        directly from completed games in `games` -- win%, last-`recent_n`
        win%, home/away split win%, and run differential per game. See the
        module docstring for why this is computed rather than pulled from
        the Stats API's historical /standings?date= endpoint. Returns a
        dict with None values (not zeros) for anything undefined when the
        team has no completed games before this date yet (early season) --
        callers/feature-builders decide how to handle that, not this
        method."""
        rows = self._conn.execute(
            "SELECT * FROM games WHERE status = 'Final' AND game_date < ? "
            "AND (home_team_id = ? OR away_team_id = ?) ORDER BY game_date DESC",
            (before_date, team_id, team_id),
        ).fetchall()
        if not rows:
            return {
                "games_played": 0, "win_pct": None, "last_n_win_pct": None,
                "home_win_pct": None, "away_win_pct": None, "run_diff_per_game": None,
            }

        wins = 0
        runs_scored = runs_allowed = 0
        home_wins = home_games = away_wins = away_games = 0
        for r in rows:
            is_home = r["home_team_id"] == team_id
            team_score = (r["home_score"] if is_home else r["away_score"]) or 0
            opp_score = (r["away_score"] if is_home else r["home_score"]) or 0
            won = r["winner_team_id"] == team_id
            wins += 1 if won else 0
            runs_scored += team_score
            runs_allowed += opp_score
            if is_home:
                home_games += 1
                home_wins += 1 if won else 0
            else:
                away_games += 1
                away_wins += 1 if won else 0

        n = len(rows)
        last_n = rows[:recent_n]
        last_n_wins = sum(1 for r in last_n if r["winner_team_id"] == team_id)

        return {
            "games_played": n,
            "win_pct": wins / n,
            "last_n_win_pct": last_n_wins / len(last_n) if last_n else None,
            "home_win_pct": (home_wins / home_games) if home_games else None,
            "away_win_pct": (away_wins / away_games) if away_games else None,
            "run_diff_per_game": (runs_scored - runs_allowed) / n,
        }

    def get_head_to_head(self, team_a_id: int, team_b_id: int, before_date: str, n: int = 10) -> list[sqlite3.Row]:
        """Up to the last `n` completed meetings between these two teams,
        strictly before `before_date`, most recent first."""
        return self._conn.execute(
            """
            SELECT * FROM games
            WHERE status = 'Final' AND game_date < ?
              AND ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
            ORDER BY game_date DESC LIMIT ?
            """,
            (before_date, team_a_id, team_b_id, team_b_id, team_a_id, n),
        ).fetchall()

    def get_last_game_for_team(self, team_id: int, before_date: str) -> sqlite3.Row | None:
        """Most recent completed game for rest-days calculation."""
        return self._conn.execute(
            "SELECT * FROM games WHERE status = 'Final' AND game_date < ? "
            "AND (home_team_id = ? OR away_team_id = ?) ORDER BY game_date DESC LIMIT 1",
            (before_date, team_id, team_id),
        ).fetchone()

    # -- general reads ------------------------------------------------------

    def get_completed_games(self, since_date: str | None = None) -> list[sqlite3.Row]:
        """All Final games, chronological -- the training set for
        fit_mlb_model.py (one row per game, both teams' as-of features get
        built by the caller before this game's own date)."""
        if since_date:
            return self._conn.execute(
                "SELECT * FROM games WHERE status = 'Final' AND game_date >= ? ORDER BY game_date ASC",
                (since_date,),
            ).fetchall()
        return self._conn.execute(
            "SELECT * FROM games WHERE status = 'Final' ORDER BY game_date ASC"
        ).fetchall()

    def get_games_on_date(self, date: str) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM games WHERE game_date = ?", (date,)).fetchall()

    def get_game_by_pk(self, game_pk: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM games WHERE game_pk = ?", (game_pk,)).fetchone()

    def get_team_by_abbreviation(self, abbreviation: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM teams WHERE abbreviation = ?", (abbreviation,)).fetchone()

    def get_team_by_id(self, team_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,)).fetchone()

    def find_games_by_teams(
        self, home_team_id: int, away_team_id: int, date_start: str, date_end: str
    ) -> list[sqlite3.Row]:
        """Games between these two teams (this specific home/away
        orientation) with game_date in [date_start, date_end] -- used by
        mlb_watchlist.py to match a Kalshi ticker's team pair to the real
        schedule row, since the ticker's own embedded date isn't reliable
        for this (see that module's docstring)."""
        return self._conn.execute(
            "SELECT * FROM games WHERE home_team_id = ? AND away_team_id = ? AND game_date BETWEEN ? AND ?",
            (home_team_id, away_team_id, date_start, date_end),
        ).fetchall()

    def team_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM teams").fetchone()["n"]

    def completed_game_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM games WHERE status = 'Final'").fetchone()["n"]
