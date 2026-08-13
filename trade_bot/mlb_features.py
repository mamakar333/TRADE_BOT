"""As-of feature engineering for MLB game predictions, built entirely from
trade_bot/mlb_data.py's MlbDataStore. Every input here is computed strictly
before the game being predicted -- see MlbDataStore.get_team_form_asof and
get_head_to_head's own anti-leakage guarantees.

Features are built PER TEAM (not per game): one call describes one team's
matchup context in a specific game (its own form, how it compares to this
specific opponent, home/away, rest, head-to-head). fit_mlb_model.py builds
TWO training rows per historical game -- one from the home team's
perspective (label = did home win) and one from the away team's (label =
did away win) -- so the model is side-agnostic, the same way the crypto
strategy's model is trained on both YES and NO candidates rather than
having a separate model per side.

Fixed-divisor scaling to roughly [-1, 1], same convention as
online_learner.build_features -- fixed scaling means early examples can't
distort what "large" means for every example after.
"""
from __future__ import annotations

from datetime import date as date_type

from .mlb_data import MlbDataStore

FEATURE_NAMES = [
    "win_pct",
    "win_pct_diff",
    "last10_pct",
    "last10_pct_diff",
    "run_diff_per_game_scaled",
    "run_diff_per_game_delta_scaled",
    "is_home",
    "context_split_pct",
    "h2h_win_pct",
    "rest_days_scaled",
    "sample_size_scaled",
    "starting_pitcher_quality_scaled",
    "starting_pitcher_quality_diff_scaled",
]

_DEFAULT_PCT = 0.5  # no real signal yet (early season / no meetings) -- coin-flip prior, not 0.0
_MAX_REST_DAYS = 5.0  # rest beyond this stops mattering much; also covers a scheduled off-day
_FULL_SEASON_GAMES = 162.0
# 2026 MLB league-average starter ERA is close enough to this for the
# purpose of a centered feature -- doesn't need to be exact, just a
# reasonable, stable reference point so the sign is intuitive (positive =
# better than average) rather than the model discovering that sign itself.
_LEAGUE_AVG_ERA = 4.20
_ERA_SPREAD = 2.0


def _pitcher_quality_scaled(store: MlbDataStore, pitcher_id: int | None, game_date: str) -> float:
    """0.0 (neutral/average) when there's no snapshot yet -- rookie/first
    month in the dataset, or no probable pitcher recorded for this game --
    same "absent = no signal" convention as every other optional feature
    here. Positive = better (lower ERA) than league average."""
    if pitcher_id is None:
        return 0.0
    snap = store.get_pitcher_stats_asof(pitcher_id, game_date)
    if snap is None or snap["era"] is None:
        return 0.0
    return (_LEAGUE_AVG_ERA - snap["era"]) / _ERA_SPREAD


def build_mlb_features(
    store: MlbDataStore, team_id: int, opponent_team_id: int, game_date: str, is_home: bool,
    pitcher_id: int | None = None, opponent_pitcher_id: int | None = None,
) -> dict[str, float]:
    team_form = store.get_team_form_asof(team_id, game_date)
    opp_form = store.get_team_form_asof(opponent_team_id, game_date)

    win_pct = team_form["win_pct"] if team_form["win_pct"] is not None else _DEFAULT_PCT
    opp_win_pct = opp_form["win_pct"] if opp_form["win_pct"] is not None else _DEFAULT_PCT
    last10_pct = team_form["last_n_win_pct"] if team_form["last_n_win_pct"] is not None else _DEFAULT_PCT
    opp_last10_pct = opp_form["last_n_win_pct"] if opp_form["last_n_win_pct"] is not None else _DEFAULT_PCT
    run_diff = team_form["run_diff_per_game"] if team_form["run_diff_per_game"] is not None else 0.0
    opp_run_diff = opp_form["run_diff_per_game"] if opp_form["run_diff_per_game"] is not None else 0.0

    if is_home:
        context_split = team_form["home_win_pct"] if team_form["home_win_pct"] is not None else _DEFAULT_PCT
    else:
        context_split = team_form["away_win_pct"] if team_form["away_win_pct"] is not None else _DEFAULT_PCT

    h2h_games = store.get_head_to_head(team_id, opponent_team_id, game_date)
    if h2h_games:
        h2h_wins = sum(1 for g in h2h_games if g["winner_team_id"] == team_id)
        h2h_win_pct = h2h_wins / len(h2h_games)
    else:
        h2h_win_pct = _DEFAULT_PCT

    last_game = store.get_last_game_for_team(team_id, game_date)
    if last_game is not None:
        rest_days = (date_type.fromisoformat(game_date) - date_type.fromisoformat(last_game["game_date"])).days
        rest_days = max(0, rest_days)
    else:
        rest_days = _MAX_REST_DAYS  # no prior game this season (e.g. opening day) -- treat as fully rested

    pitcher_quality = _pitcher_quality_scaled(store, pitcher_id, game_date)
    opp_pitcher_quality = _pitcher_quality_scaled(store, opponent_pitcher_id, game_date)

    return {
        "win_pct": win_pct,
        "win_pct_diff": win_pct - opp_win_pct,
        "last10_pct": last10_pct,
        "last10_pct_diff": last10_pct - opp_last10_pct,
        "run_diff_per_game_scaled": run_diff / 3.0,
        "run_diff_per_game_delta_scaled": (run_diff - opp_run_diff) / 3.0,
        "is_home": 1.0 if is_home else 0.0,
        "context_split_pct": context_split,
        "h2h_win_pct": h2h_win_pct,
        "rest_days_scaled": min(rest_days, _MAX_REST_DAYS) / _MAX_REST_DAYS,
        "sample_size_scaled": min(team_form["games_played"], _FULL_SEASON_GAMES) / _FULL_SEASON_GAMES,
        "starting_pitcher_quality_scaled": pitcher_quality,
        "starting_pitcher_quality_diff_scaled": pitcher_quality - opp_pitcher_quality,
    }
