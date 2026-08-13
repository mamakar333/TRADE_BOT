"""MLB game-winner probability model. Same "small, fully interpretable
linear model, hand-rolled, no black box" philosophy as
trade_bot/online_learner.py and trade_bot/data_driven_strategy.py's
PROBABILITY_MODEL_* -- every prediction traces back to a handful of named
weights.

Fit 2026-08-12, refit 2026-08-13 via fit_mlb_model.py against 1,806 real
completed 2026 regular-season games (3,612 team-rows) backfilled from the
free MLB Stats API. Chronological 80/20 holdout (362 games, 724 team-rows,
none seen while fitting): 50.0% base rate, 55.7% win rate at a 0.50
probability cutoff (361/724 taken), 60.3% at 0.52 (151/724 taken) -- the
threshold actually used live (see run_mlb_trading.py's
min_win_probability).

2026-08-13 addition: starting_pitcher_quality_scaled/_diff_scaled, from a
real fitted pitcher-quality signal (see backfill_pitcher_stats.py and
trade_bot/mlb_data.py's pitcher_stats_snapshot). Added specifically because
the user noticed live predictions clustering close to 50/50 with little
separation -- verified first that this wasn't a regularization artifact (L2
swept from 0.02 to 0.0 against the same 3,612-row dataset, holdout barely
moved, max predicted probability only 0.553 -> 0.570), confirming the
compression was a genuine "these 11 features carry modest signal" finding,
not a fitting bug, before spending the effort to add a real new signal
rather than just retuning existing ones. Net effect on holdout: modestly
better at 0.50/0.52 (won more, took slightly more), and notably better in
the model's own most-confident tail -- 0.55 cutoff went from n=3 (33.3% win
rate, too small to trust) to n=9 (55.6%), meaning the model can now
distinguish a real high-confidence tier where before it barely could.
starting_pitcher_quality_diff_scaled (this team's starter's quality RELATIVE
to the opponent's) carries essentially all of that signal (weight ~0.038);
the standalone (non-comparative) version is ~0 -- sensible, since a good
starter matters against a good OR bad opposing starter differently, not in
isolation.

Weight signs mostly match baseball intuition: last10_pct_diff (recent form)
and run_diff_per_game_delta_scaled are the two strongest positive
coefficients -- recent form predicts the next game better than stale
season-long record, a real and unsurprising sabermetric pattern. is_home is
positive (real home-field advantage). Two features came out with small,
counter-intuitive negative signs (win_pct, win_pct_diff) -- likely
multicollinearity: last-10 record is a subset of season win%, so once the
stronger last10_pct_diff signal is in the model, season win_pct's own
coefficient picks up a mild redundancy/regression effect rather than a real
reversal. Worth watching on the next refit, not treated as broken -- the
magnitude is small relative to the features that matter.

Still deferred: sportsbook/odds data (skipped entirely per explicit user
decision when this was built). Pitcher stats are monthly-granularity, not
true per-game as-of (see mlb_data.py's docstring for the API-call-volume
tradeoff behind that).

Never auto-applied -- same rule as refit_strategy_model.py: a human reviews
what changed before new coefficients go live.
"""
from __future__ import annotations

import math

from .mlb_features import FEATURE_NAMES

MLB_MODEL_BIAS = -0.07018184824439878
MLB_MODEL_WEIGHTS: dict[str, float] = {
    "win_pct": -0.024016919857847846,
    "win_pct_diff": -0.015295082638301915,
    "last10_pct": 0.04789313343016476,
    "last10_pct_diff": 0.13424956902312424,
    "run_diff_per_game_scaled": 0.03518669365929677,
    "run_diff_per_game_delta_scaled": 0.10425917042310356,
    "is_home": 0.14422030005911204,
    "context_split_pct": 0.006166188766966693,
    "h2h_win_pct": -0.03440230646920806,
    "rest_days_scaled": 0.01616420387617672,
    "sample_size_scaled": -0.007983430356474643,
    "starting_pitcher_quality_scaled": 0.0011367907557519909,
    "starting_pitcher_quality_diff_scaled": 0.03781556147265985,
}


def _sigmoid(z: float) -> float:
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def predict_mlb_win_probability(features: dict[str, float]) -> float:
    z = MLB_MODEL_BIAS + sum(MLB_MODEL_WEIGHTS.get(k, 0.0) * v for k, v in features.items())
    return _sigmoid(z)
