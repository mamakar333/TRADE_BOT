"""Fits trade_bot/mlb_model.py's probability model from mlb_data.db's real
backfilled game history -- run this manually (mirrors refit_strategy_model.py
exactly, same reasoning: a human reviews what changed before new
coefficients go live, never auto-applied).

    uv run python fit_mlb_model.py

Two training rows per completed game (one from the home team's perspective,
one from the away team's -- see trade_bot/mlb_features.py's module
docstring for why), but the chronological 80/20 holdout split happens on
GAMES first, then both of a game's rows go into whichever split its game
landed in -- splitting on individual rows would let a game's home-perspective
row train on a model that's implicitly seen its away-perspective mirror (or
vice versa), a mild form of leakage even though neither row alone leaks
outcome data.
"""
from __future__ import annotations

import math

from trade_bot.mlb_data import MlbDataStore
from trade_bot.mlb_features import FEATURE_NAMES, build_mlb_features


def _sigmoid(z: float) -> float:
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _fit(train: list[tuple[dict, int]], lr: float = 0.1, l2: float = 0.02, epochs: int = 800):
    weights = {k: 0.0 for k in FEATURE_NAMES}
    bias = 0.0
    n = len(train)
    for _ in range(epochs):
        grad_w = {k: 0.0 for k in FEATURE_NAMES}
        grad_b = 0.0
        for f, label in train:
            z = bias + sum(weights[k] * f.get(k, 0.0) for k in FEATURE_NAMES)
            p = _sigmoid(z)
            error = p - label
            for k in FEATURE_NAMES:
                grad_w[k] += error * f.get(k, 0.0) / n
            grad_b += error / n
        for k in FEATURE_NAMES:
            weights[k] -= lr * (grad_w[k] + l2 * weights[k])
        bias -= lr * grad_b
    return weights, bias


def _predict(f: dict, weights: dict, bias: float) -> float:
    z = bias + sum(weights.get(k, 0.0) * f.get(k, 0.0) for k in FEATURE_NAMES)
    return _sigmoid(z)


def _build_rows_for_game(store: MlbDataStore, game) -> list[tuple[dict, int]]:
    home_features = build_mlb_features(
        store, game["home_team_id"], game["away_team_id"], game["game_date"], True,
        pitcher_id=game["home_probable_pitcher_id"], opponent_pitcher_id=game["away_probable_pitcher_id"],
    )
    away_features = build_mlb_features(
        store, game["away_team_id"], game["home_team_id"], game["game_date"], False,
        pitcher_id=game["away_probable_pitcher_id"], opponent_pitcher_id=game["home_probable_pitcher_id"],
    )
    home_label = 1 if game["winner_team_id"] == game["home_team_id"] else 0
    return [(home_features, home_label), (away_features, 1 - home_label)]


def main() -> None:
    store = MlbDataStore()
    games = store.get_completed_games()
    print(f"Total completed regular-season games: {len(games)}")
    if len(games) < 50:
        print(
            "Fewer than 50 completed games -- a fit here would be almost pure noise. "
            "Run run_mlb_data_collector.py --backfill first, or wait for more of the season "
            "to play out. Running anyway below for reference only."
        )

    split = int(len(games) * 0.8)
    train_games, test_games = games[:split], games[split:]
    print(f"train_games={len(train_games)} test_games={len(test_games)} (chronological 80/20 split, by game)")

    train_rows: list[tuple[dict, int]] = []
    for g in train_games:
        train_rows.extend(_build_rows_for_game(store, g))

    weights, bias = _fit(train_rows)

    print("\n=== Paste into trade_bot/mlb_model.py ===")
    print(f"MLB_MODEL_BIAS = {bias!r}")
    print("MLB_MODEL_WEIGHTS: dict[str, float] = {")
    for k in FEATURE_NAMES:
        print(f'    "{k}": {weights[k]!r},')
    print("}")

    if not test_games:
        print("\nNo holdout games (too few total) -- skipping backtest section.")
        store.close()
        return

    test_rows: list[tuple[dict, int]] = []
    for g in test_games:
        test_rows.extend(_build_rows_for_game(store, g))

    print(f"\n=== Holdout backtest (n={len(test_rows)} team-rows across {len(test_games)} games) ===")
    base_rate = sum(label for _, label in test_rows) / len(test_rows)
    print(f"base rate (actual win%% across all holdout team-rows): {base_rate:.1%}")
    for thresh in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
        taken = [(f, label) for f, label in test_rows if _predict(f, weights, bias) >= thresh]
        if not taken:
            print(f"  threshold {thresh:.2f}: 0 team-rows would be taken")
            continue
        wr = sum(label for _, label in taken) / len(taken)
        print(f"  threshold {thresh:.2f}: {len(taken)}/{len(test_rows)} taken, win_rate={wr:.1%}")

    print(
        "\nRefit again on the FULL dataset (no holdout) before actually updating the "
        "constants -- the holdout above is only for judging whether this feature set "
        "still generalizes, the shipped coefficients should use every available game."
    )
    all_rows: list[tuple[dict, int]] = []
    for g in games:
        all_rows.extend(_build_rows_for_game(store, g))
    full_weights, full_bias = _fit(all_rows)
    print("\n=== Full-dataset fit (what to actually ship) ===")
    print(f"MLB_MODEL_BIAS = {full_bias!r}")
    print("MLB_MODEL_WEIGHTS: dict[str, float] = {")
    for k in FEATURE_NAMES:
        print(f'    "{k}": {full_weights[k]!r},')
    print("}")
    store.close()


if __name__ == "__main__":
    main()
