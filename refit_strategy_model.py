"""Regenerates DataDrivenCryptoStrategy's probability model from the live
ledger's actual trade history -- run this periodically (manually; there is
no scheduled job) as real trades accumulate, so the model stays a genuine
reflection of real outcomes rather than a permanent snapshot of whatever
329 trades existed on 2026-08-04.

    uv run python refit_strategy_model.py

Deliberately NOT automatic. Same principle online_learner.py already holds
itself to ("never invents a trade idea, never overrides an exit"): a
human reviews what changed and why before new coefficients go live, rather
than the strategy silently rewriting its own decision boundary. Prints the
new weights/bias ready to paste into trade_bot/data_driven_strategy.py's
PROBABILITY_MODEL_WEIGHTS/PROBABILITY_MODEL_BIAS, plus a chronological
holdout backtest so you can see whether the refit actually generalizes
before trusting it -- don't paste it in if the holdout numbers look worse
than what's already there without understanding why.

Only trades with a features_json snapshot are used (BUY decisions made by
the strategy itself -- reconciled/untracked positions and manual trades
have no entry-time feature vector and are silently excluded, same as the
original fit).
"""
from __future__ import annotations

import json
import math

from trade_bot.live_ledger import LiveLedger
from trade_bot.online_learner import FEATURE_NAMES


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


def main() -> None:
    ledger = LiveLedger()
    rows = ledger.get_closed_trades(limit=100_000)
    examples: list[tuple[dict, int, str]] = []
    for t in rows:
        if not t.features_json:
            continue
        f = json.loads(t.features_json)
        label = 1 if (t.realized_pnl or 0) > 0 else 0
        examples.append((f, label, t.closed_at or ""))
    examples.sort(key=lambda e: e[2])  # chronological

    n = len(examples)
    print(f"Total labeled trades: {n}")
    if n < 100:
        print(
            "Fewer than 100 labeled trades -- a refit here would be fitting mostly noise. "
            "The original 2026-08-04 fit used 314. Wait for more real data before trusting a refit "
            "this small; running it anyway below for reference only."
        )

    split = int(n * 0.8)
    train, test = examples[:split], examples[split:]
    print(f"train={len(train)}  test={len(test)} (chronological 80/20 split)")

    weights, bias = _fit([(f, label) for f, label, _ in train])

    print("\n=== Paste into trade_bot/data_driven_strategy.py ===")
    print(f"PROBABILITY_MODEL_BIAS = {bias!r}")
    print("PROBABILITY_MODEL_WEIGHTS: dict[str, float] = {")
    for k in FEATURE_NAMES:
        print(f'    "{k}": {weights[k]!r},')
    print("}")

    if not test:
        print("\nNo holdout examples (too few trades) -- skipping backtest section.")
        return

    print(f"\n=== Holdout backtest (n={len(test)}) ===")
    base_rate = sum(label for _, label, _ in test) / len(test)
    print(f"base rate (actual win%% across all holdout trades): {base_rate:.1%}")
    for thresh in [0.35, 0.40, 0.43, 0.45, 0.50]:
        taken = [(f, label) for f, label, _ in test if _predict(f, weights, bias) >= thresh]
        if not taken:
            print(f"  threshold {thresh:.2f}: 0 trades would be taken")
            continue
        wr = sum(label for _, label in taken) / len(taken)
        print(f"  threshold {thresh:.2f}: {len(taken)}/{len(test)} taken, win_rate={wr:.1%}")

    print(
        "\nRefit again on the FULL dataset (no holdout) before actually updating the "
        "constants -- the holdout above is only for judging whether this feature set "
        "still generalizes, the shipped coefficients should use every available trade."
    )
    full_weights, full_bias = _fit([(f, label) for f, label, _ in examples])
    print("\n=== Full-dataset fit (what to actually ship) ===")
    print(f"PROBABILITY_MODEL_BIAS = {full_bias!r}")
    print("PROBABILITY_MODEL_WEIGHTS: dict[str, float] = {")
    for k in FEATURE_NAMES:
        print(f'    "{k}": {full_weights[k]!r},')
    print("}")


if __name__ == "__main__":
    main()
