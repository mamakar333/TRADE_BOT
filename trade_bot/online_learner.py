"""Online (continuously self-updating) logistic regression over real trade
outcomes. This is the literal "updates its algorithm live without you
redeploying code" mechanism, layered on top of `PerformanceGovernor`'s
coarser strategy-wide gate (adaptive.py).

Deliberately a small, fully interpretable linear model -- hand-rolled with
plain Python floats, no external ML library, no black box. Every prediction
traces back to a handful of named weights, which get logged and shown in
the dashboard, so "why did it skip this trade" always has a concrete answer.

Honest framing, read before changing the gate thresholds:
- It predicts P(this specific candidate trade closes profitable) from
  features already computed by the strategy for its own decision -- nothing
  new is fetched or fabricated.
- It updates by one gradient step every time a real trade closes and its
  outcome is known. Real trading produces maybe a handful of labeled
  examples a day, so this learns slowly by design -- MIN_TRADES_FOR_ML_GATE
  (set by the caller) keeps it fully inert until there's enough data that a
  prediction means more than noise, exactly like PerformanceGovernor's own
  warm-up gate.
- It can only ever suppress or dampen an entry the strategy already decided
  to make -- same non-negotiable as the governor. It never invents a trade
  idea, never picks a market, never overrides an exit.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

FEATURE_NAMES = [
    "short_delta_scaled",
    "long_delta_scaled",
    "is_micro_bet",
    "volatility_scaled",
    "orderbook_imbalance",
    "minutes_to_close_scaled",
    "entry_price_scaled",
    "side_is_yes",
    # Added 2026-08-04 (the "underwriter" expansion): asset and time-of-day
    # were the two most obvious real signals the model had no way to use
    # before -- every trade got folded into one undifferentiated pool
    # regardless of whether it was BTC/ETH/BNB or 3am/3pm. Cyclical
    # (sin/cos) hour encoding avoids the discontinuity a raw 0-23 value
    # would have at the 23->0 wraparound.
    "hour_sin",
    "hour_cos",
    "asset_btc",
    "asset_eth",
]

_BTC_PREFIXES = ("KXBTC15M-", "KXBTCD-")
_ETH_PREFIXES = ("KXETH15M-", "KXETHD-")


def build_features(
    *,
    short_delta: float,
    long_delta: float | None,
    is_micro_bet: bool,
    volatility: float | None,
    orderbook_imbalance: float | None,
    minutes_to_close: float | None,
    entry_price_pct: float,
    side: str,
    hour_utc: int | None = None,
    ticker: str = "",
) -> dict[str, float]:
    """Raw signal values the strategy already computed for its own decision,
    rescaled to roughly [-1, 1] with FIXED divisors (not online-normalized)
    -- fixed scaling means a handful of early examples can't distort what
    "large" means for every example after."""
    hour_angle = 2 * math.pi * (hour_utc or 0) / 24
    return {
        "short_delta_scaled": short_delta / 5.0,
        "long_delta_scaled": (long_delta or 0.0) / 10.0,
        "is_micro_bet": 1.0 if is_micro_bet else 0.0,
        "volatility_scaled": (volatility or 0.0) / 10.0,
        "orderbook_imbalance": orderbook_imbalance or 0.0,
        "minutes_to_close_scaled": (minutes_to_close if minutes_to_close is not None else 30.0) / 60.0,
        "entry_price_scaled": entry_price_pct / 100.0,
        "side_is_yes": 1.0 if side == "YES" else 0.0,
        "hour_sin": math.sin(hour_angle) if hour_utc is not None else 0.0,
        "hour_cos": math.cos(hour_angle) if hour_utc is not None else 0.0,
        "asset_btc": 1.0 if ticker.startswith(_BTC_PREFIXES) else 0.0,
        "asset_eth": 1.0 if ticker.startswith(_ETH_PREFIXES) else 0.0,
    }


def _sigmoid(z: float) -> float:
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class OnlineLogisticLearner:
    """Weights persist via to_json()/from_json() -- LiveLedger stores this so
    learning survives a bot restart, unlike per-ticker price history (which
    intentionally does NOT survive one -- see live_engine.py)."""

    weights: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in FEATURE_NAMES})
    bias: float = 0.0
    n_updates: int = 0
    learning_rate: float = 0.05
    l2: float = 0.01

    def predict_proba(self, features: dict[str, float]) -> float:
        z = self.bias + sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        return _sigmoid(z)

    def update(self, features: dict[str, float], label: int) -> None:
        """One SGD step. `label` is 1 if the trade closed profitable, else 0."""
        p = self.predict_proba(features)
        error = label - p
        for k, v in features.items():
            w = self.weights.get(k, 0.0)
            self.weights[k] = w + self.learning_rate * (error * v - self.l2 * w)
        self.bias += self.learning_rate * error
        self.n_updates += 1

    def to_json(self) -> str:
        return json.dumps(
            {
                "weights": self.weights,
                "bias": self.bias,
                "n_updates": self.n_updates,
                "learning_rate": self.learning_rate,
                "l2": self.l2,
            }
        )

    @classmethod
    def from_json(cls, raw: str | None) -> "OnlineLogisticLearner":
        if not raw:
            return cls()
        data = json.loads(raw)
        return cls(
            weights={**{k: 0.0 for k in FEATURE_NAMES}, **data.get("weights", {})},
            bias=data.get("bias", 0.0),
            n_updates=data.get("n_updates", 0),
            learning_rate=data.get("learning_rate", 0.05),
            l2=data.get("l2", 0.01),
        )
