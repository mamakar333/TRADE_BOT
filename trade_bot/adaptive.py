"""Rolling-performance-based adaptation: the honest version of "the bot learns
and gets smarter as it goes."

This is deliberately NOT a predictive model. There is no evidence an LLM or a
trained model has genuine edge on 15-minute BTC direction, and bolting one
onto trade decisions wouldn't be learning -- it'd be guessing with extra
steps and extra latency. What IS real and verifiable: tracking whether the
strategy's own recent live trades have actually been working, and
mechanically turning position size down (or off entirely) the moment they
haven't, before more capital rides on a signal that's currently losing.
Sizing recovers automatically as real results improve. Every number this
module produces traces back to realized trades already in the ledger --
nothing here is a black box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class AdaptiveConfig:
    lookback_trades: int = 15
    min_trades_for_adaptation: int = 8
    # Trailing win rate at or below this -> stop opening new positions entirely.
    pause_win_rate: float = 0.20
    # Trailing win rate at or below this (but above pause_win_rate) -> half size.
    reduced_win_rate: float = 0.35
    # Trailing average $ P&L per trade at or below this -> pause regardless of
    # win rate (catches a strategy with occasional wins that are too small to
    # cover frequent/large losses).
    pause_expectancy_dollars: float = -3.0


@dataclass
class GovernorDecision:
    multiplier: float  # 0.0 (paused) to 1.0 (full size)
    reason: str


class PerformanceGovernor:
    """Given a function that returns a strategy's recent realized trade P&Ls,
    produces a size multiplier for its next entry. Pure function of realized
    history -- no external data, no fabricated signal."""

    def __init__(
        self,
        recent_pnls_fn: Callable[[str, int], list[float]],
        config: AdaptiveConfig | None = None,
    ):
        # recent_pnls_fn(strategy_name, limit) -> realized $ P&L per closed
        # trade, oldest first, for that strategy only.
        self._recent_pnls_fn = recent_pnls_fn
        self.config = config or AdaptiveConfig()

    def evaluate(self, strategy_name: str) -> GovernorDecision:
        cfg = self.config
        pnls = self._recent_pnls_fn(strategy_name, cfg.lookback_trades)
        if len(pnls) < cfg.min_trades_for_adaptation:
            return GovernorDecision(
                1.0, f"only {len(pnls)} closed trades so far (need {cfg.min_trades_for_adaptation}) -- no adjustment yet"
            )

        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)
        expectancy = sum(pnls) / len(pnls)

        if win_rate <= cfg.pause_win_rate or expectancy <= cfg.pause_expectancy_dollars:
            return GovernorDecision(
                0.0,
                f"trailing {len(pnls)}-trade win rate {win_rate:.0%}, expectancy ${expectancy:+.2f}/trade -- "
                "paused, not opening new positions until recent results improve",
            )
        if win_rate <= cfg.reduced_win_rate:
            return GovernorDecision(
                0.5, f"trailing {len(pnls)}-trade win rate {win_rate:.0%} -- size halved"
            )
        return GovernorDecision(
            1.0, f"trailing {len(pnls)}-trade win rate {win_rate:.0%}, expectancy ${expectancy:+.2f}/trade -- full size"
        )
