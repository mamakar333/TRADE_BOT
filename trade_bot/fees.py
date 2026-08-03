"""Kalshi's taker trading fee formula.

fee = round_up_to_$0.0001(multiplier * C * P * (1 - P))

where C = contract count, P = contract price in dollars (0.01-0.99), and
multiplier defaults to 0.07 (Kalshi's standard rate; overridable per-series
via the series' `fee_multiplier`). Confirmed 2026-07-12 against Kalshi's
public fee schedule and help docs. Since this simulator always fills by
crossing the book (buying at the ask, selling at the bid, per spec), every
simulated fill is a taker fill, so this single formula covers both entry
and exit -- there's no separate maker-fee path to model.
"""
from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

DEFAULT_FEE_MULTIPLIER = 0.07
_TICK = Decimal("0.0001")


def taker_fee_dollars(quantity: int, price_pct: float, multiplier: float = DEFAULT_FEE_MULTIPLIER) -> float:
    """`price_pct` is 0-100 (Kalshi's percent/cents-equivalent convention elsewhere in this repo).

    Uses Decimal throughout: plain float math (0.07 * 100 * 0.5 * 0.5) lands a
    hair above 1.75 due to binary float noise, and round_up-to-$0.0001 then
    amplifies that noise into a real extra $0.0001 charge per fill.
    """
    if quantity <= 0:
        return 0.0
    m = Decimal(str(multiplier))
    p = Decimal(str(price_pct)) / 100
    raw = m * quantity * p * (1 - p)
    return float(raw.quantize(_TICK, rounding=ROUND_CEILING))
