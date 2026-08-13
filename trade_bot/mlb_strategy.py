"""MLB game-winner pregame strategy. A bare `Strategy` subclass (NOT
_TakeProfitStopLossStrategy) -- that base class is built around continuous
trailing-stop/momentum management over a market's whole (short) life, which
doesn't fit this strategy's actual shape: one pregame decision, then hold
for hours until the game ends. See trade_bot/strategy.py's `Strategy`
interface and `force_settle_exit` (reused here, not reimplemented).

Entry: pregame only, one team at a time. For a given KXMLBGAME ticker
(which names exactly one team -- e.g. "...-TEX" is "does Texas win"), buys
YES if the model's predicted win probability for that specific team clears
min_win_probability. Deliberately never proposes BUY_NO -- the same
underlying view ("this other team wins") is already covered more directly
by evaluating THAT team's own ticker on its own cycle (both tickers of a
game get evaluated independently every cycle), so betting NO on this one
would just be a redundant, worse-priced way to express a view already
covered elsewhere. The engine's own same-game correlated-exposure guard
(see live_engine.py's underlying_key_fn, wired to mlb_watchlist.game_key)
is the safety net against ending up with both sides of one game anyway.

Exit: no active management at all once filled -- plain HOLD every cycle
until the market goes non-tradeable (game over), then force_settle_exit
realizes the real settlement price. This is a genuinely different bet
shape than crypto's failed `gamble_enabled` mechanic (see
data_driven_strategy.py's docstring: "0-for-41, an unambiguous no-caveats
failure") -- that was a blind hold on a rotating threshold contract with no
real model behind entry; this is a discrete team-wins-game outcome with a
real fitted model (trade_bot/mlb_model.py, holdout-validated) deciding
whether to enter at all. Worth extra caution regardless, given zero
real-money track record yet -- see run_mlb_trading.py's rollout plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .mlb_data import MlbDataStore
from .mlb_features import build_mlb_features
from .mlb_model import predict_mlb_win_probability
from .mlb_watchlist import find_game_for_ticker, parse_ticker
from .portfolio import Position
from .strategy import MarketSnapshot, Signal, Strategy, StrategyDecision, force_settle_exit


@dataclass
class MlbGameWinnerStrategy(Strategy):
    name = "mlb_pregame"

    data_store: MlbDataStore

    # Refreshed from LiveLedger.get_mlb_trading_enabled() every cycle by
    # LiveExecutionEngine (mirrors btc_dip_enabled's existing pattern) --
    # off by default, a live dashboard/app toggle, not a redeploy.
    mlb_trading_enabled: bool = False

    # Calibrated 2026-08-12 against fit_mlb_model.py's chronological holdout
    # (362 games / 724 team-rows, none seen while fitting): 0.50 showed a
    # real 56.3% win rate (vs. 50.0% base rate) on 359 taken, 0.52 showed
    # 58.5% on a smaller 147-row sample. 0.52 picked as the starting gate --
    # more selective, still a real enough sample to trust over 0.55's n=3.
    # Absolute threshold used as a ranking tool, not a "beats the market"
    # claim -- same reasoning data_driven_strategy.py's min_win_probability
    # already documents for the crypto strategy.
    min_win_probability: float = 0.52

    # Risk/reward floor -- paying deep into favorite territory for a
    # multi-hour hold leaves little room even when right. Not yet backtested
    # against real MLB entry-price outcomes (no real trades exist yet); set
    # by analogy to data_driven_strategy.py's same-spirit
    # max_entry_price_pct, revisit once there's real MLB trade history.
    max_entry_price_pct: float = 75.0

    # How far above min_win_probability the prediction needs to be for
    # sizing to reach max_dollars.
    full_confidence_margin: float = 0.08

    # Deliberately conservative PLACEHOLDER sizing -- explicit user decision
    # (see the MLB plan) was to NOT set real dollar caps until there's a
    # validated paper-trading sample to inform them. These only matter once
    # mlb_trading_enabled is ever turned on for real money, which per that
    # same plan requires an explicit separate conversation first.
    base_dollars: float = 5.0
    max_dollars: float = 10.0

    # Don't enter in the last few minutes before first pitch -- pricing can
    # get unstable/thin right at the wire, and there's no benefit to cutting
    # it close for a position that's going to sit for hours regardless.
    min_minutes_to_first_pitch: float = 15.0

    def evaluate(
        self, snapshot: MarketSnapshot, history: list[MarketSnapshot], position: Position | None,
    ) -> StrategyDecision:
        if not self.mlb_trading_enabled:
            return StrategyDecision(Signal.HOLD, reason="mlb_trading_enabled is off")

        tradeable = snapshot.status in ("active", "open")

        if position is not None:
            if not tradeable:
                return force_settle_exit(snapshot, position)
            return StrategyDecision(Signal.HOLD, reason="holding to game outcome -- no active management")

        if not tradeable:
            return StrategyDecision(
                Signal.HOLD, reason=f"market status is {snapshot.status!r}, not tradeable for new entries"
            )

        return self._evaluate_entry(snapshot)

    def _evaluate_entry(self, snapshot: MarketSnapshot) -> StrategyDecision:
        parsed = parse_ticker(snapshot.ticker)
        if parsed is None:
            return StrategyDecision(Signal.HOLD, reason="ticker doesn't match the expected KXMLBGAME format")
        _official_date, away_code, home_code, side_code = parsed

        game = find_game_for_ticker(self.data_store, snapshot.ticker)
        if game is None:
            return StrategyDecision(Signal.HOLD, reason=f"no matching scheduled game found for {away_code}@{home_code}")

        try:
            first_pitch = datetime.fromisoformat(game["game_date_utc"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return StrategyDecision(Signal.HOLD, reason="matched game has no usable first-pitch time")

        now = snapshot.timestamp if snapshot.timestamp.tzinfo else snapshot.timestamp.replace(tzinfo=timezone.utc)
        minutes_to_first_pitch = (first_pitch - now).total_seconds() / 60
        if minutes_to_first_pitch < self.min_minutes_to_first_pitch:
            return StrategyDecision(
                Signal.HOLD,
                reason=f"{minutes_to_first_pitch:.0f}min to first pitch, below the "
                f"{self.min_minutes_to_first_pitch:.0f}min pregame cutoff",
            )

        side_is_home = side_code == home_code
        side_team_id = game["home_team_id"] if side_is_home else game["away_team_id"]
        opponent_team_id = game["away_team_id"] if side_is_home else game["home_team_id"]
        side_pitcher_id = game["home_probable_pitcher_id"] if side_is_home else game["away_probable_pitcher_id"]
        opponent_pitcher_id = game["away_probable_pitcher_id"] if side_is_home else game["home_probable_pitcher_id"]

        features = build_mlb_features(
            self.data_store, side_team_id, opponent_team_id, game["game_date"], side_is_home,
            pitcher_id=side_pitcher_id, opponent_pitcher_id=opponent_pitcher_id,
        )
        p_win = predict_mlb_win_probability(features)

        if p_win < self.min_win_probability:
            return StrategyDecision(
                Signal.HOLD,
                reason=f"model predicts {p_win:.0%} win probability for {side_code}, below the "
                f"{self.min_win_probability:.0%} gate",
                features=features,
                predicted_probability=p_win,
            )

        ask = snapshot.yes_ask_pct
        if ask is None:
            return StrategyDecision(Signal.HOLD, reason="no YES ask available to size against")
        if ask > self.max_entry_price_pct:
            return StrategyDecision(
                Signal.HOLD,
                reason=f"YES ask {ask:.0f} exceeds max_entry_price_pct {self.max_entry_price_pct:.0f} -- "
                "risk/reward too asymmetric this deep into favorite territory",
                features=features, predicted_probability=p_win,
            )

        margin = p_win - self.min_win_probability
        confidence = min(1.0, margin / self.full_confidence_margin)
        dollars = self.base_dollars + (self.max_dollars - self.base_dollars) * confidence
        size = max(1, int(dollars // (ask / 100)))

        return StrategyDecision(
            Signal.BUY_YES,
            size=size,
            reason=f"model predicts {p_win:.0%} win probability for {side_code} "
            f"(gate {self.min_win_probability:.0%}, {minutes_to_first_pitch:.0f}min to first pitch) -> ${dollars:.0f}",
            features=features,
            entry_kind="mlb_pregame",
            predicted_probability=p_win,
        )
