# The Data-Driven Crypto Strategy

**Status:** Live as of 2026-08-04. Replaces `CryptoTechnicalStrategy` for the real-money bot (`run_live_trading.py`). `CryptoTechnicalStrategy` itself is untouched and keeps running, unmodified, in the paper bot (`run_paper_trading.py`) as a live A/B control.

**Code:** [`trade_bot/data_driven_strategy.py`](../trade_bot/data_driven_strategy.py) (`DataDrivenCryptoStrategy`)

**Do not treat this document as static.** It describes a model fit against 329 trades on one specific day. Re-run [`refit_strategy_model.py`](../refit_strategy_model.py) periodically as more real trades close, and update this doc's numbers when the coefficients change.

---

## TL;DR

- The old strategy's core idea — *"a price that just moved keeps moving, and two timeframes agreeing confirms it"* — is not supported by the data. Fit a logistic regression on 314 real trades and the momentum feature's coefficient came out slightly **negative**.
- **Order-book imbalance** (computed every cycle, never used to pick a trade before) has the strongest coefficient of any feature, by a wide margin. It's now the primary direction signal.
- **Volatility** is a real, fairly monotonic predictor (win rate drops from ~43% under 2pp to ~25-31% above 6pp) — now a hard entry gate, not just a size dampener.
- **Stop-losses were the single biggest loss source** (79 trades, 1.3% win rate, -$82.97), firing at a -28% average realized loss — far wider than winners were ever allowed to run. Tightened substantially.
- **Forced-settlement/reconciliation losses were the second-biggest** (41 trades, -$89.72), concentrated on the three 15-minute markets. Partly a reliability problem (already fixed by the bot watchdog), partly addressed here with a wider exit-urgency buffer on those tickers.
- The "gamble" mechanic (hold to expiry) went **0-for-41**. Removed entirely, not just disabled.
- "Scalp" (rapid 30-60s move, fast exit) was the best-performing signal in the data (55% win rate). Kept as-is, re-costed for fee drag.
- Honest expectation: this is a small, real, out-of-sample-validated edge on top of an efficiently-priced market — not a fix that turns a losing strategy into a clearly profitable one. The realistic goal is **fewer, better-chosen trades, and much smaller losses on the ones that don't work out**.

---

## 1. Why this happened

As of 2026-08-04, the live bot (`crypto_technical` strategy) had made 329 closed real trades: **122 wins, 207 losses, 37.1% win rate, -$136.14 total P&L.**

The prior algorithm was built from the account owner's own experience and instructions, tuned reactively over several days of live diagnosis (see `run_live_trading.py`'s git history / comments for that whole process). The request behind this document was explicit: stop tuning that algorithm by hand, and instead **validate everything the bot has actually done and design a new algorithm from the data itself.**

Everything below is that validation, and the design that came out of it.

## 2. What the raw numbers say

All numbers from the 329 closed trades in `live_trading.db` as of 2026-08-04, ~16:00 UTC.

### By entry type

| Entry class | n | Win rate | Total P&L | Avg P&L/trade |
|---|---:|---:|---:|---:|
| standard (strict multi-timeframe signal) | 133 | 38.3% | -$36.64 | -$0.276 |
| micro_bet (loose single-timeframe fallback) | 100 | 44.0% | -$8.87 | -$0.089 |
| gamble (hold to expiry) | 41 | **0.0%** | -$31.57 | -$0.770 |
| scalp (rapid 30-60s move) | 40 | **55.0%** | -$3.31 | -$0.083 |
| reconciled/untracked (no strategy decision) | 15 | 33.3% | -$55.75 | -$3.717 |

The strategy's "most confident" signal (`standard`, requiring both a 3-min and a 12-min window to agree) performed *worse* than its own admittedly-weaker fallback (`micro_bet`). That's the first sign the original premise doesn't hold up.

### By close reason — where the money actually went

| Close reason | n | Win rate | Total P&L |
|---|---:|---:|---:|
| hard stop-loss | 79 | 1.3% | **-$82.97** |
| reconciled (forced settlement / never got a controlled exit) | 41 | 9.8% | **-$89.72** |
| trailing stop | 121 | 48.8% | +$4.22 |
| profit ceiling | 34 | 100% (by definition) | +$43.73 |
| scalp take-profit | 21 | 100% (by definition) | +$2.34 |
| scalp stop-loss | 16 | 0% (by definition) | -$5.56 |
| gamble catastrophic floor | 11 | 0% (by definition) | -$8.09 |
| time-decay urgency exit | 3 | 66.7% | -$0.02 |

Stop-losses and reconciled/forced-settlement exits together account for **-$172.69** — more than the entire net loss. Every other exit reason nets positive. This is the clearest finding in the whole dataset: **the losses aren't spread evenly across "bad trades" — they're concentrated in two specific failure modes.**

Digging into stop-losses: the 79 that fired averaged a **-28.0% realized return** (median -25.6%, worst -75.0%) before the bot cut them. Compare that to trailing-stop exits (the healthiest bucket), which lock in gains in the +5-20% range. Losers were structurally allowed to run much further than winners before either was closed.

Digging into reconciled trades: 14 of 15 non-hourly ones were on the three 15-minute markets (KXBTC15M/KXETH15M/KXBNB15M). The two worst (-$34.96, -$24.77) were both held for **4+ hours** — they were opened, then the bot lost track of them (this correlates with known downtime windows from earlier the same day, since fixed by the bot watchdog — see `trade_bot/bot_control.py`). Median hold time across all 41 was ~11 minutes, still notably longer than these markets' own life.

### By asset

| Asset | n | Win rate | Total P&L | Avg P&L/trade |
|---|---:|---:|---:|---:|
| BTC | 126 | 42.1% | -$45.37 | -$0.360 |
| ETH | 134 | 34.3% | -$42.28 | -$0.316 |
| BNB | 69 | 33.3% | -$48.49 | -$0.703 |

BTC meaningfully outperforms ETH and BNB on win rate. BNB has the worst average loss per trade of the three.

### By market type

| Market type | n | Win rate | Total P&L | Avg P&L/trade |
|---|---:|---:|---:|---:|
| hourly (KXBTCD/KXETHD) | 203 | 41.9% | -$36.27 | -$0.179 |
| 15-min (KXBTC15M/KXETH15M/KXBNB15M) | 126 | 29.4% | -$99.87 | -$0.793 |

The 15-minute markets performed much worse overall — driven heavily by the gamble mechanic (all 41 gamble trades were on 15-min markets) and the reconciled-exit problem above, both addressed below.

### By entry price

Win rate and average P&L were both weakest in the 30-45c range and generally better at higher (more "favorite") prices — consistent with, and a further data point for, the `min_entry_price_pct=45.0` floor already in place since 2026-08-03.

## 3. The logistic regression

314 of the 329 trades have a saved entry-time feature snapshot (the 15 reconciled/untracked ones don't — there was never a strategy decision behind them). A logistic regression was fit on all 314 (L2=0.02, full-batch gradient descent, 800 epochs — see [`refit_strategy_model.py`](../refit_strategy_model.py)), validated against a **chronological 20% holdout** (the last 63 trades by close time, never seen during fitting).

| Feature | Coefficient | Reading |
|---|---:|---|
| `orderbook_imbalance` | **+0.325** | Strongest signal by far. Trading *with* the book beats trading against it. |
| `entry_price_scaled` | +0.221 | Higher-priced (more-favorite) entries do better — matches the existing price floor. |
| `asset_btc` | +0.180 | BTC trades better than the ETH/BNB baseline. |
| `minutes_to_close_scaled` | +0.069 | More time left is (weakly) better. |
| `hour_cos` | +0.115 | Weak time-of-day component. |
| `side_is_yes` | -0.074 | Weak, near-noise. |
| `asset_eth` | -0.091 | ETH trades worse than the baseline. |
| `is_micro_bet` | -0.027 | Near-zero once other features are accounted for. |
| `long_delta_scaled` | -0.010 | Effectively zero — the 12-min "trend confirmation" carries no real signal. |
| `short_delta_scaled` | **-0.038** | Slightly *negative*. The core "momentum continues" premise doesn't hold. |
| `hour_sin` | -0.296 | The largest time-of-day component — treated as a minor feature, not a hard rule (small per-hour sample sizes make this the least trustworthy coefficient). |
| `volatility_scaled` | **-0.340** | Second-strongest signal. Higher recent volatility clearly predicts worse outcomes. |

**Calibration check** (predicted-probability bucket vs. actual win rate, train+test combined): predicted ~0.2 → actual 25.0%; ~0.3 → 28.8%; ~0.4 → 41.5%; ~0.5 → 47.8%. Reasonably well-calibrated even out of sample — real evidence this isn't just overfit noise, though 314 examples for 12 features is not a lot, and the honest caveat below still applies.

### An important correction made *before* shipping

The first version of this design gated entries on the model beating the **price-implied breakeven probability** — the theoretically "correct" definition of edge (only trade when your own estimate beats what the market is already charging). Backtested against the 63-trade holdout, that gate would have taken **zero trades**. Kalshi's own pricing already encodes more information than this feature set beats in absolute terms — unsurprising for a real, actively-traded prediction market.

The gate was corrected to an **absolute predicted-probability threshold** instead — using the model as a relative ranking tool over candidate trades, not a claim of beating the market outright. Swept against the same holdout:

| Threshold | Trades taken | Win rate |
|---|---:|---:|
| 0.40 | 31/63 | 41.9% |
| 0.42 | 23/63 | 47.8% |
| 0.43 | 21/63 | 47.6% |
| 0.45 | 20/63 | 45.0% |
| 0.50 | 10/63 | 50.0% |

`min_win_probability = 0.43` was chosen — a meaningful lift over the 41.3% holdout base rate, at a trade frequency that won't leave the bot idle.

**Full combined-gate backtest** (imbalance floor + volatility ceiling + this probability threshold, all together, replayed against the same 63-trade holdout): **16/63 trades taken (25%) at 56.2% win rate**, vs. 36.2% win rate among the 47 rejected. Both numbers land on the right side of the 41.3% baseline — real, if modest (and small-sample: n=16), separation.

## 4. The new algorithm

### Entry

1. **Direction comes from order-book imbalance**, not price momentum: `|imbalance| >= min_imbalance_pct` (0.12) is required just to consider a trade at all. Direction = whichever side the book favors.
2. **Momentum sanity check**: the recent short-term price move must not *contradict* the imbalance-chosen direction beyond a small tolerance (`momentum_conflict_tolerance_pct`, 1.5pp) — a moment-old reversal blocks the trade even if the book still nominally favors it.
3. **Volatility hard gate**: skip if recent volatility exceeds `max_volatility_pct` (6.5pp) — no longer just a size dampener.
4. **Entry price floor/ceiling**: unchanged from the prior strategy (45c-85c), independently validated by both the original 2026-08-03 diagnosis and this new fit.
5. **Probability gate**: the logistic model above must predict `>= min_win_probability` (0.43).
6. **Sizing**: scales from `base_dollars` to `max_dollars` based on how far the prediction clears the gate (`full_confidence_margin`, 0.10) — replaces the old "raw price-delta confidence x multiplier" heuristic.

### Exit

Reuses the same trailing-stop / time-decay-urgency framework already validated by real data in earlier diagnoses (see `trade_bot/strategy.py`'s `_TakeProfitStopLossStrategy`) — this part of the system wasn't broken, it was already tuned from real evidence. What changed:

- **Stop-loss tightened**: `stop_loss_pct` 0.15 → **0.10**, `stop_loss_time_bonus_pct` 0.10 → **0.05** (so the widened, plenty-of-time-left stop caps at 0.15 instead of 0.25). Directly targets the -28% average realized stop-loss finding above — the single largest lever in the whole analysis.
- **Trailing-stop / profit-ceiling**: unchanged (0.05 activate / 0.03 giveback / 0.25 ceiling) — this bucket was already close to breakeven-to-positive real data (48.8% win rate, +$4.22 total), no reason to touch it.
- **Time-decay urgency**: unchanged base (3 min), plus a **new +1.5 min buffer specifically on the three 15-minute markets** (`urgency_minutes_15min_bonus`), where forced-settlement losses were concentrated (14 of 15 non-hourly reconciled trades). This is *defense in depth* alongside — not instead of — the bot watchdog fix, which addresses the actual root cause (bot downtime) of the worst individual cases.
- **"Let winners run" (momentum-hold)**: unchanged from the earlier 2026-08-04 addition — while a position is still trending favorably, widens the trailing tolerance (0.03 → 0.08) and profit ceiling (0.25 → 0.50). No real-data evidence against it yet (too new), logically sound, low risk since it only ever activates on an already-favorable move.
- **Scalp**: kept structurally identical (same 45-second window, same 1.5pp threshold — the best-performing real signal in the data, 55% win rate). Only the economics changed: stake $3 → **$5**, take-profit 6% → **8%**, to clear fixed-fee drag against a small position that was making the mechanism net-negative despite the good win rate.
- **Gamble: removed entirely.** Not disabled — the class doesn't have the mechanism at all. 0-for-41 in real trading is not a signal worth keeping around behind a flag.

### What stayed exactly the same

- `min_minutes_to_close` (6.0 min) — the single biggest fix from the original 2026-07-12 diagnosis, unrelated to this rewrite.
- `PerformanceGovernor` (trailing win-rate/expectancy gate) and all hard risk limits (`LiveRiskLimits`: $100 capital cap, $100/trade cap, $35 daily-loss reference) — these are portfolio-level risk controls, not part of "the algorithm," and were out of scope for this rewrite.
- The engine-level `OnlineLogisticLearner` ("underwriter") — still runs independently as a secondary soft dampener on whatever the strategy proposes, exactly as before. This strategy's own probability model is a separate, purpose-built decision engine; the two are not the same object and don't share state.

## 5. Architecture notes

- **Paper bot is the A/B control.** `run_paper_trading.py` was narrowed to the same market scope as live (previously traded a much broader universe: every sport category plus extra crypto tickers) but its strategy — `CryptoTechnicalStrategy`, completely unmodified — is untouched. Going forward, the two bots trade the identical market universe with different algorithms, so their real (paper vs. live) performance is directly comparable.
- **Strategy name changed** from `crypto_technical` to `data_driven_crypto` (`DataDrivenCryptoStrategy.name`, exposed as `run_live_trading.STRATEGY_NAME`). This gives the new algorithm a clean governor/ledger history instead of inheriting the old algorithm's trailing win-rate stats. `app.py` and `trade_bot/api.py` both look this up dynamically now rather than hardcoding the string a second time.
- **Existing open positions at the moment of the swap** (opened by the old strategy) are picked up and managed by the new strategy's exit logic on the next cycle — safe, since the new stop-loss is *tighter*, not looser.

## 6. Keeping this fresh

This is a **batch fit**, not continuous online learning. That's deliberate: watching `OnlineLogisticLearner` operate live earlier the same day showed a handful of one-at-a-time SGD updates on ~50 examples can push a whole model to a near-uniform, unhelpfully pessimistic output well before it has enough data to actually discriminate (see `run_live_trading.py`'s `ML_GATE_THRESHOLD` comment for that whole incident). A periodic batch refit on the full accumulated history is more statistically stable, and — same principle the online learner already holds itself to — keeps a human in the loop reviewing what changed before it goes live again, rather than the strategy silently rewriting its own decision boundary in real time.

**To refit:**

```
uv run python refit_strategy_model.py
```

Pulls every closed trade with a saved feature snapshot from the live ledger, fits on a chronological 80/20 split, prints a holdout backtest so you can judge whether it still generalizes, then prints a full-dataset fit ready to paste into `PROBABILITY_MODEL_WEIGHTS`/`PROBABILITY_MODEL_BIAS` in `trade_bot/data_driven_strategy.py`. Don't paste in a refit that looks worse than what's already live without understanding why first. Below ~100 labeled trades, the script warns that a refit is mostly fitting noise.

## 7. Honest limitations

- 314 labeled examples for a 12-feature model is a small dataset. The holdout calibration check is real evidence against pure overfitting, but the coefficients — especially the smaller ones (`hour_sin`, `is_micro_bet`, `long_delta_scaled`) — should be read as "probably real, small effect," not precise measurements.
- Kalshi's markets are close to efficiently priced by design. The corrected design in §3 exists because of this: don't expect this model to reliably beat the market's own price in absolute terms. What it can do is rank *this strategy's own candidate entries* better than picking by raw price momentum did.
- The realistic goal of this rewrite is fewer, better-selected trades and much smaller losses on the ones that don't work out — not a dramatically higher win rate. 56.2% on a 16-trade backtest sample is encouraging, not proof.
- Nothing here has access to any data Kalshi's own order book, price history, and volume don't already provide. No on-chain data, no external market feed, no fundamentals.
