> **Last updated**: 2026-03-11

# MACD + RSI HTF Spot Entry

## Status

Initial alpha research completed. The direction is **not dead**, but the original thesis needed reinterpretation. The TradingView script the user actually meant is a **weekly RSI oversold + MACD histogram reversal** signal, not a strict monthly MACD golden-cross trigger. That faithful weekly version is researchable; the strict monthly branch is not.

## Goal

Evaluate whether the user's TradingView logic, faithfully replicated as:

- `RSI[1] < 30`
- `hist[2] > hist[1] and hist[1] < hist`

can form a robust **weekly long-only crypto accumulation signal**, and whether the ATR-normalized add-on condition improves or degrades it.

## Archetype

- Type: `htf_long_only_recovery_entry`
- Integration target: `standalone_spot_entry_candidate`
- Target gap: slow-timeframe spot accumulation / cyclical recovery entry

## Hypothesis

The usable signal is not a strict MACD line golden cross. Instead, a weekly oversold condition followed by a **MACD histogram local-bottom reversal** may identify early recovery windows before the slower MACD line crossover appears. This should create more timely and more frequent crypto accumulation signals than the strict monthly interpretation.

## Causal Implementation Notes

1. The faithful Pine replication should use **weekly bars labeled at weekly open**, matching TradingView-style timestamps such as `2022-07-04`.
2. The signal is computed on completed weekly bars and entered at the **next weekly open**.
   In the daily dataset this is represented by the next Monday daily open after the signal week closes.
3. The primary condition is:
   - `RSI[1] < 30`
   - `hist[2] > hist[1] and hist[1] < hist`
4. The ATR add-on condition is:
   - `RSI[1] < 30`
   - `atrhist[1] < atrhist`
5. Use `spot` as the intended deployment domain, but compare with `futures` because the original TradingView chart was shown on `BTCUSDT.P`.

## Faithful Pine Logic

The user-provided TradingView code corresponds to:

```python
buycondition = (rsi.shift(1) < 30) & (hist.shift(2) > hist.shift(1)) & (hist.shift(1) < hist)
atrbuycondition = (rsi.shift(1) < 30) & (atrhist.shift(1) < atrhist)
```

This is materially different from a strict `MACD golden cross + RSI oversold` interpretation:

- it uses **histogram reversal**, not MACD line crossover
- it only requires `RSI[1] < 30`
- it is naturally a **weekly reversal signal**, not a monthly macro trigger

## Feasibility Findings

- Repo support already exists for `MACD` and `RSI` indicators in `src/qtrade/indicators/`.
- Repo support already exists for causal HTF alignment in `qtrade.strategy.filters.causal_resample_align()`.
- The current active config set is futures-first; there is no active `research_*.yaml` spot config for this direction yet.
- A new bootstrap config now exists at `config/research_macd_rsi_htf_spot_entry.yaml`.
- Local `spot 1d` history has now been bootstrapped for `BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT`.
- TradingView-style weekly bar labeling is reproduced by resampling with Monday-open labeling; this recreates the `2022-07-04` BTC signal.

## First-Pass EDA

Using completed **weekly** bars and entering at the **next weekly open** after each signal:

### Main signal: `buycondition`

- `futures` universe aggregate:
  - Events: `21`
  - Mean forward return: `20d +3.8%`, `60d +16.8%`, `120d +44.3%`
  - Mean 60d hit rate across symbols: `50.0%`

- `spot` universe aggregate:
  - Events: `35`
  - Mean forward return: `20d +11.0%`, `60d +50.2%`, `120d +189.6%`
  - Mean 60d hit rate across symbols: `59.1%`

- BTC weekly replication check:
  - `futures` shows `2022-07-04` and `2022-08-29`
  - `spot` shows `2022-01-31`, `2022-07-04`, and `2022-08-29`
  - Therefore the user's TradingView-observed `2022-07-04` signal is faithfully reproduced

### ATR add-on: `atrbuycondition`

- `futures` universe aggregate:
  - Events: `81`
  - Mean forward return: `20d +13.3%`, `60d +12.4%`, `120d +32.8%`
  - Mean 60d hit rate: `51.1%`

- `spot` universe aggregate:
  - Events: `150`
  - Mean forward return: `20d +10.9%`, `60d +15.8%`, `120d +76.3%`
  - Mean 60d hit rate: `57.1%`

- Verdict:
  - The ATR add-on greatly increases signal count, but it is clearly noisier than the main `buycondition`
  - It should be treated as a secondary ablation branch, not as the primary thesis

### Regime concentration

- The strongest positive outcomes are still concentrated in recovery windows and alt-led rebounds, especially:
  - `2020` spot events in `SOL/AVAX`
  - `2022` recovery entries in `AVAX/SOL`
- After correcting execution timing, BTC remains the weakest major branch:
  - `BTC futures buycondition`: `2` valid events, `60d` mean approximately `-3.2%`
  - `BTC spot buycondition` post-2021: `3` valid events, `60d` mean approximately `-1.3%`
- ETH is materially stronger than BTC:
  - `ETH futures` post-2021: `5` events, `60d` mean approximately `+42.6%`
  - `ETH spot` post-2021: `6` events, `60d` mean approximately `+35.3%`

### Robustness slices

- `futures`, all symbols:
  - `21` events
  - `60d` mean approximately `+23.2%`

- `futures`, `BTC/ETH` only:
  - `7` events
  - `60d` mean approximately `+29.5%`
  - Result survives the major-coin restriction, but is driven much more by `ETH` than `BTC`

- `spot`, all symbols:
  - `35` events
  - `60d` mean approximately `+61.3%`

- `spot`, `BTC/ETH` only:
  - `14` events
  - `60d` mean approximately `+10.9%`
  - Still positive after removing alts, but dramatically weaker than the full-universe result

- `spot`, `BTC/ETH` and `post-2021`:
  - `9` events
  - `60d` mean approximately `+23.1%`
  - This is encouraging enough to keep the direction alive, but still too small-sample for handoff

### Concentration buckets

- `BTC` alone remains weak:
  - `futures BTC-only`: `2` events, `60d` mean approximately `-3.2%`
  - `spot BTC-only post-2021`: `3` events, `60d` mean approximately `-1.3%`
  - Interpretation: this is not currently supported as a BTC-only timing rule

- `ETH` alone is much stronger:
  - `futures ETH-only`: `5` events, `60d` mean approximately `+42.6%`
  - `spot ETH-only post-2021`: `6` events, `60d` mean approximately `+35.3%`
  - Interpretation: ETH carries a meaningful share of the major-coin result

- `alts ex BTC/ETH` are the strongest branch:
  - `spot alts ex BTC/ETH post-2022`: `17` events, `60d` mean approximately `+17.8%`
  - `futures alts ex BTC/ETH`: `14` events, `60d` mean approximately `+20.1%`
  - Interpretation: the signal behaves most like a recovery-entry detector for majors/alts rather than a BTC-led macro trigger

### Cross-Asset Expansion

To test the user's broader "long-term investing" intuition, the same faithful weekly Pine logic was expanded to Yahoo Finance proxies:

- `crypto`: `BTC-USD`, `ETH-USD`, `SOL-USD`, `DOGE-USD`
- `US broad ETFs`: `SPY`, `QQQ`, `IWM`
- `US growth proxies`: `SOXX`, `ARKK`
- `Taiwan proxies`: `EWT`, `0050.TW`, `2330.TW`

Coverage was adequate across all downloaded symbols:

- Weekly coverage:
  - `crypto`: `100%`
  - `US ETFs`: `100%`
  - `Taiwan proxies`: approximately `99.4%`

First-pass 60d forward-return summary after **next weekly open** execution:

- `crypto` group:
  - Average 60d mean across tested symbols: approximately `+35.8%`
  - Strongest names remain `SOL` and `DOGE`, while `BTC` stays weak

- `US broad ETFs`:
  - `SPY`: `8` events, `60d` mean approximately `+8.2%`, hit rate `87.5%`
  - `QQQ`: `6` events, `60d` mean approximately `+4.0%`, hit rate `83.3%`
  - `IWM`: `11` events, `60d` mean approximately `+6.2%`, hit rate `81.8%`

- `US growth proxies`:
  - `SOXX`: `10` events, `60d` mean approximately `+10.1%`, hit rate `90.0%`
  - `ARKK`: `9` events, `60d` mean approximately `+1.9%`
  - `ARKK post-2021`: `7` events, `60d` mean approximately `-2.9%`

- `Taiwan proxies`:
  - `EWT`: `12` events, `60d` mean approximately `+3.0%`
  - `0050.TW`: `13` events, `60d` mean approximately `+3.8%`
  - `2330.TW`: `8` events, `60d` mean approximately `+13.2%`

Interpretation:

- This materially weakens the claim that the signal is "only ETH."
- The logic appears to generalize beyond crypto into long-only equity proxies and Taiwan proxies.
- However, the magnitude outside crypto is much smaller and more consistent with a **slow recovery-entry heuristic** than a high-alpha trading rule.
- The signal seems strongest in:
  - high-beta crypto recoveries
  - growth / semiconductor / cyclical equity regimes
- It is weakest in:
  - BTC itself
  - speculative long-duration growth proxies like `ARKK` in the post-2021 regime

### Is It Better Than Just Buying Every Week?

To evaluate whether this is a useful **active buy point** rather than just a nice-looking pattern, the signal was compared against a simple baseline:

- baseline = buy at every weekly open
- signal = buy only when the faithful Pine condition triggers
- compare forward returns after `60d` and `120d`

Key results:

- `BTC-USD`:
  - signal `60d` mean: approximately `+2.8%`
  - baseline `60d` mean: approximately `+13.9%`
  - verdict: **worse than passive weekly buying**

- `ETH-USD`:
  - signal `60d` mean: approximately `+13.2%`
  - baseline `60d` mean: approximately `+11.2%`
  - verdict: modestly better than passive buying, and clearly better post-2021

- `SOL-USD`:
  - signal `60d` mean: approximately `+93.2%`
  - baseline `60d` mean: approximately `+46.0%`
  - verdict: substantially better, but still within a high-beta crypto context

- `SPY`:
  - signal `60d` mean: approximately `+8.2%`
  - baseline `60d` mean: approximately `+3.0%`
  - verdict: promising as a selective long-term buy trigger

- `IWM`:
  - signal `60d` mean: approximately `+6.2%`
  - baseline `60d` mean: approximately `+2.5%`
  - verdict: positive active-buy improvement

- `SOXX`:
  - signal `60d` mean: approximately `+10.1%`
  - baseline `60d` mean: approximately `+5.3%`
  - verdict: one of the cleaner equity use cases

- `2330.TW`:
  - signal `60d` mean: approximately `+13.2%`
  - baseline `60d` mean: approximately `+5.5%`
  - verdict: strong Taiwan proxy candidate

- weak or mixed cases:
  - `QQQ`: roughly neutral vs baseline
  - `ARKK`: worse than baseline
  - `EWT`: only marginally positive
  - `0050.TW`: mildly positive in full sample, weaker post-2021

Interpretation:

- This supports the user's intuition that the signal can work as a **main active buy trigger** for some long-term assets.
- But it is **not universal**.
- The most credible current use case is:
  - growth / cyclical / recovery-sensitive assets
  - where you want a disciplined, low-frequency "buy the relative low" trigger
- The least credible current use case is:
  - BTC as a standalone accumulation timer
  - highly speculative growth names with unstable post-2021 behavior

## Interpretation

- The user's intuition was valid, but the operative signal is **not** the strict monthly thesis.
- The faithful Pine logic behaves like a **weekly early-reversal detector**:
  - RSI oversold on the prior week
  - MACD histogram bottoming and turning up
- This explains why the TradingView chart shows a `2022-07-04` signal while the stricter monthly/cross interpretation looked empty.
- After correcting execution to the **next weekly open**, the signal still survives, so the idea is not an artifact of same-week entry timing.
- The main `buycondition` looks materially better than `atrbuycondition`.
- However, the edge currently appears:
  - clearly weaker in BTC than in ETH / major alts
  - concentrated in major rebound windows
  - still too small-sample for immediate developer handoff
- The best current reading is:
  - **not** a universal BTC timing rule
  - **more likely** a general recovery-entry heuristic that is strongest in ETH / majors / cyclically sensitive assets
  - potentially useful as an **active accumulation trigger** when it beats unconditional weekly buying, rather than as a pure trading signal

## Next Action

1. Continue the direction as **faithful Pine weekly histogram-reversal research**, not as monthly MACD-cross research.
2. Test whether the main `buycondition` still survives when restricted to:
   - BTC/ETH only
   - post-2021 sample only
   - causal next-week or next-day-open execution variants
3. Decide whether the direction should be reframed as:
   - `ETH + majors recovery entry`, or
   - `cross-asset long-only recovery heuristic excluding BTC-led claims`
4. Keep `atrbuycondition` only as a lower-priority ablation branch.
5. If the direction continues to survive, reframe the deliverable as an **active buy framework** for suitable long-term assets rather than a universal bottom-picking strategy.

## Provisional Verdict

- Recommendation: `continue_as_faithful_weekly_pine_replication`
- Handoff: `do_not_handoff_yet`
- Reason: the TradingView logic is reproducible, survives anti-bias-correct weekly execution, and shows promising portability beyond crypto, but BTC is still weak and the current evidence is still exploratory rather than handoff-ready.
