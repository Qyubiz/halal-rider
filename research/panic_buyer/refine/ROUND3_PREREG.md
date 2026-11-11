# Round 3 pre-registration (written 22 Oct 2026, before any 2018–2020 result was seen)

2024–26 has now been looked at for about 25 picks across rounds 1–2, so it is no longer a clean test. Round 3
adds a period no test has touched: **Jan 2018 – Dec 2020** (the 2018 bear market, the 2019 recovery and the
March 2020 crash). It is used **once**, at the end, only for the candidates fixed below.

## Holdout data

Binance spot hourly candles from Dec 2017 (warm-up) to Dec 2020 for the 16 live coins that traded then: BTC ETH
XRP BNB ADA TRX LTC LINK DOGE HBAR FET BCH (BCHABC before its Nov 2019 rename) DOT AVAX NEAR SOL, each from its
Binance listing. SUI, ARB and RENDER did not exist. 43 candles that don't start on the hour (2017–19 exchange
outages, BTC/ETH/BNB/LTC only) are dropped. Same engine, costs and fill rules as before (`engine.load_pre`,
`engine.score_holdout`). The research agents are not given this data.

## What gets one look at the holdout

- **A. Live settings, backtest timing:** the strategy as reported, 2021–23 ×5.75, 2024–26 ×2.61.
- **B. Live settings, as `panic_buyer.py` is coded:** after a market buy the +1.5% sell goes up at once, and the
  48h clock starts at the hour's start (`engine.run(mtp_now=True)`): 2021–23 ×4.73, 2024–26 ×2.66.
- **C. At most 3 round-3 candidates.** Each must have passed rule steps 1–4 (chosen on 2021–23, beats A in
  2024–26, with LUNA/FTT/SRM and with fills 0.2% worse) and survived adversarial verification.

## Decisions, fixed now

1. **Market-buy timing (A vs B).** If B beats A on the holdout (higher multiple, worst drop no more than 1 point
   deeper), keep `panic_buyer.py` as coded and restate the expected results as B's. Otherwise change the bot to
   match A: place the take-profit after a market buy at the next hourly cycle, and count 48h from the end of
   the buy hour. This is a fix so the bot does what was tested; either way the user decides.
2. **A round-3 candidate** goes to the user as a change to the live bot only if it also beats A on the holdout,
   both with normal fills and with fills 0.2% worse. Otherwise it is rejected, however close.
3. **Noise.** Differences smaller than the jitter band (live settings re-run with tiny perturbations: start date,
   depth ±0.1 point, fee ±0.005%) are reported as ties.

## Round-3 hypotheses (each at most 8 variants, grids written in the scripts before running)

- **Capped split ladder:** half the coin's order at 12%, the rest deeper, with the coin's total capped at 20–25%
  of equity (round 2's uncapped 40% version failed on one LUNA trade).
- **Take-profit relative to the 24h high:** sell part of the way back to the reference high instead of at a fixed +1.5%.
- **Time of day / weekend:** whether thin hours give better or worse dips.
