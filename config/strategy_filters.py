"""Strategy-level symbol and direction filters.

Rollout policy (see README.md "Strategy B/C Filter & Regime Policy" for the
full write-up):

- Strategy B (strategy_b_fvg_breaker_15m) runs whitelist + shorts-only + the
  BTC regime gate (backtest/regime.py: only signals when BTC is currently
  "net-trending" - see is_btc_regime_trending).
- Strategy C (strategy_c_fvg_breaker_4h) runs whitelist + shorts-and-longs,
  and is intentionally NOT regime-gated.

Do not "fix" this asymmetry to make both strategies consistent. Regime
filtering was tested against historical Strategy C trades and it reduced
both win rate and total R there, because it strips out DOGEUSDT's wins,
which occur independent of market regime (DOGEUSDT is a small-sample,
high-impact symbol for Strategy C - see backtest/weekly_report.py for
ongoing monitoring of whether that edge holds up).
"""

STRATEGY_B_WHITELIST = ["BTCUSDT", "ETHUSDT", "ATOMUSDT", "INJUSDT", "SOLUSDT"]
STRATEGY_B_DIRECTIONS_ALLOWED = ["SHORT"]

STRATEGY_C_WHITELIST = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT"]
STRATEGY_C_DIRECTIONS_ALLOWED = ["SHORT", "LONG"]
