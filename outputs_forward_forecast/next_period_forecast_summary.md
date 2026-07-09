# VN-Index Next-Period Forecast

- Last data date: `2026-07-01`
- Last close: `1,865.37`
- Horizon: `30` business sessions
- Ridge daily return assumption: `0.007%`
- Ridge projected close at horizon: `1,869.36` (0.21%)
- MACD current regime: `up`
- MACD daily return assumption: `0.176%`
- MACD projected close at horizon: `1,966.51` (5.42%)

## Method Note

Ridge gives a numeric one-step next-day return forecast. For a 30-session path, the latest one-step return is compounded forward because future OHLCV features are unknown.
MACD is a directional regime rule, not a numeric price model. Its path is produced by compounding the historical average next-day return observed under the current MACD regime.

## Images

- `vnindex_next_period_forecast_lines.png`
- `next_period_return_assumptions.png`