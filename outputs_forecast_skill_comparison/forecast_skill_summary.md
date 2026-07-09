# Forecast Skill Comparison

- Window: 2022-10-04 to 2026-07-01
- Ridge forecast: predicted next-day log return > 0 means forecast up.
- MACD forecast: MACD line > signal line means forecast up; otherwise down/cash.

## Metrics

                  model accuracy balanced_accuracy precision_up recall_up specificity_down  f1_up    mcc score_corr_with_next_return forecast_up_frequency return_spread_up_minus_down
Ridge Linear Regression   54.08%            54.21%       60.52%    53.14%           55.28% 56.59% 0.0836                      0.0808                49.46%                       0.21%
           MACD 12/26/9   52.36%            51.86%       58.02%    55.81%           47.91% 56.89% 0.0370                     -0.0133                54.18%                       0.07%
       Ridge minus MACD    1.72%             2.35%        2.50%    -2.67%            7.37% -0.30% 0.0465                      0.0941                -4.72%                       0.14%

## Images

- `forecast_skill_metric_bars.png`
- `forecast_confusion_matrices.png`
- `forecast_rolling_directional_accuracy.png`
- `forecast_score_vs_next_return.png`
- `forecast_conditional_next_return.png`