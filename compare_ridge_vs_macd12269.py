from __future__ import annotations

import base64
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lr-vs-macd")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linear_regression_backtest import (
    TRADING_DAYS,
    financial_metrics,
    format_float,
    format_pct,
    load_vnindex_csv,
    max_drawdown,
)


DATA_PATH = Path("data.csv")
LR_OUTPUT = Path("outputs_linear_regression_backtest")
OUTPUT_DIR = Path("outputs_model_vs_macd12269")
COST_BPS = 5.0


def add_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TargetDate"] = out["Date"].shift(-1)
    out["target_simple_ret"] = out["Close"].shift(-1) / out["Close"] - 1
    out["target_log_ret"] = np.log(out["Close"].shift(-1) / out["Close"])
    return out.dropna(subset=["TargetDate", "target_simple_ret", "target_log_ret"]).reset_index(drop=True)


def build_macd12269_backtest(df: pd.DataFrame, target_dates: pd.Series, cost_bps: float = COST_BPS) -> pd.DataFrame:
    frame = add_return_targets(df)
    ema12 = frame["Close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["Close"].ewm(span=26, adjust=False).mean()
    frame["macd_line"] = ema12 - ema26
    frame["macd_signal"] = frame["macd_line"].ewm(span=9, adjust=False).mean()
    frame["macd_histogram"] = frame["macd_line"] - frame["macd_signal"]
    frame["position"] = (frame["macd_line"] > frame["macd_signal"]).astype(float)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["transaction_cost"] = frame["turnover"] * (cost_bps / 10000)
    frame["strategy_return"] = frame["position"] * frame["target_simple_ret"] - frame["transaction_cost"]
    frame["buy_hold_return"] = frame["target_simple_ret"]
    frame["strategy_equity"] = (1 + frame["strategy_return"]).cumprod()
    frame["buy_hold_equity"] = (1 + frame["buy_hold_return"]).cumprod()
    _, frame["strategy_drawdown"] = max_drawdown(frame["strategy_equity"])
    _, frame["buy_hold_drawdown"] = max_drawdown(frame["buy_hold_equity"])

    target_dates = pd.to_datetime(target_dates)
    frame = frame[frame["TargetDate"].isin(target_dates)].copy().reset_index(drop=True)
    frame["strategy_equity"] = (1 + frame["strategy_return"]).cumprod()
    frame["buy_hold_equity"] = (1 + frame["buy_hold_return"]).cumprod()
    _, frame["strategy_drawdown"] = max_drawdown(frame["strategy_equity"])
    _, frame["buy_hold_drawdown"] = max_drawdown(frame["buy_hold_equity"])
    return frame


def rebuild_lr_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ["Date", "TargetDate"]:
        frame[column] = pd.to_datetime(frame[column])
    frame["strategy_equity"] = (1 + frame["strategy_return"]).cumprod()
    frame["buy_hold_equity"] = (1 + frame["buy_hold_return"]).cumprod()
    _, frame["strategy_drawdown"] = max_drawdown(frame["strategy_equity"])
    _, frame["buy_hold_drawdown"] = max_drawdown(frame["buy_hold_equity"])
    return frame


def metrics_for_frame(name: str, frame: pd.DataFrame) -> dict[str, float | str]:
    metrics = financial_metrics(
        frame["strategy_return"],
        frame["buy_hold_return"],
        frame["position"],
        frame["TargetDate"],
        frame["turnover"],
    )
    return {"strategy": name, **metrics}


def build_comparison(lr: pd.DataFrame, macd: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics_for_frame("Ridge Linear Regression", lr), metrics_for_frame("MACD 12/26/9", macd)]
    comparison = pd.DataFrame(rows)
    ordered = [
        "strategy",
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "daily_var_95",
        "daily_cvar_95",
        "profit_factor",
        "win_rate_active_days",
        "exposure",
        "avg_daily_turnover",
        "number_of_trades",
        "beta_vs_buy_hold",
        "annual_alpha_vs_buy_hold",
        "information_ratio",
        "tail_ratio_95_5",
        "daily_skew",
        "daily_kurtosis",
        "benchmark_total_return",
        "benchmark_cagr",
    ]
    return comparison[ordered]


def add_advantage_row(comparison: pd.DataFrame) -> pd.DataFrame:
    ridge = comparison[comparison["strategy"] == "Ridge Linear Regression"].iloc[0]
    macd = comparison[comparison["strategy"] == "MACD 12/26/9"].iloc[0]
    advantage = {"strategy": "Ridge minus MACD"}
    for column in comparison.columns:
        if column != "strategy":
            advantage[column] = ridge[column] - macd[column]
    return pd.concat([comparison, pd.DataFrame([advantage])], ignore_index=True)


def plot_equity(lr: pd.DataFrame, macd: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "ridge_vs_macd12269_equity.png"
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(lr["TargetDate"], lr["strategy_equity"], color="#1464F4", linewidth=2.4, label="Ridge Linear Regression")
    ax.plot(macd["TargetDate"], macd["strategy_equity"], color="#D97706", linewidth=2.2, label="MACD 12/26/9")
    ax.plot(lr["TargetDate"], lr["buy_hold_equity"], color="#555555", linewidth=1.8, label="Buy & hold VN-Index")
    ax.set_title("Ridge Linear Regression vs MACD 12/26/9 - Equity Curve", fontsize=16, weight="bold")
    ax.set_ylabel("Growth of 1 VND")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_drawdown(lr: pd.DataFrame, macd: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "ridge_vs_macd12269_drawdown.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(lr["TargetDate"], lr["strategy_drawdown"], color="#1464F4", linewidth=2.2, label="Ridge Linear Regression")
    ax.plot(macd["TargetDate"], macd["strategy_drawdown"], color="#D97706", linewidth=2.2, label="MACD 12/26/9")
    ax.plot(lr["TargetDate"], lr["buy_hold_drawdown"], color="#555555", linewidth=1.5, label="Buy & hold")
    ax.fill_between(lr["TargetDate"], lr["strategy_drawdown"], 0, color="#1464F4", alpha=0.10)
    ax.fill_between(macd["TargetDate"], macd["strategy_drawdown"], 0, color="#D97706", alpha=0.10)
    ax.set_title("Drawdown Comparison", fontsize=16, weight="bold")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rolling_sharpe(lr: pd.DataFrame, macd: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "ridge_vs_macd12269_rolling_sharpe.png"
    window = 63
    lr_sharpe = lr["strategy_return"].rolling(window).mean() / lr["strategy_return"].rolling(window).std() * np.sqrt(TRADING_DAYS)
    macd_sharpe = macd["strategy_return"].rolling(window).mean() / macd["strategy_return"].rolling(window).std() * np.sqrt(TRADING_DAYS)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(lr["TargetDate"], lr_sharpe, color="#1464F4", linewidth=2.2, label="Ridge Linear Regression")
    ax.plot(macd["TargetDate"], macd_sharpe, color="#D97706", linewidth=2.2, label="MACD 12/26/9")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_title("63-Day Rolling Sharpe", fontsize=16, weight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_monthly_difference(lr: pd.DataFrame, macd: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "ridge_minus_macd12269_monthly_heatmap.png"
    lr_monthly = (1 + lr.set_index("TargetDate")["strategy_return"]).resample("ME").prod() - 1
    macd_monthly = (1 + macd.set_index("TargetDate")["strategy_return"]).resample("ME").prod() - 1
    diff = (lr_monthly - macd_monthly).to_frame("excess")
    diff["year"] = diff.index.year
    diff["month"] = diff.index.month
    heat = diff.pivot(index="year", columns="month", values="excess").reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * len(heat) + 2)))
    values = heat.to_numpy()
    vmax = np.nanpercentile(abs(values), 90) if np.isfinite(values).any() else 0.05
    image = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Monthly Excess Return: Ridge minus MACD 12/26/9", fontsize=16, weight="bold")
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(heat.index)), heat.index.astype(str))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1%}", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, format=lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_position_timeline(lr: pd.DataFrame, macd: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "ridge_vs_macd12269_positions.png"
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].fill_between(lr["TargetDate"], lr["position"], 0, color="#1464F4", alpha=0.45)
    axes[0].set_title("Ridge Linear Regression Exposure", weight="bold")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(macd["TargetDate"], macd["position"], 0, color="#D97706", alpha=0.45)
    axes[1].set_title("MACD 12/26/9 Exposure", weight="bold")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def display_table(table: pd.DataFrame) -> pd.DataFrame:
    percent_cols = {
        "total_return",
        "cagr",
        "annual_volatility",
        "max_drawdown",
        "daily_var_95",
        "daily_cvar_95",
        "win_rate_active_days",
        "exposure",
        "avg_daily_turnover",
        "annual_alpha_vs_buy_hold",
        "benchmark_total_return",
        "benchmark_cagr",
    }
    out = table.copy()
    for column in out.columns:
        if column == "strategy":
            continue
        if column in percent_cols:
            out[column] = out[column].map(format_pct)
        else:
            out[column] = out[column].map(format_float)
    return out


def write_report(comparison: pd.DataFrame, lr: pd.DataFrame, macd: pd.DataFrame, images: list[Path], output_dir: Path) -> None:
    ridge = comparison[comparison["strategy"] == "Ridge Linear Regression"].iloc[0]
    macd_row = comparison[comparison["strategy"] == "MACD 12/26/9"].iloc[0]
    winner = "Ridge Linear Regression" if ridge["sharpe"] >= macd_row["sharpe"] else "MACD 12/26/9"
    table_html = display_table(add_advantage_row(comparison)).to_html(index=False, escape=False)
    image_html = "\n".join(
        f'<h2>{path.stem.replace("_", " ").title()}</h2><img src="data:image/png;base64,{image_to_base64(path)}" />'
        for path in images
    )
    style = """
    <style>
    body { font-family: Arial, sans-serif; margin: 28px; color: #172033; background: #FAFBFC; }
    h1, h2 { color: #111827; }
    .note { background: #FFF7ED; border-left: 5px solid #D97706; padding: 12px 16px; margin: 16px 0 24px; }
    table { border-collapse: collapse; width: 100%; margin: 14px 0 30px; font-size: 13px; background: white; }
    th { background: #111827; color: white; }
    th, td { border: 1px solid #D8DEE9; padding: 8px 9px; text-align: right; white-space: nowrap; }
    td:first-child, th:first-child { text-align: left; }
    img { width: 100%; border: 1px solid #D8DEE9; margin: 14px 0 28px; background: white; }
    .small { color: #53606F; font-size: 13px; }
    </style>
    """
    html = f"""
    <!doctype html>
    <html lang="vi">
    <head><meta charset="utf-8"><title>Ridge vs MACD 12/26/9</title>{style}</head>
    <body>
    <h1>Ridge Linear Regression vs MACD 12/26/9</h1>
    <div class="note">
      <strong>Winner by Sharpe:</strong> {winner}<br>
      <strong>Comparison window:</strong> {lr["TargetDate"].min().date()} to {lr["TargetDate"].max().date()}<br>
      <span class="small">Both strategies are long/cash, close-to-close, signal at close applied to the next session, transaction cost {COST_BPS:.1f} bps on position changes.</span>
    </div>
    <h2>Financial & Statistical Metrics</h2>
    {table_html}
    {image_html}
    </body></html>
    """
    (output_dir / "ridge_vs_macd12269_report.html").write_text(html, encoding="utf-8")

    summary = [
        "# Ridge Linear Regression vs MACD 12/26/9",
        "",
        f"- Comparison window: {lr['TargetDate'].min().date()} to {lr['TargetDate'].max().date()}",
        f"- Winner by Sharpe: **{winner}**",
        f"- Ridge CAGR / Sharpe / MaxDD: **{ridge['cagr']:.2%} / {ridge['sharpe']:.2f} / {ridge['max_drawdown']:.2%}**",
        f"- MACD CAGR / Sharpe / MaxDD: **{macd_row['cagr']:.2%} / {macd_row['sharpe']:.2f} / {macd_row['max_drawdown']:.2%}**",
        "",
        "## Files",
        "",
        "- `ridge_vs_macd12269_comparison.csv`",
        "- `macd12269_test_backtest.csv`",
        "- `ridge_vs_macd12269_report.html`",
    ]
    summary.extend([f"- `{path.name}`" for path in images])
    (output_dir / "ridge_vs_macd12269_summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_path = LR_OUTPUT / "best_model_test_predictions.csv"
    if not lr_path.exists():
        raise FileNotFoundError(f"Missing {lr_path}. Run linear_regression_backtest.py first.")

    lr = rebuild_lr_frame(lr_path)
    df = load_vnindex_csv(DATA_PATH)
    macd = build_macd12269_backtest(df, lr["TargetDate"], COST_BPS)

    if len(macd) != len(lr):
        raise ValueError(f"Aligned rows differ: Ridge={len(lr)}, MACD={len(macd)}")

    comparison = build_comparison(lr, macd)
    comparison.to_csv(OUTPUT_DIR / "ridge_vs_macd12269_comparison.csv", index=False)
    macd.to_csv(OUTPUT_DIR / "macd12269_test_backtest.csv", index=False)

    images = [
        plot_equity(lr, macd, OUTPUT_DIR),
        plot_drawdown(lr, macd, OUTPUT_DIR),
        plot_rolling_sharpe(lr, macd, OUTPUT_DIR),
        plot_monthly_difference(lr, macd, OUTPUT_DIR),
        plot_position_timeline(lr, macd, OUTPUT_DIR),
    ]
    write_report(comparison, lr, macd, images, OUTPUT_DIR)

    print("Ridge vs MACD 12/26/9 comparison completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Comparison window: {lr['TargetDate'].min().date()} to {lr['TargetDate'].max().date()}")
    print(display_table(add_advantage_row(comparison)).to_string(index=False))


if __name__ == "__main__":
    main()
