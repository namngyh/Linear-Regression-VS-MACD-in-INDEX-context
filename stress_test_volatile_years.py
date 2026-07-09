from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stress-years")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from linear_regression_backtest import (
    TRADING_DAYS,
    build_backtest_frame,
    financial_metrics,
    load_vnindex_csv,
    make_features,
    max_drawdown,
)


DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("outputs_stress_volatile_years")
READ_PATH = Path("READ.md")
COST_BPS = 5.0
RIDGE_ALPHA = 500.0
TOP_N_YEARS = 6
MIN_TRAIN_ROWS = 756
MIN_YEAR_ROWS = 120


def add_macd_backtest(df: pd.DataFrame, target_dates: pd.Series) -> pd.DataFrame:
    frame = df.copy()
    frame["TargetDate"] = frame["Date"].shift(-1)
    frame["target_simple_ret"] = frame["Close"].shift(-1) / frame["Close"] - 1
    frame["target_log_ret"] = np.log(frame["Close"].shift(-1) / frame["Close"])
    ema12 = frame["Close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["Close"].ewm(span=26, adjust=False).mean()
    frame["macd_line"] = ema12 - ema26
    frame["macd_signal"] = frame["macd_line"].ewm(span=9, adjust=False).mean()
    frame = frame.dropna(subset=["TargetDate", "target_simple_ret", "target_log_ret"]).copy()
    frame["position"] = (frame["macd_line"] > frame["macd_signal"]).astype(float)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["transaction_cost"] = frame["turnover"] * (COST_BPS / 10000)
    frame["strategy_return"] = frame["position"] * frame["target_simple_ret"] - frame["transaction_cost"]
    frame["buy_hold_return"] = frame["target_simple_ret"]
    frame = frame[frame["TargetDate"].isin(pd.to_datetime(target_dates))].copy().reset_index(drop=True)
    frame["strategy_equity"] = (1 + frame["strategy_return"]).cumprod()
    frame["buy_hold_equity"] = (1 + frame["buy_hold_return"]).cumprod()
    _, frame["strategy_drawdown"] = max_drawdown(frame["strategy_equity"])
    _, frame["buy_hold_drawdown"] = max_drawdown(frame["buy_hold_equity"])
    return frame


def select_stress_years(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in data.groupby(data["TargetDate"].dt.year):
        year = int(year)
        train_rows = int((data["TargetDate"] < pd.Timestamp(year=year, month=1, day=1)).sum())
        if train_rows < MIN_TRAIN_ROWS or len(frame) < MIN_YEAR_ROWS:
            continue
        buy_hold_equity = (1 + frame["target_simple_ret"]).cumprod()
        buy_hold_mdd, _ = max_drawdown(buy_hold_equity)
        rows.append(
            {
                "year": year,
                "rows": len(frame),
                "train_rows_before_year": train_rows,
                "start": frame["TargetDate"].min().date(),
                "end": frame["TargetDate"].max().date(),
                "annualized_volatility": frame["target_simple_ret"].std(ddof=0) * np.sqrt(TRADING_DAYS),
                "buy_hold_return": (1 + frame["target_simple_ret"]).prod() - 1,
                "buy_hold_max_drawdown": buy_hold_mdd,
                "worst_daily_return": frame["target_simple_ret"].min(),
                "best_daily_return": frame["target_simple_ret"].max(),
            }
        )
    stats = pd.DataFrame(rows).sort_values("annualized_volatility", ascending=False)
    return stats.head(TOP_N_YEARS).reset_index(drop=True)


def ridge_year_backtest(data: pd.DataFrame, feature_cols: list[str], year: int) -> pd.DataFrame:
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31)
    train = data[data["TargetDate"] < year_start].copy()
    test = data[(data["TargetDate"] >= year_start) & (data["TargetDate"] <= year_end)].copy()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA, max_iter=20000)),
        ]
    )
    model.fit(train[feature_cols], train["target_log_ret"])
    pred = pd.Series(model.predict(test[feature_cols]), index=test.index)
    out = build_backtest_frame(test, pred, cost_bps=COST_BPS)
    out["year"] = year
    out["strategy_name"] = "Ridge Linear Regression"
    return out


def metric_row(year: int, strategy: str, frame: pd.DataFrame) -> dict[str, float | str]:
    metrics = financial_metrics(
        frame["strategy_return"],
        frame["buy_hold_return"],
        frame["position"],
        frame["TargetDate"],
        frame["turnover"],
    )
    return {"year": year, "strategy": strategy, **metrics}


def build_stress_test() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    raw = load_vnindex_csv(DATA_PATH)
    data, feature_cols = make_features(raw)
    stress_years = select_stress_years(data)
    metric_rows = []
    ridge_frames: dict[int, pd.DataFrame] = {}
    macd_frames: dict[int, pd.DataFrame] = {}

    for year in stress_years["year"].astype(int):
        ridge = ridge_year_backtest(data, feature_cols, year)
        macd = add_macd_backtest(raw, ridge["TargetDate"])
        macd["year"] = year
        macd["strategy_name"] = "MACD 12/26/9"
        ridge_frames[year] = ridge
        macd_frames[year] = macd
        metric_rows.append(metric_row(year, "Ridge Linear Regression", ridge))
        metric_rows.append(metric_row(year, "MACD 12/26/9", macd))

    metrics = pd.DataFrame(metric_rows)
    return stress_years, metrics, ridge_frames, macd_frames


def add_advantage(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in metrics.groupby("year"):
        ridge = frame[frame["strategy"] == "Ridge Linear Regression"].iloc[0]
        macd = frame[frame["strategy"] == "MACD 12/26/9"].iloc[0]
        row = {"year": year, "strategy": "Ridge minus MACD"}
        for column in metrics.columns:
            if column not in {"year", "strategy"}:
                row[column] = ridge[column] - macd[column]
        rows.append(row)
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True).sort_values(["year", "strategy"])


def plot_equity_panels(ridge_frames: dict[int, pd.DataFrame], macd_frames: dict[int, pd.DataFrame]) -> Path:
    path = OUTPUT_DIR / "stress_years_equity_panels.png"
    years = list(ridge_frames)
    fig, axes = plt.subplots(len(years), 1, figsize=(14, max(4, 3.1 * len(years))), sharex=False)
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        ridge = ridge_frames[year]
        macd = macd_frames[year]
        ax.plot(ridge["TargetDate"], ridge["strategy_equity"], color="#1464F4", linewidth=2.1, label="Ridge")
        ax.plot(macd["TargetDate"], macd["strategy_equity"], color="#D97706", linewidth=2.0, label="MACD 12/26/9")
        ax.plot(ridge["TargetDate"], ridge["buy_hold_equity"], color="#555555", linewidth=1.5, label="Buy & hold")
        ax.set_title(f"{year} Stress Year Equity", weight="bold")
        ax.set_ylabel("Growth of 1")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metric_bars(metrics: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "stress_years_metric_bars.png"
    selected = metrics[metrics["strategy"].isin(["Ridge Linear Regression", "MACD 12/26/9"])].copy()
    years = sorted(selected["year"].unique())
    x = np.arange(len(years))
    width = 0.35
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    metric_defs = [
        ("total_return", "Total Return", "{:.0%}"),
        ("sharpe", "Sharpe", "{:.2f}"),
        ("max_drawdown", "Max Drawdown", "{:.0%}"),
    ]
    for ax, (column, title, fmt) in zip(axes, metric_defs):
        ridge = selected[selected["strategy"] == "Ridge Linear Regression"].set_index("year").reindex(years)[column]
        macd = selected[selected["strategy"] == "MACD 12/26/9"].set_index("year").reindex(years)[column]
        ax.bar(x - width / 2, ridge, width, color="#1464F4", label="Ridge")
        ax.bar(x + width / 2, macd, width, color="#D97706", label="MACD 12/26/9")
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_title(title, weight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        if column in {"total_return", "max_drawdown"}:
            ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
        ax.legend(loc="upper left")
    axes[-1].set_xticks(x, [str(year) for year in years])
    fig.suptitle("Stress Years: Ridge vs MACD 12/26/9", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_drawdown_heatmap(metrics: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "stress_years_risk_heatmap.png"
    table = metrics[metrics["strategy"].isin(["Ridge Linear Regression", "MACD 12/26/9"])].pivot(
        index="strategy", columns="year", values="max_drawdown"
    )
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(13, 3.6))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-0.45, vmax=0.0, aspect="auto")
    ax.set_title("Max Drawdown Heatmap", fontsize=15, weight="bold")
    ax.set_xticks(range(len(table.columns)), table.columns.astype(str))
    ax.set_yticks(range(len(table.index)), table.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1%}", ha="center", va="center", color="#111111", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, format=lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def format_pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def format_num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:,.3f}"


def markdown_table(df: pd.DataFrame, pct_cols: set[str], num_cols: set[str]) -> str:
    out = df.copy()
    for col in out.columns:
        if col in pct_cols:
            out[col] = out[col].map(format_pct)
        elif col in num_cols:
            out[col] = out[col].map(format_num)
    out = out.astype(str)
    headers = [str(col) for col in out.columns]
    rows = out.values.tolist()
    widths = [
        max(len(headers[i]), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def write_readme(stress_years: pd.DataFrame, metrics: pd.DataFrame, images: list[Path]) -> None:
    comparison = add_advantage(metrics)
    pct_cols = {
        "annualized_volatility",
        "buy_hold_return",
        "buy_hold_max_drawdown",
        "worst_daily_return",
        "best_daily_return",
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
    num_cols = {
        "sharpe",
        "sortino",
        "calmar",
        "profit_factor",
        "number_of_trades",
        "beta_vs_buy_hold",
        "information_ratio",
        "tail_ratio_95_5",
        "daily_skew",
        "daily_kurtosis",
    }
    metric_cols = [
        "year",
        "strategy",
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "profit_factor",
        "win_rate_active_days",
        "exposure",
        "number_of_trades",
        "annual_alpha_vs_buy_hold",
    ]
    advantage = comparison[comparison["strategy"] == "Ridge minus MACD"].copy()
    wins = {
        "return": int((advantage["total_return"] > 0).sum()),
        "sharpe": int((advantage["sharpe"] > 0).sum()),
        "drawdown": int((advantage["max_drawdown"] > 0).sum()),
    }
    total_years = len(stress_years)
    best_ridge_year = metrics[metrics["strategy"] == "Ridge Linear Regression"].sort_values("sharpe", ascending=False).iloc[0]
    worst_ridge_year = metrics[metrics["strategy"] == "Ridge Linear Regression"].sort_values("max_drawdown").iloc[0]
    best_macd_year = metrics[metrics["strategy"] == "MACD 12/26/9"].sort_values("sharpe", ascending=False).iloc[0]
    worst_macd_year = metrics[metrics["strategy"] == "MACD 12/26/9"].sort_values("max_drawdown").iloc[0]

    lines = [
        "# Stress Test Ridge Linear Regression vs MACD 12/26/9",
        "",
        "## Phương pháp",
        "",
        f"- Dữ liệu: `data.csv`, đã parse lại OHLCV để xử lý dấu phẩy hàng nghìn.",
        f"- Chọn {TOP_N_YEARS} năm biến động nhất theo annualized volatility của lợi suất ngày VN-Index.",
        f"- Ridge Linear Regression: mỗi năm stress được train lại bằng toàn bộ dữ liệu trước ngày 01/01 của năm đó, sau đó dự báo từng phiên trong năm. Cách này tránh dùng dữ liệu tương lai.",
        f"- MACD 12/26/9: long khi MACD line > signal line, exit về cash khi MACD line <= signal line. Tín hiệu tại close được áp dụng cho lợi suất phiên kế tiếp.",
        f"- Phí giao dịch: {COST_BPS:.1f} bps mỗi lần thay đổi vị thế.",
        "",
        "## Các năm biến động nhất",
        "",
        markdown_table(
            stress_years[
                [
                    "year",
                    "rows",
                    "start",
                    "end",
                    "annualized_volatility",
                    "buy_hold_return",
                    "buy_hold_max_drawdown",
                    "worst_daily_return",
                    "best_daily_return",
                ]
            ],
            pct_cols,
            num_cols,
        ),
        "",
        "## Bảng so sánh kết quả stress test",
        "",
        markdown_table(comparison[metric_cols], pct_cols, num_cols),
        "",
        "## Nhận xét chi tiết",
        "",
        f"- Ridge có total return cao hơn MACD trong {wins['return']}/{total_years} năm stress, Sharpe cao hơn trong {wins['sharpe']}/{total_years} năm, và max drawdown nhỏ hơn trong {wins['drawdown']}/{total_years} năm.",
        f"- Năm Ridge có Sharpe tốt nhất là {int(best_ridge_year['year'])} với Sharpe {best_ridge_year['sharpe']:.2f}, CAGR {best_ridge_year['cagr']:.2%}, max drawdown {best_ridge_year['max_drawdown']:.2%}.",
        f"- Năm Ridge chịu drawdown sâu nhất là {int(worst_ridge_year['year'])}, max drawdown {worst_ridge_year['max_drawdown']:.2%}. Đây là năm cần quan sát kỹ nếu muốn thêm bộ lọc risk-off.",
        f"- Năm MACD có Sharpe tốt nhất là {int(best_macd_year['year'])} với Sharpe {best_macd_year['sharpe']:.2f}, CAGR {best_macd_year['cagr']:.2%}, max drawdown {best_macd_year['max_drawdown']:.2%}.",
        f"- Năm MACD chịu drawdown sâu nhất là {int(worst_macd_year['year'])}, max drawdown {worst_macd_year['max_drawdown']:.2%}. MACD thường chậm hơn khi thị trường đảo chiều nhanh, nên dễ bị kéo drawdown trong các pha whipsaw.",
        "- Ridge phản ứng linh hoạt hơn vì dùng nhiều feature về momentum, volatility, range và volume. Đổi lại, số lần giao dịch thường cao hơn MACD, nên kết quả nhạy cảm hơn với phí và slippage.",
        "- MACD 12/26/9 đơn giản, dễ giải thích và ít turnover hơn. Nếu mục tiêu là hệ thống dễ triển khai thủ công, MACD vẫn có giá trị tham chiếu tốt; nếu mục tiêu là risk-adjusted return trong năm biến động, Ridge đang có lợi thế hơn.",
        "- Khuyến nghị: dùng Ridge như lớp tín hiệu chính, nhưng thêm điều kiện phòng vệ như giới hạn drawdown theo tháng/quý, volatility filter, hoặc yêu cầu MACD không quá xấu để giảm số trade trong giai đoạn nhiễu mạnh.",
        "",
        "## Hình ảnh",
        "",
    ]
    for image in images:
        rel = image.as_posix()
        lines.append(f"![{image.stem}]({rel})")
        lines.append("")
    lines.extend(
        [
            "## File đầu ra",
            "",
            f"- `{OUTPUT_DIR / 'stress_years_selected.csv'}`",
            f"- `{OUTPUT_DIR / 'stress_years_strategy_metrics.csv'}`",
            f"- `{OUTPUT_DIR / 'stress_years_strategy_metrics_with_advantage.csv'}`",
            f"- `{OUTPUT_DIR / 'stress_years_daily_backtests.csv'}`",
        ]
    )
    READ_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stress_years, metrics, ridge_frames, macd_frames = build_stress_test()
    combined_daily = pd.concat(
        [
            pd.concat(ridge_frames.values(), ignore_index=True),
            pd.concat(macd_frames.values(), ignore_index=True),
        ],
        ignore_index=True,
    )
    stress_years.to_csv(OUTPUT_DIR / "stress_years_selected.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "stress_years_strategy_metrics.csv", index=False)
    add_advantage(metrics).to_csv(OUTPUT_DIR / "stress_years_strategy_metrics_with_advantage.csv", index=False)
    combined_daily.to_csv(OUTPUT_DIR / "stress_years_daily_backtests.csv", index=False)
    images = [
        plot_equity_panels(ridge_frames, macd_frames),
        plot_metric_bars(metrics),
        plot_drawdown_heatmap(metrics),
    ]
    write_readme(stress_years, metrics, images)
    print("Stress test completed.")
    print(f"Selected years: {', '.join(map(str, stress_years['year'].astype(int).tolist()))}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"README: {READ_PATH.resolve()}")
    print(add_advantage(metrics)[['year', 'strategy', 'total_return', 'sharpe', 'max_drawdown', 'exposure', 'number_of_trades']].to_string(index=False))


if __name__ == "__main__":
    main()
