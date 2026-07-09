from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-forecast-skill")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RIDGE_PATH = Path("outputs_linear_regression_backtest/best_model_test_predictions.csv")
MACD_PATH = Path("outputs_model_vs_macd12269/macd12269_test_backtest.csv")
OUTPUT_DIR = Path("outputs_forecast_skill_comparison")


def load_aligned_forecasts() -> pd.DataFrame:
    ridge = pd.read_csv(RIDGE_PATH, parse_dates=["Date", "TargetDate"])
    macd = pd.read_csv(MACD_PATH, parse_dates=["Date", "TargetDate"])
    cols = ["TargetDate", "macd_line", "macd_signal", "macd_histogram", "position"]
    merged = ridge.merge(macd[cols], on="TargetDate", how="inner", suffixes=("_ridge", "_macd"))
    merged = merged.rename(columns={"position_ridge": "ridge_position", "position_macd": "macd_position"})
    merged["actual_up"] = merged["target_log_ret"] > 0
    merged["ridge_forecast_up"] = merged["pred_log_ret"] > 0
    merged["macd_forecast_up"] = merged["macd_position"] > 0
    merged["macd_score"] = merged["macd_histogram"] / merged["Close"]
    return merged.sort_values("TargetDate").reset_index(drop=True)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def directional_metrics(frame: pd.DataFrame, pred_col: str, score_col: str | None = None) -> dict[str, float]:
    actual = frame["actual_up"].astype(bool)
    pred = frame[pred_col].astype(bool)
    tp = int((actual & pred).sum())
    tn = int((~actual & ~pred).sum())
    fp = int((~actual & pred).sum())
    fn = int((actual & ~pred).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall) if not np.isnan(precision + recall) else np.nan
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(tp * tn - fp * fn, mcc_den)
    mean_up = frame.loc[pred, "target_simple_ret"].mean()
    mean_down = frame.loc[~pred, "target_simple_ret"].mean()
    score_corr = np.nan
    if score_col is not None and frame[score_col].nunique(dropna=True) > 1:
        score_corr = float(np.corrcoef(frame[score_col], frame["target_log_ret"])[0, 1])
    return {
        "accuracy": float((actual == pred).mean()),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "precision_up": precision,
        "recall_up": recall,
        "specificity_down": specificity,
        "f1_up": f1,
        "mcc": mcc,
        "hit_rate_on_predicted_up": float(frame.loc[pred, "actual_up"].mean()) if pred.any() else np.nan,
        "hit_rate_on_predicted_down": float((~frame.loc[~pred, "actual_up"]).mean()) if (~pred).any() else np.nan,
        "mean_next_return_when_pred_up": float(mean_up),
        "mean_next_return_when_pred_down": float(mean_down),
        "return_spread_up_minus_down": float(mean_up - mean_down),
        "forecast_up_frequency": float(pred.mean()),
        "score_corr_with_next_return": score_corr,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def build_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "model": "Ridge Linear Regression",
            "rmse_log_return": float(np.sqrt(((frame["target_log_ret"] - frame["pred_log_ret"]) ** 2).mean())),
            "mae_log_return": float((frame["target_log_ret"] - frame["pred_log_ret"]).abs().mean()),
            **directional_metrics(frame, "ridge_forecast_up", "pred_log_ret"),
        },
        {
            "model": "MACD 12/26/9",
            "rmse_log_return": np.nan,
            "mae_log_return": np.nan,
            **directional_metrics(frame, "macd_forecast_up", "macd_score"),
        },
    ]
    metrics = pd.DataFrame(rows)
    ridge = metrics[metrics["model"] == "Ridge Linear Regression"].iloc[0]
    macd = metrics[metrics["model"] == "MACD 12/26/9"].iloc[0]
    diff = {"model": "Ridge minus MACD"}
    for col in metrics.columns:
        if col != "model":
            diff[col] = ridge[col] - macd[col]
    return pd.concat([metrics, pd.DataFrame([diff])], ignore_index=True)


def plot_metric_bars(metrics: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "forecast_skill_metric_bars.png"
    base = metrics[metrics["model"].isin(["Ridge Linear Regression", "MACD 12/26/9"])].set_index("model")
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_up",
        "recall_up",
        "specificity_down",
        "f1_up",
        "mcc",
        "score_corr_with_next_return",
    ]
    labels = [
        "Accuracy",
        "Balanced accuracy",
        "Precision up",
        "Recall up",
        "Specificity down",
        "F1 up",
        "MCC",
        "Score corr.",
    ]
    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(15, 7))
    ridge_vals = base.loc["Ridge Linear Regression", metric_names].astype(float).values
    macd_vals = base.loc["MACD 12/26/9", metric_names].astype(float).values
    ax.bar(x - width / 2, ridge_vals, width, color="#1464F4", label="Ridge Linear Regression")
    ax.bar(x + width / 2, macd_vals, width, color="#D97706", label="MACD 12/26/9")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9, alpha=0.65)
    ax.set_title("Forecast Skill Metrics: Ridge vs MACD 12/26/9", fontsize=17, weight="bold")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(min(-0.15, np.nanmin([ridge_vals, macd_vals]) - 0.08), max(1.05, np.nanmax([ridge_vals, macd_vals]) + 0.08))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_confusion_matrices(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "forecast_confusion_matrices.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, title, pred_col, color in [
        (axes[0], "Ridge Linear Regression", "ridge_forecast_up", "Blues"),
        (axes[1], "MACD 12/26/9", "macd_forecast_up", "Oranges"),
    ]:
        actual = frame["actual_up"].astype(bool)
        pred = frame[pred_col].astype(bool)
        mat = np.array(
            [
                [(~actual & ~pred).sum(), (~actual & pred).sum()],
                [(actual & ~pred).sum(), (actual & pred).sum()],
            ],
            dtype=float,
        )
        im = ax.imshow(mat, cmap=color)
        ax.set_title(title, fontsize=15, weight="bold")
        ax.set_xticks([0, 1], ["Forecast down/cash", "Forecast up"])
        ax.set_yticks([0, 1], ["Actual down", "Actual up"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=18, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Directional Forecast Confusion Matrices", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rolling_accuracy(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "forecast_rolling_directional_accuracy.png"
    window = 63
    ridge_hit = (frame["ridge_forecast_up"] == frame["actual_up"]).rolling(window).mean()
    macd_hit = (frame["macd_forecast_up"] == frame["actual_up"]).rolling(window).mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame["TargetDate"], ridge_hit, color="#1464F4", linewidth=2.2, label="Ridge Linear Regression")
    ax.plot(frame["TargetDate"], macd_hit, color="#D97706", linewidth=2.2, label="MACD 12/26/9")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, label="Random baseline")
    ax.set_title("63-Day Rolling Directional Accuracy", fontsize=16, weight="bold")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_score_scatter(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "forecast_score_vs_next_return.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(frame["pred_log_ret"] * 100, frame["target_log_ret"] * 100, s=18, alpha=0.6, color="#1464F4")
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].axvline(0, color="#777777", linewidth=0.8)
    axes[0].set_title("Ridge Forecast Score", weight="bold")
    axes[0].set_xlabel("Predicted next-day log return (%)")
    axes[0].set_ylabel("Actual next-day log return (%)")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(frame["macd_score"] * 100, frame["target_log_ret"] * 100, s=18, alpha=0.6, color="#D97706")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].axvline(0, color="#777777", linewidth=0.8)
    axes[1].set_title("MACD Histogram / Close Score", weight="bold")
    axes[1].set_xlabel("MACD normalized score (%)")
    axes[1].set_ylabel("Actual next-day log return (%)")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Forecast Score vs Realized Next-Day Return", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_conditional_returns(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "forecast_conditional_next_return.png"
    groups = [
        frame.loc[frame["ridge_forecast_up"], "target_simple_ret"] * 100,
        frame.loc[~frame["ridge_forecast_up"], "target_simple_ret"] * 100,
        frame.loc[frame["macd_forecast_up"], "target_simple_ret"] * 100,
        frame.loc[~frame["macd_forecast_up"], "target_simple_ret"] * 100,
    ]
    labels = ["Ridge up", "Ridge down/cash", "MACD up", "MACD down/cash"]
    colors = ["#1464F4", "#8DB5FF", "#D97706", "#F5B76B"]
    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    means = [g.mean() for g in groups]
    ax.scatter(np.arange(1, len(groups) + 1), means, color="#111111", zorder=3, label="Mean")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_title("Actual Next-Day Return Conditional on Forecast Signal", fontsize=16, weight="bold")
    ax.set_ylabel("Actual next-day return (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def format_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def format_num(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.4f}"


def write_summary(metrics: pd.DataFrame, frame: pd.DataFrame, image_paths: list[Path]) -> None:
    display = metrics.copy()
    pct_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision_up",
        "recall_up",
        "specificity_down",
        "f1_up",
        "hit_rate_on_predicted_up",
        "hit_rate_on_predicted_down",
        "mean_next_return_when_pred_up",
        "mean_next_return_when_pred_down",
        "return_spread_up_minus_down",
        "forecast_up_frequency",
    ]
    for col in display.columns:
        if col == "model":
            continue
        if col in pct_cols:
            display[col] = display[col].map(format_pct)
        else:
            display[col] = display[col].map(format_num)
    display.to_csv(OUTPUT_DIR / "forecast_skill_metrics_formatted.csv", index=False)

    lines = [
        "# Forecast Skill Comparison",
        "",
        f"- Window: {frame['TargetDate'].min().date()} to {frame['TargetDate'].max().date()}",
        "- Ridge forecast: predicted next-day log return > 0 means forecast up.",
        "- MACD forecast: MACD line > signal line means forecast up; otherwise down/cash.",
        "",
        "## Metrics",
        "",
        display[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision_up",
                "recall_up",
                "specificity_down",
                "f1_up",
                "mcc",
                "score_corr_with_next_return",
                "forecast_up_frequency",
                "return_spread_up_minus_down",
            ]
        ].to_string(index=False),
        "",
        "## Images",
        "",
    ]
    lines.extend([f"- `{p.name}`" for p in image_paths])
    (OUTPUT_DIR / "forecast_skill_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_aligned_forecasts()
    frame.to_csv(OUTPUT_DIR / "forecast_skill_daily.csv", index=False)
    metrics = build_metrics(frame)
    metrics.to_csv(OUTPUT_DIR / "forecast_skill_metrics.csv", index=False)
    image_paths = [
        plot_metric_bars(metrics),
        plot_confusion_matrices(frame),
        plot_rolling_accuracy(frame),
        plot_score_scatter(frame),
        plot_conditional_returns(frame),
    ]
    write_summary(metrics, frame, image_paths)
    print("Forecast skill comparison completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(metrics[["model", "accuracy", "balanced_accuracy", "precision_up", "recall_up", "specificity_down", "f1_up", "mcc", "score_corr_with_next_return"]].to_string(index=False))


if __name__ == "__main__":
    main()
