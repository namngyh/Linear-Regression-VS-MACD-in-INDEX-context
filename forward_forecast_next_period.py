from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-forward-forecast")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from linear_regression_backtest import load_vnindex_csv, make_features, rsi


DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("outputs_forward_forecast")
HORIZON_DAYS = 30
RIDGE_ALPHA = 500.0


def make_features_including_latest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    data["log_close"] = np.log(data["Close"])
    data["ret_1"] = data["log_close"].diff()
    data["simple_ret_1"] = data["Close"].pct_change()
    data["target_log_ret"] = data["log_close"].shift(-1) - data["log_close"]
    data["target_simple_ret"] = data["Close"].shift(-1) / data["Close"] - 1
    data["TargetDate"] = data["Date"].shift(-1)

    data["hl_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["oc_return"] = data["Close"] / data["Open"] - 1
    data["upper_shadow"] = (data["High"] - data[["Open", "Close"]].max(axis=1)) / data["Close"]
    data["lower_shadow"] = (data[["Open", "Close"]].min(axis=1) - data["Low"]) / data["Close"]
    data["volume_log"] = np.log1p(data["Volume"])
    data["volume_chg_1"] = data["volume_log"].diff()

    for window in [2, 3, 5, 10, 14, 20, 30, 60]:
        data[f"ret_lag_{window}"] = data["ret_1"].shift(window - 1)
        data[f"ret_mean_{window}"] = data["ret_1"].rolling(window).mean()
        data[f"ret_std_{window}"] = data["ret_1"].rolling(window).std()
        data[f"momentum_{window}"] = data["Close"] / data["Close"].shift(window) - 1
        sma = data["Close"].rolling(window).mean()
        data[f"close_sma_gap_{window}"] = data["Close"] / sma - 1
        data[f"range_mean_{window}"] = data["hl_range"].rolling(window).mean()
        data[f"volume_z_{window}"] = (data["volume_log"] - data["volume_log"].rolling(window).mean()) / data[
            "volume_log"
        ].rolling(window).std()

    data["rsi_14"] = rsi(data["Close"], 14)
    data["rsi_28"] = rsi(data["Close"], 28)
    data["ema_gap_12"] = data["Close"] / data["Close"].ewm(span=12, adjust=False).mean() - 1
    data["ema_gap_26"] = data["Close"] / data["Close"].ewm(span=26, adjust=False).mean() - 1
    data["macd_gap"] = data["Close"].ewm(span=12, adjust=False).mean() / data["Close"].ewm(
        span=26, adjust=False
    ).mean() - 1
    data["rolling_drawdown_60"] = data["Close"] / data["Close"].rolling(60).max() - 1

    non_features = {
        "Date",
        "TargetDate",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SourceLine",
        "log_close",
        "target_log_ret",
        "target_simple_ret",
    }
    feature_cols = [column for column in data.columns if column not in non_features]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).reset_index(drop=True)
    return data, feature_cols


def macd_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["target_log_ret"] = np.log(frame["Close"].shift(-1) / frame["Close"])
    frame["target_simple_ret"] = frame["Close"].shift(-1) / frame["Close"] - 1
    ema12 = frame["Close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["Close"].ewm(span=26, adjust=False).mean()
    frame["macd_line"] = ema12 - ema26
    frame["macd_signal"] = frame["macd_line"].ewm(span=9, adjust=False).mean()
    frame["macd_histogram"] = frame["macd_line"] - frame["macd_signal"]
    frame["macd_forecast_up"] = frame["macd_line"] > frame["macd_signal"]
    return frame


def next_business_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)


def build_forward_forecast() -> tuple[pd.DataFrame, dict[str, float | str], pd.DataFrame]:
    raw = load_vnindex_csv(DATA_PATH)
    train, feature_cols = make_features(raw)
    full_features, _ = make_features_including_latest(raw)
    latest = full_features.iloc[-1]
    last_close = float(latest["Close"])
    last_date = pd.Timestamp(latest["Date"])

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA, max_iter=20000)),
        ]
    )
    ridge.fit(train[feature_cols], train["target_log_ret"])
    latest_ridge_log_ret = float(ridge.predict(latest[feature_cols].to_frame().T)[0])
    train_pred = ridge.predict(train[feature_cols])
    ridge_rmse = float(np.sqrt(np.mean((train["target_log_ret"].to_numpy() - train_pred) ** 2)))

    macd = macd_frame(raw)
    latest_macd = macd.iloc[-1]
    current_macd_up = bool(latest_macd["macd_forecast_up"])
    macd_history = macd.dropna(subset=["target_log_ret", "macd_forecast_up"]).copy()
    conditional_log_return = float(
        macd_history.loc[macd_history["macd_forecast_up"] == current_macd_up, "target_log_ret"].mean()
    )
    macd_all_std = float(macd_history["target_log_ret"].std(ddof=0))

    dates = next_business_dates(last_date, HORIZON_DAYS)
    steps = np.arange(1, HORIZON_DAYS + 1)
    ridge_close = last_close * np.exp(latest_ridge_log_ret * steps)
    macd_close = last_close * np.exp(conditional_log_return * steps)

    forecast = pd.DataFrame(
        {
            "ForecastDate": dates,
            "RidgeForecastClose": ridge_close,
            "MACDForecastClose": macd_close,
            "RidgeDailyLogReturnAssumption": latest_ridge_log_ret,
            "MACDDailyLogReturnAssumption": conditional_log_return,
            "RidgeForecastDirection": "up" if latest_ridge_log_ret > 0 else "down",
            "MACDForecastDirection": "up" if current_macd_up else "down/cash",
        }
    )
    summary = {
        "last_data_date": str(last_date.date()),
        "last_close": last_close,
        "horizon_days": HORIZON_DAYS,
        "ridge_daily_log_return": latest_ridge_log_ret,
        "ridge_daily_simple_return": float(np.exp(latest_ridge_log_ret) - 1),
        "ridge_30d_forecast_close": float(ridge_close[-1]),
        "ridge_30d_forecast_return": float(ridge_close[-1] / last_close - 1),
        "ridge_train_rmse_log_return": ridge_rmse,
        "macd_current_regime": "up" if current_macd_up else "down/cash",
        "macd_daily_log_return_assumption": conditional_log_return,
        "macd_daily_simple_return_assumption": float(np.exp(conditional_log_return) - 1),
        "macd_30d_forecast_close": float(macd_close[-1]),
        "macd_30d_forecast_return": float(macd_close[-1] / last_close - 1),
        "macd_historical_log_return_std": macd_all_std,
    }
    return forecast, summary, raw


def plot_forward_lines(forecast: pd.DataFrame, summary: dict[str, float | str], raw: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "vnindex_next_period_forecast_lines.png"
    history = raw.tail(252).copy()
    last_date = pd.to_datetime(summary["last_data_date"])
    last_close = float(summary["last_close"])
    anchor = pd.DataFrame(
        {
            "ForecastDate": [last_date],
            "RidgeForecastClose": [last_close],
            "MACDForecastClose": [last_close],
        }
    )
    plot_forecast = pd.concat([anchor, forecast], ignore_index=True)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(history["Date"], history["Close"], color="#555555", linewidth=2.2, label="Actual VN-Index close")
    ax.plot(
        plot_forecast["ForecastDate"],
        plot_forecast["RidgeForecastClose"],
        color="#1464F4",
        linewidth=2.8,
        marker="o",
        markersize=3.5,
        label="Ridge projection",
    )
    ax.plot(
        plot_forecast["ForecastDate"],
        plot_forecast["MACDForecastClose"],
        color="#D97706",
        linewidth=2.8,
        marker="o",
        markersize=3.5,
        label="MACD regime projection",
    )
    ax.axvline(last_date, color="#222222", linestyle="--", linewidth=1.1)
    ax.annotate(
        "forecast starts",
        xy=(last_date, last_close),
        xytext=(10, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#222222"},
    )
    ax.set_title("VN-Index Next-Period Forecast: Ridge vs MACD 12/26/9", fontsize=17, weight="bold")
    ax.set_ylabel("VN-Index close")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_return_assumptions(summary: dict[str, float | str]) -> Path:
    path = OUTPUT_DIR / "next_period_return_assumptions.png"
    labels = ["Ridge", "MACD 12/26/9"]
    daily = [
        float(summary["ridge_daily_simple_return"]),
        float(summary["macd_daily_simple_return_assumption"]),
    ]
    horizon = [
        float(summary["ridge_30d_forecast_return"]),
        float(summary["macd_30d_forecast_return"]),
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, daily, width, color="#1464F4", label="Daily assumption")
    ax.bar(x + width / 2, horizon, width, color="#D97706", label=f"{HORIZON_DAYS}-session projection")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.1%}")
    ax.set_title("Forward Return Assumptions", fontsize=16, weight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_summary(forecast: pd.DataFrame, summary: dict[str, float | str], images: list[Path]) -> None:
    lines = [
        "# VN-Index Next-Period Forecast",
        "",
        f"- Last data date: `{summary['last_data_date']}`",
        f"- Last close: `{float(summary['last_close']):,.2f}`",
        f"- Horizon: `{int(summary['horizon_days'])}` business sessions",
        f"- Ridge daily return assumption: `{float(summary['ridge_daily_simple_return']):.3%}`",
        f"- Ridge projected close at horizon: `{float(summary['ridge_30d_forecast_close']):,.2f}` ({float(summary['ridge_30d_forecast_return']):.2%})",
        f"- MACD current regime: `{summary['macd_current_regime']}`",
        f"- MACD daily return assumption: `{float(summary['macd_daily_simple_return_assumption']):.3%}`",
        f"- MACD projected close at horizon: `{float(summary['macd_30d_forecast_close']):,.2f}` ({float(summary['macd_30d_forecast_return']):.2%})",
        "",
        "## Method Note",
        "",
        "Ridge gives a numeric one-step next-day return forecast. For a 30-session path, the latest one-step return is compounded forward because future OHLCV features are unknown.",
        "MACD is a directional regime rule, not a numeric price model. Its path is produced by compounding the historical average next-day return observed under the current MACD regime.",
        "",
        "## Images",
        "",
    ]
    lines.extend([f"- `{image.name}`" for image in images])
    (OUTPUT_DIR / "next_period_forecast_summary.md").write_text("\n".join(lines), encoding="utf-8")
    forecast.to_csv(OUTPUT_DIR / "vnindex_next_period_forecast.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "vnindex_next_period_forecast_summary.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecast, summary, raw = build_forward_forecast()
    images = [
        plot_forward_lines(forecast, summary, raw),
        plot_return_assumptions(summary),
    ]
    write_summary(forecast, summary, images)
    print("Forward forecast completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Last data date: {summary['last_data_date']}; last close: {float(summary['last_close']):,.2f}")
    print(
        "Ridge: "
        f"daily={float(summary['ridge_daily_simple_return']):.3%}, "
        f"{HORIZON_DAYS}d close={float(summary['ridge_30d_forecast_close']):,.2f}, "
        f"{HORIZON_DAYS}d return={float(summary['ridge_30d_forecast_return']):.2%}"
    )
    print(
        "MACD: "
        f"regime={summary['macd_current_regime']}, "
        f"daily={float(summary['macd_daily_simple_return_assumption']):.3%}, "
        f"{HORIZON_DAYS}d close={float(summary['macd_30d_forecast_close']):,.2f}, "
        f"{HORIZON_DAYS}d return={float(summary['macd_30d_forecast_return']):.2%}"
    )


if __name__ == "__main__":
    main()
