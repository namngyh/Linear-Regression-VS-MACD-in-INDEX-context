from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lr-vnindex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


TRADING_DAYS = 252
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class ModelResult:
    name: str
    params: dict
    valid_rmse: float
    estimator: Pipeline
    predictions: dict[str, pd.Series]
    prediction_metrics: dict[str, float]
    backtest_metrics: dict[str, float]
    backtest_frame: pd.DataFrame


def parse_price_tokens(tokens: list[str], start: int, n_prices: int = 4) -> tuple[list[float], int]:
    prices: list[float] = []
    i = start
    while i < len(tokens) and len(prices) < n_prices:
        token = tokens[i].strip()
        if not token:
            i += 1
            continue
        if i + 1 < len(tokens):
            nxt = tokens[i + 1].strip()
            if "." not in token and token.isdigit() and int(token) < 10 and nxt:
                if "." in nxt:
                    next_integer, next_decimal = nxt.split(".", 1)
                    padded_next = f"{next_integer.zfill(3)}.{next_decimal}"
                else:
                    padded_next = nxt.zfill(3)
                prices.append(float(f"{token}{padded_next}"))
                i += 2
                continue
        prices.append(float(token))
        i += 1
    if len(prices) != n_prices:
        raise ValueError(f"Expected {n_prices} OHLC prices, parsed {len(prices)} from tokens={tokens}")
    return prices, i


def parse_volume_tokens(tokens: Iterable[str]) -> float:
    pieces: list[str] = []
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        if "." in token:
            number = float(token)
            if not math.isnan(number):
                pieces.append(str(int(number)))
        elif token.isdigit():
            pieces.append(token)
    if not pieces:
        return np.nan
    return float("".join(pieces))


def load_vnindex_csv(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or header[0].lower() != "date":
            raise ValueError("CSV must start with a Date column.")
        for line_no, tokens in enumerate(reader, start=2):
            if not tokens or not tokens[0].strip():
                continue
            date = tokens[0].strip()
            prices, idx = parse_price_tokens(tokens, 1, 4)
            rows.append(
                {
                    "Date": date,
                    "Open": prices[0],
                    "High": prices[1],
                    "Low": prices[2],
                    "Close": prices[3],
                    "Volume": parse_volume_tokens(tokens[idx:]),
                    "SourceLine": line_no,
                }
            )

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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

    windows = [2, 3, 5, 10, 14, 20, 30, 60]
    for window in windows:
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
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["target_log_ret", "target_simple_ret"])
    return data.reset_index(drop=True), feature_cols


def split_chronological(data: pd.DataFrame, train_size: float = 0.70, valid_size: float = 0.15) -> dict[str, pd.DataFrame]:
    n = len(data)
    train_end = int(n * train_size)
    valid_end = int(n * (train_size + valid_size))
    return {
        "train": data.iloc[:train_end].copy(),
        "valid": data.iloc[train_end:valid_end].copy(),
        "test": data.iloc[valid_end:].copy(),
    }


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def prediction_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    actual_up = y_true > 0
    pred_up = y_pred > 0
    tp = int((actual_up & pred_up).sum())
    fp = int((~actual_up & pred_up).sum())
    fn = int((actual_up & ~pred_up).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not np.isnan(precision + recall) else np.nan
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "corr": float(corr),
        "direction_accuracy": float((np.sign(y_true) == np.sign(y_pred)).mean()),
        "up_precision": float(precision),
        "up_recall": float(recall),
        "up_f1": float(f1),
    }


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min()), drawdown


def financial_metrics(
    returns: pd.Series,
    benchmark: pd.Series,
    positions: pd.Series,
    dates: pd.Series,
    turnover: pd.Series,
) -> dict[str, float]:
    returns = returns.fillna(0.0)
    benchmark = benchmark.fillna(0.0)
    equity = (1 + returns).cumprod()
    bench_equity = (1 + benchmark).cumprod()
    years = max((pd.to_datetime(dates.iloc[-1]) - pd.to_datetime(dates.iloc[0])).days / 365.25, len(returns) / TRADING_DAYS)
    total_return = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else np.nan
    annual_vol = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = float((returns.mean() * TRADING_DAYS) / annual_vol) if annual_vol else np.nan
    sortino = float((returns.mean() * TRADING_DAYS) / downside) if downside else np.nan
    mdd, _ = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 and not np.isnan(cagr) else np.nan
    var_95 = float(returns.quantile(0.05))
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else np.nan
    positive = returns[returns > 0].sum()
    negative = returns[returns < 0].sum()
    profit_factor = float(positive / abs(negative)) if negative < 0 else np.nan
    active = returns - benchmark
    info_ratio = float((active.mean() * TRADING_DAYS) / (active.std(ddof=0) * np.sqrt(TRADING_DAYS))) if active.std(ddof=0) else np.nan
    beta = float(np.cov(returns, benchmark)[0, 1] / np.var(benchmark)) if np.var(benchmark) else np.nan
    alpha = float((returns.mean() - beta * benchmark.mean()) * TRADING_DAYS) if not np.isnan(beta) else np.nan
    nonzero = returns[returns != 0]
    win_rate = float((nonzero > 0).mean()) if len(nonzero) else np.nan
    tail_ratio = float(abs(returns.quantile(0.95) / returns.quantile(0.05))) if returns.quantile(0.05) else np.nan
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "daily_var_95": var_95,
        "daily_cvar_95": cvar_95,
        "profit_factor": profit_factor,
        "win_rate_active_days": win_rate,
        "exposure": float(positions.mean()),
        "avg_daily_turnover": float(turnover.mean()),
        "number_of_trades": float((turnover > 0).sum()),
        "beta_vs_buy_hold": beta,
        "annual_alpha_vs_buy_hold": alpha,
        "information_ratio": info_ratio,
        "tail_ratio_95_5": tail_ratio,
        "daily_skew": float(returns.skew()),
        "daily_kurtosis": float(returns.kurtosis()),
        "benchmark_total_return": float(bench_equity.iloc[-1] - 1),
        "benchmark_cagr": float(bench_equity.iloc[-1] ** (1 / years) - 1) if years > 0 and bench_equity.iloc[-1] > 0 else np.nan,
    }


def build_backtest_frame(
    frame: pd.DataFrame,
    pred_log_ret: pd.Series,
    cost_bps: float = 5.0,
    signal_threshold: float | None = None,
) -> pd.DataFrame:
    threshold = cost_bps / 10000 if signal_threshold is None else signal_threshold
    out = frame[["Date", "TargetDate", "Close", "target_log_ret", "target_simple_ret"]].copy()
    out["pred_log_ret"] = np.asarray(pred_log_ret)
    out["position"] = (out["pred_log_ret"] > threshold).astype(float)
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["transaction_cost"] = out["turnover"] * (cost_bps / 10000)
    out["strategy_return"] = out["position"] * out["target_simple_ret"] - out["transaction_cost"]
    out["buy_hold_return"] = out["target_simple_ret"]
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = (1 + out["buy_hold_return"]).cumprod()
    _, out["strategy_drawdown"] = max_drawdown(out["strategy_equity"])
    _, out["buy_hold_drawdown"] = max_drawdown(out["buy_hold_equity"])
    return out


def iter_param_grid(grid: dict[str, list]) -> Iterable[dict]:
    if not grid:
        yield {}
        return
    keys = list(grid)
    for values in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def model_space() -> dict[str, tuple[object, dict[str, list]]]:
    return {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(max_iter=20000), {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]}),
        "Lasso": (Lasso(max_iter=40000), {"alpha": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]}),
        "ElasticNet": (
            ElasticNet(max_iter=40000),
            {"alpha": [0.00001, 0.00005, 0.0001, 0.0005, 0.001], "l1_ratio": [0.15, 0.35, 0.50, 0.75]},
        ),
        "Huber": (HuberRegressor(max_iter=1000), {"epsilon": [1.15, 1.35, 1.50, 1.75], "alpha": [0.0001, 0.001, 0.01]}),
        "BayesianRidge": (BayesianRidge(), {}),
    }


def fit_and_select(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    cost_bps: float,
) -> list[ModelResult]:
    x_train = splits["train"][feature_cols]
    y_train = splits["train"]["target_log_ret"]
    x_valid = splits["valid"][feature_cols]
    y_valid = splits["valid"]["target_log_ret"]
    x_train_valid = pd.concat([x_train, x_valid])
    y_train_valid = pd.concat([y_train, y_valid])
    results: list[ModelResult] = []

    for name, (model, grid) in model_space().items():
        best_score = np.inf
        best_params: dict = {}
        for params in iter_param_grid(grid):
            candidate = clone(model).set_params(**params)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", candidate)])
            pipe.fit(x_train, y_train)
            valid_pred = pipe.predict(x_valid)
            score = rmse(y_valid, valid_pred)
            if score < best_score:
                best_score = score
                best_params = params

        final_pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(model).set_params(**best_params))])
        final_pipe.fit(x_train_valid, y_train_valid)
        predictions = {
            split_name: pd.Series(final_pipe.predict(split_frame[feature_cols]), index=split_frame.index)
            for split_name, split_frame in splits.items()
        }
        test_pred_metrics = prediction_metrics(splits["test"]["target_log_ret"], predictions["test"])
        backtest = build_backtest_frame(splits["test"], predictions["test"], cost_bps=cost_bps)
        bt_metrics = financial_metrics(
            backtest["strategy_return"],
            backtest["buy_hold_return"],
            backtest["position"],
            backtest["TargetDate"],
            backtest["turnover"],
        )
        results.append(
            ModelResult(
                name=name,
                params=best_params,
                valid_rmse=best_score,
                estimator=final_pipe,
                predictions=predictions,
                prediction_metrics=test_pred_metrics,
                backtest_metrics=bt_metrics,
                backtest_frame=backtest,
            )
        )
    return results


def format_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2%}"


def format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:,.4f}"


def save_tables(
    results: list[ModelResult],
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(frame),
                "start_signal_date": frame["Date"].min().date(),
                "end_signal_date": frame["Date"].max().date(),
                "start_target_date": frame["TargetDate"].min().date(),
                "end_target_date": frame["TargetDate"].max().date(),
                "mean_next_day_return": frame["target_simple_ret"].mean(),
                "annualized_vol_next_day_return": frame["target_simple_ret"].std() * np.sqrt(TRADING_DAYS),
            }
            for name, frame in splits.items()
        ]
    )

    model_rows = []
    backtest_rows = []
    for result in results:
        model_rows.append(
            {
                "model": result.name,
                "selected_params": json.dumps(result.params),
                "validation_rmse": result.valid_rmse,
                **result.prediction_metrics,
            }
        )
        backtest_rows.append(
            {
                "model": result.name,
                "selected_params": json.dumps(result.params),
                "cost_bps": cost_bps,
                **result.backtest_metrics,
            }
        )
    model_table = pd.DataFrame(model_rows).sort_values(["direction_accuracy", "rmse"], ascending=[False, True])
    backtest_table = pd.DataFrame(backtest_rows).sort_values(["sharpe", "cagr"], ascending=[False, False])

    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    model_table.to_csv(output_dir / "model_prediction_metrics.csv", index=False)
    backtest_table.to_csv(output_dir / "strategy_backtest_metrics.csv", index=False)
    return split_summary, model_table, backtest_table


def plot_equity(best: ModelResult, output_dir: Path) -> Path:
    bt = best.backtest_frame
    path = output_dir / "equity_curve_best_model.png"
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(bt["TargetDate"], bt["strategy_equity"], label=f"{best.name} strategy", linewidth=2.4, color="#1464F4")
    ax.plot(bt["TargetDate"], bt["buy_hold_equity"], label="Buy & hold VN-Index", linewidth=2.0, color="#555555")
    ax.set_title("Out-of-Sample Equity Curve", fontsize=16, weight="bold")
    ax.set_ylabel("Growth of 1 VND")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_drawdown(best: ModelResult, output_dir: Path) -> Path:
    bt = best.backtest_frame
    path = output_dir / "drawdown_best_model.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(bt["TargetDate"], bt["strategy_drawdown"], 0, alpha=0.45, color="#D94A3A", label=f"{best.name} drawdown")
    ax.plot(bt["TargetDate"], bt["buy_hold_drawdown"], color="#555555", linewidth=1.8, label="Buy & hold drawdown")
    ax.set_title("Drawdown Comparison", fontsize=16, weight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_prediction_scatter(best: ModelResult, splits: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    y = splits["test"]["target_log_ret"]
    pred = best.predictions["test"]
    path = output_dir / "prediction_scatter_best_model.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = np.where((y > 0) == (pred > 0), "#1464F4", "#D94A3A")
    ax.scatter(y * 100, pred * 100, s=18, alpha=0.65, c=colors, edgecolors="none")
    lim = max(abs(y).max(), abs(pred).max()) * 100 * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#222222", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("Predicted vs Actual Next-Day Log Return", fontsize=15, weight="bold")
    ax.set_xlabel("Actual return (%)")
    ax.set_ylabel("Predicted return (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rolling_metrics(best: ModelResult, output_dir: Path) -> Path:
    bt = best.backtest_frame.copy()
    path = output_dir / "rolling_performance_best_model.png"
    rolling_window = 63
    strat_sharpe = bt["strategy_return"].rolling(rolling_window).mean() / bt["strategy_return"].rolling(rolling_window).std() * np.sqrt(TRADING_DAYS)
    bench_sharpe = bt["buy_hold_return"].rolling(rolling_window).mean() / bt["buy_hold_return"].rolling(rolling_window).std() * np.sqrt(TRADING_DAYS)
    rolling_hit = ((bt["pred_log_ret"] > 0) == (bt["target_log_ret"] > 0)).rolling(rolling_window).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(bt["TargetDate"], strat_sharpe, color="#1464F4", label=f"{best.name} rolling Sharpe")
    axes[0].plot(bt["TargetDate"], bench_sharpe, color="#555555", label="Buy & hold rolling Sharpe")
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_title("63-Day Rolling Risk-Adjusted Performance", fontsize=15, weight="bold")
    axes[0].set_ylabel("Sharpe")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[1].plot(bt["TargetDate"], rolling_hit, color="#159447", linewidth=2.0)
    axes[1].axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Directional hit rate")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    axes[1].grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_monthly_heatmap(best: ModelResult, output_dir: Path) -> Path:
    bt = best.backtest_frame.copy()
    path = output_dir / "monthly_return_heatmap_best_model.png"
    monthly = (1 + bt.set_index("TargetDate")["strategy_return"]).resample("ME").prod() - 1
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    heat = table.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * len(heat) + 2)))
    values = heat.to_numpy()
    vmax = np.nanpercentile(abs(values), 90) if np.isfinite(values).any() else 0.1
    image = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Best Model Strategy Monthly Returns", fontsize=16, weight="bold")
    ax.set_xticks(range(12), MONTH_LABELS)
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


def plot_coefficients(best: ModelResult, feature_cols: list[str], output_dir: Path) -> Path | None:
    model = best.estimator.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None
    coefs = np.asarray(model.coef_).ravel()
    coef_table = pd.DataFrame({"feature": feature_cols, "coefficient": coefs})
    coef_table["abs_coefficient"] = coef_table["coefficient"].abs()
    coef_table = coef_table.sort_values("abs_coefficient", ascending=False).head(20).sort_values("coefficient")
    path = output_dir / "top_coefficients_best_model.png"
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = np.where(coef_table["coefficient"] >= 0, "#1464F4", "#D94A3A")
    ax.barh(coef_table["feature"], coef_table["coefficient"], color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_title("Top Standardized Linear Coefficients", fontsize=16, weight="bold")
    ax.set_xlabel("Coefficient on scaled features")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_dashboard(best: ModelResult, output_dir: Path) -> Path:
    bt = best.backtest_frame.copy()
    path = output_dir / "dashboard_best_model.png"
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(bt["TargetDate"], bt["strategy_equity"], label=f"{best.name} strategy", color="#1464F4", linewidth=2.3)
    ax1.plot(bt["TargetDate"], bt["buy_hold_equity"], label="Buy & hold", color="#555555", linewidth=1.9)
    ax1.set_title("Equity Curve", weight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.fill_between(bt["TargetDate"], bt["strategy_drawdown"], 0, color="#D94A3A", alpha=0.45)
    ax2.plot(bt["TargetDate"], bt["buy_hold_drawdown"], color="#555555", linewidth=1.5)
    ax2.set_title("Drawdown", weight="bold")
    ax2.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax2.grid(True, alpha=0.25)

    ax3.scatter(bt["target_log_ret"] * 100, bt["pred_log_ret"] * 100, s=16, alpha=0.6, color="#1464F4")
    ax3.axhline(0, color="#777777", linewidth=0.8)
    ax3.axvline(0, color="#777777", linewidth=0.8)
    ax3.set_title("Forecast vs Actual", weight="bold")
    ax3.set_xlabel("Actual next-day log return (%)")
    ax3.set_ylabel("Predicted next-day log return (%)")
    ax3.grid(True, alpha=0.25)

    rolling_hit = ((bt["pred_log_ret"] > 0) == (bt["target_log_ret"] > 0)).rolling(63).mean()
    ax4.plot(bt["TargetDate"], rolling_hit, color="#159447", linewidth=2.0)
    ax4.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
    ax4.set_title("63-Day Directional Accuracy", weight="bold")
    ax4.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax4.grid(True, alpha=0.25)

    fig.suptitle(f"VN-Index Linear Regression Backtest Dashboard - {best.name}", fontsize=19, weight="bold")
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def percent_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(format_pct)
    return out


def write_report(
    output_dir: Path,
    split_summary: pd.DataFrame,
    model_table: pd.DataFrame,
    backtest_table: pd.DataFrame,
    best: ModelResult,
    image_paths: list[Path],
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> None:
    pct_model_cols = ["r2", "corr", "direction_accuracy", "up_precision", "up_recall", "up_f1"]
    pct_bt_cols = [
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
    ]
    display_model = model_table.copy()
    display_backtest = backtest_table.copy()
    display_model = percent_columns(display_model, pct_model_cols)
    for column in display_model.columns:
        if column not in {"model", "selected_params"} and column not in pct_model_cols:
            display_model[column] = display_model[column].map(format_float)
    display_backtest = percent_columns(display_backtest, pct_bt_cols)
    for column in display_backtest.columns:
        if column not in {"model", "selected_params"} and column not in pct_bt_cols:
            display_backtest[column] = display_backtest[column].map(format_float)
    display_split = split_summary.copy()
    display_split["mean_next_day_return"] = display_split["mean_next_day_return"].map(format_pct)
    display_split["annualized_vol_next_day_return"] = display_split["annualized_vol_next_day_return"].map(format_pct)

    style = """
    <style>
    body { font-family: Arial, sans-serif; margin: 28px; color: #18202A; background: #FAFBFC; }
    h1, h2 { color: #111827; }
    .note { background: #EDF4FF; border-left: 5px solid #1464F4; padding: 12px 16px; margin: 16px 0 24px; }
    table { border-collapse: collapse; width: 100%; margin: 14px 0 30px; font-size: 13px; background: white; }
    th { background: #111827; color: white; position: sticky; top: 0; }
    th, td { border: 1px solid #D8DEE9; padding: 8px 9px; text-align: right; white-space: nowrap; }
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) { text-align: left; }
    img { width: 100%; border: 1px solid #D8DEE9; margin: 14px 0 28px; background: white; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .small { color: #53606F; font-size: 13px; }
    </style>
    """
    image_html = "\n".join(
        f'<h2>{path.stem.replace("_", " ").title()}</h2><img src="data:image/png;base64,{image_to_base64(path)}" />'
        for path in image_paths
    )
    html = f"""
    <!doctype html>
    <html lang="vi">
    <head><meta charset="utf-8"><title>VN-Index Linear Regression Backtest</title>{style}</head>
    <body>
    <h1>VN-Index Linear Regression Forecasting & Backtest</h1>
    <div class="note">
      <strong>Best model:</strong> {best.name} |
      <strong>Test Sharpe:</strong> {best.backtest_metrics["sharpe"]:.2f} |
      <strong>CAGR:</strong> {best.backtest_metrics["cagr"]:.2%} |
      <strong>Max Drawdown:</strong> {best.backtest_metrics["max_drawdown"]:.2%}
      <div class="small">Data range: {data_start.date()} to {data_end.date()}. Strategy: long VN-Index when next-day predicted log return is above the one-way cost threshold; otherwise cash.</div>
    </div>
    <h2>Data Split Summary</h2>
    {display_split.to_html(index=False, escape=False)}
    <h2>Prediction Metrics - Test Set</h2>
    {display_model.to_html(index=False, escape=False)}
    <h2>Strategy Backtest Metrics - Test Set</h2>
    {display_backtest.to_html(index=False, escape=False)}
    <div class="grid">{image_html}</div>
    </body></html>
    """
    (output_dir / "report.html").write_text(html, encoding="utf-8")

    md = [
        "# VN-Index Linear Regression Forecasting & Backtest",
        "",
        f"- Best model: **{best.name}**",
        f"- Test Sharpe: **{best.backtest_metrics['sharpe']:.2f}**",
        f"- CAGR: **{best.backtest_metrics['cagr']:.2%}**",
        f"- Max drawdown: **{best.backtest_metrics['max_drawdown']:.2%}**",
        f"- Data range: {data_start.date()} to {data_end.date()}",
        "",
        "## Output files",
        "",
        "- `model_prediction_metrics.csv`",
        "- `strategy_backtest_metrics.csv`",
        "- `best_model_test_predictions.csv`",
        "- `report.html`",
        "",
        "## Images",
        "",
    ]
    md.extend([f"- `{path.name}`" for path in image_paths])
    (output_dir / "report_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="VN-Index linear regression forecasting and strategy backtest.")
    parser.add_argument("--data", type=Path, default=Path("data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs_linear_regression_backtest"))
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_vnindex_csv(args.data)
    data, feature_cols = make_features(df)
    splits = split_chronological(data)
    results = fit_and_select(splits, feature_cols, cost_bps=args.cost_bps)
    split_summary, model_table, backtest_table = save_tables(results, splits, output_dir, args.cost_bps)

    best_name = backtest_table.iloc[0]["model"]
    best = next(result for result in results if result.name == best_name)
    best.backtest_frame.to_csv(output_dir / "best_model_test_predictions.csv", index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(output_dir / "feature_list.csv", index=False)

    image_paths = [
        plot_dashboard(best, output_dir),
        plot_equity(best, output_dir),
        plot_drawdown(best, output_dir),
        plot_prediction_scatter(best, splits, output_dir),
        plot_rolling_metrics(best, output_dir),
        plot_monthly_heatmap(best, output_dir),
    ]
    coef_path = plot_coefficients(best, feature_cols, output_dir)
    if coef_path is not None:
        image_paths.append(coef_path)

    write_report(output_dir, split_summary, model_table, backtest_table, best, image_paths, df["Date"].min(), df["Date"].max())

    print("VN-Index linear regression backtest completed.")
    print(f"Rows parsed: {len(df):,}; feature rows: {len(data):,}; features: {len(feature_cols):,}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Best model: {best.name} with params={best.params}")
    print(
        "Best strategy metrics: "
        f"CAGR={best.backtest_metrics['cagr']:.2%}, "
        f"Sharpe={best.backtest_metrics['sharpe']:.2f}, "
        f"MaxDD={best.backtest_metrics['max_drawdown']:.2%}, "
        f"TotalReturn={best.backtest_metrics['total_return']:.2%}"
    )
    print("\nTop prediction metrics:")
    print(model_table.head(6).to_string(index=False))
    print("\nTop strategy metrics:")
    cols = ["model", "total_return", "cagr", "annual_volatility", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate_active_days", "exposure", "number_of_trades"]
    print(backtest_table[cols].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
