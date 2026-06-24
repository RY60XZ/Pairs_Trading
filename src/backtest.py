import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
TRADING_DAYS = 252
TRANSACTION_COST_RATE = 0.0003

selected_pairs = pd.read_csv(PROCESSED_DATA_PATH / "selected_pairs.csv")
selected_pairs["pair"] = selected_pairs["ticker1"] + "_" + selected_pairs["ticker2"]
signals = pd.read_csv(PROCESSED_DATA_PATH / "signals.csv")
train_returns = pd.read_csv(PROCESSED_DATA_PATH / "train_returns.csv")
test_returns = pd.read_csv(PROCESSED_DATA_PATH / "test_returns.csv")

def compute_backtest(initial_capital=INITIAL_CAPITAL, transaction_cost=TRANSACTION_COST_RATE, period="train"):
    gross_exposure_per_pair = initial_capital/ len(selected_pairs)
    pair_results = []
    if period == "train":
        period_returns = train_returns
    elif period == "test":
        period_returns = test_returns
    else:
        raise ValueError("period must be 'train' or 'test'")
    period_signals = signals[signals["period"] == period]
    period_returns = period_returns.set_index("date")

    for _, pair in selected_pairs.iterrows():
        ticker_1 = pair["ticker1"]
        ticker_2 = pair["ticker2"]
        beta = pair["beta"]
        pair_name = f"{ticker_1}_{ticker_2}"
        pair_signals = period_signals[period_signals["pair"] == pair_name].copy()

        pair_signals["ticker1_dollars"] = pair_signals["signal"] * gross_exposure_per_pair / (1+beta)
        pair_signals["ticker2_dollars"] = -pair_signals["signal"] * beta * gross_exposure_per_pair / (1+beta)
        pair_signals["exposure"] = abs(pair_signals["ticker1_dollars"]) + abs(pair_signals["ticker2_dollars"])
        pair_signals["ticker1_returns"] = period_returns.loc[pair_signals["date"], ticker_1].to_numpy()
        pair_signals["ticker2_returns"] = period_returns.loc[pair_signals["date"], ticker_2].to_numpy()
        pair_signals["gross_pnl"] = (
                pair_signals["ticker1_dollars"] * pair_signals["ticker1_returns"] +
                pair_signals["ticker2_dollars"] * pair_signals["ticker2_returns"]
        )

        pair_signals["ticker1_dollars_diff"] = pair_signals["ticker1_dollars"].diff().fillna(pair_signals["ticker1_dollars"].iloc[0]).abs()
        pair_signals["ticker2_dollars_diff"] = pair_signals["ticker2_dollars"].diff().fillna(pair_signals["ticker2_dollars"].iloc[0]).abs()
        pair_signals["cost"] = (
            pair_signals["ticker1_dollars_diff"] + pair_signals["ticker2_dollars_diff"]
        ) * transaction_cost
        pair_signals["net_pnl"] = pair_signals["gross_pnl"] - pair_signals["cost"]
        pair_results.append(pair_signals)
    pair_results = pd.concat(pair_results, ignore_index=True)
    pair_results.to_csv(PROCESSED_DATA_PATH / f"{period}_backtest_pair_results.csv", index=False)
    daily_pnl = pair_results.groupby("date")[["gross_pnl", "cost", "net_pnl", "exposure"]].sum().reset_index()
    daily_pnl["daily_return"] = daily_pnl["net_pnl"] / initial_capital
    daily_pnl["equity"] = initial_capital + daily_pnl["net_pnl"].cumsum()
    daily_pnl["drawdown"] = daily_pnl["equity"] / daily_pnl["equity"].cummax() - 1
    daily_pnl.to_csv(PROCESSED_DATA_PATH / f"{period}_backtest_daily_returns.csv", index=False)

    total_return = daily_pnl["equity"].iloc[-1] / initial_capital - 1
    annualized_return = daily_pnl["daily_return"].mean() * TRADING_DAYS
    annualized_volatility = daily_pnl["daily_return"].std() * np.sqrt(TRADING_DAYS)
    sharpe_ratio = annualized_return / annualized_volatility
    max_drawdown = daily_pnl["drawdown"].min()

    metrics = pd.DataFrame([{
        "period": period,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_cost": daily_pnl["cost"].sum(),
    }])
    metrics.to_csv(PROCESSED_DATA_PATH / f"{period}_backtest_metrics.csv", index=False)


if __name__ == "__main__":
    compute_backtest(period="train")
    compute_backtest(period="test")
