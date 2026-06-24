import pandas as pd
import importlib
from pathlib import Path

import signals
import backtest

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

LOOKBACK_WINDOWS = [60, 120]
ENTRY_Z_VALUES = [2.0, 2.5]
EXIT_Z_VALUES = [0.5, 1.0]
STOP_LOSS_Z_VALUES = [3.0, 3.5]
TRANSACTION_COSTS = [0.0001, 0.0003, 0.0005]

def read_metrics(period):
    metrics = pd.read_csv(PROCESSED_DATA_PATH / f"{period}_backtest_metrics.csv").iloc[0]
    return {
        f"{period}_total_return": metrics["total_return"],
        f"{period}_annualized_return": metrics["annualized_return"],
        f"{period}_annualized_volatility": metrics["annualized_volatility"],
        f"{period}_sharpe_ratio": metrics["sharpe_ratio"],
        f"{period}_max_drawdown": metrics["max_drawdown"],
        f"{period}_total_cost": metrics["total_cost"],
    }

def run_one_config(lookback_window, entry_z, exit_z, stop_loss_z, transaction_cost):
    lookback_window = int(lookback_window)
    signals.compute_signals(
        lookback_window=lookback_window,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_loss_z=stop_loss_z,
    )

    importlib.reload(backtest)
    backtest.compute_backtest(period="train", transaction_cost=transaction_cost)
    backtest.compute_backtest(period="test", transaction_cost=transaction_cost)

    result = {
        "lookback_window": lookback_window,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "stop_loss_z": stop_loss_z,
        "transaction_cost": transaction_cost,
    }
    result.update(read_metrics("train"))
    result.update(read_metrics("test"))
    return result

def compute_validation():
    results = []
    for lookback_window in LOOKBACK_WINDOWS:
        for entry_z in ENTRY_Z_VALUES:
            for exit_z in EXIT_Z_VALUES:
                for stop_loss_z in STOP_LOSS_Z_VALUES:
                    for transaction_cost in TRANSACTION_COSTS:
                        result = run_one_config(
                            lookback_window=lookback_window,
                            entry_z=entry_z,
                            exit_z=exit_z,
                            stop_loss_z=stop_loss_z,
                            transaction_cost=transaction_cost,
                        )
                        results.append(result)

    results = pd.DataFrame(results)
    results = results.sort_values(["train_sharpe_ratio", "test_sharpe_ratio"], ascending=False)
    results.to_csv(PROCESSED_DATA_PATH / "validation_results.csv", index=False)

    best_result = results[results["transaction_cost"]==backtest.TRANSACTION_COST_RATE].iloc[0]
    pd.DataFrame([best_result]).to_csv(PROCESSED_DATA_PATH / "best_validation_result.csv", index=False)
    run_one_config(
        lookback_window=best_result["lookback_window"],
        entry_z=best_result["entry_z"],
        exit_z=best_result["exit_z"],
        stop_loss_z=best_result["stop_loss_z"],
        transaction_cost=best_result["transaction_cost"],
    )

if __name__ == "__main__":
    compute_validation()
