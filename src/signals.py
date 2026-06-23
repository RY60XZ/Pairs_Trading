import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

LOOKBACK_WINDOW = 60
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_LOSS_Z = 3.0
MIN_HALF_LIFE = 5
COINT_PVALUE_THRESHOLD = 0.05
ADF_PVALUE_THRESHOLD = 0.05

def compute_signals(lookback_window=LOOKBACK_WINDOW, entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_loss_z=STOP_LOSS_Z, min_half_life=MIN_HALF_LIFE, coint_pvalue_threshold=COINT_PVALUE_THRESHOLD, adf_pvalue_threshold=ADF_PVALUE_THRESHOLD):
    pair_results = pd.read_csv(PROCESSED_DATA_PATH / "pair_results.csv")
    pair_results = pair_results[pair_results["half_life"] <= lookback_window]
    top_pairs = pair_results[
        (pair_results["coint_pvalue"] < coint_pvalue_threshold) &
        (pair_results["adf_pvalue"] < adf_pvalue_threshold) &
        (pair_results["half_life"] >= min_half_life) &
        (pair_results["half_life"] <= lookback_window) &
        (pair_results["beta"] > 0)
    ].sort_values(["coint_pvalue", "adf_pvalue", "half_life"]).head(20)
    top_pairs.to_csv(PROCESSED_DATA_PATH / "selected_pairs.csv", index=False)

    train_log_prices = pd.read_csv(PROCESSED_DATA_PATH / "train_log_prices.csv")
    test_log_prices = pd.read_csv(PROCESSED_DATA_PATH / "test_log_prices.csv")
    pairs_signals = []
    for period, log_prices in [("train", train_log_prices), ("test", test_log_prices)]:
        for _, pair in top_pairs.iterrows():
            ticker1, ticker2 = pair["ticker1"], pair["ticker2"]
            y = log_prices[ticker1]
            x = log_prices[ticker2]
            date = pd.to_datetime(log_prices["date"])
            spread = y - pair["alpha"] - pair["beta"] * x
            spread = pd.concat([date, spread], axis=1)
            spread.columns = ["date", "spread"]
            spread["z_score"] = (spread["spread"] - spread["spread"].rolling(window=lookback_window).mean()) / spread["spread"].rolling(window=lookback_window).std()
            spread = spread.dropna()
            spread["z_score_lag_1"] = spread["z_score"].shift(1)
            spread = spread.dropna()
            signals = []
            state = 0
            for _, row in spread.iterrows():
                if not state:
                    if entry_z <= row["z_score_lag_1"] < stop_loss_z:
                        state = -1
                    elif -stop_loss_z < row["z_score_lag_1"] <= -entry_z:
                        state = 1
                else:
                    if (
                        abs(row["z_score_lag_1"]) < exit_z
                        or (state == 1 and row["z_score_lag_1"] <= -stop_loss_z)
                        or (state == -1 and row["z_score_lag_1"] >= stop_loss_z)
                    ):
                        state = 0
                signals.append(state)
            spread["signal"] = signals
            spread["period"] = period
            spread["pair"] = ticker1 + "_" + ticker2
            spread["ticker1"] = ticker1
            spread["ticker2"] = ticker2
            pairs_signals.append(spread)

    signals = pd.concat(pairs_signals, ignore_index=True)
    signals = signals[["period", "date", "pair", "ticker1", "ticker2", "spread", "z_score", "z_score_lag_1", "signal"]]
    signals.to_csv(PROCESSED_DATA_PATH / "signals.csv", index=False)

if __name__ == "__main__":
    compute_signals()
