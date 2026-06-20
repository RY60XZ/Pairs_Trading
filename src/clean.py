import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

universe = pd.read_csv(PROCESSED_DATA_PATH / "universe.csv")

min_train_rows = 1700

drop_idx = []
coverage = []
train_prices = []
test_prices = []

for idx, u in universe["ticker"].items():
    u_train = pd.read_csv(RAW_DATA_PATH / f"{u}_train.csv")
    u_test = pd.read_csv(RAW_DATA_PATH / f"{u}_test.csv")
    keep = True
    reason = ""
    if pd.isna(universe.loc[idx, "liquidity_proxy"]):
        keep = False
        reason = "missing_liquidity"
    elif len(u_train) < min_train_rows:
        keep = False
        reason = "short_train"
    elif u_train["adj_close"].isna().any() or (u_train["adj_close"] <= 0).any():
        keep = False
        reason = "bad_prices"

    coverage.append(
        {
            "ticker": u,
            "train_rows": len(u_train),
            "test_rows": len(u_test),
            "keep": keep,
            "reason": reason,
        }
    )

    if keep:
        u_train["date"] = pd.to_datetime(u_train["date"])
        u_test["date"] = pd.to_datetime(u_test["date"])
        train_prices.append(u_train.set_index("date")["adj_close"].rename(u))
        test_prices.append(u_test.set_index("date")["adj_close"].rename(u))
    else:
        drop_idx.append(idx)

universe = universe.drop(drop_idx)
universe.to_csv(PROCESSED_DATA_PATH / "cleaned_universe.csv", index=False)

train_prices = pd.concat(train_prices, axis=1).sort_index().dropna()
test_prices = pd.concat(test_prices, axis=1).sort_index().dropna()
train_log_prices = np.log(train_prices)
test_log_prices = np.log(test_prices)
train_returns = train_prices.pct_change().dropna()
test_returns = test_prices.pct_change().dropna()

train_prices.to_csv(PROCESSED_DATA_PATH / "train_prices.csv")
test_prices.to_csv(PROCESSED_DATA_PATH / "test_prices.csv")
train_log_prices.to_csv(PROCESSED_DATA_PATH / "train_log_prices.csv")
test_log_prices.to_csv(PROCESSED_DATA_PATH / "test_log_prices.csv")
train_returns.to_csv(PROCESSED_DATA_PATH / "train_returns.csv")
test_returns.to_csv(PROCESSED_DATA_PATH / "test_returns.csv")
pd.DataFrame(coverage).to_csv(PROCESSED_DATA_PATH / "coverage.csv", index=False)
