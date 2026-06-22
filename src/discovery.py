import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

universe = pd.read_csv(PROCESSED_DATA_PATH / "cleaned_universe.csv")
train_log_prices = pd.read_csv(PROCESSED_DATA_PATH / "train_log_prices.csv")

pairs = []
tickers_by_industry = dict()
pair_results = []
for _, row in universe.iterrows():
    if (row.industry not in tickers_by_industry):
        tickers_by_industry[row.industry] = [row.ticker]
    else:
        tickers_by_industry[row.industry].append(row.ticker)

for industry, tickers in tickers_by_industry.items():
    if len(tickers) < 2:
        continue
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append((tickers[i], tickers[j], industry))

for pair in pairs:
    ticker1, ticker2, industry = pair
    y = train_log_prices[ticker1]
    x = train_log_prices[ticker2]
    pair_log_prices = pd.concat([y, x], axis=1)
    pair_log_prices.columns = ["y", "x"]
    X = sm.add_constant(pair_log_prices["x"])
    model = sm.OLS(pair_log_prices["y"], X).fit()
    alpha = model.params["const"]
    beta = model.params["x"]
    spread = y - alpha - beta * x
    coint_pvalue = coint(y, x)[1]
    adf_pvalue = adfuller(spread)[1]

    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    half_life_data = pd.concat([spread_diff, spread_lag], axis=1).dropna()
    half_life_data.columns = ["spread_diff", "spread_lag"]
    half_life_model = sm.OLS(
        half_life_data["spread_diff"],
        sm.add_constant(half_life_data["spread_lag"]),
    ).fit()
    mean_reversion_speed = half_life_model.params["spread_lag"]
    if mean_reversion_speed < 0:
        half_life = -np.log(2) / mean_reversion_speed
    else:
        half_life = np.inf

    pair_results.append(
        {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "industry": industry,
            "alpha": alpha,
            "beta": beta,
            "coint_pvalue": coint_pvalue,
            "adf_pvalue": adf_pvalue,
            "half_life": half_life,
        }
    )

pair_results = pd.DataFrame(pair_results)
pair_results = pair_results.sort_values(["coint_pvalue", "adf_pvalue", "half_life"])
pair_results.to_csv(PROCESSED_DATA_PATH / "pair_results.csv", index=False)
