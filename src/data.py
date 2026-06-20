from pathlib import Path
import pandas as pd
import yfinance as yf
import requests
import json
import sys
from io import StringIO

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START_DATE = "2014-01-01"
SPLIT_DATE = "2021-01-01"
END_DATE = "2025-12-31"

def load_sp500_constituents(url: str) -> pd.DataFrame:
    try:
        response = requests.get(
            url=url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"
                )
            },
            timeout=30,
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
    except Exception as e:
        raise RuntimeError(f"Could not read S&P 500 constituents from {url}") from e
    table = next((df for df in tables if {"Symbol", "Security"}.issubset(df.columns)), None)
    table = table.rename(
        columns={
            "Symbol": "raw_ticker",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        }
    )
    table["ticker"] = table["raw_ticker"].map(lambda x: x.strip().upper().replace(".", "-"))
    table = table[["ticker", "company_name", "sector", "industry"]]
    return table

def clean_yfinance_prices(prices : pd.DataFrame) -> pd.DataFrame:
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices.reset_index(inplace=True)
    prices.columns = prices.columns.str.lower().str.replace(" ", "_")
    if "adj_close" not in prices.columns and "close" in prices.columns:
        prices["adj_close"] = prices["close"]
    return prices

def download_prices(ticker: str, start_date: str, split_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker = ticker.upper()
    ticker = ticker.replace(".", "-")
    raw_path = RAW_DATA_PATH / f"{ticker}.csv"
    prices_train = yf.download(
        ticker,
        start=start_date,
        end=split_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    prices_test = yf.download(
        ticker,
        start=split_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    prices_train = clean_yfinance_prices(prices_train)
    prices_test = clean_yfinance_prices(prices_test)
    prices_train.to_csv(RAW_DATA_PATH / f"{ticker}_train.csv", index=False)
    prices_test.to_csv(RAW_DATA_PATH / f"{ticker}_test.csv", index=False)
    print(f"Downloaded {ticker} to {raw_path}")
    return prices_train, prices_test

if __name__ == "__main__":
    universe = load_sp500_constituents(SP500_URL)
    universe["liquidity_proxy"] = 0.0
    universe["start_date"] = START_DATE
    universe["end_date"] = END_DATE
    universe["split_date"] = SPLIT_DATE
    for ticker in universe["ticker"]:
        prices_train, prices_test = download_prices(ticker, START_DATE, SPLIT_DATE, END_DATE)
        universe.loc[universe["ticker"]==ticker, "liquidity_proxy"] = (prices_train["adj_close"]*prices_train["volume"]).mean()
    universe.to_csv(PROCESSED_DATA_PATH / "universe.csv", index=False)
    
