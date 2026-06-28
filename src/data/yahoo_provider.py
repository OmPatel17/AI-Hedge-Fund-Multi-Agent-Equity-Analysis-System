from __future__ import annotations
from typing import List
import logging
import pandas as pd

from src.data.provider import MarketDataProvider
from src.data.models import Price

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except Exception:
    yf = None


class YahooFinanceProvider(MarketDataProvider):
    def __init__(self):
        if yf is None:
            logger.warning("yfinance not installed. Install with 'pip install yfinance' to use Yahoo provider.")

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs) -> List[Price]:
        if not ticker:
            return []

        if yf is None:
            raise RuntimeError("yfinance is not installed")

        try:
            tk = yf.Ticker(ticker)
            hist: pd.DataFrame = tk.history(start=start_date, end=end_date, interval="1d", actions=True)
        except Exception as e:
            logger.warning("Yahoo history fetch failed for %s: %s", ticker, e)
            return []

        if hist is None or hist.empty:
            return []

        prices: List[Price] = []
        for idx, row in hist.iterrows():
            # Handle missing values robustly
            try:
                o = None if pd.isna(row.get("Open")) else float(row.get("Open"))
                h = None if pd.isna(row.get("High")) else float(row.get("High"))
                l = None if pd.isna(row.get("Low")) else float(row.get("Low"))
                c = None if pd.isna(row.get("Close")) else float(row.get("Close"))
                v = 0 if pd.isna(row.get("Volume")) else int(row.get("Volume"))
            except Exception:
                o, h, l, c, v = 0.0, 0.0, 0.0, 0.0, 0

            prices.append(
                Price(
                    open=o or 0.0,
                    high=h or 0.0,
                    low=l or 0.0,
                    close=c or 0.0,
                    volume=v,
                    time=idx.strftime("%Y-%m-%dT%H:%M:%S"),
                )
            )

        return prices

    def get_market_cap(self, ticker: str, as_of_date: str | None = None, **kwargs) -> float | None:
        if yf is None:
            return None
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap")
            if mc:
                return float(mc)
        except Exception as e:
            logger.warning("yfinance market cap fetch failed for %s: %s", ticker, e)
        return None
