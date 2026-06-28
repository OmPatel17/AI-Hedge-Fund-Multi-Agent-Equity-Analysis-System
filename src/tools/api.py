import datetime
import logging
import os
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)

from src.data.market_data_service import market_data

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue
        
        # Return the response (whether success, other errors, or final 429)
        return response


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]
    # If not in cache, fetch using the MarketDataService
    try:
        prices = market_data.get_prices(ticker, start_date, end_date, api_key=api_key)
    except Exception as e:
        logger.warning("market_data.get_prices failed for %s: %s", ticker, e)
        return []

    if not prices:
        return []

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # If not in cache, fetch via MarketDataService
    try:
        financial_metrics = market_data.get_financial_metrics(ticker, end_date, period=period, limit=limit, api_key=api_key)
    except Exception as e:
        logger.warning("market_data.get_financial_metrics failed for %s: %s", ticker, e)
        return []

    if not financial_metrics:
        return []

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API and log data quality before returning."""
    try:
        results = market_data.search_line_items(ticker, line_items, end_date, period=period, limit=limit, api_key=api_key)
    except Exception as e:
        logger.warning("market_data.search_line_items failed for %s: %s", ticker, e)
        return []

    trimmed = results[:limit]

    # Log provenance and completeness so data gaps are visible before LLM runs
    try:
        from src.utils.data_quality import validate_required_fields
        validate_required_fields(ticker, "line_items", trimmed, line_items)
    except Exception:
        pass

    return trimmed


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]
    try:
        trades = market_data.get_insider_trades(ticker, start_date=start_date, end_date=end_date, limit=limit, api_key=api_key)
    except Exception as e:
        logger.warning("market_data.get_insider_trades failed for %s: %s", ticker, e)
        return []

    if not trades:
        return []

    _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
    return trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    try:
        news = market_data.get_company_news(ticker, start_date=start_date, end_date=end_date, limit=limit, api_key=api_key)
    except Exception as e:
        logger.warning("market_data.get_company_news failed for %s: %s", ticker, e)
        return []

    if not news:
        return []

    _cache.set_company_news(cache_key, [n.model_dump() for n in news])
    return news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from the API."""
    try:
        mc = market_data.get_market_cap(ticker, as_of_date=end_date, api_key=api_key)
        return mc
    except Exception as e:
        logger.warning("market_data.get_market_cap failed for %s: %s", ticker, e)
        # Fallback to previous behavior of deriving from financial_metrics
        financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
        if not financial_metrics:
            return None

        market_cap = financial_metrics[0].market_cap
        if not market_cap:
            return None
        return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
