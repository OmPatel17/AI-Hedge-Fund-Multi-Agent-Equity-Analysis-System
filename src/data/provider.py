from __future__ import annotations
from typing import Protocol, List

from src.data.models import (
    Price,
    FinancialMetrics,
    LineItem,
    CompanyNews,
)


class MarketDataProvider(Protocol):
    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs) -> List[Price]:
        raise NotImplementedError

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10, **kwargs) -> List[FinancialMetrics]:
        raise NotImplementedError

    def search_line_items(self, ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10, **kwargs) -> List[LineItem]:
        raise NotImplementedError

    def get_company_news(self, ticker: str, start_date: str | None, end_date: str, limit: int = 1000, **kwargs) -> List[CompanyNews]:
        raise NotImplementedError

    def get_insider_trades(self, ticker: str, start_date: str | None, end_date: str, limit: int = 1000, **kwargs):
        raise NotImplementedError

    def get_market_cap(self, ticker: str, as_of_date: str | None = None, **kwargs) -> float | None:
        raise NotImplementedError
