from __future__ import annotations

import logging
import os
from typing import List

from src.data.providers.yahoo_provider import YahooFinanceProvider
from src.data.providers.edgar_provider import EdgarProvider
from src.data.providers.gdelt_provider import GDELTProvider
from src.data.providers.fred_provider import FREDProvider
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

logger = logging.getLogger(__name__)


class MarketDataService:
    def __init__(self):
        self.yahoo = YahooFinanceProvider()
        self.edgar = EdgarProvider()
        self.gdelt = GDELTProvider()
        self.fred = FREDProvider(api_key=os.environ.get("FRED_API_KEY"))

    # ------------------------------------------------------------------
    # Prices — Yahoo only
    # ------------------------------------------------------------------

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs) -> List[Price]:
        return self.yahoo.get_prices(ticker, start_date, end_date)

    # Price-based fields that only Yahoo can supply (current market data, not period-specific)
    _PRICE_FIELDS: frozenset[str] = frozenset([
        "market_cap", "enterprise_value", "price_to_earnings_ratio",
        "price_to_book_ratio", "price_to_sales_ratio",
        "enterprise_value_to_ebitda_ratio", "enterprise_value_to_revenue_ratio",
        "free_cash_flow_yield", "peg_ratio",
    ])
    _FM_IDENTITY: frozenset[str] = frozenset(["ticker", "report_period", "period", "currency"])

    # ------------------------------------------------------------------
    # Financial metrics — EDGAR primary, merged with Yahoo market fields
    # ------------------------------------------------------------------

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
        **kwargs,
    ) -> List[FinancialMetrics]:
        edgar_results = self.edgar.get_financial_metrics(ticker, end_date, period=period, limit=limit)
        # Always fetch Yahoo as well so we can fill price-based ratios that EDGAR cannot provide.
        yahoo_results = self.yahoo.get_financial_metrics(ticker, end_date, period=period, limit=min(limit, 3))

        if not edgar_results:
            logger.debug("[MDS] EDGAR metrics empty for %s, using Yahoo only", ticker)
            return yahoo_results or []

        if not yahoo_results:
            return edgar_results

        # Build lookup tables
        yahoo_latest = yahoo_results[0]
        yahoo_by_period: dict[str, FinancialMetrics] = {y.report_period: y for y in yahoo_results}
        mergeable = [f for f in FinancialMetrics.model_fields if f not in self._FM_IDENTITY]

        for em in edgar_results:
            # Step 1: price-based market fields from Yahoo's most-recent info object
            # (these are current market values, same for every historical period)
            for field in self._PRICE_FIELDS:
                if getattr(em, field) is None:
                    val = getattr(yahoo_latest, field, None)
                    if val is not None:
                        setattr(em, field, val)

            # Step 2: fill remaining nulls from the period-matched Yahoo entry
            y_match = yahoo_by_period.get(em.report_period)
            if y_match:
                for field in mergeable:
                    if field in self._PRICE_FIELDS:
                        continue
                    if getattr(em, field) is None:
                        val = getattr(y_match, field, None)
                        if val is not None:
                            setattr(em, field, val)

        return edgar_results

    # ------------------------------------------------------------------
    # Line items — EDGAR primary, Yahoo fallback
    # ------------------------------------------------------------------

    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
        **kwargs,
    ) -> List[LineItem]:
        results = self.edgar.search_line_items(ticker, line_items, end_date, period=period, limit=limit)
        if results:
            return results
        logger.debug("[MDS] EDGAR line items empty for %s, trying Yahoo", ticker)
        return self.yahoo.search_line_items(ticker, line_items, end_date, period=period, limit=limit)

    # ------------------------------------------------------------------
    # News — GDELT primary, Yahoo fallback
    # ------------------------------------------------------------------

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int = 25,
        **kwargs,
    ) -> List[CompanyNews]:
        results = self.gdelt.get_company_news(ticker, start_date, end_date, limit=limit)
        if results:
            return results
        logger.debug("[MDS] GDELT news empty for %s, trying Yahoo fallback", ticker)
        return GDELTProvider.fallback_yahoo_news(
            ticker, start_date=start_date, end_date=end_date, limit=limit
        )

    # ------------------------------------------------------------------
    # Insider trades — EDGAR only
    # ------------------------------------------------------------------

    def get_insider_trades(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int = 1000,
        **kwargs,
    ) -> List[InsiderTrade]:
        return self.edgar.get_insider_trades(ticker, start_date, end_date, limit=limit)

    # ------------------------------------------------------------------
    # Market cap — Yahoo
    # ------------------------------------------------------------------

    def get_market_cap(self, ticker: str, as_of_date: str | None = None, **kwargs) -> float | None:
        return self.yahoo.get_market_cap(ticker, as_of_date=as_of_date)


# Singleton instance — imported by src/tools/api.py and agents
market_data = MarketDataService()
