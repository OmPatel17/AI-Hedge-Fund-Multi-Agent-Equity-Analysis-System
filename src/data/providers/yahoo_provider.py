from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from src.data.cache import get_cache
from src.data.models import CompanyNews, FinancialMetrics, LineItem, Price
from src.data.provider import MarketDataProvider

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except Exception:
    yf = None


class YahooFinanceProvider(MarketDataProvider):
    def __init__(self):
        if yf is None:
            logger.warning("yfinance not installed. Install with 'pip install yfinance'.")
        self._cache = get_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ticker(self, ticker: str):
        return yf.Ticker(ticker)

    def _frames(self, ticker: str):
        tk = self._ticker(ticker)
        income = tk.financials
        balance = tk.balance_sheet
        cashflow = tk.cashflow
        info = tk.info or {}
        return income, balance, cashflow, info

    def _frame_columns(self, income, balance, cashflow, end_date: str) -> list[pd.Timestamp]:
        cutoff = pd.to_datetime(end_date).normalize()
        seen: dict[pd.Timestamp, None] = {}
        for frame in (income, balance, cashflow):
            if frame is None or getattr(frame, "empty", True):
                continue
            for col in frame.columns:
                try:
                    ts = pd.to_datetime(col).normalize()
                except Exception:
                    continue
                if ts <= cutoff:
                    seen[ts] = None
        return sorted(seen.keys(), reverse=True)

    def _value_from_frame(self, frame, column, labels: list[str]) -> float | None:
        if frame is None or getattr(frame, "empty", True):
            return None
        for label in labels:
            if label in frame.index and column in frame.columns:
                value = frame.at[label, column]
                if pd.notna(value):
                    try:
                        return float(value)
                    except Exception:
                        pass
        return None

    @staticmethod
    def _safe_div(a: float | None, b: float | None) -> float | None:
        try:
            return a / b if a is not None and b not in (None, 0) else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs) -> List[Price]:
        if not ticker or yf is None:
            return []

        cache_key = f"yahoo_prices_{ticker}_{start_date}_{end_date}"
        cached = self._cache.get_prices(cache_key)
        if cached is not None:
            return [Price(**p) for p in cached]

        try:
            tk = self._ticker(ticker)
            # yfinance end is exclusive — always extend by 1 day so a single-date
            # request returns the bar for that date (if it's a trading day).
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
            fetch_end = (end_dt + timedelta(days=1)).isoformat()
            hist: pd.DataFrame = tk.history(start=start_date, end=fetch_end, interval="1d", actions=False)
            # If the window falls entirely on a non-trading day (weekend/holiday),
            # widen up to 7 calendar days back and keep only the last bar on or before end_date.
            if hist is None or hist.empty:
                wide_start = (start_dt - timedelta(days=7)).isoformat()
                hist = tk.history(start=wide_start, end=fetch_end, interval="1d", actions=False)
                if hist is not None and not hist.empty:
                    # Drop timezone info before comparing to avoid tz-aware vs naive errors
                    bar_dates = [idx.strftime("%Y-%m-%d") for idx in hist.index]
                    mask = [d <= end_date for d in bar_dates]
                    hist = hist[mask].iloc[-1:]
        except Exception as e:
            logger.warning("Yahoo price fetch failed for %s: %s", ticker, e)
            return []

        if hist is None or hist.empty:
            return []

        prices: list[Price] = []
        for idx, row in hist.iterrows():
            try:
                o = float(row["Open"]) if pd.notna(row.get("Open")) else 0.0
                h = float(row["High"]) if pd.notna(row.get("High")) else 0.0
                lo = float(row["Low"]) if pd.notna(row.get("Low")) else 0.0
                c = float(row["Close"]) if pd.notna(row.get("Close")) else 0.0
                v = int(row["Volume"]) if pd.notna(row.get("Volume")) else 0
            except Exception:
                o, h, lo, c, v = 0.0, 0.0, 0.0, 0.0, 0
            prices.append(Price(open=o, high=h, low=lo, close=c, volume=v, time=idx.strftime("%Y-%m-%dT%H:%M:%S")))

        self._cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
        **kwargs,
    ) -> List[FinancialMetrics]:
        if not ticker or yf is None:
            return []

        cache_key = f"yahoo_metrics_{ticker}_{end_date}_{period}_{limit}"
        cached = self._cache.get_financial_metrics(cache_key)
        if cached is not None:
            return [FinancialMetrics(**m) for m in cached]

        try:
            income, balance, cashflow, info = self._frames(ticker)
            columns = self._frame_columns(income, balance, cashflow, end_date=end_date)[:limit]
            if not columns:
                return []

            results: list[FinancialMetrics] = []
            previous_revenue: float | None = None
            previous_earnings: float | None = None
            previous_book_value: float | None = None
            previous_fcf: float | None = None
            previous_op_income: float | None = None
            previous_ebitda: float | None = None

            for col in columns:
                revenue = self._value_from_frame(income, col, ["Total Revenue", "Revenue", "Operating Revenue"])
                gross_profit = self._value_from_frame(income, col, ["Gross Profit"])
                operating_income = self._value_from_frame(income, col, ["Operating Income"])
                net_income = self._value_from_frame(income, col, ["Net Income"])
                ebitda = self._value_from_frame(income, col, ["EBITDA", "Normalized EBITDA"])
                interest_expense = self._value_from_frame(income, col, ["Interest Expense", "Interest Expense Non Operating"])
                eps = self._value_from_frame(income, col, ["Diluted EPS", "Basic EPS", "EPS"])
                dividends_paid = self._value_from_frame(cashflow, col, ["Common Stock Dividend", "Dividends Paid", "Cash Dividends Paid"])

                total_assets = self._value_from_frame(balance, col, ["Total Assets"])
                total_liabilities = self._value_from_frame(balance, col, ["Total Liab", "Total Liabilities", "Total Liabilities Net Minority Interest"])
                current_assets = self._value_from_frame(balance, col, ["Current Assets", "Total Current Assets"])
                current_liabilities = self._value_from_frame(balance, col, ["Current Liabilities", "Total Current Liabilities"])
                inventory = self._value_from_frame(balance, col, ["Inventory"])
                cash = self._value_from_frame(balance, col, ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"])
                equity = self._value_from_frame(balance, col, ["Stockholders Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"])
                total_debt = self._value_from_frame(balance, col, ["Total Debt", "Debt"])
                book_value = self._value_from_frame(balance, col, ["Book Value"])
                shares = self._value_from_frame(balance, col, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]) or info.get("sharesOutstanding")

                operating_cash_flow = self._value_from_frame(cashflow, col, ["Operating Cash Flow", "Total Cash From Operating Activities"])
                free_cash_flow = self._value_from_frame(cashflow, col, ["Free Cash Flow"])
                capex = self._value_from_frame(cashflow, col, ["Capital Expenditure", "Capital Expenditures"])

                # Derived ratios
                gross_margin = self._safe_div(gross_profit, revenue)
                operating_margin = self._safe_div(operating_income, revenue)
                net_margin = self._safe_div(net_income, revenue)
                roe = self._safe_div(net_income, equity)
                roa = self._safe_div(net_income, total_assets)
                debt_to_equity = self._safe_div(total_debt, equity)
                debt_to_assets = self._safe_div(total_debt, total_assets)
                current_ratio = self._safe_div(current_assets, current_liabilities)
                quick_ratio = self._safe_div((current_assets - inventory) if current_assets is not None and inventory is not None else None, current_liabilities)
                cash_ratio = self._safe_div(cash, current_liabilities)
                ocf_ratio = self._safe_div(operating_cash_flow, current_liabilities)
                asset_turnover = self._safe_div(revenue, total_assets)
                inventory_turnover = self._safe_div(revenue, inventory)
                receivables = self._value_from_frame(balance, col, ["Receivables", "Net Receivables", "Accounts Receivable"])
                receivables_turnover = self._safe_div(revenue, receivables)
                dso = (365.0 / receivables_turnover) if receivables_turnover not in (None, 0) else None
                working_capital = (current_assets - current_liabilities) if current_assets is not None and current_liabilities is not None else None
                working_capital_turnover = self._safe_div(revenue, working_capital)
                inventory_days = (365.0 / inventory_turnover) if inventory_turnover not in (None, 0) else None
                operating_cycle = (dso + inventory_days) if dso is not None and inventory_days is not None else None
                invested_capital = (total_assets - current_liabilities) if total_assets is not None and current_liabilities is not None else None
                roic = self._safe_div(operating_income, invested_capital)
                interest_coverage = self._safe_div(operating_income, abs(interest_expense) if interest_expense is not None else None)
                book_value_per_share = self._safe_div(book_value, shares)
                fcf_per_share = self._safe_div(free_cash_flow, shares)
                payout_ratio = self._safe_div(abs(dividends_paid) if dividends_paid is not None else None, net_income)

                # Market data from info (current, not historical per-period)
                market_cap = info.get("marketCap")
                ev = info.get("enterpriseValue")
                pe = info.get("trailingPE")
                pb = info.get("priceToBook")
                ps = info.get("priceToSalesTrailing12Months")
                ev_ebitda = info.get("enterpriseToEbitda")
                peg = info.get("pegRatio")
                fcf_yield = self._safe_div(free_cash_flow, market_cap) if market_cap else None
                ev_revenue = self._safe_div(ev, revenue) if ev else None

                # Growth rates
                revenue_growth = self._safe_div(revenue - previous_revenue, abs(previous_revenue)) if previous_revenue not in (None, 0) and revenue is not None else None
                earnings_growth = self._safe_div(net_income - previous_earnings, abs(previous_earnings)) if previous_earnings not in (None, 0) and net_income is not None else None
                book_value_growth = self._safe_div(book_value - previous_book_value, abs(previous_book_value)) if previous_book_value not in (None, 0) and book_value is not None else None
                fcf_growth = self._safe_div(free_cash_flow - previous_fcf, abs(previous_fcf)) if previous_fcf not in (None, 0) and free_cash_flow is not None else None
                op_income_growth = self._safe_div(operating_income - previous_op_income, abs(previous_op_income)) if previous_op_income not in (None, 0) and operating_income is not None else None
                ebitda_growth = self._safe_div(ebitda - previous_ebitda, abs(previous_ebitda)) if previous_ebitda not in (None, 0) and ebitda is not None else None

                results.append(FinancialMetrics(
                    ticker=ticker,
                    report_period=col.strftime("%Y-%m-%d"),
                    period="annual",
                    currency=info.get("currency") or "USD",
                    market_cap=float(market_cap) if market_cap is not None else None,
                    enterprise_value=float(ev) if ev is not None else None,
                    price_to_earnings_ratio=float(pe) if pe is not None else None,
                    price_to_book_ratio=float(pb) if pb is not None else None,
                    price_to_sales_ratio=float(ps) if ps is not None else None,
                    enterprise_value_to_ebitda_ratio=float(ev_ebitda) if ev_ebitda is not None else None,
                    enterprise_value_to_revenue_ratio=ev_revenue,
                    free_cash_flow_yield=fcf_yield,
                    peg_ratio=float(peg) if peg is not None else None,
                    gross_margin=gross_margin,
                    operating_margin=operating_margin,
                    net_margin=net_margin,
                    return_on_equity=roe,
                    return_on_assets=roa,
                    return_on_invested_capital=roic,
                    asset_turnover=asset_turnover,
                    inventory_turnover=inventory_turnover,
                    receivables_turnover=receivables_turnover,
                    days_sales_outstanding=dso,
                    operating_cycle=operating_cycle,
                    working_capital_turnover=working_capital_turnover,
                    current_ratio=current_ratio,
                    quick_ratio=quick_ratio,
                    cash_ratio=cash_ratio,
                    operating_cash_flow_ratio=ocf_ratio,
                    debt_to_equity=debt_to_equity,
                    debt_to_assets=debt_to_assets,
                    interest_coverage=interest_coverage,
                    revenue_growth=revenue_growth,
                    earnings_growth=earnings_growth,
                    book_value_growth=book_value_growth,
                    earnings_per_share_growth=None,
                    free_cash_flow_growth=fcf_growth,
                    operating_income_growth=op_income_growth,
                    ebitda_growth=ebitda_growth,
                    payout_ratio=payout_ratio,
                    earnings_per_share=eps,
                    book_value_per_share=book_value_per_share,
                    free_cash_flow_per_share=fcf_per_share,
                ))

                previous_revenue = revenue
                previous_earnings = net_income
                previous_book_value = book_value
                previous_fcf = free_cash_flow
                previous_op_income = operating_income
                previous_ebitda = ebitda

            self._cache.set_financial_metrics(cache_key, [r.model_dump() for r in results])
            return results

        except Exception:
            logger.exception("Yahoo get_financial_metrics failed for %s", ticker)
            return []

    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
        **kwargs,
    ) -> List[LineItem]:
        if not ticker or yf is None:
            return []

        items_key = "_".join(sorted(line_items))
        cache_key = f"yahoo_lineitems_{ticker}_{end_date}_{period}_{limit}_{items_key}"
        cached = self._cache.get_line_items(cache_key)
        if cached is not None:
            return [LineItem(**li) for li in cached]

        try:
            income, balance, cashflow, info = self._frames(ticker)
            columns = self._frame_columns(income, balance, cashflow, end_date=end_date)[:limit]
            if not columns:
                return []

            field_map: dict[str, list[tuple]] = {
                "revenue":                                [(income,   ["Total Revenue", "Revenue", "Operating Revenue"])],
                "net_income":                             [(income,   ["Net Income"])],
                "operating_income":                       [(income,   ["Operating Income"])],
                "gross_profit":                           [(income,   ["Gross Profit"])],
                "research_and_development":               [(income,   ["Research And Development", "Research Development"])],
                "depreciation_and_amortization":          [(income,   ["Reconciled Depreciation", "Depreciation And Amortization", "Depreciation"])],
                "operating_expense":                      [(income,   ["Total Expenses", "Operating Expense"])],
                "earnings_per_share":                     [(income,   ["Diluted EPS", "Basic EPS", "EPS"])],
                "total_assets":                           [(balance,  ["Total Assets"])],
                "total_liabilities":                      [(balance,  ["Total Liab", "Total Liabilities", "Total Liabilities Net Minority Interest"])],
                "total_debt":                             [(balance,  ["Total Debt", "Debt"])],
                "shareholders_equity":                    [(balance,  ["Stockholders Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"])],
                "current_assets":                         [(balance,  ["Current Assets", "Total Current Assets"])],
                "current_liabilities":                    [(balance,  ["Current Liabilities", "Total Current Liabilities"])],
                "cash_and_equivalents":                   [(balance,  ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"])],
                "goodwill_and_intangible_assets":         [(balance,  ["Goodwill And Other Intangible Assets", "Goodwill", "Other Intangible Assets"])],
                "book_value_per_share":                   [(balance,  ["Book Value"])],
                "outstanding_shares":                     [(balance,  ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"])],
                "shares_outstanding":                     [(balance,  ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"])],
                "free_cash_flow":                         [(cashflow, ["Free Cash Flow"])],
                "operating_cash_flow":                    [(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])],
                "capital_expenditure":                    [(cashflow, ["Capital Expenditure", "Capital Expenditures"])],
                "dividends_and_other_cash_distributions": [(cashflow, ["Common Stock Dividend", "Dividends Paid", "Cash Dividends Paid"])],
                "issuance_or_purchase_of_equity_shares":  [(cashflow, ["Repurchase Of Capital Stock", "Common Stock Repurchased", "Issuance Of Capital Stock"])],
            }

            DERIVED = {"operating_margin", "gross_margin", "net_margin", "debt_to_equity", "return_on_invested_capital"}

            results: list[LineItem] = []
            for col in columns:
                li = LineItem(
                    ticker=ticker,
                    report_period=col.strftime("%Y-%m-%d"),
                    period="annual",
                    currency=info.get("currency") or "USD",
                    source="yahoo",
                    is_fallback=True,
                    source_chain=["yahoo"],
                )
                missing: list[str] = []

                for item in line_items:
                    if item in DERIVED:
                        continue
                    value = None
                    for frame, labels in field_map.get(item, []):
                        value = self._value_from_frame(frame, col, labels)
                        if value is not None:
                            break
                    # book_value_per_share: divide book_value by shares when raw lookup returns total equity
                    if item == "book_value_per_share" and value is not None:
                        shares = self._value_from_frame(balance, col, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]) or info.get("sharesOutstanding")
                        bv_per_share = self._safe_div(value, shares)
                        if bv_per_share is not None:
                            value = bv_per_share
                    setattr(li, item, value)
                    if value is None:
                        missing.append(item)

                # Raw values needed for derived ratios
                rev = self._value_from_frame(income, col, ["Total Revenue", "Revenue", "Operating Revenue"])
                op_i = self._value_from_frame(income, col, ["Operating Income"])
                gp = self._value_from_frame(income, col, ["Gross Profit"])
                ni = self._value_from_frame(income, col, ["Net Income"])
                debt = self._value_from_frame(balance, col, ["Total Debt", "Debt"])
                eq = self._value_from_frame(balance, col, ["Stockholders Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"])
                ta = self._value_from_frame(balance, col, ["Total Assets"])
                cl = self._value_from_frame(balance, col, ["Current Liabilities", "Total Current Liabilities"])
                invested = (ta - cl) if ta is not None and cl is not None else None

                derived_map = {
                    "operating_margin":           (op_i, rev,      ["operating_income", "revenue"]),
                    "gross_margin":               (gp,   rev,      ["gross_profit", "revenue"]),
                    "net_margin":                 (ni,   rev,      ["net_income", "revenue"]),
                    "debt_to_equity":             (debt, eq,       ["total_debt", "shareholders_equity"]),
                    "return_on_invested_capital": (op_i, invested, ["operating_income", "total_assets", "current_liabilities"]),
                }
                for field_name, (num, den, inputs) in derived_map.items():
                    if field_name in line_items:
                        val = self._safe_div(num, den)
                        setattr(li, field_name, val)
                        li.derived_fields[field_name] = inputs
                        if val is None:
                            missing.append(field_name)

                li.missing_fields = missing
                results.append(li)

            self._cache.set_line_items(cache_key, [r.model_dump() for r in results])
            return results

        except Exception:
            logger.exception("Yahoo search_line_items failed for %s", ticker)
            return []

    def get_market_cap(self, ticker: str, as_of_date: str | None = None, **kwargs) -> float | None:
        if not ticker or yf is None:
            return None

        cache_key = f"yahoo_marketcap_{ticker}_{as_of_date or 'now'}"
        cached = self._cache.get_financial_metrics(cache_key)
        if cached:
            try:
                return float(cached[0].get("market_cap"))
            except Exception:
                pass

        try:
            if as_of_date:
                # Approximate historical market cap: shares * closing price on that date
                info = yf.Ticker(ticker).info or {}
                shares = info.get("sharesOutstanding")
                if shares:
                    hist = self.get_prices(ticker, as_of_date, as_of_date)
                    if hist:
                        mc = float(shares) * hist[0].close
                        self._cache.set_financial_metrics(cache_key, [{"market_cap": mc}])
                        return mc

            info = yf.Ticker(ticker).info or {}
            mc = info.get("marketCap")
            if mc is not None:
                mc = float(mc)
                self._cache.set_financial_metrics(cache_key, [{"market_cap": mc}])
                return mc
        except Exception as e:
            logger.warning("Yahoo market cap fetch failed for %s: %s", ticker, e)

        return None

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int = 10,
        **kwargs,
    ) -> List[CompanyNews]:
        if not ticker or yf is None:
            return []

        cache_key = f"yahoo_news_{ticker}_{start_date}_{end_date}_{limit}"
        cached = self._cache.get_company_news(cache_key)
        if cached is not None:
            return [CompanyNews(**n) for n in cached]

        try:
            tk = self._ticker(ticker)
            raw_news = tk.news or []

            cutoff_start = pd.to_datetime(start_date).timestamp() if start_date else None
            cutoff_end = pd.to_datetime(end_date).timestamp() if end_date else None

            results: list[CompanyNews] = []
            for item in raw_news:
                # yfinance >=0.2.x nests fields under "content"
                content = item.get("content") or item

                title = content.get("title") or item.get("title", "")
                if not title:
                    continue

                url_str = (
                    (content.get("canonicalUrl") or {}).get("url")
                    or (content.get("clickThroughUrl") or {}).get("url")
                    or item.get("link", "")
                )
                publisher = (
                    (content.get("provider") or {}).get("displayName")
                    or item.get("publisher", "yahoo")
                )

                pub_date_str = content.get("pubDate") or content.get("displayTime")
                if pub_date_str:
                    date_str = pub_date_str[:10]
                    pub_ts = pd.to_datetime(pub_date_str).timestamp()
                else:
                    publish_ts = item.get("providerPublishTime")
                    if publish_ts is None:
                        continue
                    pub_ts = float(publish_ts)
                    date_str = datetime.utcfromtimestamp(pub_ts).strftime("%Y-%m-%d")

                if cutoff_start is not None and pub_ts < cutoff_start:
                    continue
                if cutoff_end is not None and pub_ts > cutoff_end:
                    continue

                results.append(CompanyNews(
                    ticker=ticker,
                    title=title,
                    author=None,
                    source=publisher,
                    date=date_str,
                    url=url_str,
                    sentiment=None,
                ))

                if len(results) >= limit:
                    break

            self._cache.set_company_news(cache_key, [n.model_dump() for n in results])
            return results

        except Exception:
            logger.exception("Yahoo get_company_news failed for %s", ticker)
            return []

    def get_insider_trades(self, ticker: str, start_date: str | None, end_date: str, limit: int = 1000, **kwargs):
        return []
