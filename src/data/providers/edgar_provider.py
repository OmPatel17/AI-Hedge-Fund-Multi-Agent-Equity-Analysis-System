"""SEC EDGAR market data provider.

Fetches financial data from the SEC EDGAR API (free, no API key required).
Implements the MarketDataProvider protocol for:
  - get_financial_metrics
  - search_line_items
  - get_insider_trades

All other protocol methods return [] or None.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import List

import requests

from src.data.cache import get_cache
from src.data.models import FinancialMetrics, InsiderTrade, LineItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP headers — SEC enforces a descriptive User-Agent
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": "AI Hedge Fund Research (contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Pause between requests to stay under SEC's 10 req/s rate limit
_REQUEST_DELAY = 0.11

# ---------------------------------------------------------------------------
# XBRL concept → canonical field name mapping
# None means "derived from other fields"
# ---------------------------------------------------------------------------
CONCEPT_MAP: dict[str, list[str] | None] = {
    "revenue":          ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income":       ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit":     ["GrossProfit"],
    "total_assets":     ["Assets"],
    "total_liabilities":["Liabilities"],
    "shareholders_equity":["StockholdersEquity", "StockholdersEquityAttributableToParent"],
    "current_assets":   ["AssetsCurrent"],
    "current_liabilities":["LiabilitiesCurrent"],
    "cash_and_equivalents":["CashAndCashEquivalentsAtCarryingValue"],
    "total_debt":       ["LongTermDebtAndCapitalLeaseObligation", "LongTermDebt"],
    "outstanding_shares":["CommonStockSharesOutstanding"],
    "shares_outstanding":["CommonStockSharesOutstanding"],
    "free_cash_flow":   None,   # derived: operating_cash_flow - |capital_expenditure|
    "operating_cash_flow":["NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditure": [
        "PaymentsToAcquirePropertyPlantAndEquipment",   # most GAAP filers
        "PaymentsToAcquireProductiveAssets",             # NVDA switched to this post-2020
        "PaymentsForCapitalImprovements",                # real-estate/utility filers
    ],
    "depreciation_and_amortization": [
        "DepreciationDepletionAndAmortization",          # most filers
        "DepreciationAndAmortization",                   # NVDA pre-2022 and others
    ],
    "research_and_development":["ResearchAndDevelopmentExpense"],
    # Goodwill + intangibles: merge across three possible XBRL concepts
    "goodwill_and_intangible_assets":["GoodwillAndIntangibleAssets", "Goodwill", "IntangibleAssetsNetExcludingGoodwill"],
    "earnings_per_share":["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "dividends_and_other_cash_distributions":["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "issuance_or_purchase_of_equity_shares":["PaymentsForRepurchaseOfCommonStock"],
    # Valuation / DCF inputs
    "ebit":             None,   # derived: operating_income (EBIT ≈ op income for GAAP filers)
    "ebitda":           None,   # derived: operating_income + depreciation_and_amortization
    "interest_expense": ["InterestExpense", "InterestAndDebtExpense", "InterestExpenseDebt"],
    "working_capital":  None,   # derived: current_assets - current_liabilities
    "operating_expense": ["OperatingExpenses", "CostsAndExpenses"],
    # Derived ratios
    "book_value_per_share": None,
    "operating_margin": None,
    "gross_margin":     None,
    "net_margin":       None,
    "debt_to_equity":   None,
    "return_on_invested_capital": None,
}

# Share-count concepts use "shares" units, not "USD"
_SHARE_CONCEPTS = {"CommonStockSharesOutstanding"}

# EPS concepts are dimensionless (USD/shares)
_EPS_CONCEPTS = {"EarningsPerShareDiluted", "EarningsPerShareBasic"}


class EdgarProvider:
    """Market data provider backed by SEC EDGAR (no API key required)."""

    # Class-level caches so all instances share them
    _cik_map: dict[str, str] = {}          # ticker (upper) -> zero-padded CIK string
    _facts_cache: dict[str, dict] = {}     # ticker (upper) -> raw companyfacts JSON

    def __init__(self) -> None:
        self._cache = get_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str) -> requests.Response | None:
        """HTTP GET with required headers and rate-limit delay."""
        time.sleep(_REQUEST_DELAY)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            return resp
        except Exception as exc:
            logger.warning("EDGAR request failed for %s: %s", url, exc)
            return None

    def _load_cik_map(self) -> bool:
        """Download and cache the ticker->CIK mapping from SEC."""
        if EdgarProvider._cik_map:
            return True
        resp = self._get("https://www.sec.gov/files/company_tickers.json")
        if resp is None or resp.status_code != 200:
            logger.warning(
                "EDGAR: failed to fetch company_tickers.json (status %s)",
                getattr(resp, "status_code", "N/A"),
            )
            return False
        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("EDGAR: could not parse company_tickers.json: %s", exc)
            return False
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik_int = entry.get("cik_str", 0)
            if ticker and cik_int:
                EdgarProvider._cik_map[ticker] = str(cik_int).zfill(10)
        return bool(EdgarProvider._cik_map)

    def _get_cik(self, ticker: str) -> str | None:
        """Return zero-padded 10-digit CIK for *ticker*, or None."""
        ticker = ticker.upper()
        if ticker not in EdgarProvider._cik_map:
            if not self._load_cik_map():
                return None
        cik = EdgarProvider._cik_map.get(ticker)
        if not cik:
            logger.warning("EDGAR: CIK not found for ticker %s", ticker)
        return cik

    def _get_facts(self, ticker: str) -> dict | None:
        """Fetch (and cache) the full XBRL companyfacts JSON for *ticker*."""
        ticker = ticker.upper()
        if ticker in EdgarProvider._facts_cache:
            return EdgarProvider._facts_cache[ticker]

        cik = self._get_cik(ticker)
        if not cik:
            return None

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = self._get(url)
        if resp is None or resp.status_code != 200:
            logger.warning(
                "EDGAR: failed to fetch companyfacts for %s (CIK %s, status %s)",
                ticker, cik, getattr(resp, "status_code", "N/A"),
            )
            return None
        try:
            facts = resp.json()
        except Exception as exc:
            logger.warning("EDGAR: could not parse companyfacts for %s: %s", ticker, exc)
            return None

        EdgarProvider._facts_cache[ticker] = facts
        return facts

    # ------------------------------------------------------------------
    # XBRL extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _allowed_forms(period: str) -> set[str]:
        if period == "annual":
            return {"10-K"}
        # quarterly / ttm: accept both
        return {"10-K", "10-Q"}

    def _extract_series(
        self,
        facts: dict,
        concept_tags: list[str],
        end_date: str,
        period: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Return [(end_date, value), ...] merged across all matching tags.

        Collects qualifying entries from every tag in concept_tags, then
        de-duplicates by end date (latest filed wins across all tags), sorts
        descending, and caps at limit. This handles cases where a GAAP concept
        was renamed mid-history (e.g. Revenues → RevenueFromContract...).
        """
        allowed = self._allowed_forms(period)
        us_gaap: dict = facts.get("facts", {}).get("us-gaap", {})

        # Accumulate best entry per end-date across all tags
        best: dict[str, dict] = {}

        for tag in concept_tags:
            concept_data = us_gaap.get(tag)
            if not concept_data:
                continue

            # Determine which units key to use
            if tag in _SHARE_CONCEPTS:
                unit_key = "shares"
            elif tag in _EPS_CONCEPTS:
                unit_key = next(
                    (k for k in concept_data.get("units", {}) if "/" in k),
                    "USD/shares",
                )
            else:
                unit_key = "USD"

            entries = concept_data.get("units", {}).get(unit_key, [])
            if not entries:
                for k, v in concept_data.get("units", {}).items():
                    if v:
                        entries = v
                        break

            if not entries:
                continue

            for e in entries:
                form = e.get("form", "")
                fp = e.get("fp", "")
                e_end = e.get("end", "")
                if not e_end or e_end > end_date:
                    continue
                if period == "annual":
                    # Require a 10-K filing form
                    if form not in allowed:
                        continue
                    # Within 10-K, skip quarterly-period entries (fp Q1/Q2/Q3/Q4).
                    # Empty fp is allowed (balance-sheet items in some 10-K filings).
                    if fp and fp not in ("FY", "CY"):
                        continue
                    # Duration guard: income statement items have a "start" date.
                    # Some 10-K filings tag quarterly comparison periods as fp="FY".
                    # Reject any entry whose covered period is less than ~11 months
                    # (= quarterly or shorter), even if labeled FY.
                    e_start = e.get("start", "")
                    if e_start:
                        try:
                            from datetime import date as _date
                            d0 = _date.fromisoformat(e_start)
                            d1 = _date.fromisoformat(e_end)
                            if (d1 - d0).days < 330:
                                continue
                        except ValueError:
                            pass
                else:
                    if form not in allowed:
                        continue

                filed = e.get("filed", "")
                if e_end not in best or filed > best[e_end].get("filed", ""):
                    best[e_end] = e

        if not best:
            return []

        sorted_entries = sorted(best.values(), key=lambda x: x["end"], reverse=True)[:limit]
        result = []
        for e in sorted_entries:
            try:
                result.append((e["end"], float(e["val"])))
            except (KeyError, TypeError, ValueError):
                pass
        return result

    # ------------------------------------------------------------------
    # Public protocol methods
    # ------------------------------------------------------------------

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "annual",
        limit: int = 10,
        **kwargs,
    ) -> List[FinancialMetrics]:
        """Return list of FinancialMetrics built from EDGAR XBRL data."""
        cache_key = f"edgar_metrics_{ticker}_{end_date}_{period}_{limit}"
        cached = self._cache.get_financial_metrics(cache_key)
        if cached:
            return [FinancialMetrics(**m) for m in cached]

        try:
            facts = self._get_facts(ticker)
            if not facts:
                return []

            def series(tags: list[str] | None) -> list[tuple[str, float]]:
                if not tags:
                    return []
                return self._extract_series(facts, tags, end_date, period, limit)

            # Fetch raw series
            rev_series         = series(CONCEPT_MAP["revenue"])
            ni_series          = series(CONCEPT_MAP["net_income"])
            op_series          = series(CONCEPT_MAP["operating_income"])
            gp_series          = series(CONCEPT_MAP["gross_profit"])
            assets_series      = series(CONCEPT_MAP["total_assets"])
            liab_series        = series(CONCEPT_MAP["total_liabilities"])
            eq_series          = series(CONCEPT_MAP["shareholders_equity"])
            cur_assets_series  = series(CONCEPT_MAP["current_assets"])
            cur_liab_series    = series(CONCEPT_MAP["current_liabilities"])
            cash_series        = series(CONCEPT_MAP["cash_and_equivalents"])
            debt_series        = series(CONCEPT_MAP["total_debt"])
            shares_series      = series(CONCEPT_MAP["shares_outstanding"])
            ocf_series         = series(CONCEPT_MAP["operating_cash_flow"])
            capex_series       = series(CONCEPT_MAP["capital_expenditure"])
            eps_series         = series(CONCEPT_MAP["earnings_per_share"])
            da_series          = series(CONCEPT_MAP["depreciation_and_amortization"])
            # Fallback for filers (e.g. MSFT) that split D&A into separate tags
            if not da_series:
                _dep = series(["Depreciation"])
                _amort = series(["AmortizationOfIntangibleAssets"])
                _dep_d = {d: v for d, v in _dep}
                _amort_d = {d: v for d, v in _amort}
                da_series = [(d, (_dep_d.get(d) or 0) + (_amort_d.get(d) or 0))
                             for d in set(_dep_d) | set(_amort_d)]
            int_exp_series     = series(CONCEPT_MAP["interest_expense"])

            # Collect all end dates that appear in the primary series only.
            # Supplemental series (da, int_exp) are not used to drive dates — they
            # only fill values for dates already established by primary fields.
            all_dates: set[str] = set()
            for s in [rev_series, ni_series, op_series, gp_series, assets_series,
                      liab_series, eq_series, cur_assets_series, cur_liab_series,
                      cash_series, debt_series, shares_series, ocf_series,
                      capex_series, eps_series]:
                all_dates.update(d for d, _ in s)

            if not all_dates:
                return []

            def _dict(s: list[tuple[str, float]]) -> dict[str, float]:
                return {d: v for d, v in s}

            rev_d        = _dict(rev_series)
            ni_d         = _dict(ni_series)
            op_d         = _dict(op_series)
            gp_d         = _dict(gp_series)
            assets_d     = _dict(assets_series)
            liab_d       = _dict(liab_series)
            eq_d         = _dict(eq_series)
            cur_assets_d = _dict(cur_assets_series)
            cur_liab_d   = _dict(cur_liab_series)
            cash_d       = _dict(cash_series)
            debt_d       = _dict(debt_series)
            shares_d     = _dict(shares_series)
            ocf_d        = _dict(ocf_series)
            capex_d      = _dict(capex_series)
            eps_d        = _dict(eps_series)
            da_d         = _dict(da_series)
            int_exp_d    = _dict(int_exp_series)

            def _safe_div(a: float | None, b: float | None) -> float | None:
                if a is None or b is None or b == 0:
                    return None
                return a / b

            results: list[FinancialMetrics] = []
            previous_revenue: float | None = None

            for date in sorted(all_dates, reverse=True)[:limit]:
                rev       = rev_d.get(date)
                ni        = ni_d.get(date)
                op_i      = op_d.get(date)
                gp        = gp_d.get(date)
                ta        = assets_d.get(date)
                tl        = liab_d.get(date)
                eq        = eq_d.get(date)
                ca        = cur_assets_d.get(date)
                cl        = cur_liab_d.get(date)
                cash      = cash_d.get(date)
                debt      = debt_d.get(date)
                shr       = shares_d.get(date)
                ocf       = ocf_d.get(date)
                capex     = capex_d.get(date)
                eps       = eps_d.get(date)
                da        = da_d.get(date)
                int_exp   = int_exp_d.get(date)

                # Derived
                fcf              = (ocf - abs(capex)) if ocf is not None and capex is not None else None
                gross_margin     = _safe_div(gp, rev)
                operating_margin = _safe_div(op_i, rev)
                net_margin       = _safe_div(ni, rev)
                roe              = _safe_div(ni, eq)
                roa              = _safe_div(ni, ta)
                debt_to_equity   = _safe_div(debt, eq)
                debt_to_assets   = _safe_div(debt, ta)
                current_ratio    = _safe_div(ca, cl)
                invested_capital = (ta - cl) if ta is not None and cl is not None else None
                roic             = _safe_div(op_i, invested_capital)
                bvps             = _safe_div(eq, shr)
                fcfps            = _safe_div(fcf, shr)
                # interest_coverage = EBIT / interest_expense
                interest_cov = (
                    _safe_div(op_i, abs(int_exp))
                    if op_i is not None and int_exp is not None and int_exp != 0
                    else None
                )
                revenue_growth   = (
                    (rev - previous_revenue) / abs(previous_revenue)
                    if previous_revenue not in (None, 0) and rev is not None
                    else None
                )

                fm = FinancialMetrics(
                    ticker=ticker,
                    report_period=date,
                    period=period,
                    currency="USD",
                    market_cap=None,
                    enterprise_value=None,
                    price_to_earnings_ratio=None,
                    price_to_book_ratio=None,
                    price_to_sales_ratio=None,
                    enterprise_value_to_ebitda_ratio=None,
                    enterprise_value_to_revenue_ratio=None,
                    free_cash_flow_yield=None,
                    peg_ratio=None,
                    gross_margin=gross_margin,
                    operating_margin=operating_margin,
                    net_margin=net_margin,
                    return_on_equity=roe,
                    return_on_assets=roa,
                    return_on_invested_capital=roic,
                    asset_turnover=_safe_div(rev, ta),
                    inventory_turnover=None,
                    receivables_turnover=None,
                    days_sales_outstanding=None,
                    operating_cycle=None,
                    working_capital_turnover=None,
                    current_ratio=current_ratio,
                    quick_ratio=None,
                    cash_ratio=_safe_div(cash, cl),
                    operating_cash_flow_ratio=_safe_div(ocf, cl),
                    debt_to_equity=debt_to_equity,
                    debt_to_assets=debt_to_assets,
                    interest_coverage=interest_cov,
                    revenue_growth=revenue_growth,
                    earnings_growth=None,
                    book_value_growth=None,
                    earnings_per_share_growth=None,
                    free_cash_flow_growth=None,
                    operating_income_growth=None,
                    ebitda_growth=None,
                    payout_ratio=None,
                    earnings_per_share=eps,
                    book_value_per_share=bvps,
                    free_cash_flow_per_share=fcfps,
                )
                results.append(fm)
                previous_revenue = rev

            if results:
                self._cache.set_financial_metrics(cache_key, [r.model_dump() for r in results])
            return results

        except Exception as exc:
            logger.warning("EDGAR get_financial_metrics failed for %s: %s", ticker, exc)
            return []

    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "annual",
        limit: int = 10,
        **kwargs,
    ) -> List[LineItem]:
        """Return LineItem list aligned by reporting period from EDGAR XBRL."""
        items_key = "_".join(sorted(line_items))
        cache_key = f"edgar_{ticker}_{end_date}_{period}_{limit}_{items_key}"
        cached = self._cache.get_line_items(cache_key)
        if cached:
            return [LineItem(**li) for li in cached]

        try:
            facts = self._get_facts(ticker)
            if not facts:
                return []

            # Fields that are purely derived (no direct XBRL tag)
            DERIVED = {
                "free_cash_flow",
                "book_value_per_share",
                "operating_margin",
                "gross_margin",
                "net_margin",
                "debt_to_equity",
                "return_on_invested_capital",
                "ebit",          # EBIT ≈ operating_income for GAAP filers
                "ebitda",        # operating_income + depreciation_and_amortization
                "working_capital",  # current_assets - current_liabilities
            }

            # Extract raw series for every requested field that has XBRL tags
            field_series: dict[str, list[tuple[str, float]]] = {}
            for field in line_items:
                if field in DERIVED:
                    continue
                tags = CONCEPT_MAP.get(field)
                if tags is None:
                    continue
                s = self._extract_series(facts, tags, end_date, period, limit)
                if s:
                    field_series[field] = s

            # Always fetch these helper fields for derived calculations
            _helper_fields: dict[str, list[str] | None] = {
                "operating_cash_flow":           CONCEPT_MAP["operating_cash_flow"],
                "capital_expenditure":           CONCEPT_MAP["capital_expenditure"],
                "shareholders_equity":           CONCEPT_MAP["shareholders_equity"],
                "shares_outstanding":            CONCEPT_MAP["shares_outstanding"],
                "revenue":                       CONCEPT_MAP["revenue"],
                "operating_income":              CONCEPT_MAP["operating_income"],
                "gross_profit":                  CONCEPT_MAP["gross_profit"],
                "net_income":                    CONCEPT_MAP["net_income"],
                "total_debt":                    CONCEPT_MAP["total_debt"],
                "total_assets":                  CONCEPT_MAP["total_assets"],
                "current_liabilities":           CONCEPT_MAP["current_liabilities"],
                "current_assets":                CONCEPT_MAP["current_assets"],
                "depreciation_and_amortization": CONCEPT_MAP["depreciation_and_amortization"],
            }
            helper_data: dict[str, dict[str, float]] = {}
            for hf, tags in _helper_fields.items():
                if tags and hf not in field_series:
                    s = self._extract_series(facts, tags, end_date, period, limit)
                    helper_data[hf] = {d: v for d, v in s}
                elif hf in field_series:
                    helper_data[hf] = {d: v for d, v in field_series[hf]}

            # MSFT and some filers don't report a combined D&A tag — they split it into
            # Depreciation + AmortizationOfIntangibleAssets.  Sum them as a fallback.
            if not helper_data.get("depreciation_and_amortization"):
                dep_s = self._extract_series(facts, ["Depreciation"], end_date, period, limit)
                amort_s = self._extract_series(facts, ["AmortizationOfIntangibleAssets"], end_date, period, limit)
                dep_d_fb = {d: v for d, v in dep_s}
                amort_d_fb = {d: v for d, v in amort_s}
                combined_dates = set(dep_d_fb) | set(amort_d_fb)
                if combined_dates:
                    helper_data["depreciation_and_amortization"] = {
                        d: (dep_d_fb.get(d) or 0) + (amort_d_fb.get(d) or 0)
                        for d in combined_dates
                    }

            # Gather all end dates
            all_dates: set[str] = set()
            for s in field_series.values():
                all_dates.update(d for d, _ in s)
            for d_map in helper_data.values():
                all_dates.update(d_map.keys())

            if not all_dates:
                return []

            def _safe_div(a: float | None, b: float | None) -> float | None:
                if a is None or b is None or b == 0:
                    return None
                return a / b

            # Convert series lists to dicts keyed by date
            field_dicts: dict[str, dict[str, float]] = {
                f: {d: v for d, v in s} for f, s in field_series.items()
            }

            results: list[LineItem] = []
            for date in sorted(all_dates, reverse=True)[:limit]:
                missing: list[str] = []
                derived_provenance: dict[str, list[str]] = {}

                li = LineItem(
                    ticker=ticker,
                    report_period=date,
                    period=period,
                    currency="USD",
                    source="edgar",
                    is_fallback=False,
                    source_chain=["edgar"],
                )

                # Set raw fields
                for field in line_items:
                    if field in DERIVED:
                        continue
                    val = field_dicts.get(field, {}).get(date)
                    # helper_data may hold a fallback value (e.g. D&A sum for MSFT)
                    if val is None:
                        val = helper_data.get(field, {}).get(date)
                    setattr(li, field, val)
                    if val is None:
                        missing.append(field)

                # Compute derived fields if requested
                ocf   = helper_data.get("operating_cash_flow", {}).get(date)
                capex = helper_data.get("capital_expenditure", {}).get(date)
                eq    = helper_data.get("shareholders_equity", {}).get(date)
                shr   = helper_data.get("shares_outstanding", {}).get(date)
                rev   = helper_data.get("revenue", {}).get(date)
                op_i  = helper_data.get("operating_income", {}).get(date)
                gp    = helper_data.get("gross_profit", {}).get(date)
                ni    = helper_data.get("net_income", {}).get(date)
                debt  = helper_data.get("total_debt", {}).get(date)
                ta    = helper_data.get("total_assets", {}).get(date)
                cl    = helper_data.get("current_liabilities", {}).get(date)
                ca    = helper_data.get("current_assets", {}).get(date)
                da    = helper_data.get("depreciation_and_amortization", {}).get(date)

                invested_capital = (ta - cl) if ta is not None and cl is not None else None

                derived_map: dict[str, tuple[float | None, list[str]]] = {
                    "free_cash_flow": (
                        (ocf - abs(capex)) if ocf is not None and capex is not None else None,
                        ["operating_cash_flow", "capital_expenditure"],
                    ),
                    "book_value_per_share": (
                        _safe_div(eq, shr),
                        ["shareholders_equity", "shares_outstanding"],
                    ),
                    "operating_margin":           (_safe_div(op_i, rev), ["operating_income", "revenue"]),
                    "gross_margin":               (_safe_div(gp, rev),   ["gross_profit", "revenue"]),
                    "net_margin":                 (_safe_div(ni, rev),   ["net_income", "revenue"]),
                    "debt_to_equity":             (_safe_div(debt, eq),  ["total_debt", "shareholders_equity"]),
                    "return_on_invested_capital": (
                        _safe_div(op_i, invested_capital),
                        ["operating_income", "total_assets", "current_liabilities"],
                    ),
                    # EBIT ≈ operating income for GAAP reporters (interest/taxes are below op-income line)
                    "ebit": (op_i, ["operating_income"]),
                    # EBITDA = EBIT + D&A; fall back to just op_i if D&A unavailable
                    "ebitda": (
                        (op_i + abs(da)) if op_i is not None and da is not None else op_i,
                        ["operating_income", "depreciation_and_amortization"],
                    ),
                    "working_capital": (
                        (ca - cl) if ca is not None and cl is not None else None,
                        ["current_assets", "current_liabilities"],
                    ),
                }

                for field_name, (val, inputs) in derived_map.items():
                    if field_name in line_items:
                        setattr(li, field_name, val)
                        derived_provenance[field_name] = inputs
                        if val is None:
                            missing.append(field_name)

                li.missing_fields = missing
                li.derived_fields = derived_provenance
                results.append(li)

            if results:
                self._cache.set_line_items(cache_key, [r.model_dump() for r in results])
            return results

        except Exception as exc:
            logger.warning("EDGAR search_line_items failed for %s: %s", ticker, exc)
            return []

    def _extract_form4s_from_page(
        self,
        page: dict,
        ticker: str,
        start_date: str | None,
        end_date: str,
    ) -> list[dict]:
        """Return Form 4 filing dicts from a submissions page (recent or paginated)."""
        forms        = page.get("form", [])
        filing_dates = page.get("filingDate", [])
        accession_nos = page.get("accessionNumber", [])
        primary_docs  = page.get("primaryDocument", [])
        result: list[dict] = []
        for form, filing_date, accn, primary_doc in zip(
            forms, filing_dates, accession_nos, primary_docs
        ):
            if form != "4":
                continue
            if start_date and filing_date < start_date:
                continue
            if filing_date > end_date:
                continue
            result.append({
                "ticker":      ticker,
                "filing_date": filing_date,
                "accn":        accn,
                "primary_doc": primary_doc,
            })
        return result

    def get_insider_trades(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int = 1000,
        **kwargs,
    ) -> List[InsiderTrade]:
        """Fetch Form 4 insider trades from EDGAR for *ticker*.

        Handles pagination: large companies (META, JPM) have thousands of filings
        and the most-recent bucket may not reach the requested date range.
        """
        cache_key = f"edgar_insider_{ticker}_{start_date}_{end_date}_{limit}"
        cached = self._cache.get_insider_trades(cache_key)
        if cached:
            return [InsiderTrade(**t) for t in cached]

        try:
            cik = self._get_cik(ticker)
            if not cik:
                return []

            # Fetch primary submissions index
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = self._get(url)
            if resp is None or resp.status_code != 200:
                logger.warning(
                    "EDGAR: submissions fetch failed for %s (status %s)",
                    ticker, getattr(resp, "status_code", "N/A"),
                )
                return []

            try:
                submissions = resp.json()
            except Exception as exc:
                logger.warning("EDGAR: could not parse submissions for %s: %s", ticker, exc)
                return []

            recent = submissions.get("filings", {}).get("recent", {})
            form4_filings = self._extract_form4s_from_page(recent, ticker, start_date, end_date)

            # Determine whether pagination is needed.
            # If start_date falls before the oldest date in `recent`, there may be
            # additional Form 4s in the archive pages.
            recent_dates = [d for d in recent.get("filingDate", []) if d]
            oldest_recent = min(recent_dates) if recent_dates else None
            needs_more = start_date and oldest_recent and oldest_recent > start_date

            if needs_more:
                extra_files = submissions.get("filings", {}).get("files", [])
                for file_info in extra_files:
                    name = file_info.get("name", "")
                    if not name:
                        continue
                    file_url = f"https://data.sec.gov/submissions/{name}"
                    file_resp = self._get(file_url)
                    if file_resp is None or file_resp.status_code != 200:
                        continue
                    try:
                        page = file_resp.json()
                    except Exception:
                        continue
                    extra = self._extract_form4s_from_page(page, ticker, start_date, end_date)
                    form4_filings.extend(extra)
                    # Stop once this page goes past the start_date window
                    page_dates = [d for d in page.get("filingDate", []) if d]
                    if page_dates and min(page_dates) < (start_date or ""):
                        break

            trades: list[InsiderTrade] = []
            for filing in form4_filings:
                if len(trades) >= limit:
                    break
                parsed = self._parse_form4(cik, filing)
                trades.extend(parsed)

            trades = trades[:limit]
            if trades:
                self._cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
            return trades

        except Exception as exc:
            logger.warning("EDGAR get_insider_trades failed for %s: %s", ticker, exc)
            return []

    def _parse_form4(self, cik: str, filing: dict) -> list[InsiderTrade]:
        """Download and parse a single Form 4 XML filing."""
        accn_no_dashes = filing["accn"].replace("-", "")
        primary_doc    = filing["primary_doc"]
        filing_date    = filing["filing_date"]
        ticker         = filing.get("ticker", "").upper()

        # SEC submissions may return the XSLT viewer path (xslF345X05/filename.xml).
        # Strip the viewer prefix to get the raw XML path.
        raw_doc = primary_doc.split("/")[-1] if "/" in primary_doc else primary_doc
        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accn_no_dashes}/{raw_doc}"
        )
        resp = self._get(url)
        if resp is None or resp.status_code != 200:
            logger.warning(
                "EDGAR: Form 4 fetch failed %s (status %s)",
                url, getattr(resp, "status_code", "N/A"),
            )
            return []

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            logger.warning("EDGAR: Form 4 XML parse error %s: %s", url, exc)
            return []

        def _text(element, *path: str) -> str | None:
            """Walk an element path and return stripped text or None."""
            node = element
            for tag in path:
                if node is None:
                    return None
                node = node.find(tag)
            return node.text.strip() if node is not None and node.text else None

        # Issuer info
        issuer_node = root.find("issuer")
        issuer_name = _text(issuer_node, "issuerName") if issuer_node is not None else None

        # Reporting owner (take first owner)
        owner_node = root.find("reportingOwner")
        owner_name: str | None = None
        owner_title: str | None = None
        is_board_director: bool | None = None

        if owner_node is not None:
            owner_id = owner_node.find("reportingOwnerId")
            if owner_id is not None:
                owner_name = _text(owner_id, "rptOwnerName")

            rel_node = owner_node.find("reportingOwnerRelationship")
            if rel_node is not None:
                owner_title        = _text(rel_node, "officerTitle")
                is_director_text   = _text(rel_node, "isDirector")
                is_board_director  = (is_director_text == "1") if is_director_text is not None else None

        # Parse non-derivative transactions
        trades: list[InsiderTrade] = []
        non_deriv_table = root.find("nonDerivativeTable")
        if non_deriv_table is None:
            return trades

        for txn_node in non_deriv_table.findall("nonDerivativeTransaction"):
            try:
                security_title = _text(txn_node, "securityTitle", "value")
                txn_date       = _text(txn_node, "transactionDate", "value")

                txn_amounts = txn_node.find("transactionAmounts")
                shares_text = _text(txn_amounts, "transactionShares", "value") if txn_amounts is not None else None
                price_text  = _text(txn_amounts, "transactionPricePerShare", "value") if txn_amounts is not None else None
                code_text   = _text(txn_amounts, "transactionAcquiredDisposedCode", "value") if txn_amounts is not None else None

                post_amounts      = txn_node.find("postTransactionAmounts")
                shares_after_text = _text(post_amounts, "sharesOwnedFollowingTransaction", "value") if post_amounts is not None else None

                txn_shares:   float | None = float(shares_text)      if shares_text      else None
                txn_price:    float | None = float(price_text)       if price_text       else None
                shares_after: float | None = float(shares_after_text) if shares_after_text else None

                # Disposed = negative shares; Acquired = positive
                if txn_shares is not None:
                    txn_shares = -abs(txn_shares) if code_text == "D" else abs(txn_shares)

                txn_value = (
                    abs(txn_shares) * txn_price
                    if txn_shares is not None and txn_price is not None
                    else None
                )
                shares_before = (
                    shares_after - txn_shares
                    if shares_after is not None and txn_shares is not None
                    else None
                )

                trade = InsiderTrade(
                    ticker=ticker,
                    issuer=issuer_name,
                    name=owner_name,
                    title=owner_title,
                    is_board_director=is_board_director,
                    transaction_date=txn_date,
                    transaction_shares=txn_shares,
                    transaction_price_per_share=txn_price,
                    transaction_value=txn_value,
                    shares_owned_before_transaction=shares_before,
                    shares_owned_after_transaction=shares_after,
                    security_title=security_title,
                    filing_date=filing_date,
                )
                trades.append(trade)
            except Exception as exc:
                logger.warning("EDGAR: error parsing Form 4 transaction: %s", exc)
                continue

        return trades

    # ------------------------------------------------------------------
    # Unsupported protocol methods — return safe empty values
    # ------------------------------------------------------------------

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs):
        return []

    def get_company_news(self, ticker: str, start_date: str | None, end_date: str, limit: int = 1000, **kwargs):
        return []

    def get_market_cap(self, ticker: str, as_of_date: str | None = None, **kwargs) -> float | None:
        return None
