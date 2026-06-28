"""
Data quality scoring and pre-agent validation.

Scores completeness of financial inputs per ticker and logs a summary
before LLM analysis runs — without blocking execution or modifying agents.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Status = Literal["complete", "partial", "missing"]

# Minimum periods considered "complete" for each data type
_PRICE_COMPLETE_THRESHOLD = 10
_FUNDAMENTALS_COMPLETE_THRESHOLD = 3


@dataclass
class DataQualityReport:
    ticker: str
    price_data: Status = "missing"
    fundamentals: Status = "missing"
    news: Status = "missing"
    insider_trades: Status = "missing"
    fallback_used: bool = False
    sources_used: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    derived_fields: dict[str, list[str]] = field(default_factory=dict)  # field -> inputs

    def log(self) -> None:
        parts = [
            f"[DataQuality/{self.ticker}]",
            f"prices={self.price_data}",
            f"fundamentals={self.fundamentals}",
            f"news={self.news}",
            f"insider_trades={self.insider_trades}",
            f"fallback={self.fallback_used}",
            f"sources={self.sources_used or ['unknown']}",
        ]
        if self.missing_fields:
            parts.append(f"missing={sorted(set(self.missing_fields))}")
        if self.derived_fields:
            parts.append(f"derived={sorted(self.derived_fields)}")
        logger.info(" ".join(parts))

    def any_missing(self) -> bool:
        return any(s == "missing" for s in [self.price_data, self.fundamentals])


def _collect_source(li, report: DataQualityReport) -> None:
    src = getattr(li, "source", None)
    if src and src not in report.sources_used:
        report.sources_used.append(src)
    if getattr(li, "is_fallback", False):
        report.fallback_used = True
    for f_name in getattr(li, "missing_fields", []):
        if f_name not in report.missing_fields:
            report.missing_fields.append(f_name)
    for f_name, inputs in getattr(li, "derived_fields", {}).items():
        report.derived_fields.setdefault(f_name, inputs)


def score_data_quality(
    ticker: str,
    prices: list,
    metrics: list,
    line_items: list,
    news: list,
    insider_trades: list,
) -> DataQualityReport:
    """
    Score the completeness of data for a single ticker and return a report.
    Logs the result at INFO level so it appears before any LLM call that uses this data.
    """
    report = DataQualityReport(ticker=ticker)

    # Price data
    n = len(prices)
    if n >= _PRICE_COMPLETE_THRESHOLD:
        report.price_data = "complete"
    elif n > 0:
        report.price_data = "partial"

    # Fundamentals — metrics + line items together
    has_metrics = bool(metrics)
    has_items = bool(line_items)
    if has_metrics and has_items:
        n_items = len(line_items)
        report.fundamentals = "complete" if n_items >= _FUNDAMENTALS_COMPLETE_THRESHOLD else "partial"
    elif has_metrics or has_items:
        report.fundamentals = "partial"

    # News
    report.news = "complete" if news else "missing"

    # Insider trades
    report.insider_trades = "complete" if insider_trades else "missing"

    # Aggregate provenance from line items
    for li in line_items:
        _collect_source(li, report)

    report.log()
    return report


def validate_required_fields(
    ticker: str,
    analyst_type: str,
    line_items: list,
    required: list[str],
) -> list[str]:
    """
    Check that all required fields for a given analyst type are populated.
    Logs a WARNING per missing field — does not block execution.
    Returns list of field names that are absent across ALL periods.
    """
    if not line_items:
        logger.warning(
            "[PreAgent/%s/%s] No line item data available — analyst will work with empty inputs",
            analyst_type, ticker,
        )
        return required

    # A field is "present" if at least one period has a non-None value
    present: set[str] = set()
    for li in line_items:
        for f_name in required:
            val = getattr(li, f_name, None)
            if val is not None:
                present.add(f_name)

    absent = [f for f in required if f not in present]
    if absent:
        logger.warning(
            "[PreAgent/%s/%s] Required fields absent across all periods: %s",
            analyst_type, ticker, absent,
        )
    return absent
