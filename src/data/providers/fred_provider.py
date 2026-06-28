"""FRED (Federal Reserve Economic Data) macro-economic data provider.

Fetches macro-economic time series from the St. Louis Fed's public REST API.
A free API key is required and should be set via the FRED_API_KEY environment
variable (obtainable at https://fred.stlouisfed.org/docs/api/api_key.html).

If the key is absent the provider logs a warning and returns empty results rather
than raising an exception.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

_FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_ISO_DATE_FMT = "%Y-%m-%d"

# Canonical series identifiers used by get_macro_snapshot
SERIES: dict[str, str] = {
    "fed_funds_rate":     "FEDFUNDS",    # Monthly
    "cpi":                "CPIAUCSL",    # Monthly
    "unemployment":       "UNRATE",      # Monthly
    "ten_year_treasury":  "DGS10",       # Daily
    "gdp":                "GDP",         # Quarterly
    "breakeven_inflation": "T10YIE",     # Daily
    "vix":                "VIXCLS",      # Daily
}


class FREDProvider:
    """Fetches macro-economic data from the FRED REST API.

    Parameters
    ----------
    api_key:
        FRED API key.  Falls back to the FRED_API_KEY environment variable when
        not provided explicitly.  If neither is available, all methods return
        empty results and log a warning.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key: str | None = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            logger.warning(
                "FREDProvider: FRED_API_KEY is not set.  "
                "Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
                "and set it as the FRED_API_KEY environment variable."
            )
        # Simple in-memory cache: maps cache_key -> list[dict]
        self._cache: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch observations for a FRED series between two dates.

        Parameters
        ----------
        series_id:
            FRED series identifier (e.g. ``"FEDFUNDS"``).
        start_date:
            ISO date string YYYY-MM-DD (inclusive).
        end_date:
            ISO date string YYYY-MM-DD (inclusive).

        Returns
        -------
        List of ``{"date": "YYYY-MM-DD", "value": float}`` dicts sorted
        descending by date.  Returns an empty list on any error, missing key,
        or if no data is available.
        """
        if not self.api_key:
            logger.warning(
                "FREDProvider.get_series: FRED_API_KEY not set; returning empty for %s",
                series_id,
            )
            return []

        cache_key = f"fred_{series_id}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "desc",
            "limit": 100,
        }

        try:
            response = requests.get(_FRED_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            logger.warning(
                "FREDProvider.get_series: request timed out for series_id=%s", series_id
            )
            return []
        except Exception as exc:
            logger.warning(
                "FREDProvider.get_series: request failed for series_id=%s — %s",
                series_id,
                exc,
            )
            return []

        observations = payload.get("observations") or []
        if not isinstance(observations, list):
            logger.warning(
                "FREDProvider.get_series: unexpected response for %s", series_id
            )
            return []

        results: list[dict] = []
        for obs in observations:
            try:
                raw_value = obs.get("value", ".")
                # FRED uses "." to denote missing values — skip those
                if raw_value == "." or raw_value is None:
                    continue
                results.append(
                    {
                        "date": obs["date"],
                        "value": float(raw_value),
                    }
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug(
                    "FREDProvider.get_series: skipping malformed observation — %s", exc
                )
                continue

        self._cache[cache_key] = results
        return results

    def get_macro_snapshot(self, end_date: str) -> dict[str, float | None]:
        """Return the most-recent value of each key macro series as of *end_date*.

        Uses a lookback window of up to 120 days to find the latest available
        observation for each series (accommodating quarterly GDP etc.).

        Parameters
        ----------
        end_date:
            ISO date string YYYY-MM-DD.  Only observations on or before this
            date are considered.

        Returns
        -------
        Dict mapping canonical series names (e.g. ``"fed_funds_rate"``) to
        their most-recent ``float`` value, or ``None`` if no data was found or
        the fetch failed.

        Example
        -------
        ``{"fed_funds_rate": 5.33, "cpi": 314.2, "unemployment": 3.9, ...}``
        """
        try:
            end_dt = datetime.strptime(end_date, _ISO_DATE_FMT)
        except Exception:
            logger.warning(
                "FREDProvider.get_macro_snapshot: invalid end_date '%s'", end_date
            )
            return {name: None for name in SERIES}

        # Use a generous lookback to capture quarterly series (GDP)
        from datetime import timedelta
        start_dt = end_dt - timedelta(days=120)
        start_date = start_dt.strftime(_ISO_DATE_FMT)

        snapshot: dict[str, float | None] = {}
        for name, series_id in SERIES.items():
            try:
                observations = self.get_series(series_id, start_date, end_date)
                # Results are sorted descending; first entry is most recent
                snapshot[name] = observations[0]["value"] if observations else None
            except Exception as exc:
                logger.warning(
                    "FREDProvider.get_macro_snapshot: failed for %s (%s) — %s",
                    name,
                    series_id,
                    exc,
                )
                snapshot[name] = None

        return snapshot
