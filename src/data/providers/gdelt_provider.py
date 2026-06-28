"""GDELT (Global Database of Events, Language, and Tone) news provider.

Fetches company news from the GDELT Doc 2.0 API — completely free, no API key required.
Falls back to yfinance news when GDELT returns no results or is unavailable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import requests

from src.data.cache import get_cache
from src.data.models import CompanyNews

logger = logging.getLogger(__name__)

_GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_DATE_FMT = "%Y%m%d%H%M%S"  # GDELT datetime format: YYYYMMDDHHMMSS
_ISO_DATE_FMT = "%Y-%m-%d"


def _parse_gdelt_date(seendate: str) -> str:
    """Convert GDELT seendate (20240102T130000Z) to ISO date YYYY-MM-DD."""
    try:
        # Normalise: strip trailing Z, handle T separator
        clean = seendate.rstrip("Z").replace("T", "")
        dt = datetime.strptime(clean, "%Y%m%d%H%M%S")
        return dt.strftime(_ISO_DATE_FMT)
    except Exception:
        # Fallback: try just the date portion
        try:
            raw = seendate[:8]
            return raw[:4] + "-" + raw[4:6] + "-" + raw[6:8]
        except Exception:
            return seendate


def _tone_to_sentiment(tone: float | None) -> str | None:
    """Map GDELT tone score to a sentiment label."""
    if tone is None:
        return None
    if tone > 1.0:
        return "positive"
    if tone < -1.0:
        return "negative"
    return "neutral"


class GDELTProvider:
    """Fetches company news from the GDELT Doc 2.0 API."""

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int = 25,
    ) -> list[CompanyNews]:
        """Fetch news articles mentioning *ticker* between *start_date* and *end_date*.

        Parameters
        ----------
        ticker:
            Stock ticker symbol (e.g. "AAPL").
        start_date:
            ISO date string YYYY-MM-DD.  If None, defaults to 30 days before end_date.
        end_date:
            ISO date string YYYY-MM-DD (inclusive upper bound).
        limit:
            Maximum number of articles to return (capped server-side too).

        Returns
        -------
        List of :class:`~src.data.models.CompanyNews` objects, or an empty list
        on any error.
        """
        # --- resolve start_date ---
        if start_date is None:
            try:
                end_dt = datetime.strptime(end_date, _ISO_DATE_FMT)
                start_date = (end_dt - timedelta(days=30)).strftime(_ISO_DATE_FMT)
            except Exception:
                start_date = end_date

        # --- cache lookup ---
        cache_key = f"gdelt_{ticker}_{start_date}_{end_date}_{limit}"
        cache = get_cache()
        cached = cache.get_company_news(cache_key)
        if cached:
            return [CompanyNews(**item) for item in cached]

        # --- build GDELT request ---
        try:
            start_dt_str = datetime.strptime(start_date, _ISO_DATE_FMT).strftime(_DATE_FMT)
            end_dt_str = datetime.strptime(end_date, _ISO_DATE_FMT).strftime(_DATE_FMT)
        except Exception as exc:
            logger.warning("GDELTProvider: invalid date format for %s — %s", ticker, exc)
            return []

        query = quote_plus(f'"{ticker}" company')
        url = (
            f"{_GDELT_DOC_URL}"
            f"?query={query}"
            f"&mode=artlist"
            f"&maxrecords={limit}"
            f"&format=json"
            f"&startdatetime={start_dt_str}"
            f"&enddatetime={end_dt_str}"
        )

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            logger.warning("GDELTProvider: request timed out for ticker=%s", ticker)
            return []
        except Exception as exc:
            logger.warning("GDELTProvider: request failed for ticker=%s — %s", ticker, exc)
            return []

        # --- parse articles ---
        articles = payload.get("articles") or []
        if not isinstance(articles, list):
            logger.warning("GDELTProvider: unexpected response structure for %s", ticker)
            return []

        results: list[CompanyNews] = []
        for article in articles:
            try:
                # Filter to English only
                language = (article.get("language") or "").strip()
                if language and language.lower() != "english":
                    continue

                title = article.get("title") or ""
                url_str = article.get("url") or ""
                domain = article.get("domain") or ""
                seendate = article.get("seendate") or ""
                tone = article.get("tone")

                # Skip articles with missing essential fields
                if not title or not url_str:
                    continue

                iso_date = _parse_gdelt_date(seendate) if seendate else end_date
                try:
                    sentiment = _tone_to_sentiment(float(tone) if tone is not None else None)
                except (TypeError, ValueError):
                    sentiment = None

                results.append(
                    CompanyNews(
                        ticker=ticker,
                        title=title,
                        author=None,
                        source=domain,
                        date=iso_date,
                        url=url_str,
                        sentiment=sentiment,
                    )
                )

                if len(results) >= limit:
                    break

            except Exception as exc:
                logger.debug("GDELTProvider: skipping malformed article — %s", exc)
                continue

        # --- cache and return ---
        if results:
            cache.set_company_news(cache_key, [item.model_dump() for item in results])

        return results

    @staticmethod
    def fallback_yahoo_news(
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 25,
    ) -> list[CompanyNews]:
        """Fetch news for *ticker* from yfinance, respecting the requested date range.

        Yahoo Finance only returns current/recent articles, so this method returns an
        empty list when the requested date range is in the past and no articles fall
        within it — rather than silently returning wrong-era articles.
        """
        try:
            import yfinance as yf  # lazy import — optional dependency
        except ImportError:
            logger.warning(
                "GDELTProvider.fallback_yahoo_news: yfinance not installed; "
                "install with 'pip install yfinance'"
            )
            return []

        try:
            tk = yf.Ticker(ticker)
            raw_news = tk.news or []
        except Exception as exc:
            logger.warning(
                "GDELTProvider.fallback_yahoo_news: yfinance fetch failed for %s — %s",
                ticker,
                exc,
            )
            return []

        results: list[CompanyNews] = []
        # Over-fetch to allow for date filtering without a second API call
        for item in raw_news[:limit * 5]:
            try:
                # yfinance >=0.2.x nests everything under a "content" key
                content = item.get("content") or item
                title = content.get("title") or item.get("title") or ""
                if not title:
                    continue

                url_str = (
                    (content.get("canonicalUrl") or {}).get("url")
                    or (content.get("clickThroughUrl") or {}).get("url")
                    or item.get("link")
                    or ""
                )
                if not url_str:
                    continue

                publisher = (
                    (content.get("provider") or {}).get("displayName")
                    or item.get("publisher")
                    or ""
                )

                pub_date = content.get("pubDate") or content.get("displayTime")
                if pub_date:
                    iso_date = pub_date[:10]
                else:
                    publish_ts = item.get("providerPublishTime")
                    if publish_ts:
                        iso_date = datetime.utcfromtimestamp(int(publish_ts)).strftime(_ISO_DATE_FMT)
                    else:
                        iso_date = datetime.utcnow().strftime(_ISO_DATE_FMT)

                # Enforce date range — return nothing rather than wrong-era articles
                if start_date and iso_date < start_date:
                    continue
                if end_date and iso_date > end_date:
                    continue

                results.append(
                    CompanyNews(
                        ticker=ticker,
                        title=title,
                        author=None,
                        source=publisher,
                        date=iso_date,
                        url=url_str,
                        sentiment=None,
                    )
                )
                if len(results) >= limit:
                    break
            except Exception as exc:
                logger.debug(
                    "GDELTProvider.fallback_yahoo_news: skipping malformed item — %s", exc
                )
                continue

        return results
