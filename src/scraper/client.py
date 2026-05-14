"""Polite HTTP client with disk caching, retries, and rate limiting.

Why this exists as a separate module:
    Web scraping for portfolio projects fails in two predictable ways:
    1. The scraper hammers a server and gets IP-banned mid-demo.
    2. The site changes overnight and the project breaks irreproducibly.

    This module solves both. Every GET is cached to disk by URL hash, so re-running
    the pipeline doesn't re-hit the server. Failures retry with exponential backoff.
    A minimum interval between requests prevents bursting.

    In interviews: "I cached every page on first fetch so the project is reproducible
    and respectful of the source sites." This is a real engineering decision, not a
    line on a resume.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# 3 seconds between requests is conservative but safe. Basketball-Reference's
# robots.txt asks for a 3-second crawl delay; we honor it.
MIN_INTERVAL_SECONDS = 3.0

# Standard browser UA. Some sites block default `python-requests/X.Y` UA.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class PoliteClient:
    """A requests.Session wrapper with caching, retries, and pacing.

    Usage:
        client = PoliteClient(cache_dir=Path("data/raw/_cache"))
        html = client.get("https://www.basketball-reference.com/leagues/NBA_2025_per_game.html")
    """

    def __init__(
        self,
        cache_dir: Path,
        min_interval: float = MIN_INTERVAL_SECONDS,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_request_at: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def _cache_path(self, url: str) -> Path:
        """Derive a stable cache filename from a URL."""
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{key}.html"

    def _wait_for_rate_limit(self) -> None:
        """Block until enough time has elapsed since the last request."""
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def get(self, url: str, *, force_refresh: bool = False) -> str:
        """Fetch a URL, using disk cache when available.

        Args:
            url: Full URL to fetch.
            force_refresh: If True, bypass cache and refetch.

        Returns:
            HTML body as a string.

        Raises:
            requests.HTTPError: if all retries fail.
        """
        cache_path = self._cache_path(url)
        if cache_path.exists() and not force_refresh:
            logger.debug("CACHE HIT: %s", url)
            return cache_path.read_text(encoding="utf-8")

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._wait_for_rate_limit()
                logger.info("GET (attempt %d): %s", attempt, url)
                response = self._session.get(url, timeout=20)
                self._last_request_at = time.time()
                response.raise_for_status()
                cache_path.write_text(response.text, encoding="utf-8")
                return response.text
            except requests.RequestException as exc:
                last_exc = exc
                # Exponential backoff: 4s, 8s, 16s
                backoff = 2 ** (attempt + 1)
                logger.warning(
                    "Request failed (%s). Retrying in %ds...", exc, backoff
                )
                time.sleep(backoff)

        assert last_exc is not None  # for type checker
        raise last_exc
