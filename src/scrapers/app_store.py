"""
App Store scraper using the iTunes MZStore JSON API.

The old RSS feed returns only ~2 reviews per page (max ~10 total).
This implementation uses the private-but-stable MZStore API which returns
up to 500 reviews per request and supports full pagination.

SSL note: Apple's servers can fail certificate verification on some Python
installations (common on macOS Python.org installs). The scraper automatically
retries with the certifi bundle, and as a last resort disables verification
with a warning.  Run `pip install certifi` to resolve properly.
"""
import ssl
import warnings
import requests
from typing import List, Dict, Any, Optional
from .base import BaseScraper
from ..models.review import Review

# Apple storefront IDs: country code → header value
_STOREFRONTS: Dict[str, str] = {
    "hk": "143463-18,30",
    "us": "143441-1,29",
    "gb": "143444-2,29",
    "au": "143460-27,29",
    "sg": "143464-19,29",
    "tw": "143470-21,29",
}

# MZStore sort codes
_SORT_CODES: Dict[str, int] = {
    "mostRecent":  4,
    "mostHelpful": 1,
}

_PAGE_SIZE = 500   # max reviews the API returns per request
_API_URL = (
    "https://itunes.apple.com/WebObjects/MZStore.woa/wa/userReviewsRow"
    "?id={app_id}&displayable-kind=11&startIndex={start}&endIndex={end}&sort={sort}"
)


def _make_session(verify: bool = True, certifi_bundle: Optional[str] = None) -> requests.Session:
    """Return a configured requests Session."""
    s = requests.Session()
    s.headers.update({"User-Agent": "iTunes/12.0 (Macintosh; OS X 10.15)"})
    if certifi_bundle:
        s.verify = certifi_bundle
    else:
        s.verify = verify
    return s


def _best_session(logger) -> requests.Session:
    """
    Return a working Session for Apple's servers.

    Priority:
    1. Default SSL verification (uses the system/venv CA bundle)
    2. Explicit certifi CA bundle  (pip install certifi)
    3. Disabled verification with a warning (last resort)
    """
    test_url = "https://itunes.apple.com"

    # --- attempt 1: default verification ---
    try:
        s = _make_session(verify=True)
        s.get(test_url, timeout=10)
        return s
    except requests.exceptions.SSLError:
        logger.debug("Default SSL verification failed, trying certifi bundle…")

    # --- attempt 2: explicit certifi bundle ---
    try:
        import certifi
        s = _make_session(certifi_bundle=certifi.where())
        s.get(test_url, timeout=10)
        logger.info("Using certifi CA bundle for App Store requests.")
        return s
    except (ImportError, requests.exceptions.SSLError):
        pass

    # --- attempt 3: disable verification (warns loudly) ---
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logger.warning(
        "SSL certificate verification DISABLED for App Store. "
        "Fix with: pip install certifi   OR run "
        "/Applications/Python\\ 3.x/Install\\ Certificates.command"
    )
    return _make_session(verify=False)


class AppStoreScraper(BaseScraper):
    """App Store review scraper using the iTunes MZStore JSON API."""

    @property
    def platform_name(self) -> str:
        return "app_store"

    def __init__(self, config: Dict[str, Any], app_config: Dict[str, Any]):
        super().__init__(config, app_config)
        self.app_store_config = app_config.get("app_store", {})
        self.app_id = self.app_store_config.get("app_id")
        if not self.app_id:
            raise ValueError("app_store.app_id is required in app configuration")
        self.session = _best_session(self.logger)

    def scrape_reviews(self) -> List[Review]:
        """Scrape all App Store reviews across configured countries and sort methods."""
        if not self.app_store_config.get("enabled", True):
            self.logger.info("App Store scraping disabled")
            return []

        self.logger.info(f"Starting App Store scraping for app ID: {self.app_id}")

        all_reviews: List[Review] = []
        countries = self.app_store_config.get("countries", ["hk"])
        sort_methods = self.app_store_config.get("sort_methods", ["mostRecent"])

        for country in countries:
            storefront = _STOREFRONTS.get(country, _STOREFRONTS["hk"])
            self.session.headers["X-Apple-Store-Front"] = f"{storefront} t:native"

            for sort_name in sort_methods:
                sort_code = _SORT_CODES.get(sort_name, 4)
                self.logger.info(f"  Fetching country={country} sort={sort_name}")
                batch = self._fetch_all(sort_code)
                added = 0
                for r in batch:
                    if self.add_review(r):
                        all_reviews.append(r)
                        added += 1
                self.logger.info(f"  Added {added} new reviews (sort={sort_name})")
                self.rate_limit("between_sort_methods")

            self.rate_limit("between_countries")

        self.logger.info(f"App Store total unique reviews: {len(all_reviews)}")
        return self.validate_reviews(all_reviews)

    def _fetch_all(self, sort_code: int) -> List[Review]:
        """Paginate through MZStore API and return all Review objects."""
        reviews: List[Review] = []
        start = 0

        while True:
            url = _API_URL.format(
                app_id=self.app_id,
                start=start,
                end=start + _PAGE_SIZE - 1,
                sort=sort_code,
            )
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code != 200:
                    self.logger.warning(f"App Store API returned HTTP {resp.status_code}")
                    break

                batch_data = resp.json().get("userReviewList", [])
                if not batch_data:
                    break

                for item in batch_data:
                    reviews.append(Review(
                        platform="App Store",
                        review_id=str(item.get("userReviewId", "")),
                        title=item.get("title", ""),
                        content=item.get("body", ""),
                        rating=int(item.get("rating", 0)),
                        author=item.get("name", "Anonymous"),
                        date=item.get("date", ""),
                        version="",
                    ))

                self.logger.debug(f"Fetched {len(batch_data)} reviews (start={start})")

                if len(batch_data) < _PAGE_SIZE:
                    break   # last page
                start += _PAGE_SIZE
                self.rate_limit("between_requests")

            except requests.exceptions.SSLError as e:
                self.logger.error(
                    f"SSL error fetching App Store reviews: {e}\n"
                    "Fix: pip install certifi   or run the Python 'Install Certificates' script."
                )
                break
            except Exception as e:
                self.logger.error(f"App Store fetch error at start={start}: {e}")
                break

        return reviews
