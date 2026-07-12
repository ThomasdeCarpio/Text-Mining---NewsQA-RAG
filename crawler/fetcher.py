"""Browser-based page fetching for dynamic news pages."""

from __future__ import annotations

from dataclasses import dataclass

from crawler.models import RenderedPage

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class FetchConfig:
    """Runtime configuration for Playwright page fetching.

    Args:
        timeout_ms: Maximum navigation and selector wait time in milliseconds.
        wait_ms: Extra delay after page load to allow late JavaScript rendering.
        wait_selector: Optional CSS selector to wait for before reading HTML.
        headless: Whether Chromium should run in headless mode.
    """

    timeout_ms: int = 30000
    wait_ms: int = 1000
    wait_selector: str | None = None
    headless: bool = True


class PlaywrightFetcher:
    """Fetch rendered HTML pages using Playwright Chromium.

    Args:
        config: Fetch behavior and wait settings.
        user_agent: Browser user agent string sent to target sites.
    """

    def __init__(self, config: FetchConfig, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.config = config
        self.user_agent = user_agent
        self._playwright = None
        self._browser = None
        self._context = None
        self._timeout_error = TimeoutError

    def __enter__(self) -> "PlaywrightFetcher":
        """Start Chromium and return this fetcher instance."""

        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "playwright is required for dynamic crawling. "
                "Install crawler/requirements-crawler.txt and run: python -m playwright install chromium"
            ) from exc

        self._timeout_error = PlaywrightTimeoutError
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.config.headless)
        self._context = self._browser.new_context(
            user_agent=self.user_agent,
            viewport={"width": 1366, "height": 900},
            locale="en-US",
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close Chromium resources when leaving the context manager."""

        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()

    def fetch(self, url: str, wait_selector: str | None = None) -> RenderedPage:
        """Render one URL and return its final HTML.

        Args:
            url: Absolute URL to open in Chromium.
            wait_selector: Optional CSS selector overriding the default wait selector.

        Returns:
            RenderedPage with final URL, HTML, and HTTP status when available.

        Raises:
            RuntimeError: If the fetcher has not been started with a context manager.
            Exception: Propagates Playwright navigation failures for caller-level handling.
        """

        if self._context is None:
            raise RuntimeError("PlaywrightFetcher must be used as a context manager.")

        page = self._context.new_page()
        page.set_default_timeout(self.config.timeout_ms)
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=min(self.config.timeout_ms, 10000))
            except self._timeout_error:
                pass

            selector = wait_selector or self.config.wait_selector
            if selector:
                page.wait_for_selector(selector, timeout=self.config.timeout_ms)

            if self.config.wait_ms > 0:
                page.wait_for_timeout(self.config.wait_ms)

            return RenderedPage(
                requested_url=url,
                final_url=page.url,
                html=page.content(),
                status_code=response.status if response else None,
            )
        finally:
            page.close()

