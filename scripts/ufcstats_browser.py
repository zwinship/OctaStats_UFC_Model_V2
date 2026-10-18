"""
ufcstats_browser.py
Shared headless-Chromium fetcher for UFCStats.

UFCStats now serves a JavaScript "Checking your browser…" challenge to plain
HTTP clients (requests gets a ~3KB stub page with no data), so every UFCStats
page must be loaded in a real browser. This module keeps ONE browser/context
alive for the whole script run so the challenge is solved once and later pages
load fast, and it retries until the expected selector actually appears —
a page that only rendered the challenge is never treated as "no data".

Usage:
    from ufcstats_browser import fetch_html, fetch_soup, close_browser
    html = fetch_html(url, "tr.b-statistics__table-row")
"""

import atexit
import time

from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

_pw = None
_browser = None
_context = None


def _ensure_context():
    global _pw, _browser, _context
    if _context is not None:
        return _context
    from playwright.sync_api import sync_playwright
    _pw = sync_playwright().start()
    _browser = _pw.chromium.launch(headless=True)
    _context = _browser.new_context(user_agent=UA, locale="en-US")
    return _context


def close_browser():
    global _pw, _browser, _context
    try:
        if _browser is not None:
            _browser.close()
        if _pw is not None:
            _pw.stop()
    except Exception:
        pass
    _pw = _browser = _context = None


atexit.register(close_browser)


def fetch_html(url, wait_selector, timeout_ms=30000, retries=3, delay=0.5):
    """
    Load `url` and return its HTML once `wait_selector` is present.
    Returns None if the selector never appeared after `retries` attempts
    (challenge not cleared, page down, or the page genuinely lacks it).
    """
    from playwright.sync_api import TimeoutError as PWTimeout
    last_err = None
    for attempt in range(1, retries + 1):
        page = None
        try:
            page = _ensure_context().new_page()
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            page.wait_for_selector(wait_selector, timeout=timeout_ms)
            html = page.content()
            page.close()
            if delay:
                time.sleep(delay)
            return html
        except PWTimeout as e:
            last_err = f"timed out waiting for {wait_selector}"
        except Exception as e:
            last_err = str(e)[:200]
            # A crashed browser/context is unrecoverable — start a fresh one
            if "closed" in last_err.lower() or "crash" in last_err.lower():
                close_browser()
        finally:
            try:
                if page is not None and not page.is_closed():
                    page.close()
            except Exception:
                pass
        time.sleep(2 * attempt)
    print(f"  [WARN] UFCStats fetch failed after {retries} tries ({last_err}): {url}")
    return None


def fetch_soup(url, wait_selector, **kw):
    html = fetch_html(url, wait_selector, **kw)
    return BeautifulSoup(html, "lxml") if html else None


# Selectors that prove a UFCStats page actually rendered (not the challenge stub)
SEL_EVENT_LIST  = "tr.b-statistics__table-row a"
SEL_EVENT_PAGE  = "tr.b-fight-details__table-row__hover"
SEL_FIGHT_PAGE  = ".b-fight-details__person-name"
SEL_FIGHTER_PAGE = "li.b-list__box-list-item"
