#!/usr/bin/env python3
"""
Temporary read-only diagnostic: fetches every live data source the pipeline
depends on and prints what comes back. Performs NO GitHub writes and uses no
secrets. Safe to delete once the scrapers are verified.
"""
import importlib.util
import json
import os
import re
import sys
import traceback

import requests
from bs4 import BeautifulSoup

sys.argv.append("--test")  # lets the pipeline modules import without a PAT
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "diag_out")
os.makedirs(OUT, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
H  = {"User-Agent": UA, "Accept": "text/html,application/json,*/*",
      "Accept-Language": "en-US,en;q=0.9"}


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"),
                                                  os.path.join(HERE, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def save(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)


def pw_html(url, wait=None, timeout=25000):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(user_agent=UA)
        try:
            pg.goto(url, timeout=timeout, wait_until="domcontentloaded")
            if wait:
                pg.wait_for_selector(wait, timeout=timeout)
        except Exception as e:
            print(f"  [pw] {type(e).__name__}: {str(e)[:200]}")
        html = pg.content()
        b.close()
        return html


# ── 1. UFCStats ──────────────────────────────────────────────────────────────
section("1. UFCStats: requests vs Playwright")
for url, sel in [("http://www.ufcstats.com/statistics/events/upcoming", "tr.b-statistics__table-row"),
                 ("http://www.ufcstats.com/statistics/events/completed", "tr.b-statistics__table-row")]:
    try:
        r = requests.get(url, headers=H, timeout=20)
        rows = BeautifulSoup(r.text, "lxml").select(sel)
        print(f"  requests {url}: status={r.status_code} len={len(r.text)} rows={len(rows)}")
        if not rows:
            print("    body head:", re.sub(r"\s+", " ", r.text[:600]))
    except Exception as e:
        print(f"  requests {url}: ERROR {e}")
    html = pw_html(url, sel)
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select(sel)
    print(f"  playwright {url}: len={len(html)} rows={len(rows)}")
    save(url.rsplit("/", 1)[-1] + ".html", html)
    for row in rows[:4]:
        print("    ", re.sub(r"\s+", " ", row.get_text(" ", strip=True))[:150])

# Body-only wait (what 02_scrape_weekly_update.get_soup does)
html = pw_html("http://www.ufcstats.com/statistics/events/completed?page=1", "body")
print(f"  playwright wait=body completed: rows={len(BeautifulSoup(html, 'lxml').select('tr.b-statistics__table-row'))}")

# Event detail pages
comp = BeautifulSoup(open(os.path.join(OUT, "completed.html")).read(), "lxml")
links = [a.get("href") for a in comp.select("i.b-statistics__table-content a")]
upc  = BeautifulSoup(open(os.path.join(OUT, "upcoming.html")).read(), "lxml")
ulinks = [a.get("href") for a in upc.select("i.b-statistics__table-content a")]
for label, url in [("last completed", links[0] if links else None),
                   ("next upcoming", ulinks[0] if ulinks else None)]:
    if not url:
        print(f"  no {label} event link")
        continue
    html = pw_html(url, "tr.b-fight-details__table-row")
    save(f"event_{label.replace(' ', '_')}.html", html)
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover")
    print(f"  {label} {url}: {len(rows)} fight rows")
    for row in rows[:3]:
        tds = row.find_all("td")
        print("    tds:", [re.sub(r"\s+", " ", td.get_text(" ", strip=True))[:40] for td in tds])
        print("    td0 html:", re.sub(r"\s+", " ", str(tds[0]))[:300])
        if len(tds) > 6:
            print("    td6 html:", re.sub(r"\s+", " ", str(tds[6]))[:400])
    try:
        r = requests.get(url, headers=H, timeout=20)
        print(f"  requests same page: status={r.status_code} rows="
              f"{len(BeautifulSoup(r.text, 'lxml').select('tr.b-fight-details__table-row__hover'))}")
    except Exception as e:
        print(f"  requests same page: ERROR {e}")

# ── 2. BestFightOdds ─────────────────────────────────────────────────────────
section("2. BestFightOdds structure")
try:
    r = requests.get("https://www.bestfightodds.com/", headers=H, timeout=25)
    print(f"  status={r.status_code} len={len(r.text)}")
    save("bfo.html", r.text)
    soup = BeautifulSoup(r.text, "lxml")
    tables = soup.find_all("table")
    print(f"  tables={len(tables)}")
    for ti, t in enumerate(tables[:4]):
        print(f"\n  -- table {ti} class={t.get('class')} rows={len(t.find_all('tr'))}")
        thead = t.find("thead")
        if thead:
            print("   thead:", [c.get_text(strip=True) for c in thead.find_all(['th', 'td'])])
        for row in t.find_all("tr")[:7]:
            print("   ROW:", re.sub(r"\s+", " ", str(row))[:900])
    # Event headers
    for h in soup.select("div.table-header, .table-header a, h1")[:12]:
        print("  header:", h.get_text(" ", strip=True)[:100])
except Exception:
    traceback.print_exc()

section("2b. 04_predict.scrape_bestfightodds() output")
try:
    p4 = load("04_predict.py")
    ml, props = p4.scrape_bestfightodds()
    print(ml.head(60).to_string())
    print(props.head(30).to_string())
except Exception:
    traceback.print_exc()

# ── 3. DraftKings public API ────────────────────────────────────────────────
section("3. DraftKings sportsbook API")
for url in [
    "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/9034",
    "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1/leagues/9034",
    "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/9034/categories/1000",
    "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/9034?format=json",
]:
    try:
        r = requests.get(url, headers={**H, "Accept": "application/json"}, timeout=20)
        print(f"\n  {url}\n    status={r.status_code} len={len(r.text)} ctype={r.headers.get('content-type')}")
        try:
            j = r.json()
        except Exception:
            print("    body:", r.text[:300])
            continue
        save("dk_" + re.sub(r"\W+", "_", url[-40:]) + ".json", json.dumps(j)[:3_000_000])
        print("    keys:", list(j.keys())[:20] if isinstance(j, dict) else type(j))
        if isinstance(j, dict):
            for k in ("events", "markets", "selections", "subcategories", "categories"):
                v = j.get(k)
                if isinstance(v, list):
                    print(f"    {k}: {len(v)}")
                    for item in v[:3]:
                        print("      ", json.dumps(item)[:600])
            mk = j.get("markets") or []
            names = sorted({(m.get("name"), (m.get("marketType") or {}).get("name")) for m in mk})
            print("    market names:", names[:40])
            cats = j.get("categories") or []
            print("    categories:", [(c.get("id"), c.get("name")) for c in cats][:40])
            subs = j.get("subcategories") or []
            print("    subcategories:", [(c.get("id"), c.get("name"), c.get("categoryId")) for c in subs][:60])
    except Exception as e:
        print(f"  {url}: ERROR {e}")

# ── 4. Gambly (anonymous, no secrets) ───────────────────────────────────────
section("4. Gambly API (anonymous)")
try:
    r = requests.post("https://gambly.com/api/proxy/core/api/gambly-bot/start-betslip-job",
                      headers={"User-Agent": UA, "Content-Type": "application/json",
                               "Referer": "https://gambly.com/", "Origin": "https://gambly.com"},
                      json={"prompt": "jon jones moneyline", "base64Images": [],
                            "twitterUrl": "", "saveBetslipFeedPostSync": False},
                      timeout=20)
    print(f"  no-cookie POST status={r.status_code} body={r.text[:300]}")
except Exception as e:
    print(f"  ERROR {e}")

# ── 5. UFC.com rankings ─────────────────────────────────────────────────────
section("5. ufc.com rankings")
try:
    r = requests.get("https://www.ufc.com/rankings", headers=H, timeout=20)
    s = BeautifulSoup(r.text, "lxml")
    print(f"  status={r.status_code} tables={len(s.find_all('table'))}")
except Exception as e:
    print(f"  ERROR {e}")

print("\nDIAG COMPLETE")
