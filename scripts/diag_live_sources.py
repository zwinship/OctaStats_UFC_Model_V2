#!/usr/bin/env python3
"""
Temporary read-only diagnostic (round 2): DraftKings via a real browser,
Gambly UI flow, and BestFightOdds event/table mapping. No secrets, no writes.
"""
import json
import os
import re
import time
import traceback

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diag_out")
os.makedirs(OUT, exist_ok=True)


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ── A. BFO table → event mapping ─────────────────────────────────────────────
section("A. BestFightOdds event headers per table")
try:
    r = requests.get("https://www.bestfightodds.com/", headers={"User-Agent": UA}, timeout=25)
    soup = BeautifulSoup(r.text, "lxml")
    for ti, t in enumerate(soup.find_all("table")):
        hdr = t.find_previous("div", class_="table-header")
        a = hdr.find("a") if hdr else None
        n_mu = len(t.select("tr[id^=mu-]"))
        print(f"  table {ti:2d} cls={t.get('class')} matchups={n_mu} "
              f"header={hdr.get_text(' ', strip=True)[:70] if hdr else None!r} "
              f"href={a.get('href') if a else None}")
    hdr = soup.find("div", class_="table-header")
    print("  first header html:", re.sub(r"\s+", " ", str(hdr))[:1200])
    wrap = hdr.find_parent("div")
    print("  header parent:", wrap.name, wrap.get("class"), wrap.get("id"))
    th = soup.select("table.odds-table:not(.odds-table-responsive-header) thead th")
    print("  book header data-b:", [(x.get("data-b"), x.get_text(strip=True)[:12]) for x in th])
    # Method-prop labels present
    labels = sorted({th.get_text(strip=True) for th in soup.select("tr.pr th")
                     if "wins" in th.get_text().lower()})[:25]
    print("  sample prop labels:", labels)
except Exception:
    traceback.print_exc()

# ── B. DraftKings through a real browser ────────────────────────────────────
section("B. DraftKings via Playwright")
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(user_agent=UA, locale="en-US",
                            geolocation={"latitude": 40.73, "longitude": -74.17},
                            permissions=["geolocation"])
        pg = ctx.new_page()
        api_hits = []
        pg.on("response", lambda resp: api_hits.append((resp.status, resp.url))
              if "sportscontent" in resp.url or "/api/" in resp.url else None)
        try:
            pg.goto("https://sportsbook.draftkings.com/leagues/mma/ufc",
                    wait_until="domcontentloaded", timeout=30000)
            time.sleep(8)
        except Exception as e:
            print("  goto error:", e)
        html = pg.content()
        print(f"  page title={pg.title()!r} len={len(html)}")
        print("  body text:", re.sub(r"\s+", " ", pg.inner_text("body"))[:600])
        print("  event links:", sorted(set(re.findall(r"/event/[^\"'?#]+", html)))[:10])
        for s, u in api_hits[:25]:
            print("   api", s, u[:160])
        for url in ["https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/9034",
                    "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1/leagues/9034"]:
            try:
                resp = ctx.request.get(url, timeout=20000)
                body = resp.text()
                print(f"  ctx.request {url[-30:]} status={resp.status} len={len(body)} head={body[:200]!r}")
                if resp.status == 200:
                    open(os.path.join(OUT, "dk_league.json"), "w").write(body)
                    j = json.loads(body)
                    print("   keys:", list(j)[:15])
                    for k in ("events", "markets", "selections"):
                        for it in (j.get(k) or [])[:2]:
                            print(f"   {k}:", json.dumps(it)[:500])
            except Exception as e:
                print("  ctx.request error:", e)
        b.close()
except Exception:
    traceback.print_exc()

# ── C. Gambly UI flow (anonymous browser session) ───────────────────────────
section("C. Gambly bot page in Playwright")
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(user_agent=UA, locale="en-US")
        pg = ctx.new_page()
        calls = []

        def on_resp(resp):
            u = resp.url
            if "gambly" in u and ("/api/" in u) or "identitytoolkit" in u or "securetoken" in u:
                try:
                    hdrs = resp.request.headers
                except Exception:
                    hdrs = {}
                calls.append((resp.status, resp.request.method, u[:150],
                              sorted(k for k in hdrs if k.lower() in ("authorization", "cookie", "x-firebase-appcheck", "x-session"))))
        pg.on("response", on_resp)
        prompt = "alexander+volkanovski+moneyline"
        try:
            pg.goto(f"https://gambly.com/gambly-bot?auto=1&prompt={prompt}",
                    wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print("  goto error:", e)
        for i in range(12):
            time.sleep(5)
            n = pg.locator("a[href*='draftkings'], a[href*='preurl']").count()
            print(f"  t={5*(i+1)}s dk anchors={n} url={pg.url[:120]}")
            if n:
                break
        for a in pg.locator("a[href*='draftkings'], a[href*='preurl']").all()[:10]:
            print("   href:", (a.get_attribute("href") or "")[:300])
        print("  body:", re.sub(r"\s+", " ", pg.inner_text("body"))[:1200])
        for c in calls[:40]:
            print("   call:", c)
        cookies = ctx.cookies()
        print("  cookies:", [(c["name"], c["domain"]) for c in cookies])
        ls = pg.evaluate("() => Object.keys(window.localStorage)")
        print("  localStorage keys:", ls[:30])
        b.close()
except Exception:
    traceback.print_exc()

print("\nDIAG COMPLETE")
