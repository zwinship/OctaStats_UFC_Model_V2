#!/usr/bin/env python3
"""
Temporary read-only diagnostic (round 3): exercise the rewritten scrapers
against live pages. No secrets, no GitHub writes.
"""
import importlib.util
import os
import re
import sys
import traceback

import requests
from bs4 import BeautifulSoup

sys.argv.append("--test")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


section("A. ufc.com rankings table headings")
try:
    r = requests.get("https://www.ufc.com/rankings", headers={"User-Agent": UA}, timeout=25)
    soup = BeautifulSoup(r.text, "lxml")
    for i, t in enumerate(soup.select("table")):
        cap = t.find("caption")
        grp = t.find_parent(class_=re.compile("view-grouping"))
        hdr = grp.select_one(".view-grouping-header") if grp else None
        first = [a.get_text(strip=True) for a in t.select("tbody tr td a")][:3]
        print(f"  [{i:2d}] grouping-header={hdr.get_text(' ', strip=True) if hdr else None!r} "
              f"caption={re.sub(r'\\s+', ' ', cap.get_text(' ', strip=True))[:90] if cap else None!r} first={first}")
    t0 = soup.select("table")[0]
    print("  table0 ancestry:", [(p.name, p.get("class")) for p in list(t0.parents)[:4]])
    print("  table0 caption html:", re.sub(r"\s+", " ", str(t0.find("caption")))[:800])
except Exception:
    traceback.print_exc()

section("B. Gambly auth flow (static JS inspection)")
try:
    r = requests.get("https://gambly.com/gambly-bot", headers={"User-Agent": UA}, timeout=25)
    srcs = sorted(set(re.findall(r'src="(/_next/static/[^"]+\.js)"', r.text)))
    print(f"  page status={r.status_code} scripts={len(srcs)}")
    hits = {}
    for s in srcs:
        js = requests.get("https://gambly.com" + s, headers={"User-Agent": UA}, timeout=25).text
        for pat in [r"AIza[0-9A-Za-z_\-]{35}", r"auth/firebase/session[^\"'`]{0,80}",
                    r"gambly_token_set", r"firebase_session_required", r"authDomain:\"[^\"]+\"",
                    r"projectId:\"[^\"]+\"", r"getIdToken", r"signInAnonymously",
                    r"start-betslip-job.{0,200}"]:
            for m in re.findall(pat, js)[:3]:
                hits.setdefault(pat, set()).add(m[:220])
    for k, v in hits.items():
        print(f"  {k}: {sorted(v)[:4]}")
    # what does the session endpoint say?
    for method in ("GET", "POST"):
        rr = requests.request(method, "https://gambly.com/api/auth/firebase/session",
                              headers={"User-Agent": UA, "Content-Type": "application/json"},
                              json={} if method == "POST" else None, timeout=20)
        print(f"  {method} /api/auth/firebase/session → {rr.status_code} {rr.text[:200]!r}")
except Exception:
    traceback.print_exc()

section("C. 04_predict: upcoming card + BestFightOdds parse + bout matching")
try:
    p4 = load("04_predict.py", "predict04")
    ev = p4.get_upcoming_event()
    print("  event:", ev and {k: ev[k] for k in ("name", "date", "url")})
    ml, props = p4.scrape_bestfightodds()
    print(ml[ml["event"].str.startswith("UFC")].to_string()[:6000])
    if ev:
        ok = 0
        for fa, fb, wc, info in ev["fights"]:
            oa, ob = p4._match_bout_odds(fa, fb, ml)
            ok += bool(oa and ob)
            print(f"  {fa:26s} {str(oa):>6s} | {fb:26s} {str(ob):>6s} | {wc:20s} {info}")
        print(f"  → both lines found for {ok}/{len(ev['fights'])} bouts")
    print(f"  props: {len(props)} rows; sample:")
    print(props.head(12).to_string())
except Exception:
    traceback.print_exc()

section("D. 05_scrape_results --test")
try:
    p5 = load("05_scrape_results.py", "results05")
    p5.run_test()
except SystemExit as e:
    print("  run_test exit:", e)
except Exception:
    traceback.print_exc()

print("\nDIAG COMPLETE")
