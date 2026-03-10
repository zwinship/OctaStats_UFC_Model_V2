#!/usr/bin/env python3
"""
v1_predict.py
OctaStats — V1 Model Friday Prediction Runner (Automated)

Uses the same V1 prediction logic as the original manual scripts, but now:
  - Sources odds from BestFightOdds (DraftKings preferred, FanDuel fallback)
  - Reads fight card from UFCStats upcoming events
  - Loads V1 model bundle (trained by v1_train_model.py)
  - Outputs to zwinship/UFC_Model repo (PythonAnywhere site reads from there)

V1 logic preserved:
  - Predict win probability via LinearRegression(weight_class, style) group
  - Implied probability from American odds
  - Edge = predicted_prob - implied_prob
  - Discrete bet sizing: edge<5pp→0u, 5-12pp→1u, 12-20pp→2u, >20pp→3u

Environment variables:
    ZWINSHIP_PAT  — GitHub personal access token
"""

import os
import io
import re
import base64
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
import sys as _sys
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
if not GITHUB_TOKEN and "--test" not in _sys.argv:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
V2_REPO      = "OctaStats_UFC_Model_V2"
V1_REPO      = "UFC_Model"
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
}
SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection":      "keep-alive",
}
BASE_URL       = "http://www.ufcstats.com"

# V1 discrete bet sizing thresholds (edge in percentage points)
BET_THRESHOLDS = [(0.20, 3), (0.12, 2), (0.05, 1)]  # (min_edge, units)

STAT_COLS = [
    "kd", "sig_str_landed", "sig_str_att", "total_str_landed",
    "total_str_att", "td_landed", "td_att", "sub_att",
    "ctrl_seconds", "head_landed", "body_landed", "leg_landed",
    "distance_landed", "clinch_landed", "ground_landed",
]
STYLE_Z = 0.5


# ── GitHub helpers ────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path, repo=V1_REPO):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{repo}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame(), None
    data = resp.json()
    sha  = data["sha"]
    raw_content = data.get("content", "").replace("\n", "").strip()
    if not raw_content and data.get("download_url"):
        raw_resp = requests.get(data["download_url"], headers=GH_HEADERS, timeout=60)
        if raw_resp.status_code == 200:
            return pd.read_csv(io.StringIO(raw_resp.text)), sha
        return pd.DataFrame(), None
    content = base64.b64decode(raw_content).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), sha


def read_pickle_from_github(repo_path, repo=V1_REPO):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{repo}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"  [ERROR] Could not load {repo_path}: {resp.status_code}")
        return None
    data = resp.json()
    return pickle.loads(base64.b64decode(data["content"]))


def write_csv_to_github(df, repo_path, message, repo=V1_REPO):
    csv_str  = df.to_csv(index=False)
    content  = base64.b64encode(csv_str.encode()).decode()
    url      = f"https://api.github.com/repos/{REPO_OWNER}/{repo}/contents/{repo_path}"
    check    = requests.get(url, headers=GH_HEADERS, timeout=10)
    sha      = check.json().get("sha") if check.status_code == 200 else None
    payload  = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload, timeout=30)
    ok   = resp.status_code in (200, 201)
    print(f"  {'✓' if ok else '✗'} GitHub: {repo_path} ({repo}) — {resp.status_code}")
    return ok


# ── Check for upcoming event ──────────────────────────────────────────────────

def get_upcoming_event():
    """Return (event_name, event_date, fights_list) or None if no event this week."""
    url  = f"{BASE_URL}/statistics/events/upcoming"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=15)
    if resp.status_code != 200:
        return None
    soup   = BeautifulSoup(resp.text, "lxml")
    rows   = soup.select("tr.b-statistics__table-row")
    today  = datetime.utcnow().date()
    window = today + timedelta(days=7)

    for row in rows:
        if row.find("th"):
            continue
        tds = row.find_all("td")
        if not tds:
            continue
        link_el   = tds[0].select_one("i.b-statistics__table-content a")
        date_span = tds[0].select_one("span.b-statistics__date")
        if not link_el or not date_span:
            continue
        name      = link_el.get_text(strip=True)
        event_url = link_el.get("href", "")
        date_text = date_span.get_text(strip=True)
        try:
            event_date = datetime.strptime(date_text, "%B %d, %Y").date()
        except ValueError:
            continue
        if today <= event_date <= window:
            fights = scrape_event_card(event_url)
            return name, event_date, fights

    return None


def scrape_event_card(event_url):
    """Scrape fighter names and weight classes from a UFCStats event page."""
    resp = requests.get(event_url, headers=SCRAPE_HEADERS, timeout=15, allow_redirects=True)
    if resp.status_code != 200:
        return []
    soup  = BeautifulSoup(resp.text, "lxml")
    bouts = []

    all_rows = soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover") or                soup.select("tr.b-fight-details__table-row")

    WEIGHT_CLASSES = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight",
                      "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
                      "Heavyweight"]

    for row in all_rows:
        tds = row.find_all("td")
        if not tds:
            continue
        fighter_links = []
        for td_idx in [1, 0]:
            if td_idx < len(tds):
                fighter_links = tds[td_idx].select("p a.b-link") or tds[td_idx].select("a.b-link")
                if len(fighter_links) >= 2:
                    break
        if len(fighter_links) < 2:
            fighter_links = row.select("a.b-link")
        if len(fighter_links) < 2:
            continue

        f1, f2 = fighter_links[0].get_text(strip=True), fighter_links[1].get_text(strip=True)
        wc = "Unknown"
        for td in tds:
            td_text = td.get_text(strip=True)
            for wc_name in WEIGHT_CLASSES:
                if wc_name.lower() in td_text.lower():
                    wc = ("Women's " + wc_name) if "women" in td_text.lower() else wc_name
                    break
            if wc != "Unknown":
                break
        if f1 and f2:
            bouts.append({
                "fighter1":     f1,
                "fighter2":     f2,
                "weight_class": wc,
            })
    return bouts


# ── BestFightOdds scraper ────────────────────────────────────────────────────

def american_to_implied(odds):
    """Convert American odds integer to implied probability."""
    try:
        o = int(odds)
        return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)
    except (ValueError, ZeroDivisionError):
        return None


def scrape_bestfightodds(event_name=None):
    """
    Scrape odds from BestFightOdds using V2-compatible logic.
    Prefers DraftKings, falls back through BOOK_PREF list.
    Returns dict: {fighter_name_lower: american_odds_str}
    """
    url  = "https://www.bestfightodds.com/"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"  [WARN] BestFightOdds returned {resp.status_code}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    BOOK_PREF = ["draftkings", "caesars", "betrivers", "betway", "unibet", "bet365"]
    book_col_map = {}

    for row in soup.select("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        texts = [c.get_text(strip=True).lower() for c in cells]
        if sum(1 for b in BOOK_PREF if any(b in t for t in texts)) >= 2:
            for i, t in enumerate(texts):
                for b in BOOK_PREF:
                    if b in t and b not in book_col_map:
                        book_col_map[b] = i
            if book_col_map:
                break

    preferred_col, preferred_book = None, None
    for book in BOOK_PREF:
        if book in book_col_map:
            preferred_col  = book_col_map[book]
            preferred_book = book.title()
            break

    print(f"  [WARN] Using odds from: {preferred_book} (col {preferred_col})")

    PROP_WORDS = {"wins", "by", "inside", "decision", "draw", "round", "goes", "starts",
                  "doesn", "won", "ends", "either", "other", "points", "deducted",
                  "submission", "tko", "ko", "majority", "split", "unanimous", "parlay",
                  "over", "under", "not", "fight", "method"}

    odds = {}
    for row in soup.select("tr"):
        name_el = row.select_one("a")
        if not name_el:
            continue
        fighter_name = name_el.get_text(strip=True)
        if not fighter_name or len(fighter_name) < 3:
            continue
        if set(fighter_name.lower().split()) & PROP_WORDS:
            continue

        cells     = row.find_all(["td", "th"])
        odds_text = None
        if preferred_col is not None and preferred_col < len(cells):
            t = cells[preferred_col].get_text(strip=True)
            if re.match(r'^[+-]\d{2,4}$', t):
                odds_text = t

        if odds_text is None:
            m = re.search(r'[+-]\d{3,4}(?!\d)', row.get_text(" ", strip=True))
            if m:
                odds_text = m.group()

        if odds_text and re.match(r'^[+-]\d{2,4}$', odds_text):
            odds[fighter_name.lower()] = odds_text

    print(f"  Scraped {len(odds)} fighters from BestFightOdds ({preferred_book})")
    return odds


def match_fighter_odds(name, odds_dict):
    """Fuzzy-match a fighter name to the odds dict (last name first, then full)."""
    name_l = name.lower().strip()
    if name_l in odds_dict:
        return odds_dict[name_l]
    last = name_l.split()[-1]
    for k, v in odds_dict.items():
        if last in k:
            return v
    for k, v in odds_dict.items():
        parts = name_l.split()
        if any(p in k for p in parts if len(p) > 3):
            return v
    return None


# ── V1 style assignment ───────────────────────────────────────────────────────

def get_fighter_style(fighter_name, career_df):
    """Look up a fighter's most recent style from the career stats DataFrame."""
    if career_df is None or career_df.empty:
        return "Mixed"
    mask = career_df["fighter"].str.lower() == fighter_name.lower()
    rows = career_df[mask]
    if rows.empty:
        return "Mixed"
    return rows.sort_values("date").iloc[-1].get("style", "Mixed")


def get_fighter_avg_stats(fighter_name, career_df):
    """Return the most recent career-average stat row for a fighter."""
    if career_df is None or career_df.empty:
        return {}
    mask = career_df["fighter"].str.lower() == fighter_name.lower()
    rows = career_df[mask]
    if rows.empty:
        return {}
    return rows.sort_values("date").iloc[-1].to_dict()


# ── V1 prediction for one matchup ────────────────────────────────────────────

def predict_matchup_v1(f1_name, f2_name, weight_class, f1_stats, f2_stats,
                       f1_style, f2_style, models, scalers, feature_cols, style_props):
    """
    V1 prediction logic:
      1. Try (weight_class, f1_style) model — predict win prob
      2. Fallback to (weight_class, Mixed) or any model in that weight class
      3. Final fallback: style_props lookup
    Returns float in [0, 1] = probability fighter1 wins
    """
    feature_values = []
    for col in feature_cols:
        stat_name = col.replace("diff_", "")
        v1 = f1_stats.get(f"avg_{stat_name}", 0) or 0
        v2 = f2_stats.get(f"avg_{stat_name}", 0) or 0
        feature_values.append(v1 - v2)

    X = np.array(feature_values, dtype=float).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0)  # fill missing stats with 0

    # Try exact (wc, style) match
    key = (weight_class, f1_style)
    if key in models:
        scaler = scalers[key]
        X_sc   = scaler.transform(X)
        raw    = models[key].predict(X_sc)[0]
        return float(np.clip(raw, 0.01, 0.99))

    # Fallback: (wc, Mixed)
    fallback_key = (weight_class, "Mixed")
    if fallback_key in models:
        scaler = scalers[fallback_key]
        X_sc   = scaler.transform(X)
        raw    = models[fallback_key].predict(X_sc)[0]
        return float(np.clip(raw, 0.01, 0.99))

    # Fallback: any model in this weight class
    wc_models = [(k, v) for k, v in models.items() if k[0] == weight_class]
    if wc_models:
        k, m  = wc_models[0]
        X_sc  = scalers[k].transform(X)
        raw   = m.predict(X_sc)[0]
        return float(np.clip(raw, 0.01, 0.99))

    # Final fallback: style proportions lookup
    prop_key = f"{weight_class}|{f1_style}|{f2_style}"
    if prop_key in style_props:
        return float(style_props[prop_key])

    return 0.5  # last resort: coin flip


# ── V1 discrete bet sizing ───────────────────────────────────────────────────

def v1_bet_size(edge):
    """V1 discrete sizing: 0 / 1 / 2 / 3 units based on edge thresholds."""
    for threshold, units in BET_THRESHOLDS:
        if edge >= threshold:
            return units
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def _gambly_url(prompt):
    import urllib.parse
    cleaned = prompt.lower().replace(" ", "+")
    encoded = urllib.parse.quote(cleaned, safe="+")
    return f"https://gambly.com/gambly-bot?auto=1&prompt={encoded}"


def _v1_resolve_gambly_links(bet_records):
    """
    Resolve fighter names -> DraftKings deep links via Gambly for V1 bets.
    Uses the same Playwright+name-matching approach as the V2 resolve_gambly_links.
    Returns (bet_records_with_dk_link, parlay_links_dict).
    """
    import time as _time
    if not bet_records:
        return bet_records, {"dk": None}

    BASE    = "https://gambly.com/api/proxy/core/api"
    UA      = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/122.0.0.0 Safari/537.36")
    HEADERS = {
        "User-Agent":   UA,
        "Content-Type": "application/json",
        "Referer":      "https://gambly.com/",
        "Origin":       "https://gambly.com",
    }

    token_set     = os.environ.get("GAMBLY_TOKEN_SET", "").strip()
    gambly_cookie = os.environ.get("GAMBLY_COOKIE", "").strip()
    if token_set:
        cookie_str = f"gambly_token_set={token_set}"
        print("    [V1 Gambly] Using GAMBLY_TOKEN_SET")
    elif gambly_cookie:
        cookie_str = gambly_cookie
        print("    [V1 Gambly] Using GAMBLY_COOKIE")
    else:
        print("    [V1 Gambly] No auth cookie — using Gambly fallback URLs")
        for r in bet_records:
            r["dk_link"] = _gambly_url((r.get("fighter") or "") + " moneyline")
        parlay = _gambly_url(" and ".join((r.get("fighter") or "") + " moneyline" for r in bet_records))
        return bet_records, {"dk": parlay}

    names  = [r.get("fighter") or r.get("name") or "" for r in bet_records]
    prompt = " and ".join(n + " moneyline" for n in names)
    headers = {**HEADERS, "Cookie": cookie_str}

    # Start job
    try:
        r = requests.post(
            f"{BASE}/gambly-bot/start-betslip-job",
            headers=headers,
            json={"prompt": prompt, "base64Images": [],
                  "twitterUrl": "", "saveBetslipFeedPostSync": False},
            timeout=20,
        )
        if r.status_code not in (200, 202):
            raise ValueError(f"HTTP {r.status_code}")
        job_id = r.json()["jobId"]
        print(f"    [V1 Gambly] Job started: {job_id[:8]}...")
    except Exception as e:
        print(f"    [V1 Gambly] Job start failed: {e} — using fallback")
        for r2 in bet_records:
            r2["dk_link"] = _gambly_url((r2.get("fighter") or "") + " moneyline")
        return bet_records, {"dk": _gambly_url(prompt)}

    # Poll
    for _ in range(20):
        _time.sleep(2)
        try:
            s = requests.get(f"{BASE}/gambly-bot/betslip-job-status/{job_id}",
                             headers=headers, timeout=10).json().get("status")
            if s == "completed": break
            if s == "failed":    raise ValueError("job failed")
        except Exception as e:
            print(f"    [V1 Gambly] Poll error: {e}")
            break
    else:
        print("    [V1 Gambly] Timed out — using fallback")
        for r2 in bet_records:
            r2["dk_link"] = _gambly_url((r2.get("fighter") or "") + " moneyline")
        return bet_records, {"dk": _gambly_url(prompt)}

    # Get result
    try:
        wr = requests.get(f"{BASE}/gambly-bot/webpage-result/{job_id}",
                          headers=headers, timeout=10)
        wr.raise_for_status()
        wr_data    = wr.json()
        share_url  = wr_data.get("shareUrl", "")
        qs         = wr_data.get("queryString", "")
    except Exception as e:
        print(f"    [V1 Gambly] Result fetch failed: {e}")
        for r2 in bet_records:
            r2["dk_link"] = _gambly_url((r2.get("fighter") or "") + " moneyline")
        return bet_records, {"dk": _gambly_url(prompt)}

    bet_ids = []
    for part in qs.split("&"):
        if part.startswith("betOfferIdsPoints="):
            for token in part.split("=", 1)[1].split("|"):
                bid = token.split("_")[0]
                if bid.lstrip("-").isdigit():
                    bet_ids.append(bid)

    dk_urls_by_bet_id = {}

    # Playwright: name-based matching on share page
    if share_url:
        try:
            import urllib.parse as _up
            from playwright.sync_api import sync_playwright as _spw
            with _spw() as _pw:
                _br  = _pw.chromium.launch(headless=True)
                _ctx = _br.new_context(user_agent=UA)
                _ctx.add_cookies([{
                    "name": "gambly_token_set",
                    "value": cookie_str.split("gambly_token_set=")[-1].split(";")[0],
                    "domain": "gambly.com", "path": "/",
                    "secure": True, "sameSite": "Lax",
                }])
                _pg = _ctx.new_page()
                _pg.goto(share_url, wait_until="domcontentloaded", timeout=25000)
                _time.sleep(3)
                try:
                    _dk_btn = _pg.locator("img[src*='draft-kings'], img[alt*='DraftKings']").first
                    _dk_btn.click(timeout=5000)
                    _time.sleep(2)
                except Exception:
                    pass

                def _decode_preurl(href):
                    if not href: return None
                    if "preurl=" in href:
                        _qs2 = href.split("?", 1)[1] if "?" in href else ""
                        _pm  = dict(p2.split("=", 1) for p2 in _qs2.split("&") if "=" in p2)
                        _r2  = _up.unquote(_pm.get("preurl", ""))
                        return _r2 if "draftkings" in _r2 else None
                    if "draftkings" in href and "/event/" in href:
                        return href
                    return None

                _name_to_dk = {}
                for _nel in _pg.locator(".bet-name-renderer").all():
                    try:
                        _fname  = (_nel.inner_text() or "").strip()
                        _parent = _nel.locator("..").locator("..")
                        _href   = _parent.locator("a[href*='draftkings']").first.get_attribute("href", timeout=2000) or ""
                        _real   = _decode_preurl(_href)
                        if _real and _fname:
                            _name_to_dk[_fname.lower()] = _real
                            print(f"    [V1 Gambly] {_fname} → {_real}")
                    except Exception:
                        pass

                for _name2, _bid in zip(names, bet_ids):
                    _url = _name_to_dk.get(_name2.lower())
                    if not _url:
                        _last = _name2.lower().split()[-1]
                        _url = next((u for k, u in _name_to_dk.items() if _last in k), None)
                    if _url:
                        dk_urls_by_bet_id[_bid] = _url
                _br.close()
        except Exception as e:
            print(f"    [V1 Gambly] Share page scrape failed: {e}")

    # Map URLs to fighters
    for i, r2 in enumerate(bet_records):
        name = r2.get("fighter") or r2.get("name") or ""
        bid  = bet_ids[i] if i < len(bet_ids) else None
        dk_url = dk_urls_by_bet_id.get(bid) if bid else None
        r2["dk_link"] = dk_url or _gambly_url(name + " moneyline")
        status = "✓" if dk_url else "↩ fallback"
        print(f"    [V1 Gambly] {status} {name}: {r2['dk_link']}")

    # Build parlay URL
    parlay_url = None
    if len(bet_ids) >= 2:
        try:
            all_outcomes, event_id = [], None
            for bid in bet_ids:
                url = dk_urls_by_bet_id.get(bid, "")
                if "outcomes=" in url:
                    all_outcomes.append(url.split("outcomes=")[1].split("&")[0])
                    if not event_id and "/event/" in url:
                        event_id = url.split("/event/")[1].split("?")[0]
            if all_outcomes and event_id:
                parlay_url = f"https://sportsbook.draftkings.com/event/{event_id}?outcomes={'+'.join(all_outcomes)}"
        except Exception:
            pass

    parlay_links = {"dk": parlay_url or _gambly_url(prompt)}
    print(f"    [V1 Gambly] Parlay: {parlay_links['dk']}")
    return bet_records, parlay_links


def main():
    print("=" * 60)
    print("V1 Model — Friday Prediction Runner")
    print("=" * 60)

    # 1. Check for upcoming event
    print("\n[1] Checking UFCStats for upcoming event…")
    event_info = get_upcoming_event()
    if event_info is None:
        print("  No UFC event scheduled this week. Exiting.")
        return
    event_name, event_date, fights = event_info
    print(f"  Found: {event_name} ({event_date}) — {len(fights)} bouts")

    if not fights:
        print("  No fight card data available yet. Exiting.")
        return

    # 2. Scrape odds
    print("\n[2] Scraping BestFightOdds…")
    odds_dict = scrape_bestfightodds(event_name)

    # 3. Load V1 model bundle
    print("\n[3] Loading V1 model bundle from UFC_Model repo…")
    bundle = read_pickle_from_github("data/v1_model_bundle.pkl", repo=V1_REPO)
    if bundle is None:
        print("  [ERROR] No V1 model bundle found. Run v1_train_model.py first.")
        return
    models       = bundle["models"]
    scalers      = bundle["scalers"]
    feature_cols = bundle["feature_cols"]
    style_props  = bundle["style_props"]
    print(f"  Loaded {len(models)} group models (trained {bundle.get('trained_at', '?')})")

    # 4. Load career stats
    print("\n[4] Loading career stats…")
    career_df, _ = read_csv_from_github("data/career_styled_v1.csv", repo=V1_REPO)
    if career_df.empty:
        print("  [WARN] No career stats found — style fallback to Mixed")

    # 5. Generate predictions
    print("\n[5] Generating predictions…")
    predictions = []

    MIN_FIGHTS = 3

    for bout in fights:
        f1, f2, wc = bout["fighter1"], bout["fighter2"], bout["weight_class"]
        print(f"  {f1} vs {f2} ({wc})")

        f1_stats = get_fighter_avg_stats(f1, career_df)
        f2_stats = get_fighter_avg_stats(f2, career_df)

        # Skip fighters with insufficient data
        if not f1_stats or not f2_stats:
            print(f"    [SKIP] No career data found for {f1 if not f1_stats else f2}")
            continue
        # Count rows in career_df = number of fights on record
        f1_fights = len(career_df[career_df["fighter"].str.lower() == f1.lower()]) if not career_df.empty else 0
        f2_fights = len(career_df[career_df["fighter"].str.lower() == f2.lower()]) if not career_df.empty else 0
        if f1_fights < MIN_FIGHTS or f2_fights < MIN_FIGHTS:
            short = f1 if f1_fights < MIN_FIGHTS else f2
            print(f"    [SKIP] Insufficient data for {short} ({min(f1_fights,f2_fights)} fights < {MIN_FIGHTS})")
            continue

        f1_style = get_fighter_style(f1, career_df)
        f2_style = get_fighter_style(f2, career_df)

        # Predict for both fighters
        p1 = predict_matchup_v1(f1, f2, wc, f1_stats, f2_stats,
                                 f1_style, f2_style, models, scalers,
                                 feature_cols, style_props)
        p2 = 1.0 - p1

        # Get odds
        o1 = match_fighter_odds(f1, odds_dict)
        o2 = match_fighter_odds(f2, odds_dict)

        for fighter, pred_prob, opp_prob, odds_raw, style, opp_style in [
            (f1, p1, p2, o1, f1_style, f2_style),
            (f2, p2, p1, o2, f2_style, f1_style),
        ]:
            implied = american_to_implied(odds_raw) if odds_raw is not None else None
            edge    = (pred_prob - implied) if implied is not None else None
            bet     = v1_bet_size(edge) if edge is not None else 0

            predictions.append({
                "event":            event_name,
                "event_date":       str(event_date),
                "fighter":          fighter,
                "opponent":         f2 if fighter == f1 else f1,
                "weight_class":     wc,
                "style":            style,
                "opp_style":        opp_style,
                "predicted_prob":   round(pred_prob, 4),
                "Odds":             odds_raw if odds_raw is not None else "N/A",
                "implied_prob":     round(implied, 4) if implied is not None else None,
                "betting_edge":     round(edge, 4)    if edge    is not None else None,
                "bet_size":         bet,
            })

    pred_df = pd.DataFrame(predictions)
    bets    = pred_df[pred_df["bet_size"] > 0]
    print(f"\n  Total fighters: {len(pred_df)}")
    print(f"  Bets recommended: {len(bets)}")
    if len(bets):
        print(bets[["fighter", "predicted_prob", "implied_prob", "betting_edge", "bet_size"]].to_string(index=False))

    # 5b. Resolve DraftKings deep links via Gambly
    print("\n[5b] Resolving DraftKings deep links via Gambly...")
    bet_records = bets.to_dict("records")
    bet_records, parlay_links = _v1_resolve_gambly_links(bet_records)
    # Write dk_link back into pred_df
    dk_map = {r["fighter"]: r.get("dk_link", "") for r in bet_records}
    pred_df["dk_link"]        = pred_df["fighter"].map(dk_map).fillna("")
    pred_df["dk_parlay_link"] = parlay_links.get("dk", "")

    # 6. Push to UFC_Model repo
    print("\n[6] Pushing predictions to UFC_Model repo…")
    safe_name = re.sub(r"[^a-z0-9_]", "_", event_name.lower())
    write_csv_to_github(
        pred_df,
        f"predictions/v1_predictions_{safe_name}.csv",
        f"feat: V1 predictions for {event_name}",
        repo=V1_REPO,
    )
    # Also write/overwrite a fixed "latest" file so PythonAnywhere can always find it
    write_csv_to_github(
        pred_df,
        "predictions/v1_predictions_latest.csv",
        f"chore: update latest V1 predictions ({event_name})",
        repo=V1_REPO,
    )

    print(f"\n✓ V1 predictions complete — {len(bets)} bet(s) recommended for {event_name}.")


def run_test():
    """
    --test mode: tests every function in v1_predict without writing to GitHub.
    Tests:
      1. UFCStats upcoming event scraping (live)
      2. BestFightOdds scraping (live)
      3. Odds matching logic (offline)
      4. Prediction logic with synthetic career data (offline)
      5. Bet sizing logic (offline)
    """
    print("=== TEST MODE: v1_predict ===\n")
    all_pass = True

    # [1] UFCStats upcoming event
    print("[1] UFCStats upcoming event scraping...")
    event_info = get_upcoming_event()
    if event_info is None:
        print("  ○ No UFC event this week — expected mid-week")
        print("  ✓ Function returned None cleanly (correct behaviour)")
    else:
        event_name, event_date, fights = event_info
        print(f"  ✓ Found: {event_name} ({event_date}) — {len(fights)} bouts")
        for b in fights[:3]:
            print(f"    {b['fighter1']} vs {b['fighter2']} ({b['weight_class']})")

    # [2] BestFightOdds scraping (live)
    print("\n[2] BestFightOdds odds scraping...")
    try:
        name = event_info[0] if event_info else "UFC"
        odds_dict = scrape_bestfightodds(name)
        if not odds_dict:
            print("  ○ No odds found — expected if no upcoming event")
        else:
            sample = list(odds_dict.items())[:5]
            print(f"  ✓ {len(odds_dict)} fighters with odds")
            for fighter, odds in sample:
                print(f"    {fighter}: {odds}")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [3] Odds matching logic (offline)
    print("\n[3] Odds matching logic (offline)...")
    try:
        # match_fighter_odds expects lowercased keys (mirrors scrape_bestfightodds output)
        test_dict = {
            "jon jones":        "+150",
            "stipe miocic":     "-175",
            "charles oliveira": "-200",
        }
        cases = [
            ("Jon Jones",         "+150"),   # casing normalised internally
            ("jones",             "+150"),   # partial last name
            ("Unknown Fighter",   None),     # no match
        ]
        all_match = True
        for name, expected in cases:
            got = match_fighter_odds(name, test_dict)
            ok  = got == expected
            print(f"  {'✓' if ok else '✗'}  '{name}' → {got!r} (expect {expected!r})")
            if not ok: all_match = False; all_pass = False
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [4] Prediction logic with synthetic career data (offline)
    print("\n[4] Prediction logic — synthetic career data (offline)...")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        # Synthetic career_df matching what v1_train_model produces
        career_data = []
        for fighter, wc, style in [
            ("Jon Jones",    "Heavyweight", "Wrestler"),
            ("Stipe Miocic", "Heavyweight", "Striker"),
        ]:
            for i in range(8):
                row = {"fighter": fighter, "date": pd.Timestamp(f"2023-0{i+1}-01"),
                       "weight_class": wc, "style": style}
                for col in STAT_COLS:
                    row[f"avg_{col}"] = np.random.uniform(0.5, 5.0)
                career_data.append(row)
        career_df = pd.DataFrame(career_data)

        f1_stats = get_fighter_avg_stats("Jon Jones",    career_df)
        f2_stats = get_fighter_avg_stats("Stipe Miocic", career_df)
        assert f1_stats is not None, "f1_stats is None"
        assert f2_stats is not None, "f2_stats is None"
        print(f"  ✓ get_fighter_avg_stats: {len(f1_stats)} avg columns")

        # Build a minimal synthetic model
        feat_cols = [f"diff_{c}" for c in STAT_COLS]
        X_train   = np.random.randn(20, len(feat_cols))
        y_train   = np.random.randint(0, 2, 20).astype(float)
        sc = StandardScaler(); X_sc = sc.fit_transform(X_train)
        lr = LinearRegression(); lr.fit(X_sc, y_train)
        models  = {("Heavyweight","Wrestler"): lr, ("Heavyweight","Striker"): lr}
        scalers = {("Heavyweight","Wrestler"): sc, ("Heavyweight","Striker"): sc}

        prob = predict_matchup_v1(
            "Jon Jones", "Stipe Miocic", "Heavyweight",
            f1_stats, f2_stats, "Wrestler", "Striker",
            models, scalers, feat_cols, {}
        )
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
        print(f"  ✓ predict_matchup_v1: prob={prob:.4f} (in [0,1])")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [5] Bet sizing thresholds (offline)
    print("\n[5] v1_bet_size() thresholds...")
    try:
        cases = [
            (0.02,  0, "edge < 5pp → 0u"),
            (0.06,  1, "edge 5-12pp → 1u"),
            (0.14,  2, "edge 12-20pp → 2u"),
            (0.22,  3, "edge > 20pp → 3u"),
            (-0.10, 0, "negative edge → 0u"),
        ]
        all_ok = True
        for edge, expected, desc in cases:
            got = v1_bet_size(edge)
            ok  = got == expected
            print(f"  {'✓' if ok else '✗'}  {desc}: got {got}u")
            if not ok: all_ok = False; all_pass = False
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [6] american_to_implied()
    print("\n[6] american_to_implied() accuracy...")
    try:
        cases = [
            (-150, round(150/250, 4)),
            (+130, round(100/230, 4)),
            (-110, round(110/210, 4)),
            (+100, round(100/200, 4)),
        ]
        for odds, expected in cases:
            got = round(american_to_implied(odds), 4)
            ok  = abs(got - expected) < 0.001
            print(f"  {'✓' if ok else '✗'}  {odds:+d} → {got:.4f} (expect {expected:.4f})")
            if not ok: all_pass = False
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    print()
    print("=" * 60)
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED — see above'}")
    print("=" * 60)
    print("=== END TEST ===")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        main()
