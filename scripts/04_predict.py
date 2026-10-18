#!/usr/bin/env python3
"""
04_predict.py
OctaStats V2 — Friday fight week prediction runner

Runs Friday at noon if a UFC event is scheduled this week. Steps:
  1. Check UFCStats upcoming events — skip if no fight this week
  2. Scrape odds from BestFightOdds (DraftKings preferred)
  3. Load model bundle and most recent career stats
  4. Build matchup features for this week's card (mirrors 03_train_model exactly)
  5. Run dynamic logit win predictions
  6. Run multinomial finish-type predictions (KO/TKO | Sub | Decision)
  7. Apply style-shift detection (display if confidence > 0.78)
  8. Compute continuous bet sizes via bounded Kelly criterion
  9. Run prop model predictions with strict thresholds
 10. Push predictions CSV and fight titles JSON to GitHub

Environment variables:
    ZWINSHIP_PAT  — GitHub personal access token for zwinship account
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
from datetime import datetime, timedelta, timezone
from scipy.special import expit
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from name_match import match_name, norm_name
from bfo_odds import parse_bfo_moneylines, match_bout_odds
from rankings import relabel_history, division_from_heading, current_rank
from dk_links import resolve_gambly_links, _gambly_url, _fallback_links
from ufcstats_browser import fetch_html, SEL_EVENT_LIST, SEL_EVENT_PAGE

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
import sys as _sys
if not GITHUB_TOKEN and "--test" not in _sys.argv:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
REPO_NAME    = "OctaStats_UFC_Model_V2"
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
if not GITHUB_TOKEN:
    GH_HEADERS.pop("Authorization")   # --test runs read the public repo anonymously
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

MAX_UNITS      = 5.0
STEEPNESS      = 3.5
MIN_EDGE_PCT   = 0.04
PROP_MIN_EDGE  = 0.15
PROP_MIN_CONF  = 0.70
PROP_MAX_IMPL  = 0.65
STYLE_ORDER    = ["BJJ", "Mixed", "Muay_Thai", "Sniper", "Striker", "Wrestler"]
FINISH_CLASSES = ["KO_TKO", "Submission", "Decision"]


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path, repo_name=None):
    rn   = repo_name or REPO_NAME
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{rn}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        return pd.DataFrame(), None
    data = resp.json()
    sha  = data["sha"]
    raw_content = data.get("content", "").replace("\n", "").strip()
    if not raw_content and data.get("download_url"):
        raw_resp = requests.get(url, headers={**GH_HEADERS, "Accept": "application/vnd.github.raw"}, timeout=120)
        if raw_resp.status_code == 200:
            return pd.read_csv(io.StringIO(raw_resp.text)), sha
        return pd.DataFrame(), None
    content = base64.b64decode(raw_content).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), sha


def read_pickle_from_github(repo_path):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    # Raw media type works for files over the 1 MB inline-content limit
    resp = requests.get(url, headers={**GH_HEADERS, "Accept": "application/vnd.github.raw"}, timeout=120)
    if resp.status_code != 200:
        print(f"  [ERROR] Could not load {repo_path}: {resp.status_code}")
        return None
    return pickle.loads(resp.content)


def write_csv_to_github(df, repo_path, message, repo_name=None, sha=None):
    rn      = repo_name or REPO_NAME
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{rn}/contents/{repo_path}"
    if sha is None:
        check = requests.get(url, headers=GH_HEADERS)
        sha   = check.json().get("sha") if check.status_code == 200 else None
    payload = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    ok   = resp.status_code in (200, 201)
    print(f"  {'✓' if ok else '✗'} GitHub: {repo_path} ({rn})")
    return ok


def write_json_to_github(obj, repo_path, message, repo_name=None):
    import json
    rn      = repo_name or REPO_NAME
    content = base64.b64encode(json.dumps(obj, indent=2).encode()).decode()
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{rn}/contents/{repo_path}"
    check   = requests.get(url, headers=GH_HEADERS)
    sha     = check.json().get("sha") if check.status_code == 200 else None
    payload = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    ok   = resp.status_code in (200, 201)
    print(f"  {'✓' if ok else '✗'} GitHub: {repo_path} ({rn})")


# ── Step 1: Check for upcoming event ─────────────────────────────────────────

def get_upcoming_event():
    url  = f"{BASE_URL}/statistics/events/upcoming"
    print("  Fetching UFCStats upcoming events (Playwright)...")
    html = fetch_html(url, SEL_EVENT_LIST)
    if not html:
        raise RuntimeError("UFCStats upcoming-events page did not load "
                           "(browser check not cleared) — refusing to report 'no event'.")
    soup      = BeautifulSoup(html, "lxml")
    rows      = soup.select("tr.b-statistics__table-row")
    today     = datetime.now(timezone.utc).date()
    next_week = today + timedelta(days=7)
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
        try:
            event_date = datetime.strptime(date_span.get_text(strip=True), "%B %d, %Y").date()
        except ValueError:
            continue
        if today <= event_date <= next_week:
            event_url = link_el.get("href", "")
            return {
                "name":   link_el.get_text(strip=True),
                "date":   event_date.isoformat(),
                "url":    event_url,
                "fights": get_fights_from_event(event_url),
            }
    return None


WEIGHT_CLASSES = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight",
                  "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
                  "Heavyweight", "Catch Weight"]


def get_fights_from_event(event_url):
    """
    [(fighter_a, fighter_b, weight_class, info), ...] in card order.
    info = {"title_bout": 0/1, "number_of_rounds": 3/5}. Title bouts carry a
    belt icon in the weight-class cell; the first bout is the main event
    (always five rounds).
    """
    print(f"  Fetching event page (Playwright): {event_url}")
    html = fetch_html(event_url, SEL_EVENT_PAGE)
    if not html:
        raise RuntimeError(f"UFCStats event page did not load: {event_url}")
    soup  = BeautifulSoup(html, "lxml")
    bouts = []
    for row in soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover"):
        tds = row.find_all("td")
        if len(tds) < 7:
            continue
        links = tds[1].select("p a") or row.select("a.b-link")
        if len(links) < 2:
            continue
        f1, f2  = links[0].get_text(strip=True), links[1].get_text(strip=True)
        wc_cell = tds[6]
        wc_text = wc_cell.get_text(" ", strip=True)
        wc = "Unknown"
        # Longest name first so "Light Heavyweight" is not read as "Heavyweight"
        for wc_name in sorted(WEIGHT_CLASSES, key=len, reverse=True):
            if wc_name.lower() in wc_text.lower():
                wc = ("Women's " + wc_name) if "women" in wc_text.lower() else wc_name
                break
        is_title = int(bool(wc_cell.select_one("img[src*='belt']")))
        info = {"title_bout": is_title,
                "number_of_rounds": 5 if (is_title or not bouts) else 3}
        if f1 and f2:
            bouts.append((f1, f2, wc, info))
    return bouts


# ── Step 2: Scrape odds from BestFightOdds ────────────────────────────────────

# Prop-specific sportsbook priority (DK first, then fallbacks)
PROP_BOOK_PREF = ["draftkings", "caesars", "fanduel", "betway"]

# Prop label patterns → internal prop_type key
PROP_PATTERNS = [
    (re.compile(r"wins by tko/ko$",                  re.IGNORECASE), "wins_by_ko"),
    (re.compile(r"wins by submission$",              re.IGNORECASE), "wins_by_sub"),
    (re.compile(r"wins by decision$",               re.IGNORECASE), "wins_by_dec"),
    (re.compile(r"^fight goes to decision$",         re.IGNORECASE), "fight_goes_distance"),
    (re.compile(r"^fight doesn.t go to decision$",   re.IGNORECASE), "fight_inside_distance"),
]
PROP_SKIP_WORDS = {"either", "any other"}


def _bfo_get_odds_at_col(cells, col_idx):
    """Return American odds string at a column index, or None. Checks <span> first."""
    if col_idx is None or col_idx >= len(cells):
        return None
    cell = cells[col_idx]
    span = cell.find("span")
    t = span.get_text(strip=True) if span else cell.get_text(strip=True)
    return t if re.match(r'^[+-]\d{2,4}$', t) else None


def _bfo_name_tokens(full_name):
    return set(full_name.lower().split()) if full_name else set()


def scrape_bestfightodds():
    """
    Single fetch of BestFightOdds. Returns:
        (moneyline_df, prop_odds_df)

    moneyline_df  — columns: fighter, odds
    prop_odds_df  — columns: fighter_a, fighter_b, matchup, prop_type,
                             prop_label, odds_str, market_implied_prob,
                             sportsbook_used
    """
    url  = "https://www.bestfightodds.com/"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"  [WARN] BestFightOdds returned status {resp.status_code}")
        return pd.DataFrame(columns=["fighter", "odds"]), pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")

    ml_df = parse_bfo_moneylines(soup)
    books = ml_df["book"].value_counts().to_dict() if not ml_df.empty else {}
    print(f"  Scraped {len(ml_df)} moneyline odds across "
          f"{ml_df['matchup_id'].nunique() if not ml_df.empty else 0} matchups (books: {books})")

    # ── Prop odds scraping ────────────────────────────────────────────────────
    prop_records = []
    tables = (soup.select("div.div-content table") or
              soup.select("table.moneyline-table") or
              soup.find_all("table"))

    for table in tables:
        # Detect book columns from <thead> <th> <a> anchors
        local_book_col_map = {}
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                for col_idx, th_cell in enumerate(header_row.find_all("th")):
                    anchor = th_cell.find("a")
                    if anchor:
                        bname = anchor.get_text(strip=True).lower()
                        for b in PROP_BOOK_PREF:
                            if b in bname and b not in local_book_col_map:
                                local_book_col_map[b] = col_idx
                                break

        if not local_book_col_map:
            # Fallback: scan first few rows
            for row in table.find_all("tr")[:5]:
                cells = row.find_all(["td", "th"])
                texts = [c.get_text(strip=True).lower() for c in cells]
                for i, t in enumerate(texts):
                    for b in PROP_BOOK_PREF:
                        if b in t and b not in local_book_col_map:
                            local_book_col_map[b] = i

        if not local_book_col_map:
            continue

        current_fa, current_fb = None, None
        tbody = table.find("tbody") or table

        for row in tbody.find_all("tr"):
            th = row.find("th")
            if not th:
                continue
            th_text  = th.get_text(strip=True)
            th_lower = th_text.lower()

            if any(skip in th_lower for skip in PROP_SKIP_WORDS):
                continue

            # Fighter name rows
            name_span   = th.select_one("a span.t-b-fcc")
            name_anchor = th.select_one("a") if not name_span else None
            if name_span or (name_anchor and name_anchor.get("href", "").startswith("/fighters/")):
                name = (name_span or name_anchor).get_text(strip=True)
                if name and len(name) > 2:
                    if current_fa is None:
                        current_fa = name
                    elif current_fb is None:
                        current_fb = name
                    else:
                        current_fa = name
                        current_fb = None
                continue

            if current_fa is None:
                continue

            label_tokens = set(re.split(r'[\s/]+', th_lower))

            def _matches(fname):
                return bool(label_tokens & _bfo_name_tokens(fname))

            for pattern, prop_key in PROP_PATTERNS:
                if not pattern.search(th_text):
                    continue

                all_cells = row.find_all(["td", "th"])

                # Per-row fallback: try each book in priority order
                odds_str  = None
                book_used = "none"
                for book in PROP_BOOK_PREF:
                    col = local_book_col_map.get(book)
                    candidate = _bfo_get_odds_at_col(all_cells, col)
                    if candidate:
                        odds_str  = candidate
                        book_used = book.title()
                        break

                # Convert to implied probability
                imp_prob = None
                if odds_str:
                    n = float(odds_str)
                    imp_prob = round(100 / (n + 100) if n > 0 else abs(n) / (abs(n) + 100), 4)

                # Assign fighter_a / fighter_b for per-fighter props
                if prop_key in ("wins_by_ko", "wins_by_sub", "wins_by_dec"):
                    if _matches(current_fa):
                        assigned = "a"
                    elif current_fb and _matches(current_fb):
                        assigned = "b"
                    else:
                        assigned = "unknown"
                    type_map = {
                        "wins_by_ko":  {"a": "fighter_a_ko",  "b": "fighter_b_ko",  "unknown": None},
                        "wins_by_sub": {"a": "fighter_a_sub", "b": "fighter_b_sub", "unknown": None},
                        "wins_by_dec": {"a": "fighter_a_dec", "b": "fighter_b_dec", "unknown": None},
                    }
                    prop_type = type_map[prop_key][assigned]
                    if prop_type is None:
                        break  # couldn't assign — skip row
                else:
                    prop_type = prop_key

                prop_records.append({
                    "fighter_a":           current_fa or "",
                    "fighter_b":           current_fb or "",
                    "matchup":             f"{current_fa} vs {current_fb}",
                    "prop_type":           prop_type,
                    "prop_label":          th_text,
                    "odds_str":            odds_str or "n/a",
                    "market_implied_prob": imp_prob,
                    "sportsbook_used":     book_used,
                })
                break

    prop_df = pd.DataFrame(prop_records) if prop_records else pd.DataFrame(
        columns=["fighter_a", "fighter_b", "matchup", "prop_type",
                 "prop_label", "odds_str", "market_implied_prob", "sportsbook_used"])

    # Debug: print all decision props so we can verify correct fighter/odds assignment
    if not prop_df.empty:
        dec_props = prop_df[prop_df["prop_type"].isin(["fighter_a_dec", "fighter_b_dec"])]
        if not dec_props.empty:
            print(f"  [DEBUG] Decision prop assignments:")
            for _, r in dec_props.iterrows():
                print(f"    {r['prop_type']:20s} | {r['prop_label']:40s} | {r['odds_str']:>7} | imp={r['market_implied_prob']}")

    with_odds = prop_df[prop_df["market_implied_prob"].notna()] if not prop_df.empty else prop_df
    books_used = set(with_odds["sportsbook_used"].unique()) if not with_odds.empty else set()
    print(f"  Scraped {len(with_odds)} prop odds across "
          f"{prop_df['matchup'].nunique() if not prop_df.empty else 0} matchups "
          f"(books: {', '.join(books_used) or 'none'})")

    return ml_df, prop_df


def convert_odds(odds_str):
    if pd.isna(odds_str):
        return np.nan
    try:
        return float(str(odds_str).strip())
    except ValueError:
        return np.nan


def implied_prob_from_odds(odds):
    if np.isnan(odds):
        return np.nan
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


# ── Step 3-4: Build features for upcoming fights ──────────────────────────────

def _load_train_module():
    """03_train_model.py holds the canonical career-stat / style code."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_model", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_train_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_weight_class(weight_class, title_bout):
    """Card weight class → the label the model was trained on.
    fight_stats stores title fights as 'UFC <Division> Title' and
    03_train_model.load_raw title-cases every label ("Women'S Flyweight")."""
    wc = f"UFC {weight_class} Title" if title_bout else weight_class
    return pd.Series([wc]).str.strip().str.replace(r"\s+", " ", regex=True).str.title().iloc[0]


def build_current_career(raw_df, fights, event_date, rankings_df=None):
    """
    Career state of every fighter on the card GOING INTO this event.

    career_styled.csv rows are pre-fight snapshots (leak-free for training),
    so a fighter's last row excludes their most recent fight. Here each
    fighter gets a placeholder row dated on the event and the SAME functions
    used in training (compute_career_stats / assign_fighting_styles) produce
    their stats — every completed fight included — and their locked style.
    Physical attributes, current age and current rankings are attached.
    """
    tm  = _load_train_module()
    df  = tm.load_raw(raw_df)
    ev  = pd.Timestamp(event_date)
    placeholders = []
    for fa, fb, wc, info in fights:
        url = f"upcoming::{fa}::{fb}"
        for f, o in ((fa, fb), (fb, fa)):
            placeholders.append({
                "fight_url": url, "event_date": ev, "fighter": f, "opponent": o,
                "weight_class": _train_weight_class(wc, info["title_bout"]),
                "won": 0, "title_bout": info["title_bout"],
                "number_of_rounds": info["number_of_rounds"],
            })
    all_df = pd.concat([df, pd.DataFrame(placeholders)], ignore_index=True)
    career = tm.compute_career_stats(all_df)
    styled = tm.assign_fighting_styles(all_df, career)
    cur    = styled[styled["fight_url"].astype(str).str.startswith("upcoming::")].copy()

    # Latest known physical attributes; age rolled forward to the event date
    hist = df.sort_values("event_date")
    phys = hist.groupby("fighter")[["height_cms", "reach_cms"]].last()
    aged = hist.dropna(subset=["age"]).groupby("fighter")[["age", "event_date"]].last()
    cur["height_cms"] = cur["fighter"].map(phys["height_cms"])
    cur["reach_cms"]  = cur["fighter"].map(phys["reach_cms"])
    cur["age"]        = cur["fighter"].map(
        aged["age"] + (ev - aged["event_date"]).dt.days / 365.25)

    # Current rankings (champion = 0, unranked = NaN — as in training)
    division = {}
    for fa, fb, wc, info in fights:
        division[fa] = division[fb] = division_from_heading(wc)
    cur["fighter_rank_wc"] = [
        current_rank(rankings_df, f, division.get(f)) for f in cur["fighter"]]
    cur["fighter_rank_pfp"] = [
        current_rank(rankings_df, f, ("Women's " if str(division.get(f, "")).startswith("Women")
                                      else "") + "Pound-for-Pound")
        for f in cur["fighter"]]
    cur["fight_key"] = cur["fight_url"]
    return cur.set_index(["fight_key", "fighter"]), df


def _per_fight_wc_stats(df):
    """Per-weight-class mean/std of single-fight stats — the scale the
    style-shift model's std_* features were standardised on in training."""
    stat_cols = [c for c in _CAREER_STAT_COLS if c in df.columns]
    agg = df.groupby("weight_class")[stat_cols].agg(["mean", "std"])
    return agg


_CAREER_STAT_COLS = [
    "kd", "sig_str_landed", "sig_str_att", "sig_str_pct",
    "total_str_landed", "total_str_att", "td_landed", "td_att", "td_pct",
    "sub_att", "reversals", "ctrl_seconds", "head_landed", "head_att",
    "body_landed", "body_att", "leg_landed", "leg_att",
    "distance_landed", "distance_att", "clinch_landed", "clinch_att",
    "ground_landed", "ground_att",
]


def _live_method_probs(fa, fb, prop_odds_df):
    """{fighter: {"ko": p, "sub": p, "dec": p}} from live BFO method props."""
    out = {fa: {}, fb: {}}
    if prop_odds_df is None or prop_odds_df.empty:
        return out
    for _, row in prop_odds_df.iterrows():
        names = [str(row["fighter_a"]), str(row["fighter_b"])]
        if not (match_name(fa, names) and match_name(fb, names)):
            continue
        method = {"ko": "ko", "sub": "sub", "dec": "dec"}.get(str(row["prop_type"]).rsplit("_", 1)[-1])
        imp    = row.get("market_implied_prob")
        if method is None or imp is None or pd.isna(imp) or str(row["prop_type"]).startswith("fight_"):
            continue
        subject = row["fighter_a"] if str(row["prop_type"]).startswith("fighter_a") else row["fighter_b"]
        who = fa if match_name(fa, [subject]) else fb if match_name(fb, [subject]) else None
        if who is not None:
            out[who][method] = float(imp)
    return out


def build_upcoming_features(fights, current, odds_df, model_bundle, prop_odds_df=None, train_df=None):
    """
    Mirrors the feature construction in 03_train_model.build_matchup_features,
    using each fighter's career state going into this event (build_current_career).
    """
    wc_stats = _per_fight_wc_stats(train_df) if train_df is not None else None
    # Weight-class share of wins by each method (fallback for missing props;
    # same shape as the model-imputed implied_prob_* used in training)
    method_prior = {}
    if train_df is not None and "method" in train_df.columns:
        winners = train_df[train_df["won"] == 1]
        m = winners["method"].astype(str).str.lower()
        for wc, idx in winners.groupby("weight_class").groups.items():
            mm = m.loc[idx]
            if len(mm) >= 10:
                method_prior[wc] = {"ko": mm.str.contains("ko").mean(),
                                    "sub": mm.str.contains("sub").mean(),
                                    "dec": mm.str.contains("dec").mean()}

    records = []

    for (fa, fb, weight_class, info) in fights:
        key = f"upcoming::{fa}::{fb}"
        if (key, fa) not in current.index or (key, fb) not in current.index:
            print(f"  [WARN] Missing career data for {fa} or {fb}. Skipping fight.")
            continue
        row_a = current.loc[(key, fa)]
        row_b = current.loc[(key, fb)]

        # Skip fighters with fewer than 2 UFC fights in our dataset
        MIN_FIGHTS = 2
        ra_fights = int(row_a.get("career_fights_before", 0) or 0)
        rb_fights = int(row_b.get("career_fights_before", 0) or 0)
        if ra_fights < MIN_FIGHTS or rb_fights < MIN_FIGHTS:
            short = fa if ra_fights < MIN_FIGHTS else fb
            n     = min(ra_fights, rb_fights)
            print(f"  [SKIP] Insufficient data for {short} ({n} fights in dataset < {MIN_FIGHTS}). Skipping fight.")
            continue

        raw_a, raw_b = match_bout_odds(fa, fb, odds_df)
        odds_a = convert_odds(raw_a)
        odds_b = convert_odds(raw_b)

        # Both lines are required: the market implied prob is the model's
        # strongest input and the edge is measured against the vig-normalised
        # pair, so a one-sided line cannot be priced correctly.
        if np.isnan(odds_a) or np.isnan(odds_b):
            missing = " and ".join(n for n, o in [(fa, odds_a), (fb, odds_b)] if np.isnan(o))
            print(f"  [WARN] No sportsbook odds for {missing} ({fa} vs {fb}). Skipping fight.")
            continue

        wc_train = row_a.get("weight_class")
        live     = _live_method_probs(fa, fb, prop_odds_df)
        ip       = {fa: implied_prob_from_odds(odds_a), fb: implied_prob_from_odds(odds_b)}

        def _method_prob(name, row, method):
            if method in live[name]:
                return live[name][method]
            prior = method_prior.get(wc_train, {}).get(method, np.nan)
            col   = {"ko": "career_win_by_ko", "sub": "career_win_by_sub", "dec": "career_win_by_dec"}[method]
            wins  = float(row.get("career_wins", 0) or 0)
            rate  = float(row.get(col, 0) or 0) / wins if wins > 0 else prior
            alpha = min(wins / 10.0, 1.0)
            return (alpha * rate + (1 - alpha) * prior) * ip[name]

        for (fighter, opponent, row_f, row_o, odds_f, odds_o) in [
            (fa, fb, row_a, row_b, odds_a, odds_b),
            (fb, fa, row_b, row_a, odds_b, odds_a),
        ]:
            feat = {
                "fighter":            fighter,
                "opponent":           opponent,
                "weight_class":       weight_class,
                "weight_class_train": wc_train,
                "career_style":       row_f.get("career_style", "Mixed"),
                "opp_career_style":   row_o.get("career_style", "Mixed"),
                "odds_str":           str(int(odds_f)),
                "odds_numeric":       odds_f,
            }

            # Career stat diffs
            career_cols = [c for c in row_f.index
                           if c.startswith("career_") or c in (
                               "current_win_streak", "current_lose_streak",
                               "longest_win_streak", "avg_scorecard_margin")]
            for col in career_cols:
                try:
                    feat[f"diff_{col}"] = float(row_f.get(col, np.nan)) - float(row_o.get(col, np.nan))
                except (TypeError, ValueError):
                    feat[f"diff_{col}"] = np.nan

            # Physical + ranking diffs
            for col in ["height_cms", "reach_cms", "age", "fighter_rank_wc", "fighter_rank_pfp"]:
                try:
                    feat[f"diff_{col}"] = float(row_f.get(col, np.nan)) - float(row_o.get(col, np.nan))
                except (TypeError, ValueError):
                    feat[f"diff_{col}"] = np.nan

            # Market implied prob (from current sportsbook odds)
            ip_f = implied_prob_from_odds(odds_f)
            ip_o = implied_prob_from_odds(odds_o)
            feat["diff_implied_prob"]     = ip_f - ip_o
            feat["implied_prob_a"]        = ip_f
            feat["implied_prob_b"]        = ip_o
            feat["implied_prob_source_a"] = 1   # live sportsbook odds = market
            feat["diff_historical_odds"]  = odds_f - odds_o

            # Method-of-victory implied probs: live BFO props when posted,
            # otherwise the career-rate estimate used for unpriced training rows
            for method in ("ko", "sub", "dec"):
                pf = _method_prob(fighter, row_f, method)
                po = _method_prob(opponent, row_o, method)
                feat[f"diff_implied_prob_{method}"] = pf - po

            # Fight context
            feat["title_bout"]       = int(info["title_bout"])
            feat["empty_arena"]      = 0
            feat["number_of_rounds"] = int(info["number_of_rounds"])

            # Interaction terms (mirror 03_train_model)
            diff_ws = feat.get("diff_current_win_streak", 0.0) or 0.0
            feat["odds_x_win_streak"] = (feat.get("diff_implied_prob", 0) or 0) * diff_ws
            feat["reach_x_ip"]  = (feat.get("diff_reach_cms", 0) or 0) * (feat.get("diff_implied_prob", 0) or 0)
            feat["height_x_ip"] = (feat.get("diff_height_cms", 0) or 0) * (feat.get("diff_implied_prob", 0) or 0)
            feat["title_x_fights"] = feat["title_bout"] * (feat.get("diff_career_fights_before", 0) or 0)
            feat["title_x_rank"]   = feat["title_bout"] * (feat.get("diff_fighter_rank_wc", 0) or 0)
            feat["ko_x_scorecard"] = (feat.get("diff_career_win_by_ko", 0) or 0) * (feat.get("diff_avg_scorecard_margin", 0) or 0)

            # Style-shift model inputs: expected in-fight stats (career
            # averages) standardised on the weight class's per-fight scale,
            # plus the career-style one-hot — the training feature layout
            for c in _CAREER_STAT_COLS:
                val = row_f.get(f"career_avg_{c}", np.nan)
                try:
                    mu, sd = wc_stats.loc[wc_train, (c, "mean")], wc_stats.loc[wc_train, (c, "std")]
                    feat[f"std_{c}"] = (float(val) - mu) / sd if sd and sd > 0 else 0.0
                except (KeyError, TypeError, ValueError, AttributeError):
                    feat[f"std_{c}"] = 0.0
            for s in STYLE_ORDER:
                feat[f"cs_{s}"] = int(feat["career_style"] == s)

            records.append(feat)

    return records


def _fight_feature_vector(feat_cols, rec_a, rec_b):
    """Fight-level feature row for the finish-type / prop models
    (03_train_model.train_finish_model & train_prop_models layout)."""
    vec = []
    for col in feat_cols:
        if col.startswith("diff_"):
            v = rec_a.get(col, 0.0)
        elif col.startswith("style_a_"):
            v = int(rec_a.get("career_style", "") == col[8:])
        elif col.startswith("style_b_"):
            v = int(rec_b.get("career_style", "") == col[8:])
        elif col.startswith("wc_"):
            v = int(rec_a.get("weight_class_train", "") == col[3:])
        elif col in ("title_bout", "number_of_rounds"):
            v = rec_a.get(col, 0)
        else:
            v = 0.0
        try:
            vec.append(float(v) if v is not None else 0.0)
        except (TypeError, ValueError):
            vec.append(0.0)
    return np.nan_to_num(np.array(vec, dtype=float), nan=0.0).reshape(1, -1)


# ── Step 5: Run win predictions ───────────────────────────────────────────────

def run_predictions(upcoming_records, model_bundle, career_styled_df):
    main_model   = model_bundle["main_model"]
    feature_cols = model_bundle["feature_cols"]
    shift_model  = model_bundle.get("shift_model")
    shift_feats  = model_bundle.get("shift_features", [])
    shift_thresh = model_bundle.get("shift_threshold", 0.78)
    style_props  = model_bundle.get("style_proportions", {})
    style_order  = model_bundle.get("style_order", STYLE_ORDER)

    pairs = {}
    for rec in upcoming_records:
        key = tuple(sorted([rec["fighter"], rec["opponent"]]))
        pairs.setdefault(key, []).append(rec)

    for fight_key, pair in pairs.items():
        if len(pair) != 2:
            continue

        for rec in pair:
            fighter  = rec["fighter"]
            opponent = rec["opponent"]

            feat_vec = {}
            for col in feature_cols:
                feat_vec[col] = rec.get(col, 0.0) or 0.0

            # Style one-hots
            for s in style_order:
                feat_vec[f"style_a_{s}"] = int(rec["career_style"] == s)
                feat_vec[f"style_b_{s}"] = int(rec["opp_career_style"] == s)
                for sb in style_order:
                    feat_vec[f"matchup_{s}_vs_{sb}"] = int(
                        rec["career_style"] == s and rec["opp_career_style"] == sb)

            # Style × implied_prob interactions
            ip_diff = feat_vec.get("diff_implied_prob", 0) or 0
            for s in style_order:
                feat_vec[f"style_a_{s}_x_ip"] = feat_vec.get(f"style_a_{s}", 0) * ip_diff
                feat_vec[f"style_b_{s}_x_ip"] = feat_vec.get(f"style_b_{s}", 0) * ip_diff

            # Style × reach
            reach_diff = feat_vec.get("diff_reach_cms", 0) or 0
            for s in ["Striker", "Sniper", "Muay_Thai"]:
                feat_vec[f"style_a_{s}_x_reach"] = feat_vec.get(f"style_a_{s}", 0) * reach_diff

            # Lag variables
            fighter_history = (
                career_styled_df[career_styled_df["fighter"] == fighter]
                .sort_values("event_date")["career_style"].tolist()
            )
            opp_history = (
                career_styled_df[career_styled_df["fighter"] == opponent]
                .sort_values("event_date")["career_style"].tolist()
            )
            for k in range(1, 4):
                lag_style_a = fighter_history[-(k)] if len(fighter_history) >= k else "Unknown"
                lag_style_b = opp_history[-(k)] if len(opp_history) >= k else "Unknown"
                for s in style_order + ["Unknown"]:
                    feat_vec[f"style_a_lag{k}_{s}"] = int(lag_style_a == s)
                    feat_vec[f"style_b_lag{k}_{s}"] = int(lag_style_b == s)

            # Weight class dummies (training label, e.g. "Women'S Flyweight")
            fight_wc = rec.get("weight_class_train", "Unknown")
            for col in feature_cols:
                if col.startswith("wc_"):
                    feat_vec[col] = int(fight_wc == col[3:])

            X_row = np.array([feat_vec.get(c, 0.0) for c in feature_cols], dtype=float)
            X_row = np.nan_to_num(X_row, nan=0.0)

            win_prob_raw = main_model.predict_proba(X_row.reshape(1, -1))[0][1]
            rec["win_prob_raw"] = win_prob_raw

            # Style-shift detection
            rec["style_shift_predicted"]   = False
            rec["style_shift_probability"] = 0.0
            rec["predicted_infight_style"] = rec["career_style"]

            lookup_key = (rec["career_style"], rec["opp_career_style"])
            if lookup_key in style_props:
                rec["predicted_infight_style"] = max(
                    style_props[lookup_key], key=style_props[lookup_key].get)

            if shift_model is not None and shift_feats:
                shift_vec = np.array([rec.get(f, 0.0) for f in shift_feats], dtype=float)
                shift_vec = np.nan_to_num(shift_vec, nan=0.0)
                shift_prob = shift_model.predict_proba(shift_vec.reshape(1, -1))[0][1]
                rec["style_shift_probability"] = float(shift_prob)
                if shift_prob >= shift_thresh:
                    rec["style_shift_predicted"] = True

    return upcoming_records


# ── Step 6: Finish-type predictions (multinomial) ─────────────────────────────

def run_finish_predictions(upcoming_records, model_bundle):
    """
    For each matchup (fight-level), predict P(KO/TKO), P(Submission), P(Decision).
    Returns dict keyed by sorted fighter pair.
    """
    finish_info = model_bundle.get("finish_model")
    if not finish_info:
        return {}

    pipeline    = finish_info["pipeline"]
    feat_cols   = finish_info["feature_cols"]
    class_names = finish_info["class_names"]

    pairs = {}
    for rec in upcoming_records:
        key = tuple(sorted([rec["fighter"], rec["opponent"]]))
        pairs.setdefault(key, []).append(rec)

    finish_preds = {}
    for fight_key, pair in pairs.items():
        if len(pair) != 2:
            continue
        rec_a, rec_b = pair[0], pair[1]
        X = _fight_feature_vector(feat_cols, rec_a, rec_b)
        probs = pipeline.predict_proba(X)[0]
        finish_preds[fight_key] = {cls: float(p) for cls, p in zip(class_names, probs)}

    return finish_preds


# ── Step 7: Scale predictions & compute edge ──────────────────────────────────

def scale_predictions(results):
    pairs = {}
    for r in results:
        key = tuple(sorted([r["fighter"], r["opponent"]]))
        pairs.setdefault(key, []).append(r)

    scaled = []
    for key, pair in pairs.items():
        if len(pair) != 2:
            continue
        total_prob = sum(r["win_prob_raw"] for r in pair)
        for r in pair:
            r["win_prob_scaled"] = r["win_prob_raw"] / total_prob if total_prob > 0 else 0.5

        raw_imps = []
        for r in pair:
            odds = r.get("odds_numeric", np.nan)
            imp  = implied_prob_from_odds(odds) if not np.isnan(odds) else 0.5
            raw_imps.append(imp)
        total_imp = sum(raw_imps) or 1.0

        for r, raw_imp in zip(pair, raw_imps):
            r["implied_prob_scaled"] = raw_imp / total_imp
            r["betting_edge"]        = r["win_prob_scaled"] - r["implied_prob_scaled"]

        scaled.extend(pair)
    return scaled


# ── Step 8: Continuous bounded Kelly bet sizing ───────────────────────────────

def compute_bet_sizes(results):
    for r in results:
        edge = r.get("betting_edge", 0)
        p    = r.get("win_prob_scaled", 0.5)
        odds = r.get("odds_numeric", np.nan)

        if edge < MIN_EDGE_PCT or np.isnan(odds):
            r["bet_size"] = 0.0
            continue

        decimal_odds = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
        b = decimal_odds - 1
        if b <= 0:
            r["bet_size"] = 0.0
            continue

        kelly      = max((p * b - (1 - p)) / b, 0)
        confidence = max(abs(p - 0.5) * 2, 0.1)
        units      = MAX_UNITS * expit(STEEPNESS * confidence * kelly)
        units      = min(max(round(units, 2), 1.0), MAX_UNITS - 0.01)
        r["bet_size"] = units
    return results


# ── Step 9: Prop predictions ──────────────────────────────────────────────────

def predict_props(upcoming_records, model_bundle, event_fights, prop_odds_df=None):
    """
    Generate 8 prop probabilities per fight using the multinomial finish model
    combined with individual win probabilities.

    Market implied probs now come from live BFO prop odds (prop_odds_df) when
    available. Falls back to career method rates if BFO has no line for a prop.

    A prop is flagged when model edge vs market implied prob exceeds thresholds.
    """
    finish_info = model_bundle.get("finish_model")
    if not finish_info:
        return []

    finish_pipeline  = finish_info["pipeline"]
    finish_feat_cols = finish_info["feature_cols"]
    finish_classes   = finish_info["class_names"]   # ["KO_TKO","Submission","Decision"]

    # Build live prop lookup keyed by (matchup_last_names, fighter_last_name, method)
    # Avoids fighter_a/b ordering mismatches between BFO and UFCStats.
    # Distance props use (matchup_last_names, prop_type) since no fighter name needed.
    live_prop_lookup       = {}   # (frozenset(last_names), fighter_last, method) → imp_prob
    live_prop_lookup_fight = {}   # (frozenset(last_names), prop_type) → imp_prob

    if prop_odds_df is not None and not prop_odds_df.empty:
        for _, row in prop_odds_df.iterrows():
            imp = row.get("market_implied_prob")
            if imp is None or str(imp) in ("nan", "None", ""):
                continue
            fa_bfo   = str(row["fighter_a"]).strip()
            fb_bfo   = str(row["fighter_b"]).strip()
            pt       = row["prop_type"]
            val      = float(imp)
            odds_str = str(row.get("odds_str", "") or "").strip()

            last_a = fa_bfo.split()[-1].lower() if fa_bfo else ""
            last_b = fb_bfo.split()[-1].lower() if fb_bfo else ""
            if not last_a or not last_b:
                continue
            matchup_lasts = frozenset([last_a, last_b])

            if pt in {"fight_goes_distance", "fight_inside_distance"}:
                live_prop_lookup_fight[(matchup_lasts, pt)] = (val, odds_str)
            else:
                # For fighter-specific props, key on (matchup, fighter_last_name, method)
                # Extract fighter last name from the prop label to avoid fighter_a/b ordering issues
                method = {"fighter_a_ko":"ko","fighter_b_ko":"ko",
                          "fighter_a_sub":"sub","fighter_b_sub":"sub",
                          "fighter_a_dec":"dec","fighter_b_dec":"dec"}.get(pt)
                if not method:
                    continue
                label  = str(row.get("prop_label", "")).lower()
                m      = re.match(r'^([a-z\s\'\.\-]+?)\s+wins\s+by', label)
                if m:
                    prop_fighter_last = m.group(1).strip().split()[-1]
                else:
                    prop_fighter_last = last_a if last_a in label else last_b
                live_prop_lookup[(matchup_lasts, prop_fighter_last, method)] = (val, odds_str)

    print(f"  Built prop lookup: {len(live_prop_lookup)} method + {len(live_prop_lookup_fight)} distance entries")

    fight_feats = {}
    for (fa, fb, *_) in event_fights:
        rec_a = next((r for r in upcoming_records if r["fighter"] == fa), None)
        rec_b = next((r for r in upcoming_records if r["fighter"] == fb), None)
        if rec_a and rec_b:
            fight_feats[(fa, fb)] = (rec_a, rec_b)

    prop_recs = []
    for (fa, fb), (rec_a, rec_b) in fight_feats.items():
        # Win probs (already scaled in results; look them up from rec)
        p_a = float(rec_a.get("win_prob_scaled") or rec_a.get("win_prob_raw") or 0.5)
        p_b = float(rec_b.get("win_prob_scaled") or rec_b.get("win_prob_raw") or 0.5)
        total = p_a + p_b
        if total > 0:
            p_a /= total; p_b /= total

        # Finish-type probabilities from multinomial model (fight-level)
        X = _fight_feature_vector(finish_feat_cols, rec_a, rec_b)
        try:
            finish_probs = finish_pipeline.predict_proba(X)[0]
            fp = {cls: float(p) for cls, p in zip(finish_classes, finish_probs)}
        except Exception:
            continue

        p_ko  = fp.get("KO_TKO",     0.0)
        p_sub = fp.get("Submission", 0.0)
        p_dec = fp.get("Decision",   0.0)

        matchup_key = frozenset([fa, fb])

        def _live_market_prob(prop_type):
            """Look up live BFO market implied prob using fighter last name + method."""
            last_a = fa.split()[-1].lower() if fa else ""
            last_b = fb.split()[-1].lower() if fb else ""
            matchup_lasts = frozenset([last_a, last_b])
            if prop_type in {"fight_goes_distance", "fight_inside_distance"}:
                result = live_prop_lookup_fight.get((matchup_lasts, prop_type))
                return result[0] if result else None
            method = {"fighter_a_ko":"ko","fighter_b_ko":"ko",
                      "fighter_a_sub":"sub","fighter_b_sub":"sub",
                      "fighter_a_dec":"dec","fighter_b_dec":"dec"}.get(prop_type)
            if not method:
                return None
            fighter = fa if prop_type.startswith("fighter_a") else fb
            fighter_last = fighter.split()[-1].lower() if fighter else ""
            result = live_prop_lookup.get((matchup_lasts, fighter_last, method))
            return result[0] if result else None

        def _live_market_odds_str(prop_type):
            """Look up the raw American odds string from BFO for this prop."""
            last_a = fa.split()[-1].lower() if fa else ""
            last_b = fb.split()[-1].lower() if fb else ""
            matchup_lasts = frozenset([last_a, last_b])
            if prop_type in {"fight_goes_distance", "fight_inside_distance"}:
                result = live_prop_lookup_fight.get((matchup_lasts, prop_type))
                return result[1] if result else None
            method = {"fighter_a_ko":"ko","fighter_b_ko":"ko",
                      "fighter_a_sub":"sub","fighter_b_sub":"sub",
                      "fighter_a_dec":"dec","fighter_b_dec":"dec"}.get(prop_type)
            if not method:
                return None
            fighter = fa if prop_type.startswith("fighter_a") else fb
            fighter_last = fighter.split()[-1].lower() if fighter else ""
            result = live_prop_lookup.get((matchup_lasts, fighter_last, method))
            return result[1] if result else None

        def _career_market_prob(rec, method):
            """Fallback: career method rate from career stats."""
            col_map = {"ko": "implied_prob_ko", "sub": "implied_prob_sub", "dec": "implied_prob_dec"}
            col = col_map.get(method, "")
            v = rec.get(col, None)
            try:
                return float(v) if v is not None and str(v) not in ("nan", "None", "") else None
            except (TypeError, ValueError):
                return None

        def _market_prob(prop_type, rec=None, method=None):
            """Live BFO odds first; fall back to career rate if not available."""
            live = _live_market_prob(prop_type)
            if live is not None:
                return live
            if rec is not None and method is not None:
                return _career_market_prob(rec, method)
            return None

        # Build all 8 props (6 fighter-specific + 2 fight-level distance props)
        PROPS = [
            (fa, fb, "fighter_a_ko",  f"{fa} wins by KO/TKO",    round(p_a * p_ko,  4), _market_prob("fighter_a_ko",  rec_a, "ko")),
            (fa, fb, "fighter_a_sub", f"{fa} wins by Submission", round(p_a * p_sub, 4), _market_prob("fighter_a_sub", rec_a, "sub")),
            (fa, fb, "fighter_a_dec", f"{fa} wins by Decision",   round(p_a * p_dec, 4), _market_prob("fighter_a_dec", rec_a, "dec")),
            (fb, fa, "fighter_b_ko",  f"{fb} wins by KO/TKO",    round(p_b * p_ko,  4), _market_prob("fighter_b_ko",  rec_b, "ko")),
            (fb, fa, "fighter_b_sub", f"{fb} wins by Submission", round(p_b * p_sub, 4), _market_prob("fighter_b_sub", rec_b, "sub")),
            (fb, fa, "fighter_b_dec", f"{fb} wins by Decision",   round(p_b * p_dec, 4), _market_prob("fighter_b_dec", rec_b, "dec")),
            (fa, fb, "fight_goes_distance",   "Fight Goes to Decision",     round(p_dec,        4), _market_prob("fight_goes_distance")),
            (fa, fb, "fight_inside_distance", "Fight Ends Inside Distance", round(p_ko + p_sub, 4), _market_prob("fight_inside_distance")),
        ]

        for (fighter, opponent, prop_type, prop_label, model_prob, market_imp) in PROPS:
            market_edge = round(model_prob - market_imp, 4) if market_imp is not None else None
            flagged = (
                market_imp is not None
                and market_edge is not None
                and market_edge > 0.10
                and model_prob > 0.15
            )

            # Kelly-sized bet using prop odds (derived from market_implied_prob)
            # market_imp is the implied prob → decimal odds = 1 / market_imp
            # We use the same bounded Kelly as compute_bet_sizes but capped at 3u for props
            prop_bet_size = 0.0
            if flagged and market_imp is not None and market_imp > 0:
                try:
                    decimal_odds = 1.0 / market_imp
                    b = decimal_odds - 1
                    p = model_prob
                    if b > 0 and p > 0:
                        kelly      = max((p * b - (1 - p)) / b, 0)
                        confidence = max(abs(p - 0.5) * 2, 0.1)
                        PROP_MAX   = 3.0
                        units      = PROP_MAX * expit(STEEPNESS * confidence * kelly)
                        prop_bet_size = min(max(round(units, 2), 0.5), PROP_MAX)
                except Exception:
                    prop_bet_size = 1.0

            prop_recs.append({
                "fighter_a":           fa,
                "fighter_b":           fb,
                "fighter":             fighter,
                "opponent":            opponent,
                "matchup":             f"{fa} vs {fb}",
                "prop_type":           prop_type,
                "prop_label":          prop_label,
                "model_prob":          model_prob,
                "market_implied_prob": round(market_imp, 4) if market_imp is not None else None,
                "market_odds_str":     _live_market_odds_str(prop_type) or "",
                "market_edge":         market_edge,
                "flagged":             flagged,
                "bet_size":            prop_bet_size,
            })

    def _in_live_lookup(p):
        la = p["fighter_a"].split()[-1].lower() if p["fighter_a"] else ""
        lb = p["fighter_b"].split()[-1].lower() if p["fighter_b"] else ""
        ml = frozenset([la, lb])
        pt = p["prop_type"]
        if pt in {"fight_goes_distance", "fight_inside_distance"}:
            return (ml, pt) in live_prop_lookup_fight
        method = {"fighter_a_ko":"ko","fighter_b_ko":"ko",
                  "fighter_a_sub":"sub","fighter_b_sub":"sub",
                  "fighter_a_dec":"dec","fighter_b_dec":"dec"}.get(pt)
        if not method:
            return False
        fighter = p["fighter_a"] if pt.startswith("fighter_a") else p["fighter_b"]
        fl = fighter.split()[-1].lower() if fighter else ""
        return (ml, fl, method) in live_prop_lookup

    live_count   = sum(1 for p in prop_recs if p["market_implied_prob"] is not None and _in_live_lookup(p))
    career_count = sum(1 for p in prop_recs if p["market_implied_prob"] is not None and not _in_live_lookup(p))
    print(f"  {live_count} props used live BFO odds, {career_count} fell back to career rates")

    # ── Select exactly 3 props: 1 distance + 2 method ─────────────────────────
    # First unflag everything, then selectively re-flag the winners
    for p in prop_recs:
        p["flagged"] = False
        p["bet_size"] = 0.0

    DISTANCE_TYPES = {"fight_goes_distance", "fight_inside_distance"}
    METHOD_TYPES   = {"fighter_a_ko", "fighter_b_ko",
                      "fighter_a_sub", "fighter_b_sub",
                      "fighter_a_dec", "fighter_b_dec"}

    def _has_edge(p):
        return (p["market_implied_prob"] is not None
                and p["market_edge"] is not None
                and p["market_edge"] > 0.10
                and p["model_prob"] > 0.15)

    # Best 1 distance prop across all fights
    distance_candidates = sorted(
        [p for p in prop_recs if p["prop_type"] in DISTANCE_TYPES and _has_edge(p)],
        key=lambda p: p["market_edge"], reverse=True
    )
    # Best 2 method props across all fights
    method_candidates = sorted(
        [p for p in prop_recs if p["prop_type"] in METHOD_TYPES and _has_edge(p)],
        key=lambda p: p["market_edge"], reverse=True
    )

    selected = distance_candidates[:1] + method_candidates[:3]
    for p in selected:
        p["flagged"] = True
        # Recompute Kelly-sized bet now that we know this prop is selected
        market_imp = p.get("market_implied_prob")
        model_prob = p.get("model_prob", 0)
        kelly_size = 1.0  # fallback
        if market_imp and market_imp > 0 and model_prob > 0:
            try:
                decimal_odds = 1.0 / market_imp
                b = decimal_odds - 1
                if b > 0:
                    kelly      = max((model_prob * b - (1 - model_prob)) / b, 0)
                    confidence = max(abs(model_prob - 0.5) * 2, 0.1)
                    PROP_MAX   = 3.0
                    units      = PROP_MAX * expit(STEEPNESS * confidence * kelly)
                    kelly_size = min(max(round(units, 2), 0.5), PROP_MAX)
            except Exception:
                kelly_size = 1.0
        p["bet_size"] = kelly_size

    print(f"  Selected {len(selected)} prop(s): "
          f"{len(distance_candidates[:1])} distance + {len(method_candidates[:3])} method")
    return prop_recs



# ── Parlay bet sizing ──────────────────────────────────────────────────────────

def compute_parlay_bet(bet_rows):
    """
    Kelly-sized bet on the full parlay of all flagged single bets.
    Combined prob = product of individual win probs.
    Combined decimal odds = product of individual decimal odds.
    Returns (units: float, american_odds: int | None).
    """
    if not bet_rows or len(bet_rows) < 2:
        return 0.0, None

    KELLY_FRACTION = 0.40
    MAX_PARLAY     = 3.0

    combined_prob = 1.0
    combined_dec  = 1.0
    for r in bet_rows:
        p    = r.get("win_prob_scaled", 0.5)
        odds = r.get("odds_numeric", np.nan)
        if np.isnan(odds):
            return 0.0, None
        dec = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
        combined_prob *= p
        combined_dec  *= dec

    b = combined_dec - 1
    if b <= 0 or combined_prob <= 0:
        return 0.0, None

    kelly = max((combined_prob * b - (1 - combined_prob)) / b, 0)
    if kelly <= 0:
        return 0.0, None

    kelly     *= KELLY_FRACTION
    confidence = max(abs(combined_prob - 0.5) * 2, 0.1)
    units      = MAX_PARLAY * expit(STEEPNESS * confidence * kelly)
    units      = min(max(round(units, 2), 0.5), MAX_PARLAY)

    # Convert combined decimal odds → American
    if combined_dec >= 2.0:
        american = round((combined_dec - 1) * 100)
    else:
        american = round(-100 / (combined_dec - 1))

    return units, american


def save_card_odds(event, odds_df, prop_odds_df):
    """
    Append this card's sportsbook lines to data/raw/bfo_odds_history.csv so the
    Wednesday updater can attach market odds to every fight on the card (not
    only the ones the model ended up pricing). One row per fighter.
    """
    rows = []
    for fa, fb, wc, info in event["fights"]:
        oa, ob = match_bout_odds(fa, fb, odds_df)
        if not (oa and ob):
            continue
        live = _live_method_probs(fa, fb, prop_odds_df)
        for f, o, odds in [(fa, fb, oa), (fb, fa, ob)]:
            hit = match_name(f, odds_df["fighter"].tolist())
            rows.append({
                "event_date": event["date"], "event_name": event["name"],
                "fighter": f, "opponent": o, "odds": int(float(odds)),
                "book": odds_df[odds_df["fighter"] == hit]["book"].iloc[0] if hit else "",
                "implied_prob_ko":  live[f].get("ko"),
                "implied_prob_sub": live[f].get("sub"),
                "implied_prob_dec": live[f].get("dec"),
                "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            })
    if not rows:
        return
    hist, sha = read_csv_from_github("data/raw/bfo_odds_history.csv")
    new = pd.DataFrame(rows)
    combined = new if hist.empty else pd.concat([hist, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["event_date", "fighter"], keep="last")
    write_csv_to_github(combined, "data/raw/bfo_odds_history.csv",
                        f"Card odds: {event['name']}", sha=sha)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Prediction Runner")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    print("\n[1] Checking for upcoming UFC event this week...")
    event = get_upcoming_event()
    if event is None:
        print("  No UFC event this week. Exiting.")
        return
    print(f"  Event found: {event['name']} on {event['date']}")
    print(f"  {len(event['fights'])} fights on the card")

    print("\n[2] Scraping fight odds from BestFightOdds...")
    odds_df, prop_odds_df = scrape_bestfightodds()

    print("\n[3] Loading model bundle from GitHub...")
    model_bundle = read_pickle_from_github("data/model_bundle.pkl")
    if model_bundle is None:
        print("  [ERROR] No model bundle found. Run 03_train_model.py first.")
        return

    print("\n[4] Loading career stats dataset...")
    career_df, _ = read_csv_from_github("data/processed/career_styled.csv")
    if career_df.empty:
        print("  [ERROR] No career_styled.csv found.")
        return
    career_df["event_date"] = pd.to_datetime(career_df["event_date"])

    raw_df, _ = read_csv_from_github("data/raw/fight_stats.csv")
    if raw_df.empty:
        print("  [ERROR] No fight_stats.csv found.")
        return
    rankings_df, _ = read_csv_from_github("data/raw/rankings_history.csv")
    rankings_df = relabel_history(rankings_df) if not rankings_df.empty else rankings_df
    current, train_df = build_current_career(raw_df, event["fights"], event["date"], rankings_df)

    print("\n[5] Building matchup features...")
    save_card_odds(event, odds_df, prop_odds_df)
    records = build_upcoming_features(event["fights"], current, odds_df, model_bundle,
                                      prop_odds_df=prop_odds_df, train_df=train_df)
    print(f"  Built features for {len(records)} fighter entries")

    print("\n[6] Running win predictions...")
    results = run_predictions(records, model_bundle, career_df)
    results = scale_predictions(results)
    results = compute_bet_sizes(results)

    print("\n[7] Running finish-type predictions (KO/Sub/Dec)...")
    finish_preds = run_finish_predictions(records, model_bundle)
    print(f"  Finish predictions for {len(finish_preds)} matchups")

    print("\n[8] Running prop model predictions...")
    prop_recs     = predict_props(records, model_bundle, event["fights"], prop_odds_df=prop_odds_df)
    flagged_props = [p for p in prop_recs if p["flagged"]]
    print(f"  {len(prop_recs)} prop predictions, {len(flagged_props)} flagged")

    bets = [r for r in results if r.get("bet_size", 0) > 0]
    print(f"  {len(bets)} win bet recommendations")

    parlay_bet_size, parlay_combined_odds = compute_parlay_bet(bets) if len(bets) >= 2 else (0.0, None)
    if parlay_bet_size > 0:
        odds_str = (f"+{parlay_combined_odds}" if parlay_combined_odds and parlay_combined_odds > 0 else str(parlay_combined_odds)) if parlay_combined_odds else "n/a"
        print(f"  Parlay: {parlay_bet_size:.2f}u at {odds_str}")

    print("\n[9] Resolving DraftKings deep links via Gambly...")
    results, prop_recs, parlay_links = resolve_gambly_links(results, prop_recs)

    # Build output
    output_rows = []
    for r in results:
        fight_key     = tuple(sorted([r["fighter"], r["opponent"]]))
        fp            = finish_preds.get(fight_key, {})
        output_rows.append({
            "matchup_id":               f"{r['fighter']}_vs_{r['opponent']}",
            "name":                     r["fighter"],   # alias: frontend reads r.fighter || r.name
            "fighter":                  r["fighter"],
            "opponent":                 r["opponent"],
            "weight_class":             r.get("weight_class", ""),
            "career_style":             r.get("career_style"),
            "predicted_infight_style":  r.get("predicted_infight_style"),
            "style_shift_predicted":    r.get("style_shift_predicted", False),
            "style_shift_probability":  round(r.get("style_shift_probability", 0), 4),
            "Odds":                     r.get("odds_str"),
            "odds_numeric":             r.get("odds_numeric"),
            "implied_probability":      round(r.get("implied_prob_scaled", 0.5), 4),
            "predicted_probability":    round(r.get("win_prob_scaled", 0.5), 4),
            "betting_edge":             round(r.get("betting_edge", 0), 4),
            "bet_size":                 r.get("bet_size", 0),
            # Finish type probabilities from multinomial model
            "prob_ko_tko":              round(fp.get("KO_TKO", 0), 4),
            "prob_submission":          round(fp.get("Submission", 0), 4),
            "prob_decision":            round(fp.get("Decision", 0), 4),
            "event_name":               event["name"],
            "event_date":               event["date"],
            # Sportsbook deep links (resolved via Gambly; fallback = Gambly URL)
            "dk_link":                  r.get("dk_link", ""),
            "dk_parlay_link":           parlay_links.get("dk", ""),
            "parlay_bet_size":          parlay_bet_size,
            "parlay_combined_odds":     parlay_combined_odds if parlay_combined_odds is not None else "",
                    })

    df_out     = pd.DataFrame(output_rows)
    clean_title = re.sub(r'[^\w\s]', '', event["name"]).replace(' ', '_')

    write_csv_to_github(
        df_out,
        f"predictions/v2_betting_recommendations_{clean_title}.csv",
        f"V2 predictions: {event['name']} — {datetime.now().strftime('%Y-%m-%d')}",
    )

    if prop_recs:
        write_csv_to_github(
            pd.DataFrame(prop_recs),
            f"predictions/v2_prop_recommendations_{clean_title}.csv",
            f"V2 prop predictions: {event['name']} — {datetime.now().strftime('%Y-%m-%d')}",
        )

    # Update fight_titles.json
    import json
    existing_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/titles/fight_titles.json"
    existing = requests.get(existing_url, headers=GH_HEADERS)
    recent_name     = "Unknown"
    existing_titles = {}
    if existing.status_code == 200:
        try:
            # keep every key (e.g. recent_date written by 05_scrape_results)
            existing_titles = json.loads(base64.b64decode(existing.json()["content"]).decode())
            recent_name     = existing_titles.get("recent", "Unknown")
        except Exception:
            pass

    titles = existing_titles if isinstance(existing_titles, dict) else {}
    titles.update({"upcoming": event["name"], "upcoming_date": event["date"],
                   "recent": recent_name, "updated_at": datetime.now().isoformat()})
    write_json_to_github(
        titles,
        "titles/fight_titles.json",
        f"Update fight titles — {event['name']}",
    )

    print(f"\n{'='*60}")
    print(f"✓ Predictions saved for: {event['name']}")
    print(f"  Win bets: {len(bets)}  |  Prop bets flagged: {len(flagged_props)}")
    if bets:
        print(f"  Total units at risk: {sum(r['bet_size'] for r in bets):.2f}")
    print(f"{'='*60}")


def run_test():
    """
    --test mode: validates the full prediction → betting-slip pipeline
    without needing a GitHub token or live model.

    Two sub-modes:
      --test          → scrape live UFCStats + BestFightOdds, then run mock
                        predictions on whatever fighters are found
      --test --mock   → skip all scraping, use hardcoded mock card so you
                        can verify Gambly URLs offline
    """
    import sys
    use_mock_card = "--mock" in sys.argv

    print("=" * 60)
    print("OctaStats V2 — TEST MODE")
    print("=" * 60)

    # ── 1. Event + odds ───────────────────────────────────────────
    if use_mock_card:
        print("\n[1] Using mock fight card (--mock flag set)\n")
        event = {
            "name":   "UFC Fight Night: Emmett vs Sy (2026-03-15)",
            "date":   "2026-03-15",
            "fights": [
                ("Josh Emmett", "Oumar Sy", "Featherweight", {"title_bout": 0, "number_of_rounds": 5}),
            ],
        }
        mock_odds = {
            "Josh Emmett": "+220",
            "Oumar Sy":    "-270",
        }
        print(f"  Event : {event['name']} ({event['date']})")
        print(f"  Fights: {len(event['fights'])}")
        for f1, f2, wc, *_ in event["fights"]:
            print(f"    {f1} vs {f2}  ({wc})")
        odds_df = pd.DataFrame(
            [{"fighter": k, "odds": v, "matchup_id": "mock"} for k, v in mock_odds.items()]
        )
        print(f"\n  Odds  : {len(odds_df)} fighters (mock)")
    else:
        print("\n[1] Checking UFCStats for upcoming event...")
        event = get_upcoming_event()
        if event is None:
            print("  No UFC event found within 7-day window.")
            print("  Re-run with --mock to use a hardcoded card instead.")
            event = None
        else:
            print(f"  ✓ Found: {event['name']} on {event['date']}")
            for f1, f2, wc, *_ in event["fights"]:
                print(f"    {f1} vs {f2}  ({wc})")

        print("\n[2] Scraping BestFightOdds...")
        odds_df, prop_odds_df = scrape_bestfightodds()
        if odds_df.empty:
            print("  No odds returned — re-run with --mock to skip scraping.")
        else:
            print(f"  ✓ {len(odds_df)} fighters\n")
            print(odds_df.head(12).to_string(index=False))

        if event is None:
            print("\n=== END TEST (no event found) ===")
            return

    # ── 2. Build mock predictions ─────────────────────────────────
    print("\n[3] Building mock prediction rows (no model needed)...\n")

    mock_preds = []
    for (f1, f2, wc, *_) in event["fights"]:
        odds_f1_str = None
        odds_f2_str = None

        if not odds_df.empty:
            o1, o2 = match_bout_odds(f1, f2, odds_df)
            odds_f1_str = str(o1) if o1 else None
            odds_f2_str = str(o2) if o2 else None
            print(f"    odds: {f1} {odds_f1_str} | {f2} {odds_f2_str}")

        # Fallback synthetic odds if not scraped
        if odds_f1_str is None:
            odds_f1_str = "-150"
        if odds_f2_str is None:
            odds_f2_str = "+125"

        odds_f1 = convert_odds(odds_f1_str)
        odds_f2 = convert_odds(odds_f2_str)
        ip_f1   = implied_prob_from_odds(odds_f1)
        ip_f2   = implied_prob_from_odds(odds_f2)

        # Synthetic model probabilities (slight upset lean for testing)
        model_p1 = round(ip_f1 * 0.95 + 0.10, 4)
        model_p2 = round(1.0 - model_p1, 4)
        edge_f1  = round(model_p1 - ip_f1, 4)
        edge_f2  = round(model_p2 - ip_f2, 4)

        for fighter, opp, odds_s, odds_n, model_p, edge in [
            (f1, f2, odds_f1_str, odds_f1, model_p1, edge_f1),
            (f2, f1, odds_f2_str, odds_f2, model_p2, edge_f2),
        ]:
            bet_sz = round(max(0.0, min(3.0, edge * 10)), 2) if edge >= MIN_EDGE_PCT else 0.0
            mock_preds.append({
                "matchup_id":              f"{fighter}_vs_{opp}",
                "name":                    fighter,   # ← alias for frontend r.name
                "fighter":                 fighter,
                "opponent":                opp,
                "weight_class":            wc,
                "career_style":            "Mixed",
                "predicted_infight_style": "Mixed",
                "style_shift_predicted":   False,
                "style_shift_probability": 0.0,
                "Odds":                    odds_s,
                "odds_numeric":            odds_n,
                "implied_probability":     round(ip_f1 if fighter == f1 else ip_f2, 4),
                "predicted_probability":   model_p,
                "betting_edge":            edge,
                "bet_size":                bet_sz,
                "prob_ko_tko":             0.35,
                "prob_submission":         0.20,
                "prob_decision":           0.45,
                "event_name":              event["name"],
                "event_date":              event["date"],
            })

    df_preds = pd.DataFrame(mock_preds)
    bets     = df_preds[df_preds["bet_size"] > 0]

    print(f"  Total rows : {len(df_preds)}")
    print(f"  Bet rows   : {len(bets)}")
    print()
    print(df_preds[["fighter", "opponent", "Odds", "predicted_probability",
                     "betting_edge", "bet_size"]].to_string(index=False))

    # ── 3. Mock props ─────────────────────────────────────────────
    mock_props = []
    for (f1, f2, wc, *_) in event["fights"][:2]:   # flag first two fights as having props
        mock_props.append({
            "fighter_a":          f1,
            "fighter_b":          f2,
            "matchup":            f"{f1} vs {f2}",
            "prop_type":          "inside_distance",
            "prop_label":         "wins inside the distance",
            "model_prob":         0.72,
            "base_rate":          0.52,
            "model_edge_vs_base": 0.20,
            "confidence":         0.80,
            "market_implied_prob": None,
            "market_edge":        None,
            "flagged":            True,
            "bet_size":           1.0,
            "cv_auc":             0.63,
        })

    flagged_props = [p for p in mock_props if p["flagged"]]

    # ── 4. Simulate Gambly URL generation (mirrors frontend exactly) ──
    print("\n" + "=" * 60)
    print("GAMBLY LINK PREVIEW  (mirrors frontend buildBettingSlips)")
    print("=" * 60)

    ml_bets = bets.to_dict("records")

    if ml_bets:
        print(f"\n── Moneyline singles ({len(ml_bets)} bets) ──")
        for r in ml_bets:
            name   = r.get("fighter") or r.get("name") or ""
            prompt = name + " moneyline"
            url    = _gambly_url(prompt)
            print(f"  {name:<28}  {r['Odds']:<6}  {r['bet_size']} u")
            print(f"    DK/FD → {url}")

        if len(ml_bets) >= 2:
            parlay_prompt  = " and ".join(
                (r.get("fighter") or r.get("name") or "") + " moneyline"
                for r in ml_bets
            )
            singles_prompt = parlay_prompt   # same prompt, Gambly handles routing
            print(f"\n  Parlay URL   → {_gambly_url(parlay_prompt)}")
            print(f"  Singles URL  → {_gambly_url(singles_prompt)}")
    else:
        print("\n  No moneyline bets flagged (all edges below threshold)")

    if flagged_props:
        print(f"\n── Props ({len(flagged_props)} flagged) ──")
        for p in flagged_props:
            fighter  = p.get("fighter_a") or p.get("fighter") or ""
            prop_lbl = p.get("prop_label") or p.get("prop_type") or "prop"
            prompt   = fighter + " " + prop_lbl
            url      = _gambly_url(prompt)
            print(f"  {fighter:<28}  {prop_lbl}")
            print(f"    DK/FD → {url}")

        all_prop_prompt = " and ".join(
            (p.get("fighter_a") or p.get("fighter") or "") + " " +
            (p.get("prop_label") or p.get("prop_type") or "prop")
            for p in flagged_props
        )
        print(f"\n  All props URL → {_gambly_url(all_prop_prompt)}")
    else:
        print("\n  No props flagged")

    # ── 5. CSV field audit ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CSV FIELD AUDIT  (checks frontend <-> CSV column alignment)")
    print("=" * 60)
    required_ml_cols   = ["fighter", "name", "opponent", "Odds", "odds_numeric",
                           "bet_size", "predicted_probability", "betting_edge",
                           "weight_class", "event_name", "event_date",
                           "dk_link", "dk_parlay_link"]
    required_prop_cols = ["fighter_a", "fighter_b", "prop_type",
                          "prop_label", "flagged", "bet_size", "dk_link"]

    missing_ml   = [c for c in required_ml_cols   if c not in df_preds.columns]
    prop_df      = pd.DataFrame(mock_props)
    missing_prop = [c for c in required_prop_cols if c not in prop_df.columns]

    print(f"\n  Predictions CSV columns  : {list(df_preds.columns)}")
    print(f"  Missing (expected by UI) : {missing_ml or '✓ none'}")
    print(f"\n  Props CSV columns        : {list(prop_df.columns)}")
    print(f"  Missing (expected by UI) : {missing_prop or '✓ none'}")

    # ── Optional: live Gambly API test ───────────────────────────────────
    if "--gambly" in sys.argv:
        print("\n" + "=" * 60)
        print("GAMBLY API TEST  (--gambly flag set)")
        print("Attempting to resolve real DK/FD links for mock bets...")
        print("=" * 60)
        test_bets = bets.to_dict("records") if not bets.empty else []
        test_props = mock_props
        enriched_bets, enriched_props, parlay_lnks = resolve_gambly_links(
            test_bets, test_props, timeout_ms=20000
        )
        print("\n  Results:")
        for r in enriched_bets:
            name = r.get("fighter") or r.get("name") or ""
            print(f"    {name}")
            print(f"      DK: {r.get('dk_link','—')}")
        print(f"\n  Parlay DK: {parlay_lnks.get('dk','—')}")

    print("\n" + "=" * 60)
    print("✓ TEST COMPLETE — no GitHub writes performed")
    if "--gambly" not in sys.argv:
        print("  Tip: run with --gambly to also test Gambly API link resolution")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        main()
