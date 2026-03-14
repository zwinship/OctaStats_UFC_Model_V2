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
from datetime import datetime, timedelta
from scipy.special import expit

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
        raw_resp = requests.get(data["download_url"], headers=GH_HEADERS, timeout=60)
        if raw_resp.status_code == 200:
            return pd.read_csv(io.StringIO(raw_resp.text)), sha
        return pd.DataFrame(), None
    content = base64.b64decode(raw_content).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), sha


def read_pickle_from_github(repo_path):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        print(f"  [ERROR] Could not load {repo_path}")
        return None
    data = resp.json()
    raw  = base64.b64decode(data["content"])
    return pickle.loads(raw)


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
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=15)
    if resp.status_code != 200:
        return None

    soup      = BeautifulSoup(resp.text, "lxml")
    rows      = soup.select("tr.b-statistics__table-row")
    today     = datetime.utcnow().date()
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


def get_fights_from_event(event_url):
    resp = requests.get(event_url, headers=SCRAPE_HEADERS, timeout=15, allow_redirects=True)
    if resp.status_code != 200:
        return []
    soup     = BeautifulSoup(resp.text, "lxml")
    bouts    = []
    all_rows = soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover") or \
               soup.select("tr.b-fight-details__table-row")

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
            bouts.append((f1, f2, wc))
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

    # ── Moneyline scraping (unchanged logic) ──────────────────────────────────
    ML_BOOK_PREF = ["draftkings", "caesars", "betrivers", "betway", "unibet", "bet365"]
    book_col_map = {}

    for row in soup.select("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        texts = [c.get_text(strip=True).lower() for c in cells]
        if sum(1 for b in ML_BOOK_PREF if any(b in t for t in texts)) >= 2:
            for i, t in enumerate(texts):
                for b in ML_BOOK_PREF:
                    if b in t and b not in book_col_map:
                        book_col_map[b] = i
            if book_col_map:
                break

    preferred_col, preferred_book = None, None
    for book in ML_BOOK_PREF:
        if book in book_col_map:
            preferred_col  = book_col_map[book]
            preferred_book = book.title()
            break

    print(f"  Moneyline odds from: {preferred_book} (col {preferred_col})")

    PROP_WORDS = {"wins", "by", "inside", "decision", "draw", "round", "goes", "starts",
                  "doesn", "won", "ends", "either", "other", "points", "deducted",
                  "submission", "tko", "ko", "majority", "split", "unanimous", "parlay",
                  "over", "under", "not", "fight", "method"}

    fighters, odds_list = [], []
    for row in soup.select("tr"):
        cells    = row.find_all(["td", "th"])
        name_el  = row.select_one("a")
        if not name_el:
            continue
        fighter_name = name_el.get_text(strip=True)
        if not fighter_name or len(fighter_name) < 3:
            continue
        if set(fighter_name.lower().split()) & PROP_WORDS:
            continue

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
            fighters.append(fighter_name)
            odds_list.append(odds_text)

    ml_df = pd.DataFrame({"fighter": fighters, "odds": odds_list}).drop_duplicates(
        subset="fighter", keep="first")
    print(f"  Scraped {len(ml_df)} moneyline odds ({preferred_book})")

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


def _match_fighter_odds(name, odds_df):
    if odds_df.empty:
        return None
    exact = odds_df[odds_df["fighter"].str.lower() == name.lower()]
    if not exact.empty:
        return exact.iloc[0]["odds"]
    last_name = name.split()[-1].lower()
    partial   = odds_df[odds_df["fighter"].str.lower().str.contains(last_name, na=False)]
    if not partial.empty:
        return partial.iloc[0]["odds"]
    return None


# ── Step 3-4: Build features for upcoming fights ──────────────────────────────

def build_upcoming_features(fights, career_styled_df, odds_df, model_bundle):
    """
    Mirrors the feature construction in 03_train_model.build_matchup_features
    exactly, using most-recent career stats per fighter.
    """
    latest = (
        career_styled_df
        .sort_values("event_date")
        .groupby("fighter")
        .last()
        .reset_index()
    )

    records = []

    for (fa, fb, weight_class) in fights:
        row_a = latest[latest["fighter"] == fa]
        row_b = latest[latest["fighter"] == fb]
        if row_a.empty or row_b.empty:
            print(f"  [WARN] Missing career data for {fa} or {fb}. Skipping fight.")
            continue
        row_a = row_a.iloc[0]
        row_b = row_b.iloc[0]

        # Skip fighters with fewer than 3 fights in our dataset
        MIN_FIGHTS = 2
        ra_fights = (row_a.get("career_fights_before", 0) or 0) + 1
        rb_fights = (row_b.get("career_fights_before", 0) or 0) + 1
        if ra_fights < MIN_FIGHTS or rb_fights < MIN_FIGHTS:
            short = fa if ra_fights < MIN_FIGHTS else fb
            n     = min(ra_fights, rb_fights)
            print(f"  [SKIP] Insufficient data for {short} ({n} fights in dataset < {MIN_FIGHTS}). Skipping fight.")
            continue

        odds_a = convert_odds(_match_fighter_odds(fa, odds_df))
        odds_b = convert_odds(_match_fighter_odds(fb, odds_df))

        if np.isnan(odds_a) and np.isnan(odds_b):
            print(f"  [WARN] No odds found for {fa} vs {fb}")
            continue

        for (fighter, opponent, row_f, row_o, odds_f, odds_o) in [
            (fa, fb, row_a, row_b, odds_a, odds_b),
            (fb, fa, row_b, row_a, odds_b, odds_a),
        ]:
            feat = {
                "fighter":          fighter,
                "opponent":         opponent,
                "weight_class":     weight_class,
                "career_style":     row_f.get("career_style", "Mixed"),
                "opp_career_style": row_o.get("career_style", "Mixed"),
                "odds_str":         str(int(odds_f)) if not np.isnan(odds_f) else None,
                "odds_numeric":     odds_f,
            }

            # Career stat diffs
            career_cols = [c for c in row_f.index if c.startswith("career_")]
            for col in career_cols:
                try:
                    feat[f"diff_{col}"] = float(row_f.get(col, np.nan)) - float(row_o.get(col, np.nan))
                except (TypeError, ValueError):
                    feat[f"diff_{col}"] = np.nan

            # Physical diffs
            for col in ["height_cms", "reach_cms", "age"]:
                try:
                    feat[f"diff_{col}"] = float(row_f.get(col, np.nan)) - float(row_o.get(col, np.nan))
                except (TypeError, ValueError):
                    feat[f"diff_{col}"] = np.nan

            # Ranking diffs
            for rank_col in ["fighter_rank_wc", "fighter_rank_pfp"]:
                try:
                    feat[f"diff_{rank_col}"] = float(row_f.get(rank_col, np.nan)) - float(row_o.get(rank_col, np.nan))
                except (TypeError, ValueError):
                    feat[f"diff_{rank_col}"] = np.nan

            # Market implied prob (from current BFO odds)
            ip_f = implied_prob_from_odds(odds_f) if not np.isnan(odds_f) else np.nan
            ip_o = implied_prob_from_odds(odds_o) if not np.isnan(odds_o) else np.nan
            feat["diff_implied_prob"]     = (ip_f - ip_o) if not (np.isnan(ip_f) or np.isnan(ip_o)) else np.nan
            feat["implied_prob_a"]        = ip_f if not np.isnan(ip_f) else 0.5
            feat["implied_prob_b"]        = ip_o if not np.isnan(ip_o) else 0.5
            feat["implied_prob_source_a"] = 1   # live BFO odds = market
            feat["diff_historical_odds"]  = (odds_f - odds_o) if not (np.isnan(odds_f) or np.isnan(odds_o)) else np.nan

            # Prop implied probs — use career method rates as proxies at prediction time
            # (no BFO prop odds scraping yet — use model-imputed values from career stats)
            for prop_col in ["implied_prob_ko", "implied_prob_sub", "implied_prob_dec"]:
                try:
                    pf = float(row_f.get(prop_col, np.nan))
                    po = float(row_o.get(prop_col, np.nan))
                    feat[f"diff_{prop_col}"] = (pf - po) if not (np.isnan(pf) or np.isnan(po)) else np.nan
                except (TypeError, ValueError):
                    feat[f"diff_{prop_col}"] = np.nan

            # Fight context (assume standard card; could be updated if we scrape fight details)
            feat["title_bout"]       = 0
            feat["empty_arena"]      = 0
            feat["number_of_rounds"] = 5 if "title" in weight_class.lower() else 3

            # Interaction terms (mirror 03_train_model)
            diff_ws = feat.get("diff_current_win_streak", 0.0) or 0.0
            feat["odds_x_win_streak"] = (feat.get("diff_implied_prob", 0) or 0) * diff_ws
            feat["reach_x_ip"]  = (feat.get("diff_reach_cms", 0) or 0) * (feat.get("diff_implied_prob", 0) or 0)
            feat["height_x_ip"] = (feat.get("diff_height_cms", 0) or 0) * (feat.get("diff_implied_prob", 0) or 0)
            feat["title_x_fights"] = feat["title_bout"] * (feat.get("diff_career_fights_before", 0) or 0)
            feat["title_x_rank"]   = feat["title_bout"] * (feat.get("diff_fighter_rank_wc", 0) or 0)
            feat["ko_x_scorecard"] = (feat.get("diff_career_win_by_ko", 0) or 0) * (feat.get("diff_avg_scorecard_margin", 0) or 0)

            records.append(feat)

    return records


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

            # Weight class dummies
            fight_wc = rec.get("weight_class", "Unknown")
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

        feat_vec = []
        for col in feat_cols:
            if col.startswith("diff_"):
                raw = col[5:]
                try:
                    feat_vec.append(float(rec_a.get(raw, 0.0) or 0.0) - float(rec_b.get(raw, 0.0) or 0.0))
                except (TypeError, ValueError):
                    feat_vec.append(0.0)
            elif col.startswith("style_a_") and not "_lag" in col and not "_x_" in col:
                feat_vec.append(int(rec_a.get("career_style", "") == col[8:]))
            elif col.startswith("style_b_") and not "_lag" in col and not "_x_" in col:
                feat_vec.append(int(rec_b.get("career_style", "") == col[8:]))
            elif col.startswith("wc_"):
                feat_vec.append(int(rec_a.get("weight_class", "") == col[3:]))
            elif col in ["title_bout", "number_of_rounds"]:
                feat_vec.append(float(rec_a.get(col, 0) or 0))
            else:
                feat_vec.append(0.0)

        X = np.array(feat_vec, dtype=float).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
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
            fa_bfo = str(row["fighter_a"]).strip()
            fb_bfo = str(row["fighter_b"]).strip()
            pt     = row["prop_type"]
            val    = float(imp)

            last_a = fa_bfo.split()[-1].lower() if fa_bfo else ""
            last_b = fb_bfo.split()[-1].lower() if fb_bfo else ""
            if not last_a or not last_b:
                continue
            matchup_lasts = frozenset([last_a, last_b])

            if pt in {"fight_goes_distance", "fight_inside_distance"}:
                live_prop_lookup_fight[(matchup_lasts, pt)] = val
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
                live_prop_lookup[(matchup_lasts, prop_fighter_last, method)] = val

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
        feat_vec = []
        for col in finish_feat_cols:
            if col.startswith("diff_"):
                raw = col[5:]
                try:
                    feat_vec.append(float(rec_a.get(raw, 0.0) or 0.0) - float(rec_b.get(raw, 0.0) or 0.0))
                except (TypeError, ValueError):
                    feat_vec.append(0.0)
            elif col.startswith("style_a_"):
                feat_vec.append(int(rec_a.get("career_style", "") == col[8:]))
            elif col.startswith("style_b_"):
                feat_vec.append(int(rec_b.get("career_style", "") == col[8:]))
            elif col.startswith("wc_"):
                feat_vec.append(int(rec_a.get("weight_class", "") == col[3:]))
            elif col in ["title_bout", "number_of_rounds"]:
                feat_vec.append(float(rec_a.get(col, 0) or 0))
            else:
                feat_vec.append(0.0)

        X = np.array(feat_vec, dtype=float).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
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
                return live_prop_lookup_fight.get((matchup_lasts, prop_type))

            method = {"fighter_a_ko":"ko","fighter_b_ko":"ko",
                      "fighter_a_sub":"sub","fighter_b_sub":"sub",
                      "fighter_a_dec":"dec","fighter_b_dec":"dec"}.get(prop_type)
            if not method:
                return None
            # Fighter this prop is for: fa for fighter_a_*, fb for fighter_b_*
            fighter = fa if prop_type.startswith("fighter_a") else fb
            fighter_last = fighter.split()[-1].lower() if fighter else ""
            return live_prop_lookup.get((matchup_lasts, fighter_last, method))

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

    # Merge latest rankings into career_df
    raw_df, _ = read_csv_from_github("data/raw/fight_stats.csv")
    if not raw_df.empty:
        rank_cols = [c for c in ["fighter", "event_date", "fighter_rank_wc",
                                  "fighter_rank_pfp", "implied_prob_ko",
                                  "implied_prob_sub", "implied_prob_dec"]
                     if c in raw_df.columns]
        if len(rank_cols) > 2:
            raw_df["event_date"] = pd.to_datetime(raw_df["event_date"], errors="coerce")
            snap = (raw_df[rank_cols].sort_values("event_date")
                    .groupby("fighter").last().reset_index())
            for rc in rank_cols[2:]:
                if rc in snap.columns:
                    career_df = career_df.merge(
                        snap[["fighter", rc]], on="fighter", how="left",
                        suffixes=("_old", ""))
                    if f"{rc}_old" in career_df.columns:
                        career_df = career_df.drop(columns=[f"{rc}_old"])

    print("\n[5] Building matchup features...")
    records = build_upcoming_features(event["fights"], career_df, odds_df, model_bundle)
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
    recent_name = "Unknown"
    if existing.status_code == 200:
        try:
            recent_name = json.loads(base64.b64decode(existing.json()["content"]).decode()).get("recent", "Unknown")
        except Exception:
            pass

    write_json_to_github(
        {"upcoming": event["name"], "upcoming_date": event["date"],
         "recent": recent_name, "updated_at": datetime.now().isoformat()},
        "titles/fight_titles.json",
        f"Update fight titles — {event['name']}",
    )

    print(f"\n{'='*60}")
    print(f"✓ Predictions saved for: {event['name']}")
    print(f"  Win bets: {len(bets)}  |  Prop bets flagged: {len(flagged_props)}")
    if bets:
        print(f"  Total units at risk: {sum(r['bet_size'] for r in bets):.2f}")
    print(f"{'='*60}")


def _gambly_url(prompt):
    """Mirror the frontend gamblyUrl() function exactly."""
    import urllib.parse
    cleaned = prompt.lower().replace(" ", "+")
    encoded = urllib.parse.quote(cleaned, safe="+")
    return f"https://gambly.com/gambly-bot?auto=1&prompt={encoded}"


# ── Gambly link resolution (API) ─────────────────────────────────────────────

def resolve_gambly_links(bet_rows, prop_rows=None, timeout_ms=45000):
    """
    Resolve fighter names -> DraftKings deep links via Gambly's JSON API.

    Flow:
      1. Use Playwright (headless) to load gambly.com once -> get Cloudflare
         clearance cookies (anon-session-token, cf_clearance, etc.)
      2. POST start-betslip-job with those cookies via requests
      3. Poll betslip-job-status until completed
      4. GET  webpage-result -> queryString with bet IDs
      5. GET  get-alternate-offers-by-group-hash -> sourceData has raw DK URLs
    """
    import time as _time
    import json as _json

    ml_bets   = [r for r in bet_rows  if float(r.get("bet_size", 0)) > 0]
    prop_bets = [p for p in (prop_rows or []) if p.get("flagged")]

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

    # ── Step 1: get session cookies ──────────────────────────────────────────
    def _get_cf_cookies():
        import traceback as _tb

        # Prefer authenticated token from GitHub secret (enables DK deeplinks)
        # gambly_token_set expires in ~1 year; refreshToken inside it auto-renews
        token_set = os.environ.get("GAMBLY_TOKEN_SET", "").strip()
        if token_set:
            print(f"    [INFO] Using GAMBLY_TOKEN_SET secret (authenticated session)")
            return f"gambly_token_set={token_set}"

        # Legacy: GAMBLY_COOKIE secret (full cookie string)
        gambly_cookie = os.environ.get("GAMBLY_COOKIE", "").strip()
        if gambly_cookie:
            print(f"    [INFO] Using GAMBLY_COOKIE secret (authenticated session)")
            return gambly_cookie

        # Fallback: get anonymous session via headless Playwright
        print(f"    [INFO] Launching headless Chromium for CF cookies...")
        try:
            from playwright.sync_api import sync_playwright
            print(f"    [INFO] Playwright imported OK")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                print(f"    [INFO] Browser launched")
                ctx     = browser.new_context(user_agent=UA)
                page    = ctx.new_page()
                page.goto("https://gambly.com/gambly-bot",
                          wait_until="domcontentloaded", timeout=20000)
                print(f"    [INFO] Page loaded")
                _time.sleep(2)

                # Force state=ME so Gambly enables DK deeplinks
                page.evaluate("""async () => {
                    try {
                        await fetch('/api/get-books-priority-by-state?state=ME');
                        await fetch('/api/proxy/core/api/books/get-books-by-location?location=ME');
                    } catch(e) {}
                }""")
                _time.sleep(1)

                cookies = ctx.cookies()
                browser.close()

            cookie_dict = {co["name"]: co["value"] for co in cookies}
            sp = cookie_dict.get("searchParams", "N/A")
            print(f"    [INFO] searchParams cookie: {sp[:150]}")
            cookie_str = "; ".join(f"{co['name']}={co['value']}" for co in cookies)
            names = [co["name"] for co in cookies]
            print(f"    [INFO] Got {len(cookies)} anon cookies: {names}")
            print(f"    [WARN] Anonymous session — DK deeplinks may be unavailable")
            return cookie_str
        except Exception as e:
            print(f"    [WARN] Could not get CF cookies: {e}")
            _tb.print_exc()
            return None

    # ── Step 2-5: call Gambly API with cookies ────────────────────────────────
    def _get_dk_links(prompt, names, cookie_str):
        headers = {**HEADERS, "Cookie": cookie_str}

        # Start job
        try:
            print(f"    [INFO] POSTing start-betslip-job...")
            r = requests.post(
                f"{BASE}/gambly-bot/start-betslip-job",
                headers=headers,
                json={"prompt": prompt, "base64Images": [],
                      "twitterUrl": "", "saveBetslipFeedPostSync": False},
                timeout=20,
            )
            print(f"    [INFO] POST status: {r.status_code}")
            if r.status_code not in (200, 202):
                print(f"    [WARN] Unexpected status {r.status_code}: {r.text[:200]}")
                return None, {}
            job_id = r.json()["jobId"]
            print(f"    [INFO] Job started: {job_id[:8]}...")
        except Exception as e:
            print(f"    [WARN] Gambly job start failed: {e}")
            import traceback; traceback.print_exc()
            return None, {}

        # Poll until completed (max 40s)
        for attempt in range(20):
            _time.sleep(2)
            try:
                r = requests.get(
                    f"{BASE}/gambly-bot/betslip-job-status/{job_id}",
                    headers=headers, timeout=10,
                )
                s = r.json().get("status")
                if s == "completed":
                    print(f"    [INFO] Job completed after {(attempt+1)*2}s")
                    break
                elif s == "failed":
                    print(f"    [WARN] Gambly job failed")
                    return None, {}
            except Exception as e:
                print(f"    [WARN] Poll error: {e}")
                return None, {}
        else:
            print(f"    [WARN] Gambly job timed out after 40s")
            return None, {}

        # Get result
        try:
            r = requests.get(
                f"{BASE}/gambly-bot/webpage-result/{job_id}",
                headers=headers, timeout=10,
            )
            r.raise_for_status()
            import json as _json3
            _wr = r.json()

            # Find DraftKings betOffers specifically and dump them
            _bet_offers = _wr.get("betOffers", [])
            _share_url = _wr.get("shareUrl", "")
            qs = _wr.get("queryString", "")
        except Exception as e:
            print(f"    [WARN] Gambly result fetch failed: {e}")
            return None, {}

        # Parse bet IDs and group hashes
        bet_ids, group_hash_ids = [], []
        for part in qs.split("&"):
            if part.startswith("betOfferIdsPoints="):
                for token in part.split("=", 1)[1].split("|"):
                    bid = token.split("_")[0]
                    if bid.lstrip("-").isdigit():
                        bet_ids.append(bid)
            elif part.startswith("groupHashIds="):
                group_hash_ids = [h for h in part.split("=", 1)[1].split("|") if h]

        if not bet_ids:
            print(f"    [WARN] No bet IDs in queryString")
            return None, {}

        print(f"    [INFO] {len(bet_ids)} bet IDs, {len(group_hash_ids)} group hashes")
        print(f"    [INFO] First 3 bet IDs: {bet_ids[:3]}")
        print(f"    [INFO] First 3 hashes: {group_hash_ids[:3]}")

        # Get raw DK URLs from sourceData
        dk_urls_by_bet_id = {}
        if group_hash_ids:
            try:
                hashes_param = "%2C".join(group_hash_ids)
                r = requests.get(
                    f"{BASE}/bets/get-alternate-offers-by-group-hash"
                    f"?groupHashIds={hashes_param}",
                    headers=headers, timeout=10,
                )
                print(f"    [INFO] get-alternate-offers status: {r.status_code}")
                r.raise_for_status()
                resp_json = r.json()
                print(f"    [INFO] Offers groups: {len(resp_json)}")
                for group_offers in resp_json.values():
                    for offer in (group_offers or []):
                        bid = str(offer.get("id", ""))
                        sd  = offer.get("sourceData", "")
                        if sd:
                            try:
                                sd_obj = _json.loads(sd)
                                dk_url = sd_obj.get("Desktop") or sd_obj.get("Android")
                                if dk_url and "draftkings" in dk_url:
                                    dk_urls_by_bet_id[bid] = dk_url
                            except Exception:
                                pass
                print(f"    [INFO] DK URLs found: {len(dk_urls_by_bet_id)}")
                print(f"    [INFO] bet_ids[:5]: {bet_ids[:5]}")
                print(f"    [INFO] dk_urls keys[:5]: {list(dk_urls_by_bet_id.keys())[:5]}")
            except Exception as e:
                print(f"    [WARN] get-alternate-offers failed: {e}")
                import traceback; traceback.print_exc()
        else:
            # ── Playwright: load Gambly share page, click DK icon, scrape preurl ──
            # The shareUrl page shows FanDuel by default but has DK links loaded.
            # Clicking the DK book icon swaps to DK links — the real DK event URL
            # is encoded in the href as a `preurl` query param.
            if _share_url:
                print(f"    [INFO] Loading share page to click DK icon and scrape links...")
                try:
                    import urllib.parse as _up, re as _re2
                    from playwright.sync_api import sync_playwright as _spw
                    with _spw() as _pw:
                        _br  = _pw.chromium.launch(headless=True)
                        _ctx = _br.new_context(user_agent=UA)
                        # Inject auth cookie so Gambly treats us as logged-in
                        _ctx.add_cookies([{
                            "name": "gambly_token_set", "value": cookie_str.split("gambly_token_set=")[-1].split(";")[0],
                            "domain": "gambly.com", "path": "/",
                            "secure": True, "sameSite": "Lax",
                        }])
                        _pg = _ctx.new_page()
                        _pg.goto(_share_url, wait_until="domcontentloaded", timeout=25000)
                        _time.sleep(3)

                        # Click the DraftKings book icon
                        # It's a small book-switcher button — find it by looking for
                        # an element containing "draftkings" in its image src or aria-label
                        try:
                            _dk_btn = _pg.locator("img[src*='draft-kings'], img[alt*='DraftKings'], button[aria-label*='DraftKings']").first
                            if _dk_btn.count() == 0:
                                # fallback: find by logo URL pattern in any clickable element
                                _dk_btn = _pg.locator("[src*='draft-kings']").first
                            _dk_btn.click(timeout=5000)
                            print(f"    [INFO] Clicked DK book icon")
                            _time.sleep(2)
                        except Exception as _ce:
                            print(f"    [WARN] DK icon click failed: {_ce} — trying xpath")
                            try:
                                # Use the xpath area the user identified for the book switcher
                                _pg.click("xpath=/html/body/main/div[2]/div/div/div/div[2]/div[1]/div[1]/div/div[1]/div/div/div[2]", timeout=5000)
                                print(f"    [INFO] Clicked DK via xpath")
                                _time.sleep(2)
                            except Exception as _ce2:
                                print(f"    [WARN] xpath click also failed: {_ce2}")

                        # Match by fighter name using the bet-name-renderer class.
                        # Structure: .bet-name-renderer (name) and sibling <a> (link)
                        # are both children of the same parent div.
                        def _decode_preurl(href):
                            if not href: return None
                            if "preurl=" in href:
                                _qs = href.split("?", 1)[1] if "?" in href else ""
                                _pm = dict(p.split("=", 1) for p in _qs.split("&") if "=" in p)
                                _r  = _up.unquote(_pm.get("preurl", ""))
                                return _r if "draftkings" in _r else None
                            if "draftkings" in href and "/event/" in href:
                                return href
                            return None

                        # Build a map of name → DK URL from the page.
                        # Two element types to handle:
                        #   1. Fighter-specific props: name in .bet-name-renderer
                        #      e.g. "K. Vallejos", "C. Johnson"
                        #   2. Fight-level props (goes/doesn't go distance): DK puts
                        #      the matchup string in a sibling div, NOT .bet-name-renderer
                        #      e.g. "Andre Fili vs. Jose Miguel Delgado"
                        _name_to_dk = {}

                        # All bet card root elements — each card contains both the
                        # name/matchup div AND the anchor link
                        _all_cards = _pg.locator("a[href*='draftkings'], a[href*='preurl']").all()
                        print(f"    [INFO] Found {len(_all_cards)} DK anchor elements on page")

                        for _card in _all_cards:
                            try:
                                _href = _card.get_attribute("href", timeout=2000) or ""
                                _real = _decode_preurl(_href)
                                if not _real:
                                    continue
                                # Walk up to find the containing bet card, then grab
                                # any text label — either .bet-name-renderer or the
                                # matchup div (first non-empty div text in the card)
                                _container = _card.locator("..").locator("..")
                                # Try .bet-name-renderer first
                                _label_el = _container.locator(".bet-name-renderer").first
                                _label = ""
                                try:
                                    _label = (_label_el.inner_text(timeout=1000) or "").strip()
                                except Exception:
                                    pass
                                # If empty, grab all inner text of the container and
                                # take the first non-trivial line — this catches the
                                # matchup div "Andre Fili vs. Jose Miguel Delgado"
                                if not _label:
                                    _full = (_container.inner_text(timeout=1000) or "").strip()
                                    for _line in _full.split("\n"):
                                        _line = _line.strip()
                                        if len(_line) > 3 and _line not in ("Yes", "No", "Over", "Under"):
                                            _label = _line
                                            break
                                if _label:
                                    _name_to_dk[_label.lower()] = _real
                                    print(f"    [INFO] Scraped: {_label} → {_real}")
                            except Exception:
                                pass

                        # Also try the original .bet-name-renderer approach as a
                        # supplementary pass (catches abbreviated names like "K. Vallejos")
                        _name_els = _pg.locator(".bet-name-renderer").all()
                        print(f"    [INFO] Found {len(_name_els)} bet-name-renderer elements")
                        for _nel in _name_els:
                            try:
                                _fighter = (_nel.inner_text() or "").strip()
                                _parent  = _nel.locator("..").locator("..")
                                _a       = _parent.locator("a[href*='draftkings'], a[href*='preurl']").first
                                _href    = _a.get_attribute("href", timeout=2000) or ""
                                _real    = _decode_preurl(_href)
                                if _real and _fighter:
                                    _name_to_dk[_fighter.lower()] = _real
                                    print(f"    [INFO] Scraped (renderer): {_fighter} → {_real}")
                            except Exception:
                                pass

                        print(f"    [INFO] Name→URL map: {list(_name_to_dk.keys())}")

                        # Match our bet names to the scraped map.
                        # For fighter-specific props: try full name, then last name.
                        # For fight-level props: try matching either fighter's last name
                        # against the matchup string keys (e.g. "fili" in "andre fili vs. jose...")
                        for _name, _bid in zip(names, bet_ids):
                            _nl   = _name.lower()
                            _url  = _name_to_dk.get(_nl)
                            if not _url:
                                # Last name match against all keys
                                _last = _nl.split()[-1]
                                _url  = next((u for k, u in _name_to_dk.items() if _last in k), None)
                            if not _url:
                                # First name match (catches "Andre" in "Andre Fili vs. ...")
                                _first = _nl.split()[0]
                                _url   = next((u for k, u in _name_to_dk.items() if _first in k), None)
                            if _url:
                                dk_urls_by_bet_id[_bid] = _url
                                print(f"    [INFO] {_name} → {_url}")
                            else:
                                print(f"    [WARN] {_name}: no match in name→URL map")
                        _br.close()
                    print(f"    [INFO] DK URLs from share page: {len(dk_urls_by_bet_id)}")
                except Exception as e:
                    print(f"    [WARN] Share page scrape failed: {e}")
                    import traceback; traceback.print_exc()
            else:
                print(f"    [WARN] No shareUrl available")



        # Map to fighter names by position (bet_ids order = prompt order = names order)
        name_to_url = {}
        for i, name in enumerate(names):
            if i < len(bet_ids):
                name_to_url[name] = dk_urls_by_bet_id.get(bet_ids[i])

        # Parlay: use __parlay__ if set by deeplink/v3, otherwise construct from outcomes
        parlay_url = dk_urls_by_bet_id.pop("__parlay__", None)
        if not parlay_url and len(bet_ids) >= 2:
            try:
                all_outcomes, event_id = [], None
                for bid in bet_ids:
                    url = dk_urls_by_bet_id.get(bid, "")
                    if "outcomes=" in url:
                        all_outcomes.append(url.split("outcomes=")[1].split("&")[0])
                        if not event_id and "/event/" in url:
                            event_id = url.split("/event/")[1].split("?")[0]
                if all_outcomes and event_id:
                    parlay_url = (
                        f"https://sportsbook.draftkings.com/event/{event_id}"
                        f"?outcomes={'+'.join(all_outcomes)}"
                    )
            except Exception:
                pass

        return parlay_url, name_to_url

    # ── Main ──────────────────────────────────────────────────────────────────
    print("\n[GAMBLY] Resolving DraftKings deep links via Gambly API...")
    parlay_links = {"dk": None}

    cookie_str = _get_cf_cookies()
    if not cookie_str:
        print("  [WARN] No CF cookies — falling back to Gambly URLs")
        return _fallback_links(bet_rows, prop_rows)

    if ml_bets:
        names  = [r.get("fighter") or r.get("name") or "" for r in ml_bets]
        prompt = " and ".join(n + " moneyline" for n in names)
        gurl   = _gambly_url(prompt)
        print(f"  Moneylines ({len(ml_bets)} bets): {prompt}")

        parlay_url, name_to_url = _get_dk_links(prompt, names, cookie_str)

        parlay_links["dk"] = parlay_url or gurl
        print(f"  {'✓' if parlay_url else '↩ fallback'}  Parlay DK:")
        print(f"    {parlay_links['dk']}")

        for r in ml_bets:
            name    = r.get("fighter") or r.get("name") or ""
            dk_href = name_to_url.get(name)
            r["dk_link"] = dk_href or _gambly_url(name + " moneyline")
            print(f"  {'✓' if dk_href else '↩ fallback (not yet on DK)'}  {name}:")
            print(f"    {r['dk_link']}")

    if prop_bets:
        # Map prop_type → DraftKings-style prompt phrasing Gambly understands
        DK_PROP_PHRASE = {
            "fighter_a_ko":          "wins by KO TKO",
            "fighter_b_ko":          "wins by KO TKO",
            "fighter_a_sub":         "wins by submission",
            "fighter_b_sub":         "wins by submission",
            "fighter_a_dec":         "wins by decision",
            "fighter_b_dec":         "wins by decision",
            "fight_goes_distance":   "fight goes the distance",
            "fight_inside_distance": "fight does not go the distance",
        }

        def _prop_prompt_str(p):
            prop_type = p.get("prop_type", "")
            phrase    = DK_PROP_PHRASE.get(prop_type)
            if phrase:
                # For fighter-specific props use the actual fighter name
                # For fight-level props use fighter_a as the fight anchor
                name = p.get("fighter") or p.get("fighter_a") or ""
                return f"{name} {phrase}"
            # Fallback: use stored prop_label
            name = p.get("fighter_a") or p.get("fighter") or ""
            return name + " " + (p.get("prop_label") or prop_type or "prop")

        names  = [p.get("fighter") or p.get("fighter_a") or "" for p in prop_bets]
        prompt = " and ".join(_prop_prompt_str(p) for p in prop_bets)
        print(f"  Props ({len(prop_bets)} bets): {prompt}")

        # For the name→URL lookup, distance props use "yes"/"no" as the DK label
        # not the fighter name — build a parallel lookup_names list accordingly
        DISTANCE_TYPES = {"fight_goes_distance", "fight_inside_distance"}
        lookup_names = []
        for p in prop_bets:
            pt = p.get("prop_type", "")
            if pt == "fight_goes_distance":
                lookup_names.append("yes")
            elif pt == "fight_inside_distance":
                lookup_names.append("no")
            else:
                lookup_names.append(p.get("fighter") or p.get("fighter_a") or "")

        _, name_to_url = _get_dk_links(prompt, lookup_names, cookie_str)

        for p, name, lname in zip(prop_bets, names, lookup_names):
            dk_href      = name_to_url.get(lname)
            p["dk_link"] = dk_href or _gambly_url(_prop_prompt_str(p))
            print(f"  {'✓' if dk_href else '↩ fallback (not yet on DK)'}  {name}:")
            print(f"    {p['dk_link']}")

    print(f"  [GAMBLY] Done — {len(ml_bets)} ML bets + {len(prop_bets)} props resolved")
    return bet_rows, prop_rows or [], parlay_links

def _fallback_links(bet_rows, prop_rows):
    """Populate dk_link with individual Gambly URLs when Playwright unavailable.
    Each fighter gets their own single-bet Gambly URL so users can click directly
    to their specific bet, rather than the full parlay URL."""
    ml_bets = [r for r in bet_rows if float(r.get("bet_size", 0)) > 0]
    gurl_parlay = _gambly_url(" and ".join(
        (r.get("fighter") or r.get("name") or "") + " moneyline"
        for r in ml_bets
    )) if ml_bets else ""
    for r in ml_bets:
        name = r.get("fighter") or r.get("name") or ""
        r["dk_link"] = _gambly_url(name + " moneyline")
    parlay_links = {"dk": gurl_parlay if len(ml_bets) >= 2 else None}
    if prop_rows:
        DK_PROP_PHRASE = {
            "fighter_a_ko": "wins by KO TKO", "fighter_b_ko": "wins by KO TKO",
            "fighter_a_sub": "wins by submission", "fighter_b_sub": "wins by submission",
            "fighter_a_dec": "wins by decision", "fighter_b_dec": "wins by decision",
            "fight_goes_distance": "fight goes the distance",
            "fight_inside_distance": "fight does not go the distance",
        }
        for p in prop_rows:
            if p.get("flagged"):
                name   = p.get("fighter") or p.get("fighter_a") or ""
                phrase = DK_PROP_PHRASE.get(p.get("prop_type", ""),
                         p.get("prop_label") or p.get("prop_type") or "prop")
                p["dk_link"] = _gambly_url(f"{name} {phrase}")
    return bet_rows, prop_rows or [], parlay_links


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
                ("Josh Emmett", "Oumar Sy", "Featherweight"),
            ],
        }
        mock_odds = {
            "Josh Emmett": "+220",
            "Oumar Sy":    "-270",
        }
        print(f"  Event : {event['name']} ({event['date']})")
        print(f"  Fights: {len(event['fights'])}")
        for f1, f2, wc in event["fights"]:
            print(f"    {f1} vs {f2}  ({wc})")
        odds_df = pd.DataFrame(
            [{"fighter": k, "odds": v} for k, v in mock_odds.items()]
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
            for f1, f2, wc in event["fights"]:
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
    for (f1, f2, wc) in event["fights"]:
        odds_f1_str = None
        odds_f2_str = None

        if not odds_df.empty:
            m1 = odds_df[odds_df["fighter"].str.lower() == f1.lower()]
            m2 = odds_df[odds_df["fighter"].str.lower() == f2.lower()]
            if not m1.empty:
                odds_f1_str = str(m1.iloc[0]["odds"])
            if not m2.empty:
                odds_f2_str = str(m2.iloc[0]["odds"])

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
    for (f1, f2, wc) in event["fights"][:2]:   # flag first two fights as having props
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
