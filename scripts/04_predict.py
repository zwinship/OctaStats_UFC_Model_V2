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

def scrape_bestfightodds():
    url  = "https://www.bestfightodds.com/"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"  [WARN] BestFightOdds returned status {resp.status_code}")
        return pd.DataFrame(columns=["fighter", "odds"])

    soup      = BeautifulSoup(resp.text, "lxml")
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

    print(f"  Using odds from: {preferred_book} (col {preferred_col})")

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

    result = pd.DataFrame({"fighter": fighters, "odds": odds_list}).drop_duplicates(
        subset="fighter", keep="first")
    print(f"  Scraped {len(result)} fighter odds from BestFightOdds ({preferred_book})")
    return result


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
        MIN_FIGHTS = 3
        ra_fights = row_a.get("career_fights_before", 0) or 0
        rb_fights = row_b.get("career_fights_before", 0) or 0
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

def predict_props(upcoming_records, model_bundle, event_fights):
    prop_models = model_bundle.get("prop_models", {})
    if not prop_models:
        return []

    fight_feats = {}
    for (fa, fb, *_) in event_fights:
        rec_a = next((r for r in upcoming_records if r["fighter"] == fa), None)
        rec_b = next((r for r in upcoming_records if r["fighter"] == fb), None)
        if rec_a and rec_b:
            fight_feats[(fa, fb)] = (rec_a, rec_b)

    prop_recs = []
    for (fa, fb), (rec_a, rec_b) in fight_feats.items():
        for model_key, model_info in prop_models.items():
            pipeline     = model_info["pipeline"]
            feature_cols = model_info["feature_cols"]
            base_rate    = model_info["base_rate"]
            label        = model_info["label"]

            feat_vec, missing = [], 0
            for col in feature_cols:
                if col.startswith("diff_"):
                    raw = col[5:]
                    try:
                        feat_vec.append(float(rec_a.get(raw, 0.0) or 0.0) - float(rec_b.get(raw, 0.0) or 0.0))
                    except (TypeError, ValueError):
                        feat_vec.append(0.0); missing += 1
                elif col.startswith("style_a_"):
                    feat_vec.append(1 if rec_a.get("career_style") == col[8:] else 0)
                elif col.startswith("style_b_"):
                    feat_vec.append(1 if rec_b.get("career_style") == col[8:] else 0)
                elif col.startswith("wc_"):
                    feat_vec.append(1 if rec_a.get("weight_class", "") == col[3:] else 0)
                elif col in ["title_bout", "number_of_rounds"]:
                    feat_vec.append(float(rec_a.get(col, 0) or 0))
                else:
                    feat_vec.append(0.0); missing += 1

            if missing > len(feature_cols) // 3:
                continue

            X = np.array(feat_vec, dtype=float).reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0)
            try:
                prob = pipeline.predict_proba(X)[0][1]
            except Exception:
                continue

            confidence  = abs(prob - base_rate) / max(base_rate, 1 - base_rate)
            model_edge  = abs(prob - base_rate)
            flagged     = model_edge > 0.20 and confidence > 0.75

            prop_recs.append({
                "fighter_a":           fa,
                "fighter_b":           fb,
                "matchup":             f"{fa} vs {fb}",
                "prop_type":           model_key,
                "prop_label":          label,
                "model_prob":          round(prob, 4),
                "base_rate":           round(base_rate, 4),
                "model_edge_vs_base":  round(model_edge, 4),
                "confidence":          round(confidence, 4),
                "market_implied_prob": None,
                "market_edge":         None,
                "flagged":             flagged,
                "bet_size":            1.0 if flagged else 0.0,
                "cv_auc":              round(model_info["cv_auc"], 4),
            })

    return prop_recs


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
    odds_df = scrape_bestfightodds()

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
    prop_recs     = predict_props(records, model_bundle, event["fights"])
    flagged_props = [p for p in prop_recs if p["flagged"]]
    print(f"  {len(prop_recs)} prop predictions, {len(flagged_props)} flagged")

    bets = [r for r in results if r.get("bet_size", 0) > 0]
    print(f"  {len(bets)} win bet recommendations")

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
    No browser required. Flow:
      1. POST start-betslip-job   { prompt: "fighter moneyline and ..." }
      2. Poll betslip-job-status  until status == "completed"
      3. GET  webpage-result      -> queryString contains bet IDs (e.g. -88511843_0.0|...)
      4. GET  get-alternate-offers-by-group-hash  -> sourceData has raw DK URLs per bet
    """
    import time as _time
    import json as _json

    ml_bets   = [r for r in bet_rows  if float(r.get("bet_size", 0)) > 0]
    prop_bets = [p for p in (prop_rows or []) if p.get("flagged")]

    BASE    = "https://gambly.com/api/proxy/core/api"
    HEADERS = {
        "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
        "Referer":      "https://gambly.com/",
        "Origin":       "https://gambly.com",
    }

    def _get_dk_links(prompt, names):
        """
        Returns: (parlay_url_or_None, {name: dk_url_or_None})
        Uses get-alternate-offers-by-group-hash which returns raw DK desktop URLs
        directly in sourceData — no deeplink call needed.
        """
        # Step 1: start job
        try:
            r = requests.post(
                f"{BASE}/gambly-bot/start-betslip-job",
                headers=HEADERS,
                json={"prompt": prompt, "base64Images": [],
                      "twitterUrl": "", "saveBetslipFeedPostSync": False},
                timeout=15,
            )
            r.raise_for_status()
            job_id = r.json()["jobId"]
            print(f"    [INFO] Job started: {job_id[:8]}...")
        except Exception as e:
            print(f"    [WARN] Gambly job start failed: {e}")
            return None, {}

        # Step 2: poll until completed (max 40s)
        qs = None
        for attempt in range(20):
            _time.sleep(2)
            try:
                r = requests.get(
                    f"{BASE}/gambly-bot/betslip-job-status/{job_id}",
                    headers=HEADERS, timeout=10,
                )
                status = r.json()
                if status.get("status") == "completed":
                    print(f"    [INFO] Job completed after {(attempt+1)*2}s")
                    break
                elif status.get("status") == "failed":
                    print(f"    [WARN] Gambly job failed: {status.get('error')}")
                    return None, {}
            except Exception as e:
                print(f"    [WARN] Poll error: {e}")
                return None, {}
        else:
            print(f"    [WARN] Gambly job timed out after 40s")
            return None, {}

        # Step 3: get webpage-result -> queryString
        try:
            r = requests.get(
                f"{BASE}/gambly-bot/webpage-result/{job_id}",
                headers=HEADERS, timeout=10,
            )
            r.raise_for_status()
            wp = r.json()
            qs = wp.get("queryString", "")
        except Exception as e:
            print(f"    [WARN] Gambly result fetch failed: {e}")
            return None, {}

        # Parse bet IDs and group hashes from queryString
        # Format: betOfferIdsPoints=-88511843_0.0|-88531854_0.0|...
        #         groupHashIds=abc123|def456|...
        bet_ids    = []
        group_hash_ids = []
        for part in qs.split("&"):
            if part.startswith("betOfferIdsPoints="):
                for token in part.split("=", 1)[1].split("|"):
                    bid = token.split("_")[0]
                    if bid.lstrip("-").isdigit():
                        bet_ids.append(bid)
            elif part.startswith("groupHashIds="):
                group_hash_ids = part.split("=", 1)[1].split("|")

        if not bet_ids:
            print(f"    [WARN] No bet IDs in queryString")
            return None, {}

        print(f"    [INFO] Found {len(bet_ids)} bet IDs, {len(group_hash_ids)} group hashes")

        # Step 4: get-alternate-offers-by-group-hash -> raw DK desktop URLs in sourceData
        # This avoids the deeplink/v3 call entirely — sourceData has the URL directly
        dk_urls_by_bet_id = {}
        if group_hash_ids:
            try:
                hashes_param = "%2C".join(group_hash_ids)
                r = requests.get(
                    f"{BASE}/bets/get-alternate-offers-by-group-hash"
                    f"?groupHashIds={hashes_param}",
                    headers=HEADERS, timeout=10,
                )
                r.raise_for_status()
                offers = r.json()  # {groupHash: [{id, price, sourceData}, ...]}
                for group_offers in offers.values():
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
            except Exception as e:
                print(f"    [WARN] get-alternate-offers failed: {e}")

        # Map bet IDs back to fighter names by position order
        # bet_ids order matches the prompt order (same order as names list)
        name_to_url = {}
        for i, name in enumerate(names):
            if i < len(bet_ids):
                bid = bet_ids[i]
                name_to_url[name] = dk_urls_by_bet_id.get(bid)

        # Build parlay URL from all DK individual URLs
        parlay_url = None
        if len(bet_ids) >= 2:
            # Parlay: DK event URL with all outcomes combined
            # e.g. https://sportsbook.draftkings.com/event/33766391?outcomes=0ML83745398_3+0ML83745388_3
            # Construct from individual URLs by extracting outcomes
            try:
                all_outcomes = []
                event_id = None
                for bid in bet_ids:
                    url = dk_urls_by_bet_id.get(bid, "")
                    if "outcomes=" in url:
                        outcome = url.split("outcomes=")[1].split("&")[0]
                        all_outcomes.append(outcome)
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

    print("\n[GAMBLY] Resolving DraftKings deep links via Gambly API (no browser)...")
    parlay_links = {"dk": None}

    if ml_bets:
        names  = [r.get("fighter") or r.get("name") or "" for r in ml_bets]
        prompt = " and ".join(n + " moneyline" for n in names)
        gurl   = _gambly_url(prompt)
        print(f"  Moneylines ({len(ml_bets)} bets): {prompt}")

        parlay_url, name_to_url = _get_dk_links(prompt, names)

        parlay_links["dk"] = parlay_url or gurl
        status = "✓" if parlay_url else "↩ fallback"
        print(f"  {status}  Parlay DK:")
        print(f"    {parlay_links['dk']}")

        for r in ml_bets:
            name    = r.get("fighter") or r.get("name") or ""
            dk_href = name_to_url.get(name)
            r["dk_link"] = dk_href or _gambly_url(name + " moneyline")
            status  = "✓" if dk_href else "↩ fallback (not yet on DK)"
            print(f"  {status}  {name}:")
            print(f"    {r['dk_link']}")

    if prop_bets:
        names  = [p.get("fighter_a") or p.get("fighter") or "" for p in prop_bets]
        prompt = " and ".join(
            n + " " + (p.get("prop_label") or p.get("prop_type") or "prop")
            for n, p in zip(names, prop_bets)
        )
        gurl = _gambly_url(prompt)
        print(f"  Props ({len(prop_bets)} bets): {prompt}")

        _, name_to_url = _get_dk_links(prompt, names)

        for p, name in zip(prop_bets, names):
            dk_href     = name_to_url.get(name)
            p["dk_link"] = dk_href or _gambly_url(
                name + " " + (p.get("prop_label") or p.get("prop_type") or "prop")
            )
            status = "✓" if dk_href else "↩ fallback (not yet on DK)"
            print(f"  {status}  {name}:")
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
        for p in prop_rows:
            if p.get("flagged"):
                fighter  = p.get("fighter_a") or ""
                prop_lbl = p.get("prop_label") or p.get("prop_type") or "prop"
                p["dk_link"] = _gambly_url(fighter + " " + prop_lbl)
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
        odds_df = scrape_bestfightodds()
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
