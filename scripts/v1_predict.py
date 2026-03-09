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
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT", "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")
REPO_OWNER   = "zwinship"
V2_REPO      = "OctaStats_UFC_Model_V2"
V1_REPO      = "UFC_Model"
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
}
SCRAPE_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OctaStats/1.0)"}
BASE_URL       = "http://www.ufcstats.com"

# V1 discrete bet sizing thresholds (edge in percentage points)
BET_THRESHOLDS = [(0.20, 3), (0.12, 2), (0.05, 1)]  # (min_edge, units)

STAT_COLS = [
    "KD", "sig_str_landed", "sig_str_attempted", "total_str_landed",
    "total_str_attempted", "td_landed", "td_attempted", "sub_att",
    "ctrl_sec", "head_landed", "body_landed", "leg_landed",
    "distance_landed", "clinch_landed", "ground_landed",
]
STYLE_Z = 0.5


# ── GitHub helpers ────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path, repo=V1_REPO):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{repo}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame(), None
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), data["sha"]


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
        link = row.select_one("a.b-link")
        date_td = row.select("td")
        if not link or len(date_td) < 2:
            continue
        name      = link.get_text(strip=True)
        date_text = date_td[1].get_text(strip=True)
        try:
            event_date = datetime.strptime(date_text, "%B %d, %Y").date()
        except ValueError:
            continue
        if today <= event_date <= window:
            event_url = link["href"]
            fights    = scrape_event_card(event_url)
            return name, event_date, fights

    return None


def scrape_event_card(event_url):
    """Scrape fighter names and weight classes from a UFCStats event page."""
    resp = requests.get(event_url, headers=SCRAPE_HEADERS, timeout=15)
    if resp.status_code != 200:
        return []
    soup   = BeautifulSoup(resp.text, "lxml")
    bouts  = []
    rows   = soup.select("tr.b-fight-details__table-row")

    for row in rows:
        cells = row.select("td")
        if len(cells) < 7:
            continue
        fighters = row.select("a.b-link.b-fight-details__person-link")
        if len(fighters) < 2:
            continue
        wc_text = cells[6].get_text(strip=True) if len(cells) > 6 else "Unknown"
        bouts.append({
            "fighter1":     fighters[0].get_text(strip=True),
            "fighter2":     fighters[1].get_text(strip=True),
            "weight_class": wc_text,
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


def scrape_bestfightodds(event_name):
    """
    Scrape DraftKings (preferred) or FanDuel (fallback) odds from BestFightOdds.
    Returns dict: {fighter_name_lower: american_odds_int}
    """
    url  = "https://www.bestfightodds.com/"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"  [WARN] BestFightOdds returned {resp.status_code}")
        return {}

    soup  = BeautifulSoup(resp.text, "lxml")
    odds  = {}

    # Find the table matching this event
    tables = soup.select("table.odds-table")
    target = None
    for tbl in tables:
        caption = tbl.select_one("caption, th.event-header, .table-header")
        if caption and event_name.split(":")[0].lower() in caption.get_text().lower():
            target = tbl
            break
    if target is None and tables:
        target = tables[0]  # fall back to first table (most recent/upcoming event)
    if target is None:
        print("  [WARN] No odds table found on BestFightOdds")
        return {}

    # Identify DraftKings or FanDuel column index
    headers = [th.get_text(strip=True).lower() for th in target.select("thead th")]
    dk_idx = next((i for i, h in enumerate(headers) if "draftkings" in h), None)
    fd_idx = next((i for i, h in enumerate(headers) if "fanduel" in h), None)
    col_idx = dk_idx if dk_idx is not None else fd_idx
    if col_idx is None:
        # Fall back: use first numeric column
        col_idx = 1
        print("  [WARN] Neither DraftKings nor FanDuel column found — using first odds column")

    for row in target.select("tbody tr"):
        cells = row.select("td")
        name_cell = row.select_one("td.fighter-name, td:first-child a, td a")
        if not name_cell or col_idx >= len(cells):
            continue
        name   = name_cell.get_text(strip=True).lower()
        raw    = cells[col_idx].get_text(strip=True).replace("−", "-").replace("–", "-")
        match  = re.search(r"[-+]?\d{3,}", raw)
        if match:
            odds[name] = int(match.group())

    print(f"  Scraped {len(odds)} fighters from BestFightOdds")
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

    X = np.array(feature_values).reshape(1, -1)

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

    for bout in fights:
        f1, f2, wc = bout["fighter1"], bout["fighter2"], bout["weight_class"]
        print(f"  {f1} vs {f2} ({wc})")

        f1_stats = get_fighter_avg_stats(f1, career_df)
        f2_stats = get_fighter_avg_stats(f2, career_df)
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


if __name__ == "__main__":
    main()
