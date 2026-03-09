#!/usr/bin/env python3
"""
04_predict.py
OctaStats V2 — Friday fight week prediction runner

Runs Friday at noon if a UFC event is scheduled this week. Steps:
  1. Check UFCStats upcoming events — skip if no fight this week
  2. Scrape odds from BestFightOdds (DraftKings preferred, FanDuel fallback)
  3. Load model bundle and most recent career stats
  4. Build matchup features for this week's card
  5. Run dynamic logit predictions
  6. Apply style-shift detection (display if confidence > 0.78)
  7. Compute continuous bet sizes via bounded Kelly criterion
  8. Push predictions CSV and fight titles JSON to GitHub

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
if not GITHUB_TOKEN:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
REPO_NAME    = "OctaStats_UFC_Model_V2"
V1_REPO      = "UFC_Model"          # V1 still writes to the old repo
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

# Bet sizing constants — logistic-bounded Kelly
MAX_UNITS      = 5.0     # absolute ceiling (never reached)
STEEPNESS      = 3.5     # controls how quickly units grow (higher = more conservative)
MIN_EDGE_PCT   = 0.04    # minimum edge (4pp) to place any bet
VARIANCE_FLOOR = 0.01    # prevents blow-up when model variance is tiny


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path, repo_name=None):
    rn  = repo_name or REPO_NAME
    url = f"https://api.github.com/repos/{REPO_OWNER}/{rn}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        return pd.DataFrame(), None
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), data["sha"]


def read_pickle_from_github(repo_path):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        print(f"  [ERROR] Could not load {repo_path}")
        return None
    data    = resp.json()
    raw     = base64.b64decode(data["content"])
    return pickle.loads(raw)


def write_csv_to_github(df, repo_path, message, repo_name=None, sha=None):
    rn       = repo_name or REPO_NAME
    csv_str  = df.to_csv(index=False)
    content  = base64.b64encode(csv_str.encode()).decode()
    url      = f"https://api.github.com/repos/{REPO_OWNER}/{rn}/contents/{repo_path}"
    if sha is None:
        check = requests.get(url, headers=GH_HEADERS)
        sha   = check.json().get("sha") if check.status_code == 200 else None
    payload  = {"message": message, "content": content}
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


# ── Step 1: Check if there's a fight this week ────────────────────────────────

def get_upcoming_event():
    """
    Return (event_name, event_date, event_url, fights) for the next upcoming
    UFC event, or None if there's no event within the next 7 days.
    """
    url  = f"{BASE_URL}/statistics/events/upcoming"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=15)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    rows = soup.select("tr.b-statistics__table-row")
    if not rows:
        return None

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
        date_text  = date_span.get_text(strip=True)
        event_url  = link_el.get("href", "")
        event_name = link_el.get_text(strip=True)
        try:
            event_date = datetime.strptime(date_text, "%B %d, %Y").date()
        except ValueError:
            continue

        if today <= event_date <= next_week:

            # Scrape the fight card
            fights = get_fights_from_event(event_url)
            return {
                "name":  event_name,
                "date":  event_date.isoformat(),
                "url":   event_url,
                "fights": fights,
            }

    return None


def get_fights_from_event(event_url):
    """
    Return list of (fighter_a, fighter_b) tuples from an event page.
    Each td in the fight card has fighter names in <p> tags.
    """
    resp = requests.get(event_url, headers=SCRAPE_HEADERS, timeout=15, allow_redirects=True)
    if resp.status_code != 200:
        return []
    soup  = BeautifulSoup(resp.text, "lxml")
    bouts = []
    for row in soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover"):
        tds = row.find_all("td")
        if not tds:
            continue
        # Fighter names are in <p><a> tags inside the first td
        fighter_links = tds[0].select("p a.b-link")
        if len(fighter_links) < 2:
            continue
        f1 = fighter_links[0].get_text(strip=True)
        f2 = fighter_links[1].get_text(strip=True)
        if f1 and f2:
            bouts.append((f1, f2))
    return bouts


# ── Step 2: Scrape odds from BestFightOdds ────────────────────────────────────

def scrape_bestfightodds():
    """
    Scrape fighter odds from bestfightodds.com.
    Returns a DataFrame with columns: fighter, odds (American format string).
    Prefers DraftKings column; falls back to FanDuel.
    """
    url  = "https://www.bestfightodds.com/"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"  [WARN] BestFightOdds returned status {resp.status_code}")
        return pd.DataFrame(columns=["fighter", "odds"])

    soup = BeautifulSoup(resp.text, "lxml")

    # Identify column index for DraftKings or FanDuel
    dk_idx = fd_idx = None
    header_cells = soup.select("table.odds-table thead th, .table-header th")
    for i, th in enumerate(header_cells):
        text = th.get_text(strip=True).lower()
        if "draftkings" in text and dk_idx is None:
            dk_idx = i
        if "fanduel" in text and fd_idx is None:
            fd_idx = i

    odds_col_idx = dk_idx if dk_idx is not None else fd_idx
    book_name    = "DraftKings" if dk_idx is not None else ("FanDuel" if fd_idx is not None else "Unknown")

    if odds_col_idx is None:
        print("  [WARN] Could not find DraftKings or FanDuel column on BestFightOdds")
        return pd.DataFrame(columns=["fighter", "odds"])

    print(f"  Using odds from: {book_name}")

    fighters, odds_list = [], []
    rows = soup.select("table.odds-table tbody tr, .table-body tr")

    for row in rows:
        cells     = row.select("td")
        name_el   = row.select_one(".fighter-name a, .name a, td:first-child a")
        if not name_el or len(cells) <= odds_col_idx:
            continue

        fighter_name = name_el.get_text(strip=True)
        odds_cell    = cells[odds_col_idx]
        odds_text    = odds_cell.get_text(strip=True)

        # Validate odds format (+XXX or -XXX)
        if re.match(r'^[+-]\d+$', odds_text):
            fighters.append(fighter_name)
            odds_list.append(odds_text)

    result = pd.DataFrame({"fighter": fighters, "odds": odds_list})
    print(f"  Scraped {len(result)} fighter odds from BestFightOdds ({book_name})")
    return result


def convert_odds(odds_str):
    """Convert American odds string to float. Returns np.nan on failure."""
    if pd.isna(odds_str):
        return np.nan
    try:
        return float(str(odds_str).strip())
    except ValueError:
        return np.nan


def implied_prob_from_odds(odds):
    """Convert American odds to raw implied probability."""
    if np.isnan(odds):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


# ── Step 3-4: Build features for upcoming fights ──────────────────────────────

def build_upcoming_features(fights, career_styled_df, odds_df, model_bundle):
    """
    For each upcoming fight, build the feature vector using the most recent
    career stats for each fighter.

    Returns a list of dicts (one per fighter), each containing:
      - name, opponent, weight_class, career stats, odds, features for prediction
    """
    # Get the most recent career stats row per fighter
    latest = (
        career_styled_df
        .sort_values("event_date")
        .groupby("fighter")
        .last()
        .reset_index()
    )

    style_order = model_bundle.get("style_order", [])
    records     = []

    for (fa, fb) in fights:
        row_a = latest[latest["fighter"] == fa]
        row_b = latest[latest["fighter"] == fb]

        if row_a.empty or row_b.empty:
            print(f"  [WARN] Missing career data for {fa} or {fb}. Skipping fight.")
            continue

        row_a = row_a.iloc[0]
        row_b = row_b.iloc[0]

        # Odds
        odds_a_str = _match_fighter_odds(fa, odds_df)
        odds_b_str = _match_fighter_odds(fb, odds_df)

        odds_a = convert_odds(odds_a_str)
        odds_b = convert_odds(odds_b_str)

        # Skip if neither fighter has odds
        if np.isnan(odds_a) and np.isnan(odds_b):
            print(f"  [WARN] No odds found for {fa} vs {fb}")
            continue

        for (fighter, opponent, row_f, row_o, odds_f, odds_o) in [
            (fa, fb, row_a, row_b, odds_a, odds_b),
            (fb, fa, row_b, row_a, odds_b, odds_a),
        ]:
            # Diff features
            diff_cols = [c for c in row_f.index if c.startswith("career_")]
            feat = {
                "fighter":            fighter,
                "opponent":           opponent,
                "career_style":       row_f.get("career_style", "Mixed"),
                "opp_career_style":   row_o.get("career_style", "Mixed"),
                "odds_str":           odds_f if not pd.isna(odds_f) else None,
                "odds_numeric":       odds_f,
            }

            for col in diff_cols:
                try:
                    feat[f"diff_{col}"] = float(row_f.get(col, np.nan)) - float(row_o.get(col, np.nan))
                except (TypeError, ValueError):
                    feat[f"diff_{col}"] = np.nan

            records.append(feat)

    return records


def _match_fighter_odds(name, odds_df):
    """Fuzzy match a fighter name against the odds DataFrame."""
    if odds_df.empty:
        return None
    # Exact match first
    exact = odds_df[odds_df["fighter"].str.lower() == name.lower()]
    if not exact.empty:
        return exact.iloc[0]["odds"]

    # Partial match (last name)
    last_name = name.split()[-1].lower()
    partial   = odds_df[odds_df["fighter"].str.lower().str.contains(last_name, na=False)]
    if not partial.empty:
        return partial.iloc[0]["odds"]

    return None


# ── Step 5: Run predictions ───────────────────────────────────────────────────

def run_predictions(upcoming_records, model_bundle, career_styled_df):
    """
    Apply the dynamic logit model to each matchup, returning scaled
    win probabilities and style-shift predictions.
    """
    main_model    = model_bundle["main_model"]
    feature_cols  = model_bundle["feature_cols"]
    shift_model   = model_bundle.get("shift_model")
    shift_feats   = model_bundle.get("shift_features", [])
    shift_thresh  = model_bundle.get("shift_threshold", STYLE_SHIFT_THRESHOLD)
    style_props   = model_bundle.get("style_proportions", {})
    style_order   = model_bundle.get("style_order", [])

    # Latest career styled data for lag variables
    latest = (
        career_styled_df
        .sort_values("event_date")
        .groupby("fighter")
        .last()
        .reset_index()
    )

    results = []

    # Group records into matchup pairs
    fighters_seen = {}
    for rec in upcoming_records:
        fight_key = tuple(sorted([rec["fighter"], rec["opponent"]]))
        if fight_key not in fighters_seen:
            fighters_seen[fight_key] = []
        fighters_seen[fight_key].append(rec)

    for fight_key, pair in fighters_seen.items():
        if len(pair) != 2:
            continue

        for rec in pair:
            fighter = rec["fighter"]
            opponent = rec["opponent"]

            # Build feature vector
            feat_vec = {}
            for col in feature_cols:
                feat_vec[col] = rec.get(col, np.nan)

            # Style one-hots
            for s in style_order:
                feat_vec[f"style_a_{s}"] = int(rec["career_style"] == s)
                feat_vec[f"style_b_{s}"] = int(rec["opp_career_style"] == s)
                feat_vec[f"matchup_{s}_vs_{rec['opp_career_style']}"] = int(rec["career_style"] == s)

            # Lag variables (use last 3 events from career_styled)
            fighter_history = (
                career_styled_df[career_styled_df["fighter"] == fighter]
                .sort_values("event_date")["career_style"]
                .tolist()
            )
            for k in range(1, 4):
                lag_style = fighter_history[-(k)] if len(fighter_history) >= k else "Unknown"
                for s in style_order + ["Unknown"]:
                    feat_vec[f"style_a_lag{k}_{s}"] = int(lag_style == s)

            opp_history = (
                career_styled_df[career_styled_df["fighter"] == opponent]
                .sort_values("event_date")["career_style"]
                .tolist()
            )
            for k in range(1, 4):
                lag_style = opp_history[-(k)] if len(opp_history) >= k else "Unknown"
                for s in style_order + ["Unknown"]:
                    feat_vec[f"style_b_lag{k}_{s}"] = int(lag_style == s)

            # Build array in exact column order
            X_row = np.array([feat_vec.get(c, 0.0) for c in feature_cols], dtype=float)

            # Predict win probability
            if np.isnan(X_row).any():
                # Fall back to 0.5 if too many missing features
                win_prob_raw = 0.5
            else:
                win_prob_raw = main_model.predict_proba(X_row.reshape(1, -1))[0][1]

            rec["win_prob_raw"] = win_prob_raw

            # Style-shift detection
            rec["style_shift_predicted"]    = False
            rec["style_shift_probability"]  = 0.0
            rec["predicted_infight_style"]  = rec["career_style"]

            # First try style_props lookup
            lookup_key = (rec["career_style"], rec["opp_career_style"])
            if lookup_key in style_props:
                predicted = max(style_props[lookup_key], key=style_props[lookup_key].get)
                rec["predicted_infight_style"] = predicted

            # Then check shift model
            if shift_model is not None and shift_feats:
                shift_vec = np.array([rec.get(f, 0.0) for f in shift_feats], dtype=float)
                if not np.isnan(shift_vec).any():
                    shift_prob = shift_model.predict_proba(shift_vec.reshape(1, -1))[0][1]
                    rec["style_shift_probability"] = float(shift_prob)
                    if shift_prob >= shift_thresh:
                        rec["style_shift_predicted"] = True

            results.append(rec)

    return results


# ── Step 6: Scale predictions per matchup & compute implied probs ─────────────

def scale_predictions(results):
    """
    For each matchup pair, scale predicted win probs to sum to 1.
    Compute implied probability from odds and compute edge.
    """
    # Group by matchup pair
    pairs = {}
    for r in results:
        key = tuple(sorted([r["fighter"], r["opponent"]]))
        pairs.setdefault(key, []).append(r)

    scaled = []
    for key, pair in pairs.items():
        if len(pair) != 2:
            continue

        total_prob = sum(r["win_prob_raw"] for r in pair)
        if total_prob <= 0:
            for r in pair:
                r["win_prob_scaled"] = 0.5
        else:
            for r in pair:
                r["win_prob_scaled"] = r["win_prob_raw"] / total_prob

        # Implied probability from odds (remove vig by scaling)
        raw_imps = []
        for r in pair:
            imp = implied_prob_from_odds(r["odds_numeric"]) if not np.isnan(r.get("odds_numeric", np.nan)) else 0.5
            raw_imps.append(imp)
        total_imp = sum(raw_imps) if sum(raw_imps) > 0 else 1.0

        for r, raw_imp in zip(pair, raw_imps):
            r["implied_prob_scaled"] = raw_imp / total_imp
            r["betting_edge"]        = r["win_prob_scaled"] - r["implied_prob_scaled"]

        scaled.extend(pair)

    return scaled


# ── Step 7: Continuous bounded Kelly bet sizing ───────────────────────────────

def compute_bet_sizes(results):
    """
    Compute continuous bet sizes using a logistic-bounded Kelly criterion.

    Standard fractional Kelly:
        f* = (p - q/b) / 1   where b = decimal odds - 1, p = win prob, q = 1-p

    We then pass this through a logistic transform to bound units below MAX_UNITS:
        units = MAX_UNITS * sigmoid(STEEPNESS * kelly_fraction)

    This means:
      - Units approach but never reach MAX_UNITS (logistic asymptote)
      - Negative kelly (bet is negative EV) maps to ~0 units
      - High-confidence heavy favourites with 2pp edge get more units than
        long shots with 2pp edge (because kelly naturally accounts for odds)
      - Variance in predictions reduces units (confidence weighting)

    Additional constraints:
      - Edge must exceed MIN_EDGE_PCT to place any bet
      - Model confidence (distance from 0.5) modulates STEEPNESS
    """
    for r in results:
        edge = r.get("betting_edge", 0)
        p    = r.get("win_prob_scaled", 0.5)
        odds = r.get("odds_numeric", np.nan)

        if edge < MIN_EDGE_PCT or np.isnan(odds):
            r["bet_size"] = 0.0
            continue

        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        b = decimal_odds - 1  # net decimal return per unit staked
        if b <= 0:
            r["bet_size"] = 0.0
            continue

        q = 1.0 - p

        # Kelly fraction
        kelly = (p * b - q) / b
        kelly = max(kelly, 0)  # no negative bets

        # Confidence weight: how far from 0.5 is the prediction?
        # This reduces bet size when the model is uncertain (close to 0.5)
        confidence = abs(p - 0.5) * 2   # maps [0.5, 1.0] → [0.0, 1.0]
        confidence = max(confidence, 0.1)  # floor so even uncertain bets can happen

        # Adjusted steepness
        adj_steepness = STEEPNESS * confidence

        # Logistic-bounded units
        units = MAX_UNITS * expit(adj_steepness * kelly)

        # Round to 2 decimal places (continuous bet sizing)
        units = round(units, 2)

        # Final hard cap — should never trigger due to logistic, but safety net
        units = min(units, MAX_UNITS - 0.01)

        r["bet_size"] = units

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Prediction Runner")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Check for upcoming event this week
    print("\n[1] Checking for upcoming UFC event this week...")
    event = get_upcoming_event()
    if event is None:
        print("  No UFC event this week. Exiting.")
        return

    print(f"  Event found: {event['name']} on {event['date']}")
    print(f"  {len(event['fights'])} fights on the card")

    # 2. Scrape odds
    print("\n[2] Scraping fight odds from BestFightOdds...")
    odds_df = scrape_bestfightodds()

    # 3. Load model bundle
    print("\n[3] Loading model bundle from GitHub...")
    model_bundle = read_pickle_from_github("data/model_bundle.pkl")
    if model_bundle is None:
        print("  [ERROR] No model bundle found. Run 03_train_model.py first.")
        return

    # 4. Load career styled dataset
    print("\n[4] Loading career stats dataset...")
    career_df, _ = read_csv_from_github("data/processed/career_styled.csv")
    if career_df.empty:
        print("  [ERROR] No career_styled.csv found.")
        return
    career_df["event_date"] = pd.to_datetime(career_df["event_date"])

    # 5. Build features for upcoming fights
    print("\n[5] Building matchup features...")
    records = build_upcoming_features(event["fights"], career_df, odds_df, model_bundle)
    print(f"  Built features for {len(records)} fighter entries")

    # 6. Run predictions
    print("\n[6] Running dynamic logit predictions...")
    results = run_predictions(records, model_bundle, career_df)
    results = scale_predictions(results)
    results = compute_bet_sizes(results)

    # Filter to fighters with bets
    bets = [r for r in results if r.get("bet_size", 0) > 0]
    print(f"  {len(bets)} bet recommendations")

    # Build output DataFrame
    output_rows = []
    for r in results:
        output_rows.append({
            "matchup_id":              f"{r['fighter']}_vs_{r['opponent']}",
            "name":                    r["fighter"],
            "opponent":                r["opponent"],
            "career_style":            r.get("career_style"),
            "predicted_infight_style": r.get("predicted_infight_style"),
            "style_shift_predicted":   r.get("style_shift_predicted", False),
            "style_shift_probability": round(r.get("style_shift_probability", 0), 4),
            "Odds":                    r.get("odds_str"),
            "odds_numeric":            r.get("odds_numeric"),
            "implied_probability":     round(r.get("implied_prob_scaled", 0.5), 4),
            "predicted_probability":   round(r.get("win_prob_scaled", 0.5), 4),
            "betting_edge":            round(r.get("betting_edge", 0), 4),
            "bet_size":                r.get("bet_size", 0),
            "event_name":              event["name"],
            "event_date":              event["date"],
        })

    df_out = pd.DataFrame(output_rows)

    # Clean title for filename
    clean_title = re.sub(r'[^\w\s]', '', event["name"]).replace(' ', '_')

    # Push V2 predictions
    write_csv_to_github(
        df_out,
        f"predictions/v2_betting_recommendations_{clean_title}.csv",
        f"V2 predictions: {event['name']} — {datetime.now().strftime('%Y-%m-%d')}",
    )

    # Update fight_titles.json (shared between V1 and V2)
    # Read existing recent event from V2 repo
    existing_titles_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/titles/fight_titles.json"
    existing = requests.get(existing_titles_url, headers=GH_HEADERS)
    recent_name = "Unknown"
    if existing.status_code == 200:
        import json
        existing_content = json.loads(base64.b64decode(existing.json()["content"]).decode())
        recent_name = existing_content.get("recent", "Unknown")

    fight_titles = {
        "upcoming": event["name"],
        "upcoming_date": event["date"],
        "recent": recent_name,
        "updated_at": datetime.now().isoformat(),
    }
    write_json_to_github(
        fight_titles,
        "titles/fight_titles.json",
        f"Update fight titles — {event['name']}",
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"✓ Predictions saved for: {event['name']}")
    print(f"  Total bets: {len(bets)}")
    if bets:
        total_units = sum(r["bet_size"] for r in bets)
        print(f"  Total units at risk: {total_units:.2f}")
        print(f"  Largest single bet:  {max(r['bet_size'] for r in bets):.2f} units")
        shifts = [r for r in bets if r.get("style_shift_predicted")]
        if shifts:
            print(f"  Style shift alerts:  {len(shifts)} fighter(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
