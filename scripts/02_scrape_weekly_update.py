#!/usr/bin/env python3
"""
02_scrape_weekly_update.py
OctaStats V2 — Wednesday incremental data updater

Run every Wednesday at noon via GitHub Actions. Checks for any new
completed events since the last scrape date, appends new fight rows
to data/raw/fight_stats.csv, and pushes the update to GitHub.

Environment variables (set as GitHub Actions secrets):
    ZWINSHIP_PAT  — personal access token for zwinship account
"""

import requests
import pandas as pd
import numpy as np
import base64
import os
import io
import time
import re
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
import sys as _sys
if not GITHUB_TOKEN and "--test" not in _sys.argv:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
REPO_NAME    = "OctaStats_UFC_Model_V2"
BASE_URL     = "http://www.ufcstats.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection":      "keep-alive",
}
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path):
    """Read a CSV from GitHub, returning a DataFrame and the file SHA."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        print(f"  [WARN] Could not read {repo_path} from GitHub (status {resp.status_code})")
        return pd.DataFrame(), None
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    sha     = data["sha"]
    return pd.read_csv(io.StringIO(content)), sha


def write_csv_to_github(df, repo_path, message, sha=None):
    """Write a DataFrame as CSV to GitHub, updating if sha is provided."""
    csv_str  = df.to_csv(index=False)
    content  = base64.b64encode(csv_str.encode()).decode()
    url      = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    payload  = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    if resp.status_code in (200, 201):
        print(f"  ✓ GitHub: {repo_path} {'updated' if sha else 'created'}")
    else:
        print(f"  ✗ GitHub error {resp.status_code}: {resp.json().get('message')}")


# ── Shared scraping logic (mirrors 01_scrape_historical.py) ───────────────────

_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)


def get_soup(url, retries=3, delay=1.2):
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=20, allow_redirects=True)
            if resp.status_code == 200:
                time.sleep(delay)
                return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"  [WARN] {e}")
            time.sleep(delay * 2)
    return None


def _split_of(cell_str):
    if not cell_str or cell_str.strip() in ('---', '--', ''):
        return None, None
    if ' of ' in cell_str:
        parts = cell_str.strip().split(' of ')
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    try:
        return float(cell_str.strip()), None
    except ValueError:
        return None, None


def parse_stat(s):
    if not s or s.strip() in ('---', '--', ''):
        return None, None
    s = s.strip()
    if ' of ' in s:
        p = s.split(' of ')
        try:    return int(p[0]), int(p[1])
        except: return None, None
    if s.endswith('%'):
        try:    return float(s.rstrip('%')) / 100, None
        except: return None, None
    if re.match(r'^\d+:\d{2}$', s):
        p = s.split(':')
        return int(p[0]) * 60 + int(p[1]), None
    try:    return float(s), None
    except: return s, None


def _cell_val(td, fighter_idx):
    """
    UFCStats fight pages store BOTH fighters\'  values in a single <td>,
    each inside a <p> tag. fighter_idx 0 = fighter on left, 1 = fighter on right.
    Falls back to raw td text if no <p> tags found.
    """
    ps = td.select("p")
    if len(ps) >= 2:
        return ps[fighter_idx].get_text(strip=True)
    elif len(ps) == 1:
        return ps[0].get_text(strip=True)
    return td.get_text(separator=" ", strip=True)


def parse_fight_page(fight_url, event_name, event_date, event_location):
    """
    Scrape a single fight page and return a list of row dicts — one per fighter.

    Key UFCStats structure insight: each <td> in the stats tables contains
    BOTH fighters\' values in separate <p> tags — index 0 = fighter 1, index 1 = fighter 2.
    """
    soup = get_soup(fight_url)
    if soup is None:
        return []

    # ── Fight metadata ────────────────────────────────────────────────────────
    method, round_num, time_str, time_fmt, referee = None, None, None, None, None

    for item in soup.select(".b-fight-details__text-item"):
        label_el = item.select_one(".b-fight-details__label")
        if not label_el:
            continue
        lbl = label_el.get_text(strip=True).lower()
        # value = everything in the item after stripping the label text
        raw = item.get_text(strip=True)
        lbl_text = label_el.get_text(strip=True)
        v = raw[len(lbl_text):].strip()
        if "method" in lbl:
            method = v
        elif "round" in lbl and "format" not in lbl:
            try: round_num = int(v)
            except: pass
        elif "time" in lbl and "format" not in lbl:
            time_str = v
        elif "format" in lbl:
            time_fmt = v
        elif "referee" in lbl:
            referee = v

    # Weight class and title bout
    weight_class, title_bout = None, False
    bout_title_el = soup.select_one(".b-fight-details__fight-title")
    if bout_title_el:
        bt = bout_title_el.get_text(strip=True)
        title_bout   = "title" in bt.lower() or "championship" in bt.lower()
        weight_class = re.sub(r"(ufc\s+)?(interim\s+)?(title\s+)?bout", "",
                               bt, flags=re.IGNORECASE).strip()

    # Fighter names and results
    fighter_names, fighter_results = [], []
    for el in soup.select(".b-fight-details__person"):
        name_el   = el.select_one(".b-fight-details__person-name a")
        result_el = el.select_one(".b-fight-details__person-status")
        if name_el:
            fighter_names.append(name_el.get_text(strip=True))
            fighter_results.append(result_el.get_text(strip=True) if result_el else "")

    if len(fighter_names) < 2:
        return []

    # ── Stats tables ──────────────────────────────────────────────────────────
    # Table 0 = totals, Table 1 = significant strikes breakdown
    tables = soup.select(".b-fight-details__table")
    totals_table = tables[0] if len(tables) > 0 else None
    sig_table    = tables[1] if len(tables) > 1 else None

    # Get the single data row from each table (one row contains both fighters)
    def get_data_row(table):
        if table is None:
            return None
        for tr in table.select("tbody tr"):
            tds = tr.select("td")
            if tds and any(td.select("p") for td in tds):
                return tds
        return None

    totals_tds = get_data_row(totals_table)
    sig_tds    = get_data_row(sig_table)

    rows = []
    for idx in range(2):
        name   = fighter_names[idx]
        result = fighter_results[idx]
        won    = 1 if result.upper() == "W" else 0
        opp    = fighter_names[1 - idx]

        # ── Totals ────────────────────────────────────────────────────────────
        # Column order: Fighter | KD | Sig.Str. | Sig.Str.% | Total Str. | Td | Td% | Sub. Att | Rev. | Ctrl
        kd_val = sig_landed = sig_att = sig_pct = None
        tot_landed = tot_att = td_landed = td_att = td_pct = None
        sub_att_val = rev_val = ctrl_secs = None

        if totals_tds and len(totals_tds) >= 9:
            kd_val,              _ = parse_stat(_cell_val(totals_tds[1], idx))
            sig_landed,    sig_att = _split_of(_cell_val(totals_tds[2], idx))
            sig_pct,             _ = parse_stat(_cell_val(totals_tds[3], idx))
            tot_landed,    tot_att = _split_of(_cell_val(totals_tds[4], idx))
            td_landed,      td_att = _split_of(_cell_val(totals_tds[5], idx))
            td_pct,              _ = parse_stat(_cell_val(totals_tds[6], idx))
            sub_att_val,         _ = parse_stat(_cell_val(totals_tds[7], idx))
            rev_val,             _ = parse_stat(_cell_val(totals_tds[8], idx))
            if len(totals_tds) > 9:
                ctrl_raw = _cell_val(totals_tds[9], idx)
                ctrl_secs, _ = parse_stat(ctrl_raw)

        # ── Sig strike breakdown ──────────────────────────────────────────────
        # Column order: Fighter | Sig.Str. | Sig.Str.% | Head | Body | Leg | Distance | Clinch | Ground
        head_l = head_a = body_l = body_a = leg_l = leg_a = None
        dist_l = dist_a = clinch_l = clinch_a = ground_l = ground_a = None

        if sig_tds and len(sig_tds) >= 9:
            head_l,   head_a   = _split_of(_cell_val(sig_tds[3], idx))
            body_l,   body_a   = _split_of(_cell_val(sig_tds[4], idx))
            leg_l,    leg_a    = _split_of(_cell_val(sig_tds[5], idx))
            dist_l,   dist_a   = _split_of(_cell_val(sig_tds[6], idx))
            clinch_l, clinch_a = _split_of(_cell_val(sig_tds[7], idx))
            if len(sig_tds) > 8:
                ground_l, ground_a = _split_of(_cell_val(sig_tds[8], idx))

        rows.append({
            "fight_url":        fight_url,
            "event_name":       event_name,
            "event_date":       event_date,
            "event_location":   event_location,
            "fighter":          name,
            "opponent":         opp,
            "won":              won,
            "weight_class":     weight_class,
            "title_bout":       int(title_bout),
            "method":           method,
            "finish_round":     round_num,
            "finish_time":      time_str,
            "time_format":      time_fmt,
            "referee":          referee,
            "kd":               kd_val,
            "sig_str_landed":   sig_landed,
            "sig_str_att":      sig_att,
            "sig_str_pct":      sig_pct,
            "total_str_landed": tot_landed,
            "total_str_att":    tot_att,
            "td_landed":        td_landed,
            "td_att":           td_att,
            "td_pct":           td_pct,
            "sub_att":          sub_att_val,
            "reversals":        rev_val,
            "ctrl_seconds":     ctrl_secs,
            "head_landed":      head_l,
            "head_att":         head_a,
            "body_landed":      body_l,
            "body_att":         body_a,
            "leg_landed":       leg_l,
            "leg_att":          leg_a,
            "distance_landed":  dist_l,
            "distance_att":     dist_a,
            "clinch_landed":    clinch_l,
            "clinch_att":       clinch_a,
            "ground_landed":    ground_l,
            "ground_att":       ground_a,
        })

    return rows



def get_fight_urls_from_event(event_url):
    soup = get_soup(event_url, delay=0.8)
    if soup is None:
        return []
    return [
        row.get("data-link")
        for row in soup.select(".b-fight-details__table-row.b-fight-details__table-row__hover")
        if row.get("data-link") and "fight-details" in row.get("data-link")
    ]


# ── Master CSV + Rankings merge (weekly) ──────────────────────────────────────

def _norm_name(s):
    return re.sub(r'\s+', ' ', str(s).lower().strip())


def _safe_float(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


MASTER_ODDS_COLS = [
    "historical_odds", "historical_odds_opp", "expected_value",
    "dec_odds", "dec_odds_opp", "sub_odds", "sub_odds_opp",
    "ko_odds", "ko_odds_opp",
]
MASTER_FIGHTER_COLS = [
    "height_cms", "reach_cms", "weight_lbs", "stance", "age",
    "wins", "losses", "draws", "current_win_streak", "current_lose_streak",
    "longest_win_streak", "wins_by_ko", "wins_by_sub",
    "wins_by_dec_unanimous", "wins_by_dec_split", "wins_by_dec_majority",
    "wins_by_tko_doctor", "total_rounds_fought", "total_title_bouts",
]
MASTER_FIGHT_COLS = [
    "number_of_rounds", "empty_arena", "finish_details", "total_fight_time_secs",
]
MASTER_DIFF_COLS = [
    "height_dif", "reach_dif", "age_dif", "win_dif", "loss_dif",
    "win_streak_dif", "lose_streak_dif", "longest_win_streak_dif",
    "ko_dif", "sub_dif", "total_round_dif", "total_title_bout_dif",
    "sig_str_dif", "avg_sub_att_dif", "avg_td_dif",
]
ALL_MASTER_COLS = MASTER_ODDS_COLS + MASTER_FIGHTER_COLS + MASTER_FIGHT_COLS + MASTER_DIFF_COLS


def load_master_flat_from_github():
    df, _ = read_csv_from_github("data/raw/ufc-master.csv")
    if df.empty:
        print("    [WARN] ufc-master.csv not found in GitHub repo — skipping")
        return pd.DataFrame()
    print(f"    Loaded ufc-master.csv: {len(df):,} rows")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    records = []
    for _, row in df.iterrows():
        for color, opp_color in [("Red", "Blue"), ("Blue", "Red")]:
            sign = 1 if color == "Red" else -1
            r = {
                "fighter_norm":        _norm_name(row.get(f"{color}Fighter", "")),
                "event_date":          row["Date"],
                "historical_odds":     _safe_float(row.get(f"{color}Odds")),
                "historical_odds_opp": _safe_float(row.get(f"{opp_color}Odds")),
                "expected_value":      _safe_float(row.get(f"{color}ExpectedValue")),
                "dec_odds":     _safe_float(row.get(f"{color[0]}DecOdds")),
                "dec_odds_opp": _safe_float(row.get(f"{opp_color[0]}DecOdds")),
                "sub_odds":     _safe_float(row.get(f"{color[0]}SubOdds")),
                "sub_odds_opp": _safe_float(row.get(f"{opp_color[0]}SubOdds")),
                "ko_odds":      _safe_float(row.get(f"{color[0]}KOOdds")),
                "ko_odds_opp":  _safe_float(row.get(f"{opp_color[0]}KOOdds")),
                "height_cms":   _safe_float(row.get(f"{color}HeightCms")),
                "reach_cms":    _safe_float(row.get(f"{color}ReachCms")),
                "weight_lbs":   _safe_float(row.get(f"{color}WeightLbs")),
                "stance":       str(row.get(f"{color}Stance", "")).strip() or None,
                "age":          _safe_float(row.get(f"{color}Age")),
                "wins":                  _safe_float(row.get(f"{color}Wins")),
                "losses":                _safe_float(row.get(f"{color}Losses")),
                "draws":                 _safe_float(row.get(f"{color}Draws")),
                "current_win_streak":    _safe_float(row.get(f"{color}CurrentWinStreak")),
                "current_lose_streak":   _safe_float(row.get(f"{color}CurrentLoseStreak")),
                "longest_win_streak":    _safe_float(row.get(f"{color}LongestWinStreak")),
                "wins_by_ko":            _safe_float(row.get(f"{color}WinsByKO")),
                "wins_by_sub":           _safe_float(row.get(f"{color}WinsBySubmission")),
                "wins_by_dec_unanimous": _safe_float(row.get(f"{color}WinsByDecisionUnanimous")),
                "wins_by_dec_split":     _safe_float(row.get(f"{color}WinsByDecisionSplit")),
                "wins_by_dec_majority":  _safe_float(row.get(f"{color}WinsByDecisionMajority")),
                "wins_by_tko_doctor":    _safe_float(row.get(f"{color}WinsByTKODoctorStoppage")),
                "total_rounds_fought":   _safe_float(row.get(f"{color}TotalRoundsFought")),
                "total_title_bouts":     _safe_float(row.get(f"{color}TotalTitleBouts")),
                "number_of_rounds":      _safe_float(row.get("NumberOfRounds")),
                "empty_arena":           int(bool(row.get("EmptyArena", False))),
                "finish_details":        str(row.get("FinishDetails", "")).strip() or None,
                "total_fight_time_secs": _safe_float(row.get("TotalFightTimeSecs")),
                "height_dif":            _safe_float(row.get("HeightDif")) * sign,
                "reach_dif":             _safe_float(row.get("ReachDif"))  * sign,
                "age_dif":               _safe_float(row.get("AgeDif"))    * sign,
                "win_dif":               _safe_float(row.get("WinDif"))    * sign,
                "loss_dif":              _safe_float(row.get("LossDif"))   * sign,
                "win_streak_dif":        _safe_float(row.get("WinStreakDif")) * sign,
                "lose_streak_dif":       _safe_float(row.get("LoseStreakDif")) * sign,
                "longest_win_streak_dif":_safe_float(row.get("LongestWinStreakDif")) * sign,
                "ko_dif":                _safe_float(row.get("KODif"))     * sign,
                "sub_dif":               _safe_float(row.get("SubDif"))    * sign,
                "total_round_dif":       _safe_float(row.get("TotalRoundDif")) * sign,
                "total_title_bout_dif":  _safe_float(row.get("TotalTitleBoutDif")) * sign,
                "sig_str_dif":           _safe_float(row.get("SigStrDif")) * sign,
                "avg_sub_att_dif":       _safe_float(row.get("AvgSubAttDif")) * sign,
                "avg_td_dif":            _safe_float(row.get("AvgTDDif"))  * sign,
            }
            records.append(r)
    flat = pd.DataFrame(records)
    flat["event_date"] = pd.to_datetime(flat["event_date"], errors="coerce")
    return flat.dropna(subset=["fighter_norm", "event_date"])


def merge_master_into_new(new_df, master_flat):
    for col in ALL_MASTER_COLS:
        if col not in new_df.columns:
            new_df[col] = np.nan
    if master_flat.empty:
        return new_df
    new_df = new_df.copy()
    new_df["_f_norm"] = new_df["fighter"].apply(_norm_name)
    new_df["_date"]   = pd.to_datetime(new_df["event_date"], errors="coerce")
    matched = 0
    for idx, row in new_df.iterrows():
        fname  = row["_f_norm"]
        fdate  = row["_date"]
        window = master_flat[
            (master_flat["event_date"] >= fdate - pd.Timedelta(days=3)) &
            (master_flat["event_date"] <= fdate + pd.Timedelta(days=3))
        ]
        hit = window[window["fighter_norm"] == fname]
        if hit.empty:
            last = fname.split()[-1] if " " in fname else fname
            hit  = window[window["fighter_norm"].str.contains(last, na=False, regex=False)]
        if hit.empty:
            continue
        src = hit.iloc[0]
        for col in ALL_MASTER_COLS:
            if col in src.index and pd.isna(new_df.at[idx, col]):
                new_df.at[idx, col] = src[col]
        matched += 1
    new_df = new_df.drop(columns=["_f_norm", "_date"], errors="ignore")
    print(f"    Master merge: {matched}/{len(new_df)} new rows matched")
    return new_df


def backfill_bfo_odds(fight_df):
    if "historical_odds" not in fight_df.columns:
        fight_df["historical_odds"] = np.nan
    if not fight_df["historical_odds"].isna().any():
        return fight_df
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/predictions"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        return fight_df
    files = [f["name"] for f in resp.json()
             if f["name"].startswith("v2_betting_recommendations_") and f["name"].endswith(".csv")]
    if not files:
        return fight_df
    all_odds = []
    for fname in files:
        pred_df, _ = read_csv_from_github(f"predictions/{fname}")
        if pred_df.empty or not all(c in pred_df.columns for c in ["name","odds_numeric","event_date"]):
            continue
        sub = pred_df[["name","odds_numeric","event_date"]].copy()
        sub.columns = ["fighter","odds_numeric","event_date"]
        sub["event_date"] = pd.to_datetime(sub["event_date"], errors="coerce")
        all_odds.append(sub.dropna())
    if not all_odds:
        return fight_df
    pool = pd.concat(all_odds, ignore_index=True)
    pool["f_norm"] = pool["fighter"].apply(_norm_name)
    fight_df = fight_df.copy()
    fight_df["_date"]   = pd.to_datetime(fight_df["event_date"], errors="coerce")
    fight_df["_f_norm"] = fight_df["fighter"].apply(_norm_name)
    filled = 0
    for idx in fight_df[fight_df["historical_odds"].isna()].index:
        row    = fight_df.loc[idx]
        window = pool[
            (pool["event_date"] >= row["_date"] - pd.Timedelta(days=3)) &
            (pool["event_date"] <= row["_date"] + pd.Timedelta(days=3))
        ]
        hit = window[window["f_norm"] == row["_f_norm"]]
        if hit.empty:
            last = row["_f_norm"].split()[-1]
            hit  = window[window["f_norm"].str.contains(last, na=False, regex=False)]
        if not hit.empty:
            fight_df.at[idx, "historical_odds"] = hit.iloc[0]["odds_numeric"]
            filled += 1
    fight_df = fight_df.drop(columns=["_date","_f_norm"], errors="ignore")
    print(f"    BFO backfill: {filled} additional odds")
    return fight_df


def scrape_ufc_rankings():
    UFC_RANKINGS_URL = "https://www.ufc.com/rankings"
    UFC_DIV_ORDER = [
        "Flyweight","Bantamweight","Featherweight","Lightweight",
        "Welterweight","Middleweight","Light Heavyweight","Heavyweight",
        "Women's Strawweight","Women's Flyweight","Women's Bantamweight",
        "Women's Featherweight","Pound-for-Pound","Women's Pound-for-Pound",
    ]
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        resp = sess.get(UFC_RANKINGS_URL, timeout=20)
        if resp.status_code != 200:
            print(f"    [WARN] UFC rankings HTTP {resp.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"    [WARN] UFC rankings: {e}")
        return pd.DataFrame()
    soup  = BeautifulSoup(resp.text, "lxml")
    today = datetime.now().strftime("%Y-%m-%d")
    rows  = []
    for t_idx, table in enumerate(soup.select("table")):
        if t_idx >= len(UFC_DIV_ORDER):
            break
        wc      = UFC_DIV_ORDER[t_idx]
        caption = table.find("caption")
        if caption:
            champ_link = caption.select_one("h5 a")
            if champ_link:
                rows.append({"date": today, "weightclass": wc,
                             "fighter": champ_link.get_text(strip=True), "rank": 0})
        for tr in table.select("tbody tr"):
            tds = tr.select("td")
            if len(tds) < 2:
                continue
            try:
                rank_num = int(tds[0].get_text(strip=True))
            except ValueError:
                continue
            link = tds[1].find("a")
            if link and link.get_text(strip=True):
                rows.append({"date": today, "weightclass": wc,
                             "fighter": link.get_text(strip=True), "rank": rank_num})
    if not rows:
        print("    [WARN] No rankings found from ufc.com")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(f"    ✓ UFC rankings: {len(df)} entries, {df['weightclass'].nunique()} divisions")
    return df


def update_rankings_on_github(new_ranks_df):
    if new_ranks_df.empty:
        return
    existing_df, sha = read_csv_from_github("data/raw/rankings_history.csv")
    today = datetime.now().strftime("%Y-%m-%d")
    if not existing_df.empty and "date" in existing_df.columns:
        existing_df["date"] = existing_df["date"].astype(str)
        if today in existing_df["date"].values:
            print(f"    Rankings for {today} already saved — skipping")
            return
        combined = pd.concat([existing_df, new_ranks_df], ignore_index=True)
    else:
        combined = new_ranks_df
    write_csv_to_github(combined, "data/raw/rankings_history.csv",
                        f"Weekly rankings — {today}", sha=sha)
    print(f"    Rankings history: {len(combined):,} total rows")


def merge_rankings_into_new(new_df, rankings_df):
    WC_MAP = {
        "Strawweight":"Strawweight","Women's Strawweight":"Women's Strawweight",
        "Flyweight":"Flyweight","Women's Flyweight":"Women's Flyweight",
        "Bantamweight":"Bantamweight","Women's Bantamweight":"Women's Bantamweight",
        "Featherweight":"Featherweight","Women's Featherweight":"Women's Featherweight",
        "Lightweight":"Lightweight","Welterweight":"Welterweight",
        "Middleweight":"Middleweight","Light Heavyweight":"Light Heavyweight",
        "Heavyweight":"Heavyweight",
    }
    for col in ["fighter_rank_wc","fighter_rank_pfp","diff_rank_wc","diff_rank_pfp"]:
        if col not in new_df.columns:
            new_df[col] = np.nan
    if rankings_df.empty:
        return new_df
    rankings_df = rankings_df.copy()
    rankings_df["date"]         = pd.to_datetime(rankings_df["date"], errors="coerce")
    rankings_df["fighter_norm"] = rankings_df["fighter"].apply(_norm_name)
    rankings_df["rank"]         = pd.to_numeric(rankings_df.get("rank", np.nan), errors="coerce")
    pfp = rankings_df[rankings_df["weightclass"] == "Pound-for-Pound"]
    wc  = rankings_df[rankings_df["weightclass"] != "Pound-for-Pound"]
    def get_rank(name_norm, fight_date, wc_str, df_sub):
        past = df_sub[(df_sub["fighter_norm"] == name_norm) & (df_sub["date"] <= fight_date)]
        if past.empty:
            last = name_norm.split()[-1] if " " in name_norm else name_norm
            past = df_sub[df_sub["fighter_norm"].str.contains(last, na=False, regex=False) &
                          (df_sub["date"] <= fight_date)]
            if wc_str and not past.empty:
                wm = past[past["weightclass"] == wc_str]
                if not wm.empty:
                    past = wm
        if past.empty:
            return np.nan
        latest = past[past["date"] == past["date"].max()]
        if wc_str:
            wm = latest[latest["weightclass"] == wc_str]
            if not wm.empty:
                return float(wm["rank"].iloc[0])
        return float(latest["rank"].iloc[0])
    new_df = new_df.copy()
    new_df["_f_norm"] = new_df["fighter"].apply(_norm_name)
    new_df["_o_norm"] = new_df["opponent"].apply(_norm_name)
    new_df["_date"]   = pd.to_datetime(new_df["event_date"], errors="coerce")
    new_df["_wc"]     = new_df["weight_class"].apply(lambda x: WC_MAP.get(str(x).strip(), str(x).strip()))
    rank_wc_vals, rank_pfp_vals = [], []
    for _, row in new_df.iterrows():
        rank_wc_vals.append(get_rank(row["_f_norm"], row["_date"], row["_wc"], wc))
        rank_pfp_vals.append(get_rank(row["_f_norm"], row["_date"], None, pfp))
    new_df["fighter_rank_wc"]  = rank_wc_vals
    new_df["fighter_rank_pfp"] = rank_pfp_vals
    new_df["_rank_wc"]         = rank_wc_vals
    new_df["_rank_pfp"]        = rank_pfp_vals
    opp_wc  = dict(zip(zip(new_df["_o_norm"], new_df["fight_url"]), new_df["_rank_wc"]))
    opp_pfp = dict(zip(zip(new_df["_o_norm"], new_df["fight_url"]), new_df["_rank_pfp"]))
    new_df["diff_rank_wc"] = [
        (r["_rank_wc"] - opp_wc.get((r["_f_norm"], r["fight_url"]), np.nan))
        if not (np.isnan(r["_rank_wc"]) or np.isnan(opp_wc.get((r["_f_norm"], r["fight_url"]), np.nan)))
        else np.nan for _, r in new_df.iterrows()
    ]
    new_df["diff_rank_pfp"] = [
        (r["_rank_pfp"] - opp_pfp.get((r["_f_norm"], r["fight_url"]), np.nan))
        if not (np.isnan(r["_rank_pfp"]) or np.isnan(opp_pfp.get((r["_f_norm"], r["fight_url"]), np.nan)))
        else np.nan for _, r in new_df.iterrows()
    ]
    new_df = new_df.drop(columns=["_f_norm","_o_norm","_date","_wc","_rank_wc","_rank_pfp"], errors="ignore")
    print(f"    Rankings: {new_df['fighter_rank_wc'].notna().sum()}/{len(new_df)} WC ranks attached")
    return new_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Weekly Data Update")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    existing_df, existing_sha = read_csv_from_github("data/raw/fight_stats.csv")
    if existing_df.empty:
        print("[ERROR] No existing fight_stats.csv. Run 01_scrape_historical.py first.")
        return

    latest_date   = pd.to_datetime(existing_df["event_date"]).max()
    existing_urls = set(existing_df["fight_url"].unique())
    print(f"Most recent event in dataset: {latest_date.strftime('%Y-%m-%d')}")

    # A: Scrape + save current UFC rankings
    print("\n[A] Scraping current UFC rankings from ufc.com...")
    new_ranks = scrape_ufc_rankings()
    update_rankings_on_github(new_ranks)

    # B: Scrape new UFCStats fight rows
    print("\n[B] Checking for new UFC fights...")
    new_rows = []
    stop = False
    for page in range(1, 6):
        if stop:
            break
        url  = f"{BASE_URL}/statistics/events/completed?page={page}"
        soup = get_soup(url, delay=0.8)
        if soup is None:
            break
        for row in soup.select("tr.b-statistics__table-row"):
            if row.find("th"):
                continue
            tds = row.find_all("td")
            if not tds:
                continue
            link_el = tds[0].select_one("i.b-statistics__table-content a")
            if not link_el:
                continue
            event_url  = link_el.get("href", "").strip()
            event_name = link_el.get_text(strip=True)
            if not event_url or "event-details" not in event_url:
                continue
            date_span = tds[0].select_one("span.b-statistics__date")
            if not date_span:
                continue
            date_text  = date_span.get_text(strip=True)
            event_date = None
            for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
                try:
                    event_date = datetime.strptime(date_text, fmt)
                    break
                except ValueError:
                    continue
            if event_date is None:
                continue
            if event_date <= latest_date:
                stop = True
                break
            location = tds[1].get_text(strip=True) if len(tds) > 1 else ""
            date_str  = event_date.strftime("%Y-%m-%d")
            print(f"  New event: {event_name} ({date_str})")
            fight_urls     = get_fight_urls_from_event(event_url)
            new_fight_urls = [u for u in fight_urls if u not in existing_urls]
            for fight_url in new_fight_urls:
                rows = parse_fight_page(fight_url, event_name, date_str, location)
                new_rows.extend(rows)
                existing_urls.add(fight_url)

    if not new_rows:
        print("No new fight data found. Dataset is up to date.")
        return

    new_df = pd.DataFrame(new_rows)
    print(f"  {len(new_rows)} new fighter-fight rows scraped")

    # C: Merge master CSV attributes for new rows
    print("\n[C] Merging master CSV data...")
    master_flat = load_master_flat_from_github()
    new_df = merge_master_into_new(new_df, master_flat)

    # D: Backfill BFO odds for fights newer than master cutoff
    print("\n[D] Backfilling BFO odds for new fights...")
    new_df = backfill_bfo_odds(new_df)

    # E: Attach rankings
    print("\n[E] Attaching rankings to new fight rows...")
    rankings_df, _ = read_csv_from_github("data/raw/rankings_history.csv")
    new_df = merge_rankings_into_new(new_df, rankings_df)

    # F: Align columns and combine
    for col in new_df.columns:
        if col not in existing_df.columns:
            existing_df[col] = np.nan
    for col in existing_df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    combined = pd.concat([new_df, existing_df], ignore_index=True)
    combined = combined.sort_values("event_date", ascending=False).reset_index(drop=True)

    write_csv_to_github(combined, "data/raw/fight_stats.csv",
                        f"Weekly data update — {datetime.now().strftime('%Y-%m-%d')}",
                        sha=existing_sha)
    print(f"\n✓ Added {len(new_rows)} new rows. Total: {len(combined):,}")



def run_test():
    """
    --test mode: verifies all data pipeline components without touching GitHub writes.

    Tests:
      1. UFCStats completed events page (connectivity + parsing)
      2. Fight card URL scraping for most recent event
      3. Fight page stat parsing (all 38 columns)
      4. UFC.com rankings scraper (live request)
      5. Master CSV merge logic (offline, synthetic data)
      6. Rankings merge logic (offline, synthetic data)
    """
    print("=== TEST MODE: 02_scrape_weekly_update ===\n")
    all_pass = True

    # ── 1. UFCStats event page ────────────────────────────────────────────────
    print("[1] UFCStats completed events page...")
    url  = f"{BASE_URL}/statistics/events/completed?page=1"
    soup = get_soup(url, delay=0.5)
    if soup is None:
        print("  ✗ FAIL — could not reach UFCStats")
        all_pass = False
    else:
        event_url = event_name = event_date_str = location = None
        for row in soup.select("tr.b-statistics__table-row"):
            if row.find("th"):
                continue
            tds = row.find_all("td")
            if not tds:
                continue
            link_el   = tds[0].select_one("i.b-statistics__table-content a")
            date_span = tds[0].select_one("span.b-statistics__date")
            if not link_el or not date_span:
                continue
            event_url      = link_el.get("href", "")
            event_name     = link_el.get_text(strip=True)
            event_date_str = date_span.get_text(strip=True)
            location       = tds[1].get_text(strip=True) if len(tds) > 1 else ""
            break
        if event_url:
            print(f"  ✓ Most recent: {event_name} ({event_date_str})")
        else:
            print("  ✗ No events found")
            all_pass = False

    # ── 2. Fight card URLs ────────────────────────────────────────────────────
    print("\n[2] Fight card scraping...")
    fight_urls = []
    if event_url:
        fight_urls = get_fight_urls_from_event(event_url)
        if fight_urls:
            print(f"  ✓ {len(fight_urls)} fights found on card")
        else:
            print("  ✗ No fight URLs found")
            all_pass = False

    # ── 3. Fight page stat parsing ────────────────────────────────────────────
    print("\n[3] Fight page parsing (all columns)...")
    if fight_urls:
        rows = parse_fight_page(fight_urls[0], event_name, event_date_str, location)
        if rows:
            print(f"  ✓ {len(rows)} fighter rows, {len(rows[0])} columns each")
            none_count = sum(1 for r in rows for v in r.values() if v is None)
            print(f"  None values: {none_count}")
            print(f"  Columns: {list(rows[0].keys())}")
        else:
            print("  ✗ parse_fight_page returned no rows")
            all_pass = False

    # ── 4. UFC.com rankings scraper ───────────────────────────────────────────
    print("\n[4] UFC.com rankings scraper (live)...")
    ranks = scrape_ufc_rankings()
    if ranks.empty:
        print("  ✗ No rankings returned — check ufc.com layout")
        all_pass = False
    else:
        print(f"  ✓ {len(ranks)} entries, {ranks['weightclass'].nunique()} divisions")
        champions = ranks[ranks["rank"] == 0]
        print(f"  Champions found ({len(champions)}):")
        for _, r in champions.iterrows():
            print(f"    {r['weightclass']:<28} {r['fighter']}")
        ranked_counts = ranks[ranks["rank"] > 0].groupby("weightclass").size()
        if ranked_counts.min() < 5:
            print(f"  ⚠ Some divisions have fewer than 5 ranked fighters — check layout")

    # ── 5. Master CSV merge logic (offline) ───────────────────────────────────
    print("\n[5] Master CSV merge logic (offline, synthetic data)...")
    test_fights = pd.DataFrame([
        {"fight_url": "u1", "fighter": "Alexandre Pantoja", "opponent": "Kai Asakura",
         "event_date": "2024-12-07", "weight_class": "Flyweight"},
        {"fight_url": "u1", "fighter": "Kai Asakura", "opponent": "Alexandre Pantoja",
         "event_date": "2024-12-07", "weight_class": "Flyweight"},
        {"fight_url": "u2", "fighter": "Shavkat Rakhmonov", "opponent": "Ian Machado Garry",
         "event_date": "2024-12-07", "weight_class": "Welterweight"},
        {"fight_url": "u3", "fighter": "Nobody Known", "opponent": "Also Unknown",
         "event_date": "2020-01-01", "weight_class": "Lightweight"},
    ])
    # Synthetic master flat with the expected values from the sample you shared
    test_master = pd.DataFrame([
        {"fighter_norm": "alexandre pantoja", "event_date": pd.Timestamp("2024-12-07"),
         "historical_odds": -250.0, "historical_odds_opp": 215.0,
         "ko_odds": 400.0, "sub_odds": 150.0, "dec_odds": 300.0,
         "height_cms": 165.1, "reach_cms": 170.18, "age": 34.0,
         "height_dif": 7.62, "age_dif": 3.0,
         **{c: np.nan for c in ALL_MASTER_COLS if c not in ["historical_odds","historical_odds_opp","ko_odds","sub_odds","dec_odds","height_cms","reach_cms","age","height_dif","age_dif"]}},
        {"fighter_norm": "shavkat rakhmonov", "event_date": pd.Timestamp("2024-12-07"),
         "historical_odds": -210.0, "historical_odds_opp": 295.0,
         "ko_odds": 240.0, "sub_odds": np.nan, "dec_odds": 250.0,
         "height_cms": 190.5, "reach_cms": 187.96, "age": 30.0,
         "height_dif": 5.08, "age_dif": -3.0,
         **{c: np.nan for c in ALL_MASTER_COLS if c not in ["historical_odds","historical_odds_opp","ko_odds","sub_odds","dec_odds","height_cms","reach_cms","age","height_dif","age_dif"]}},
    ])
    result = merge_master_into_new(test_fights.copy(), test_master)
    pantoja_odds = result.loc[result["fighter"]=="Alexandre Pantoja","historical_odds"].values
    rakh_odds    = result.loc[result["fighter"]=="Shavkat Rakhmonov","historical_odds"].values
    unknown_odds = result.loc[result["fighter"]=="Nobody Known","historical_odds"].values

    p_ok = len(pantoja_odds) > 0 and abs(pantoja_odds[0] - (-250)) < 5
    r_ok = len(rakh_odds) > 0 and abs(rakh_odds[0] - (-210)) < 5
    u_ok = len(unknown_odds) > 0 and pd.isna(unknown_odds[0])
    print(f"  Pantoja ML: {pantoja_odds[0] if len(pantoja_odds) else 'N/A'} — {'✓' if p_ok else '✗'} (expect ~-250)")
    print(f"  Rakhmonov ML: {rakh_odds[0] if len(rakh_odds) else 'N/A'} — {'✓' if r_ok else '✗'} (expect ~-210)")
    print(f"  Nobody Known: {unknown_odds[0] if len(unknown_odds) else 'N/A'} — {'✓' if u_ok else '✗'} (expect NaN)")
    print(f"  Pantoja height_cms: {result.loc[0,'height_cms']}, age: {result.loc[0,'age']}, height_dif: {result.loc[0,'height_dif']}")
    if not (p_ok and r_ok and u_ok):
        all_pass = False

    # ── 6. Rankings merge logic (offline) ────────────────────────────────────
    print("\n[6] Rankings merge logic (offline, synthetic data)...")
    test_rankings = pd.DataFrame([
        {"date": "2024-11-01", "weightclass": "Flyweight", "fighter": "Alexandre Pantoja", "rank": 0},
        {"date": "2024-11-01", "weightclass": "Flyweight", "fighter": "Amir Albazi", "rank": 1},
        {"date": "2024-11-01", "weightclass": "Pound-for-Pound", "fighter": "Alexandre Pantoja", "rank": 3},
        {"date": "2024-11-01", "weightclass": "Welterweight", "fighter": "Shavkat Rakhmonov", "rank": 1},
        {"date": "2024-11-01", "weightclass": "Pound-for-Pound", "fighter": "Shavkat Rakhmonov", "rank": 5},
    ])
    result2 = merge_rankings_into_new(test_fights.copy(), test_rankings)
    p_wc  = result2.loc[result2["fighter"]=="Alexandre Pantoja","fighter_rank_wc"].values
    p_pfp = result2.loc[result2["fighter"]=="Alexandre Pantoja","fighter_rank_pfp"].values
    r_wc  = result2.loc[result2["fighter"]=="Shavkat Rakhmonov","fighter_rank_wc"].values
    unk_wc= result2.loc[result2["fighter"]=="Nobody Known","fighter_rank_wc"].values

    pw_ok  = len(p_wc)  > 0 and p_wc[0]  == 0.0
    pp_ok  = len(p_pfp) > 0 and p_pfp[0] == 3.0
    rw_ok  = len(r_wc)  > 0 and r_wc[0]  == 1.0
    uw_ok  = len(unk_wc)> 0 and pd.isna(unk_wc[0])
    print(f"  Pantoja WC rank:   {p_wc[0]  if len(p_wc)  else 'N/A'} — {'✓' if pw_ok else '✗'} (expect 0=champion)")
    print(f"  Pantoja PFP rank:  {p_pfp[0] if len(p_pfp) else 'N/A'} — {'✓' if pp_ok else '✗'} (expect 3)")
    print(f"  Rakhmonov WC rank: {r_wc[0]  if len(r_wc)  else 'N/A'} — {'✓' if rw_ok else '✗'} (expect 1)")
    print(f"  Nobody Known rank: {unk_wc[0] if len(unk_wc) else 'N/A'} — {'✓' if uw_ok else '✗'} (expect NaN)")
    diff_check = result2.loc[result2["fighter"]=="Alexandre Pantoja","diff_rank_wc"].values
    print(f"  Pantoja diff_rank_wc (vs Kai Asakura, unranked): {diff_check[0] if len(diff_check) else 'N/A'} (expect NaN — Asakura unranked)")
    if not (pw_ok and pp_ok and rw_ok and uw_ok):
        all_pass = False

    # ── Summary ───────────────────────────────────────────────────────────────
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
