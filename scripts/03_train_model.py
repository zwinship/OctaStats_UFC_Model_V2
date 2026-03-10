#!/usr/bin/env python3
"""
03_train_model.py
OctaStats V2 — Feature engineering + Dynamic Logit model training

Runs every Wednesday after the data update. Steps:
  1. Load raw fight_stats.csv (output of 01/02 scripts)
  2. Compute rolling career stats per fighter (no data leakage)
  3. Assign fighting styles (career-level, locked per historical fight)
  4. Build matchup-level feature matrix — ALL available features,
     let LASSO trim the irrelevant ones
  5. Train discrete-choice dynamic logit with LASSO regularization
     — state variables: fighter's last 1-3 in-fight styles (style momentum)
     — LASSO filters the covariate space
  6. Train multinomial logit for finish type prediction
     (KO/TKO | Submission | Decision — fight-level, not fighter-level)
  7. Train style-shift prediction model
  8. Serialize models to data/ and push to GitHub

New vs previous version:
  - FULL feature set: implied_prob + prop implied probs + scorecard_margin
    + all physical attributes + all career stat diffs + interaction terms
    (odds × style, odds × rank, momentum × career, physical × style, etc.)
  - Multinomial logit model for finish type prediction (3-class)
  - implied_prob from 01_scrape_historical.py used directly as a feature
    (encodes market information the model should use)

Environment variables:
    ZWINSHIP_PAT  — GitHub personal access token for zwinship account
"""

import os
import io
import base64
import pickle
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import expit  # sigmoid

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
import sys as _sys
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
if not GITHUB_TOKEN and "--test" not in _sys.argv:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
REPO_NAME    = "OctaStats_UFC_Model_V2"
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

STYLE_THRESHOLDS = {
    "Striker":   {"sig_str_att": -0.35, "total_str_att": -0.35, "head_att": -0.35, "kd": -0.35},
    "Wrestler":  {"ctrl_seconds": 0.0,   "ground_att": 0.0,      "td_att": 0.0},
    "BJJ":       {"sub_att": -0.45,      "ctrl_seconds": -0.45,  "td_att": -0.45},
    "Muay_Thai": {"clinch_att": -0.43,   "body_att": -0.43,      "leg_att": -0.43, "head_att": -0.43},
    "Sniper":    {"distance_att": -0.7,  "sig_str_pct": -0.7,    "head_att": -0.7,
                  "body_att": -0.7,      "leg_att": -0.7},
}

STYLE_SHIFT_THRESHOLD = 0.78

STYLE_ORDER = ["BJJ", "Mixed", "Muay_Thai", "Sniper", "Striker", "Wrestler"]

FINISH_CLASSES = ["KO_TKO", "Submission", "Decision"]   # multinomial target labels


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path, repo=None):
    _repo = repo or REPO_NAME
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{_repo}/contents/{repo_path}"
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


def upload_pickle_to_github(obj, repo_path, message, existing_sha=None):
    buf     = pickle.dumps(obj)
    content = base64.b64encode(buf).decode()
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    if existing_sha is None:
        check = requests.get(url, headers=GH_HEADERS)
        existing_sha = check.json().get("sha") if check.status_code == 200 else None
    payload = {"message": message, "content": content}
    if existing_sha:
        payload["sha"] = existing_sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    ok   = resp.status_code in (200, 201)
    print(f"  {'✓' if ok else '✗'} GitHub: {repo_path}")
    return ok


def write_csv_to_github(df, repo_path, message, sha=None):
    csv_str  = df.to_csv(index=False)
    content  = base64.b64encode(csv_str.encode()).decode()
    url      = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    if sha is None:
        check = requests.get(url, headers=GH_HEADERS)
        sha   = check.json().get("sha") if check.status_code == 200 else None
    payload  = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    print(f"  {'✓' if resp.status_code in (200,201) else '✗'} GitHub: {repo_path}")


# ── Step 1: Load & clean raw data ─────────────────────────────────────────────

def load_raw(df):
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["weight_class"] = (
        df["weight_class"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )
    stat_cols = ["sig_str_landed", "sig_str_att", "td_landed", "td_att", "ctrl_seconds"]
    df = df.dropna(subset=stat_cols, how="all")

    # Ensure numeric types for implied prob columns (from 01 script)
    for col in ["implied_prob", "implied_prob_ko", "implied_prob_sub", "implied_prob_dec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure scorecard_margin is numeric
    if "scorecard_margin" in df.columns:
        df["scorecard_margin"] = pd.to_numeric(df["scorecard_margin"], errors="coerce")

    return df


# ── Step 2: Rolling career stats (leak-free) ──────────────────────────────────

CAREER_STAT_COLS = [
    "kd", "sig_str_landed", "sig_str_att", "sig_str_pct",
    "total_str_landed", "total_str_att",
    "td_landed", "td_att", "td_pct",
    "sub_att", "reversals", "ctrl_seconds",
    "head_landed", "head_att",
    "body_landed", "body_att",
    "leg_landed",  "leg_att",
    "distance_landed", "distance_att",
    "clinch_landed",   "clinch_att",
    "ground_landed",   "ground_att",
]


def compute_career_stats(df):
    df = df.sort_values(["fighter", "event_date"]).copy()
    records = []

    for fighter, group in df.groupby("fighter"):
        group = group.sort_values("event_date").reset_index(drop=True)

        career_wins           = 0
        career_losses         = 0
        career_title_bouts    = 0
        current_win_streak    = 0
        current_lose_streak   = 0
        longest_win_streak    = 0
        win_by_ko             = 0
        win_by_sub            = 0
        win_by_dec            = 0
        total_scorecard_margin = 0.0
        decision_fights       = 0
        running_stats         = {col: [] for col in CAREER_STAT_COLS}

        for i, row in group.iterrows():
            n_prev    = len(running_stats["sig_str_landed"])
            career_avgs = {}
            for col in CAREER_STAT_COLS:
                vals = [v for v in running_stats[col]
                        if v is not None and not (isinstance(v, float) and np.isnan(v))]
                career_avgs[f"career_avg_{col}"] = np.mean(vals) if vals else np.nan

            records.append({
                "fighter":               fighter,
                "fight_url":             row["fight_url"],
                "event_date":            row["event_date"],
                "career_fights_before":  n_prev,
                "career_wins":           career_wins,
                "career_losses":         career_losses,
                "career_title_bouts":    career_title_bouts,
                "current_win_streak":    current_win_streak,
                "current_lose_streak":   current_lose_streak,
                "longest_win_streak":    longest_win_streak,
                "career_win_by_ko":      win_by_ko,
                "career_win_by_sub":     win_by_sub,
                "career_win_by_dec":     win_by_dec,
                "avg_scorecard_margin":  total_scorecard_margin / decision_fights if decision_fights > 0 else np.nan,
                **career_avgs,
            })

            # Update accumulators AFTER recording (leak-free)
            for col in CAREER_STAT_COLS:
                running_stats[col].append(row.get(col))

            # Track scorecard margin (decision dominance)
            sm = row.get("scorecard_margin")
            if sm is not None and not (isinstance(sm, float) and np.isnan(sm)):
                try:
                    total_scorecard_margin += float(sm)
                    decision_fights        += 1
                except (TypeError, ValueError):
                    pass

            if row["won"] == 1:
                career_wins        += 1
                current_win_streak  += 1
                current_lose_streak = 0
                longest_win_streak  = max(longest_win_streak, current_win_streak)
                method = str(row.get("method", "")).lower()
                if "ko" in method or "tko" in method:
                    win_by_ko += 1
                elif "sub" in method:
                    win_by_sub += 1
                elif "dec" in method:
                    win_by_dec += 1
            else:
                career_losses       += 1
                current_lose_streak += 1
                current_win_streak  = 0

            if row["title_bout"] == 1:
                career_title_bouts += 1

    return pd.DataFrame(records)


# ── Step 3: Fighting style assignment ─────────────────────────────────────────

def assign_fighting_styles(df, career_df):
    meta   = df[["fight_url", "fighter", "weight_class"]].drop_duplicates()
    merged = career_df.merge(meta, on=["fighter", "fight_url"], how="left")

    stat_cols  = [c for c in merged.columns if c.startswith("career_avg_")]
    has_history = merged["career_fights_before"] >= 3

    for col in stat_cols:
        raw_col = "std_" + col
        merged[raw_col] = np.nan
        for wc, grp_idx in merged[has_history].groupby("weight_class").groups.items():
            vals = merged.loc[grp_idx, col]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                merged.loc[grp_idx, raw_col] = (merged.loc[grp_idx, col] - mean) / std

    def classify_style(row):
        def z(field):
            col = f"std_career_avg_{field}"
            val = row.get(col)
            return val if (val is not None and not (isinstance(val, float) and np.isnan(val))) else -999

        t = STYLE_THRESHOLDS
        if (z("kd")            >= t["Striker"]["kd"] and
            z("sig_str_att")   >= t["Striker"]["sig_str_att"] and
            z("total_str_att") >= t["Striker"]["total_str_att"] and
            z("head_att")      >= t["Striker"]["head_att"]):
            return "Striker"
        if (z("ctrl_seconds") >= t["Wrestler"]["ctrl_seconds"] and
            z("ground_att")   >= t["Wrestler"]["ground_att"] and
            z("td_att")       >= t["Wrestler"]["td_att"]):
            return "Wrestler"
        if (z("sub_att")      >= t["BJJ"]["sub_att"] and
            z("ctrl_seconds") >= t["BJJ"]["ctrl_seconds"] and
            z("td_att")       >= t["BJJ"]["td_att"]):
            return "BJJ"
        if (z("clinch_att") >= t["Muay_Thai"]["clinch_att"] and
            z("body_att")   >= t["Muay_Thai"]["body_att"] and
            z("leg_att")    >= t["Muay_Thai"]["leg_att"] and
            z("head_att")   >= t["Muay_Thai"]["head_att"]):
            return "Muay_Thai"
        if (z("distance_att") >= t["Sniper"]["distance_att"] and
            z("sig_str_pct")  >= t["Sniper"]["sig_str_pct"] and
            z("head_att")     >= t["Sniper"]["head_att"] and
            z("body_att")     >= t["Sniper"]["body_att"] and
            z("leg_att")      >= t["Sniper"]["leg_att"]):
            return "Sniper"
        return "Mixed"

    merged["career_style"] = merged.apply(classify_style, axis=1)
    return merged


# ── Step 4: Build matchup feature matrix ──────────────────────────────────────

def build_matchup_features(df, career_styled):
    """
    Constructs the richest possible matchup feature matrix.
    LASSO will zero out features with no predictive value.

    Features included:
      1. Career stat diffs (rolling, leak-free)
      2. Physical attribute diffs (height, reach, age)
      3. Ranking diffs (WC + P4P)
      4. Market odds features (implied_prob from 01 script — market info)
      5. Prop implied prob features (ko/sub/dec market probs)
      6. Style matchup dummies (6×6 = 36 interactions)
      7. Style lag state variables (dynamic logit component)
      8. Weight class dummies
      9. Interaction terms:
           - odds × career momentum (market adjusts for recent form?)
           - physical × style (reach advantage × striker more predictive?)
           - title_bout × career stats (pressure handling)
           - empty_arena × home advantage proxy
           - scorecard_margin × decision tendency
    """
    fight_pairs = []

    # Build lookup for implied_prob columns from raw df
    ip_lookup = {}
    for cols_needed in ["implied_prob", "implied_prob_ko", "implied_prob_sub",
                        "implied_prob_dec", "implied_prob_source",
                        "historical_odds", "scorecard_margin",
                        "empty_arena", "title_bout", "number_of_rounds",
                        "height_cms", "reach_cms", "age",
                        "fighter_rank_wc", "fighter_rank_pfp"]:
        pass  # we'll pull these from df rows directly

    for fight_url, group in df.groupby("fight_url"):
        if len(group) != 2:
            continue
        a = group.iloc[0].copy()
        b = group.iloc[1].copy()

        ca = career_styled[career_styled["fight_url"] == fight_url]
        ca = ca.set_index("fighter")
        if a["fighter"] not in ca.index or b["fighter"] not in ca.index:
            continue

        row_a = ca.loc[a["fighter"]]
        row_b = ca.loc[b["fighter"]]

        feat = {
            "fight_url":       fight_url,
            "event_date":      a["event_date"],
            "weight_class":    a["weight_class"],
            "fighter_a":       a["fighter"],
            "fighter_b":       b["fighter"],
            "fighter_a_won":   int(a["won"]),
            "style_a":         row_a.get("career_style", "Mixed"),
            "style_b":         row_b.get("career_style", "Mixed"),
            # Store for multinomial model building
            "method_raw":      str(a.get("method", "")),
        }

        # ── Career stat diffs ──────────────────────────────────────────────────
        career_diff_cols = [
            "career_avg_kd", "career_avg_sig_str_landed", "career_avg_sig_str_att",
            "career_avg_sig_str_pct", "career_avg_td_landed", "career_avg_td_att",
            "career_avg_td_pct", "career_avg_sub_att", "career_avg_ctrl_seconds",
            "career_avg_head_att", "career_avg_body_att", "career_avg_leg_att",
            "career_avg_distance_att", "career_avg_clinch_att", "career_avg_ground_att",
            "career_avg_reversals",
            "career_wins", "career_losses", "career_title_bouts",
            "current_win_streak", "current_lose_streak", "longest_win_streak",
            "career_win_by_ko", "career_win_by_sub", "career_win_by_dec",
            "career_fights_before",
            "avg_scorecard_margin",  # decision dominance
        ]
        for col in career_diff_cols:
            try:
                feat[f"diff_{col}"] = float(row_a.get(col, np.nan)) - float(row_b.get(col, np.nan))
            except (TypeError, ValueError):
                feat[f"diff_{col}"] = np.nan

        # ── Physical attribute diffs ───────────────────────────────────────────
        for col in ["height_cms", "reach_cms", "age"]:
            try:
                feat[f"diff_{col}"] = float(a.get(col, np.nan)) - float(b.get(col, np.nan))
            except (TypeError, ValueError):
                feat[f"diff_{col}"] = np.nan

        # ── Ranking diffs ──────────────────────────────────────────────────────
        for rank_col in ["fighter_rank_wc", "fighter_rank_pfp"]:
            try:
                feat[f"diff_{rank_col}"] = float(a.get(rank_col, np.nan)) - float(b.get(rank_col, np.nan))
            except (TypeError, ValueError):
                feat[f"diff_{rank_col}"] = np.nan

        # ── Market odds features (from implied_prob column in fight_stats) ─────
        # implied_prob is ALWAYS filled (market where available, model-imputed otherwise)
        # This encodes the combined market signal directly into the model.
        ip_a = float(a.get("implied_prob", np.nan)) if pd.notna(a.get("implied_prob")) else np.nan
        ip_b = float(b.get("implied_prob", np.nan)) if pd.notna(b.get("implied_prob")) else np.nan
        feat["diff_implied_prob"]     = (ip_a - ip_b) if not (np.isnan(ip_a) or np.isnan(ip_b)) else np.nan
        feat["implied_prob_a"]        = ip_a   # absolute, not just diff
        feat["implied_prob_b"]        = ip_b
        feat["implied_prob_source_a"] = 1 if str(a.get("implied_prob_source", "")) == "market" else 0

        # ── Prop implied prob diffs ────────────────────────────────────────────
        for prop in ["implied_prob_ko", "implied_prob_sub", "implied_prob_dec"]:
            try:
                pa = float(a.get(prop, np.nan))
                pb = float(b.get(prop, np.nan))
                feat[f"diff_{prop}"] = (pa - pb) if not (np.isnan(pa) or np.isnan(pb)) else np.nan
            except (TypeError, ValueError):
                feat[f"diff_{prop}"] = np.nan

        # ── Historical American odds diff (raw, for completeness) ──────────────
        try:
            oa = float(a.get("historical_odds", np.nan))
            ob = float(b.get("historical_odds", np.nan))
            feat["diff_historical_odds"] = (oa - ob) if not (np.isnan(oa) or np.isnan(ob)) else np.nan
        except (TypeError, ValueError):
            feat["diff_historical_odds"] = np.nan

        # ── Fight-level context features ───────────────────────────────────────
        try:
            feat["title_bout"]       = int(a.get("title_bout", 0))
            feat["empty_arena"]      = int(a.get("empty_arena", 0))
            feat["number_of_rounds"] = int(a.get("number_of_rounds", 3))
        except (TypeError, ValueError):
            feat["title_bout"]       = 0
            feat["empty_arena"]      = 0
            feat["number_of_rounds"] = 3

        # ── Interaction terms ──────────────────────────────────────────────────
        # Market odds × career momentum: does the market overweight recent form?
        diff_ws = feat.get("diff_current_win_streak", 0.0) or 0.0
        feat["odds_x_win_streak"]   = (feat.get("diff_implied_prob", 0) or 0) * diff_ws

        # Physical × style (reach advantage more important for strikers/snipers?)
        # Will be computed post style-encoding in encode_styles step below,
        # but we store the raw components here for convenience
        feat["reach_x_ip"]   = (feat.get("diff_reach_cms", 0) or 0) * (feat.get("diff_implied_prob", 0) or 0)
        feat["height_x_ip"]  = (feat.get("diff_height_cms", 0) or 0) * (feat.get("diff_implied_prob", 0) or 0)

        # Title bout × career stats interaction
        feat["title_x_fights"] = feat["title_bout"] * (feat.get("diff_career_fights_before", 0) or 0)
        feat["title_x_rank"]   = feat["title_bout"] * (feat.get("diff_fighter_rank_wc", 0) or 0)

        # Scorecard margin × KO tendency (powerful finish types)
        feat["ko_x_scorecard"] = (feat.get("diff_career_win_by_ko", 0) or 0) * (feat.get("diff_avg_scorecard_margin", 0) or 0)

        fight_pairs.append(feat)

    return pd.DataFrame(fight_pairs)


# ── Step 5: Dynamic Logit with LASSO ──────────────────────────────────────────

def encode_styles(df):
    df = df.copy()
    for s in STYLE_ORDER:
        df[f"style_a_{s}"] = (df["style_a"] == s).astype(int)
        df[f"style_b_{s}"] = (df["style_b"] == s).astype(int)

    # Full 36 matchup interaction dummies
    for sa in STYLE_ORDER:
        for sb in STYLE_ORDER:
            df[f"matchup_{sa}_vs_{sb}"] = (
                (df["style_a"] == sa) & (df["style_b"] == sb)
            ).astype(int)

    # Style × implied_prob interactions (does market discount certain styles?)
    ip_diff = df.get("diff_implied_prob", pd.Series(0.0, index=df.index)).fillna(0)
    for s in STYLE_ORDER:
        df[f"style_a_{s}_x_ip"] = df[f"style_a_{s}"] * ip_diff
        df[f"style_b_{s}_x_ip"] = df[f"style_b_{s}"] * ip_diff

    # Style × reach diff (reach matters more for striking styles)
    reach_diff = df.get("diff_reach_cms", pd.Series(0.0, index=df.index)).fillna(0)
    for s in ["Striker", "Sniper", "Muay_Thai"]:
        df[f"style_a_{s}_x_reach"] = df[f"style_a_{s}"] * reach_diff

    return df


def add_state_variables(matchup_df):
    df = matchup_df.sort_values("event_date").copy()

    for fighter_col in ["fighter_a", "fighter_b"]:
        prefix    = "a" if fighter_col == "fighter_a" else "b"
        style_col = f"style_{prefix}"

        style_history = {}
        lag_records   = {f"style_{prefix}_lag{k}": [] for k in range(1, 4)}

        for idx, row in df.iterrows():
            fighter = row[fighter_col]
            hist    = style_history.get(fighter, [])
            for k in range(1, 4):
                lag_records[f"style_{prefix}_lag{k}"].append(
                    hist[-(k)] if len(hist) >= k else "Unknown"
                )
            style_history[fighter] = hist + [row[style_col]]

        for col, vals in lag_records.items():
            df[col] = vals

        for k in range(1, 4):
            lag_col = f"style_{prefix}_lag{k}"
            for s in STYLE_ORDER + ["Unknown"]:
                df[f"{lag_col}_{s}"] = (df[lag_col] == s).astype(int)

    return df


def train_dynamic_logit(matchup_df):
    df = encode_styles(matchup_df)
    df = add_state_variables(df)

    # ── Feature groups ────────────────────────────────────────────────────────
    diff_cols      = [c for c in df.columns if c.startswith("diff_")]
    # Exclude raw string columns style_a/style_b (already one-hot encoded) and raw lag strings
    style_cols     = [c for c in df.columns if (c.startswith("style_") or c.startswith("matchup_"))
                      and "_lag" not in c and c not in ("style_a", "style_b")]
    lag_hot_cols   = [c for c in df.columns if "_lag" in c and any(s in c for s in STYLE_ORDER + ["Unknown"])]
    abs_feat_cols  = ["implied_prob_a", "implied_prob_b", "implied_prob_source_a",
                      "title_bout", "empty_arena", "number_of_rounds",
                      "odds_x_win_streak", "reach_x_ip", "height_x_ip",
                      "title_x_fights", "title_x_rank", "ko_x_scorecard"]
    abs_feat_cols  = [c for c in abs_feat_cols if c in df.columns]

    # Weight class one-hot (drop Lightweight as reference)
    wc_dummies = pd.get_dummies(df["weight_class"], prefix="wc", drop_first=False)
    if "wc_Lightweight" in wc_dummies.columns:
        wc_dummies = wc_dummies.drop(columns=["wc_Lightweight"])
    wc_cols = list(wc_dummies.columns)
    df = pd.concat([df.reset_index(drop=True), wc_dummies.reset_index(drop=True)], axis=1)

    feature_cols = diff_cols + style_cols + lag_hot_cols + abs_feat_cols + wc_cols
    # Remove duplicates while preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    model_df = df.dropna(subset=["fighter_a_won"])
    model_df = model_df[model_df["career_fights_before"].fillna(0).apply(
        lambda x: x >= 3 if not np.isnan(x) else False
    ) if "career_fights_before" in model_df.columns else pd.Series(True, index=model_df.index)]

    # Fill NaN features with 0 (LASSO will zero-weight unreliable sparse features)
    for col in feature_cols:
        if col not in model_df.columns:
            model_df[col] = 0.0
        else:
            model_df[col] = model_df[col].fillna(0.0)

    if len(model_df) < 50:
        print(f"  [WARN] Only {len(model_df)} rows after filtering — model may be unreliable")

    X = model_df[feature_cols].values
    y = model_df["fighter_a_won"].values

    # ── Slight recency weighting ───────────────────────────────────────────────
    # Linear decay: oldest fight gets weight 0.55, newest gets 1.0.
    # This is intentionally mild — we don't want to overfit to recent noise,
    # just nudge the model to weight the modern era a touch more.
    _dates = pd.to_datetime(model_df["event_date"])
    _d_min = _dates.min()
    _d_max = _dates.max()
    _span  = (_d_max - _d_min).days or 1
    _recency = ((_dates - _d_min).dt.days / _span)          # 0 → 1
    sample_weights = (0.55 + 0.45 * _recency).values        # 0.55 → 1.0

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso_logit", LogisticRegression(
            C=0.1,          # Fixed regularisation — no CV grid search needed weekly
            penalty="l1",
            solver="saga",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y, lasso_logit__sample_weight=sample_weights)

    cv_auc = cross_val_score(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        params={"lasso_logit__sample_weight": sample_weights},
    ).mean()

    n_nonzero = (pipeline.named_steps["lasso_logit"].coef_[0] != 0).sum()
    print(f"  Dynamic logit: AUC={cv_auc:.4f}, C=0.1, non-zero features={n_nonzero}/{len(feature_cols)}")

    return pipeline, feature_cols, float(cv_auc)


# ── Step 6: Multinomial finish-type model ─────────────────────────────────────

def _method_to_finish_class(method_str):
    """Map UFCStats method string → one of KO_TKO | Submission | Decision."""
    m = str(method_str).lower()
    if "ko" in m or "tko" in m:
        return "KO_TKO"
    if "sub" in m:
        return "Submission"
    if "dec" in m:
        return "Decision"
    return None  # No contest, DQ, etc. — exclude from training


def train_finish_model(df, career_styled, matchup_df):
    """
    Multinomial logistic regression: predict finish type (KO/TKO, Sub, Decision).

    This is FIGHT-LEVEL (one row per fight, not per fighter). We use the
    same feature set as the main model but target finish type, not who wins.

    Returns: {pipeline, feature_cols, cv_accuracy, class_names, base_rates}
    """
    # Build finish-type label per fight
    fight_methods = {}
    for fight_url, grp in df.groupby("fight_url"):
        winner = grp[grp["won"] == 1]
        if winner.empty:
            continue
        cls = _method_to_finish_class(winner.iloc[0].get("method", ""))
        if cls:
            fight_methods[fight_url] = cls

    if len(fight_methods) < 100:
        print("  [WARN] Insufficient finish data for multinomial model")
        return None

    # Add finish class to matchup_df
    mdf = matchup_df.copy()
    mdf["finish_class"] = mdf["fight_url"].map(fight_methods)
    mdf = mdf.dropna(subset=["finish_class"])
    mdf = encode_styles(mdf)

    # Features — use career tendency features + style + weight class
    # (avoid odds features here — finish type is more about fighter style than line)
    feat_cols = []
    tendency_diffs = [
        "diff_career_win_by_ko", "diff_career_win_by_sub", "diff_career_win_by_dec",
        "diff_career_avg_td_landed", "diff_career_avg_sub_att", "diff_career_avg_ctrl_seconds",
        "diff_career_avg_kd", "diff_career_avg_sig_str_landed", "diff_career_avg_sig_str_att",
        "diff_career_avg_distance_att", "diff_career_avg_clinch_att", "diff_career_avg_ground_att",
        "diff_career_avg_leg_att", "diff_career_avg_head_att",
        "diff_career_fights_before", "diff_career_wins",
        "diff_height_cms", "diff_reach_cms", "diff_age",
        "diff_avg_scorecard_margin",
        "diff_implied_prob_ko", "diff_implied_prob_sub", "diff_implied_prob_dec",
        "title_bout", "number_of_rounds",
    ]
    for col in tendency_diffs:
        if col in mdf.columns:
            feat_cols.append(col)

    style_individual = [c for c in mdf.columns
                        if (c.startswith("style_a_") or c.startswith("style_b_"))
                        and not "_lag" in c and not "_x_" in c]
    feat_cols += style_individual

    # Weight class dummies
    wc_dummies = pd.get_dummies(mdf["weight_class"], prefix="wc", drop_first=False)
    if "wc_Lightweight" in wc_dummies.columns:
        wc_dummies = wc_dummies.drop(columns=["wc_Lightweight"])
    wc_cols = list(wc_dummies.columns)
    mdf = pd.concat([mdf.reset_index(drop=True), wc_dummies.reset_index(drop=True)], axis=1)
    feat_cols += wc_cols

    feat_cols = list(dict.fromkeys(c for c in feat_cols if c in mdf.columns))

    for col in feat_cols:
        mdf[col] = mdf[col].fillna(0.0)

    mdf = mdf.dropna(subset=["finish_class"])

    X = mdf[feat_cols].values
    y = mdf["finish_class"].values

    # Base rates
    base_rates = {cls: float((y == cls).mean()) for cls in FINISH_CLASSES}
    print(f"  Finish base rates: " + ", ".join(f"{k}:{v:.1%}" for k, v in base_rates.items()))

    # Multinomial logistic with L2 (less sparsity needed for 3-class problem)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,          # Fixed C for finish model
            penalty="l2",
            solver="lbfgs",  # lbfgs handles multiclass natively in sklearn>=1.7
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)

    # Cross-validated accuracy
    cv_acc = cross_val_score(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    ).mean()

    # Class order from the fitted model
    classes = list(pipeline.named_steps["clf"].classes_)
    print(f"  Finish model: CV accuracy={cv_acc:.4f}, classes={classes}")

    return {
        "pipeline":     pipeline,
        "feature_cols": feat_cols,
        "cv_accuracy":  float(cv_acc),
        "class_names":  classes,
        "base_rates":   base_rates,
    }


# ── Step 7: Style-shift prediction model ──────────────────────────────────────

def train_style_shift_model(df, career_styled):
    per_fight_stats = df.copy()
    stat_cols = [c for c in CAREER_STAT_COLS if c in per_fight_stats.columns]

    per_fight_stats[[f"std_{c}" for c in stat_cols]] = np.nan
    for wc, grp_idx in per_fight_stats.groupby("weight_class").groups.items():
        for col in stat_cols:
            vals = per_fight_stats.loc[grp_idx, col]
            mean, std = vals.mean(), vals.std()
            if std and std > 0:
                per_fight_stats.loc[grp_idx, f"std_{col}"] = (vals - mean) / std

    def classify_infight(row):
        def z(f):
            val = row.get(f"std_{f}")
            return val if (val is not None and not (isinstance(val, float) and np.isnan(val))) else -999
        if z("sig_str_att") >= 0.2 and z("total_str_att") >= 0.2 and z("head_att") >= 0.2:
            return "Striker"
        if z("ctrl_seconds") >= -0.5 and z("ground_att") >= -0.5 and z("td_att") >= -0.5:
            return "Wrestler"
        if z("sub_att") >= -0.75 and z("ctrl_seconds") >= -0.75 and z("td_att") >= -0.75:
            return "BJJ"
        if z("clinch_att") >= -0.5 and z("body_att") >= -0.5 and z("leg_att") >= -0.5:
            return "Muay_Thai"
        if z("distance_att") >= -0.8 and z("sig_str_pct") >= -0.8 and z("head_att") >= -0.8:
            return "Sniper"
        return "Mixed"

    per_fight_stats["infight_style"] = per_fight_stats.apply(classify_infight, axis=1)

    cs     = career_styled[["fighter", "fight_url", "career_style"]].copy()
    merged = per_fight_stats.merge(cs, on=["fighter", "fight_url"], how="inner")
    merged["style_shifted"] = (merged["infight_style"] != merged["career_style"]).astype(int)

    print(f"  Style shift rate in training data: {merged['style_shifted'].mean():.2%}")

    feature_cols = [f"std_{c}" for c in stat_cols if f"std_{c}" in merged.columns]
    for s in STYLE_ORDER:
        merged[f"cs_{s}"] = (merged["career_style"] == s).astype(int)
        feature_cols.append(f"cs_{s}")

    model_data = merged.dropna(subset=feature_cols + ["style_shifted"])
    if len(model_data) < 50:
        print("  [WARN] Insufficient data for style-shift model")
        return None, None

    X = model_data[feature_cols].values
    y = model_data["style_shifted"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, penalty="l1", solver="saga",
            max_iter=2000, random_state=42,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(X, y)

    cv_auc = cross_val_score(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
    ).mean()

    print(f"  Style-shift model: AUC={cv_auc:.4f}")
    return pipeline, feature_cols


# ── Step 8: Style proportions lookup ──────────────────────────────────────────

def build_style_proportions(df, career_styled):
    cs = career_styled[["fighter", "fight_url", "career_style"]].copy()

    stat_cols = [c for c in CAREER_STAT_COLS if c in df.columns]
    pf        = df.copy()
    for wc, grp_idx in pf.groupby("weight_class").groups.items():
        for col in stat_cols:
            vals = pf.loc[grp_idx, col]
            mean, std = vals.mean(), vals.std()
            if std and std > 0:
                pf.loc[grp_idx, f"std_{col}"] = (vals - mean) / std

    def classify_infight(row):
        def z(f):
            v = row.get(f"std_{f}")
            return v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else -999
        if z("sig_str_att") >= 0.2 and z("total_str_att") >= 0.2 and z("head_att") >= 0.2:
            return "Striker"
        if z("ctrl_seconds") >= -0.5 and z("ground_att") >= -0.5 and z("td_att") >= -0.5:
            return "Wrestler"
        if z("sub_att") >= -0.75 and z("ctrl_seconds") >= -0.75 and z("td_att") >= -0.75:
            return "BJJ"
        if z("clinch_att") >= -0.5 and z("body_att") >= -0.5 and z("leg_att") >= -0.5:
            return "Muay_Thai"
        if z("distance_att") >= -0.8 and z("sig_str_pct") >= -0.8 and z("head_att") >= -0.8:
            return "Sniper"
        return "Mixed"

    pf["infight_style"] = pf.apply(classify_infight, axis=1)
    merged = pf.merge(cs, on=["fighter", "fight_url"], how="inner")
    opp_cs = cs.rename(columns={"fighter": "opponent", "career_style": "opp_career_style"})
    merged = merged.merge(opp_cs, on=["fight_url", "opponent"], how="left")

    props = {}
    for (cs_val, opp_cs_val), grp in merged.groupby(["career_style", "opp_career_style"]):
        counts = grp["infight_style"].value_counts(normalize=True)
        props[(cs_val, opp_cs_val)] = counts.to_dict()
    return props


# ── Step 9: Prop models (goes the distance / KO / Sub) ───────────────────────

def train_prop_models(df, career_styled, matchup_df):
    fight_meta = df[["fight_url", "fighter", "method", "won"]].copy()
    fight_meta["method_lower"] = fight_meta["method"].fillna("").str.lower()

    outcomes = {}
    for fight_url, grp in fight_meta.groupby("fight_url"):
        winner_row = grp[grp["won"] == 1]
        if winner_row.empty:
            continue
        method = winner_row.iloc[0]["method_lower"]
        outcomes[fight_url] = {
            "went_distance": 1 if "dec" in method else 0,
            "ko_finish":     1 if ("ko" in method or "tko" in method) else 0,
            "sub_finish":    1 if "sub" in method else 0,
        }

    outcome_df = pd.DataFrame.from_dict(outcomes, orient="index").reset_index()
    outcome_df.columns = ["fight_url", "went_distance", "ko_finish", "sub_finish"]
    mdf = matchup_df.merge(outcome_df, on="fight_url", how="inner")
    mdf = encode_styles(mdf)

    prop_diff_cols = [
        "diff_career_win_by_ko", "diff_career_win_by_sub", "diff_career_win_by_dec",
        "diff_career_avg_kd", "diff_career_avg_sig_str_landed",
        "diff_career_avg_td_landed", "diff_career_avg_sub_att", "diff_career_avg_ctrl_seconds",
        "diff_career_avg_distance_att", "diff_career_avg_clinch_att", "diff_career_avg_ground_att",
        "diff_career_fights_before",
        "diff_implied_prob_ko", "diff_implied_prob_sub", "diff_implied_prob_dec",
        "diff_avg_scorecard_margin",
        "title_bout", "number_of_rounds",
    ]

    wc_dummies = pd.get_dummies(mdf["weight_class"], prefix="wc", drop_first=False)
    if "wc_Lightweight" in wc_dummies.columns:
        wc_dummies = wc_dummies.drop(columns=["wc_Lightweight"])
    wc_cols = list(wc_dummies.columns)
    mdf = pd.concat([mdf.reset_index(drop=True), wc_dummies.reset_index(drop=True)], axis=1)

    style_individual = [c for c in mdf.columns
                        if (c.startswith("style_a_") or c.startswith("style_b_"))
                        and not "_lag" in c and not "_x_" in c]

    avail_diff       = [c for c in prop_diff_cols if c in mdf.columns]
    prop_feature_cols = list(dict.fromkeys(avail_diff + style_individual + wc_cols))

    for col in prop_feature_cols:
        if col in mdf.columns:
            mdf[col] = mdf[col].fillna(0.0)
    mdf = mdf.dropna(subset=prop_feature_cols)

    prop_models = {}
    for target, label in [("went_distance", "Goes the Distance"),
                           ("ko_finish",     "KO/TKO Finish"),
                           ("sub_finish",    "Submission Finish")]:
        if target not in mdf.columns:
            continue
        y = mdf[target].values
        if y.sum() < 20:
            print(f"  [WARN] Insufficient positive examples for prop model: {label}")
            continue

        X = mdf[prop_feature_cols].values
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.5, penalty="l2", solver="lbfgs",
                max_iter=2000, random_state=42,
                class_weight="balanced",
            )),
        ])
        pipeline.fit(X, y)

        cv_auc    = cross_val_score(
            pipeline, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc",
        ).mean()
        base_rate = y.mean()
        print(f"  Prop model [{label}]: AUC={cv_auc:.4f}, base_rate={base_rate:.1%}")

        prop_models[target] = {
            "pipeline":     pipeline,
            "feature_cols": prop_feature_cols,
            "cv_auc":       float(cv_auc),
            "base_rate":    float(base_rate),
            "label":        label,
        }

    return prop_models


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Model Training Pipeline")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    print("\n[1/8] Loading raw fight stats...")
    raw_df, _ = read_csv_from_github("data/raw/fight_stats.csv")
    if raw_df.empty:
        print("[ERROR] No fight_stats.csv found. Run scripts 01 or 02 first.")
        return
    df = load_raw(raw_df)
    print(f"  Loaded {len(df)} fighter-fight rows ({df['fight_url'].nunique()} fights)")

    # Report implied_prob coverage
    if "implied_prob" in df.columns:
        mkt = (df.get("implied_prob_source", "") == "market").sum()
        mod = (df.get("implied_prob_source", "") == "model").sum()
        print(f"  implied_prob: {mkt} market, {mod} model-filled, "
              f"{df['implied_prob'].isna().sum()} missing")

    print("\n[2/8] Computing rolling career stats (leak-free)...")
    career_df = compute_career_stats(df)
    print(f"  Career stats computed for {career_df['fighter'].nunique()} fighters")

    print("\n[3/8] Assigning fighting styles (locked)...")
    career_styled = assign_fighting_styles(df, career_df)
    style_dist = career_styled["career_style"].value_counts(normalize=True)
    print("  Style distribution:")
    for style, pct in style_dist.items():
        print(f"    {style}: {pct:.1%}")

    print("\n[4/8] Building matchup feature matrix...")
    matchup_df = build_matchup_features(df, career_styled)
    print(f"  {len(matchup_df)} matchup rows")

    print("\n[5/8] Training dynamic logit + LASSO model...")
    main_model, feature_cols, cv_auc = train_dynamic_logit(matchup_df)

    print("\n[6/8] Training multinomial finish-type model...")
    finish_model = train_finish_model(df, career_styled, matchup_df)

    print("\n[7/8] Training style-shift prediction model...")
    shift_model, shift_features = train_style_shift_model(df, career_styled)

    print("\n[8/8] Training prop models (distance / KO / submission)...")
    prop_models    = train_prop_models(df, career_styled, matchup_df)
    style_props    = build_style_proportions(df, career_styled)

    # Enrich career_styled with physical attributes for prediction script
    phys_rank_cols = [c for c in ["height_cms", "reach_cms", "age",
                                   "fighter_rank_wc", "fighter_rank_pfp"]
                      if c in df.columns]
    if phys_rank_cols:
        phys_df = df[["fighter", "fight_url"] + phys_rank_cols].drop_duplicates(
            subset=["fighter", "fight_url"])
        career_styled = career_styled.merge(phys_df, on=["fighter", "fight_url"],
                                            how="left", suffixes=("_cs", ""))
        career_styled = career_styled[[c for c in career_styled.columns if not c.endswith("_cs")]]

    model_bundle = {
        "main_model":            main_model,
        "feature_cols":          feature_cols,
        "cv_auc":                cv_auc,
        "finish_model":          finish_model,   # multinomial: KO_TKO | Submission | Decision
        "shift_model":           shift_model,
        "shift_features":        shift_features,
        "shift_threshold":       STYLE_SHIFT_THRESHOLD,
        "style_proportions":     style_props,
        "style_order":           STYLE_ORDER,
        "finish_classes":        FINISH_CLASSES,
        "prop_models":           prop_models,
        "career_stats_snapshot": career_styled[[
            "fighter", "fight_url", "event_date",
            "career_style", "career_fights_before",
            "career_wins", "career_losses",
        ]],
        "trained_at": datetime.now().isoformat(),
    }

    write_csv_to_github(
        career_styled,
        "data/processed/career_styled.csv",
        f"Career-styled dataset — {datetime.now().strftime('%Y-%m-%d')}",
    )
    upload_pickle_to_github(
        model_bundle,
        "data/model_bundle.pkl",
        f"Model retrain — {datetime.now().strftime('%Y-%m-%d')} — CV AUC {cv_auc:.4f}",
    )

    print(f"\n✓ Training complete.")
    print(f"  Win model CV AUC:       {cv_auc:.4f}")
    if finish_model:
        print(f"  Finish model CV Acc:    {finish_model['cv_accuracy']:.4f}")
    print("  Model bundle saved to GitHub.")


def run_test():
    """
    --test mode: loads real fight_stats.csv from GitHub, runs every pipeline
    stage (career stats, style assignment, matchup features, model training,
    pickle serialization) and reports pass/fail for each step.
    No data is written back to GitHub.
    """
    import sys, pickle, io, base64
    print("=== TEST MODE: 03_train_model ===\n")
    all_pass = True

    # [1] Load fight_stats.csv from GitHub
    print("[1] Loading fight_stats.csv from GitHub...")
    raw_df, _ = read_csv_from_github("data/raw/fight_stats.csv")
    if raw_df.empty:
        print("  ✗ fight_stats.csv not found — push data/raw/fight_stats.csv first")
        return
    print(f"  ✓ Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")

    # [2] load_raw()
    print("\n[2] load_raw() — type coercion and cleaning...")
    try:
        df = load_raw(raw_df)
        assert len(df) > 0, "All rows dropped after cleaning"
        assert pd.api.types.is_datetime64_any_dtype(df["event_date"]), "event_date not datetime"
        print(f"  ✓ {len(df)} rows after cleaning")
    except Exception as e:
        print(f"  ✗ {e}")
        all_pass = False
        return

    # [3] compute_career_stats() — leak check
    print("\n[3] compute_career_stats() — rolling stats + leak check...")
    try:
        career_df = compute_career_stats(df)
        assert len(career_df) == len(df), f"Row count mismatch: {len(career_df)} vs {len(df)}"
        first_fights = career_df[career_df["career_fights_before"] == 0]
        assert first_fights["career_avg_sig_str_landed"].isna().all(), \
            "CRITICAL: data leakage — first fight has non-NaN career averages"
        print(f"  ✓ {len(career_df)} rows, leak-check passed")
        print(f"  ✓ {career_df['fighter'].nunique()} unique fighters")
    except Exception as e:
        print(f"  ✗ {e}")
        all_pass = False
        return

    # [4] assign_fighting_styles()
    print("\n[4] assign_fighting_styles()...")
    try:
        career_styled = assign_fighting_styles(df, career_df)
        assert "career_style" in career_styled.columns
        dist = career_styled["career_style"].value_counts()
        print(f"  ✓ Style distribution: {dict(dist)}")
    except Exception as e:
        print(f"  ✗ {e}")
        all_pass = False
        return

    # [5] build_matchup_features()
    print("\n[5] build_matchup_features()...")
    try:
        matchup_df = build_matchup_features(df, career_styled)
        if len(matchup_df) == 0:
            print("  ✗ No matchups built — check fight_url pairing")
            all_pass = False
        else:
            diff_cols = [c for c in matchup_df.columns if c.startswith("diff_")]
            print(f"  ✓ {len(matchup_df)} matchups, {len(diff_cols)} diff features")
    except Exception as e:
        print(f"  ✗ {e}")
        all_pass = False
        return

    # [6] train_dynamic_logit() — skip if < 50 rows (expected with sample data)
    print("\n[6] train_dynamic_logit()...")
    main_model = feature_cols = cv_auc = None
    y_vals = matchup_df["fighter_a_won"].dropna().values if len(matchup_df) > 0 else []
    if len(matchup_df) < 10:
        print(f"  [SKIP] Only {len(matchup_df)} matchups — need ≥50 for real training (fine with sample data)")
    elif len(set(y_vals)) < 2:
        print(f"  [SKIP] Training data has only one class — need wins AND losses in fight_stats.csv")
    else:
        try:
            main_model, feature_cols, cv_auc = train_dynamic_logit(matchup_df)
            assert cv_auc > 0
            print(f"  ✓ CV AUC={cv_auc:.4f}")
            if cv_auc < 0.52:
                print(f"  ⚠ AUC near random — expected with small sample data")
        except Exception as e:
            print(f"  ✗ {e}")
            all_pass = False

    # [7] train_finish_model() — skip if insufficient data
    print("\n[7] train_finish_model()...")
    finish_model = None
    try:
        finish_model = train_finish_model(df, career_styled, matchup_df)
        if finish_model is None:
            print(f"  [SKIP] Insufficient data (need ≥100 fights) — expected with sample data")
        else:
            print(f"  ✓ CV accuracy={finish_model['cv_accuracy']:.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        all_pass = False

    # [8] train_style_shift_model()
    print("\n[8] train_style_shift_model()...")
    shift_model = shift_features = None
    try:
        shift_model, shift_features = train_style_shift_model(df, career_styled)
        if shift_model is None:
            print(f"  [SKIP] Insufficient data (need ≥50 rows) — expected with sample data")
        else:
            print(f"  ✓ Trained, {len(shift_features)} features")
    except Exception as e:
        print(f"  ✗ {e}")
        all_pass = False

    # [9] train_prop_models()
    print("\n[9] train_prop_models()...")
    try:
        prop_models = train_prop_models(df, career_styled, matchup_df)
        if not prop_models:
            print(f"  [SKIP] Insufficient data (need ≥20 positive examples per prop) — expected with sample data")
        else:
            for target, m in prop_models.items():
                print(f"  ✓ [{target}] AUC={m['cv_auc']:.4f}, base_rate={m['base_rate']:.1%}")
    except Exception as e:
        print(f"  ✗ {e}")
        all_pass = False

    # [10] model_bundle serialization
    print("\n[10] model_bundle pickle serialization...")
    try:
        bundle = {
            "main_model":        main_model,
            "feature_cols":      feature_cols,
            "cv_auc":            cv_auc,
            "finish_model":      finish_model,
            "shift_model":       shift_model,
            "shift_features":    shift_features,
            "shift_threshold":   STYLE_SHIFT_THRESHOLD,
            "style_proportions": {},
            "style_order":       STYLE_ORDER,
            "finish_classes":    FINISH_CLASSES,
            "prop_models":       {},
            "career_stats_snapshot": career_styled[["fighter","fight_url","event_date","career_style","career_fights_before","career_wins","career_losses"]],
            "trained_at":        datetime.now().isoformat(),
        }
        raw_bytes = pickle.dumps(bundle, protocol=4)
        restored  = pickle.loads(raw_bytes)
        assert "trained_at" in restored
        print(f"  ✓ Serializes OK ({len(raw_bytes):,} bytes)")
    except Exception as e:
        print(f"  ✗ {e}")
        all_pass = False

    # [11] career_styled.csv structure check
    print("\n[11] career_styled.csv output structure...")
    required_career_cols = ["fighter", "fight_url", "event_date", "career_style",
                            "career_fights_before", "career_wins", "career_losses"]
    missing = [c for c in required_career_cols if c not in career_styled.columns]
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        all_pass = False
    else:
        print(f"  ✓ All required career_styled columns present ({len(career_styled.columns)} total)")

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
