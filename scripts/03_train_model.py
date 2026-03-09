#!/usr/bin/env python3
"""
03_train_model.py
OctaStats V2 — Feature engineering + Dynamic Logit model training

Runs every Wednesday after the data update. Steps:
  1. Load raw fight_stats.csv
  2. Compute rolling career stats per fighter (no data leakage — only uses
     fights BEFORE each event date when computing that fight's features)
  3. Assign fighting styles (career-level, locked per historical fight)
  4. Build matchup-level feature matrix (diffs, style encodings)
  5. Train discrete-choice dynamic logit with LASSO regularization
     — state variables: fighter's last 3 in-fight styles (style momentum)
     — LASSO filters the covariate space
  6. Train style-shift prediction model (predicts whether a fighter will
     use a different in-fight style vs their career style — only flagged
     at high confidence threshold)
  7. Serialize models to data/ and push to GitHub

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
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import expit  # sigmoid

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT", "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")
REPO_OWNER   = "zwinship"
REPO_NAME    = "OctaStats_UFC_Model_V2"
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# Style classification thresholds (standardised z-scores within weight class)
# These mirror V1 thresholds — tuning happens via cross-validated LASSO downstream
STYLE_THRESHOLDS = {
    "Striker":   {"sig_str_att": -0.35, "total_str_att": -0.35, "head_att": -0.35, "kd": -0.35},
    "Wrestler":  {"ctrl_seconds": 0.0,   "ground_att": 0.0,      "td_att": 0.0},
    "BJJ":       {"sub_att": -0.45,      "ctrl_seconds": -0.45,  "td_att": -0.45},
    "Muay_Thai": {"clinch_att": -0.43,   "body_att": -0.43,      "leg_att": -0.43, "head_att": -0.43},
    "Sniper":    {"distance_att": -0.7,  "sig_str_pct": -0.7,    "head_att": -0.7,
                  "body_att": -0.7,      "leg_att": -0.7},
}

# Style-shift prediction: only display on website if model confidence > this
STYLE_SHIFT_THRESHOLD = 0.78

WEIGHT_CLASS_LIMITS = [
    (57.6,  "Flyweight"),
    (62.1,  "Bantamweight"),
    (66.7,  "Featherweight"),
    (71.2,  "Lightweight"),
    (78.0,  "Welterweight"),
    (84.8,  "Middleweight"),
    (93.9,  "Light Heavyweight"),
    (121.1, "Heavyweight"),
]


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        return pd.DataFrame(), None
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), data["sha"]


def upload_pickle_to_github(obj, repo_path, message, existing_sha=None):
    """Serialize obj with pickle and upload to GitHub."""
    buf     = pickle.dumps(obj)
    content = base64.b64encode(buf).decode()
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"

    # Check for existing SHA if not provided
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
    """Basic cleaning of the scraped fight stats."""
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Normalise weight class string
    df["weight_class"] = (
        df["weight_class"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )

    # Drop fights with no usable stats
    stat_cols = ["sig_str_landed", "sig_str_att", "td_landed", "td_att", "ctrl_seconds"]
    df = df.dropna(subset=stat_cols, how="all")

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
    """
    For each fighter-fight row, compute career average stats using only
    fights STRICTLY BEFORE that event_date (no leakage).

    Returns a DataFrame with columns: fighter, fight_url, career_avg_{stat}
    plus win/loss totals, win streaks, title bouts, etc.
    """
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
        running_stats         = {col: [] for col in CAREER_STAT_COLS}

        for i, row in group.iterrows():
            # --- Career stats AT THE TIME of this fight (before this fight) ---
            n_prev = len(running_stats["sig_str_landed"])

            career_avgs = {}
            for col in CAREER_STAT_COLS:
                vals = [v for v in running_stats[col] if v is not None and not (isinstance(v, float) and np.isnan(v))]
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
                **career_avgs,
            })

            # --- Update accumulators AFTER recording ---
            for col in CAREER_STAT_COLS:
                running_stats[col].append(row.get(col))

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
    """
    Compute a per-fight fighting style for each fighter using career averages
    up to (but not including) that fight, standardised within weight class.

    Key constraint: once computed, a historical fight's style label is LOCKED.
    The style for an upcoming fight uses the most recent career stats.

    Returns career_df with columns: career_style, inst_style_predicted
    """
    # Merge career stats back with weight class info
    meta = df[["fight_url", "fighter", "weight_class"]].drop_duplicates()
    merged = career_df.merge(meta, on=["fighter", "fight_url"], how="left")

    # Standardise career avg stats within weight class
    stat_cols = [c for c in merged.columns if c.startswith("career_avg_")]

    # Only standardise rows with enough prior fights (≥3)
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
        """Rule-based style classification on standardised career averages."""
        def z(field):
            col = f"std_career_avg_{field}"
            val = row.get(col)
            return val if (val is not None and not (isinstance(val, float) and np.isnan(val))) else -999

        # Priority order matters — check most distinctive styles first
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

        if (z("sub_att")       >= t["BJJ"]["sub_att"] and
            z("ctrl_seconds")  >= t["BJJ"]["ctrl_seconds"] and
            z("td_att")        >= t["BJJ"]["td_att"]):
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
    Join two fighter rows per fight into one matchup row.
    Compute difference features and encode style matchup.
    Returns a DataFrame where each row is one fight (from fighter A's perspective).
    """
    # We'll treat Red as fighter A (just need one row per fight)
    # Use the fact that each fight has exactly 2 rows, pair them by fight_url
    fight_pairs = []

    for fight_url, group in df.groupby("fight_url"):
        if len(group) != 2:
            continue
        a = group.iloc[0].copy()
        b = group.iloc[1].copy()

        # Ensure a is the winner when building training label
        # (doesn't matter for symmetry — both perspectives are in the data)
        ca = career_styled[career_styled["fight_url"] == fight_url]
        ca = ca.set_index("fighter")

        if a["fighter"] not in ca.index or b["fighter"] not in ca.index:
            continue

        row_a = ca.loc[a["fighter"]]
        row_b = ca.loc[b["fighter"]]

        diff_cols = [
            "career_avg_kd", "career_avg_sig_str_landed", "career_avg_sig_str_att",
            "career_avg_sig_str_pct", "career_avg_td_landed", "career_avg_td_att",
            "career_avg_td_pct", "career_avg_sub_att", "career_avg_ctrl_seconds",
            "career_avg_head_att", "career_avg_body_att", "career_avg_leg_att",
            "career_avg_distance_att", "career_avg_clinch_att", "career_avg_ground_att",
            "career_wins", "career_losses", "career_title_bouts",
            "current_win_streak", "current_lose_streak", "longest_win_streak",
            "career_win_by_ko", "career_win_by_sub",
        ]

        feat = {
            "fight_url":       fight_url,
            "event_date":      a["event_date"],
            "weight_class":    a["weight_class"],
            "fighter_a":       a["fighter"],
            "fighter_b":       b["fighter"],
            "fighter_a_won":   int(a["won"]),
            "style_a":         row_a.get("career_style", "Mixed"),
            "style_b":         row_b.get("career_style", "Mixed"),
        }

        for col in diff_cols:
            val_a = row_a.get(col, np.nan)
            val_b = row_b.get(col, np.nan)
            try:
                feat[f"diff_{col}"] = float(val_a) - float(val_b)
            except (TypeError, ValueError):
                feat[f"diff_{col}"] = np.nan

        fight_pairs.append(feat)

    return pd.DataFrame(fight_pairs)


# ── Step 5: Dynamic Logit with LASSO ──────────────────────────────────────────

STYLE_ORDER = ["BJJ", "Mixed", "Muay_Thai", "Sniper", "Striker", "Wrestler"]


def encode_styles(df):
    """One-hot encode style matchup (style_a × style_b interaction dummy)."""
    df = df.copy()
    for s in STYLE_ORDER:
        df[f"style_a_{s}"] = (df["style_a"] == s).astype(int)
        df[f"style_b_{s}"] = (df["style_b"] == s).astype(int)

    # Interaction terms: style_a × style_b captures matchup dynamics
    for sa in STYLE_ORDER:
        for sb in STYLE_ORDER:
            df[f"matchup_{sa}_vs_{sb}"] = (
                (df["style_a"] == sa) & (df["style_b"] == sb)
            ).astype(int)
    return df


def add_state_variables(matchup_df):
    """
    Dynamic logit state: for each fighter, append their last 1-3 career styles
    as lag variables. These capture "style momentum" — whether a fighter has
    been consistently fighting in a given style or recently shifting.

    This is the key dynamic component: the model can detect if a fighter's
    recent style history deviates from their career style, which predicts
    adaptation in the next fight.
    """
    df = matchup_df.sort_values("event_date").copy()

    for fighter_col in ["fighter_a", "fighter_b"]:
        prefix = "a" if fighter_col == "fighter_a" else "b"
        style_col = f"style_{prefix}"

        # Build per-fighter chronological style sequence
        style_history = {}  # fighter → [style_t-1, style_t-2, style_t-3]

        lag_records = {f"style_{prefix}_lag{k}": [] for k in range(1, 4)}

        for idx, row in df.iterrows():
            fighter = row[fighter_col]
            hist    = style_history.get(fighter, [])

            for k in range(1, 4):
                lag_records[f"style_{prefix}_lag{k}"].append(
                    hist[-(k)] if len(hist) >= k else "Unknown"
                )

            # Update history
            style_history[fighter] = hist + [row[style_col]]

        for col, vals in lag_records.items():
            df[col] = vals

        # One-hot encode lags
        for k in range(1, 4):
            lag_col = f"style_{prefix}_lag{k}"
            for s in STYLE_ORDER + ["Unknown"]:
                df[f"{lag_col}_{s}"] = (df[lag_col] == s).astype(int)

    return df


def train_dynamic_logit(matchup_df):
    """
    Train a LASSO-regularised logistic regression on matchup features.

    The model is a discrete choice model where:
      - Choice outcome: fighter_a wins (1) or loses (0)
      - Covariates: diff features + style matchup + lag state variables
      - LASSO selects the most predictive subset automatically
      - LassoCV chooses the regularisation strength via cross-validation

    Returns: trained Pipeline (scaler + logistic LASSO), feature list, cv_auc
    """
    df = encode_styles(matchup_df)
    df = add_state_variables(df)

    # ── Feature selection ─────────────────────────────────────────────────────
    diff_cols    = [c for c in df.columns if c.startswith("diff_")]
    style_cols   = [c for c in df.columns if c.startswith("style_") or c.startswith("matchup_")]
    lag_hot_cols = [c for c in df.columns if "_lag" in c and any(s in c for s in STYLE_ORDER + ["Unknown"])]

    feature_cols = diff_cols + style_cols + lag_hot_cols

    # Drop rows with NaN in any feature or target
    model_df = df.dropna(subset=feature_cols + ["fighter_a_won"])
    # Filter to minimum fights (avoid noise from debut fights)
    model_df = model_df[model_df.apply(
        lambda r: _min_fights_filter(r, matchup_df), axis=1
    )]

    if len(model_df) < 50:
        print(f"  [WARN] Only {len(model_df)} rows after filtering — model may be unreliable")

    X = model_df[feature_cols].values
    y = model_df["fighter_a_won"].values

    # ── LASSO logistic regression with cross-validated C ─────────────────────
    # C = 1/lambda; LassoCV equivalent for logistic is LogisticRegressionCV with l1
    from sklearn.linear_model import LogisticRegressionCV

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso_logit", LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 20),   # search over 20 regularisation strengths
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            penalty="l1",
            solver="saga",
            max_iter=2000,
            scoring="roc_auc",
            refit=True,
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)

    # Cross-validated AUC
    cv_auc = cross_val_score(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
    ).mean()

    best_C   = pipeline.named_steps["lasso_logit"].C_[0]
    n_nonzero = (pipeline.named_steps["lasso_logit"].coef_[0] != 0).sum()

    print(f"  Dynamic logit trained: AUC={cv_auc:.4f}, best_C={best_C:.4f}, "
          f"non-zero features={n_nonzero}/{len(feature_cols)}")

    return pipeline, feature_cols, float(cv_auc)


def _min_fights_filter(row, matchup_df):
    """Only include fight rows where both fighters have ≥3 prior fights."""
    # Since career_fights_before isn't in matchup_df directly, proxy by
    # checking that diff cols aren't all NaN (means both had stats)
    diff_cols = [c for c in row.index if c.startswith("diff_career_avg")]
    n_valid   = sum(1 for c in diff_cols if not (isinstance(row[c], float) and np.isnan(row[c])))
    return n_valid >= (len(diff_cols) // 2)


# ── Step 6: Style-shift prediction model ──────────────────────────────────────

def train_style_shift_model(df, career_styled):
    """
    Predict whether a fighter's in-fight style will differ from their career style.

    This is a binary classifier (shifted / didn't shift). Only surfaced on the
    website when predicted probability exceeds STYLE_SHIFT_THRESHOLD (0.78).

    Features: career style, opponent career style, career stat diffs,
              recent style lag variables (already computed upstream).

    Returns trained Pipeline or None if insufficient data.
    """
    # We need actual in-fight style labels — these come from V1's inst_style
    # We approximate in-fight style using the same rule-based classifier
    # applied to per-fight stats (not career averages) — this is the "actual"
    # style used in that fight.

    per_fight_stats = df.copy()
    # Standardise per-fight stats within weight class
    stat_cols = [c for c in CAREER_STAT_COLS if c in per_fight_stats.columns]

    per_fight_stats[["std_" + c for c in stat_cols]] = np.nan
    for wc, grp_idx in per_fight_stats.groupby("weight_class").groups.items():
        for col in stat_cols:
            vals = per_fight_stats.loc[grp_idx, col]
            mean, std = vals.mean(), vals.std()
            if std and std > 0:
                per_fight_stats.loc[grp_idx, "std_" + col] = (vals - mean) / std

    # In-fight style thresholds (slightly different from career thresholds)
    def classify_infight(row):
        def z(field):
            val = row.get("std_" + field)
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

    # Merge career style
    cs = career_styled[["fighter", "fight_url", "career_style"]].copy()
    merged = per_fight_stats.merge(cs, on=["fighter", "fight_url"], how="inner")
    merged["style_shifted"] = (merged["infight_style"] != merged["career_style"]).astype(int)

    shift_rate = merged["style_shifted"].mean()
    print(f"  Style shift rate in training data: {shift_rate:.2%}")

    feature_cols = ["std_" + c for c in stat_cols if "std_" + c in merged.columns]
    # Add career style one-hot
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
            class_weight="balanced",  # fights where style shifts are rare
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


# ── Step 7: Style proportions lookup table (V1 compatibility) ─────────────────

def build_style_proportions(df, career_styled):
    """
    Build the (career_style, opponent_career_style) → {infight_style: pct} lookup.
    Used as a fast reference for in-fight style prediction where the logit
    model has insufficient data for a weight class.
    """
    cs = career_styled[["fighter", "fight_url", "career_style"]].copy()

    # Per-fight style (same as in style-shift model)
    stat_cols = [c for c in CAREER_STAT_COLS if c in df.columns]
    pf = df.copy()
    for wc, grp_idx in pf.groupby("weight_class").groups.items():
        for col in stat_cols:
            vals = pf.loc[grp_idx, col]
            mean, std = vals.mean(), vals.std()
            if std and std > 0:
                pf.loc[grp_idx, "std_" + col] = (vals - mean) / std

    def classify_infight(row):
        def z(f):
            v = row.get("std_" + f)
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

    # Get opponent career style
    opp_cs = cs.rename(columns={"fighter": "opponent", "career_style": "opp_career_style"})
    merged = merged.merge(opp_cs, on=["fight_url", "opponent"], how="left")

    props = {}
    for (cs_val, opp_cs_val), grp in merged.groupby(["career_style", "opp_career_style"]):
        counts = grp["infight_style"].value_counts(normalize=True)
        props[(cs_val, opp_cs_val)] = counts.to_dict()

    return props


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Model Training Pipeline")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Load raw data
    print("\n[1/6] Loading raw fight stats...")
    raw_df, _ = read_csv_from_github("data/raw/fight_stats.csv")
    if raw_df.empty:
        print("[ERROR] No fight_stats.csv found. Run scripts 01 or 02 first.")
        return
    df = load_raw(raw_df)
    print(f"  Loaded {len(df)} fighter-fight rows ({df['fight_url'].nunique()} fights)")

    # 2. Compute rolling career stats (leak-free)
    print("\n[2/6] Computing rolling career stats (leak-free)...")
    career_df = compute_career_stats(df)
    print(f"  Career stats computed for {career_df['fighter'].nunique()} fighters")

    # 3. Assign fighting styles (LOCKED for historical fights)
    print("\n[3/6] Assigning fighting styles (career-level, locked)...")
    career_styled = assign_fighting_styles(df, career_df)
    style_dist = career_styled["career_style"].value_counts(normalize=True)
    print("  Style distribution:")
    for style, pct in style_dist.items():
        print(f"    {style}: {pct:.1%}")

    # 4. Build matchup feature matrix
    print("\n[4/6] Building matchup feature matrix...")
    matchup_df = build_matchup_features(df, career_styled)
    print(f"  {len(matchup_df)} matchup rows")

    # 5. Train dynamic logit with LASSO
    print("\n[5/6] Training dynamic logit + LASSO model...")
    main_model, feature_cols, cv_auc = train_dynamic_logit(matchup_df)

    # 6. Train style-shift model
    print("\n[6/6] Training style-shift prediction model...")
    shift_model, shift_features = train_style_shift_model(df, career_styled)

    # Build style proportions lookup (fallback for prediction)
    style_props = build_style_proportions(df, career_styled)

    # Package everything into a single model bundle
    model_bundle = {
        "main_model":           main_model,
        "feature_cols":         feature_cols,
        "cv_auc":               cv_auc,
        "shift_model":          shift_model,
        "shift_features":       shift_features,
        "shift_threshold":      STYLE_SHIFT_THRESHOLD,
        "style_proportions":    style_props,
        "style_order":          STYLE_ORDER,
        "career_stats_snapshot": career_styled[[
            "fighter", "fight_url", "event_date",
            "career_style", "career_fights_before",
            "career_wins", "career_losses",
        ]],
        "trained_at": datetime.now().isoformat(),
    }

    # Save career styled dataset for predictions script
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

    print(f"\n✓ Training complete. Cross-validated AUC: {cv_auc:.4f}")
    print("  Model bundle saved to GitHub.")


if __name__ == "__main__":
    main()
