#!/usr/bin/env python3
"""
v1_train_model.py
OctaStats — V1 Model Retrainer (Automated)

Reads the shared fight_stats.csv (scraped by V2 pipeline), engineers the same
features the original V1 model used, and trains separate LinearRegression models
per (weight_class, fighter_style) group. Pushes model bundle to zwinship/UFC_Model.

V1 modeling logic is intentionally preserved unchanged:
  - 6 fighting styles: Striker, Wrestler, BJJ, Muay_Thai, Sniper, Mixed
  - LinearRegression per (weight_class, style) group — predicts win probability
  - Discrete bet sizing: 0 / 1 / 2 / 3 units based on edge thresholds
  - Style assigned from career averages within weight class (z-scored)

Environment variables:
    ZWINSHIP_PAT  — GitHub personal access token
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
import sys as _sys
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
if not GITHUB_TOKEN and "--test" not in _sys.argv:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
V2_REPO      = "OctaStats_UFC_Model_V2"   # source of fight_stats.csv
V1_REPO      = "UFC_Model"                # destination for V1 model bundle
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
}

# Style thresholds — same z-score cutoffs as V1
STYLE_Z = 0.5

# V1 feature columns derived from fight_stats.csv
STAT_COLS = [
    "kd", "sig_str_landed", "sig_str_att", "total_str_landed",
    "total_str_att", "td_landed", "td_att", "sub_att",
    "ctrl_seconds", "head_landed", "body_landed", "leg_landed",
    "distance_landed", "clinch_landed", "ground_landed",
]


# ── GitHub helpers ────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path, repo=V2_REPO):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{repo}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame(), None
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), data["sha"]


def write_pickle_to_github(obj, repo_path, message, repo=V1_REPO):
    raw     = pickle.dumps(obj, protocol=4)
    content = base64.b64encode(raw).decode()
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{repo}/contents/{repo_path}"
    check   = requests.get(url, headers=GH_HEADERS, timeout=10)
    sha     = check.json().get("sha") if check.status_code == 200 else None
    payload = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload, timeout=30)
    ok   = resp.status_code in (200, 201)
    print(f"  {'✓' if ok else '✗'} GitHub: {repo_path} ({repo}) — {resp.status_code}")
    return ok


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


# ── Feature engineering ───────────────────────────────────────────────────────

def safe_col(df, col):
    """Return column as numeric or zeros if missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index)


def compute_career_averages(df):
    """
    Compute per-fighter career averages up to (but not including) each fight.
    Returns a DataFrame indexed the same as df with career avg columns prefixed 'avg_'.
    Leak-free: only uses fights strictly before the current fight date.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    career_rows = []
    for _, row in df.iterrows():
        fighter  = row["fighter"]
        fight_dt = row["date"]
        prior    = df[(df["fighter"] == fighter) & (df["date"] < fight_dt)]
        rec      = {"fighter": fighter, "date": fight_dt, "event": row.get("event", "")}
        for col in STAT_COLS:
            if col in df.columns:
                rec[f"avg_{col}"] = prior[col].mean() if len(prior) else np.nan
        career_rows.append(rec)

    return pd.DataFrame(career_rows)


def assign_style(df_career, weight_class):
    """
    Z-score career averages within weight class, then assign style by the
    same rule-based logic as V1. Returns series of style labels.
    """
    wc_mask = df_career["weight_class"] == weight_class if "weight_class" in df_career.columns else pd.Series(True, index=df_career.index)
    subset  = df_career[wc_mask].copy()

    avg_cols = [c for c in subset.columns if c.startswith("avg_")]
    if not avg_cols:
        return pd.Series("Mixed", index=subset.index)

    z = subset[avg_cols].copy()
    for col in avg_cols:
        mu, sigma = z[col].mean(), z[col].std()
        z[col] = (z[col] - mu) / (sigma + 1e-9)

    def label_row(r):
        kd       = r.get("avg_kd", 0)
        sig      = r.get("avg_sig_str_landed", 0)
        td       = r.get("avg_td_landed", 0)
        ctrl     = r.get("avg_ctrl_sec", 0)
        sub      = r.get("avg_sub_att", 0)
        clinch   = r.get("avg_clinch_landed", 0)
        body     = r.get("avg_body_landed", 0)
        leg      = r.get("avg_leg_landed", 0)
        distance = r.get("avg_distance_landed", 0)
        head     = r.get("avg_head_landed", 0)

        # Same V1 rules (z-scored, threshold = STYLE_Z)
        if kd > STYLE_Z and sig > STYLE_Z:
            return "Striker"
        if td > STYLE_Z and ctrl > STYLE_Z and sub < STYLE_Z:
            return "Wrestler"
        if sub > STYLE_Z and ctrl > STYLE_Z and td > STYLE_Z * 0.5:
            return "BJJ"
        if clinch > STYLE_Z and (body > STYLE_Z or leg > STYLE_Z or head > STYLE_Z):
            return "Muay_Thai"
        if distance > STYLE_Z and sig > STYLE_Z and td < STYLE_Z:
            return "Sniper"
        return "Mixed"

    z["weight_class"] = subset["weight_class"].values if "weight_class" in subset.columns else weight_class
    styles = z.apply(label_row, axis=1)
    result = pd.Series(index=df_career.index, dtype=str)
    result[wc_mask] = styles.values
    return result


# ── Build matchup features (V1 style) ────────────────────────────────────────

def build_matchup_features(df):
    """
    Each row of df has one fighter's stats. Match fighters into bouts,
    compute differential features, return matchup-level DataFrame.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Group by (event, bout_id or sequential pair)
    matchups = []
    for (event, date), grp in df.groupby(["event", "date"]):
        fighters = grp.reset_index(drop=True)
        if len(fighters) < 2:
            continue
        for i in range(0, len(fighters) - 1, 2):
            f1 = fighters.iloc[i]
            f2 = fighters.iloc[i + 1]
            row = {
                "event":        event,
                "date":         date,
                "fighter1":     f1["fighter"],
                "fighter2":     f2["fighter"],
                "style1":       f1.get("style", "Mixed"),
                "style2":       f2.get("style", "Mixed"),
                "weight_class": f1.get("weight_class", "Unknown"),
                "winner":       f1["fighter"] if f1.get("result", "") == "W" else f2["fighter"],
                "f1_won":       1 if f1.get("result", "") == "W" else 0,
            }
            for col in STAT_COLS:
                ac = f"avg_{col}"
                row[f"diff_{col}"] = f1.get(ac, 0) - f2.get(ac, 0)
            matchups.append(row)

    return pd.DataFrame(matchups)


# ── Train V1 models ───────────────────────────────────────────────────────────

def train_v1_models(matchup_df):
    """
    Train one LinearRegression per (weight_class, style1) group, predicting f1_won.
    Returns dict: {(weight_class, style): fitted LinearRegression}.
    """
    feature_cols = [c for c in matchup_df.columns if c.startswith("diff_")]
    models       = {}
    scalers      = {}

    for (wc, style), grp in matchup_df.groupby(["weight_class", "style1"]):
        grp = grp.dropna(subset=feature_cols + ["f1_won"])
        if len(grp) < 10:
            continue
        X = grp[feature_cols].values
        y = grp["f1_won"].values
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        model  = LinearRegression()
        model.fit(X_sc, y)
        models[(wc, style)]  = model
        scalers[(wc, style)] = scaler
        print(f"  Trained ({wc}, {style}): n={len(grp)}, "
              f"coef_range=[{(X_sc @ model.coef_).min():.3f}, {(X_sc @ model.coef_).max():.3f}]")

    return models, scalers, feature_cols


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("V1 Model Retrainer")
    print("=" * 60)

    # 1. Load fight_stats.csv from V2 repo
    print("\n[1] Loading fight_stats.csv from V2 repo…")
    fights, _ = read_csv_from_github("data/raw/fight_stats.csv", repo=V2_REPO)
    if fights.empty:
        print("  [ERROR] No fight data found. Run historical scrape first.")
        return
    print(f"  Loaded {len(fights):,} fight rows")

    # fight_stats.csv uses event_date/event_name/won — alias to what V1 expects
    if "date" not in fights.columns and "event_date" in fights.columns:
        fights["date"] = pd.to_datetime(fights["event_date"], errors="coerce")
    if "event" not in fights.columns and "event_name" in fights.columns:
        fights["event"] = fights["event_name"]
    if "result" not in fights.columns and "won" in fights.columns:
        fights["result"] = fights["won"].map({1: "W", 0: "L"})

    # 2. Compute per-fighter career averages (leak-free)
    print("\n[2] Computing career averages (leak-free)…")
    career = compute_career_averages(fights)
    # Re-attach weight_class and result
    meta_cols = ["fighter", "date", "event", "weight_class", "result"]
    for col in meta_cols:
        if col in fights.columns and col not in career.columns:
            career = career.merge(fights[["fighter", "date", col]].drop_duplicates(),
                                  on=["fighter", "date"], how="left")

    # 3. Assign V1 styles within weight class
    print("\n[3] Assigning V1 fighting styles…")
    career["style"] = "Mixed"
    if "weight_class" in career.columns:
        for wc in career["weight_class"].dropna().unique():
            mask = career["weight_class"] == wc
            career.loc[mask, "style"] = assign_style(career[mask], wc).values
    else:
        career["style"] = assign_style(career, "all").values

    style_counts = career["style"].value_counts()
    print(f"  Style distribution:\n{style_counts.to_string()}")

    # Push career stats to V1 repo for reference
    write_csv_to_github(career, "data/career_styled_v1.csv",
                        "chore: update V1 career stats", repo=V1_REPO)

    # 4. Build matchup features
    print("\n[4] Building matchup features…")
    matchups = build_matchup_features(career)
    print(f"  Built {len(matchups):,} matchups")

    if matchups.empty:
        print("  [ERROR] No matchups generated. Check data structure.")
        return

    # 5. Train V1 models
    print("\n[5] Training V1 LinearRegression models…")
    models, scalers, feature_cols = train_v1_models(matchups)
    print(f"  Trained {len(models)} (weight_class, style) group models")

    # 6. Build style proportions lookup for prediction fallback
    print("\n[6] Building style proportions lookup…")
    style_props = {}
    for (wc, s1), grp in matchups.groupby(["weight_class", "style1"]):
        for s2, sgrp in grp.groupby("style2"):
            key         = f"{wc}|{s1}|{s2}"
            wr          = sgrp["f1_won"].mean()
            style_props[key] = round(wr, 4)

    # 7. Bundle and push
    print("\n[7] Pushing V1 model bundle to UFC_Model repo…")
    bundle = {
        "models":        models,
        "scalers":       scalers,
        "feature_cols":  feature_cols,
        "style_props":   style_props,
        "trained_at":    datetime.utcnow().isoformat(),
        "n_fights":      len(fights),
        "n_matchups":    len(matchups),
        "style_counts":  style_counts.to_dict(),
        "version":       "v1_automated",
    }
    write_pickle_to_github(bundle, "data/v1_model_bundle.pkl",
                           "chore: retrain V1 model with latest data")

    print("\n✓ V1 model training complete.")
    print(f"  Models trained: {len(models)}")
    print(f"  Fights used:    {len(fights):,}")
    print(f"  Matchups used:  {len(matchups):,}")


def run_test():
    """
    --test mode: loads real fight_stats.csv from GitHub, applies column aliases,
    runs every V1 pipeline stage, and validates the model bundle structure.
    No data is written back to GitHub.
    """
    print("=== TEST MODE: v1_train_model ===\n")
    all_pass = True

    # [1] Load fight_stats.csv
    print("[1] Loading fight_stats.csv from GitHub...")
    fights, _ = read_csv_from_github("data/raw/fight_stats.csv", repo=V2_REPO)
    if fights.empty:
        print("  ✗ fight_stats.csv not found")
        return
    print(f"  ✓ Loaded {len(fights)} rows")

    # [2] Column aliases
    print("\n[2] Applying column aliases (event_date→date, event_name→event, won→result)...")
    try:
        if "date" not in fights.columns and "event_date" in fights.columns:
            fights["date"] = pd.to_datetime(fights["event_date"], errors="coerce")
        if "event" not in fights.columns and "event_name" in fights.columns:
            fights["event"] = fights["event_name"]
        if "result" not in fights.columns and "won" in fights.columns:
            fights["result"] = fights["won"].map({1: "W", 0: "L"})
        assert "date"   in fights.columns
        assert "event"  in fights.columns
        assert "result" in fights.columns
        print("  ✓ Aliases applied")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False; return

    # [3] compute_career_averages()
    print("\n[3] compute_career_averages()...")
    try:
        career = compute_career_averages(fights)
        assert len(career) == len(fights), f"Row count mismatch: {len(career)} vs {len(fights)}"
        avg_cols = [c for c in career.columns if c.startswith("avg_")]
        print(f"  ✓ {len(career)} rows, {len(avg_cols)} avg columns")
        # Leak check: first fight for each fighter should have all-NaN avgs
        first_fights = career[career.groupby("fighter")["date"].transform("min") == career["date"]]
        if avg_cols:
            leaky = first_fights[avg_cols[0]].notna().sum()
            if leaky > 0:
                print(f"  ✗ LEAK: {leaky} first-fight rows have non-NaN {avg_cols[0]}")
                all_pass = False
            else:
                print(f"  ✓ Leak check passed")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False; return

    # [4] assign_style()
    print("\n[4] assign_style()...")
    try:
        career["style"] = "Mixed"
        if "weight_class" in fights.columns:
            career = career.merge(
                fights[["fighter","date","weight_class"]].drop_duplicates(),
                on=["fighter","date"], how="left"
            )
            for wc in career["weight_class"].dropna().unique():
                mask = career["weight_class"] == wc
                career.loc[mask, "style"] = assign_style(career[mask], wc).values
        dist = career["style"].value_counts()
        print(f"  ✓ Style distribution: {dict(dist)}")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False; return

    # [5] build_matchup_features()
    print("\n[5] build_matchup_features()...")
    try:
        # Re-attach result column needed by build_matchup_features
        if "result" not in career.columns:
            career = career.merge(
                fights[["fighter","date","result"]].drop_duplicates(),
                on=["fighter","date"], how="left"
            )
        matchups = build_matchup_features(career)
        if matchups.empty:
            print(f"  ✗ No matchups built — check column aliases and fight pairing")
            all_pass = False
        else:
            diff_cols = [c for c in matchups.columns if c.startswith("diff_")]
            print(f"  ✓ {len(matchups)} matchups, {len(diff_cols)} diff features")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False; return

    # [6] train_v1_models()
    print("\n[6] train_v1_models()...")
    try:
        models, scalers, feature_cols = train_v1_models(matchups)
        if not models:
            print(f"  [SKIP] No groups had ≥10 matchups — expected with sample data")
        else:
            print(f"  ✓ Trained {len(models)} (weight_class, style) group models")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [7] Bundle serialization
    print("\n[7] Model bundle pickle serialization...")
    try:
        import pickle
        bundle = {
            "models":       models if 'models' in dir() else {},
            "scalers":      scalers if 'scalers' in dir() else {},
            "feature_cols": feature_cols if 'feature_cols' in dir() else [],
            "style_props":  {},
            "trained_at":   datetime.now().isoformat(),
        }
        raw = pickle.dumps(bundle, protocol=4)
        assert pickle.loads(raw)["trained_at"]
        print(f"  ✓ Serializes OK ({len(raw):,} bytes)")
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
