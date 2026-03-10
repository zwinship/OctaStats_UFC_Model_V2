#!/usr/bin/env python3
"""
v1_statistics.py
OctaStats — V1 Model Statistics Generator (Automated)

Reads v1_all_betting_results.csv from UFC_Model repo, computes the same
statistical analysis as V2 (t-test, calibration, monthly P&L, etc.),
and pushes v1_statistical_analysis.json back to UFC_Model.

Environment variables:
    ZWINSHIP_PAT  — GitHub personal access token
"""

import os
import io
import json
import base64
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")

GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
if not GITHUB_TOKEN:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
V1_REPO      = "UFC_Model"
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
}
BREAKEVEN = 0.96  # implied vig breakeven (same as V2)


# ── GitHub helpers ────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{V1_REPO}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame()
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(content))


def write_json_to_github(obj, repo_path, message):
    content = base64.b64encode(json.dumps(obj, indent=2).encode()).decode()
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{V1_REPO}/contents/{repo_path}"
    check   = requests.get(url, headers=GH_HEADERS, timeout=10)
    sha     = check.json().get("sha") if check.status_code == 200 else None
    payload = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload, timeout=30)
    ok   = resp.status_code in (200, 201)
    print(f"  {'✓' if ok else '✗'} GitHub: {repo_path} — {resp.status_code}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def american_to_decimal(odds):
    try:
        o = int(odds)
        return (100 / abs(o) + 1) if o < 0 else (o / 100 + 1)
    except (ValueError, TypeError):
        return None


def compute_rtp(row):
    """RTP multiplier per unit staked. Win: decimal odds. Loss: 0."""
    bet = int(row.get("bet_size", 0))
    if bet == 0:
        return None
    won = row.get("won", 0) == 1 or row.get("result", "") == "W"
    dec = american_to_decimal(row.get("Odds"))
    if dec is None:
        return None
    return float(dec) if won else 0.0


def safe(v):
    """JSON-safe conversion for numpy types."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("V1 Model — Statistics Generator")
    print("=" * 60)

    # 1. Load all results
    print("\n[1] Loading V1 all-time results…")
    df = read_csv_from_github("results/v1_all_betting_results.csv")
    if df.empty:
        print("  No V1 results yet. Exiting.")
        return
    print(f"  Loaded {len(df):,} rows")

    # Filter to bets only
    bets = df[pd.to_numeric(df.get("bet_size", pd.Series(dtype=float)), errors="coerce") > 0].copy()
    if bets.empty:
        print("  No bets recorded yet.")
        return
    print(f"  Bets: {len(bets)}")

    # Normalise won column
    if "won" not in bets.columns:
        bets["won"] = (bets.get("result", pd.Series(dtype=str)) == "W").astype(int)

    # 2. Compute RTP multiplier per bet
    bets["rtp"] = bets.apply(compute_rtp, axis=1)
    rtp_series  = bets["rtp"].dropna()

    if len(rtp_series) == 0:
        print("  No RTP values computed — check odds format.")
        return

    # 3. One-sided t-test: H0: mean(RTP) = BREAKEVEN
    t_stat, p_two = stats.ttest_1samp(rtp_series, BREAKEVEN)
    p_one         = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    conclusion    = "Reject H0" if p_one < 0.05 else "Fail to reject H0"

    # 4. Overall performance
    bets["net"] = bets.apply(
        lambda r: float(american_to_decimal(r.get("Odds")) - 1) * int(r.get("bet_size", 0))
                  if r.get("won", 0) == 1 or r.get("result") == "W"
                  else -int(r.get("bet_size", 0)),
        axis=1,
    )
    total_bets    = len(bets)
    total_wins    = int(bets["won"].sum())
    total_losses  = total_bets - total_wins
    units_staked  = float(bets["bet_size"].astype(float).sum())
    net_units     = float(bets["net"].sum())
    win_rate_pct  = round(total_wins / total_bets * 100, 2) if total_bets else 0
    rtp_pct       = round((1 + net_units / units_staked) * 100, 2) if units_staked else 0

    # 5. Basic stats on RTP series
    basic_stats = {
        "count":  int(len(rtp_series)),
        "mean":   safe(rtp_series.mean()),
        "median": safe(rtp_series.median()),
        "std":    safe(rtp_series.std()),
        "min":    safe(rtp_series.min()),
        "max":    safe(rtp_series.max()),
        "range":  safe(rtp_series.max() - rtp_series.min()),
        "iqr":    safe(rtp_series.quantile(0.75) - rtp_series.quantile(0.25)),
    }

    # 6. Monthly performance
    monthly = []
    if "event_date" in bets.columns:
        bets["month"] = pd.to_datetime(bets["event_date"], errors="coerce").dt.to_period("M").astype(str)
        for month, grp in bets.groupby("month"):
            monthly.append({
                "month":       month,
                "n_bets":      int(len(grp)),
                "wins":        int(grp["won"].sum()),
                "net_units":   safe(grp["net"].sum()),
                "units_staked":safe(grp["bet_size"].astype(float).sum()),
                "win_rate":    safe(grp["won"].mean()),
            })

    # 7. Calibration (predicted prob vs actual win rate)
    calibration = []
    if "predicted_prob" in bets.columns:
        bets["pred_bucket"] = (bets["predicted_prob"].astype(float) * 10).apply(int) / 10
        for bucket, grp in bets.groupby("pred_bucket"):
            if len(grp) >= 3:
                calibration.append({
                    "prob_bucket":      safe(bucket),
                    "n":                int(len(grp)),
                    "avg_predicted":    safe(grp["predicted_prob"].astype(float).mean()),
                    "actual_win_rate":  safe(grp["won"].mean()),
                })

    # 8. Bet size distribution
    bet_dist = {}
    for units, grp in bets.groupby("bet_size"):
        bet_dist[str(int(units))] = int(len(grp))

    # 9. Assemble and push
    output = {
        "analysis_date":    datetime.utcnow().isoformat(),
        "model":            "V1",
        "t_test": {
            "null_hypothesis_value": BREAKEVEN,
            "t_statistic":           safe(t_stat),
            "p_value_one_sided":     safe(p_one),
            "conclusion":            conclusion,
        },
        "overall_performance": {
            "total_bets":    total_bets,
            "total_wins":    total_wins,
            "total_losses":  total_losses,
            "win_rate_pct":  win_rate_pct,
            "units_staked":  safe(units_staked),
            "net_units":     safe(net_units),
            "rtp_pct":       rtp_pct,
        },
        "basic_statistics":   basic_stats,
        "monthly_performance": monthly,
        "calibration":         calibration,
        "bet_size_distribution": {
            "units_by_bucket": bet_dist,
        },
    }

    print("\n[2] Pushing v1_statistical_analysis.json…")
    write_json_to_github(
        output,
        "statistics/v1_statistical_analysis.json",
        "chore: update V1 statistical analysis",
    )

    print(f"\n✓ V1 statistics complete.")
    print(f"  Total bets:  {total_bets}")
    print(f"  Win rate:    {win_rate_pct}%")
    print(f"  Net P&L:     {net_units:+.2f}u")
    print(f"  RTP:         {rtp_pct}%")
    print(f"  t-stat:      {t_stat:.3f}, p={p_one:.4f} ({conclusion})")


if __name__ == "__main__":
    main()
