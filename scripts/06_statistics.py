#!/usr/bin/env python3
"""
06_statistics.py
OctaStats V2 — Statistical analysis of model performance

Runs Sunday after results are recorded. Computes:
  - Basic return-to-player (RTP) statistics
  - One-sample t-test: is model profit significantly > 0.96 (break-even after vig)?
  - Monthly performance trends
  - Style matchup accuracy (how often did style predictions match in-fight style?)
  - Model calibration (are predicted probs aligned with actual win rates?)
  - Writes JSON to statistics/v2_statistical_analysis.json for website consumption

Environment variables:
    ZWINSHIP_PAT  — GitHub personal access token for zwinship account
"""

import os
import io
import base64
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from scipy import stats as scipy_stats

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

# Break-even threshold: account for typical bookmaker margin (~4%)
BREAKEVEN_THRESHOLD = 0.96


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def read_csv(repo_path):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
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


def write_json(obj, repo_path, message):
    content = base64.b64encode(json.dumps(obj, indent=2, default=str).encode()).decode()
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    check   = requests.get(url, headers=GH_HEADERS)
    sha     = check.json().get("sha") if check.status_code == 200 else None
    payload = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    ok   = resp.status_code in (200, 201)
    print(f"  {'✓' if ok else '✗'} GitHub: {repo_path}")


# ── Analysis functions ────────────────────────────────────────────────────────

def basic_stats(df):
    """Compute basic statistics on profit_multiplier (RTP metric)."""
    profits = df["profit_multiplier"].dropna()
    if len(profits) == 0:
        return {}

    return {
        "count":    int(len(profits)),
        "mean":     float(profits.mean()),
        "median":   float(profits.median()),
        "std":      float(profits.std()),
        "min":      float(profits.min()),
        "max":      float(profits.max()),
        "range":    float(profits.max() - profits.min()),
        "iqr":      float(profits.quantile(0.75) - profits.quantile(0.25)),
    }


def rtp_ttest(df):
    """
    One-sample t-test: H0: mean profit_multiplier = 0.96, H1: > 0.96.
    We also compute against 1.0 (strict breakeven) for reference.
    """
    profits = df["profit_multiplier"].dropna()
    if len(profits) < 5:
        return {"error": "Insufficient data for t-test"}

    t_stat,  p_two  = scipy_stats.ttest_1samp(profits, BREAKEVEN_THRESHOLD)
    p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)

    t_stat2, p_two2 = scipy_stats.ttest_1samp(profits, 1.0)
    p_one2 = p_two2 / 2 if t_stat2 > 0 else 1 - (p_two2 / 2)

    return {
        "null_hypothesis_value": BREAKEVEN_THRESHOLD,
        "t_statistic":           float(t_stat),
        "p_value_one_sided":     float(p_one),
        "conclusion":            "Reject H0" if p_one < 0.05 else "Fail to reject H0",
        "vs_strict_breakeven": {
            "null_hypothesis_value": 1.0,
            "t_statistic":           float(t_stat2),
            "p_value_one_sided":     float(p_one2),
            "conclusion":            "Reject H0" if p_one2 < 0.05 else "Fail to reject H0",
        },
    }


def monthly_performance(df):
    """Break down win rate, units staked, and net P&L by calendar month."""
    df = df.copy()
    df["result_date"] = pd.to_datetime(df["result_date"], errors="coerce")
    df = df.dropna(subset=["result_date"])
    df["month"] = df["result_date"].dt.to_period("M").astype(str)

    bet_df = df[df["bet_size"] > 0]
    if bet_df.empty:
        return []

    monthly = (
        bet_df
        .groupby("month")
        .agg(
            bets=("name", "count"),
            wins=("won", "sum"),
            units_staked=("bet_size", "sum"),
            units_returned=("total_return", "sum"),
        )
        .reset_index()
    )
    monthly["win_rate"]  = (monthly["wins"] / monthly["bets"] * 100).round(2)
    monthly["net_units"] = (monthly["units_returned"] - monthly["units_staked"]).round(3)
    monthly["rtp"]       = (monthly["units_returned"] / monthly["units_staked"] * 100).round(2)

    return monthly.to_dict("records")


def calibration_analysis(df):
    """
    Model calibration: group bets by predicted probability bucket (decile)
    and compute actual win rate per bucket. Well-calibrated models have
    actual win rates close to the predicted probabilities.
    """
    df = df.copy()
    if "predicted_probability" not in df.columns or len(df) < 10:
        return []

    df["prob_bucket"] = pd.cut(
        df["predicted_probability"],
        bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        labels=["0-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"],
    )

    cal = (
        df.groupby("prob_bucket", observed=True)
        .agg(
            count=("won", "count"),
            actual_wins=("won", "sum"),
            avg_predicted=("predicted_probability", "mean"),
        )
        .reset_index()
    )
    cal["actual_win_rate"] = (cal["actual_wins"] / cal["count"]).round(4)
    cal["avg_predicted"]   = cal["avg_predicted"].round(4)
    cal["calibration_err"] = (cal["avg_predicted"] - cal["actual_win_rate"]).abs().round(4)

    return cal.to_dict("records")


def style_accuracy(df):
    """
    How often did the predicted_infight_style actually occur?
    (Requires matching against post-fight data, approximated here.)
    """
    if "predicted_infight_style" not in df.columns or "career_style" not in df.columns:
        return {}

    n_shift_predicted = df["style_shift_predicted"].sum() if "style_shift_predicted" in df.columns else 0

    return {
        "style_distribution": df["career_style"].value_counts().to_dict(),
        "predicted_shift_count": int(n_shift_predicted),
    }


def bet_size_analysis(df):
    """Distribution of bet sizes placed."""
    bet_df = df[df["bet_size"] > 0].copy() if "bet_size" in df.columns else pd.DataFrame()
    if bet_df.empty:
        return {}

    return {
        "mean_units":   round(float(bet_df["bet_size"].mean()), 3),
        "max_units":    round(float(bet_df["bet_size"].max()), 3),
        "min_units":    round(float(bet_df["bet_size"].min()), 3),
        "units_by_bucket": {
            "0-1":   int((bet_df["bet_size"] < 1).sum()),
            "1-2":   int(((bet_df["bet_size"] >= 1) & (bet_df["bet_size"] < 2)).sum()),
            "2-3":   int(((bet_df["bet_size"] >= 2) & (bet_df["bet_size"] < 3)).sum()),
            "3-4":   int(((bet_df["bet_size"] >= 3) & (bet_df["bet_size"] < 4)).sum()),
            "4-5":   int((bet_df["bet_size"] >= 4).sum()),
        },
    }


def overall_performance(df):
    """High-level performance summary."""
    bet_df = df[df["bet_size"] > 0] if "bet_size" in df.columns else df
    if bet_df.empty:
        return {}

    total_staked   = float(bet_df["bet_size"].sum())
    total_returned = float(bet_df["total_return"].sum())
    net            = total_returned - total_staked

    return {
        "total_bets":    int(len(bet_df)),
        "total_wins":    int(bet_df["won"].sum()),
        "total_losses":  int((bet_df["won"] == 0).sum()),
        "win_rate_pct":  round(float(bet_df["won"].mean()) * 100, 2),
        "units_staked":  round(total_staked, 2),
        "units_returned": round(total_returned, 2),
        "net_units":     round(net, 2),
        "rtp_pct":       round(total_returned / total_staked * 100, 2) if total_staked > 0 else 0,
        "avg_odds":      round(float(bet_df["odds_numeric"].mean()), 1) if "odds_numeric" in bet_df.columns else None,
    }


def prop_performance(df):
    """Overall performance stats for prop bets."""
    if df.empty:
        return {}
    bet_df = df[df["bet_size"].astype(float) > 0].copy()
    valid  = bet_df[bet_df["won"].notna()].copy()
    if valid.empty:
        return {}

    total_staked   = float(valid["bet_size"].sum())
    total_returned = float(valid["total_return"].sum())
    net            = total_returned - total_staked

    return {
        "total_props":   int(len(valid)),
        "total_wins":    int(valid["won"].sum()),
        "total_losses":  int((valid["won"] == 0).sum()),
        "win_rate_pct":  round(float(valid["won"].mean()) * 100, 2),
        "units_staked":  round(total_staked, 2),
        "units_returned": round(total_returned, 2),
        "net_units":     round(net, 2),
        "rtp_pct":       round(total_returned / total_staked * 100, 2) if total_staked > 0 else 0,
        "by_type": (
            valid.groupby("prop_type")
            .agg(count=("won","count"), wins=("won","sum"),
                 staked=("bet_size","sum"), returned=("total_return","sum"))
            .assign(win_rate=lambda x: (x["wins"]/x["count"]*100).round(1),
                    net=lambda x: (x["returned"]-x["staked"]).round(3))
            .reset_index()
            .to_dict("records")
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Statistical Analysis")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Load master results
    print("\n[1] Loading all betting results...")
    df, _ = read_csv("results/v2_all_betting_results.csv")

    if df.empty:
        print("  [WARN] No results data found. Saving empty stats.")
        stats = {"error": "No results data available", "analysis_date": datetime.now().isoformat()}
        write_json(stats, "statistics/v2_statistical_analysis.json",
                   f"Empty stats — {datetime.now().strftime('%Y-%m-%d')}")
        return

    # Only analyse rows where we actually placed bets
    bet_df = df[df["bet_size"] > 0].copy() if "bet_size" in df.columns else df.copy()
    print(f"  Loaded {len(df)} total rows, {len(bet_df)} with bets placed")

    # Load prop results
    print("\n[1b] Loading prop results...")
    prop_df, _ = read_csv("results/v2_all_prop_results.csv")
    if prop_df.empty:
        print("  No prop results yet")
    else:
        print(f"  Loaded {len(prop_df)} prop result rows")

    # Run all analyses
    print("\n[2] Running analyses...")
    output = {
        "analysis_date":         datetime.now().isoformat(),
        "overall_performance":   overall_performance(bet_df),
        "basic_statistics":      basic_stats(bet_df),
        "t_test":                rtp_ttest(bet_df),
        "monthly_performance":   monthly_performance(df),
        "calibration":           calibration_analysis(bet_df),
        "style_analysis":        style_accuracy(bet_df),
        "bet_size_distribution": bet_size_analysis(bet_df),
        "prop_performance":      prop_performance(prop_df) if not prop_df.empty else {},
    }

    # Print key stats
    perf = output["overall_performance"]
    if perf:
        print(f"\n  ── Performance Summary ──────────────────────────")
        print(f"  Total bets:     {perf.get('total_bets', 0)}")
        print(f"  Win rate:       {perf.get('win_rate_pct', 0):.1f}%")
        print(f"  Net units:      {perf.get('net_units', 0):+.2f}")
        print(f"  RTP:            {perf.get('rtp_pct', 0):.1f}%")
        print(f"  ────────────────────────────────────────────────")

    ttest = output["t_test"]
    if ttest and "conclusion" in ttest:
        print(f"\n  T-test vs {BREAKEVEN_THRESHOLD}: {ttest['conclusion']}")
        print(f"  p-value (one-sided): {ttest.get('p_value_one_sided', 'N/A')}")

    prop_perf = output["prop_performance"]
    if prop_perf:
        print(f"\n  ── Prop Summary ─────────────────────────────────")
        print(f"  Total props:    {prop_perf.get('total_props', 0)}")
        print(f"  Win rate:       {prop_perf.get('win_rate_pct', 0):.1f}%")
        print(f"  Net units:      {prop_perf.get('net_units', 0):+.2f}")
        print(f"  RTP:            {prop_perf.get('rtp_pct', 0):.1f}%")
        print(f"  ────────────────────────────────────────────────")

    # Save to GitHub
    print("\n[3] Saving statistics to GitHub...")
    write_json(
        output,
        "statistics/v2_statistical_analysis.json",
        f"V2 stats update — {datetime.now().strftime('%Y-%m-%d')}",
    )

    print("\n✓ Statistics analysis complete.")


def run_test():
    """
    --test mode: runs every analysis function against synthetic betting results
    and validates the JSON structure that update_site.yml expects.
    No GitHub reads or writes.
    """
    print("=== TEST MODE: 06_statistics ===\n")
    all_pass = True

    # Build synthetic results data matching v2_all_betting_results.csv schema
    np.random.seed(42)
    n = 80
    dates  = pd.date_range("2024-06-01", periods=n, freq="7D")
    odds   = np.random.choice([-150,-130,-110,-140,110,130,150,200,-200,-180], n)
    bets   = np.random.choice([1, 2, 3], n).astype(float)
    probs  = np.array([abs(o)/(abs(o)+100) if o < 0 else 100/(o+100) for o in odds])
    wins   = np.random.binomial(1, np.minimum(probs * 1.08, 0.99), n)

    def pnl(won, odd, bet):
        if won:
            return bet * ((100/abs(odd)) if odd < 0 else (odd/100))
        return -bet

    returns = np.array([pnl(wins[i], odds[i], bets[i]) + bets[i] for i in range(n)])
    mult    = returns / bets

    df = pd.DataFrame({
        "name":                 [f"Fighter {i}" for i in range(n)],
        "result_date":          dates,
        "event_date":           dates,
        "won":                  wins,
        "bet_size":             bets,
        "total_return":         returns,
        "profit_multiplier":    mult,
        "odds_numeric":         odds.astype(float),
        "predicted_probability": probs,
        "career_style":         np.random.choice(["Striker","Wrestler","BJJ","Mixed"], n),
    })
    bet_df = df[df["bet_size"] > 0].copy()

    # [1] overall_performance()
    print("[1] overall_performance()...")
    try:
        perf = overall_performance(bet_df)
        assert "total_bets"   in perf
        assert "win_rate_pct" in perf
        assert "net_units"    in perf
        assert "rtp_pct"      in perf
        print(f"  ✓ total_bets={perf['total_bets']}, win_rate={perf['win_rate_pct']}%, net={perf['net_units']:+.2f}u")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [2] rtp_ttest()
    print("\n[2] rtp_ttest()...")
    try:
        ttest = rtp_ttest(bet_df)
        assert "t_statistic"       in ttest
        assert "p_value_one_sided" in ttest
        assert "conclusion"        in ttest
        assert ttest["conclusion"] in ("Reject H0", "Fail to reject H0")
        print(f"  ✓ t={ttest['t_statistic']:.3f}, p={ttest['p_value_one_sided']:.4f}, {ttest['conclusion']}")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [3] basic_stats()
    print("\n[3] basic_stats()...")
    try:
        bs = basic_stats(bet_df)
        for k in ["count","mean","median","std","min","max"]:
            assert k in bs, f"Missing key: {k}"
        print(f"  ✓ count={bs['count']}, mean={bs['mean']:.4f}, std={bs['std']:.4f}")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [4] monthly_performance()
    print("\n[4] monthly_performance()...")
    try:
        monthly = monthly_performance(df)
        assert isinstance(monthly, list)
        if monthly:
            assert "month" in monthly[0] and "bets" in monthly[0]
        print(f"  ✓ {len(monthly)} months of data")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [5] calibration_analysis()
    print("\n[5] calibration_analysis()...")
    try:
        cal = calibration_analysis(bet_df)
        assert isinstance(cal, list)
        print(f"  ✓ {len(cal)} probability buckets")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [6] Full JSON output structure (what update_site.yml reads)
    print("\n[6] Full JSON output structure...")
    try:
        import json
        output = {
            "analysis_date":       datetime.now().isoformat(),
            "overall_performance": overall_performance(bet_df),
            "basic_statistics":    basic_stats(bet_df),
            "t_test":              rtp_ttest(bet_df),
            "monthly_performance": monthly_performance(df),
            "calibration":         calibration_analysis(bet_df),
            "style_analysis":      style_accuracy(bet_df),
            "bet_size_distribution": bet_size_analysis(bet_df),
        }
        serialized = json.dumps(output, default=str)
        restored   = json.loads(serialized)
        # Validate keys that update_site.yml and the frontend read
        assert "overall_performance" in restored
        assert "t_test"              in restored
        assert "win_rate_pct"        in restored["overall_performance"]
        assert "conclusion"          in restored["t_test"]
        print(f"  ✓ JSON serializes OK ({len(serialized):,} chars)")
        print(f"  ✓ All required site-data keys present")
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
