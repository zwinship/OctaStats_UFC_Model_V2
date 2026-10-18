#!/usr/bin/env python3
"""
05_scrape_results.py
OctaStats V2 — Sunday post-fight results recorder

Runs Sunday at noon if a UFC event happened this week. Steps:
  1. Check if a fight happened this week (most recent completed event)
  2. Load V2 predictions CSV for that event
  3. Scrape fight results from UFCStats (who won each bout)
  4. Match winners against predictions, compute P&L
  5. Append results to master all_betting_results CSV
  6. Update fight_titles.json with new recent event

Environment variables:
    ZWINSHIP_PAT  — GitHub personal access token for zwinship account
"""

import os
import io
import re
import base64
import json
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from name_match import match_name, norm_name, last_name
from ufcstats_browser import fetch_html, SEL_EVENT_LIST, SEL_EVENT_PAGE

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
if not GITHUB_TOKEN:
    GH_HEADERS.pop("Authorization")   # --test runs read the public repo anonymously
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


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def github_get(repo_path):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        return None, None
    data = resp.json()
    sha  = data["sha"]
    raw_content = data.get("content", "").replace("\n", "").strip()
    if not raw_content and data.get("download_url"):
        raw_resp = requests.get(url, headers={**GH_HEADERS, "Accept": "application/vnd.github.raw"}, timeout=120)
        return (raw_resp.text if raw_resp.status_code == 200 else None), sha
    return base64.b64decode(raw_content).decode("utf-8"), sha


def github_put(repo_path, content_str, message, sha=None):
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    payload = {
        "message": content_str if message is None else message,
        "content": base64.b64encode(content_str.encode()).decode(),
    }
    if sha:
        payload["sha"] = sha
    if message:
        payload["message"] = message
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    ok   = resp.status_code in (200, 201)
    print(f"  {'✓' if ok else '✗'} GitHub: {repo_path}")
    return ok


def read_csv(repo_path):
    content, sha = github_get(repo_path)
    if content is None:
        return pd.DataFrame(), None
    return pd.read_csv(io.StringIO(content)), sha


def write_csv(df, repo_path, message, sha=None):
    if sha is None:
        _, sha = github_get(repo_path)
    github_put(repo_path, df.to_csv(index=False), message, sha)


# ── Step 1: Find most recent completed event ──────────────────────────────────

def _event_list_rows(html):
    soup = BeautifulSoup(html, "lxml")
    out  = []
    for row in soup.select("tr.b-statistics__table-row"):
        tds = row.find_all("td")
        if not tds:
            continue
        link_el   = tds[0].select_one("i.b-statistics__table-content a")
        date_span = tds[0].select_one("span.b-statistics__date")
        if not link_el or not date_span:
            continue
        date_text = date_span.get_text(strip=True)
        event_date = None
        for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
            try:
                event_date = datetime.strptime(date_text, fmt).date()
                break
            except ValueError:
                continue
        if event_date is None:
            continue
        out.append({"name": link_el.get_text(strip=True),
                    "date": event_date,
                    "url":  link_el.get("href", "").strip()})
    return out


def get_recent_completed_event():
    url  = f"{BASE_URL}/statistics/events/completed"
    print("  Fetching UFCStats completed events (Playwright)...")
    html = fetch_html(url, SEL_EVENT_LIST)
    if not html:
        raise RuntimeError("UFCStats completed-events page did not load "
                           "(browser check not cleared) — refusing to report 'no event'.")
    today    = datetime.now(timezone.utc).date()
    week_ago = today - timedelta(days=7)
    for ev in _event_list_rows(html):
        # The completed list also shows the next upcoming event at the top
        if week_ago <= ev["date"] <= today:
            return {"name": ev["name"], "date": ev["date"].isoformat(), "url": ev["url"]}
    return None


_EVENT_CACHE = {}


def get_event_results(event_url):
    """
    Parse a completed UFCStats event page into one record per bout:
        {fighter_1, fighter_2, outcome, method}
    outcome: 'win' (fighter_1 won — UFCStats lists the winner first),
             'draw', 'nc'
    method:  'KO/TKO' | 'Submission' | 'Decision' | 'DQ' | 'No Contest' | raw
    """
    if event_url in _EVENT_CACHE:
        return _EVENT_CACHE[event_url]
    print(f"  Fetching event results page (Playwright): {event_url}")
    html = fetch_html(event_url, SEL_EVENT_PAGE)
    if not html:
        raise RuntimeError(f"UFCStats event page did not load: {event_url}")
    soup  = BeautifulSoup(html, "lxml")
    bouts = []
    for row in soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover"):
        tds = row.find_all("td")
        if len(tds) < 8:
            continue
        names = [a.get_text(strip=True) for a in tds[1].select("p a")]
        if len(names) < 2:
            continue
        flag = tds[0].get_text(" ", strip=True).lower()
        if "draw" in flag:
            outcome = "draw"
        elif "nc" in flag.split() or "no contest" in flag:
            outcome = "nc"
        elif "win" in flag:
            outcome = "win"
        else:
            outcome = None   # not yet fought / page not updated
        # td7 = <p>METHOD</p><p>details</p>; classify on the method line only
        method_p   = tds[7].find("p")
        method_raw = (method_p or tds[7]).get_text(" ", strip=True).upper()
        if "DQ" in method_raw:
            method = "DQ"
        elif "OVERTURNED" in method_raw or "CNC" in method_raw or outcome == "nc":
            method = "No Contest"
        elif "KO" in method_raw:
            method = "KO/TKO"
        elif "SUB" in method_raw:
            method = "Submission"
        elif "DEC" in method_raw:
            method = "Decision"
        else:
            method = method_raw or "Unknown"
        bouts.append({"fighter_1": names[0], "fighter_2": names[1],
                      "outcome": outcome, "method": method})
    _EVENT_CACHE[event_url] = bouts
    return bouts


def get_winners_from_event(event_url):
    """{fighter_lower: 'W' | 'L' | 'D' | 'NC'} for every fighter who fought."""
    results = {}
    for b in get_event_results(event_url):
        f1, f2 = b["fighter_1"].lower(), b["fighter_2"].lower()
        if b["outcome"] == "win":
            results[f1], results[f2] = "W", "L"
        elif b["outcome"] == "draw":
            results[f1] = results[f2] = "D"
        elif b["outcome"] == "nc":
            results[f1] = results[f2] = "NC"
    return results


# ── Step 3-4: Match results to predictions and compute P&L ───────────────────

def _bet_result(name, winners):
    """'W' | 'L' | 'D' | 'NC', or None when the fighter is not on the results card."""
    hit = match_name(name, list(winners))
    return winners[hit] if hit is not None else None

def calculate_pnl(row, bet_size_col="bet_size", odds_col="odds_numeric"):
    """
    Calculate net P&L for a single bet row.
    Returns (potential_winnings, total_return, net, profit_multiplier)
    """
    bet   = row[bet_size_col]
    won   = row["won"]
    odds  = row[odds_col]

    if won == 1:
        if odds > 0:
            winnings = (odds / 100) * bet
        else:
            winnings = (100 / abs(odds)) * bet
        total_return = bet + winnings
    else:
        winnings     = 0
        total_return = 0

    net    = total_return - bet
    profit = (total_return / bet) if bet > 0 else 0.0

    return round(winnings, 4), round(total_return, 4), round(net, 4), round(profit, 4)


# ── Prop results helpers ──────────────────────────────────────────────────────

def get_fight_methods(event_url):
    """
    fight_methods:   frozenset({fighter_a_lower, fighter_b_lower}) → method
    winner_by_fight: frozenset → winner_lower (None for draws / no contests)
    method is one of: 'KO/TKO', 'Submission', 'Decision', 'No Contest', 'DQ'
    """
    fight_methods   = {}
    winner_by_fight = {}
    for b in get_event_results(event_url):
        if b["outcome"] is None:
            continue
        key = frozenset([b["fighter_1"].lower(), b["fighter_2"].lower()])
        fight_methods[key]   = "Draw" if b["outcome"] == "draw" else b["method"]
        winner_by_fight[key] = b["fighter_1"].lower() if b["outcome"] == "win" else None
    return fight_methods, winner_by_fight


def _find_fight(fa, fb, fight_methods):
    """Key in fight_methods for the bout between fa and fb (both must match)."""
    for key in fight_methods:
        names = list(key)
        ma, mb = match_name(fa, names), match_name(fb, names)
        if ma and mb and ma != mb:
            return key
    return None


def _prop_won(prop_row, fight_methods, winner_by_fight):
    """
    Determine if a prop bet won based on actual fight result.

    Returns 1 (win), 0 (loss), or None (fight not found / no contest → void).
    """
    fa  = str(prop_row.get("fighter_a") or "")
    fb  = str(prop_row.get("fighter_b") or "")
    pt  = prop_row.get("prop_type", "")
    key = _find_fight(fa, fb, fight_methods)
    if key is None:
        return None  # bout cancelled / not on the card → void

    method = fight_methods.get(key)
    winner = winner_by_fight.get(key)
    if method in (None, "No Contest", "Unknown"):
        return None

    if pt == "fight_goes_distance":
        return 1 if method in ("Decision", "Draw") else 0
    if pt == "fight_inside_distance":
        return 0 if method in ("Decision", "Draw") else 1

    wanted = {"ko": "KO/TKO", "sub": "Submission", "dec": "Decision"}.get(pt.rsplit("_", 1)[-1])
    if wanted is None:
        return 0
    fighter = str(prop_row.get("fighter") or "")
    won_fight = bool(winner) and match_name(fighter, [winner]) is not None
    return 1 if (won_fight and method == wanted) else 0


def track_prop_results(prop_df, fight_methods, winner_by_fight, event, clean_title):
    """
    Compute P&L for flagged props, write per-event CSV and append to master.
    Uses market_implied_prob to derive decimal odds for payout calculation.
    """
    if prop_df.empty:
        print("  No prop predictions found — skipping prop tracking")
        return

    bet_props = prop_df[prop_df["bet_size"].astype(float) > 0].copy()
    if bet_props.empty:
        print("  No flagged prop bets — skipping prop tracking")
        return

    print(f"  Processing {len(bet_props)} flagged prop bets...")

    results_rows = []
    for _, row in bet_props.iterrows():
        won_val = _prop_won(row, fight_methods, winner_by_fight)
        bet     = float(row.get("bet_size", 0))
        imp     = float(row.get("market_implied_prob") or 0)

        # Derive decimal odds from implied prob; guard against div/zero
        if imp > 0 and imp < 1:
            decimal_odds = 1.0 / imp
        else:
            decimal_odds = 2.0  # fallback even money

        if won_val is None:
            # No contest / void — return stake
            net          = 0.0
            total_return = bet
            won_flag     = None
        elif won_val == 1:
            winnings     = (decimal_odds - 1) * bet
            total_return = round(bet + winnings, 4)
            net          = round(winnings, 4)
            won_flag     = 1
        else:
            total_return = 0.0
            net          = round(-bet, 4)
            won_flag     = 0

        profit_mult = round(total_return / bet, 4) if bet > 0 else 0.0

        results_rows.append({
            "event_name":          event["name"],
            "event_date":          event["date"],
            "result_date":         datetime.now().strftime("%Y-%m-%d"),
            "fighter_a":           row.get("fighter_a", ""),
            "fighter_b":           row.get("fighter_b", ""),
            "fighter":             row.get("fighter", ""),
            "matchup":             row.get("matchup", ""),
            "prop_type":           row.get("prop_type", ""),
            "prop_label":          row.get("prop_label", ""),
            "model_prob":          row.get("model_prob"),
            "market_implied_prob": row.get("market_implied_prob"),
            "market_edge":         row.get("market_edge"),
            "bet_size":            bet,
            "won":                 won_flag,
            "total_return":        total_return,
            "net":                 net,
            "profit_multiplier":   profit_mult,
        })

    result_df = pd.DataFrame(results_rows)

    # Summary
    valid  = result_df[result_df["won"].notna()]
    won_ct = int(valid["won"].sum())
    print(f"\n  ── Prop Summary ──────────────────────────────")
    print(f"  Props bet:      {len(result_df)}")
    print(f"  Props won:      {won_ct} ({won_ct/max(len(valid),1)*100:.1f}%)")
    print(f"  Units staked:   {result_df['bet_size'].sum():.2f}")
    print(f"  Net P&L:        {result_df['net'].sum():+.2f} units")
    print(f"  ──────────────────────────────────────────────")

    # Save per-event prop results
    write_csv(
        result_df,
        f"results/v2_prop_results_{clean_title}.csv",
        f"V2 prop results: {event['name']} — {datetime.now().strftime('%Y-%m-%d')}",
    )

    # Append to master prop results
    master_df, master_sha = read_csv("results/v2_all_prop_results.csv")
    if master_df.empty:
        combined = result_df
    else:
        combined = pd.concat([master_df, result_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["prop_label", "fighter_a", "fighter_b", "event_name"], keep="last"
        )

    write_csv(
        combined,
        "results/v2_all_prop_results.csv",
        f"Append V2 prop results: {event['name']}",
        sha=master_sha,
    )

def _prop_payout(row, won_val):
    """(total_return, net, profit_multiplier) for one prop bet."""
    bet = float(row.get("bet_size", 0) or 0)
    try:
        imp = float(row.get("market_implied_prob") or 0)
    except (TypeError, ValueError):
        imp = 0.0
    decimal_odds = 1.0 / imp if 0 < imp < 1 else 2.0
    if won_val is None:
        total_return, net = bet, 0.0
    elif won_val == 1:
        net          = round((decimal_odds - 1) * bet, 4)
        total_return = round(bet + net, 4)
    else:
        total_return, net = 0.0, round(-bet, 4)
    return total_return, net, (round(total_return / bet, 4) if bet > 0 else 0.0)


def regrade_ungraded_props(dry_run=False):
    """
    Grade prop bets that were recorded as void because the old requests-based
    method scraper was blocked by UFCStats' browser check (every event from
    Noche UFC 2026 onward was affected). Re-reads each event page with
    Playwright and grades with the same rules as track_prop_results.
    Bets that are genuinely void (cancelled bout / no contest) stay void.
    """
    master, master_sha = read_csv("results/v2_all_prop_results.csv")
    if master.empty or "won" not in master.columns:
        return
    todo = master[master["won"].isna() & (pd.to_numeric(master["bet_size"], errors="coerce") > 0)]
    if todo.empty:
        print("  No ungraded prop bets.")
        return
    print(f"  {len(todo)} ungraded prop bets across {todo['event_name'].nunique()} events")

    html = fetch_html(f"{BASE_URL}/statistics/events/completed?page=all", SEL_EVENT_LIST)
    if not html:
        print("  [WARN] Could not load UFCStats event list — skipping regrade")
        return
    event_urls = {norm_name(e["name"]): e["url"] for e in _event_list_rows(html)}

    changed = 0
    for event_name, grp in todo.groupby("event_name"):
        url = event_urls.get(norm_name(event_name))
        if not url:
            print(f"  [WARN] {event_name}: not found on UFCStats")
            continue
        fight_methods, winner_by_fight = get_fight_methods(url)
        for idx, row in grp.iterrows():
            won_val = _prop_won(row, fight_methods, winner_by_fight)
            if won_val is None:
                continue
            tr, net, pm = _prop_payout(row, won_val)
            print(f"    {event_name[:38]:38s} {row['prop_label'][:42]:42s} → "
                  f"{'WON ' if won_val else 'LOST'} {net:+.2f}u")
            if not dry_run:
                master.loc[idx, ["won", "total_return", "net", "profit_multiplier"]] = [won_val, tr, net, pm]
            changed += 1

    print(f"  Regraded {changed} prop bets{' (dry run — nothing written)' if dry_run else ''}")
    if dry_run or not changed:
        return
    write_csv(master, "results/v2_all_prop_results.csv",
              f"Regrade previously ungraded V2 prop results ({changed} bets)", sha=master_sha)
    for event_name, grp in master[master.index.isin(todo.index)].groupby("event_name"):
        clean = re.sub(r"[^\w\s]", "", event_name).replace(" ", "_")
        per_event, sha = read_csv(f"results/v2_prop_results_{clean}.csv")
        if per_event.empty:
            continue
        cols      = ["won", "total_return", "net", "profit_multiplier"]
        key       = ["prop_label", "fighter_a", "fighter_b"]
        col_order = list(per_event.columns)
        per_event = per_event.set_index(key)
        per_event.update(grp.set_index(key)[cols])
        write_csv(per_event.reset_index()[col_order],
                  f"results/v2_prop_results_{clean}.csv",
                  f"Regrade V2 prop results: {event_name}", sha=sha)


def _track_parlay_result(pred_df, results, event, clean_title):
    """
    Check if the OCTASTATS PARLAY (all bet legs) hit, and append to parlay results CSV.
    A parlay wins only if every single bet leg in that event was correct.
    Parlay data comes from the first bet row which carries parlay_bet_size / parlay_combined_odds.
    """
    bet_rows = pred_df[pred_df["bet_size"] > 0].copy() if not pred_df.empty else pd.DataFrame()
    if bet_rows.empty:
        print("  No bet rows — skipping parlay tracking")
        return

    # Get parlay metadata from first row
    parlay_bet_size    = float(bet_rows.iloc[0].get("parlay_bet_size",    0) or 0)
    parlay_combined_odds = bet_rows.iloc[0].get("parlay_combined_odds", "")
    try:
        parlay_combined_odds = float(parlay_combined_odds) if str(parlay_combined_odds) not in ("", "nan", "None") else None
    except (TypeError, ValueError):
        parlay_combined_odds = None

    if parlay_bet_size <= 0 or parlay_combined_odds is None:
        print("  No parlay data in predictions — skipping")
        return

    # Did every leg win? A void leg (cancelled / draw / NC) drops out of the
    # parlay and its price is removed from the combined odds.
    legs = []
    combined_dec = (parlay_combined_odds / 100 + 1) if parlay_combined_odds > 0 \
        else (100 / abs(parlay_combined_odds) + 1)
    for _, row in bet_rows.iterrows():
        name   = row.get("name") or row.get("fighter")
        result = results.get(name)
        if result not in ("W", "L"):
            o = float(row.get("odds_numeric") or 0)
            if o:
                combined_dec /= (o / 100 + 1) if o > 0 else (100 / abs(o) + 1)
            legs.append({"name": name, "won": None})
            continue
        legs.append({"name": name, "won": result == "W"})

    live_legs = [l for l in legs if l["won"] is not None]
    all_won   = bool(live_legs) and all(l["won"] for l in live_legs)

    # Compute parlay P&L
    if not live_legs:
        net, total_return = 0.0, round(parlay_bet_size, 4)
    elif all_won:
        winnings     = (combined_dec - 1) * parlay_bet_size
        net          = round(winnings, 4)
        total_return = round(parlay_bet_size + winnings, 4)
    else:
        net = round(-parlay_bet_size, 4)
        total_return = 0.0

    parlay_row = {
        "event_name":      event["name"],
        "event_date":      event["date"],
        "result_date":     datetime.now().strftime("%Y-%m-%d"),
        "legs":            " & ".join(l["name"] for l in legs),
        "num_legs":        len(legs),
        "combined_odds":   parlay_combined_odds,
        "bet_size":        parlay_bet_size,
        "all_won":         int(all_won),
        "net":             net,
        "total_return":    total_return,
        "leg_detail":      " | ".join(
            f"{l['name']} ({'VOID' if l['won'] is None else 'W' if l['won'] else 'L'})" for l in legs),
    }

    print(f"  Parlay legs: {parlay_row['leg_detail']}")
    print(f"  Result: {'✓ WON' if all_won else '✗ LOST'}  Net: {net:+.2f}u")

    # Append to parlay results CSV
    master_parlay, parlay_sha = read_csv("results/v2_parlay_results.csv")
    new_row_df = pd.DataFrame([parlay_row])
    if master_parlay.empty:
        combined = new_row_df
    else:
        combined = pd.concat([master_parlay, new_row_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_name"], keep="last")

    write_csv(
        combined,
        "results/v2_parlay_results.csv",
        f"Parlay result: {event['name']} — {'WON' if all_won else 'LOST'}",
        sha=parlay_sha,
    )



def main():
    print("=" * 60)
    print("OctaStats V2 — Post-Fight Results Recorder")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Find the recent event
    print("\n[1] Checking for recent completed UFC event...")
    event = get_recent_completed_event()
    if event is None:
        print("  No UFC event found in the last 7 days.")
        print("\n[5c] Checking for previously ungraded prop bets...")
        regrade_ungraded_props()
        return
    print(f"  Event: {event['name']} ({event['date']})")

    # 2. Load V2 predictions for this event
    print("\n[2] Loading V2 predictions for this event...")
    clean_title = re.sub(r'[^\w\s]', '', event["name"]).replace(' ', '_')
    pred_path   = f"predictions/v2_betting_recommendations_{clean_title}.csv"
    pred_df, _  = read_csv(pred_path)

    if pred_df.empty:
        print(f"  [WARN] No V2 predictions found at {pred_path}")
        # Still update fight titles
    else:
        print(f"  Loaded {len(pred_df)} prediction rows")

        # 3. Scrape winners + fight methods
        print("\n[3] Scraping fight results from UFCStats...")
        bouts = get_event_results(event["url"])
        pending = [f"{b['fighter_1']} vs {b['fighter_2']}" for b in bouts if b["outcome"] is None]
        if pending:
            print(f"  [ERROR] UFCStats has not posted results for: {', '.join(pending)}")
            print("  Not writing partial results — re-run once the event page is complete.")
            raise SystemExit(1)
        winners = get_winners_from_event(event["url"])
        fight_methods, winner_by_fight = get_fight_methods(event["url"])
        print(f"  Found {len(bouts)} bouts, {sum(v == 'W' for v in winners.values())} winners: "
              f"{', '.join(sorted(k for k, v in winners.items() if v == 'W'))}")
        print(f"  Found {len(fight_methods)} fight methods")

        # 4. Match winners and compute P&L
        print("\n[4] Computing P&L...")
        pred_df["_result"] = pred_df["name"].apply(lambda n: _bet_result(n, winners))
        pred_df["won"]     = (pred_df["_result"] == "W").astype(int)

        # Only compute P&L for rows where we placed a bet
        pred_df["potential_winnings"] = 0.0
        pred_df["total_return"]       = 0.0
        pred_df["net"]                = 0.0
        pred_df["profit_multiplier"]  = 0.0

        bet_rows = pred_df[pred_df["bet_size"] > 0]
        for idx, row in bet_rows.iterrows():
            if pd.isna(row.get("odds_numeric")):
                continue
            if row["_result"] not in ("W", "L"):
                # Cancelled bout, draw or no contest → moneyline is a push
                print(f"  ↺ {row['name']}: {row['_result'] or 'bout not on card'} — stake returned")
                pred_df.at[idx, "total_return"]      = float(row["bet_size"])
                pred_df.at[idx, "profit_multiplier"] = 1.0
                continue
            w, tr, n, pm = calculate_pnl(row)
            pred_df.at[idx, "potential_winnings"] = w
            pred_df.at[idx, "total_return"]       = tr
            pred_df.at[idx, "net"]                = n
            pred_df.at[idx, "profit_multiplier"]  = pm

        results_by_name = dict(zip(pred_df["name"], pred_df.pop("_result")))
        pred_df["event_name"]  = event["name"]
        pred_df["result_date"] = datetime.now().strftime("%Y-%m-%d")

        # Summary
        bet_df  = pred_df[pred_df["bet_size"] > 0]
        won_df  = bet_df[bet_df["won"] == 1]
        total_staked  = bet_df["bet_size"].sum()
        total_returned = bet_df["total_return"].sum()
        net_pl        = total_returned - total_staked

        print(f"\n  ── Event Summary ──────────────────────────────")
        print(f"  Bets placed:    {len(bet_df)}")
        print(f"  Bets won:       {len(won_df)} ({len(won_df)/max(len(bet_df),1)*100:.1f}%)")
        print(f"  Units staked:   {total_staked:.2f}")
        print(f"  Units returned: {total_returned:.2f}")
        print(f"  Net P&L:        {net_pl:+.2f} units")
        print(f"  ──────────────────────────────────────────────")

        # 5. Save event-specific results
        write_csv(
            pred_df,
            f"results/v2_betting_results_{clean_title}.csv",
            f"V2 results: {event['name']} — {datetime.now().strftime('%Y-%m-%d')}",
        )

        # 6. Append to master all_betting_results CSV
        print("\n[5] Updating master results file...")
        master_df, master_sha = read_csv("results/v2_all_betting_results.csv")

        if master_df.empty:
            combined = pred_df
        else:
            combined = pd.concat([master_df, pred_df], ignore_index=True)
            # Deduplicate by name + event_name
            combined = combined.drop_duplicates(
                subset=["name", "event_name"], keep="last"
            )

        write_csv(
            combined,
            "results/v2_all_betting_results.csv",
            f"Append V2 results: {event['name']}",
            sha=master_sha,
        )

        # 5b. Track prop results
        print("\n[5b] Processing prop bet results...")
        prop_path = f"predictions/v2_prop_recommendations_{clean_title}.csv"
        prop_df, _ = read_csv(prop_path)
        if prop_df.empty:
            print(f"  [WARN] No prop predictions found at {prop_path}")
        else:
            print(f"  Loaded {len(prop_df)} prop rows")
            track_prop_results(prop_df, fight_methods, winner_by_fight, event, clean_title)

    # 6b. Track parlay result
    print("\n[5b] Tracking parlay result...")
    if not pred_df.empty:
        _track_parlay_result(pred_df, results_by_name, event, clean_title)

    # 6c. Grade any older prop bets that were left ungraded
    print("\n[5c] Checking for previously ungraded prop bets...")
    regrade_ungraded_props()

    # 7. Update fight_titles.json
    print("\n[6] Updating fight_titles.json...")
    titles_content, titles_sha = github_get("titles/fight_titles.json")
    existing_titles = json.loads(titles_content) if titles_content else {}

    existing_titles["recent"]      = event["name"]
    existing_titles["recent_date"] = event["date"]
    existing_titles["updated_at"]  = datetime.now().isoformat()

    github_put(
        "titles/fight_titles.json",
        json.dumps(existing_titles, indent=2),
        f"Update recent event title: {event['name']}",
        sha=titles_sha,
    )

    print(f"\n✓ Results processing complete for: {event['name']}")


def run_test():
    """
    --test mode: live UFCStats results parsing + offline matching logic.
    No GitHub writes.
    """
    print("=== TEST MODE: 05_scrape_results ===\n")
    all_pass = True

    print("[1] Recent completed event detection (live)...")
    event = get_recent_completed_event()
    print(f"  {'✓ Found: ' + event['name'] + ' (' + event['date'] + ')' if event else '○ No event in last 7 days'}")

    print("\n[2] Parsing the most recent completed event page (live)...")
    html = fetch_html(f"{BASE_URL}/statistics/events/completed", SEL_EVENT_LIST)
    today = datetime.now(timezone.utc).date()
    past = [e for e in _event_list_rows(html or "") if e["date"] <= today]
    if not past:
        print("  ✗ Could not load completed events"); all_pass = False
    else:
        ev = past[0]
        bouts = get_event_results(ev["url"])
        print(f"  {ev['name']} ({ev['date']}): {len(bouts)} bouts")
        for b in bouts:
            print(f"    {b['outcome'] or '?':5s} {b['fighter_1']:28s} def. {b['fighter_2']:28s} {b['method']}")
        if not bouts or any(b["outcome"] is None for b in bouts):
            print("  ✗ Missing bouts or outcomes"); all_pass = False
        fm, wf = get_fight_methods(ev["url"])
        print(f"  ✓ {len(fm)} fight methods parsed")

    print("\n[3] Name matching / cancelled-bout logic (offline)...")
    winners = {"jon jones": "W", "stipe miocic": "L", "jean silva": "W", "bruno silva": "L",
               "allen frye jr.": "D", "suyoung you": "NC"}
    cases = [("Jon Jones", "W"), ("Stipe Miocic", "L"), ("Jean Silva", "W"),
             ("Bruno Silva", "L"), ("Silva", None), ("Allen Frye", "D"),
             ("SuYoung You", "NC"), ("Cancelled Fighter", None), ("", None)]
    for name, expected in cases:
        got = _bet_result(name, winners)
        ok  = got == expected
        all_pass &= ok
        print(f"  {'✓' if ok else '✗'}  {name!r:22s} → {got!r}")

    print("\n[4] Prop regrade preview (reads GitHub, writes nothing)...")
    try:
        regrade_ungraded_props(dry_run=True)
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    print()
    print("=" * 60)
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED — see above'}")
    print("=" * 60)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        main()

