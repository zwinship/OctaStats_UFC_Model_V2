#!/usr/bin/env python3
"""
v1_scrape_results.py
OctaStats — V1 Model Sunday Results Recorder (Automated)

Finds the most recently completed UFC event, loads V1 predictions, scrapes
winners from UFCStats, computes P&L using V1 discrete bet sizing, and pushes
results to zwinship/UFC_Model.

V1 P&L logic (preserved):
  - Win: units * (1 / implied_prob - 1)   → actual payout at those odds
  - Loss: -units

CRITICAL FIX vs original:
  - Scraping now returns both winners_dict AND fought_set (set of all fighters
    confirmed as having competed in a bout that was recorded).
  - A fighter is only marked "L" if their bout is confirmed as having happened.
    If a fighter appears in predictions but their bout was cancelled / not
    yet recorded on UFCStats, they get result=None and are excluded from P&L.
  - This prevents cancelled or postponed bouts from being incorrectly scored
    as losses.

Environment variables:
    ZWINSHIP_PAT  — GitHub personal access token
"""

import os
import io
import re
import base64
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
if not GITHUB_TOKEN:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
V1_REPO      = "UFC_Model"
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
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


# ── GitHub helpers ────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path, repo=V1_REPO):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{repo}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame(), None
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), data["sha"]


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


# ── Scrape recent event results ───────────────────────────────────────────────

def get_recent_completed_event():
    """Return dict(event_name, event_date, event_url) for the most recently
    completed UFC event (within 8 days), or None."""
    for page in range(1, 4):
        url  = f"{BASE_URL}/statistics/events/completed?page={page}"
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=15)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("tr.b-statistics__table-row")
        for row in rows:
            link = row.select_one("a.b-link")
            if not link:
                continue
            name      = link.get_text(strip=True)
            event_url = link["href"]
            date_tds  = row.select("td")
            if len(date_tds) < 2:
                continue
            date_text = date_tds[1].get_text(strip=True)
            try:
                event_date = datetime.strptime(date_text, "%B %d, %Y").date()
            except ValueError:
                continue
            if (datetime.utcnow().date() - event_date).days <= 8:
                return {"name": name, "date": event_date, "url": event_url}
    return None


def scrape_fight_results(event_url):
    """
    Scrape the winner and loser of each fight from a UFCStats event page.

    Returns:
      winners_dict : {fighter_name_lower: "W" or "L"}
      fought_set   : set of fighter_name_lower for every fighter whose bout
                     was confirmed as completed (both corners present).

    IMPORTANT: Only fighters in fought_set should be marked W/L.
    A fighter in predictions but NOT in fought_set had their bout cancelled,
    no-contested, or not yet scraped — do NOT record them as a loss.
    """
    resp = requests.get(event_url, headers=SCRAPE_HEADERS, timeout=15)
    if resp.status_code != 200:
        return {}, set()

    soup    = BeautifulSoup(resp.text, "lxml")
    winners = {}
    fought  = set()

    rows = soup.select("tr.b-fight-details__table-row")
    for row in rows:
        fighters = row.select("a.b-link.b-fight-details__person-link")
        if len(fighters) < 2:
            # Try broader link selector
            fighters = row.select("a.b-link")
        if len(fighters) < 2:
            continue

        winner_name = fighters[0].get_text(strip=True)
        loser_name  = fighters[1].get_text(strip=True)
        if not winner_name or not loser_name:
            continue

        w_lower = winner_name.lower()
        l_lower = loser_name.lower()
        winners[w_lower] = "W"
        winners[l_lower] = "L"
        fought.add(w_lower)
        fought.add(l_lower)

    return winners, fought


# ── Match names ───────────────────────────────────────────────────────────────

def match_result(fighter_name, winners_dict, fought_set):
    """
    Return "W", "L", or None.
    None means the fight is NOT confirmed as having happened —
    caller must NOT record this as a loss.
    """
    name_l = fighter_name.lower().strip() if fighter_name else ""

    # Exact match in fought_set
    if name_l in fought_set:
        return winners_dict.get(name_l)

    # Last-name fuzzy match within fought_set only
    last = name_l.split()[-1] if name_l else ""
    for k in fought_set:
        if last and last in k:
            return winners_dict.get(k)

    # Partial token match within fought_set only
    parts = [p for p in name_l.split() if len(p) > 3]
    for k in fought_set:
        if any(p in k for p in parts):
            return winners_dict.get(k)

    # Fight not confirmed — do not mark as loss
    return None


# ── P&L computation ───────────────────────────────────────────────────────────

def american_to_decimal(odds):
    """Convert American odds to decimal (payout per unit staked, including stake)."""
    try:
        o = int(odds)
        return (100 / abs(o) + 1) if o < 0 else (o / 100 + 1)
    except (ValueError, TypeError):
        return None


def compute_pnl(row):
    """Compute net P&L for one V1 bet row. Returns 0.0 if result is None/unknown."""
    bet    = int(row.get("bet_size", 0))
    if bet == 0:
        return 0.0
    result = row.get("result")
    # FIX: if result is None (bout not confirmed), do not record a loss
    if result is None or (isinstance(result, float) and pd.isna(result)):
        return 0.0
    won  = result == "W"
    odds = row.get("Odds")
    dec  = american_to_decimal(odds)
    if dec is None:
        return 0.0
    return round(bet * (dec - 1), 2) if won else round(-bet, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("V1 Model — Sunday Results Recorder")
    print("=" * 60)

    # 1. Find most recent event
    print("\n[1] Checking for recent completed UFC event…")
    event_info = get_recent_completed_event()
    if event_info is None:
        print("  No UFC event in the last 8 days. Exiting.")
        return
    event_name = event_info["name"]
    event_date = event_info["date"]
    event_url  = event_info["url"]
    print(f"  Found: {event_name} ({event_date})")

    # 2. Load predictions for this event
    print("\n[2] Loading V1 predictions from UFC_Model repo…")
    preds, _ = read_csv_from_github("predictions/v1_predictions_latest.csv")
    if preds.empty:
        print("  No V1 predictions found. Exiting.")
        return
    print(f"  Loaded {len(preds)} rows")

    if "event" in preds.columns:
        events_in_preds = preds["event"].dropna().unique()
        match = any(event_name.lower()[:15] in e.lower() for e in events_in_preds)
        if not match:
            print(f"  [WARN] Predictions don't match event name '{event_name}' — proceeding anyway")

    # 3. Scrape fight results
    print("\n[3] Scraping fight results from UFCStats…")
    winners_dict, fought_set = scrape_fight_results(event_url)
    print(f"  Confirmed bouts: {len(fought_set) // 2}  ({len(fought_set)} fighters)")

    if not winners_dict:
        print("  [ERROR] No results scraped. Exiting.")
        return

    # 4. Match results to predictions
    print("\n[4] Matching results to predictions…")
    fighter_col = "fighter" if "fighter" in preds.columns else "name"
    preds["result"] = preds[fighter_col].apply(
        lambda n: match_result(n, winners_dict, fought_set))

    matched    = preds["result"].notna().sum()
    unmatched  = preds["result"].isna().sum()
    print(f"  Matched (fight confirmed): {matched}/{len(preds)}")
    if unmatched > 0:
        unmatched_names = preds[preds["result"].isna()][fighter_col].tolist()
        print(f"  Unmatched (bout not confirmed, excluded from P&L): {unmatched_names}")

    # 5. Compute P&L — only for confirmed results
    preds["net"] = preds.apply(compute_pnl, axis=1)
    preds["won"] = preds["result"].map({"W": 1, "L": 0})

    confirmed_bets = preds[
        (preds["bet_size"] > 0) & preds["result"].notna()
    ]
    skipped_bets = preds[
        (preds["bet_size"] > 0) & preds["result"].isna()
    ]
    net_pnl = confirmed_bets["net"].sum()
    staked  = confirmed_bets["bet_size"].sum()

    print(f"\n  Confirmed bets:  {len(confirmed_bets)}")
    print(f"  Skipped bets:    {len(skipped_bets)}  (bout not confirmed, not scored)")
    print(f"  Units staked:    {staked:.1f}u")
    print(f"  Net P&L:         {net_pnl:+.2f}u")
    if len(confirmed_bets):
        print(confirmed_bets[[fighter_col, "bet_size", "result", "net"]].to_string(index=False))

    # 6. Append to master results file
    print("\n[5] Appending to V1 master results file…")
    all_results, _ = read_csv_from_github("results/v1_all_betting_results.csv")
    preds["event_name"] = event_name
    preds["event_date"] = str(event_date)

    if not all_results.empty:
        event_col = "event" if "event" in all_results.columns else "event_name"
        already = all_results[
            all_results.get(event_col, pd.Series(dtype=str))
            .str.contains(event_name[:20], na=False)
        ]
        if len(already) > 0:
            print(f"  [WARN] Results for {event_name} already in master file — overwriting")
            all_results = all_results[
                ~all_results.get(event_col, pd.Series(dtype=str))
                .str.contains(event_name[:20], na=False)
            ]
        updated = pd.concat([all_results, preds], ignore_index=True)
    else:
        updated = preds.copy()

    write_csv_to_github(
        updated,
        "results/v1_all_betting_results.csv",
        f"feat: V1 results for {event_name}",
        repo=V1_REPO,
    )

    # Also write event-specific results file
    safe_name = re.sub(r"[^a-z0-9_]", "_", event_name.lower())
    write_csv_to_github(
        preds,
        f"results/v1_results_{safe_name}.csv",
        f"feat: V1 results — {event_name}",
        repo=V1_REPO,
    )

    print(f"\n✓ V1 results complete — {net_pnl:+.2f}u net for {event_name}.")


def run_test():
    """--test mode: scrape most recent event and validate fight-existence logic."""
    print("=== TEST MODE: v1_scrape_results.py ===\n")

    print("[1] Finding recent completed event…")
    event_info = get_recent_completed_event()
    if event_info is None:
        print("  ✗ No event in the last 8 days")
        print("  (Expected mid-week — run again on Sunday)")
        return

    print(f"  ✓ Found: {event_info['name']} ({event_info['date']})")

    print("\n[2] Scraping fight results…")
    winners, fought = scrape_fight_results(event_info["url"])
    print(f"  Confirmed bouts: {len(fought) // 2}  ({len(fought)} fighters)")
    for f in sorted(fought):
        print(f"    {winners.get(f, '?')}  {f}")

    print("\n[3] Validating fight-existence check…")
    fake = "nobody fake fighter"
    r = match_result(fake, winners, fought)
    ok = r is None
    print(f"  {'✓' if ok else '✗'}  Fake fighter → None (not 'L'): {r}")

    if fought:
        real_winner = next(f for f in fought if winners.get(f) == "W")
        rw = match_result(real_winner, winners, fought)
        print(f"  {'✓' if rw == 'W' else '✗'}  Real winner '{real_winner}' → '{rw}'")

    print("\n=== END TEST ===")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        main()
