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
import sys as _sys
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
if not GITHUB_TOKEN and "--test" not in _sys.argv:
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
    """Return (event_name, event_date, event_url) for the most recent completed UFC event
    within the last 8 days, or None if not found."""
    from datetime import timezone
    url  = f"{BASE_URL}/statistics/events/completed"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=15)
    if resp.status_code != 200:
        return None

    soup     = BeautifulSoup(resp.text, "lxml")
    rows     = soup.select("tr.b-statistics__table-row")
    today    = datetime.now(timezone.utc).date()
    week_ago = today - timedelta(days=8)

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
        date_text = date_span.get_text(strip=True)
        try:
            event_date = datetime.strptime(date_text, "%B %d, %Y").date()
        except ValueError:
            try:
                event_date = datetime.strptime(date_text, "%b. %d, %Y").date()
            except ValueError:
                continue

        if week_ago <= event_date <= today:
            return link_el.get_text(strip=True), event_date, link_el.get("href", "")

    return None


def _scrape_fight_winner_from_page(fight_url):
    """Scrape the winner from an individual fight detail page (fallback)."""
    import time
    time.sleep(0.8)
    resp = requests.get(fight_url, headers=SCRAPE_HEADERS, timeout=15)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    for person_el in soup.select(".b-fight-details__person"):
        status = person_el.select_one(".b-fight-details__person-status")
        name   = person_el.select_one(".b-fight-details__person-name a")
        if status and name and status.get_text(strip=True).upper() == "W":
            return name.get_text(strip=True)
    return None


def scrape_fight_results(event_url):
    """
    Scrape the winner of each fight from a UFCStats event page.
    Returns dict: {fighter_name_lower: "W" or "L"}
    Mirrors V2's get_winners_from_event() exactly, including tds[1]
    fallback and individual fight page fallback.
    """
    resp = requests.get(event_url, headers=SCRAPE_HEADERS, timeout=15, allow_redirects=True)
    if resp.status_code != 200:
        return {}

    soup    = BeautifulSoup(resp.text, "lxml")
    results = {}

    for row in soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover"):
        tds = row.find_all("td")
        if not tds:
            continue
        fighter_links = tds[0].select("p a.b-link") or (tds[1].select("p a.b-link") if len(tds) > 1 else [])
        if len(fighter_links) >= 2:
            winner = fighter_links[0].get_text(strip=True)
            loser  = fighter_links[1].get_text(strip=True)
            results[winner.lower()] = "W"
            results[loser.lower()]  = "L"
        else:
            fight_link = row.get("data-link", "")
            if fight_link:
                winner = _scrape_fight_winner_from_page(fight_link)
                if winner:
                    results[winner.lower()] = "W"

    return results


# ── Match names ───────────────────────────────────────────────────────────────

def match_result(fighter_name, results_dict):
    """Fuzzy-match a fighter name to the results dict."""
    if not fighter_name or not fighter_name.strip():
        return None
    name_l = fighter_name.lower().strip()
    if name_l in results_dict:
        return results_dict[name_l]
    parts = name_l.split()
    if not parts:
        return None
    last = parts[-1]
    for k, v in results_dict.items():
        if last in k:
            return v
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
    """Compute net P&L for one V1 bet row."""
    bet   = int(row.get("bet_size", 0))
    if bet == 0:
        return 0.0
    won   = row.get("result") == "W"
    # Handle both 'Odds' and 'odds' column names
    odds  = row.get("Odds") if row.get("Odds") is not None else row.get("odds")
    dec   = american_to_decimal(odds)
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
    event_name, event_date, event_url = event_info
    print(f"  Found: {event_name} ({event_date})")

    # 2. Load predictions for this event
    print("\n[2] Loading V1 predictions from UFC_Model repo…")
    preds, _ = read_csv_from_github("predictions/v1_predictions_latest.csv")
    if preds.empty:
        print("  No V1 predictions found. Exiting.")
        return
    print(f"  Loaded {len(preds)} rows")

    # Verify predictions match this event (check event name overlap)
    if "event" in preds.columns:
        events_in_preds = preds["event"].dropna().unique()
        match = any(event_name.lower()[:15] in e.lower() for e in events_in_preds)
        if not match:
            print(f"  [WARN] Predictions don't match event name '{event_name}' — proceeding anyway")

    # 3. Scrape fight results
    print("\n[3] Scraping fight results from UFCStats…")
    results_dict = scrape_fight_results(event_url)
    print(f"  Scraped results for {len(results_dict)} fighters")

    if not results_dict:
        print("  [ERROR] No results scraped. Exiting.")
        return

    # 4. Match results to predictions
    print("\n[4] Matching results to predictions…")
    fighter_col = "fighter" if "fighter" in preds.columns else "name"
    preds["result"] = preds[fighter_col].apply(lambda n: match_result(n, results_dict))
    matched = preds["result"].notna().sum()
    print(f"  Matched {matched}/{len(preds)} fighters")

    # 5. Compute P&L
    preds["net"] = preds.apply(compute_pnl, axis=1)
    preds["won"] = (preds["result"] == "W").astype(int)

    bets   = preds[preds["bet_size"] > 0]
    net_pnl = bets["net"].sum()
    staked  = bets["bet_size"].sum()

    print(f"\n  Bets this event:  {len(bets)}")
    print(f"  Units staked:     {staked:.1f}u")
    print(f"  Net P&L:          {net_pnl:+.2f}u")
    if len(bets):
        print(bets[[fighter_col, "bet_size", "result", "net"]].to_string(index=False))

    # 6. Append to shared master results file
    print("\n[5] Appending to master results file…")
    all_results, _ = read_csv_from_github("results/all_betting_results.csv")

    # Normalize V1 preds to match master schema before appending
    preds["event_name"] = event_name
    preds["event_date"] = str(event_date)
    # Master uses 'name' — V1 preds use 'fighter'
    if "name" not in preds.columns:
        preds["name"] = preds["fighter"]
    # result col: match_result returns "W"/"L"/None — rename to match master
    if "result" not in preds.columns:
        preds["result"] = preds.get("result")
    # won col: 1 if W else 0
    if "won" not in preds.columns:
        preds["won"] = (preds["result"] == "W").astype(int)

    if not all_results.empty:
        combined = pd.concat([all_results, preds], ignore_index=True)
        combined = combined.drop_duplicates(subset=["name", "event_name"], keep="last")
    else:
        combined = preds.copy()

    write_csv_to_github(
        combined,
        "results/all_betting_results.csv",
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
    """
    --test mode: tests UFCStats results scraping and match_result logic.
    No GitHub writes.
    """
    print("=== TEST MODE: v1_scrape_results ===\n")
    all_pass = True

    # [1] Recent completed event (live)
    print("[1] Looking for recent completed event (within last 8 days)...")
    event_info = get_recent_completed_event()
    if event_info is None:
        print("  ○ No event in last 8 days — expected mid-week")
        print("  ✓ Function returned None cleanly")
    else:
        event_name, event_date, event_url = event_info
        print(f"  ✓ Found: {event_name} ({event_date})")
        print(f"  URL: {event_url}")

        # [2] Scrape results (live)
        print("\n[2] Scraping fight results...")
        try:
            results = scrape_fight_results(event_url)
            if not results:
                print("  ✗ No results scraped")
                all_pass = False
            else:
                print(f"  ✓ {len(results)} fighters scraped")
                for name, result in list(results.items())[:6]:
                    print(f"    {result}  {name}")
        except Exception as e:
            print(f"  ✗ {e}"); all_pass = False

    # [3] match_result() logic (offline)
    print("\n[3] match_result() logic (offline)...")
    try:
        results_dict = {
            "jon jones":        "W",
            "stipe miocic":     "L",
            "charles oliveira": "W",
            "michael chandler": "L",
        }
        cases = [
            ("Jon Jones",          "W",  "Exact match winner"),
            ("Stipe Miocic",       "L",  "Exact match loser"),
            ("Jones",              "W",  "Last-name match"),
            ("Charles Oliveira",   "W",  "Full name match"),
            ("Cancelled Fighter",  None, "Cancelled — must be None not L"),
            ("Unknown Fighter",    None, "Unknown — must be None"),
            ("",                   None, "Empty name — must be None"),
        ]
        for name, expected, desc in cases:
            got = match_result(name, results_dict)
            ok  = got == expected
            print(f"  {'✓' if ok else '✗'}  {desc}: got={got!r}")
            if not ok: all_pass = False

        # Critical: cancelled fighter must never return "L"
        assert match_result("Cancelled Fighter", results_dict) is None, \
            "CRITICAL: cancelled fighter must be None not L"
        print("  ✓ Cancelled fighter safety check passed")
    except Exception as e:
        print(f"  ✗ {e}"); all_pass = False

    # [4] american_to_decimal() (offline)
    print("\n[4] american_to_decimal() accuracy...")
    try:
        cases = [
            (-150, round(1 + 100/150, 4)),
            (+130, round(1 + 130/100, 4)),
            (-110, round(1 + 100/110, 4)),
            (None, None),
        ]
        for odds, expected in cases:
            got = american_to_decimal(odds)
            if expected is None:
                ok = got is None
            else:
                ok = got is not None and abs(got - expected) < 0.01
            print(f"  {'✓' if ok else '✗'}  {odds} → {got} (expect {expected})")
            if not ok: all_pass = False
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
