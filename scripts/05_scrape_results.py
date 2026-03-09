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
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
if not GITHUB_TOKEN:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
REPO_NAME    = "OctaStats_UFC_Model_V2"
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


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def github_get(repo_path):
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        return None, None
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


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

def get_recent_completed_event():
    """
    Return the most recent completed event if it happened within the last 7 days,
    else return None.
    """
    url  = f"{BASE_URL}/statistics/events/completed"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=15)
    if resp.status_code != 200:
        return None

    soup  = BeautifulSoup(resp.text, "lxml")
    rows  = soup.select("tr.b-statistics__table-row")
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)

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
            return {
                "name": link_el.get_text(strip=True),
                "date": event_date.isoformat(),
                "url":  link_el.get("href", ""),
            }

    return None


# ── Step 2: Scrape winners from the completed event ───────────────────────────

def get_winners_from_event(event_url):
    """
    Return dict of {fighter_name_lower: "W"/"L"} from a completed event page.

    UFCStats event pages: each fight row has fighter names in <p><a> tags inside
    the first <td>. The first <p> is always the winner on completed event pages.
    Falls back to scraping individual fight pages if needed.
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
        # Fighter names are in <p><a> tags inside the first td
        fighter_links = tds[0].select("p a.b-link")
        if len(fighter_links) >= 2:
            # First fighter = winner on completed event pages
            winner = fighter_links[0].get_text(strip=True)
            loser  = fighter_links[1].get_text(strip=True)
            results[winner.lower()] = "W"
            results[loser.lower()]  = "L"
        else:
            # Fallback: scrape individual fight page
            fight_link = row.get("data-link", "")
            if fight_link:
                winner = _scrape_fight_winner(fight_link)
                if winner:
                    results[winner.lower()] = "W"

    return results


def _scrape_fight_winner(fight_url):
    """Scrape the winner from an individual fight page."""
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


# ── Step 3-4: Match results to predictions and compute P&L ───────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Post-Fight Results Recorder")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Find the recent event
    print("\n[1] Checking for recent completed UFC event...")
    event = get_recent_completed_event()
    if event is None:
        print("  No UFC event found in the last 7 days. Exiting.")
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

        # 3. Scrape winners
        print("\n[3] Scraping fight results from UFCStats...")
        winners = get_winners_from_event(event["url"])
        print(f"  Found {len(winners)} winners: {', '.join(sorted(winners))}")

        # 4. Match winners and compute P&L
        print("\n[4] Computing P&L...")
        pred_df["won"] = pred_df["name"].apply(lambda x: 1 if x in winners else 0)

        # Only compute P&L for rows where we placed a bet
        pred_df["potential_winnings"] = 0.0
        pred_df["total_return"]       = 0.0
        pred_df["net"]                = 0.0
        pred_df["profit_multiplier"]  = 0.0

        bet_rows = pred_df[pred_df["bet_size"] > 0]
        for idx, row in bet_rows.iterrows():
            if pd.isna(row.get("odds_numeric")):
                continue
            w, tr, n, pm = calculate_pnl(row)
            pred_df.at[idx, "potential_winnings"] = w
            pred_df.at[idx, "total_return"]       = tr
            pred_df.at[idx, "net"]                = n
            pred_df.at[idx, "profit_multiplier"]  = pm

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


if __name__ == "__main__":
    main()
