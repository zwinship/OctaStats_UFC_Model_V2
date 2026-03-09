#!/usr/bin/env python3
"""
01_scrape_historical.py
OctaStats V2 — One-time historical data scraper

Run this ONCE locally to build the initial dataset (2015-present).
This scrapes every fight card and fight from UFCStats, recording all
per-round stats and outcomes. The output is committed to the V2 repo
as data/raw/fight_stats.csv.

Usage:
    python 01_scrape_historical.py

Requirements:
    pip install requests beautifulsoup4 lxml pandas tqdm
"""

import requests
import pandas as pd
import time
import re
import json
import base64
import os
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

# ── GitHub config ────────────────────────────────────────────────────────────
GITHUB_TOKEN   = "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"   # zwinship PAT
REPO_OWNER     = "zwinship"
REPO_NAME      = "OctaStats_UFC_Model_V2"
CUTOFF_YEAR    = 2015          # Only scrape fights from this year onward

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OctaStats/2.0)"}
BASE_URL = "http://www.ufcstats.com"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_soup(url, retries=3, delay=1.2):
    """Fetch a URL and return a BeautifulSoup object. Retries on failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                time.sleep(delay)
                return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"  [WARN] Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(delay * 2)
    print(f"  [ERROR] Could not fetch: {url}")
    return None


def parse_stat(value_str):
    """
    Parse a stat that may look like '53 of 143', '37%', '0:39', or a plain number.
    Returns a tuple (landed, attempted) for 'X of Y', float for % and time, else raw.
    """
    if not value_str or value_str.strip() in ('---', '--', ''):
        return None, None

    s = value_str.strip()

    # "X of Y" format → return (landed, attempted)
    if ' of ' in s:
        parts = s.split(' of ')
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None

    # Percentage
    if s.endswith('%'):
        try:
            return float(s.rstrip('%')) / 100, None
        except ValueError:
            return None, None

    # Time "M:SS"
    if re.match(r'^\d+:\d{2}$', s):
        parts = s.split(':')
        return int(parts[0]) * 60 + int(parts[1]), None

    # Plain number
    try:
        return float(s), None
    except ValueError:
        return s, None


def parse_fight_page(fight_url, event_name, event_date, event_location):
    """
    Scrape a single fight page and return a list of row dicts — one per fighter.
    Captures totals only (not per-round) to keep the dataset manageable.
    """
    soup = get_soup(fight_url)
    if soup is None:
        return []

    rows = []

    # ── Fight metadata ────────────────────────────────────────────────────────
    method_el  = soup.select_one(".b-fight-details__text-item_first .b-fight-details__text-item-value")
    method     = method_el.get_text(strip=True) if method_el else None

    detail_items = soup.select(".b-fight-details__text-item")
    round_num, time_str, time_fmt, referee = None, None, None, None
    for item in detail_items:
        label = item.select_one(".b-fight-details__label")
        val   = item.select_one(".b-fight-details__text-item-value") or item
        if label:
            lbl = label.get_text(strip=True).lower()
            v   = val.get_text(strip=True).replace(label.get_text(strip=True), "").strip()
            if "round:" in lbl:
                try:    round_num = int(v)
                except: pass
            elif "time:" in lbl and "format" not in lbl:
                time_str = v
            elif "time format" in lbl:
                time_fmt = v
            elif "referee" in lbl:
                referee = v

    # Weight class and title bout
    bout_title = soup.select_one(".b-fight-details__fight-title")
    weight_class, title_bout = None, False
    if bout_title:
        bt = bout_title.get_text(strip=True)
        title_bout = "title" in bt.lower() or "championship" in bt.lower()
        # strip "Title Bout" and similar
        weight_class = re.sub(r'(ufc\s+)?(interim\s+)?(title\s+)?bout', '', bt, flags=re.IGNORECASE).strip()

    # Fighter names and winner
    fighter_els = soup.select(".b-fight-details__person")
    fighter_names, fighter_results = [], []
    for el in fighter_els:
        name_el   = el.select_one(".b-fight-details__person-name a")
        result_el = el.select_one(".b-fight-details__person-status")
        if name_el:
            fighter_names.append(name_el.get_text(strip=True))
            fighter_results.append(result_el.get_text(strip=True) if result_el else "")

    if len(fighter_names) < 2:
        return []

    # ── Totals table ──────────────────────────────────────────────────────────
    tables = soup.select(".b-fight-details__table")
    if not tables:
        return []

    totals_table = tables[0]
    t_rows = totals_table.select("tbody tr")
    if len(t_rows) < 1:
        return []

    # Significant strikes table (second table)
    sig_table = tables[1] if len(tables) > 1 else None

    def extract_table_row(row):
        """Extract td text values from a table row."""
        return [td.get_text(separator=' ', strip=True) for td in row.select("td")]

    for idx, fighter_row in enumerate(t_rows[:2]):  # max 2 fighters
        cells = extract_table_row(fighter_row)
        if len(cells) < 9:
            continue

        # Totals columns: KD | Sig.Str | Sig.Str% | Total Str | TD | TD% | Sub Att | Rev | Ctrl
        kd_val,        _  = parse_stat(cells[1])
        sig_landed,  sig_att = _split_of(cells[2])
        sig_pct,       _  = parse_stat(cells[3])
        tot_landed, tot_att = _split_of(cells[4])
        td_landed,   td_att = _split_of(cells[5])
        td_pct,        _  = parse_stat(cells[6])
        sub_att,       _  = parse_stat(cells[7])
        rev_val,       _  = parse_stat(cells[8])
        ctrl_secs,     _  = parse_stat(cells[9]) if len(cells) > 9 else (None, None)

        # Significant strikes breakdown
        head_l = head_a = body_l = body_a = leg_l = leg_a = None
        dist_l = dist_a = clinch_l = clinch_a = ground_l = ground_a = None

        if sig_table:
            sig_rows = sig_table.select("tbody tr")
            if idx < len(sig_rows):
                sc = extract_table_row(sig_rows[idx])
                if len(sc) >= 8:
                    head_l,   head_a   = _split_of(sc[3])
                    body_l,   body_a   = _split_of(sc[4])
                    leg_l,    leg_a    = _split_of(sc[5])
                    dist_l,   dist_a   = _split_of(sc[6])
                    clinch_l, clinch_a = _split_of(sc[7])
                    ground_l, ground_a = _split_of(sc[8]) if len(sc) > 8 else (None, None)

        name   = fighter_names[idx] if idx < len(fighter_names) else None
        result = fighter_results[idx] if idx < len(fighter_results) else ""
        won    = 1 if result.upper() == "W" else 0

        opp_idx  = 1 - idx
        opp_name = fighter_names[opp_idx] if opp_idx < len(fighter_names) else None

        rows.append({
            "fight_url":        fight_url,
            "event_name":       event_name,
            "event_date":       event_date,
            "event_location":   event_location,
            "fighter":          name,
            "opponent":         opp_name,
            "won":              won,
            "weight_class":     weight_class,
            "title_bout":       int(title_bout),
            "method":           method,
            "finish_round":     round_num,
            "finish_time":      time_str,
            "time_format":      time_fmt,
            "referee":          referee,
            # Striking totals
            "kd":               kd_val,
            "sig_str_landed":   sig_landed,
            "sig_str_att":      sig_att,
            "sig_str_pct":      sig_pct,
            "total_str_landed": tot_landed,
            "total_str_att":    tot_att,
            "td_landed":        td_landed,
            "td_att":           td_att,
            "td_pct":           td_pct,
            "sub_att":          sub_att,
            "reversals":        rev_val,
            "ctrl_seconds":     ctrl_secs,
            # Sig strike breakdown
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


def _split_of(cell_str):
    """Helper: '53 of 143' → (53, 143). Returns (None, None) on failure."""
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


# ── Event list scraper ────────────────────────────────────────────────────────

def get_all_event_urls(cutoff_year=2015):
    """
    Iterate through all pages of completed events and collect event URLs
    for events on or after cutoff_year.
    """
    event_urls = []
    page = 1

    print(f"Collecting event URLs from {cutoff_year} onward...")

    while True:
        url  = f"{BASE_URL}/statistics/events/completed?page={page}"
        soup = get_soup(url, delay=0.8)
        if soup is None:
            break

        rows = soup.select(".b-statistics__table-events tbody tr")
        if not rows:
            break

        found_any = False
        stop = False

        for row in rows:
            # Date column
            date_el = row.select_one("td:nth-child(2) span") or row.select_one("td:nth-child(2)")
            if not date_el:
                continue
            date_text = date_el.get_text(strip=True)
            try:
                event_date = datetime.strptime(date_text, "%B %d, %Y")
            except ValueError:
                # Try alternate format
                try:
                    event_date = datetime.strptime(date_text, "%b. %d, %Y")
                except ValueError:
                    continue

            if event_date.year < cutoff_year:
                stop = True
                break

            link_el = row.select_one("td:nth-child(1) a")
            loc_el  = row.select_one("td:nth-child(2)")
            if link_el:
                href     = link_el.get("href", "")
                name     = link_el.get_text(strip=True)
                location = loc_el.get_text(strip=True) if loc_el else ""
                event_urls.append({
                    "url":      href,
                    "name":     name,
                    "date":     event_date.strftime("%Y-%m-%d"),
                    "location": location,
                })
                found_any = True

        if stop or not found_any:
            break

        # Check for next page
        next_page = soup.select_one(".b-statistics__paginate-item_next a")
        if not next_page:
            break
        page += 1

    print(f"  Found {len(event_urls)} events")
    return event_urls


def get_fight_urls_from_event(event_url):
    """Return list of (fight_url) from an event page."""
    soup = get_soup(event_url, delay=0.8)
    if soup is None:
        return []

    fights = []
    for row in soup.select(".b-fight-details__table-row.b-fight-details__table-row__hover"):
        link = row.get("data-link") or ""
        if "fight-details" in link:
            fights.append(link)
    return fights


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Historical Data Scraper")
    print(f"Scraping all UFC fights from {CUTOFF_YEAR} to present")
    print("=" * 60)

    # Step 1: collect all event URLs
    events = get_all_event_urls(CUTOFF_YEAR)

    all_rows = []

    # Step 2: for each event, collect fight URLs then scrape each fight
    for event in tqdm(events, desc="Events"):
        fight_urls = get_fight_urls_from_event(event["url"])
        for fight_url in fight_urls:
            fight_rows = parse_fight_page(
                fight_url,
                event["name"],
                event["date"],
                event["location"],
            )
            all_rows.extend(fight_rows)

    if not all_rows:
        print("[ERROR] No data scraped. Check network and UFCStats availability.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values("event_date", ascending=False).reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/fight_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)} fighter-fight rows to {out_path}")

    # Step 3: upload to GitHub
    print("\nUploading to GitHub...")
    upload_csv_to_github(out_path, f"data/raw/fight_stats.csv",
                         f"Initial historical scrape — {datetime.now().strftime('%Y-%m-%d')}")
    print("✓ Done!")


def upload_csv_to_github(local_path, repo_path, message):
    """Upload a local file to the V2 GitHub repo."""
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()

    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    gh_headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Check for existing SHA
    check = requests.get(api_url, headers=gh_headers)
    payload = {"message": message, "content": content_b64}
    if check.status_code == 200:
        payload["sha"] = check.json()["sha"]

    resp = requests.put(api_url, headers=gh_headers, json=payload)
    if resp.status_code in (200, 201):
        print(f"  GitHub: {repo_path} {'updated' if check.status_code == 200 else 'created'}")
    else:
        print(f"  GitHub error {resp.status_code}: {resp.json().get('message')}")


if __name__ == "__main__":
    main()
