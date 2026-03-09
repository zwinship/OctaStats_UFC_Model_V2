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
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
if not GITHUB_TOKEN:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER     = "zwinship"
REPO_NAME      = "OctaStats_UFC_Model_V2"
CUTOFF_YEAR    = 2015          # Only scrape fights from this year onward

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection":      "keep-alive",
}
BASE_URL = "http://www.ufcstats.com"


# ── Helpers ───────────────────────────────────────────────────────────────────

# Reuse a single session so cookies/keep-alive are preserved
_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)


def get_soup(url, retries=3, delay=1.2):
    """Fetch a URL and return a BeautifulSoup object. Retries on failure."""
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=20, allow_redirects=True)
            if resp.status_code == 200:
                time.sleep(delay)
                return BeautifulSoup(resp.text, "lxml")
            else:
                print(f"  [WARN] HTTP {resp.status_code} for {url}")
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


def _cell_val(td, fighter_idx):
    """
    UFCStats fight pages store BOTH fighters\'  values in a single <td>,
    each inside a <p> tag. fighter_idx 0 = fighter on left, 1 = fighter on right.
    Falls back to raw td text if no <p> tags found.
    """
    ps = td.select("p")
    if len(ps) >= 2:
        return ps[fighter_idx].get_text(strip=True)
    elif len(ps) == 1:
        return ps[0].get_text(strip=True)
    return td.get_text(separator=" ", strip=True)


def parse_fight_page(fight_url, event_name, event_date, event_location):
    """
    Scrape a single fight page and return a list of row dicts — one per fighter.

    Key UFCStats structure insight: each <td> in the stats tables contains
    BOTH fighters\' values in separate <p> tags — index 0 = fighter 1, index 1 = fighter 2.
    """
    soup = get_soup(fight_url)
    if soup is None:
        return []

    # ── Fight metadata ────────────────────────────────────────────────────────
    method, round_num, time_str, time_fmt, referee = None, None, None, None, None

    for item in soup.select(".b-fight-details__text-item"):
        label_el = item.select_one(".b-fight-details__label")
        if not label_el:
            continue
        lbl = label_el.get_text(strip=True).lower()
        # value = everything in the item after stripping the label text
        raw = item.get_text(strip=True)
        lbl_text = label_el.get_text(strip=True)
        v = raw[len(lbl_text):].strip()
        if "method" in lbl:
            method = v
        elif "round" in lbl and "format" not in lbl:
            try: round_num = int(v)
            except: pass
        elif "time" in lbl and "format" not in lbl:
            time_str = v
        elif "format" in lbl:
            time_fmt = v
        elif "referee" in lbl:
            referee = v

    # Weight class and title bout
    weight_class, title_bout = None, False
    bout_title_el = soup.select_one(".b-fight-details__fight-title")
    if bout_title_el:
        bt = bout_title_el.get_text(strip=True)
        title_bout   = "title" in bt.lower() or "championship" in bt.lower()
        weight_class = re.sub(r"(ufc\s+)?(interim\s+)?(title\s+)?bout", "",
                               bt, flags=re.IGNORECASE).strip()

    # Fighter names and results
    fighter_names, fighter_results = [], []
    for el in soup.select(".b-fight-details__person"):
        name_el   = el.select_one(".b-fight-details__person-name a")
        result_el = el.select_one(".b-fight-details__person-status")
        if name_el:
            fighter_names.append(name_el.get_text(strip=True))
            fighter_results.append(result_el.get_text(strip=True) if result_el else "")

    if len(fighter_names) < 2:
        return []

    # ── Stats tables ──────────────────────────────────────────────────────────
    # Table 0 = totals, Table 1 = significant strikes breakdown
    tables = soup.select(".b-fight-details__table")
    totals_table = tables[0] if len(tables) > 0 else None
    sig_table    = tables[1] if len(tables) > 1 else None

    # Get the single data row from each table (one row contains both fighters)
    def get_data_row(table):
        if table is None:
            return None
        for tr in table.select("tbody tr"):
            tds = tr.select("td")
            if tds and any(td.select("p") for td in tds):
                return tds
        return None

    totals_tds = get_data_row(totals_table)
    sig_tds    = get_data_row(sig_table)

    rows = []
    for idx in range(2):
        name   = fighter_names[idx]
        result = fighter_results[idx]
        won    = 1 if result.upper() == "W" else 0
        opp    = fighter_names[1 - idx]

        # ── Totals ────────────────────────────────────────────────────────────
        # Column order: Fighter | KD | Sig.Str. | Sig.Str.% | Total Str. | Td | Td% | Sub. Att | Rev. | Ctrl
        kd_val = sig_landed = sig_att = sig_pct = None
        tot_landed = tot_att = td_landed = td_att = td_pct = None
        sub_att_val = rev_val = ctrl_secs = None

        if totals_tds and len(totals_tds) >= 9:
            kd_val,              _ = parse_stat(_cell_val(totals_tds[1], idx))
            sig_landed,    sig_att = _split_of(_cell_val(totals_tds[2], idx))
            sig_pct,             _ = parse_stat(_cell_val(totals_tds[3], idx))
            tot_landed,    tot_att = _split_of(_cell_val(totals_tds[4], idx))
            td_landed,      td_att = _split_of(_cell_val(totals_tds[5], idx))
            td_pct,              _ = parse_stat(_cell_val(totals_tds[6], idx))
            sub_att_val,         _ = parse_stat(_cell_val(totals_tds[7], idx))
            rev_val,             _ = parse_stat(_cell_val(totals_tds[8], idx))
            if len(totals_tds) > 9:
                ctrl_raw = _cell_val(totals_tds[9], idx)
                ctrl_secs, _ = parse_stat(ctrl_raw)

        # ── Sig strike breakdown ──────────────────────────────────────────────
        # Column order: Fighter | Sig.Str. | Sig.Str.% | Head | Body | Leg | Distance | Clinch | Ground
        head_l = head_a = body_l = body_a = leg_l = leg_a = None
        dist_l = dist_a = clinch_l = clinch_a = ground_l = ground_a = None

        if sig_tds and len(sig_tds) >= 9:
            head_l,   head_a   = _split_of(_cell_val(sig_tds[3], idx))
            body_l,   body_a   = _split_of(_cell_val(sig_tds[4], idx))
            leg_l,    leg_a    = _split_of(_cell_val(sig_tds[5], idx))
            dist_l,   dist_a   = _split_of(_cell_val(sig_tds[6], idx))
            clinch_l, clinch_a = _split_of(_cell_val(sig_tds[7], idx))
            if len(sig_tds) > 8:
                ground_l, ground_a = _split_of(_cell_val(sig_tds[8], idx))

        rows.append({
            "fight_url":        fight_url,
            "event_name":       event_name,
            "event_date":       event_date,
            "event_location":   event_location,
            "fighter":          name,
            "opponent":         opp,
            "won":              won,
            "weight_class":     weight_class,
            "title_bout":       int(title_bout),
            "method":           method,
            "finish_round":     round_num,
            "finish_time":      time_str,
            "time_format":      time_fmt,
            "referee":          referee,
            "kd":               kd_val,
            "sig_str_landed":   sig_landed,
            "sig_str_att":      sig_att,
            "sig_str_pct":      sig_pct,
            "total_str_landed": tot_landed,
            "total_str_att":    tot_att,
            "td_landed":        td_landed,
            "td_att":           td_att,
            "td_pct":           td_pct,
            "sub_att":          sub_att_val,
            "reversals":        rev_val,
            "ctrl_seconds":     ctrl_secs,
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
    Iterate through all pages of completed events and collect event URLs.

    Confirmed UFCStats HTML structure:
      <tr class="b-statistics__table-row">
        <td class="b-statistics__table-col">
          <i class="b-statistics__table-content">
            <a href="http://www.ufcstats.com/event-details/...">Event Name</a>
            <span class="b-statistics__date">March 07, 2026</span>
          </i>
        </td>
        <td class="b-statistics__table-col ...">Las Vegas, Nevada, USA</td>
      </tr>
    """
    event_urls = []
    page = 1

    print(f"Collecting event URLs from {cutoff_year} onward...")

    while True:
        url  = f"{BASE_URL}/statistics/events/completed?page={page}"
        print(f"  Fetching page {page}...")
        soup = get_soup(url, delay=1.0)
        if soup is None:
            print(f"  [WARN] Could not load page {page}, stopping.")
            break

        rows = soup.select("tr.b-statistics__table-row")
        if not rows:
            print(f"  No rows found on page {page} — stopping.")
            break

        found_any = False
        stop      = False

        for row in rows:
            # Skip header row (contains <th> not <td>)
            if row.find("th"):
                continue

            tds = row.find_all("td")
            if not tds:
                continue

            first_td = tds[0]

            # Link is inside <i class="b-statistics__table-content"> > <a>
            link_el = first_td.select_one("i.b-statistics__table-content a")
            if not link_el:
                continue
            href = link_el.get("href", "").strip()
            name = link_el.get_text(strip=True)
            if not href or "event-details" not in href:
                continue

            # Date is in <span class="b-statistics__date"> inside same td
            date_span = first_td.select_one("span.b-statistics__date")
            if not date_span:
                continue
            date_text = date_span.get_text(strip=True)

            # Location is in second td
            location = tds[1].get_text(strip=True) if len(tds) > 1 else ""

            # Parse date
            event_date = None
            for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
                try:
                    event_date = datetime.strptime(date_text, fmt)
                    break
                except ValueError:
                    continue

            if event_date is None:
                continue

            if event_date.year < cutoff_year:
                stop = True
                break

            event_urls.append({
                "url":      href,
                "name":     name,
                "date":     event_date.strftime("%Y-%m-%d"),
                "location": location,
            })
            found_any = True

        if stop:
            print(f"  Reached events before {cutoff_year} — stopping.")
            break

        if not found_any:
            print(f"  No valid events on page {page} — stopping.")
            break

        # Just increment the page number — stop when no events are found
        page += 1
        time.sleep(0.5)

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
    import sys
    if "--test" in sys.argv:
        # Diagnostic mode: fetch the events page and print what we find
        print("=== DIAGNOSTIC MODE ===")
        url  = f"{BASE_URL}/statistics/events/completed?page=1"
        resp = _SESSION.get(url, timeout=20, allow_redirects=True)
        print(f"HTTP status:  {resp.status_code}")
        print(f"Final URL:    {resp.url}")
        print(f"Content len:  {len(resp.text)} chars")
        soup = BeautifulSoup(resp.text, "lxml")
        # Find first real event
        rows = soup.select("tr.b-statistics__table-row")
        first_event = None
        for row in rows:
            if row.find("th"):
                continue
            link = row.select_one("i.b-statistics__table-content a")
            date = row.select_one("span.b-statistics__date")
            if link and date:
                first_event = {"url": link.get("href",""), "name": link.get_text(strip=True),
                               "date": date.get_text(strip=True), "location": ""}
                tds = row.find_all("td")
                if len(tds) > 1:
                    first_event["location"] = tds[1].get_text(strip=True)
                break

        if not first_event:
            print("\n✗ Could not find any events.")
            print("=== END DIAGNOSTIC ===")
        else:
            print(f"\n✓ First event found: {first_event['name']} ({first_event['date']})")
            print(f"  URL: {first_event['url']}")
            print(f"\nScraping fight card for that event...")
            fight_urls = get_fight_urls_from_event(first_event["url"])
            print(f"  Found {len(fight_urls)} fights")

            if not fight_urls:
                print("  ✗ No fight URLs found — check get_fight_urls_from_event()")
                print("=== END DIAGNOSTIC ===")
            else:
                print(f"\nScraping first fight: {fight_urls[0]}")
                rows_data = parse_fight_page(
                    fight_urls[0],
                    first_event["name"],
                    first_event["date"],
                    first_event["location"],
                )
                if not rows_data:
                    print("  ✗ No data returned from fight page")
                else:
                    print(f"\n✓ Got {len(rows_data)} fighter rows")
                    print(f"\n{'='*60}")
                    print("ALL COLUMNS AND VALUES:")
                    print(f"{'='*60}")
                    for fighter_row in rows_data:
                        print(f"\n  Fighter: {fighter_row.get('fighter','?')}")
                        print(f"  {'─'*50}")
                        for col, val in fighter_row.items():
                            print(f"  {col:<22} {repr(val)}")
                    print(f"\n{'='*60}")
                    print(f"Total columns: {len(rows_data[0])}")
                    print("\n✓ Structure looks good — run without --test to scrape all data.")
        print("=== END DIAGNOSTIC ===")
    else:
        main()
