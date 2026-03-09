#!/usr/bin/env python3
"""
02_scrape_weekly_update.py
OctaStats V2 — Wednesday incremental data updater

Run every Wednesday at noon via GitHub Actions. Checks for any new
completed events since the last scrape date, appends new fight rows
to data/raw/fight_stats.csv, and pushes the update to GitHub.

Environment variables (set as GitHub Actions secrets):
    ZWINSHIP_PAT  — personal access token for zwinship account
"""

import requests
import pandas as pd
import base64
import os
import io
import time
import re
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT", "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")
REPO_OWNER   = "zwinship"
REPO_NAME    = "OctaStats_UFC_Model_V2"
BASE_URL     = "http://www.ufcstats.com"
HEADERS      = {"User-Agent": "Mozilla/5.0 (compatible; OctaStats/2.0)"}
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path):
    """Read a CSV from GitHub, returning a DataFrame and the file SHA."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        print(f"  [WARN] Could not read {repo_path} from GitHub (status {resp.status_code})")
        return pd.DataFrame(), None
    data    = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    sha     = data["sha"]
    return pd.read_csv(io.StringIO(content)), sha


def write_csv_to_github(df, repo_path, message, sha=None):
    """Write a DataFrame as CSV to GitHub, updating if sha is provided."""
    csv_str  = df.to_csv(index=False)
    content  = base64.b64encode(csv_str.encode()).decode()
    url      = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    payload  = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    if resp.status_code in (200, 201):
        print(f"  ✓ GitHub: {repo_path} {'updated' if sha else 'created'}")
    else:
        print(f"  ✗ GitHub error {resp.status_code}: {resp.json().get('message')}")


# ── Shared scraping logic (mirrors 01_scrape_historical.py) ───────────────────

def get_soup(url, retries=3, delay=1.2):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                time.sleep(delay)
                return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"  [WARN] {e}")
            time.sleep(delay * 2)
    return None


def _split_of(cell_str):
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


def parse_stat(s):
    if not s or s.strip() in ('---', '--', ''):
        return None, None
    s = s.strip()
    if ' of ' in s:
        p = s.split(' of ')
        try:    return int(p[0]), int(p[1])
        except: return None, None
    if s.endswith('%'):
        try:    return float(s.rstrip('%')) / 100, None
        except: return None, None
    if re.match(r'^\d+:\d{2}$', s):
        p = s.split(':')
        return int(p[0]) * 60 + int(p[1]), None
    try:    return float(s), None
    except: return s, None


def parse_fight_page(fight_url, event_name, event_date, event_location):
    """Scrape a single fight page — identical logic to script 01."""
    soup = get_soup(fight_url)
    if soup is None:
        return []
    rows = []

    method_el = soup.select_one(".b-fight-details__text-item_first .b-fight-details__text-item-value")
    method = method_el.get_text(strip=True) if method_el else None

    round_num, time_str, time_fmt, referee = None, None, None, None
    for item in soup.select(".b-fight-details__text-item"):
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

    bout_title = soup.select_one(".b-fight-details__fight-title")
    weight_class, title_bout = None, False
    if bout_title:
        bt = bout_title.get_text(strip=True)
        title_bout   = "title" in bt.lower() or "championship" in bt.lower()
        weight_class = re.sub(r'(ufc\s+)?(interim\s+)?(title\s+)?bout', '', bt, flags=re.IGNORECASE).strip()

    fighter_names, fighter_results = [], []
    for el in soup.select(".b-fight-details__person"):
        name_el   = el.select_one(".b-fight-details__person-name a")
        result_el = el.select_one(".b-fight-details__person-status")
        if name_el:
            fighter_names.append(name_el.get_text(strip=True))
            fighter_results.append(result_el.get_text(strip=True) if result_el else "")

    if len(fighter_names) < 2:
        return []

    tables    = soup.select(".b-fight-details__table")
    if not tables:
        return []
    sig_table = tables[1] if len(tables) > 1 else None

    def row_cells(row):
        return [td.get_text(separator=' ', strip=True) for td in row.select("td")]

    for idx, fighter_row in enumerate(tables[0].select("tbody tr")[:2]):
        cells = row_cells(fighter_row)
        if len(cells) < 9:
            continue

        kd_val,        _   = parse_stat(cells[1])
        sig_landed, sig_att = _split_of(cells[2])
        sig_pct,       _   = parse_stat(cells[3])
        tot_landed,tot_att  = _split_of(cells[4])
        td_landed,  td_att  = _split_of(cells[5])
        td_pct,        _   = parse_stat(cells[6])
        sub_att,       _   = parse_stat(cells[7])
        rev_val,       _   = parse_stat(cells[8])
        ctrl_secs,     _   = parse_stat(cells[9]) if len(cells) > 9 else (None, None)

        head_l=head_a=body_l=body_a=leg_l=leg_a=None
        dist_l=dist_a=clinch_l=clinch_a=ground_l=ground_a=None
        if sig_table:
            sig_rows = sig_table.select("tbody tr")
            if idx < len(sig_rows):
                sc = row_cells(sig_rows[idx])
                if len(sc) >= 8:
                    head_l,   head_a   = _split_of(sc[3])
                    body_l,   body_a   = _split_of(sc[4])
                    leg_l,    leg_a    = _split_of(sc[5])
                    dist_l,   dist_a   = _split_of(sc[6])
                    clinch_l, clinch_a = _split_of(sc[7])
                    ground_l, ground_a = _split_of(sc[8]) if len(sc) > 8 else (None, None)

        rows.append({
            "fight_url": fight_url, "event_name": event_name,
            "event_date": event_date, "event_location": event_location,
            "fighter": fighter_names[idx],
            "opponent": fighter_names[1 - idx],
            "won": 1 if (fighter_results[idx].upper() == "W") else 0,
            "weight_class": weight_class, "title_bout": int(title_bout),
            "method": method, "finish_round": round_num,
            "finish_time": time_str, "time_format": time_fmt, "referee": referee,
            "kd": kd_val,
            "sig_str_landed": sig_landed, "sig_str_att": sig_att, "sig_str_pct": sig_pct,
            "total_str_landed": tot_landed, "total_str_att": tot_att,
            "td_landed": td_landed, "td_att": td_att, "td_pct": td_pct,
            "sub_att": sub_att, "reversals": rev_val, "ctrl_seconds": ctrl_secs,
            "head_landed": head_l, "head_att": head_a,
            "body_landed": body_l, "body_att": body_a,
            "leg_landed": leg_l,  "leg_att": leg_a,
            "distance_landed": dist_l, "distance_att": dist_a,
            "clinch_landed": clinch_l, "clinch_att": clinch_a,
            "ground_landed": ground_l, "ground_att": ground_a,
        })
    return rows


def get_fight_urls_from_event(event_url):
    soup = get_soup(event_url, delay=0.8)
    if soup is None:
        return []
    return [
        row.get("data-link")
        for row in soup.select(".b-fight-details__table-row.b-fight-details__table-row__hover")
        if row.get("data-link") and "fight-details" in row.get("data-link")
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Weekly Data Update")
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Load existing data to find the most recent event date
    existing_df, existing_sha = read_csv_from_github("data/raw/fight_stats.csv")

    if existing_df.empty:
        print("[ERROR] No existing fight_stats.csv found. Run 01_scrape_historical.py first.")
        return

    latest_date = pd.to_datetime(existing_df["event_date"]).max()
    print(f"Most recent event in dataset: {latest_date.strftime('%Y-%m-%d')}")

    existing_urls = set(existing_df["fight_url"].unique())

    # Scrape the first 2 pages of completed events (only need recent ones)
    new_rows = []
    stop     = False

    for page in range(1, 4):
        if stop:
            break
        url  = f"{BASE_URL}/statistics/events/completed?page={page}"
        soup = get_soup(url, delay=0.8)
        if soup is None:
            break

        for row in soup.select(".b-statistics__table-events tbody tr"):
            date_el = row.select_one("td:nth-child(2) span") or row.select_one("td:nth-child(2)")
            if not date_el:
                continue
            date_text = date_el.get_text(strip=True)
            try:
                event_date = datetime.strptime(date_text, "%B %d, %Y")
            except ValueError:
                try:
                    event_date = datetime.strptime(date_text, "%b. %d, %Y")
                except ValueError:
                    continue

            # Stop once we're at or before the last scraped date
            if event_date <= latest_date:
                stop = True
                break

            link_el = row.select_one("td:nth-child(1) a")
            loc_el  = row.select_one("td:nth-child(2)")
            if not link_el:
                continue

            event_url  = link_el.get("href", "")
            event_name = link_el.get_text(strip=True)
            location   = loc_el.get_text(strip=True) if loc_el else ""
            date_str   = event_date.strftime("%Y-%m-%d")

            print(f"  New event: {event_name} ({date_str})")
            fight_urls = get_fight_urls_from_event(event_url)
            new_fight_urls = [u for u in fight_urls if u not in existing_urls]

            for fight_url in new_fight_urls:
                rows = parse_fight_page(fight_url, event_name, date_str, location)
                new_rows.extend(rows)
                existing_urls.add(fight_url)

    if not new_rows:
        print("No new fight data found. Dataset is up to date.")
        return

    new_df   = pd.DataFrame(new_rows)
    combined = pd.concat([new_df, existing_df], ignore_index=True)
    combined = combined.sort_values("event_date", ascending=False).reset_index(drop=True)

    write_csv_to_github(
        combined,
        "data/raw/fight_stats.csv",
        f"Weekly data update — {datetime.now().strftime('%Y-%m-%d')}",
        sha=existing_sha,
    )
    print(f"\n✓ Added {len(new_rows)} new fighter-fight rows. Total: {len(combined)}")


if __name__ == "__main__":
    main()
