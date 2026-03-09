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

_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)


def get_soup(url, retries=3, delay=1.2):
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=20, allow_redirects=True)
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

    for page in range(1, 6):  # check up to 5 pages, stop when we hit old events
        if stop:
            break
        url  = f"{BASE_URL}/statistics/events/completed?page={page}"
        soup = get_soup(url, delay=0.8)
        if soup is None:
            break

        for row in soup.select("tr.b-statistics__table-row"):
            if row.find("th"):  # skip header row
                continue
            tds = row.find_all("td")
            if not tds:
                continue

            # Name + link inside <i class="b-statistics__table-content"> > <a>
            link_el = tds[0].select_one("i.b-statistics__table-content a")
            if not link_el:
                continue
            event_url  = link_el.get("href", "").strip()
            event_name = link_el.get_text(strip=True)
            if not event_url or "event-details" not in event_url:
                continue

            # Date inside <span class="b-statistics__date">
            date_span = tds[0].select_one("span.b-statistics__date")
            if not date_span:
                continue
            date_text = date_span.get_text(strip=True)

            event_date = None
            for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
                try:
                    event_date = datetime.strptime(date_text, fmt)
                    break
                except ValueError:
                    continue
            if event_date is None:
                continue

            # Stop once we reach events we already have
            if event_date <= latest_date:
                stop = True
                break

            location = tds[1].get_text(strip=True) if len(tds) > 1 else ""
            date_str = event_date.strftime("%Y-%m-%d")

            print(f"  New event: {event_name} ({date_str})")
            fight_urls     = get_fight_urls_from_event(event_url)
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
