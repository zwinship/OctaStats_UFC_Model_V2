#!/usr/bin/env python3
"""
01_scrape_historical.py
OctaStats V2 — One-time historical data scraper

Run this ONCE locally to build the initial dataset (2015-present).
Scrapes UFCStats for fight-level stats, then merges in:
  - ufc-master.csv   : historical moneyline + prop odds, fighter attributes
  - rankings_history.csv : weekly UFC rankings snapshots

Output: data/raw/fight_stats.csv (committed to GitHub)

Usage:
    python 01_scrape_historical.py           # full run
    python 01_scrape_historical.py --test    # UFCStats scrape diagnostic
    python 01_scrape_historical.py --test-merge  # offline merge logic test

Required files in same directory as this script:
    ufc-master.csv        (historical odds + attributes)
    rankings_history.csv  (weekly rankings snapshots)

Requirements:
    pip install requests beautifulsoup4 lxml pandas tqdm
"""

import requests
import pandas as pd
import numpy as np
import time
import re
import json
import base64
import os
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

# ── GitHub config ─────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")  # optional — upload to GitHub is manual
REPO_OWNER  = "zwinship"
REPO_NAME   = "OctaStats_UFC_Model_V2"
CUTOFF_YEAR = 2015

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

_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_soup(url, retries=3, delay=1.2):
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
    if not value_str or value_str.strip() in ('---', '--', ''):
        return None, None
    s = value_str.strip()
    if ' of ' in s:
        parts = s.split(' of ')
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    if s.endswith('%'):
        try:
            return float(s.rstrip('%')) / 100, None
        except ValueError:
            return None, None
    if re.match(r'^\d+:\d{2}$', s):
        parts = s.split(':')
        return int(parts[0]) * 60 + int(parts[1]), None
    try:
        return float(s), None
    except ValueError:
        return s, None


def _cell_val(td, fighter_idx):
    ps = td.select("p")
    if len(ps) >= 2:
        return ps[fighter_idx].get_text(strip=True)
    elif len(ps) == 1:
        return ps[0].get_text(strip=True)
    return td.get_text(separator=" ", strip=True)


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


# ── Fight page parser ─────────────────────────────────────────────────────────

def parse_fight_page(fight_url, event_name, event_date, event_location):
    soup = get_soup(fight_url)
    if soup is None:
        return []

    method, round_num, time_str, time_fmt, referee, finish_details = None, None, None, None, None, None
    judge_scores = []   # list of "Judge Name: score1 - score2" strings

    # UFCStats fight detail page has two <p> blocks in .b-fight-details__content:
    #   p[0]: Method / Round / Time / Time format / Referee / Details
    #   p[1]: Scorecards (one item per judge, no label)
    # Each item is <i class="b-fight-details__text-item">
    # Labels are <i class="b-fight-details__label">Label:</i>
    # Values are plain text nodes (NOT in a child <i> tag — just whitespace-separated text)

    for item in soup.select(".b-fight-details__text-item"):
        label_el = item.select_one(".b-fight-details__label")

        if label_el is None:
            # No label = scorecard line: "Judge Name  score1 - score2"
            raw = item.get_text(" ", strip=True)
            # Match "Name 49 - 46" or "Name  49-46"
            m = re.match(r"^(.+?)\s+(\d+\s*[-–]\s*\d+)\s*\.$", raw)
            if not m:
                m = re.match(r"^(.+?)\s+(\d+\s*[-–]\s*\d+)$", raw)
            if m:
                judge_scores.append(f"{m.group(1).strip()}: {m.group(2).strip()}")
            elif raw and not raw.startswith("http"):
                # Still capture whatever's there
                judge_scores.append(raw)
            continue

        lbl = label_el.get_text(strip=True).lower()

        # Value = all text in this item EXCEPT the label text
        # Use get_text with separator to avoid concatenation artifacts
        full_text = item.get_text(" ", strip=True)
        label_text = label_el.get_text(strip=True)
        v = full_text[len(label_text):].strip().lstrip(":").strip()

        if not v or v in ("---", "--"):
            continue

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
        elif "detail" in lbl:
            finish_details = v or None

    # ── Method + Details fallback: parse from full content block ────────────
    # UFCStats renders Method and Details outside .b-fight-details__text-item
    # on some page variants. The full content area is a flat text string like:
    # "Method: KO/TKO Round: 1 Time: 4:27 Time format: 3 Rnd (5-5-5) Referee: Herb Dean Details: Punches"
    # or for decisions:
    # "Method: Decision - Unanimous Round: 5 Time: 5:00 ... Details: Mike Bell 29 - 28. ..."
    # We parse by splitting on known label keywords.
    content_el = soup.select_one(".b-fight-details__content")
    if content_el:
        full = content_el.get_text(" ", strip=True)
        # Extract each labelled field by matching up to the next label or end-of-string
        LABELS = r"(?:Method|Round|Time format|Time|Referee|Details)"
        field_re = re.compile(
            r"(Method|Round|Time format|Time|Referee|Details):\s*"
            r"(.*?)"
            r"(?=\s*(?:Method|Round|Time format|Time|Referee|Details):|$)",
            re.IGNORECASE
        )
        for fm in field_re.finditer(full):
            lbl_found = fm.group(1).lower()
            val_found = fm.group(2).strip()
            if not val_found or val_found in ("---", "--"):
                continue
            if lbl_found == "method" and method is None:
                method = val_found
            elif lbl_found == "details" and finish_details is None:
                # Only treat as finish_details if it's NOT judge score format
                # (judge scores look like "Name 29 - 28." with digits)
                if not re.match(r"^[A-Za-z\s']+\s+\d+\s*[-–]\s*\d+", val_found):
                    finish_details = val_found

    weight_class, title_bout = None, False
    bout_title_el = soup.select_one(".b-fight-details__fight-title")
    if bout_title_el:
        bt = bout_title_el.get_text(strip=True)
        bt_lower = bt.lower()
        title_bout = (
            "title" in bt_lower or
            "championship" in bt_lower or
            "interim" in bt_lower
        )
        # Strip UFC prefix, interim/title qualifiers, and "Bout" suffix to get clean weight class
        weight_class = re.sub(
            r"(?i)(ufc\s+)?(interim\s+)?(women'?s?\s+)?(title\s+)?championship\s+",
            "", bt
        )
        weight_class = re.sub(r"(?i)\s*bout\s*$", "", weight_class).strip()
        # If we ended up with just empty or very short string, use the raw text minus "Bout"
        if len(weight_class) < 3:
            weight_class = re.sub(r"(?i)\s*bout\s*$", "", bt).strip() or None

    fighter_names, fighter_results, fighter_urls = [], [], []
    for el in soup.select(".b-fight-details__person"):
        name_el   = el.select_one(".b-fight-details__person-name a")
        result_el = el.select_one(".b-fight-details__person-status")
        if name_el:
            fighter_names.append(name_el.get_text(strip=True))
            fighter_results.append(result_el.get_text(strip=True) if result_el else "")
            fighter_urls.append(name_el.get("href", "").strip())

    if len(fighter_names) < 2:
        return []

    tables     = soup.select(".b-fight-details__table")
    totals_tds = None
    sig_tds    = None

    def get_data_row(table):
        if table is None:
            return None
        for tr in table.select("tbody tr"):
            tds = tr.select("td")
            if tds and any(td.select("p") for td in tds):
                return tds
        return None

    if len(tables) > 0: totals_tds = get_data_row(tables[0])
    if len(tables) > 1: sig_tds    = get_data_row(tables[1])

    # ── Scorecard margin: sum of per-judge |winner_score - loser_score| ─────
    # Scores on UFCStats are always listed as fighter[0]_score - fighter[1]_score
    # (page order matches fighter_names order).
    # For a 29-28; 29-28; 29-28 unanimous: margin = 1+1+1 = 3
    # For a 30-27; 30-27; 30-27 dominant:  margin = 3+3+3 = 9
    # For a 49-46 (5-round, 3 judges):      margin = 3+3+3 = 9
    # We store the raw margin (always positive at fight level).
    # In the row dict: winner gets +margin, loser gets -margin.
    scorecard_margin = None
    if judge_scores:
        total = 0
        parsed = 0
        for js in judge_scores:
            # Parse "Judge Name: 29 - 28" or "Judge Name: 49-46" etc.
            m = re.search(r"(\d+)\s*[-–]\s*(\d+)", js)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                total += abs(a - b)
                parsed += 1
        if parsed > 0:
            scorecard_margin = total

    rows = []
    for idx in range(2):
        name   = fighter_names[idx]
        result = fighter_results[idx]
        won    = 1 if result.upper() == "W" else 0
        opp    = fighter_names[1 - idx]

        kd_val = sig_landed = sig_att = sig_pct = None
        tot_landed = tot_att = td_landed = td_att = td_pct = None
        sub_att_val = rev_val = ctrl_secs = None

        if totals_tds and len(totals_tds) >= 9:
            kd_val,          _ = parse_stat(_cell_val(totals_tds[1], idx))
            sig_landed, sig_att = _split_of(_cell_val(totals_tds[2], idx))
            sig_pct,         _ = parse_stat(_cell_val(totals_tds[3], idx))
            tot_landed, tot_att = _split_of(_cell_val(totals_tds[4], idx))
            td_landed,  td_att  = _split_of(_cell_val(totals_tds[5], idx))
            td_pct,          _ = parse_stat(_cell_val(totals_tds[6], idx))
            sub_att_val,     _ = parse_stat(_cell_val(totals_tds[7], idx))
            rev_val,         _ = parse_stat(_cell_val(totals_tds[8], idx))
            if len(totals_tds) > 9:
                ctrl_secs, _ = parse_stat(_cell_val(totals_tds[9], idx))

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
            "fighter_url":      fighter_urls[idx] if idx < len(fighter_urls) else "",
            "opponent":         opp,
            "won":              won,
            "weight_class":     weight_class,
            "title_bout":       int(title_bout),
            "method":           method,
            "finish_round":     round_num,
            "finish_time":      time_str,
            "time_format":      time_fmt,
            "referee":          referee,
            "finish_details":   finish_details,
            "judge_scores":     "; ".join(judge_scores) if judge_scores else None,
            # Positive for winner, negative for loser; None for non-decisions
            "scorecard_margin": scorecard_margin * (1 if won else -1) if scorecard_margin is not None else None,
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


# ── Event list scraper ────────────────────────────────────────────────────────

def get_all_event_urls(cutoff_year=2015):
    event_urls = []
    page = 1
    print(f"Collecting event URLs from {cutoff_year} onward...")

    while True:
        url  = f"{BASE_URL}/statistics/events/completed?page={page}"
        print(f"  Fetching page {page}...")
        soup = get_soup(url, delay=1.0)
        if soup is None:
            break

        rows      = soup.select("tr.b-statistics__table-row")
        found_any = False
        stop      = False

        for row in rows:
            if row.find("th"):
                continue
            tds = row.find_all("td")
            if not tds:
                continue
            link_el = tds[0].select_one("i.b-statistics__table-content a")
            if not link_el:
                continue
            href = link_el.get("href", "").strip()
            name = link_el.get_text(strip=True)
            if not href or "event-details" not in href:
                continue

            date_span = tds[0].select_one("span.b-statistics__date")
            if not date_span:
                continue
            date_text = date_span.get_text(strip=True)
            location  = tds[1].get_text(strip=True) if len(tds) > 1 else ""

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

        page += 1
        time.sleep(0.5)

    print(f"  Found {len(event_urls)} events")
    return event_urls


def get_fight_urls_from_event(event_url):
    soup = get_soup(event_url, delay=0.8)
    if soup is None:
        return []
    fights = []
    for row in soup.select(".b-fight-details__table-row.b-fight-details__table-row__hover"):
        link = row.get("data-link") or ""
        if "fight-details" in link:
            fights.append(link)
    return fights


# ── Fighter page scraper ──────────────────────────────────────────────────────

def _parse_height(s):
    """Convert '6\\' 4"' or '5\\' 11"' to centimetres. Returns np.nan on failure."""
    if not s or str(s).strip() in ("--", "---", ""):
        return np.nan
    m = re.search(r"(\d+)'\s*(\d+)", str(s))
    if m:
        feet, inches = int(m.group(1)), int(m.group(2))
        return round((feet * 12 + inches) * 2.54, 2)
    return np.nan


def _parse_reach(s):
    """Convert '77"' to centimetres. Returns np.nan on failure."""
    if not s or str(s).strip() in ("--", "---", ""):
        return np.nan
    m = re.search(r"([\d.]+)", str(s))
    if m:
        return round(float(m.group(1)) * 2.54, 2)
    return np.nan


def _parse_weight(s):
    """Convert '205 lbs.' to float pounds. Returns np.nan on failure."""
    if not s or str(s).strip() in ("--", "---", ""):
        return np.nan
    m = re.search(r"([\d.]+)", str(s))
    if m:
        return float(m.group(1))
    return np.nan


def _parse_dob(s):
    """Parse 'Nov 07, 1990' → datetime. Returns None on failure."""
    if not s or str(s).strip() in ("--", "---", ""):
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except ValueError:
            pass
    return None


def scrape_fighter_page(fighter_url):
    """
    Scrape a UFCStats fighter detail page and return a dict with:
        height_cms, reach_cms, weight_lbs, stance, dob
    Returns an empty dict on failure.
    """
    if not fighter_url or "fighter-details" not in fighter_url:
        return {}
    soup = get_soup(fighter_url, delay=0.6)
    if soup is None:
        return {}

    info = {}
    # The stats are in <li class="b-list__box-list-item b-list__box-list-item_type_block">
    # Each item contains a <i class="b-list__box-item-title"> label + plain text value
    for li in soup.select("li.b-list__box-list-item"):
        label_el = li.select_one("i.b-list__box-item-title")
        if not label_el:
            continue
        lbl = label_el.get_text(strip=True).lower().rstrip(":")
        # Value = everything in the li after stripping the label text
        full_text = li.get_text(strip=True)
        val = full_text[len(label_el.get_text(strip=True)):].strip()

        if "height" in lbl:
            info["height_cms"] = _parse_height(val)
        elif "weight" in lbl:
            info["weight_lbs"] = _parse_weight(val)
        elif "reach" in lbl:
            info["reach_cms"] = _parse_reach(val)
        elif "stance" in lbl:
            info["stance"] = val if val not in ("--", "---", "") else None
        elif "dob" in lbl or "date of birth" in lbl:
            info["dob"] = _parse_dob(val)

    return info


def scrape_all_fighter_pages(df):
    """
    Scrape each unique fighter URL once.
    Returns a dict: fighter_url → {height_cms, reach_cms, weight_lbs, stance, dob}
    """
    # Build url→name mapping from the df (for progress display)
    url_map = (
        df[["fighter", "fighter_url"]]
        .drop_duplicates("fighter_url")
        .dropna(subset=["fighter_url"])
    )
    url_map = url_map[url_map["fighter_url"].str.contains("fighter-details", na=False)]

    total = len(url_map)
    print(f"  Scraping {total:,} fighter pages...")

    cache = {}
    for i, (_, row) in enumerate(url_map.iterrows(), 1):
        url  = row["fighter_url"]
        name = row["fighter"]
        if url in cache:
            continue
        info = scrape_fighter_page(url)
        cache[url] = info
        if i % 100 == 0 or i == total:
            print(f"    {i:,}/{total:,} done")

    hits = sum(1 for v in cache.values() if v)
    print(f"  ✓ Fighter pages: {hits:,}/{total:,} with data")
    return cache


# ── Self-derived columns (computed from UFCStats data alone) ──────────────────
#
# These replace ALL non-odds columns that previously came from ufc-master.csv.
# Computed after scraping is complete — no data leakage (each fight uses only
# information available BEFORE that fight date).
#
# Fighter-level (per-row, pre-fight state):
#   height_cms, reach_cms, weight_lbs, stance  ← fighter page
#   age                                         ← DOB from fighter page + event_date
#   wins, losses, draws                         ← rolling count of prior results
#   current_win_streak, current_lose_streak,
#   longest_win_streak                          ← rolling streaks
#   wins_by_ko, wins_by_sub,
#   wins_by_dec_unanimous, wins_by_dec_split,
#   wins_by_dec_majority, wins_by_tko_doctor    ← rolling method counts
#   total_rounds_fought                         ← rolling sum of finish_round
#   total_title_bouts                           ← rolling count of title_bout==1
#
# Fight-level (same for both fighters in a fight):
#   number_of_rounds   ← parsed from time_format ("5 Rnd (5-5-5-5-5)" → 5)
#   empty_arena        ← event_date between 2020-03-14 and 2021-07-09
#   finish_details     ← already in fight page (method detail line) — kept as-is
#   total_fight_time_secs ← (finish_round - 1) * round_duration + finish_seconds
#
# Matchup-level differences (computed from the above, fight perspective):
#   height_dif, reach_dif, age_dif, win_dif, loss_dif,
#   win_streak_dif, lose_streak_dif, longest_win_streak_dif,
#   ko_dif, sub_dif, total_round_dif, total_title_bout_dif,
#   sig_str_dif, avg_sub_att_dif, avg_td_dif

# COVID empty-arena date range (Fight Island + Apex without fans)
EMPTY_ARENA_START = datetime(2020, 3, 14)
EMPTY_ARENA_END   = datetime(2021, 7,  9)


def _parse_number_of_rounds(time_fmt):
    """
    Parse scheduled rounds from time_format string.
    '5 Rnd (5-5-5-5-5)' → 5
    '3 Rnd (5-5-5)'     → 3
    Falls back to 3 if parse fails.
    """
    if not time_fmt:
        return 3
    m = re.match(r"(\d+)\s+Rnd", str(time_fmt).strip())
    if m:
        return int(m.group(1))
    return 3


def _finish_time_to_secs(time_str):
    """Convert 'M:SS' finish time string to seconds. Returns 0 on failure."""
    if not time_str:
        return 0
    m = re.match(r"(\d+):(\d+)", str(time_str).strip())
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return 0


def _method_category(method_str):
    """Classify a method string into one of the win-type buckets."""
    if not method_str:
        return "other"
    m = str(method_str).lower()
    if "tko" in m and "doctor" in m:
        return "tko_doctor"
    if "ko" in m or "tko" in m:
        return "ko"
    if "sub" in m:
        return "sub"
    if "u-dec" in m or "unanimous" in m:
        return "dec_unanimous"
    if "s-dec" in m or "split" in m:
        return "dec_split"
    if "m-dec" in m or "majority" in m:
        return "dec_majority"
    if "dec" in m:
        return "dec_unanimous"   # default dec bucket
    return "other"


def compute_self_derived_cols(df, fighter_cache):
    """
    Add all self-derived columns to df in-place (returns new df).

    Two passes:
      Pass 1 — attach height/reach/weight/stance/dob from fighter_cache,
               compute fight-level cols (number_of_rounds, empty_arena,
               total_fight_time_secs).
      Pass 2 — compute rolling pre-fight record stats per fighter (leak-free).
      Pass 3 — compute matchup diff columns.
    """
    df = df.copy()
    df["event_date_dt"] = pd.to_datetime(df["event_date"], errors="coerce")

    # ── Pass 1A: physical attributes from fighter pages ───────────────────────
    phys_cols = ["height_cms", "reach_cms", "weight_lbs", "stance", "age"]
    for col in phys_cols:
        if col not in df.columns:
            df[col] = np.nan if col != "stance" else None

    # Build name→url lookup (some fighters share names — url is the safe key)
    url_col = "fighter_url" if "fighter_url" in df.columns else None

    for idx, row in df.iterrows():
        furl = row.get("fighter_url", "") if url_col else ""
        info = fighter_cache.get(furl, {})
        if not info:
            continue

        if pd.isna(df.at[idx, "height_cms"]) or df.at[idx, "height_cms"] is None:
            df.at[idx, "height_cms"] = info.get("height_cms", np.nan)
        if pd.isna(df.at[idx, "reach_cms"]) or df.at[idx, "reach_cms"] is None:
            df.at[idx, "reach_cms"]  = info.get("reach_cms",  np.nan)
        if pd.isna(df.at[idx, "weight_lbs"]) or df.at[idx, "weight_lbs"] is None:
            df.at[idx, "weight_lbs"] = info.get("weight_lbs", np.nan)

        stance_val = info.get("stance")
        if stance_val and (df.at[idx, "stance"] is None or
                           (isinstance(df.at[idx, "stance"], float) and
                            np.isnan(df.at[idx, "stance"]))):
            df.at[idx, "stance"] = stance_val

        dob = info.get("dob")
        if dob and not pd.isna(row["event_date_dt"]):
            age_years = (row["event_date_dt"] - dob).days / 365.25
            df.at[idx, "age"] = round(age_years, 2)

    # ── Pass 1B: fight-level derived cols ─────────────────────────────────────
    df["number_of_rounds"] = df["time_format"].apply(_parse_number_of_rounds)

    df["empty_arena"] = df["event_date_dt"].apply(
        lambda d: 1 if (pd.notna(d) and EMPTY_ARENA_START <= d <= EMPTY_ARENA_END) else 0
    )

    # total_fight_time_secs: (finish_round - 1) * round_minutes*60 + finish_seconds
    # Round duration is always 5 minutes in UFC
    def _calc_fight_time(row):
        r = row.get("finish_round")
        t = row.get("finish_time")
        if r is None or pd.isna(r):
            return np.nan
        return (int(r) - 1) * 300 + _finish_time_to_secs(t)

    df["total_fight_time_secs"] = df.apply(_calc_fight_time, axis=1)

    # finish_details is now scraped directly in parse_fight_page from the "Details:" label

    # ── Pass 2: rolling pre-fight record stats ────────────────────────────────
    record_cols = [
        "wins", "losses", "draws",
        "current_win_streak", "current_lose_streak", "longest_win_streak",
        "wins_by_ko", "wins_by_sub",
        "wins_by_dec_unanimous", "wins_by_dec_split",
        "wins_by_dec_majority", "wins_by_tko_doctor",
        "total_rounds_fought", "total_title_bouts",
    ]
    for col in record_cols:
        df[col] = np.nan

    df = df.sort_values(["fighter", "event_date_dt"]).reset_index(drop=True)

    for fighter, grp in df.groupby("fighter"):
        grp = grp.sort_values("event_date_dt")

        wins = losses = draws = 0
        cur_win = cur_lose = longest_win = 0
        ko = sub = dec_u = dec_s = dec_m = tko_doc = 0
        rounds_fought = title_bouts = 0

        for i, row in grp.iterrows():
            # Record the PRE-FIGHT state (before this fight)
            df.at[i, "wins"]                  = wins
            df.at[i, "losses"]                = losses
            df.at[i, "draws"]                 = draws
            df.at[i, "current_win_streak"]    = cur_win
            df.at[i, "current_lose_streak"]   = cur_lose
            df.at[i, "longest_win_streak"]    = longest_win
            df.at[i, "wins_by_ko"]            = ko
            df.at[i, "wins_by_sub"]           = sub
            df.at[i, "wins_by_dec_unanimous"] = dec_u
            df.at[i, "wins_by_dec_split"]     = dec_s
            df.at[i, "wins_by_dec_majority"]  = dec_m
            df.at[i, "wins_by_tko_doctor"]    = tko_doc
            df.at[i, "total_rounds_fought"]   = rounds_fought
            df.at[i, "total_title_bouts"]     = title_bouts

            # Update accumulators AFTER recording (no leakage)
            r = row.get("finish_round")
            if r is not None and not (isinstance(r, float) and np.isnan(r)):
                rounds_fought += int(r)
            if row.get("title_bout") == 1:
                title_bouts += 1

            if row.get("won") == 1:
                wins    += 1
                cur_win += 1
                cur_lose = 0
                longest_win = max(longest_win, cur_win)
                cat = _method_category(row.get("method", ""))
                if cat == "ko":           ko     += 1
                elif cat == "sub":        sub    += 1
                elif cat == "dec_unanimous": dec_u += 1
                elif cat == "dec_split":  dec_s  += 1
                elif cat == "dec_majority": dec_m += 1
                elif cat == "tko_doctor": tko_doc += 1
            else:
                result_str = str(row.get("won", "")).strip()
                if result_str == "0":
                    losses  += 1
                    cur_lose += 1
                    cur_win  = 0
                else:
                    draws += 1

    # ── Pass 3: matchup diff columns ──────────────────────────────────────────
    # Build a lookup: (fight_url, fighter) → row index for fast pairing
    df = df.sort_values("fight_url").reset_index(drop=True)

    diff_map = {
        "height_dif":             "height_cms",
        "reach_dif":              "reach_cms",
        "age_dif":                "age",
        "win_dif":                "wins",
        "loss_dif":               "losses",
        "win_streak_dif":         "current_win_streak",
        "lose_streak_dif":        "current_lose_streak",
        "longest_win_streak_dif": "longest_win_streak",
        "ko_dif":                 "wins_by_ko",
        "sub_dif":                "wins_by_sub",
        "total_round_dif":        "total_rounds_fought",
        "total_title_bout_dif":   "total_title_bouts",
    }
    for col in diff_map:
        df[col] = np.nan

    # sig_str_dif / avg_sub_att_dif / avg_td_dif — compute from our per-fight
    # rolling stats.  We use the cumulative sig_str_landed / sig_str_att and
    # sub_att / td_landed averages from prior fights.  Since we don't have these
    # rolling averages directly in fight_stats (they're computed in 03_train_model),
    # we approximate here using per-fight values normalised by rounds.
    # These will be NaN for debut fights (no prior data).
    for col in ["sig_str_dif", "avg_sub_att_dif", "avg_td_dif"]:
        df[col] = np.nan

    for fight_url, grp in df.groupby("fight_url"):
        if len(grp) != 2:
            continue
        idx_a, idx_b = grp.index[0], grp.index[1]
        row_a, row_b = df.loc[idx_a], df.loc[idx_b]

        for diff_col, src_col in diff_map.items():
            try:
                va = float(row_a[src_col])
                vb = float(row_b[src_col])
                df.at[idx_a, diff_col] =  (va - vb) + 0.0
                df.at[idx_b, diff_col] = -(va - vb) + 0.0
            except (TypeError, ValueError):
                pass

        # Approximate sig_str_dif: per-fight sig_str_landed diff
        # (master uses career avg per min — we use per-fight total as proxy)
        for diff_col, src_col in [
            ("sig_str_dif",    "sig_str_landed"),
            ("avg_sub_att_dif","sub_att"),
            ("avg_td_dif",     "td_landed"),
        ]:
            try:
                va = float(row_a.get(src_col, np.nan))
                vb = float(row_b.get(src_col, np.nan))
                if not np.isnan(va) and not np.isnan(vb):
                    df.at[idx_a, diff_col] =  (va - vb)
                    df.at[idx_b, diff_col] = -(va - vb)
            except (TypeError, ValueError):
                pass

    df = df.drop(columns=["event_date_dt"], errors="ignore")
    print(f"  ✓ Self-derived cols computed. Height coverage: "
          f"{df['height_cms'].notna().sum():,}/{len(df):,} rows")
    return df


# ── ufc-master.csv merge ──────────────────────────────────────────────────────
#
# Columns pulled from master (all odds + useful attributes not in UFCStats):
#   Moneyline : RedOdds, BlueOdds  → historical_odds_red, historical_odds_blue
#   Props     : RedDecOdds, BlueDecOdds, RSubOdds, BSubOdds, RKOOdds, BKOOdds
#   EV        : RedExpectedValue, BlueExpectedValue
#   Fighter   : HeightCms, ReachCms, WeightLbs, Stance, Age → for both fighters
#   Streak    : CurrentWinStreak, CurrentLoseStreak, LongestWinStreak
#   Record    : Wins, Losses, Draws, WinsByKO, WinsBySubmission, WinsByDecision*
#   Rounds    : TotalRoundsFought, TotalTitleBouts, NumberOfRounds
#   Fight     : EmptyArena, TitleBout, FinishDetails, FinishRound, FinishRoundTime
#               TotalFightTimeSecs
#
# Columns we derive as DIFFERENCES from master (for model use):
#   HeightDif, ReachDif, AgeDif already in master — we keep those too
#   WinDif, LossDif, etc. already present in master as diff columns
#
# Note: master uses Red/Blue framing; our data uses fighter/opponent framing.
# We match on (RedFighter==fighter AND event_date within ±3 days) OR
# (BlueFighter==fighter AND event_date within ±3 days), then assign
# columns from the correct perspective.

MASTER_ODDS_COLS = [
    # Moneyline
    "historical_odds",          # fighter's moneyline (American)
    "historical_odds_opp",      # opponent's moneyline
    "expected_value",           # EV for this fighter's bet
    # Props
    "dec_odds",                 # fighter wins by decision odds
    "dec_odds_opp",
    "sub_odds",                 # fighter wins by submission odds
    "sub_odds_opp",
    "ko_odds",                  # fighter wins by KO/TKO odds
    "ko_odds_opp",
]

# All non-odds columns are now self-derived (see compute_self_derived_cols).
# These lists are kept for backward compatibility with 02_scrape_weekly_update.py
# which imports the same column names.
MASTER_FIGHTER_COLS = []
MASTER_FIGHT_COLS   = []
MASTER_DIFF_COLS    = []


def _norm_name(s):
    """Lowercase, strip, collapse whitespace for fighter name matching."""
    return re.sub(r'\s+', ' ', str(s).lower().strip())


def load_master_csv(path="ufc-master.csv"):
    """
    Load ufc-master.csv and extract ONLY the 9 odds columns.
    All other attributes (height, record, streaks, etc.) are now self-derived
    from UFCStats data in compute_self_derived_cols().

    Returns a flat DataFrame with one row per fighter per fight,
    or an empty DataFrame if the file is not found.
    """
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — odds columns will be NaN")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return pd.DataFrame()

    print(f"  Loaded ufc-master.csv: {len(df):,} rows, {len(df.columns)} columns")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    records = []
    for _, row in df.iterrows():
        for color, opp_color in [("Red", "Blue"), ("Blue", "Red")]:
            r = {
                "fighter_norm":        _norm_name(row.get(f"{color}Fighter", "")),
                "event_date":          row["Date"],
                # ── Odds only ────────────────────────────────────────────────
                "historical_odds":     _safe_float(row.get(f"{color}Odds")),
                "historical_odds_opp": _safe_float(row.get(f"{opp_color}Odds")),
                "expected_value":      _safe_float(row.get(f"{color}ExpectedValue")),
                "dec_odds":     _safe_float(row.get(f"{color[0]}DecOdds")),
                "dec_odds_opp": _safe_float(row.get(f"{opp_color[0]}DecOdds")),
                "sub_odds":     _safe_float(row.get(f"{color[0]}SubOdds")),
                "sub_odds_opp": _safe_float(row.get(f"{opp_color[0]}SubOdds")),
                "ko_odds":      _safe_float(row.get(f"{color[0]}KOOdds")),
                "ko_odds_opp":  _safe_float(row.get(f"{opp_color[0]}KOOdds")),
            }
            records.append(r)

    master_flat = pd.DataFrame(records)
    master_flat["event_date"] = pd.to_datetime(master_flat["event_date"], errors="coerce")
    master_flat = master_flat.dropna(subset=["fighter_norm", "event_date"])
    print(f"  Master flat: {len(master_flat):,} fighter-fight odds rows")
    return master_flat


def _safe_float(v):
    """Convert to float, returning np.nan on failure."""
    try:
        f = float(v)
        return f if not np.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def merge_master_csv(fight_df, master_flat):
    """
    Merge ONLY the 9 odds columns from ufc-master.csv into fight_df.
    Matching: fighter_norm + ±3-day date window → exact name → last-name fallback.
    All non-odds columns are handled by compute_self_derived_cols().
    """
    for col in MASTER_ODDS_COLS:
        if col not in fight_df.columns:
            fight_df[col] = np.nan

    if master_flat.empty:
        return fight_df

    fdf = fight_df.copy()
    fdf["_f_norm"] = fdf["fighter"].apply(_norm_name)
    fdf["_date"]   = pd.to_datetime(fdf["event_date"], errors="coerce")

    matched = 0
    for idx, row in fdf.iterrows():
        fname = row["_f_norm"]
        fdate = row["_date"]

        window = master_flat[
            (master_flat["event_date"] >= fdate - pd.Timedelta(days=3)) &
            (master_flat["event_date"] <= fdate + pd.Timedelta(days=3))
        ]
        hit = window[window["fighter_norm"] == fname]
        if hit.empty:
            last = fname.split()[-1] if " " in fname else fname
            hit  = window[window["fighter_norm"].str.contains(last, na=False, regex=False)]
        if hit.empty:
            continue

        src = hit.iloc[0]
        for col in MASTER_ODDS_COLS:
            if col in src.index and pd.isna(fdf.at[idx, col]):
                fdf.at[idx, col] = src[col]
        matched += 1

    fdf = fdf.drop(columns=["_f_norm", "_date"], errors="ignore")
    print(f"  Odds merge: {matched:,}/{len(fdf):,} rows matched "
          f"({matched/len(fdf):.1%}), "
          f"{fdf['historical_odds'].notna().sum():,} moneylines, "
          f"{fdf['ko_odds'].notna().sum():,} KO props")
    return fdf


# ── rankings_history.csv merge ────────────────────────────────────────────────
#
# For each fighter-fight row, find the most recent ranking snapshot BEFORE
# the fight date and attach:
#   fighter_rank_wc     : rank in their weight class (0 = champion, NaN = unranked)
#   fighter_rank_pfp    : pound-for-pound rank (NaN if not ranked)
#
# Weight class name matching: rankings_history uses "Light Heavyweight" etc.
# Our fight_stats uses weight_class from UFCStats (similar format).

# Map UFCStats weight class strings → rankings_history weightclass strings
WC_MAP = {
    "Strawweight":           "Strawweight",
    "Women's Strawweight":   "Women's Strawweight",
    "Flyweight":             "Flyweight",
    "Women's Flyweight":     "Women's Flyweight",
    "Bantamweight":          "Bantamweight",
    "Women's Bantamweight":  "Women's Bantamweight",
    "Featherweight":         "Featherweight",
    "Women's Featherweight": "Women's Featherweight",
    "Lightweight":           "Lightweight",
    "Welterweight":          "Welterweight",
    "Middleweight":          "Middleweight",
    "Light Heavyweight":     "Light Heavyweight",
    "Heavyweight":           "Heavyweight",
}


def load_rankings_history(path="rankings_history.csv"):
    """
    Load rankings_history.csv.
    Returns DataFrame with columns: date (datetime), weightclass, fighter_norm, rank (int)
    """
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — skipping rankings merge")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=None, engine="python")  # handles tab or comma
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return pd.DataFrame()

    # Normalise columns (handles both tab-sep and comma-sep layouts)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "fighter" not in df.columns:
        print(f"  [WARN] rankings_history.csv missing expected columns: {list(df.columns)}")
        return pd.DataFrame()

    df["date"]         = pd.to_datetime(df["date"], errors="coerce")
    df["fighter_norm"] = df["fighter"].apply(_norm_name)
    df["rank"]         = pd.to_numeric(df.get("rank", df.get("ranking", np.nan)), errors="coerce")
    df["weightclass"]  = df.get("weightclass", df.get("weight_class", "")).astype(str).str.strip()

    df = df.dropna(subset=["date", "fighter_norm", "rank"])
    print(f"  Loaded rankings_history.csv: {len(df):,} rows, "
          f"{df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
    return df


def merge_rankings(fight_df, rankings_df):
    """
    For each fighter-fight row, attach their most recent ranking snapshot
    BEFORE the fight date in their weight class AND in PFP.

    Adds columns:
      fighter_rank_wc    (int or NaN — 0 = champion, 1-15 = ranked, NaN = unranked)
      fighter_rank_pfp   (int or NaN)
      diff_rank_wc       (fighter_rank_wc - opponent_rank_wc, NaN if either missing)
      diff_rank_pfp      (fighter_rank_pfp - opp_rank_pfp)
    """
    if rankings_df.empty:
        for col in ["fighter_rank_wc", "fighter_rank_pfp",
                    "diff_rank_wc", "diff_rank_pfp"]:
            fight_df[col] = np.nan
        return fight_df

    fdf = fight_df.copy()
    fdf["_f_norm"] = fdf["fighter"].apply(_norm_name)
    fdf["_o_norm"] = fdf["opponent"].apply(_norm_name)
    fdf["_date"]   = pd.to_datetime(fdf["event_date"], errors="coerce")
    fdf["_wc"]     = fdf["weight_class"].apply(
        lambda x: WC_MAP.get(str(x).strip(), str(x).strip())
    )

    pfp = rankings_df[rankings_df["weightclass"] == "Pound-for-Pound"].copy()
    wc  = rankings_df[rankings_df["weightclass"] != "Pound-for-Pound"].copy()

    def get_rank(name_norm, fight_date, wc_str, df_subset):
        """Most recent rank for fighter in given division before fight date."""
        past = df_subset[
            (df_subset["fighter_norm"] == name_norm) &
            (df_subset["date"] <= fight_date)
        ]
        if past.empty:
            # Try last-name only
            last = name_norm.split()[-1] if " " in name_norm else name_norm
            past = df_subset[
                (df_subset["fighter_norm"].str.contains(last, na=False, regex=False)) &
                (df_subset["date"] <= fight_date)
            ]
            if wc_str and not past.empty:
                wc_match = past[past["weightclass"] == wc_str]
                if not wc_match.empty:
                    past = wc_match
        if past.empty:
            return np.nan
        # Most recent snapshot
        latest_date = past["date"].max()
        latest = past[past["date"] == latest_date]
        # If weight class provided, filter to it
        if wc_str:
            wc_latest = latest[latest["weightclass"] == wc_str]
            if not wc_latest.empty:
                return float(wc_latest["rank"].iloc[0])
        return float(latest["rank"].iloc[0])

    rank_wc_vals  = []
    rank_pfp_vals = []

    for _, row in fdf.iterrows():
        fname = row["_f_norm"]
        fdate = row["_date"]
        fwc   = row["_wc"]

        rank_wc_vals.append(get_rank(fname, fdate, fwc, wc))
        rank_pfp_vals.append(get_rank(fname, fdate, None, pfp))

    fdf["fighter_rank_wc"]  = rank_wc_vals
    fdf["fighter_rank_pfp"] = rank_pfp_vals

    # Compute diff vs opponent (need a join)
    fdf["_rank_wc"]  = rank_wc_vals
    fdf["_rank_pfp"] = rank_pfp_vals

    # Build opponent rank lookup from the same data
    opp_wc_lookup  = dict(zip(zip(fdf["_o_norm"], fdf["fight_url"]), fdf["_rank_wc"]))
    opp_pfp_lookup = dict(zip(zip(fdf["_o_norm"], fdf["fight_url"]), fdf["_rank_pfp"]))

    diff_wc  = []
    diff_pfp = []
    for _, row in fdf.iterrows():
        f_wc   = row["_rank_wc"]
        o_wc   = opp_wc_lookup.get((row["_f_norm"], row["fight_url"]), np.nan)
        f_pfp  = row["_rank_pfp"]
        o_pfp  = opp_pfp_lookup.get((row["_f_norm"], row["fight_url"]), np.nan)

        diff_wc.append(
            (f_wc - o_wc) if (not np.isnan(f_wc) and not np.isnan(o_wc)) else np.nan
        )
        diff_pfp.append(
            (f_pfp - o_pfp) if (not np.isnan(f_pfp) and not np.isnan(o_pfp)) else np.nan
        )

    fdf["diff_rank_wc"]  = diff_wc
    fdf["diff_rank_pfp"] = diff_pfp

    fdf = fdf.drop(columns=["_f_norm", "_o_norm", "_date", "_wc",
                              "_rank_wc", "_rank_pfp"], errors="ignore")

    n_wc  = fdf["fighter_rank_wc"].notna().sum()
    n_pfp = fdf["fighter_rank_pfp"].notna().sum()
    print(f"  Rankings merge: {n_wc:,}/{len(fdf):,} WC ranks, "
          f"{n_pfp:,}/{len(fdf):,} PFP ranks attached")
    return fdf


# ── UFC.com rankings scraper (for weekly update use) ─────────────────────────
#
# ufc.com/rankings page structure (from your XPaths):
#   Each division = a table block
#   Champion : caption > div > div[1] > h5 > a  (text = fighter name)
#   Rank 1   : tbody > tr[1] > td[2] > a
#   Rank N   : tbody > tr[N] > td[2] > a
#   Divisions appear in document order: Flyweight, Bantamweight, ..., Heavyweight,
#   then Women's, then P4P Men, P4P Women at the bottom.

UFC_RANKINGS_URL = "https://www.ufc.com/rankings"

# Map div index (0-based) on the rankings page to weight class name
# Order matches the actual page layout — verify if UFC changes it
UFC_RANKINGS_DIV_ORDER = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight",
    "Pound-for-Pound",          # Men's P4P
    "Women's Pound-for-Pound",  # Women's P4P
]


def scrape_ufc_rankings():
    """
    Scrape current UFC rankings from ufc.com/rankings.
    Returns DataFrame: date (today), weightclass, fighter, rank (0=champion)
    Returns empty DataFrame on failure.
    """
    ufc_session = requests.Session()
    ufc_session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })

    try:
        resp = ufc_session.get(UFC_RANKINGS_URL, timeout=20)
        if resp.status_code != 200:
            print(f"  [WARN] UFC rankings: HTTP {resp.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  [WARN] UFC rankings fetch failed: {e}")
        return pd.DataFrame()

    soup  = BeautifulSoup(resp.text, "lxml")
    today = datetime.now().strftime("%Y-%m-%d")
    rows  = []

    # Each division block is a <div class="view-grouping"> or similar
    # Most reliable: find all <table> elements — each is one division
    tables = soup.select("table")

    for t_idx, table in enumerate(tables):
        if t_idx >= len(UFC_RANKINGS_DIV_ORDER):
            break
        wc = UFC_RANKINGS_DIV_ORDER[t_idx]

        # Champion: <caption> → h5 > a  OR  caption > div > div > h5 > a
        champion = None
        caption  = table.find("caption")
        if caption:
            champ_link = caption.select_one("h5 a")
            if champ_link:
                champion = champ_link.get_text(strip=True)

        if champion:
            rows.append({
                "date": today, "weightclass": wc,
                "fighter": champion, "rank": 0
            })

        # Ranked fighters: tbody > tr > td[1] a  (index 1 = name col, index 0 = rank #)
        for tr in table.select("tbody tr"):
            tds      = tr.select("td")
            rank_td  = tds[0] if len(tds) > 0 else None
            name_td  = tds[1] if len(tds) > 1 else None
            if rank_td is None or name_td is None:
                continue
            try:
                rank_num = int(rank_td.get_text(strip=True))
            except ValueError:
                continue
            name_link = name_td.find("a")
            if not name_link:
                continue
            fighter_name = name_link.get_text(strip=True)
            if fighter_name:
                rows.append({
                    "date": today, "weightclass": wc,
                    "fighter": fighter_name, "rank": rank_num
                })

    if not rows:
        print("  [WARN] No rankings found — ufc.com layout may have changed")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"  ✓ UFC rankings scraped: {len(df)} entries, "
          f"{df['weightclass'].nunique()} divisions")
    return df


# ── Odds imputation ───────────────────────────────────────────────────────────
#
# American odds are discontinuous (-110 → +100 skips through zero), so we work
# entirely in implied probability space [0, 1].
#
# Moneyline implied prob:
#   Train a calibrated LogisticRegression on the ~86% of rows where we have
#   historical_odds. Features = numeric diff cols + weight-class dummies.
#   Predict implied_prob for ALL rows (including those with known odds, so the
#   column is always complete). For rows with known odds, the original market
#   price is preserved in historical_odds; implied_prob_market comes from the
#   model for consistency across all rows.
#
# Prop implied probs (ko / sub / dec):
#   Sparse data → use a two-stage approach:
#     Stage 1: compute historical method rates per weight class (strong prior)
#     Stage 2: logistic adjustment using fighter-level KO/sub/dec tendencies
#   This gives stable estimates even for fighters with few fights.
#
# All implied probs are stored as new columns:
#   implied_prob          — P(fighter wins) from moneyline model
#   implied_prob_ko       — P(fighter wins by KO/TKO)
#   implied_prob_sub      — P(fighter wins by submission)
#   implied_prob_dec      — P(fighter wins by decision)
#
# American odds back-conversion (for display / downstream use):
#   american_odds_from_prob(p) → standard American integer odds

def _american_to_prob(odds):
    """Convert American moneyline odds to implied probability (no vig removed)."""
    try:
        o = float(odds)
        if o >= 100:
            return 100 / (o + 100)
        else:
            return abs(o) / (abs(o) + 100)
    except (TypeError, ValueError):
        return np.nan


def american_odds_from_prob(p):
    """Convert implied probability back to American odds (rounded to nearest 5)."""
    try:
        p = float(p)
        p = max(0.01, min(0.99, p))
        if p >= 0.5:
            raw = -(p / (1 - p)) * 100
        else:
            raw = ((1 - p) / p) * 100
        # Round to nearest 5 (standard odds increment)
        return int(round(raw / 5) * 5)
    except (TypeError, ValueError):
        return np.nan


# Features used for the moneyline implied-prob model
# These are all available at fight time (no leakage) and numeric
_ODDS_MODEL_FEATURES = [
    "height_dif", "reach_dif", "age_dif",
    "win_dif", "loss_dif",
    "win_streak_dif", "lose_streak_dif", "longest_win_streak_dif",
    "ko_dif", "sub_dif",
    "total_round_dif", "total_title_bout_dif",
    "sig_str_dif", "avg_sub_att_dif", "avg_td_dif",
]

# Weight class encoding for the odds model (used as additional feature)
_WC_ORDER = [
    "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
    "Lightweight", "Welterweight", "Middleweight",
    "Light Heavyweight", "Heavyweight",
    "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
]
_WC_TO_IDX = {wc: i for i, wc in enumerate(_WC_ORDER)}


def _build_feature_matrix(df, features=_ODDS_MODEL_FEATURES):
    """
    Build a numeric feature matrix from df.
    Adds weight_class_idx as an extra feature.
    Returns (X np.array, valid_mask bool array).
    """
    X_parts = []
    for col in features:
        if col in df.columns:
            X_parts.append(df[col].values.reshape(-1, 1))
        else:
            X_parts.append(np.zeros((len(df), 1)))

    # Weight class as ordinal index (roughly ascending by size/power)
    wc_idx = df["weight_class"].map(_WC_TO_IDX).fillna(4).values.reshape(-1, 1)
    X_parts.append(wc_idx)

    X = np.hstack(X_parts).astype(float)
    # Valid mask: row has no NaN in any feature
    valid_mask = ~np.isnan(X).any(axis=1)
    # Impute remaining NaNs with column median for prediction rows
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nan_rows = np.isnan(X[:, j])
        X[nan_rows, j] = col_medians[j]

    return X, valid_mask


def impute_odds(df):
    """
    Add implied probability columns to df, filling all rows (including those
    with known odds). Returns df with new columns added.

    New columns:
        implied_prob          P(fighter wins) — moneyline model
        implied_prob_ko       P(fighter wins by KO/TKO)
        implied_prob_sub      P(fighter wins by submission)
        implied_prob_dec      P(fighter wins by decision)
        implied_prob_source   'market' | 'model' (for implied_prob)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    df = df.copy()

    # ── 1. Convert known American odds to implied probability ─────────────────
    df["_ip_raw"] = df["historical_odds"].apply(_american_to_prob)

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    X, valid_mask = _build_feature_matrix(df)

    # ── 3. Train moneyline model on rows with known odds ──────────────────────
    train_mask = valid_mask & df["_ip_raw"].notna()
    n_train = train_mask.sum()
    print(f"  Moneyline model: {n_train:,} training rows "
          f"({n_train/len(df):.1%} of dataset)")

    # Target: implied probability from market odds (not raw win/loss)
    # We binarise at 0.5 for the classifier, then use predict_proba
    # which gives us calibrated probabilities matching the market's scale
    y_train = (df.loc[train_mask, "_ip_raw"] > 0.5).astype(int).values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            method="isotonic", cv=5
        )),
    ])
    pipe.fit(X[train_mask], y_train)

    # Predict for ALL rows
    prob_all = pipe.predict_proba(X)[:, 1]

    # ── 4. Fill implied_prob ──────────────────────────────────────────────────
    # Use market price where available, model where not
    df["implied_prob"] = np.where(
        df["_ip_raw"].notna(),
        df["_ip_raw"],          # keep exact market price
        prob_all                # model prediction
    )
    df["implied_prob_source"] = np.where(
        df["_ip_raw"].notna(), "market", "model"
    )

    n_market = (df["implied_prob_source"] == "market").sum()
    n_model  = (df["implied_prob_source"] == "model").sum()
    print(f"  implied_prob: {n_market:,} market, {n_model:,} model-filled")

    # ── 5. Prop implied probs: KO / Sub / Dec ─────────────────────────────────
    # Method rates by weight class (computed from training data with known methods)
    # Then adjusted by fighter tendency (rolling wins_by_ko / wins / total)

    def _method_rate_col(df, method_wins_col):
        """
        Returns a column of P(fighter wins by this method | fight happens).
        = (rolling method wins / rolling total wins) * P(fighter wins)
        Falls back to weight-class average where no history.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            # Fraction of wins by this method (career tendency)
            fighter_rate = np.where(
                df["wins"] > 0,
                df[method_wins_col] / df["wins"],
                np.nan
            )

        # Weight-class average rate (from rows with known method)
        method_known = df["method"].notna()
        wc_rates = {}
        for wc in df["weight_class"].unique():
            wc_mask = method_known & (df["weight_class"] == wc)
            if wc_mask.sum() < 10:
                continue
            wc_df = df[wc_mask]
            # Rate among winners
            winner_mask = wc_df["won"] == 1
            if winner_mask.sum() == 0:
                continue
            wc_rate = wc_df[winner_mask][method_wins_col].sum() / winner_mask.sum()
            wc_rates[wc] = wc_rate

        overall_rate = np.nanmean(list(wc_rates.values())) if wc_rates else 0.3
        wc_avg = df["weight_class"].map(wc_rates).fillna(overall_rate).values

        # Blend: 70% fighter history, 30% weight-class prior
        # (shrink toward prior for fighters with little history)
        n_wins = df["wins"].fillna(0).values
        alpha = np.minimum(n_wins / 10.0, 1.0)  # full fighter weight at 10+ wins
        blended_rate = alpha * np.where(np.isnan(fighter_rate), wc_avg, fighter_rate) \
                     + (1 - alpha) * wc_avg

        # P(wins by method) = blended_rate * P(wins)
        return blended_rate * df["implied_prob"].values

    # KO prop implied probs
    if "ko_odds" in df.columns:
        ip_ko_market = df["ko_odds"].apply(_american_to_prob)
    else:
        ip_ko_market = pd.Series(np.nan, index=df.index)

    ip_ko_model = _method_rate_col(df, "wins_by_ko")
    df["implied_prob_ko"] = np.where(
        ip_ko_market.notna(), ip_ko_market, ip_ko_model
    )

    # Sub prop implied probs
    if "sub_odds" in df.columns:
        ip_sub_market = df["sub_odds"].apply(_american_to_prob)
    else:
        ip_sub_market = pd.Series(np.nan, index=df.index)

    ip_sub_model = _method_rate_col(df, "wins_by_sub")
    df["implied_prob_sub"] = np.where(
        ip_sub_market.notna(), ip_sub_market, ip_sub_model
    )

    # Dec prop implied probs — use wins_by_dec_unanimous + split + majority
    df["_wins_by_dec_total"] = (
        df.get("wins_by_dec_unanimous", 0).fillna(0) +
        df.get("wins_by_dec_split",     0).fillna(0) +
        df.get("wins_by_dec_majority",  0).fillna(0)
    )
    if "dec_odds" in df.columns:
        ip_dec_market = df["dec_odds"].apply(_american_to_prob)
    else:
        ip_dec_market = pd.Series(np.nan, index=df.index)

    ip_dec_model = _method_rate_col(df, "_wins_by_dec_total")
    df["implied_prob_dec"] = np.where(
        ip_dec_market.notna(), ip_dec_market, ip_dec_model
    )

    # Coverage
    for col in ["implied_prob", "implied_prob_ko", "implied_prob_sub", "implied_prob_dec"]:
        pct = df[col].notna().sum() / len(df)
        print(f"  {col:<28} {pct:.1%} coverage")

    # Clean up temp cols
    df = df.drop(columns=["_ip_raw", "_wins_by_dec_total"], errors="ignore")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OctaStats V2 — Historical Data Scraper")
    print(f"Scraping all UFC fights from {CUTOFF_YEAR} to present")
    print("=" * 60)

    # Step 1: UFCStats fight-level scrape
    print("\n[1/6] Scraping UFCStats fight pages...")
    events   = get_all_event_urls(CUTOFF_YEAR)
    all_rows = []
    for event in tqdm(events, desc="Events"):
        fight_urls = get_fight_urls_from_event(event["url"])
        for fight_url in fight_urls:
            fight_rows = parse_fight_page(
                fight_url, event["name"], event["date"], event["location"],
            )
            all_rows.extend(fight_rows)

    if not all_rows:
        print("[ERROR] No data scraped.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values("event_date", ascending=False).reset_index(drop=True)
    print(f"\n✓ UFCStats: {len(df):,} fighter-fight rows, "
          f"{df['fight_url'].nunique():,} fights")

    # Step 2: Scrape fighter pages (height/reach/weight/stance/DOB)
    print("\n[2/6] Scraping UFCStats fighter detail pages...")
    fighter_cache = scrape_all_fighter_pages(df)

    # Step 3: Compute all self-derived columns
    print("\n[3/6] Computing self-derived columns (record, streaks, diffs)...")
    df = compute_self_derived_cols(df, fighter_cache)

    # Step 4: Merge ufc-master.csv (odds only)
    print("\n[4/6] Merging ufc-master.csv (odds columns only)...")
    master_flat = load_master_csv("ufc-master.csv")
    df = merge_master_csv(df, master_flat)

    # Step 5: Merge rankings history
    print("\n[5/6] Merging rankings history...")
    rankings_df = load_rankings_history("rankings_history.csv")
    df = merge_rankings(df, rankings_df)

    # Step 6: Impute missing odds → implied probability columns
    print("\n[6/6] Imputing missing odds (implied probability model)...")
    try:
        df = impute_odds(df)
    except ImportError:
        print("  [WARN] scikit-learn not installed — skipping odds imputation")
        print("  Run: pip install scikit-learn")
        for col in ["implied_prob", "implied_prob_ko",
                    "implied_prob_sub", "implied_prob_dec", "implied_prob_source"]:
            df[col] = np.nan

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/fight_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df):,} rows, {len(df.columns)} columns → {out_path}")

    # Coverage report
    key_cols = ["height_cms", "reach_cms", "stance", "age",
                "wins", "losses", "historical_odds",
                "implied_prob", "implied_prob_ko", "implied_prob_source"]
    print("\n  Coverage report:")
    for col in key_cols:
        if col in df.columns:
            pct = df[col].notna().sum() / len(df)
            print(f"    {col:<28} {pct:.1%}")

    print("\n" + "=" * 60)
    print("NEXT STEPS — upload fight_stats.csv to GitHub manually:")
    print(f"  Repo: https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/main/data/raw")
    print("  Option A: git add data/raw/fight_stats.csv && git commit && git push")
    print("  Option B: drag-and-drop the file via GitHub.com UI")
    print("  Also upload: data/raw/ufc-master.csv, data/raw/rankings_history.csv")
    print("  Then: trigger the Wednesday Actions workflow to retrain the model.")
    print("=" * 60)
    print("\n✓ Done!")


# ── Test mode ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # ── UFCStats scrape diagnostic ────────────────────────────────────────
        print("=== TEST: UFCStats scraper + fighter page ===\n")
        url  = f"{BASE_URL}/statistics/events/completed?page=1"
        resp = _SESSION.get(url, timeout=20)
        print(f"HTTP: {resp.status_code} | Chars: {len(resp.text):,}")
        soup = BeautifulSoup(resp.text, "lxml")

        first_event = None
        for row in soup.select("tr.b-statistics__table-row"):
            if row.find("th"):
                continue
            tds = row.find_all("td")
            if not tds:
                continue
            link = tds[0].select_one("i.b-statistics__table-content a")
            date = tds[0].select_one("span.b-statistics__date")
            if link and date:
                first_event = {
                    "url":      link.get("href", ""),
                    "name":     link.get_text(strip=True),
                    "date":     date.get_text(strip=True),
                    "location": tds[1].get_text(strip=True) if len(tds) > 1 else "",
                }
                break

        if not first_event:
            print("✗ No events found")
        else:
            print(f"✓ First event: {first_event['name']} ({first_event['date']})")
            fight_urls = get_fight_urls_from_event(first_event["url"])
            print(f"✓ Found {len(fight_urls)} fights")
            if fight_urls:
                rows_data = parse_fight_page(
                    fight_urls[0], first_event["name"],
                    first_event["date"], first_event["location"],
                )
                print(f"✓ Fight rows: {len(rows_data)}")
                for frow in rows_data:
                    print(f"\n  {frow['fighter']}:")
                    for k, v in frow.items():
                        if k != "fighter":
                            print(f"    {k:<22} {repr(v)}")

                # ── Fighter page test ─────────────────────────────────────────
                print("\n[Fighter page test]")
                for frow in rows_data:
                    furl = frow.get("fighter_url", "")
                    if furl and "fighter-details" in furl:
                        print(f"\n  Scraping: {frow['fighter']} → {furl}")
                        info = scrape_fighter_page(furl)
                        if info:
                            print(f"    height_cms:  {info.get('height_cms')}")
                            print(f"    reach_cms:   {info.get('reach_cms')}")
                            print(f"    weight_lbs:  {info.get('weight_lbs')}")
                            print(f"    stance:      {info.get('stance')}")
                            print(f"    dob:         {info.get('dob')}")
                            # Spot-check: height should be a reasonable number
                            h = info.get("height_cms", 0) or 0
                            ok = 150 < h < 220
                            print(f"    height range check: {'✓' if ok else '✗'} ({h} cm)")
                        else:
                            print("    ✗ No data returned")
                        break   # test one fighter is enough
        print("\n=== END TEST ===")

    elif "--test-full-card" in sys.argv:
        # ── Full card column coverage test ───────────────────────────────────
        # Scrapes the most recent completed event (all fights + all fighter
        # pages), runs compute_self_derived_cols, then prints a per-column
        # coverage report so you can see exactly what's populated vs missing.
        print("=== TEST: Full card column coverage ===\n")

        # All columns we expect in the final fight_stats.csv
        ALL_EXPECTED_COLS = [
            # ── Fight metadata ────────────────────────────────────────────────
            "fight_url", "event_name", "event_date", "event_location",
            "fighter", "fighter_url", "opponent", "won",
            "weight_class", "title_bout",
            "method", "finish_round", "finish_time", "time_format",
            "referee", "finish_details", "judge_scores", "scorecard_margin",
            # ── Per-round fight stats (from UFCStats totals table) ────────────
            "kd", "sig_str_landed", "sig_str_att", "sig_str_pct",
            "total_str_landed", "total_str_att",
            "td_landed", "td_att", "td_pct",
            "sub_att", "reversals", "ctrl_seconds",
            # ── Significant strike breakdown ──────────────────────────────────
            "head_landed", "head_att", "body_landed", "body_att",
            "leg_landed", "leg_att",
            "distance_landed", "distance_att",
            "clinch_landed", "clinch_att",
            "ground_landed", "ground_att",
            # ── Self-derived: physical (from fighter pages) ───────────────────
            "height_cms", "reach_cms", "weight_lbs", "stance", "age",
            # ── Self-derived: fight-level ─────────────────────────────────────
            "number_of_rounds", "empty_arena", "total_fight_time_secs",
            # ── Self-derived: rolling pre-fight record ────────────────────────
            "wins", "losses", "draws",
            "current_win_streak", "current_lose_streak", "longest_win_streak",
            "wins_by_ko", "wins_by_sub",
            "wins_by_dec_unanimous", "wins_by_dec_split",
            "wins_by_dec_majority", "wins_by_tko_doctor",
            "total_rounds_fought", "total_title_bouts",
            # ── Self-derived: matchup diffs ───────────────────────────────────
            "height_dif", "reach_dif", "age_dif",
            "win_dif", "loss_dif",
            "win_streak_dif", "lose_streak_dif", "longest_win_streak_dif",
            "ko_dif", "sub_dif",
            "total_round_dif", "total_title_bout_dif",
            "sig_str_dif", "avg_sub_att_dif", "avg_td_dif",
            # ── Implied probability (filled for all rows) ─────────────────────
            "implied_prob", "implied_prob_ko", "implied_prob_sub",
            "implied_prob_dec", "implied_prob_source",
            "historical_odds", "historical_odds_opp", "expected_value",
            "dec_odds", "dec_odds_opp",
            "sub_odds", "sub_odds_opp",
            "ko_odds", "ko_odds_opp",
            # ── Rankings (from rankings_history.csv) ─────────────────────────
            "fighter_rank_wc", "fighter_rank_pfp",
            "diff_rank_wc", "diff_rank_pfp",
        ]

        # Step 1: Get most recent event
        print("[1/4] Finding most recent event...")
        url  = f"{BASE_URL}/statistics/events/completed?page=1"
        resp = _SESSION.get(url, timeout=20)
        soup = BeautifulSoup(resp.text, "lxml")

        event_info = None
        for row in soup.select("tr.b-statistics__table-row"):
            if row.find("th"):
                continue
            tds = row.find_all("td")
            if not tds:
                continue
            link = tds[0].select_one("i.b-statistics__table-content a")
            date = tds[0].select_one("span.b-statistics__date")
            if link and date:
                event_info = {
                    "url":      link.get("href", ""),
                    "name":     link.get_text(strip=True),
                    "date":     date.get_text(strip=True),
                    "location": tds[1].get_text(strip=True) if len(tds) > 1 else "",
                }
                break

        if not event_info:
            print("✗ Could not find event")
            sys.exit(1)

        print(f"  Event: {event_info['name']} ({event_info['date']})")

        # Step 2: Scrape all fights on the card
        print("\n[2/4] Scraping all fights on card...")
        fight_urls = get_fight_urls_from_event(event_info["url"])
        print(f"  Found {len(fight_urls)} fights")

        all_rows = []
        for fu in fight_urls:
            rows_data = parse_fight_page(
                fu, event_info["name"], event_info["date"], event_info["location"]
            )
            all_rows.extend(rows_data)

        if not all_rows:
            print("✗ No fight rows scraped")
            sys.exit(1)

        df = pd.DataFrame(all_rows)
        print(f"  ✓ {len(df)} fighter-fight rows ({df['fight_url'].nunique()} fights)")

        # Step 3: Fighter pages
        print("\n[3/4] Scraping fighter pages...")
        fighter_cache = scrape_all_fighter_pages(df)

        # Step 4: Compute self-derived cols
        print("\n[4/4] Computing self-derived columns...")
        df = compute_self_derived_cols(df, fighter_cache)

        # Step 5: Impute odds
        print("\n[5/5] Imputing implied probability columns...")
        # Merge dummy odds cols so imputer has something to work with
        for col in ["historical_odds", "ko_odds", "sub_odds", "dec_odds"]:
            if col not in df.columns:
                df[col] = np.nan
        try:
            df = impute_odds(df)
        except ImportError:
            print("  [WARN] scikit-learn not installed — skipping")
            for col in ["implied_prob","implied_prob_ko","implied_prob_sub",
                        "implied_prob_dec","implied_prob_source"]:
                df[col] = np.nan

        # Add placeholder odds + ranking cols so coverage table is complete
        for col in ["historical_odds", "historical_odds_opp", "expected_value",
                    "dec_odds", "dec_odds_opp", "sub_odds", "sub_odds_opp",
                    "ko_odds", "ko_odds_opp",
                    "fighter_rank_wc", "fighter_rank_pfp",
                    "diff_rank_wc", "diff_rank_pfp"]:
            if col not in df.columns:
                df[col] = np.nan

        # ── Coverage report ───────────────────────────────────────────────────
        print("\n" + "=" * 64)
        print(f"{'COLUMN':<35} {'POPULATED':>10}  {'SAMPLE VALUE'}")
        print("=" * 64)

        # Group columns by section for readability
        sections = {
            "Fight metadata": [
                "fight_url", "event_name", "event_date", "event_location",
                "fighter", "fighter_url", "opponent", "won",
                "weight_class", "title_bout",
                "method", "finish_round", "finish_time", "time_format",
                "referee", "finish_details", "judge_scores", "scorecard_margin",
            ],
            "Fight stats (totals)": [
                "kd", "sig_str_landed", "sig_str_att", "sig_str_pct",
                "total_str_landed", "total_str_att",
                "td_landed", "td_att", "td_pct",
                "sub_att", "reversals", "ctrl_seconds",
            ],
            "Sig strike breakdown": [
                "head_landed", "head_att", "body_landed", "body_att",
                "leg_landed", "leg_att", "distance_landed", "distance_att",
                "clinch_landed", "clinch_att", "ground_landed", "ground_att",
            ],
            "Physical (fighter pg)": [
                "height_cms", "reach_cms", "weight_lbs", "stance", "age",
            ],
            "Fight-level derived": [
                "number_of_rounds", "empty_arena", "total_fight_time_secs",
            ],
            "Rolling record": [
                "wins", "losses", "draws",
                "current_win_streak", "current_lose_streak", "longest_win_streak",
                "wins_by_ko", "wins_by_sub",
                "wins_by_dec_unanimous", "wins_by_dec_split",
                "wins_by_dec_majority", "wins_by_tko_doctor",
                "total_rounds_fought", "total_title_bouts",
            ],
            "Matchup diffs": [
                "height_dif", "reach_dif", "age_dif",
                "win_dif", "loss_dif",
                "win_streak_dif", "lose_streak_dif", "longest_win_streak_dif",
                "ko_dif", "sub_dif",
                "total_round_dif", "total_title_bout_dif",
                "sig_str_dif", "avg_sub_att_dif", "avg_td_dif",
            ],
            "Implied probability": [
                "implied_prob", "implied_prob_ko",
                "implied_prob_sub", "implied_prob_dec", "implied_prob_source",
            ],
            "Odds (needs master)": [
                "historical_odds", "historical_odds_opp", "expected_value",
                "dec_odds", "dec_odds_opp",
                "sub_odds", "sub_odds_opp",
                "ko_odds", "ko_odds_opp",
            ],
            "Rankings (needs hist)": [
                "fighter_rank_wc", "fighter_rank_pfp",
                "diff_rank_wc", "diff_rank_pfp",
            ],
        }

        # Columns only populated for certain fight types (partial coverage is correct)
        DECISION_ONLY_COLS  = {"judge_scores", "scorecard_margin"}
        FINISH_ONLY_COLS    = {"finish_details"}   # only KO/TKO/Sub finishes have detail text

        issues = []
        for section, cols in sections.items():
            print(f"\n  ── {section} ──")
            for col in cols:
                if col not in df.columns:
                    print(f"  {'✗ MISSING':<8} {col:<35} (column not in df)")
                    issues.append(f"MISSING COLUMN: {col}")
                    continue
                n_pop   = df[col].notna().sum()
                n_total = len(df)
                pct     = n_pop / n_total if n_total else 0
                # Sample: first non-null value
                sample = df[col].dropna().iloc[0] if n_pop > 0 else "—"
                if isinstance(sample, float):
                    sample = f"{sample:.3g}"
                sample = str(sample)[:30]

                # Decide pass/warn/fail
                if col in ("historical_odds", "historical_odds_opp", "expected_value",
                           "dec_odds", "dec_odds_opp", "sub_odds", "sub_odds_opp",
                           "ko_odds", "ko_odds_opp",
                           "fighter_rank_wc", "fighter_rank_pfp",
                           "diff_rank_wc", "diff_rank_pfp"):
                    icon = "○"   # expected empty without source files
                    issues.append(f"EXPECTED_EMPTY: {col}")
                elif col in DECISION_ONLY_COLS or col in FINISH_ONLY_COLS:
                    # Partial coverage is correct — only present for certain fight types
                    icon = "~" if pct > 0 else "~"   # always ~ for these
                elif pct == 0:
                    icon = "✗"
                    issues.append(f"ZERO COVERAGE: {col}")
                elif pct < 0.5:
                    icon = "~"   # partial — acceptable for e.g. ctrl_seconds, td_pct
                else:
                    icon = "✓"

                print(f"  {icon}  {col:<35} {n_pop:>3}/{n_total:<3}  {sample}")

        # ── Per-fighter detailed view (first fight only) ──────────────────────
        print("\n" + "=" * 64)
        print("FIRST FIGHT — per-fighter detail")
        print("=" * 64)
        first_fight_url = df["fight_url"].iloc[0]
        fight_rows = df[df["fight_url"] == first_fight_url]
        for _, frow in fight_rows.iterrows():
            print(f"\n  {frow['fighter']} (vs {frow['opponent']}):")
            for col in ALL_EXPECTED_COLS:
                val = frow.get(col, "—MISSING—")
                if isinstance(val, float) and np.isnan(val):
                    val = "NaN"
                print(f"    {col:<35} {repr(str(val)[:50])}")

        # ── Summary ──────────────────────────────────────────────────────────
        print("\n" + "=" * 64)
        real_issues = [i for i in issues if not i.startswith("EXPECTED_EMPTY")]
        if real_issues:
            print(f"⚠  {len(real_issues)} issue(s) found:")
            for i in real_issues:
                print(f"   {i}")
        else:
            print("✓  All self-derived columns populated — ready for full run!")
        print(f"   (Odds + rankings cols will fill once source files are provided)")
        print("\n=== END TEST ===")

    elif "--test-merge" in sys.argv:
        # ── Offline merge logic test (no network needed) ──────────────────────
        print("=== TEST: Master CSV + Rankings merge logic ===\n")

        # ── 1. Simulate fight_df ──────────────────────────────────────────────
        print("[1] Building synthetic fight_df...")
        fight_df = pd.DataFrame([
            {"fight_url": "url1", "fighter": "Alexandre Pantoja",
             "opponent": "Kai Asakura", "event_date": "2024-12-07",
             "weight_class": "Flyweight", "won": 1},
            {"fight_url": "url1", "fighter": "Kai Asakura",
             "opponent": "Alexandre Pantoja", "event_date": "2024-12-07",
             "weight_class": "Flyweight", "won": 0},
            {"fight_url": "url2", "fighter": "Shavkat Rakhmonov",
             "opponent": "Ian Machado Garry", "event_date": "2024-12-07",
             "weight_class": "Welterweight", "won": 1},
            {"fight_url": "url2", "fighter": "Ian Machado Garry",
             "opponent": "Shavkat Rakhmonov", "event_date": "2024-12-07",
             "weight_class": "Welterweight", "won": 0},
            {"fight_url": "url3", "fighter": "Unknown Fighter",
             "opponent": "Also Unknown", "event_date": "2020-01-01",
             "weight_class": "Lightweight", "won": 0},
        ])
        print(f"  {len(fight_df)} rows created\n")

        # ── 2. Test master CSV loading ────────────────────────────────────────
        print("[2] Testing ufc-master.csv load + merge...")
        if os.path.exists("ufc-master.csv"):
            master_flat = load_master_csv("ufc-master.csv")
            result = merge_master_csv(fight_df.copy(), master_flat)
            print(f"\n  Pantoja historical_odds: {result.loc[0,'historical_odds']}")
            print(f"  Pantoja ko_odds:         {result.loc[0,'ko_odds']}")
            print(f"  Pantoja age:             {result.loc[0,'age']}")
            print(f"  Pantoja stance:          {result.loc[0,'stance']}")
            print(f"  Pantoja height_cms:      {result.loc[0,'height_cms']}")
            print(f"  Pantoja reach_cms:       {result.loc[0,'reach_cms']}")
            print(f"  height_dif:              {result.loc[0,'height_dif']}")
            print(f"  age_dif:                 {result.loc[0,'age_dif']}")
            print(f"  Rakhmonov historical_odds: {result.loc[2,'historical_odds']}")
            print(f"  Unknown Fighter odds:    {result.loc[4,'historical_odds']} (expect NaN)")

            # Verify expected values from the sample data you provided
            expected_pantoja_odds = -250.0
            got = result.loc[0, 'historical_odds']
            mark = "✓" if (abs(got - expected_pantoja_odds) < 5 if not pd.isna(got) else False) else "?"
            print(f"\n  {mark} Pantoja ML odds: got {got}, expected ~{expected_pantoja_odds}")
            expected_rakh_odds = -210.0
            got2 = result.loc[2, 'historical_odds']
            mark2 = "✓" if (abs(got2 - expected_rakh_odds) < 5 if not pd.isna(got2) else False) else "?"
            print(f"  {mark2} Rakhmonov ML odds: got {got2}, expected ~{expected_rakh_odds}")
        else:
            print("  ⚠ ufc-master.csv not found in current directory — skipping")
            print("  Place ufc-master.csv in the same folder as this script to test")

        print()

        # ── 3. Test rankings_history loading ─────────────────────────────────
        print("[3] Testing rankings_history.csv load + merge...")
        if os.path.exists("rankings_history.csv"):
            rankings_df = load_rankings_history("rankings_history.csv")
            result2 = merge_rankings(fight_df.copy(), rankings_df)
            print(f"\n  Pantoja rank_wc:   {result2.loc[0,'fighter_rank_wc']}")
            print(f"  Pantoja rank_pfp:  {result2.loc[0,'fighter_rank_pfp']}")
            print(f"  Pantoja diff_rank: {result2.loc[0,'diff_rank_wc']}")
            print(f"  Rakh rank_wc:      {result2.loc[2,'fighter_rank_wc']}")
            print(f"  Unknown rank_wc:   {result2.loc[4,'fighter_rank_wc']} (expect NaN)")
        else:
            print("  ⚠ rankings_history.csv not found — skipping")
            print("  Place rankings_history.csv in the same folder as this script to test")

        print()

        # ── 4. Test UFC.com rankings scraper ─────────────────────────────────
        print("[4] Testing UFC.com rankings scraper (live request)...")
        ufc_ranks = scrape_ufc_rankings()
        if ufc_ranks.empty:
            print("  ✗ No rankings returned — check ufc.com layout")
        else:
            print(f"  ✓ {len(ufc_ranks)} entries scraped")
            print(f"  Sample (first 10):")
            print(ufc_ranks.head(10).to_string(index=False))
            divisions = ufc_ranks["weightclass"].unique()
            print(f"\n  Divisions found ({len(divisions)}): {list(divisions)}")
            champions = ufc_ranks[ufc_ranks["rank"] == 0]
            print(f"\n  Champions ({len(champions)}):")
            for _, r in champions.iterrows():
                print(f"    {r['weightclass']:<28} {r['fighter']}")

        print("\n=== END TEST ===")

    else:
        main()
