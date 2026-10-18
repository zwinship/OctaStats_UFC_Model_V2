#!/usr/bin/env python3
"""
02_scrape_weekly_update.py
OctaStats V2 — Wednesday incremental data updater

Run every Wednesday at noon via GitHub Actions. Checks for any new
completed events since the last scrape date, appends new fight rows
to data/raw/fight_stats.csv, and pushes the update to GitHub.

Each run also repairs recent rows: fights scraped since REPAIR_SINCE that
are missing their method / physical attributes / implied probabilities
(the old weekly parser never captured them), and fights that were silently
skipped when a UFCStats page failed to render, are re-scraped.

Modes:
    python 02_scrape_weekly_update.py                  weekly update + repair
    python 02_scrape_weekly_update.py --rebuild-stats  re-parse EVERY fight page
        (one-off: rows scraped before Oct 2026 hold ROUND-1 stats instead of
         full-fight totals — see 01_scrape_historical.parse_fight_page)
    python 02_scrape_weekly_update.py --test           read-only checks

Environment variables (set as GitHub Actions secrets):
    ZWINSHIP_PAT  — personal access token for zwinship account
"""

import importlib.util
import requests
import pandas as pd
import numpy as np
import base64
import json
import os
import io
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("ZWINSHIP_PAT")
import sys as _sys
if not GITHUB_TOKEN and "--test" not in _sys.argv:
    raise EnvironmentError("ZWINSHIP_PAT environment variable is not set.")
REPO_OWNER   = "zwinship"
REPO_NAME    = "OctaStats_UFC_Model_V2"
BASE_URL     = "http://www.ufcstats.com"
GH_HEADERS   = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
if not GITHUB_TOKEN:
    GH_HEADERS.pop("Authorization")   # --test runs read the public repo anonymously

# Rows on/after this date were written by the old weekly parser (no method,
# details, judges, fighter URL or physicals) and are re-scraped when found.
REPAIR_SINCE = "2026-03-01"
# Recent cards are always re-checked for fights that failed to scrape.
RECHECK_DAYS = 60

_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
from name_match import match_name
from rankings import relabel_history, division_from_heading
from ufcstats_browser import fetch_soup, SEL_EVENT_LIST, SEL_EVENT_PAGE


def _load_historical_module():
    """01_scrape_historical.py holds the canonical fight / fighter page
    parsers and derived-column logic; reuse them so weekly rows match the
    historical rows exactly."""
    spec = importlib.util.spec_from_file_location(
        "scrape_historical", os.path.join(_HERE, "01_scrape_historical.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H01 = _load_historical_module()


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def read_csv_from_github(repo_path):
    """Read a CSV from GitHub, returning a DataFrame and the file SHA.
    Large files (>1MB) come back with empty inline content, so they are
    re-requested with the raw media type (not the CDN download_url, which can
    serve a stale copy for several minutes)."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        print(f"  [WARN] Could not read {repo_path} from GitHub (status {resp.status_code})")
        return pd.DataFrame(), None
    data = resp.json()
    sha  = data["sha"]

    raw_content = data.get("content", "").replace("\n", "").strip()
    if not raw_content and data.get("download_url"):
        raw_resp = requests.get(url, headers={**GH_HEADERS, "Accept": "application/vnd.github.raw"}, timeout=120)
        if raw_resp.status_code == 200:
            return pd.read_csv(io.StringIO(raw_resp.text), low_memory=False), sha
        print(f"  [WARN] Large file fallback failed (status {raw_resp.status_code})")
        return pd.DataFrame(), None

    content = base64.b64decode(raw_content).decode("utf-8")
    return pd.read_csv(io.StringIO(content), low_memory=False), sha


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
        return True
    raise RuntimeError(f"GitHub write failed for {repo_path}: "
                       f"{resp.status_code} {resp.text[:300]}")


META_PATH = "data/raw/fight_stats_meta.json"


def read_meta():
    """fight_stats_meta.json records that the one-off full-fight stats rebuild
    ran, plus any fights whose pages failed during it (retried weekly)."""
    url  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{META_PATH}"
    resp = requests.get(url, headers=GH_HEADERS)
    if resp.status_code != 200:
        return {}, None
    data = resp.json()
    return json.loads(base64.b64decode(data["content"]).decode()), data["sha"]


def write_meta(meta, sha):
    url     = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{META_PATH}"
    payload = {"message": "fight_stats metadata",
               "content": base64.b64encode(json.dumps(meta, indent=2).encode()).decode()}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADERS, json=payload)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"GitHub write failed for {META_PATH}: {resp.status_code} {resp.text[:300]}")


# ── UFCStats event list / cards ───────────────────────────────────────────────

def list_completed_events():
    """All completed UFC events: [{name, date (datetime), url, location}], newest first."""
    soup = fetch_soup(f"{BASE_URL}/statistics/events/completed?page=all", SEL_EVENT_LIST)
    if soup is None:
        raise RuntimeError("UFCStats completed-events page did not load (browser check "
                           "not cleared) — refusing to report 'no new fights'.")
    today  = datetime.now()
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        tds = row.find_all("td")
        if not tds:
            continue
        link_el   = tds[0].select_one("i.b-statistics__table-content a")
        date_span = tds[0].select_one("span.b-statistics__date")
        if not link_el or not date_span:
            continue
        event_date = None
        for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
            try:
                event_date = datetime.strptime(date_span.get_text(strip=True), fmt)
                break
            except ValueError:
                continue
        # The list also shows the next (not yet fought) event at the top
        if event_date is None or event_date.date() >= today.date():
            continue
        events.append({
            "name":     link_el.get_text(strip=True),
            "date":     event_date,
            "url":      link_el.get("href", "").strip(),
            "location": tds[1].get_text(strip=True) if len(tds) > 1 else "",
        })
    if not events:
        raise RuntimeError("UFCStats completed-events page rendered no events.")
    return events


def get_fight_urls_from_event(event_url):
    soup = fetch_soup(event_url, SEL_EVENT_PAGE)
    if soup is None:
        return None
    return [
        row.get("data-link")
        for row in soup.select(".b-fight-details__table-row.b-fight-details__table-row__hover")
        if row.get("data-link") and "fight-details" in row.get("data-link")
    ]


def scrape_fights(jobs):
    """jobs: [(fight_url, event)] → (rows, failed_urls)"""
    rows, failed = [], []
    for i, (fight_url, ev) in enumerate(jobs, 1):
        parsed = H01.parse_fight_page(fight_url, ev["name"],
                                      ev["date"].strftime("%Y-%m-%d"), ev["location"])
        if parsed:
            rows.extend(parsed)
        else:
            failed.append(fight_url)
        if i % 50 == 0 or i == len(jobs):
            print(f"    {i}/{len(jobs)} fight pages parsed ({len(failed)} failed)")
    return rows, failed


# ── Odds backfill ─────────────────────────────────────────────────────────────

def _american_to_prob(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(o):
        return np.nan
    return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)


def _load_odds_pool():
    """Moneylines captured by the Friday predictor:
    data/raw/bfo_odds_history.csv (every bout on the card) plus the odds
    stored in the prediction CSVs (bouts the model priced)."""
    pools = []
    hist, _ = read_csv_from_github("data/raw/bfo_odds_history.csv")
    if not hist.empty:
        pools.append(hist.rename(columns={"odds": "odds_numeric"})[
            ["fighter", "opponent", "odds_numeric", "event_date"]])
    resp = requests.get(f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/predictions",
                        headers=GH_HEADERS)
    if resp.status_code == 200:
        for f in resp.json():
            name = f["name"]
            if not (name.startswith("v2_betting_recommendations_") and name.endswith(".csv")):
                continue
            pred, _ = read_csv_from_github(f"predictions/{name}")
            if pred.empty or not {"name", "opponent", "odds_numeric", "event_date"} <= set(pred.columns):
                continue
            pools.append(pred[["name", "opponent", "odds_numeric", "event_date"]]
                         .rename(columns={"name": "fighter"}))
    if not pools:
        return pd.DataFrame(columns=["fighter", "opponent", "odds_numeric", "event_date"])
    pool = pd.concat(pools, ignore_index=True)
    pool["odds_numeric"] = pd.to_numeric(pool["odds_numeric"], errors="coerce")
    pool = pool.dropna(subset=["odds_numeric"])
    pool["event_date"] = pd.to_datetime(pool["event_date"], errors="coerce")
    return pool


def backfill_bfo_odds(fight_df, row_mask):
    """
    Fill historical_odds / historical_odds_opp for rows in row_mask.

    A line is only accepted when BOTH fighters of the bout have one and the
    pair's implied probabilities sum to a sportsbook-like overround
    (1.00–1.15). Before Oct 2026 the predictor sometimes stored prediction-
    market prices or one-sided lines, and those must not enter training data.
    """
    for col in ["historical_odds", "historical_odds_opp"]:
        if col not in fight_df.columns:
            fight_df[col] = np.nan
    pool = _load_odds_pool()
    if pool.empty:
        print("    BFO backfill: no stored odds available")
        return fight_df

    fight_df = fight_df.copy()
    dates    = pd.to_datetime(fight_df["event_date"], errors="coerce")
    filled   = 0
    todo     = fight_df[row_mask & fight_df["historical_odds"].isna()]
    for fight_url, grp in todo.groupby("fight_url"):
        if len(grp) != 2:
            continue
        d = dates[grp.index[0]]
        window = pool[(pool["event_date"] - d).abs() <= pd.Timedelta(days=3)]
        if window.empty:
            continue
        names = window["fighter"].unique().tolist()
        idx_a, idx_b = grp.index
        ha = match_name(fight_df.at[idx_a, "fighter"], names)
        hb = match_name(fight_df.at[idx_b, "fighter"], names)
        if not ha or not hb or ha == hb:
            continue
        oa = window[window["fighter"] == ha]["odds_numeric"].iloc[-1]
        ob = window[window["fighter"] == hb]["odds_numeric"].iloc[-1]
        overround = _american_to_prob(oa) + _american_to_prob(ob)
        if not (1.0 <= overround <= 1.15):
            continue
        fight_df.loc[idx_a, ["historical_odds", "historical_odds_opp"]] = [oa, ob]
        fight_df.loc[idx_b, ["historical_odds", "historical_odds_opp"]] = [ob, oa]
        filled += 2
    print(f"    BFO backfill: {filled} odds filled")
    return fight_df


# ── UFC.com rankings ──────────────────────────────────────────────────────────

def scrape_ufc_rankings():
    """
    Current UFC rankings. Each ufc.com table's division is read from its own
    caption (<h4>Lightweight</h4> / <h4>Men's Pound-for-Pound</h4>) — the old
    positional mapping shifted every division by one (see rankings.py).
    """
    UFC_RANKINGS_URL = "https://www.ufc.com/rankings"
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        resp = sess.get(UFC_RANKINGS_URL, timeout=20)
        if resp.status_code != 200:
            print(f"    [WARN] UFC rankings HTTP {resp.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"    [WARN] UFC rankings: {e}")
        return pd.DataFrame()
    soup  = BeautifulSoup(resp.text, "lxml")
    today = datetime.now().strftime("%Y-%m-%d")
    rows  = []
    seen  = set()
    for table in soup.select("table"):
        caption = table.find("caption")
        heading = caption.find("h4") if caption else None
        wc      = division_from_heading(heading.get_text(" ", strip=True) if heading else "")
        if wc is None or wc in seen:
            continue
        seen.add(wc)
        # Division captions show the champion (rank 0); P4P captions repeat #1
        if "Pound-for-Pound" not in wc:
            champ_link = caption.select_one("h5 a")
            if champ_link:
                rows.append({"date": today, "weightclass": wc,
                             "fighter": champ_link.get_text(strip=True), "rank": 0})
        for tr in table.select("tbody tr"):
            tds = tr.select("td")
            if len(tds) < 2:
                continue
            try:
                rank_num = int(tds[0].get_text(strip=True))
            except ValueError:
                continue
            link = tds[1].find("a")
            if link and link.get_text(strip=True):
                rows.append({"date": today, "weightclass": wc,
                             "fighter": link.get_text(strip=True), "rank": rank_num})
    if not rows:
        print("    [WARN] No rankings found from ufc.com")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["labels_fixed"] = 1
    print(f"    ✓ UFC rankings: {len(df)} entries, {df['weightclass'].nunique()} divisions")
    return df


def update_rankings_on_github(new_ranks_df):
    """Append today's snapshot and repair the mislabelled older snapshots.
    Returns the full (corrected) history."""
    existing_df, sha = read_csv_from_github("data/raw/rankings_history.csv")
    fixed = relabel_history(existing_df) if not existing_df.empty else existing_df
    today = datetime.now().strftime("%Y-%m-%d")
    already = (not fixed.empty and today in fixed["date"].astype(str).values)
    if new_ranks_df.empty or already:
        combined = fixed
    else:
        combined = pd.concat([fixed, new_ranks_df], ignore_index=True)
    if not combined.equals(existing_df):
        write_csv_to_github(combined, "data/raw/rankings_history.csv",
                            f"Weekly rankings — {today}", sha=sha)
        print(f"    Rankings history: {len(combined):,} total rows")
    return combined


def _division_of(weight_class):
    """'UFC Women's Flyweight Title' → "Women's Flyweight"; tournaments /
    catch weights have no ranking division."""
    wc = str(weight_class or "")
    if "tournament" in wc.lower() or "catch" in wc.lower():
        return None
    return division_from_heading(wc)


def merge_rankings_into_new(new_df, rankings_df):
    """Attach each fighter's division and P4P rank as of the fight date."""
    for col in ["fighter_rank_wc", "fighter_rank_pfp", "diff_rank_wc", "diff_rank_pfp"]:
        if col not in new_df.columns:
            new_df[col] = np.nan
    if rankings_df is None or rankings_df.empty or new_df.empty:
        return new_df
    rk = rankings_df.copy()
    rk["date"] = pd.to_datetime(rk["date"], errors="coerce")
    rk["rank"] = pd.to_numeric(rk["rank"], errors="coerce")
    snapshots  = sorted(rk["date"].dropna().unique())

    def _rank(name, fight_date, division):
        if division is None:
            return np.nan
        past = [d for d in snapshots if d <= fight_date]
        if not past:
            return np.nan
        snap = rk[(rk["date"] == past[-1]) & (rk["weightclass"] == division)]
        hit  = match_name(name, snap["fighter"].tolist())
        return float(snap[snap["fighter"] == hit]["rank"].min()) if hit else np.nan

    new_df = new_df.copy()
    dates  = pd.to_datetime(new_df["event_date"], errors="coerce")
    wc_rank, pfp_rank = [], []
    for idx, row in new_df.iterrows():
        div = _division_of(row.get("weight_class"))
        pfp = None if div is None else ("Women's Pound-for-Pound" if div.startswith("Women") else "Pound-for-Pound")
        wc_rank.append(_rank(row["fighter"], dates[idx], div))
        pfp_rank.append(_rank(row["fighter"], dates[idx], pfp))
    new_df["fighter_rank_wc"]  = wc_rank
    new_df["fighter_rank_pfp"] = pfp_rank
    opp = new_df.set_index(["fight_url", "fighter"])[["fighter_rank_wc", "fighter_rank_pfp"]]
    for col, src in [("diff_rank_wc", "fighter_rank_wc"), ("diff_rank_pfp", "fighter_rank_pfp")]:
        new_df[col] = [
            r[src] - opp[src].get((r["fight_url"], r["opponent"]), np.nan)
            for _, r in new_df.iterrows()
        ]
    print(f"    Rankings: {new_df['fighter_rank_wc'].notna().sum()}/{len(new_df)} WC ranks attached")
    return new_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main(rebuild_stats=False):
    print("=" * 60)
    print("OctaStats V2 — Weekly Data Update" + (" (FULL STATS REBUILD)" if rebuild_stats else ""))
    print(f"Running: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    existing_df, existing_sha = read_csv_from_github("data/raw/fight_stats.csv")
    if existing_df.empty:
        raise RuntimeError("No existing fight_stats.csv. Run 01_scrape_historical.py first.")
    original_cols = list(existing_df.columns)

    dates         = pd.to_datetime(existing_df["event_date"])
    latest_date   = dates.max()
    existing_urls = set(existing_df["fight_url"].unique())
    print(f"Most recent event in dataset: {latest_date.strftime('%Y-%m-%d')} "
          f"({existing_df['fight_url'].nunique():,} fights)")

    meta, meta_sha = read_meta()
    if meta.get("stat_totals") != "full_fight" and not rebuild_stats:
        print("\n  fight_stats.csv still holds ROUND-1 stats (pre-Oct-2026 parser) —")
        print("  running the one-time full-fight stats rebuild of every fight.")
        rebuild_stats = True
    pending = set(meta.get("pending_rebuild", []))

    # A: Scrape + save current UFC rankings (also repairs old snapshot labels)
    print("\n[A] Scraping current UFC rankings from ufc.com...")
    rankings_df = update_rankings_on_github(scrape_ufc_rankings())

    # B: Decide what to scrape
    print("\n[B] Checking UFCStats for new / incomplete fights...")
    events       = list_completed_events()
    ev_by_name   = {e["name"]: e for e in events}
    needs_repair = existing_df["method"].isna() & (dates >= REPAIR_SINCE)
    print(f"  Rows needing repair (no method since {REPAIR_SINCE}): {int(needs_repair.sum())}")

    recheck_from = datetime.now() - timedelta(days=RECHECK_DAYS)
    if needs_repair.any():
        recheck_from = min(recheck_from, pd.Timestamp(REPAIR_SINCE).to_pydatetime())
    jobs = []           # (fight_url, event) to (re)scrape
    seen = set()
    for ev in events:
        if not (ev["date"] > latest_date or ev["date"] >= recheck_from):
            continue
        urls = get_fight_urls_from_event(ev["url"])
        if urls is None:
            raise RuntimeError(f"UFCStats event page did not load: {ev['url']}")
        new_urls = [u for u in urls if u not in existing_urls]
        if new_urls:
            print(f"  {ev['name']} ({ev['date']:%Y-%m-%d}): {len(new_urls)} fight(s) not in dataset")
        for u in new_urls:
            jobs.append((u, ev))
            seen.add(u)

    reparse = existing_df if rebuild_stats else existing_df[
        needs_repair | existing_df["fight_url"].isin(pending)]
    for fight_url, grp in reparse.groupby("fight_url"):
        if fight_url in seen:
            continue
        r  = grp.iloc[0]
        ev = ev_by_name.get(r["event_name"]) or {
            "name": r["event_name"], "date": pd.Timestamp(r["event_date"]).to_pydatetime()}
        ev = {**ev, "location": r.get("event_location", "") if pd.notna(r.get("event_location")) else ""}
        jobs.append((fight_url, ev))
        seen.add(fight_url)

    if not jobs:
        print("No new or incomplete fights. Dataset is up to date.")
        if pending:
            write_meta({**meta, "pending_rebuild": []}, meta_sha)
        return
    print(f"  Scraping {len(jobs)} fight page(s)...")
    rows, failed = scrape_fights(jobs)
    if not rows:
        raise RuntimeError(f"All {len(failed)} fight pages failed to load.")
    fresh = pd.DataFrame(rows)
    fresh["event_date"] = pd.to_datetime(fresh["event_date"]).dt.strftime("%Y-%m-%d")

    # C: Replace scraped columns of re-parsed fights, keeping non-scraped
    #    columns (stored odds, implied probs) from the existing rows
    print("\n[C] Merging scraped rows...")
    keep_cols = [c for c in existing_df.columns if c not in fresh.columns]
    replaced  = existing_df[existing_df["fight_url"].isin(fresh["fight_url"])]
    fresh     = fresh.merge(replaced[["fight_url", "fighter"] + keep_cols],
                            on=["fight_url", "fighter"], how="left")
    base      = existing_df[~existing_df["fight_url"].isin(fresh["fight_url"])]
    combined  = pd.concat([base, fresh], ignore_index=True)
    fresh_urls = set(fresh["fight_url"])
    n_new     = int((~fresh["fight_url"].isin(existing_urls)).sum())
    n_fixed   = len(fresh) - n_new
    print(f"  {n_new} new rows, {n_fixed} re-scraped rows")

    # D: Fighter pages (height / reach / stance / DOB) for touched rows
    print("\n[D] Scraping fighter pages for physical attributes...")
    touched   = combined["fight_url"].isin(fresh_urls)
    need_phys = combined[touched & (combined["height_cms"].isna() | combined["age"].isna())]
    urls      = need_phys["fighter_url"].dropna().unique().tolist()
    cache     = {}
    for i, u in enumerate(urls, 1):
        cache[u] = H01.scrape_fighter_page(u)
        if i % 50 == 0 or i == len(urls):
            print(f"    {i}/{len(urls)} fighter pages")

    # E: Derived columns (records, streaks, rounds, diffs) — same code as 01
    print("\n[E] Recomputing derived columns...")
    combined = H01.compute_self_derived_cols(combined, cache).reset_index(drop=True)
    touched  = combined["fight_url"].isin(fresh_urls)

    # F: Odds + implied probabilities for touched rows
    print("\n[F] Backfilling sportsbook odds + implied probabilities...")
    # Odds stored for rows since REPAIR_SINCE came from the old predictor's
    # backfill (prediction-market prices / one-sided lines) — re-derive them
    recent = touched & (pd.to_datetime(combined["event_date"]) >= REPAIR_SINCE)
    combined.loc[recent, ["historical_odds", "historical_odds_opp", "implied_prob",
                          "implied_prob_ko", "implied_prob_sub", "implied_prob_dec",
                          "implied_prob_source"]] = np.nan
    combined = backfill_bfo_odds(combined, touched)
    if combined.loc[touched, "implied_prob"].isna().any():
        imputed = H01.impute_odds(combined)
        cols    = ["implied_prob", "implied_prob_ko", "implied_prob_sub",
                   "implied_prob_dec", "implied_prob_source"]
        fill    = touched & combined["implied_prob"].isna()
        combined.loc[fill, cols] = imputed.loc[fill, cols].values

    # G: Rankings for touched rows (corrected division labels)
    print("\n[G] Attaching rankings...")
    rank_cols = ["fighter_rank_wc", "fighter_rank_pfp", "diff_rank_wc", "diff_rank_pfp"]
    ranked = merge_rankings_into_new(combined[touched], rankings_df)
    combined.loc[touched, rank_cols] = ranked[rank_cols].values

    # H: Align to the existing schema and save
    for col in original_cols:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[original_cols + [c for c in combined.columns if c not in original_cols]]
    combined = combined.sort_values(["event_date", "fight_url"], ascending=[False, True]).reset_index(drop=True)

    write_csv_to_github(combined, "data/raw/fight_stats.csv",
                        f"Weekly data update — {datetime.now().strftime('%Y-%m-%d')}"
                        + (" (full stats rebuild)" if rebuild_stats else ""),
                        sha=existing_sha)
    touched = combined["fight_url"].isin(fresh_urls)
    print(f"\n✓ {n_new} new rows, {n_fixed} repaired. Total: {len(combined):,}")
    for col in ["method", "height_cms", "age", "implied_prob", "historical_odds", "fighter_rank_wc"]:
        print(f"    {col:<18} coverage in touched rows: {combined.loc[touched, col].notna().mean():.0%}")

    if rebuild_stats or pending:
        # Existing fights whose page could not be re-parsed keep their old
        # (round-1) stats; remember them and retry next week
        still_pending = {u for u in failed if u in existing_urls} | (pending - fresh_urls)
        write_meta({
            "stat_totals":     "full_fight",
            "rebuilt_at":      datetime.now().strftime("%Y-%m-%d %H:%M UTC") if rebuild_stats
                               else meta.get("rebuilt_at"),
            "pending_rebuild": sorted(still_pending),
        }, meta_sha)

    if failed:
        msg = (f"{len(failed)}/{len(jobs)} fight page(s) failed to scrape and will be retried "
               f"next run: {failed[:10]}")
        if len(failed) > max(5, 0.05 * len(jobs)):
            raise RuntimeError(msg)
        print(f"::warning title=UFCStats pages failed::{msg}")


def run_test():
    """
    --test mode: read-only checks of every live source the updater uses.
    No GitHub writes.
    """
    print("=== TEST MODE: 02_scrape_weekly_update ===\n")
    all_pass = True

    print("[1] UFCStats completed events (browser)...")
    try:
        events = list_completed_events()
        print(f"  ✓ {len(events)} events; latest: {events[0]['name']} ({events[0]['date']:%Y-%m-%d})")
    except Exception as e:
        print(f"  ✗ {e}")
        return False

    print("\n[2] Fight page parsing — full-fight totals, not round 1...")
    urls = get_fight_urls_from_event(events[0]["url"]) or []
    print(f"  {len(urls)} fights on {events[0]['name']}")
    if urls:
        rows = H01.parse_fight_page(urls[0], events[0]["name"], events[0]["date"].strftime("%Y-%m-%d"),
                                    events[0]["location"])
        for r in rows:
            print(f"    {r['fighter']:26s} won={r['won']} sig={r['sig_str_landed']}/{r['sig_str_att']} "
                  f"td={r['td_landed']} ctrl={r['ctrl_seconds']}s method={r['method']!r} "
                  f"rnd={r['finish_round']} fmt={r['time_format']!r} wc={r['weight_class']!r}")
        ok = len(rows) == 2 and bool(rows[0]["method"]) and bool(rows[0]["fighter_url"])
        print(f"  {'✓' if ok else '✗'} method + fighter_url captured")
        all_pass &= ok
        if rows and rows[0].get("fighter_url"):
            info = H01.scrape_fighter_page(rows[0]["fighter_url"])
            print(f"  {'✓' if info.get('height_cms') else '✗'} fighter page: {info}")
            all_pass &= bool(info.get("height_cms"))

    print("\n[3] UFC.com rankings (caption-based divisions)...")
    rk = scrape_ufc_rankings()
    if rk.empty:
        print("  ✗ no rankings")
        all_pass = False
    else:
        for wc, g in rk.groupby("weightclass", sort=False):
            print(f"    {wc:24s} " + ", ".join(f"{r.rank}:{r.fighter}" for r in g.head(3).itertuples()))

    print("\n[4] Repair plan (reads GitHub, writes nothing)...")
    df, _ = read_csv_from_github("data/raw/fight_stats.csv")
    if df.empty:
        print("  ✗ could not read fight_stats.csv")
        all_pass = False
    else:
        d    = pd.to_datetime(df["event_date"])
        need = df["method"].isna() & (d >= REPAIR_SINCE)
        print(f"  rows needing repair: {int(need.sum())} ({df[need]['fight_url'].nunique()} fights)")
        have, missing = set(df["fight_url"]), 0
        for ev in [e for e in events if e["date"] >= pd.Timestamp(REPAIR_SINCE)]:
            miss = [u for u in (get_fight_urls_from_event(ev["url"]) or []) if u not in have]
            if miss:
                missing += len(miss)
                print(f"    {ev['name']} ({ev['date']:%Y-%m-%d}): {len(miss)} fight(s) missing from dataset")
        print(f"  fights missing from dataset since {REPAIR_SINCE}: {missing}")

    print()
    print("=" * 60)
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED — see above'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        sys.exit(0 if run_test() else 1)
    else:
        main(rebuild_stats="--rebuild-stats" in sys.argv)
