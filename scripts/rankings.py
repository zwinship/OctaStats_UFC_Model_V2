"""
rankings.py
UFC rankings helpers shared by the weekly updater and the predictor.

Until Oct 2026 the ufc.com scraper labelled divisions by table position, but
ufc.com lists the men's pound-for-pound table first, so every snapshot was
shifted by one division ("Flyweight" held the men's P4P list, "Bantamweight"
held flyweight, … "Women's Strawweight" held heavyweight). relabel_history()
repairs those snapshots; scrape_ufc_rankings() now reads each table's own
heading instead.
"""

import re

import numpy as np
import pandas as pd

from name_match import match_name

# Position-based label written by the old scraper → the division actually
# in that table. The 14th table (added mid-2026) duplicates a division list
# and is dropped.
_OLD_LABEL_TO_TRUE = {
    "Flyweight":               "Pound-for-Pound",
    "Bantamweight":            "Flyweight",
    "Featherweight":           "Bantamweight",
    "Lightweight":             "Featherweight",
    "Welterweight":            "Lightweight",
    "Middleweight":            "Welterweight",
    "Light Heavyweight":       "Middleweight",
    "Heavyweight":             "Light Heavyweight",
    "Women's Strawweight":     "Heavyweight",
    "Women's Flyweight":       "Women's Pound-for-Pound",
    "Women's Bantamweight":    "Women's Strawweight",
    "Women's Featherweight":   "Women's Flyweight",
    "Pound-for-Pound":         "Women's Bantamweight",
    "Women's Pound-for-Pound": None,
}


def relabel_history(df):
    """Return rankings_history with the shifted snapshots corrected.
    Idempotent: rows already carrying labels_fixed == 1 (repaired, or written
    by the heading-based scraper) are left alone."""
    if df.empty or "weightclass" not in df.columns:
        return df
    df = df.copy()
    if "labels_fixed" not in df.columns:
        df["labels_fixed"] = 0
    old = pd.to_numeric(df["labels_fixed"], errors="coerce").fillna(0).astype(int) == 0
    df.loc[old, "weightclass"] = df.loc[old, "weightclass"].map(_OLD_LABEL_TO_TRUE)
    # P4P tables list the #1 fighter in the caption AND at #1 — keep the ranked row
    p4p = df["weightclass"].fillna("").str.contains("Pound-for-Pound")
    df = df[~(old & (df["weightclass"].isna() | (p4p & (df["rank"] == 0))))].copy()
    df["labels_fixed"] = 1
    return df.reset_index(drop=True)


def division_from_heading(text):
    """Normalise a ufc.com table heading to our division names."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    tl = t.lower()
    if "pound" in tl:
        return "Women's Pound-for-Pound" if "women" in tl else "Pound-for-Pound"
    for name in ["Light Heavyweight", "Heavyweight", "Middleweight", "Welterweight",
                 "Lightweight", "Featherweight", "Bantamweight", "Flyweight", "Strawweight"]:
        if name.lower() in tl:
            return ("Women's " + name) if "women" in tl else name
    return None


def current_rank(rankings_df, fighter, division):
    """Latest-snapshot rank of `fighter` in `division` (champion = 0), or NaN."""
    if rankings_df is None or rankings_df.empty:
        return np.nan
    latest = rankings_df[rankings_df["date"].astype(str) == rankings_df["date"].astype(str).max()]
    pool = latest[latest["weightclass"] == division]
    hit = match_name(fighter, pool["fighter"].tolist())
    if hit is None:
        return np.nan
    return float(pd.to_numeric(pool[pool["fighter"] == hit]["rank"], errors="coerce").min())
