"""
bfo_odds.py
BestFightOdds front-page moneyline parsing shared by the V1 and V2 predictors.
"""

import re

import pandas as pd

from name_match import match_name

# Moneyline sportsbook priority. Prediction markets (Polymarket, Kalshi) are
# listed on BFO too but are never used — their prices are not sportsbook lines.
ML_BOOK_PREF = ["draftkings", "fanduel", "caesars", "betmgm", "betrivers", "betway", "unibet"]
_ODDS_RE     = re.compile(r"^[+-]\d{3,5}$")


def _bfo_cell_odds(cell):
    """American odds in a BFO odds cell. The price lives in the first <span>;
    a sibling <span> holds the ▲/▼ line-movement arrow, so reading the whole
    cell text ('-340▲') does not work."""
    if cell is None:
        return None
    span = cell.find("span")
    t = (span.get_text(strip=True) if span else cell.get_text(strip=True)).replace("−", "-")
    return t if _ODDS_RE.match(t) else None


def _bfo_book_columns(table):
    """{book_key: column index} from an odds table's header row."""
    cols = {}
    thead = table.find("thead")
    header = thead.find("tr") if thead else None
    if header is None:
        return cols
    for idx, th in enumerate(header.find_all(["th", "td"])):
        label = th.get_text(" ", strip=True).lower().replace(" ", "")
        for book in ML_BOOK_PREF + ["polymarket", "kalshi"]:
            if label.startswith(book) and book not in cols:
                cols[book] = idx
    return cols


def parse_bfo_moneylines(soup):
    """
    Parse every matchup on the BestFightOdds front page.

    BFO renders each event twice (a names-only responsive table, then the real
    odds table). Each matchup is two consecutive rows; the first row's <th>
    also contains a hidden admin link (<a class="bfo-admin-link">45250</a>)
    before the fighter link, so names must come from span.t-b-fcc.

    Returns DataFrame: fighter, odds, opponent, book, matchup_id, event
    Both fighters in a matchup are priced from the SAME book (first book in
    ML_BOOK_PREF quoting both sides) so the two implied probs are consistent.
    """
    out = []
    for table in soup.select("table.odds-table"):
        if "odds-table-responsive-header" in (table.get("class") or []):
            continue
        cols = _bfo_book_columns(table)
        if not any(b in cols for b in ML_BOOK_PREF):
            continue
        hdr   = table.find_previous("div", class_="table-header")
        event = hdr.get_text(" ", strip=True) if hdr else ""

        matchups = {}   # matchup_id -> [(name, cells), ...] in page order
        order    = []
        for row in (table.find("tbody") or table).find_all("tr", recursive=False):
            if "pr" in (row.get("class") or []):
                continue
            th = row.find("th")
            name_el = th.select_one("span.t-b-fcc") if th else None
            if not name_el:
                continue
            cells = row.find_all(["th", "td"], recursive=False)
            mid = None
            for td in cells:
                m = re.match(r"\[\s*\d+\s*,\s*\d+\s*,\s*(\d+)", td.get("data-li", ""))
                if m:
                    mid = m.group(1)
                    break
            if mid is None:
                # No priced cell: first row carries id="mu-<id>", second row
                # belongs to whichever matchup is still waiting for a partner
                rid = row.get("id", "")
                if rid.startswith("mu-"):
                    mid = rid[3:]
                elif order and len(matchups[order[-1]]) == 1:
                    mid = order[-1]
                else:
                    mid = f"x{len(order)}"
            if mid not in matchups:
                matchups[mid] = []
                order.append(mid)
            matchups[mid].append((name_el.get_text(strip=True), cells))

        for mid in order:
            pair = matchups[mid]
            if len(pair) != 2:
                continue
            chosen = None
            for book in ML_BOOK_PREF:
                col = cols.get(book)
                if col is None:
                    continue
                o1 = _bfo_cell_odds(pair[0][1][col] if col < len(pair[0][1]) else None)
                o2 = _bfo_cell_odds(pair[1][1][col] if col < len(pair[1][1]) else None)
                if o1 and o2:
                    chosen = (book, o1, o2)
                    break
            if chosen is None:
                continue
            book, o1, o2 = chosen
            for (name, _), odds, opp in [(pair[0], o1, pair[1][0]), (pair[1], o2, pair[0][0])]:
                out.append({"fighter": name, "odds": odds, "opponent": opp,
                            "book": book.title(), "matchup_id": mid, "event": event})

    return pd.DataFrame(out, columns=["fighter", "odds", "opponent", "book", "matchup_id", "event"])


def match_bout_odds(fa, fb, odds_df):
    """
    (odds_a, odds_b) for the BFO matchup containing these two fighters.
    Matching on the PAIR avoids pulling a same-surname fighter's line from a
    different fight or promotion on the BFO front page.
    """
    if odds_df is None or odds_df.empty:
        return None, None
    for mid, grp in odds_df.groupby("matchup_id", sort=False):
        names = grp["fighter"].tolist()
        if len(names) != 2:
            continue
        ma, mb = match_name(fa, names), match_name(fb, names)
        if ma and mb and ma != mb:
            g = grp.set_index("fighter")
            return g.loc[ma, "odds"], g.loc[mb, "odds"]
    # Pair not found (e.g. late replacement listed under another name):
    # accept a single-fighter match only when it is unique on the page.
    oa = ob = None
    for name, key in [(fa, "a"), (fb, "b")]:
        hit = match_name(name, odds_df["fighter"].unique().tolist())
        if hit is not None:
            val = odds_df[odds_df["fighter"] == hit].iloc[0]["odds"]
            if key == "a":
                oa = val
            else:
                ob = val
    return oa, ob
