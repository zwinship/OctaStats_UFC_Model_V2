"""
ufcstats_results.py
Completed-event parsing shared by the V1 and V2 results recorders.
"""

from datetime import datetime

from bs4 import BeautifulSoup

from ufcstats_browser import fetch_html, SEL_EVENT_PAGE


def event_list_rows(html):
    soup = BeautifulSoup(html, "lxml")
    out  = []
    for row in soup.select("tr.b-statistics__table-row"):
        tds = row.find_all("td")
        if not tds:
            continue
        link_el   = tds[0].select_one("i.b-statistics__table-content a")
        date_span = tds[0].select_one("span.b-statistics__date")
        if not link_el or not date_span:
            continue
        date_text = date_span.get_text(strip=True)
        event_date = None
        for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
            try:
                event_date = datetime.strptime(date_text, fmt).date()
                break
            except ValueError:
                continue
        if event_date is None:
            continue
        out.append({"name": link_el.get_text(strip=True),
                    "date": event_date,
                    "url":  link_el.get("href", "").strip()})
    return out


_EVENT_CACHE = {}


def get_event_results(event_url):
    """
    Parse a completed UFCStats event page into one record per bout:
        {fighter_1, fighter_2, outcome, method}
    outcome: 'win' (fighter_1 won — UFCStats lists the winner first),
             'draw', 'nc'
    method:  'KO/TKO' | 'Submission' | 'Decision' | 'DQ' | 'No Contest' | raw
    """
    if event_url in _EVENT_CACHE:
        return _EVENT_CACHE[event_url]
    print(f"  Fetching event results page (Playwright): {event_url}")
    html = fetch_html(event_url, SEL_EVENT_PAGE)
    if not html:
        raise RuntimeError(f"UFCStats event page did not load: {event_url}")
    soup  = BeautifulSoup(html, "lxml")
    bouts = []
    for row in soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover"):
        tds = row.find_all("td")
        if len(tds) < 8:
            continue
        names = [a.get_text(strip=True) for a in tds[1].select("p a")]
        if len(names) < 2:
            continue
        flag = tds[0].get_text(" ", strip=True).lower()
        if "draw" in flag:
            outcome = "draw"
        elif "nc" in flag.split() or "no contest" in flag:
            outcome = "nc"
        elif "win" in flag:
            outcome = "win"
        else:
            outcome = None   # not yet fought / page not updated
        # td7 = <p>METHOD</p><p>details</p>; classify on the method line only
        method_p   = tds[7].find("p")
        method_raw = (method_p or tds[7]).get_text(" ", strip=True).upper()
        if "DQ" in method_raw:
            method = "DQ"
        elif "OVERTURNED" in method_raw or "CNC" in method_raw or outcome == "nc":
            method = "No Contest"
        elif "KO" in method_raw:
            method = "KO/TKO"
        elif "SUB" in method_raw:
            method = "Submission"
        elif "DEC" in method_raw:
            method = "Decision"
        else:
            method = method_raw or "Unknown"
        bouts.append({"fighter_1": names[0], "fighter_2": names[1],
                      "outcome": outcome, "method": method})
    _EVENT_CACHE[event_url] = bouts
    return bouts


def get_winners_from_event(event_url):
    """{fighter_lower: 'W' | 'L' | 'D' | 'NC'} for every fighter who fought."""
    results = {}
    for b in get_event_results(event_url):
        f1, f2 = b["fighter_1"].lower(), b["fighter_2"].lower()
        if b["outcome"] == "win":
            results[f1], results[f2] = "W", "L"
        elif b["outcome"] == "draw":
            results[f1] = results[f2] = "D"
        elif b["outcome"] == "nc":
            results[f1] = results[f2] = "NC"
    return results
