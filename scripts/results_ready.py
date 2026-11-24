#!/usr/bin/env python3
"""
results_ready.py
Gate for the results workflow: are this weekend's fight results complete on
UFCStats and not yet recorded?

Prints the decision and writes `ready=true|false` (plus `event=<name>`) to
$GITHUB_OUTPUT when running in GitHub Actions.

    python scripts/results_ready.py --precheck  # fight-night poll, cheap step only
    python scripts/results_ready.py             # fight-night poll, full check
    python scripts/results_ready.py --fallback  # Sunday noon / manual run

Fight-night polls first do a cheap check with no browser: titles/fight_titles.json
(written by the Friday predictor) must name an upcoming event dated today, yesterday
or the day before, and that event must not already be recorded as "recent". Only
then is UFCStats loaded. The fallback mode skips the cheap check, so it still
catches an event the Friday run missed.

Results count as complete when the event page lists every bout with an outcome
(win / draw / NC). Cancelled bouts drop off the page, so they don't block it.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE_URL      = "http://www.ufcstats.com"
TITLES_PATH   = os.path.join(HERE, "..", "titles", "fight_titles.json")
WINDOW_DAYS   = 2     # an event counts as "this weekend" for this many days


def set_output(ready, event="", reason="", key="ready"):
    print(f"{key}={str(ready).lower()}  event={event!r}  ({reason})")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"{key}={str(ready).lower()}\n")
            f.write(f"event={event}\n")


def load_titles():
    try:
        with open(TITLES_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main(fallback=False, precheck=False):
    today  = datetime.now(timezone.utc).date()
    titles = load_titles()
    recent = titles.get("recent", "")

    if not fallback:
        upcoming = titles.get("upcoming", "")
        try:
            up_date = datetime.strptime(titles.get("upcoming_date", ""), "%Y-%m-%d").date()
        except ValueError:
            up_date = None
        if not upcoming or up_date is None or not (0 <= (today - up_date).days <= WINDOW_DAYS):
            return set_output(False, reason="no event scheduled this weekend",
                              key="poll" if precheck else "ready")
        if upcoming == recent:
            return set_output(False, upcoming, "results already recorded", key="poll" if precheck else "ready")
        if precheck:
            return set_output(True, upcoming, "event this weekend — check UFCStats", key="poll")

    # Only now pay for a browser
    from ufcstats_browser import fetch_html, SEL_EVENT_LIST
    from ufcstats_results import event_list_rows, get_event_results

    html = fetch_html(f"{BASE_URL}/statistics/events/completed", SEL_EVENT_LIST)
    if not html:
        return set_output(False, reason="UFCStats did not load — will retry next poll")
    window = [e for e in event_list_rows(html)
              if timedelta(0) <= today - e["date"] <= timedelta(days=WINDOW_DAYS)]
    if not window:
        return set_output(False, reason="no UFC event in the last few days")
    event = window[0]
    if event["name"] == recent:
        return set_output(False, event["name"], "results already recorded")

    try:
        bouts = get_event_results(event["url"])
    except RuntimeError as e:
        return set_output(False, event["name"], f"event page did not load ({e})")
    pending = [f"{b['fighter_1']} vs {b['fighter_2']}" for b in bouts if b["outcome"] is None]
    if not bouts:
        return set_output(False, event["name"], "no bouts on the event page yet")
    if pending:
        return set_output(False, event["name"],
                          f"{len(bouts) - len(pending)}/{len(bouts)} bouts final; waiting on {pending[0]}")
    return set_output(True, event["name"], f"all {len(bouts)} bouts have results")


if __name__ == "__main__":
    main(fallback="--fallback" in sys.argv, precheck="--precheck" in sys.argv)
