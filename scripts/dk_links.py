"""
dk_links.py
DraftKings deep links for recommended bets, resolved through Gambly's
bet-slip API. Shared by the V1 and V2 predictors.

Gambly requires a logged-in session: set the GAMBLY_COOKIE secret to the
Cookie header of a logged-in gambly.com browser session (GAMBLY_TOKEN_SET,
the legacy single-cookie secret, is still tried). When neither is accepted
every bet falls back to a Gambly prompt URL, which still opens a pre-filled
bet slip for a logged-in visitor.
"""

import os

import requests


def _gambly_url(prompt):
    """Mirror the frontend gamblyUrl() function exactly."""
    import urllib.parse
    cleaned = prompt.lower().replace(" ", "+")
    encoded = urllib.parse.quote(cleaned, safe="+")
    return f"https://gambly.com/gambly-bot?auto=1&prompt={encoded}"


# ── Gambly link resolution (API) ─────────────────────────────────────────────

def resolve_gambly_links(bet_rows, prop_rows=None, timeout_ms=45000):
    """
    Resolve fighter names -> DraftKings deep links via Gambly's JSON API.

    Flow:
      1. Use Playwright (headless) to load gambly.com once -> get Cloudflare
         clearance cookies (anon-session-token, cf_clearance, etc.)
      2. POST start-betslip-job with those cookies via requests
      3. Poll betslip-job-status until completed
      4. GET  webpage-result -> queryString with bet IDs
      5. GET  get-alternate-offers-by-group-hash -> sourceData has raw DK URLs
    """
    import time as _time
    import json as _json

    ml_bets   = [r for r in bet_rows  if float(r.get("bet_size", 0)) > 0]
    prop_bets = [p for p in (prop_rows or []) if p.get("flagged")]

    auth_state = {"rejected": False}
    BASE    = "https://gambly.com/api/proxy/core/api"
    UA      = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/122.0.0.0 Safari/537.36")
    HEADERS = {
        "User-Agent":   UA,
        "Content-Type": "application/json",
        "Referer":      "https://gambly.com/",
        "Origin":       "https://gambly.com",
    }

    # ── Step 1: get session cookies ──────────────────────────────────────────
    def _candidate_sessions():
        """
        Gambly only builds bet slips for a logged-in (Firebase) session —
        anonymous sessions get 401 'Firebase session required' (verified
        Oct 2026), so the old headless-anonymous fallback can never return
        DraftKings links. Credentials, tried in order:
          GAMBLY_COOKIE    — full Cookie header copied from a logged-in
                             browser session on gambly.com
          GAMBLY_TOKEN_SET — legacy gambly_token_set cookie value
        """
        sessions = []
        cookie = os.environ.get("GAMBLY_COOKIE", "").strip()
        if cookie:
            sessions.append(("GAMBLY_COOKIE", cookie))
        token_set = os.environ.get("GAMBLY_TOKEN_SET", "").strip()
        if token_set:
            sessions.append(("GAMBLY_TOKEN_SET", f"gambly_token_set={token_set}"))
        return sessions

    # ── Step 2-5: call Gambly API with cookies ────────────────────────────────
    def _get_dk_links(prompt, names, cookie_str):
        headers = {**HEADERS, "Cookie": cookie_str}

        # Start job
        try:
            print(f"    [INFO] POSTing start-betslip-job...")
            r = requests.post(
                f"{BASE}/gambly-bot/start-betslip-job",
                headers=headers,
                json={"prompt": prompt, "base64Images": [],
                      "twitterUrl": "", "saveBetslipFeedPostSync": False},
                timeout=20,
            )
            print(f"    [INFO] POST status: {r.status_code}")
            if r.status_code in (401, 403):
                auth_state["rejected"] = True
                print(f"    [WARN] Gambly rejected the session ({r.status_code}): {r.text[:120]}")
                return None, {}
            if r.status_code not in (200, 202):
                print(f"    [WARN] Unexpected status {r.status_code}: {r.text[:200]}")
                return None, {}
            job_id = r.json()["jobId"]
            print(f"    [INFO] Job started: {job_id[:8]}...")
        except Exception as e:
            print(f"    [WARN] Gambly job start failed: {e}")
            import traceback; traceback.print_exc()
            return None, {}

        # Poll until completed (max 40s)
        for attempt in range(20):
            _time.sleep(2)
            try:
                r = requests.get(
                    f"{BASE}/gambly-bot/betslip-job-status/{job_id}",
                    headers=headers, timeout=10,
                )
                s = r.json().get("status")
                if s == "completed":
                    print(f"    [INFO] Job completed after {(attempt+1)*2}s")
                    break
                elif s == "failed":
                    print(f"    [WARN] Gambly job failed")
                    return None, {}
            except Exception as e:
                print(f"    [WARN] Poll error: {e}")
                return None, {}
        else:
            print(f"    [WARN] Gambly job timed out after 40s")
            return None, {}

        # Get result
        try:
            r = requests.get(
                f"{BASE}/gambly-bot/webpage-result/{job_id}",
                headers=headers, timeout=10,
            )
            r.raise_for_status()
            import json as _json3
            _wr = r.json()

            # Find DraftKings betOffers specifically and dump them
            _bet_offers = _wr.get("betOffers", [])
            _share_url = _wr.get("shareUrl", "")
            qs = _wr.get("queryString", "")
        except Exception as e:
            print(f"    [WARN] Gambly result fetch failed: {e}")
            return None, {}

        # Parse bet IDs and group hashes
        bet_ids, group_hash_ids = [], []
        for part in qs.split("&"):
            if part.startswith("betOfferIdsPoints="):
                for token in part.split("=", 1)[1].split("|"):
                    bid = token.split("_")[0]
                    if bid.lstrip("-").isdigit():
                        bet_ids.append(bid)
            elif part.startswith("groupHashIds="):
                group_hash_ids = [h for h in part.split("=", 1)[1].split("|") if h]

        if not bet_ids:
            print(f"    [WARN] No bet IDs in queryString")
            return None, {}

        print(f"    [INFO] {len(bet_ids)} bet IDs, {len(group_hash_ids)} group hashes")
        print(f"    [INFO] First 3 bet IDs: {bet_ids[:3]}")
        print(f"    [INFO] First 3 hashes: {group_hash_ids[:3]}")

        # Get raw DK URLs from sourceData
        dk_urls_by_bet_id = {}
        if group_hash_ids:
            try:
                hashes_param = "%2C".join(group_hash_ids)
                r = requests.get(
                    f"{BASE}/bets/get-alternate-offers-by-group-hash"
                    f"?groupHashIds={hashes_param}",
                    headers=headers, timeout=10,
                )
                print(f"    [INFO] get-alternate-offers status: {r.status_code}")
                r.raise_for_status()
                resp_json = r.json()
                print(f"    [INFO] Offers groups: {len(resp_json)}")
                for group_offers in resp_json.values():
                    for offer in (group_offers or []):
                        bid = str(offer.get("id", ""))
                        sd  = offer.get("sourceData", "")
                        if sd:
                            try:
                                sd_obj = _json.loads(sd)
                                dk_url = sd_obj.get("Desktop") or sd_obj.get("Android")
                                if dk_url and "draftkings" in dk_url:
                                    dk_urls_by_bet_id[bid] = dk_url
                            except Exception:
                                pass
                print(f"    [INFO] DK URLs found: {len(dk_urls_by_bet_id)}")
                print(f"    [INFO] bet_ids[:5]: {bet_ids[:5]}")
                print(f"    [INFO] dk_urls keys[:5]: {list(dk_urls_by_bet_id.keys())[:5]}")
            except Exception as e:
                print(f"    [WARN] get-alternate-offers failed: {e}")
                import traceback; traceback.print_exc()
        else:
            # ── Playwright: load Gambly share page, click DK icon, scrape preurl ──
            # The shareUrl page shows FanDuel by default but has DK links loaded.
            # Clicking the DK book icon swaps to DK links — the real DK event URL
            # is encoded in the href as a `preurl` query param.
            if _share_url:
                print(f"    [INFO] Loading share page to click DK icon and scrape links...")
                try:
                    import urllib.parse as _up, re as _re2
                    from playwright.sync_api import sync_playwright as _spw
                    with _spw() as _pw:
                        _br  = _pw.chromium.launch(headless=True)
                        _ctx = _br.new_context(user_agent=UA)
                        # Inject the session cookies so Gambly treats us as logged-in
                        _ctx.add_cookies([
                            {"name": k.strip(), "value": v.strip(), "domain": "gambly.com",
                             "path": "/", "secure": True, "sameSite": "Lax"}
                            for k, v in (part.split("=", 1) for part in cookie_str.split(";") if "=" in part)
                        ])
                        _pg = _ctx.new_page()
                        _pg.goto(_share_url, wait_until="domcontentloaded", timeout=25000)
                        _time.sleep(3)

                        # Click the DraftKings book icon
                        # It's a small book-switcher button — find it by looking for
                        # an element containing "draftkings" in its image src or aria-label
                        try:
                            _dk_btn = _pg.locator("img[src*='draft-kings'], img[alt*='DraftKings'], button[aria-label*='DraftKings']").first
                            if _dk_btn.count() == 0:
                                # fallback: find by logo URL pattern in any clickable element
                                _dk_btn = _pg.locator("[src*='draft-kings']").first
                            _dk_btn.click(timeout=5000)
                            print(f"    [INFO] Clicked DK book icon")
                            _time.sleep(2)
                        except Exception as _ce:
                            print(f"    [WARN] DK icon click failed: {_ce} — trying xpath")
                            try:
                                # Use the xpath area the user identified for the book switcher
                                _pg.click("xpath=/html/body/main/div[2]/div/div/div/div[2]/div[1]/div[1]/div/div[1]/div/div/div[2]", timeout=5000)
                                print(f"    [INFO] Clicked DK via xpath")
                                _time.sleep(2)
                            except Exception as _ce2:
                                print(f"    [WARN] xpath click also failed: {_ce2}")

                        # Match by fighter name using the bet-name-renderer class.
                        # Structure: .bet-name-renderer (name) and sibling <a> (link)
                        # are both children of the same parent div.
                        def _decode_preurl(href):
                            if not href: return None
                            if "preurl=" in href:
                                _qs = href.split("?", 1)[1] if "?" in href else ""
                                _pm = dict(p.split("=", 1) for p in _qs.split("&") if "=" in p)
                                _r  = _up.unquote(_pm.get("preurl", ""))
                                return _r if "draftkings" in _r else None
                            if "draftkings" in href and "/event/" in href:
                                return href
                            return None

                        # Build a map of name → DK URL from the page.
                        # Two element types to handle:
                        #   1. Fighter-specific props: name in .bet-name-renderer
                        #      e.g. "K. Vallejos", "C. Johnson"
                        #   2. Fight-level props (goes/doesn't go distance): DK puts
                        #      the matchup string in a sibling div, NOT .bet-name-renderer
                        #      e.g. "Andre Fili vs. Jose Miguel Delgado"
                        _name_to_dk = {}

                        # All bet card root elements — each card contains both the
                        # name/matchup div AND the anchor link
                        _all_cards = _pg.locator("a[href*='draftkings'], a[href*='preurl']").all()
                        print(f"    [INFO] Found {len(_all_cards)} DK anchor elements on page")

                        for _card in _all_cards:
                            try:
                                _href = _card.get_attribute("href", timeout=2000) or ""
                                _real = _decode_preurl(_href)
                                if not _real:
                                    continue
                                # Walk up to find the containing bet card, then grab
                                # any text label — either .bet-name-renderer or the
                                # matchup div (first non-empty div text in the card)
                                _container = _card.locator("..").locator("..")
                                # Try .bet-name-renderer first
                                _label_el = _container.locator(".bet-name-renderer").first
                                _label = ""
                                try:
                                    _label = (_label_el.inner_text(timeout=1000) or "").strip()
                                except Exception:
                                    pass
                                # If empty, grab all inner text of the container and
                                # take the first non-trivial line — this catches the
                                # matchup div "Andre Fili vs. Jose Miguel Delgado"
                                if not _label:
                                    _full = (_container.inner_text(timeout=1000) or "").strip()
                                    for _line in _full.split("\n"):
                                        _line = _line.strip()
                                        if len(_line) > 3 and _line not in ("Yes", "No", "Over", "Under"):
                                            _label = _line
                                            break
                                if _label:
                                    _name_to_dk[_label.lower()] = _real
                                    print(f"    [INFO] Scraped: {_label} → {_real}")
                            except Exception:
                                pass

                        # Also try the original .bet-name-renderer approach as a
                        # supplementary pass (catches abbreviated names like "K. Vallejos")
                        _name_els = _pg.locator(".bet-name-renderer").all()
                        print(f"    [INFO] Found {len(_name_els)} bet-name-renderer elements")
                        for _nel in _name_els:
                            try:
                                _fighter = (_nel.inner_text() or "").strip()
                                _parent  = _nel.locator("..").locator("..")
                                _a       = _parent.locator("a[href*='draftkings'], a[href*='preurl']").first
                                _href    = _a.get_attribute("href", timeout=2000) or ""
                                _real    = _decode_preurl(_href)
                                if _real and _fighter:
                                    _name_to_dk[_fighter.lower()] = _real
                                    print(f"    [INFO] Scraped (renderer): {_fighter} → {_real}")
                            except Exception:
                                pass

                        print(f"    [INFO] Name→URL map: {list(_name_to_dk.keys())}")

                        # Match our bet names to the scraped map.
                        # For fighter-specific props: try full name, then last name.
                        # For fight-level props: try matching either fighter's last name
                        # against the matchup string keys (e.g. "fili" in "andre fili vs. jose...")
                        for _name, _bid in zip(names, bet_ids):
                            _nl   = _name.lower()
                            _url  = _name_to_dk.get(_nl)
                            if not _url:
                                # Last name match against all keys
                                _last = _nl.split()[-1]
                                _url  = next((u for k, u in _name_to_dk.items() if _last in k), None)
                            if not _url:
                                # First name match (catches "Andre" in "Andre Fili vs. ...")
                                _first = _nl.split()[0]
                                _url   = next((u for k, u in _name_to_dk.items() if _first in k), None)
                            if _url:
                                dk_urls_by_bet_id[_bid] = _url
                                print(f"    [INFO] {_name} → {_url}")
                            else:
                                print(f"    [WARN] {_name}: no match in name→URL map")
                        _br.close()
                    print(f"    [INFO] DK URLs from share page: {len(dk_urls_by_bet_id)}")
                except Exception as e:
                    print(f"    [WARN] Share page scrape failed: {e}")
                    import traceback; traceback.print_exc()
            else:
                print(f"    [WARN] No shareUrl available")



        # Map to fighter names by position (bet_ids order = prompt order = names order)
        name_to_url = {}
        for i, name in enumerate(names):
            if i < len(bet_ids):
                name_to_url[name] = dk_urls_by_bet_id.get(bet_ids[i])

        # Parlay: use __parlay__ if set by deeplink/v3, otherwise construct from outcomes
        parlay_url = dk_urls_by_bet_id.pop("__parlay__", None)
        if not parlay_url and len(bet_ids) >= 2:
            try:
                all_outcomes, event_id = [], None
                for bid in bet_ids:
                    url = dk_urls_by_bet_id.get(bid, "")
                    if "outcomes=" in url:
                        all_outcomes.append(url.split("outcomes=")[1].split("&")[0])
                        if not event_id and "/event/" in url:
                            event_id = url.split("/event/")[1].split("?")[0]
                if all_outcomes and event_id:
                    parlay_url = (
                        f"https://sportsbook.draftkings.com/event/{event_id}"
                        f"?outcomes={'+'.join(all_outcomes)}"
                    )
            except Exception:
                pass

        return parlay_url, name_to_url

    # ── Main ──────────────────────────────────────────────────────────────────
    print("\n[GAMBLY] Resolving DraftKings deep links via Gambly API...")
    parlay_links = {"dk": None}

    sessions = _candidate_sessions()
    if not sessions:
        print("  [WARN] No Gambly credentials set (GAMBLY_COOKIE / GAMBLY_TOKEN_SET) — "
              "DraftKings deep links need a logged-in Gambly session. Using Gambly URLs.")
        return _fallback_links(bet_rows, prop_rows)

    def _with_session(prompt, names):
        """Run a bet-slip job with the first credential Gambly accepts."""
        nonlocal sessions
        while sessions:
            label, cookie = sessions[0]
            auth_state["rejected"] = False
            print(f"    [INFO] Using {label}")
            result = _get_dk_links(prompt, names, cookie)
            if not auth_state["rejected"]:
                return result
            print(f"::warning title=Gambly session expired::{label} was rejected by Gambly — "
                  f"refresh the secret (see README → DraftKings links).")
            sessions = sessions[1:]
        return None, {}

    if ml_bets:
        names  = [r.get("fighter") or r.get("name") or "" for r in ml_bets]
        prompt = " and ".join(n + " moneyline" for n in names)
        gurl   = _gambly_url(prompt)
        print(f"  Moneylines ({len(ml_bets)} bets): {prompt}")

        parlay_url, name_to_url = _with_session(prompt, names)

        parlay_links["dk"] = parlay_url or gurl
        print(f"  {'✓' if parlay_url else '↩ fallback'}  Parlay DK:")
        print(f"    {parlay_links['dk']}")

        for r in ml_bets:
            name    = r.get("fighter") or r.get("name") or ""
            dk_href = name_to_url.get(name)
            r["dk_link"] = dk_href or _gambly_url(name + " moneyline")
            print(f"  {'✓' if dk_href else '↩ fallback (not yet on DK)'}  {name}:")
            print(f"    {r['dk_link']}")

    if prop_bets:
        # Map prop_type → DraftKings-style prompt phrasing Gambly understands
        DK_PROP_PHRASE = {
            "fighter_a_ko":          "wins by KO TKO",
            "fighter_b_ko":          "wins by KO TKO",
            "fighter_a_sub":         "wins by submission",
            "fighter_b_sub":         "wins by submission",
            "fighter_a_dec":         "wins by decision",
            "fighter_b_dec":         "wins by decision",
            "fight_goes_distance":   "fight goes the distance",
            "fight_inside_distance": "fight does not go the distance",
        }

        def _prop_prompt_str(p):
            prop_type = p.get("prop_type", "")
            phrase    = DK_PROP_PHRASE.get(prop_type)
            if phrase:
                # For fighter-specific props use the actual fighter name
                # For fight-level props use fighter_a as the fight anchor
                name = p.get("fighter") or p.get("fighter_a") or ""
                return f"{name} {phrase}"
            # Fallback: use stored prop_label
            name = p.get("fighter_a") or p.get("fighter") or ""
            return name + " " + (p.get("prop_label") or prop_type or "prop")

        names  = [p.get("fighter") or p.get("fighter_a") or "" for p in prop_bets]
        prompt = " and ".join(_prop_prompt_str(p) for p in prop_bets)
        print(f"  Props ({len(prop_bets)} bets): {prompt}")

        # For the name→URL lookup, distance props use "yes"/"no" as the DK label
        # not the fighter name — build a parallel lookup_names list accordingly
        DISTANCE_TYPES = {"fight_goes_distance", "fight_inside_distance"}
        lookup_names = []
        for p in prop_bets:
            pt = p.get("prop_type", "")
            if pt == "fight_goes_distance":
                lookup_names.append("yes")
            elif pt == "fight_inside_distance":
                lookup_names.append("no")
            else:
                lookup_names.append(p.get("fighter") or p.get("fighter_a") or "")

        _, name_to_url = _with_session(prompt, lookup_names)

        for p, name, lname in zip(prop_bets, names, lookup_names):
            dk_href      = name_to_url.get(lname)
            p["dk_link"] = dk_href or _gambly_url(_prop_prompt_str(p))
            print(f"  {'✓' if dk_href else '↩ fallback (not yet on DK)'}  {name}:")
            print(f"    {p['dk_link']}")

    print(f"  [GAMBLY] Done — {len(ml_bets)} ML bets + {len(prop_bets)} props resolved")
    return bet_rows, prop_rows or [], parlay_links

def _fallback_links(bet_rows, prop_rows):
    """Populate dk_link with individual Gambly URLs when Playwright unavailable.
    Each fighter gets their own single-bet Gambly URL so users can click directly
    to their specific bet, rather than the full parlay URL."""
    ml_bets = [r for r in bet_rows if float(r.get("bet_size", 0)) > 0]
    gurl_parlay = _gambly_url(" and ".join(
        (r.get("fighter") or r.get("name") or "") + " moneyline"
        for r in ml_bets
    )) if ml_bets else ""
    for r in ml_bets:
        name = r.get("fighter") or r.get("name") or ""
        r["dk_link"] = _gambly_url(name + " moneyline")
    parlay_links = {"dk": gurl_parlay if len(ml_bets) >= 2 else None}
    if prop_rows:
        DK_PROP_PHRASE = {
            "fighter_a_ko": "wins by KO TKO", "fighter_b_ko": "wins by KO TKO",
            "fighter_a_sub": "wins by submission", "fighter_b_sub": "wins by submission",
            "fighter_a_dec": "wins by decision", "fighter_b_dec": "wins by decision",
            "fight_goes_distance": "fight goes the distance",
            "fight_inside_distance": "fight does not go the distance",
        }
        for p in prop_rows:
            if p.get("flagged"):
                name   = p.get("fighter") or p.get("fighter_a") or ""
                phrase = DK_PROP_PHRASE.get(p.get("prop_type", ""),
                         p.get("prop_label") or p.get("prop_type") or "prop")
                p["dk_link"] = _gambly_url(f"{name} {phrase}")
    return bet_rows, prop_rows or [], parlay_links
