"""
name_match.py
Fighter-name normalisation and matching shared by the scrapers.

Different sources spell the same fighter differently ("Suyoung You" vs
"SuYoung You", "Allen Frye Jr." vs "Allen Frye", "José" vs "Jose"), and
the old per-script matchers fell back to "first name containing the last
name", which silently picks the wrong fighter when two share a surname
(two Silvas, two Chandlers on one page). These helpers only fall back to
a last-name match when it is unambiguous.
"""

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def norm_name(name):
    """'José Aldo Jr.' → 'jose aldo'"""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[.\'’`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)


def last_name(name):
    parts = norm_name(name).split()
    return parts[-1] if parts else ""


def match_name(name, candidates):
    """
    Return the candidate that refers to `name`, or None.
      1. exact normalised match
      2. same set of name tokens in a different order / spacing
         ("Song Yadong" vs "Yadong Song", "SuYoung You" vs "Su Young You")
      3. unique last-name match
    """
    target = norm_name(name)
    if not target:
        return None
    normed = {}
    for c in candidates:
        normed.setdefault(norm_name(c), c)
    if target in normed:
        return normed[target]

    squashed = target.replace(" ", "")
    tset = set(target.split())
    for n, c in normed.items():
        if n.replace(" ", "") == squashed or (len(tset) > 1 and set(n.split()) == tset):
            return c

    last = target.split()[-1]
    hits = [c for n, c in normed.items() if n.split() and n.split()[-1] == last]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        first = target.split()[0]
        narrowed = [c for c in hits if norm_name(c).split()[0][:1] == first[:1]]
        if len(narrowed) == 1:
            return narrowed[0]
    return None


def same_fighter(a, b):
    return match_name(a, [b]) is not None
