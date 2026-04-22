"""First-author surname lookup from arxiv's Atom API, with on-disk cache.

Used by refine to build canonical bibkeys as `{surname}{year}{shortname}`.
Kept out of refine.py because external API + HTTP cache + XML parsing
aren't refine concerns.
"""

from __future__ import annotations

import json
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch_surnames(arxiv_ids: list[str], cache_path: Path) -> dict[str, str | None]:
    """Return ``{arxiv_id: first_author_surname_or_None}`` for every id.

    Uses an on-disk JSON cache at *cache_path*; only misses hit the arxiv
    API. A 3-second interval between requests keeps within arxiv's
    published etiquette.
    """
    cache: dict[str, str | None] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [aid for aid in arxiv_ids if aid and aid not in cache]

    def _flush() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")

    # Flush every 20 fetches so a crash doesn't throw away hundreds
    # of successful lookups each costing ~3s of rate-limited API time.
    for i, aid in enumerate(missing):
        if i:
            time.sleep(3.1)
        cache[aid] = _fetch_one(aid)
        if (i + 1) % 20 == 0:
            _flush()
    if missing:
        _flush()
    return {aid: cache.get(aid) for aid in arxiv_ids if aid}


def _fetch_one(arxiv_id: str) -> str | None:
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                xml = r.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(15)
                continue
            return None
        except Exception:
            return None
    else:
        return None
    try:
        author = ET.fromstring(xml).find("a:entry/a:author/a:name", _ATOM_NS)
        if author is None or not author.text:
            return None
        words = author.text.strip().split()
        if not words:
            return None
        # Skip trailing generational suffixes (Jr., Sr., III, etc.) so we
        # keep the actual surname — seen on a handful of VLA authors.
        last = words[-1]
        if len(words) > 1 and last.lower().rstrip(".") in {"jr", "sr", "i", "ii", "iii", "iv", "v"}:
            last = words[-2]
        last = last.split("-")[0]
        ascii_only = "".join(c for c in unicodedata.normalize("NFKD", last) if c.isalpha() and c.isascii())
        return ascii_only.lower() or None
    except Exception:
        return None
