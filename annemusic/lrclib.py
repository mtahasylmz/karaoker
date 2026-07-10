"""LRCLIB lookup: free, no-auth, community-synced lyrics.

Returns synced lines [{text, start, end}] parsed from LRC, or None when
LRCLIB has no synced match. Harvested from the old transcribe stage
(commit 619e0a6); httpx swapped for stdlib urllib (ponytail: no dep for
one GET).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Iterable

RETRY_PAUSE_S = 2.0  # LRCLIB flakes transiently; one retry after a short pause

API = os.environ.get("LRCLIB_URL", "https://lrclib.net/api/get")


def fetch(
    title: str | None,
    artist: str | None,
    duration_s: float | None = None,
    timeout: float = 30.0,  # pod-to-lrclib latency swings 7-17s; be generous
) -> list[dict] | None:
    """Synced lines for (artist, title), or None. Any failure -> None so
    the caller falls back to ASR."""
    if not title or not artist:
        return None
    body = _get(title, artist, duration_s, timeout)
    if body is None:
        return None
    synced = body.get("syncedLyrics")
    if not synced:
        return None
    lines = list(parse_lrc(synced))
    return lines or None


def _get(title: str, artist: str, duration_s: float | None, timeout: float) -> dict | None:
    # Offline/acceptance hook: a canned response file stands in for the API.
    fixture = os.environ.get("ANNEMUSIC_LRC_FIXTURE")
    if fixture:
        try:
            return json.loads(open(fixture, encoding="utf-8").read())
        except OSError:
            return None
    params = {"track_name": title, "artist_name": artist}
    if duration_s is not None:
        params["duration"] = str(int(round(duration_s)))
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "annemusic (github.com/annemusic)"})
    # LRCLIB flakes transiently: retry the GET once (a transient miss otherwise
    # costs the whole human-timed lyrics path). Give up after that -> ASR.
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if attempt == 0:
                time.sleep(RETRY_PAUSE_S)
    return None


_TS_RE = re.compile(r"^\[(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)$")


def _parse_row(raw: str) -> tuple[float, str] | None:
    """One '[mm:ss.xx] text' line -> (start_seconds, text), or None if it has
    no leading timestamp."""
    m = _TS_RE.match(raw.strip())
    if not m:
        return None
    mm, ss, cs, text = m.groups()
    frac = int(cs) / 10 ** len(cs) if cs else 0.0
    return int(mm) * 60 + int(ss) + frac, text.strip()


def parse_lrc(lrc: str) -> Iterable[dict]:
    """'[mm:ss.xx] line' -> {text, start, end}. End of line N = start of
    N+1 (gapless); last line gets a +3 s tail. Shared by synced.py (the
    secondary-provider adapter) — one LRC parser, one place."""
    rows = [r for r in map(_parse_row, lrc.splitlines()) if r is not None]
    rows.sort(key=lambda r: r[0])
    for i, (start, text) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else start + 3.0
        if text:
            yield {"text": text, "start": float(start), "end": float(end)}
