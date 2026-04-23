"""LRCLIB lookup: free, open, community-synced lyrics.

Returns a list of segments compatible with our Segment contract. If no
synced match is found we return None so the caller falls through to
Whisper.
"""

from __future__ import annotations

import re
from typing import Iterable

import httpx

API = "https://lrclib.net/api/get"


def fetch(
    title: str | None,
    artist: str | None,
    duration_s: float | None = None,
    timeout: float = 5.0,
) -> list[dict] | None:
    """Return [{text, start, end}] parsed from a synced LRC, or None if
    LRCLIB has no synced match for (artist, title)."""
    if not title or not artist:
        return None
    params: dict[str, str] = {"track_name": title, "artist_name": artist}
    if duration_s is not None:
        params["duration"] = str(int(round(duration_s)))
    try:
        r = httpx.get(API, params=params, timeout=timeout)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    body = r.json()
    synced = body.get("syncedLyrics")
    if not synced:
        return None
    segments = list(_parse_lrc(synced))
    return segments or None


_TS_RE = re.compile(r"^\[(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)$")


def _parse_lrc(lrc: str) -> Iterable[dict]:
    """Turn Musixmatch/LRCLIB '[mm:ss.xx] line' lines into segments.

    End of line N = start of line N+1 (gapless). Last line gets +3s tail.
    """
    rows: list[tuple[float, str]] = []
    for raw in lrc.splitlines():
        m = _TS_RE.match(raw.strip())
        if not m:
            continue
        mm, ss, cs, text = m.groups()
        start = int(mm) * 60 + int(ss) + (int(cs) / 10 ** len(cs) if cs else 0.0)
        rows.append((start, text.strip()))
    rows.sort(key=lambda r: r[0])
    for i, (start, text) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else start + 3.0
        if not text:
            continue
        yield {"text": text, "start": float(start), "end": float(end)}
