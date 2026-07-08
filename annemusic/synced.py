"""Secondary synced-lyrics lookup via the `syncedlyrics` library.

Tried AFTER LRCLIB misses and BEFORE the ASR fallback. Returns
``(provider_name, lines)`` — the provider that hit ("musixmatch", "netease",
...) and synced lines [{text, start, end}] — or None when nothing has synced
lyrics. Any failure (network, parse, missing metadata) degrades to None so the
caller falls back to ASR silently.

Structured like lrclib.py: one public function, all failures -> None, the
network library imported lazily inside the call, and the same offline fixture
hook (ANNEMUSIC_SYNCED_FIXTURE) for tests. LRC parsing is reused from
lrclib.parse_lrc — one parser, one place.
"""

from __future__ import annotations

import os

from annemusic.lrclib import parse_lrc

# Providers tried, in order. LRCLIB itself is excluded — our lrclib.py already
# tried it first (and pipeline-10 asserts "lrclib" for the fixture).
_PROVIDER_NAMES = ("Musixmatch", "NetEase", "Megalobiz", "Deezer")


def fetch(
    title: str | None,
    artist: str | None,
    duration_s: float | None = None,
) -> tuple[str, list[dict]] | None:
    """First secondary provider with synced lyrics for (artist, title), as
    ``(provider_name, lines)``, or None. ``duration_s`` is accepted for a
    uniform adapter signature; the syncedlyrics providers don't use it."""
    if not title or not artist:
        return None
    canned = _fixture_lines()
    if canned is not None:
        return canned
    search_term = f"{artist} {title}"
    for cls in _provider_classes():
        lines = _try_provider(cls, search_term)
        if lines:
            return cls.__name__.lower(), lines
    return None


def _fixture_lines() -> tuple[str, list[dict]] | None:
    """Offline/acceptance hook: a canned LRC file stands in for the network."""
    fixture = os.environ.get("ANNEMUSIC_SYNCED_FIXTURE")
    if not fixture:
        return None
    try:
        lrc = open(fixture, encoding="utf-8").read()
    except OSError:
        return None
    lines = list(parse_lrc(lrc))
    return ("synced", lines) if lines else None


def _provider_classes() -> list:
    """The syncedlyrics provider classes, imported lazily. Empty on failure."""
    try:
        import syncedlyrics
    except Exception:
        return []
    return [getattr(syncedlyrics, name) for name in _PROVIDER_NAMES
            if hasattr(syncedlyrics, name)]


def _try_provider(cls, search_term: str) -> list[dict]:
    """LRC lines from one provider, or [] on any miss/failure."""
    try:
        lyrics = cls().get_lrc(search_term)
        synced = getattr(lyrics, "synced", None)
    except Exception:
        return []
    if not synced:
        return []
    return list(parse_lrc(synced))
