"""Unit tests for the secondary synced-lyrics adapter (synced.py).

The syncedlyrics library is mocked: no network. We assert provider-chain
order, provider naming, LRC parsing reuse, and silent degradation to None.
"""

from __future__ import annotations

import sys
import types

from annemusic import synced


class _Lyrics:
    def __init__(self, synced_text):
        self.synced = synced_text


def _FakeProvider(name, synced_text=None, boom=False):
    """Build a stand-in syncedlyrics provider *class* named ``name``."""

    def get_lrc(self, search_term):
        if boom:
            raise RuntimeError("network down")
        return _Lyrics(synced_text)

    return type(name, (), {"get_lrc": get_lrc})


_LRC = "[00:01.00]hello\n[00:03.00]world\n"


def _install(monkeypatch, providers):
    monkeypatch.setattr(synced, "_provider_classes", lambda: providers)


def test_first_synced_provider_wins_and_is_named(monkeypatch):
    miss = _FakeProvider("Musixmatch", synced_text=None)
    hit = _FakeProvider("NetEase", synced_text=_LRC)
    _install(monkeypatch, [miss, hit])
    result = synced.fetch("Title", "Artist")
    assert result is not None
    name, lines = result
    assert name == "netease"  # lowercased provider class name
    assert [ln["text"] for ln in lines] == ["hello", "world"]


def test_provider_order_is_honored(monkeypatch):
    first = _FakeProvider("Musixmatch", synced_text=_LRC)
    second = _FakeProvider("NetEase", synced_text=_LRC)
    _install(monkeypatch, [first, second])
    name, _ = synced.fetch("T", "A")
    assert name == "musixmatch"  # the earlier provider is returned


def test_all_misses_return_none(monkeypatch):
    _install(monkeypatch, [
        _FakeProvider("Musixmatch", synced_text=None),
        _FakeProvider("NetEase", synced_text=""),
    ])
    assert synced.fetch("T", "A") is None


def test_provider_exception_is_swallowed_and_chain_continues(monkeypatch):
    _install(monkeypatch, [
        _FakeProvider("Musixmatch", boom=True),
        _FakeProvider("NetEase", synced_text=_LRC),
    ])
    name, _ = synced.fetch("T", "A")
    assert name == "netease"  # the boom provider didn't abort the chain


def test_empty_lrc_parse_yields_none(monkeypatch):
    # A provider returns synced text with no parseable timestamps -> None.
    _install(monkeypatch, [_FakeProvider("Musixmatch", synced_text="no timestamps")])
    assert synced.fetch("T", "A") is None


def test_missing_title_or_artist_returns_none(monkeypatch):
    _install(monkeypatch, [_FakeProvider("Musixmatch", synced_text=_LRC)])
    assert synced.fetch(None, "A") is None
    assert synced.fetch("T", None) is None


def test_fixture_hook_short_circuits_the_network(monkeypatch, tmp_path):
    # ANNEMUSIC_SYNCED_FIXTURE points at a canned LRC; providers are never hit.
    fixture = tmp_path / "canned.lrc"
    fixture.write_text(_LRC, encoding="utf-8")
    monkeypatch.setenv("ANNEMUSIC_SYNCED_FIXTURE", str(fixture))
    _install(monkeypatch, [_FakeProvider("Musixmatch", boom=True)])  # would raise
    name, lines = synced.fetch("T", "A")
    assert name == "synced"  # fixture provider name
    assert [ln["text"] for ln in lines] == ["hello", "world"]


def test_fixture_hook_missing_file_falls_through_to_providers(monkeypatch, tmp_path):
    # An unreadable fixture path yields None from the hook; fetch then tries the
    # real provider chain instead of erroring.
    monkeypatch.setenv("ANNEMUSIC_SYNCED_FIXTURE", str(tmp_path / "absent.lrc"))
    _install(monkeypatch, [_FakeProvider("NetEase", synced_text=_LRC)])
    name, lines = synced.fetch("T", "A")
    assert name == "netease"
    assert [ln["text"] for ln in lines] == ["hello", "world"]


# --------------------------------------------------------------------------- #
# _provider_classes: the real lazy-import body (no _install mock here)
# --------------------------------------------------------------------------- #


def test_provider_classes_selects_only_present_names(monkeypatch):
    # Fake syncedlyrics module exposing two of the four wanted providers: the
    # selector returns exactly those, in _PROVIDER_NAMES order.
    fake = types.ModuleType("syncedlyrics")
    fake.Musixmatch = type("Musixmatch", (), {})
    fake.Deezer = type("Deezer", (), {})  # NetEase/Megalobiz absent
    monkeypatch.setitem(sys.modules, "syncedlyrics", fake)
    classes = synced._provider_classes()
    assert [c.__name__ for c in classes] == ["Musixmatch", "Deezer"]


def test_provider_classes_empty_when_library_missing(monkeypatch):
    # Import failure degrades to an empty list (caller then falls back to ASR).
    monkeypatch.setitem(sys.modules, "syncedlyrics", None)  # import -> ImportError
    assert synced._provider_classes() == []


# --------------------------------------------------------------------------- #
# Hard wall-clock cap: a hung provider must degrade to None, never block the
# pipeline (a timeout-less syncedlyrics request stalled a run for 1.5 days).
# --------------------------------------------------------------------------- #


def test_hung_provider_times_out_and_degrades(monkeypatch):
    import time

    def slow_get_lrc(self, search_term):
        time.sleep(5)  # far past the sub-second budget below
        return _Lyrics(_LRC)

    hung = type("Musixmatch", (), {"get_lrc": slow_get_lrc})
    _install(monkeypatch, [hung])
    monkeypatch.setenv("SYNCED_TIMEOUT_S", "0.2")

    started = time.monotonic()
    result = synced.fetch("T", "A")
    elapsed = time.monotonic() - started

    assert result is None  # degraded to ASR instead of hanging
    assert elapsed < 4  # returned on the budget, did not wait out the 5 s sleep
