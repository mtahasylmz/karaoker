"""Unit tests for LRCLIB lookup (lrclib.py).

No network: the fixture hook (ANNEMUSIC_LRC_FIXTURE) and a stubbed urlopen
drive every branch of fetch/_get, plus parse_lrc directly.
"""

from __future__ import annotations

import io
import json

from annemusic import lrclib


def _write_fixture(tmp_path, payload: dict):
    p = tmp_path / "lrc.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


_LRC = "[00:01.00]hello\n[00:03.50]world\n"


# --------------------------------------------------------------------------- #
# fetch: metadata guard, syncedLyrics present/absent, empty-parse
# --------------------------------------------------------------------------- #


def test_fetch_returns_none_without_title_or_artist():
    assert lrclib.fetch(None, "Artist") is None
    assert lrclib.fetch("Title", None) is None


def test_fetch_via_fixture_returns_parsed_lines(monkeypatch, tmp_path):
    fixture = _write_fixture(tmp_path, {"syncedLyrics": _LRC})
    monkeypatch.setenv("ANNEMUSIC_LRC_FIXTURE", str(fixture))
    lines = lrclib.fetch("Title", "Artist")
    assert [ln["text"] for ln in lines] == ["hello", "world"]
    assert lines[0]["start"] == 1.0 and lines[1]["start"] == 3.5


def test_fetch_none_when_body_has_no_synced_lyrics(monkeypatch, tmp_path):
    fixture = _write_fixture(tmp_path, {"plainLyrics": "no timing here"})
    monkeypatch.setenv("ANNEMUSIC_LRC_FIXTURE", str(fixture))
    assert lrclib.fetch("Title", "Artist") is None


def test_fetch_none_when_synced_lyrics_have_no_timestamps(monkeypatch, tmp_path):
    fixture = _write_fixture(tmp_path, {"syncedLyrics": "just prose, no marks"})
    monkeypatch.setenv("ANNEMUSIC_LRC_FIXTURE", str(fixture))
    assert lrclib.fetch("Title", "Artist") is None


def test_fetch_none_when_fixture_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ANNEMUSIC_LRC_FIXTURE", str(tmp_path / "absent.json"))
    assert lrclib.fetch("Title", "Artist") is None


# --------------------------------------------------------------------------- #
# _get network path: stubbed urlopen, including the duration parameter
# --------------------------------------------------------------------------- #


def test_get_hits_the_api_and_includes_duration(monkeypatch):
    seen = {}

    class _Resp:
        def __enter__(self):
            return io.BytesIO(json.dumps({"syncedLyrics": _LRC}).encode())

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.delenv("ANNEMUSIC_LRC_FIXTURE", raising=False)
    monkeypatch.setattr(lrclib.urllib.request, "urlopen", fake_urlopen)
    lines = lrclib.fetch("My Song", "The Band", duration_s=241.6, timeout=7.0)
    assert [ln["text"] for ln in lines] == ["hello", "world"]
    assert "track_name=My+Song" in seen["url"]
    assert "artist_name=The+Band" in seen["url"]
    assert "duration=242" in seen["url"]  # rounded to nearest second
    assert seen["timeout"] == 7.0


def test_get_omits_duration_when_not_supplied(monkeypatch):
    seen = {}

    class _Resp:
        def __enter__(self):
            return io.BytesIO(json.dumps({"syncedLyrics": _LRC}).encode())

        def __exit__(self, *a):
            return False

    monkeypatch.delenv("ANNEMUSIC_LRC_FIXTURE", raising=False)
    monkeypatch.setattr(
        lrclib.urllib.request, "urlopen",
        lambda req, timeout: seen.setdefault("url", req.full_url) or _Resp(),
    )
    lrclib.fetch("S", "A")
    assert "duration=" not in seen["url"]


def test_get_returns_none_on_network_error(monkeypatch):
    def boom(req, timeout):
        raise OSError("connection refused")

    monkeypatch.delenv("ANNEMUSIC_LRC_FIXTURE", raising=False)
    monkeypatch.setattr(lrclib.urllib.request, "urlopen", boom)
    monkeypatch.setattr(lrclib.time, "sleep", lambda _s: None)
    assert lrclib.fetch("S", "A") is None


def test_get_retries_once_after_a_transient_failure(monkeypatch):
    # LRCLIB flakes transiently: first GET dies, the retry lands. One retry
    # must recover the synced lines instead of silently falling back to ASR.
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return io.BytesIO(json.dumps({"syncedLyrics": _LRC}).encode())

        def __exit__(self, *a):
            return False

    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("connection reset")
        return _Resp()

    monkeypatch.delenv("ANNEMUSIC_LRC_FIXTURE", raising=False)
    monkeypatch.setattr(lrclib.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(lrclib.time, "sleep", lambda _s: None)  # keep tests fast
    lines = lrclib.fetch("S", "A")
    assert [ln["text"] for ln in lines] == ["hello", "world"]
    assert calls["n"] == 2  # exactly one retry


def test_get_gives_up_after_two_consecutive_failures(monkeypatch):
    # Not an infinite retry loop: two failures in a row -> None (ASR fallback).
    calls = {"n": 0}

    def always_boom(req, timeout):
        calls["n"] += 1
        raise OSError("still down")

    monkeypatch.delenv("ANNEMUSIC_LRC_FIXTURE", raising=False)
    monkeypatch.setattr(lrclib.urllib.request, "urlopen", always_boom)
    monkeypatch.setattr(lrclib.time, "sleep", lambda _s: None)
    assert lrclib.fetch("S", "A") is None
    assert calls["n"] == 2  # original attempt + one retry, then give up


# --------------------------------------------------------------------------- #
# parse_lrc: timestamp arithmetic, gapless end, last-line tail, sorting
# --------------------------------------------------------------------------- #


def test_parse_lrc_end_is_next_start_gapless():
    lines = list(lrclib.parse_lrc("[00:01.00]a\n[00:02.50]b\n"))
    assert lines[0] == {"text": "a", "start": 1.0, "end": 2.5}


def test_parse_lrc_last_line_gets_three_second_tail():
    lines = list(lrclib.parse_lrc("[00:10.00]end\n"))
    assert lines[-1]["end"] == 13.0


def test_parse_lrc_sorts_out_of_order_lines():
    lines = list(lrclib.parse_lrc("[00:05.00]second\n[00:01.00]first\n"))
    assert [ln["text"] for ln in lines] == ["first", "second"]


def test_parse_lrc_handles_minutes_and_variable_centis():
    # mm:ss and a single-digit fractional field ([..:..\.x]).
    lines = list(lrclib.parse_lrc("[01:05.5]late\n"))
    assert lines[0]["start"] == 65.5


def test_parse_lrc_skips_blank_and_untimed_lines():
    lines = list(lrclib.parse_lrc("not a timestamp\n[00:01.00]\n[00:02.00]kept\n"))
    assert [ln["text"] for ln in lines] == ["kept"]  # empty-text line dropped


# --------------------------------------------------------------------------- #
# fetch clamps LRC lines to the video's audio duration: LRC timestamps come
# from the song *recording*, which can run past the *video's* audio (the last
# line's +3 s tail or a next-line gap overruns [0, duration]).
# --------------------------------------------------------------------------- #


def test_fetch_clamps_line_end_overrunning_duration(monkeypatch, tmp_path):
    # Last line starts before duration but its +3 s tail runs past it: the end
    # must be pulled back to duration_s so no word lands outside [0, duration].
    fixture = _write_fixture(tmp_path, {"syncedLyrics": "[04:02.29]son\n"})
    monkeypatch.setenv("ANNEMUSIC_LRC_FIXTURE", str(fixture))
    lines = lrclib.fetch("T", "A", duration_s=242.95)
    assert lines[-1]["start"] == 242.29
    assert lines[-1]["end"] == 242.95  # clamped down from 242.29 + 3.0


def test_fetch_drops_line_starting_at_or_past_duration(monkeypatch, tmp_path):
    # A line whose start is already at/past the video audio has no room: drop
    # it entirely rather than emit a zero/negative-width window.
    fixture = _write_fixture(
        tmp_path, {"syncedLyrics": "[00:01.00]inside\n[04:03.00]past\n"}
    )
    monkeypatch.setenv("ANNEMUSIC_LRC_FIXTURE", str(fixture))
    lines = lrclib.fetch("T", "A", duration_s=242.95)
    assert [ln["text"] for ln in lines] == ["inside"]
    assert lines[0]["end"] == 242.95  # the survivor's tail is also clamped


def test_fetch_without_duration_keeps_current_behavior(monkeypatch, tmp_path):
    # duration_s not provided: no clamp/drop, last line keeps its +3 s tail.
    fixture = _write_fixture(tmp_path, {"syncedLyrics": "[04:02.29]son\n"})
    monkeypatch.setenv("ANNEMUSIC_LRC_FIXTURE", str(fixture))
    lines = lrclib.fetch("T", "A")
    assert lines[-1]["start"] == 242.29
    assert lines[-1]["end"] == 245.29  # 242.29 + 3.0, unclamped
