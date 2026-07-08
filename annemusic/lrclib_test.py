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
    assert lrclib.fetch("S", "A") is None


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
