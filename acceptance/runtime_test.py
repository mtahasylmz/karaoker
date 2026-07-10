"""Guard for the acceptance runtime's token expansion.

Not collected by the unit gate (testpaths = annemusic/); invoke explicitly:
    uv run pytest acceptance/runtime_test.py -q
"""

from __future__ import annotations

from acceptance.runtime import _tokens


def test_tokens_expands_all_env_vars(monkeypatch):
    monkeypatch.setenv("ANNEMUSIC_ARTIST", "Sezen Aksu")
    monkeypatch.setenv("ANNEMUSIC_FIXTURE", "/tmp/song.mp4")
    tokens = _tokens('annemusic $ANNEMUSIC_FIXTURE --artist "$ANNEMUSIC_ARTIST"')
    assert "Sezen Aksu" in tokens
    assert not any("$" in t for t in tokens)
