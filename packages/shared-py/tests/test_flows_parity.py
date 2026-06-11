"""flows.py is a hand-written mirror of packages/contracts/src/flows.ts.

contracts:build snapshots the TS routing tables into json-schema/flows.json;
this test turns any drift between the two implementations into a failure
instead of a silently different backend for some language's pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.flows import (
    DEFAULT_FLOW,
    _QWEN_ALIGN_LANGS,
    _QWEN_TRANSCRIBE_LANGS,
    flow_for,
)

_FLOWS_JSON = (
    Path(__file__).resolve().parents[2] / "contracts" / "json-schema" / "flows.json"
)


def _snapshot() -> dict:
    return json.loads(_FLOWS_JSON.read_text())


def test_language_sets_match_ts() -> None:
    data = _snapshot()
    assert sorted(_QWEN_ALIGN_LANGS) == data["qwen_align_langs"]
    assert sorted(_QWEN_TRANSCRIBE_LANGS) == data["qwen_transcribe_langs"]


def test_default_flow_matches_ts() -> None:
    d = _snapshot()["default_flow"]
    assert DEFAULT_FLOW.transcribe == d["transcribe"]
    assert DEFAULT_FLOW.transcribe_input == d["transcribe_input"]
    assert DEFAULT_FLOW.align == d["align"]


def test_flow_for_routing_invariants() -> None:
    # tr: qwen3 transcribes (mix input) but has no qwen aligner.
    tr = flow_for("tr")
    assert tr.transcribe == "qwen3"
    assert tr.transcribe_input == "mix"
    assert tr.align == "whisperx"
    # en: both backends qwen3.
    assert flow_for("en").align == "qwen3"
    # Unknown language and missing hint fall to the whisper default.
    assert flow_for("xx") == DEFAULT_FLOW
    assert flow_for(None) == DEFAULT_FLOW
    # Case-insensitive.
    assert flow_for("TR") == flow_for("tr")
