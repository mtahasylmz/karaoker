"""Filter-graph builders: each fragment produces the expected ffmpeg syntax."""

from __future__ import annotations

import pytest

from record_mix import filter_chain


LN = "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-22:measured_TP=-3.5:measured_LRA=9:measured_thresh=-32:offset=0:linear=true:print_format=summary"


def test_vocal_chain_substitutes_loudnorm_and_knobs():
    out = filter_chain.vocal_chain(
        in_label="0:a", out_label="v_post_eq",
        loudnorm_filter=LN, vocal_gain_db=0, presence_db=2,
    )
    assert out.startswith("[0:a]loudnorm=")
    assert "highpass=f=80" in out
    assert "equalizer=f=4000:width_type=q:w=1:g=2" in out
    assert "acompressor=threshold=-18dB:ratio=3:attack=5:release=80" in out
    assert "volume=0dB" in out
    assert out.endswith("[v_post_eq]")


def test_instrumental_chain_applies_gain():
    out = filter_chain.instrumental_chain(
        in_label="1:a", out_label="i_pre",
        loudnorm_filter=LN, instrumental_gain_db=-3,
    )
    assert out.startswith("[1:a]loudnorm=")
    assert "volume=-3dB" in out
    assert out.endswith("[i_pre]")


def test_reverb_chain_weights_dry_wet():
    out = filter_chain.reverb_chain(
        in_label="v_post_eq", out_label="v_post_rev",
        ir_input="2:a", reverb_wet=0.25,
    )
    assert "[v_post_eq]asplit=2[v_dry][v_for_rev]" in out
    assert "[v_for_rev][2:a]afir=irnorm=1[v_wet]" in out
    assert "weights=0.750 0.250" in out
    assert out.endswith("[v_post_rev]")


def test_reverb_chain_rejects_zero_wet():
    with pytest.raises(ValueError):
        filter_chain.reverb_chain(
            in_label="v", out_label="o", ir_input="2:a", reverb_wet=0.0,
        )


def test_ducking_chain_maps_duck_db_to_ratio():
    out = filter_chain.ducking_chain(
        vocal_label="v_post_eq", instrumental_label="i_pre",
        vocal_out="v_mix", ducked_out="i_ducked", duck_db=4,
    )
    # ratio = 1 + 4*0.5 = 3.0
    assert "sidechaincompress=threshold=0.05:ratio=3.000:attack=20:release=250" in out
    assert "[v_post_eq]asplit=2[v_mix][v_sc]" in out
    assert out.endswith("[i_ducked]")


def test_ducking_chain_zero_duck_clamps_to_unity():
    out = filter_chain.ducking_chain(
        vocal_label="v", instrumental_label="i",
        vocal_out="vm", ducked_out="id", duck_db=0,
    )
    assert "ratio=1.000" in out


def test_bus_chain_master_gain_and_limiter():
    out = filter_chain.bus_chain(
        vocal_label="v_mix", instrumental_label="i_ducked",
        out_label="out", master_gain_db=-1.5,
    )
    assert "amix=inputs=2:duration=longest:dropout_transition=2" in out
    assert "volume=-1.5dB" in out
    assert "alimiter=limit=0.94" in out
    assert out.endswith("[out]")


def test_assemble_joins_with_semicolons():
    a = filter_chain.vocal_chain(
        in_label="0:a", out_label="v_post_eq",
        loudnorm_filter=LN, vocal_gain_db=0, presence_db=2,
    )
    b = filter_chain.instrumental_chain(
        in_label="1:a", out_label="i_pre", loudnorm_filter=LN, instrumental_gain_db=0,
    )
    graph = filter_chain.assemble([a, "", b])  # empty fragment dropped
    assert graph == f"{a};{b}"
