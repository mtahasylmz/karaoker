"""Composable ffmpeg filter-graph fragments for the record-mix bus.

Each builder returns a single `;`-separated filter-graph string fragment with
its own input/output labels. `assemble()` glues fragments together with `;`
so the full chain is still one `-filter_complex` argument.

Kept string-only (no numpy, no ffmpeg exec) so tests can assert the exact
syntax for known knob values without spawning a subprocess.
"""

from __future__ import annotations


def vocal_chain(
    *,
    in_label: str,
    out_label: str,
    loudnorm_filter: str,
    vocal_gain_db: float,
    presence_db: float,
) -> str:
    """Pre-mix vocal chain: loudnorm → highpass → presence EQ → comp → gain."""
    return (
        f"[{in_label}]{loudnorm_filter},"
        f"highpass=f=80,"
        f"equalizer=f=4000:width_type=q:w=1:g={presence_db},"
        f"acompressor=threshold=-18dB:ratio=3:attack=5:release=80,"
        f"volume={vocal_gain_db}dB"
        f"[{out_label}]"
    )


def reverb_chain(
    *,
    in_label: str,
    out_label: str,
    ir_input: str,
    reverb_wet: float,
) -> str:
    """Convolution reverb via afir. Caller must ensure reverb_wet > 0.

    `ir_input` is the ffmpeg stream label for the plate IR input (e.g. "2:a").
    """
    if reverb_wet <= 0.0:
        raise ValueError("reverb_chain requires reverb_wet > 0; caller should skip this fragment")
    dry = max(0.0, 1.0 - reverb_wet)
    wet = reverb_wet
    return (
        f"[{in_label}]asplit=2[v_dry][v_for_rev];"
        f"[v_for_rev][{ir_input}]afir=irnorm=1[v_wet];"
        f"[v_dry][v_wet]amix=inputs=2:weights={dry:.3f} {wet:.3f}"
        f"[{out_label}]"
    )


def instrumental_chain(
    *,
    in_label: str,
    out_label: str,
    loudnorm_filter: str,
    instrumental_gain_db: float,
) -> str:
    """Pre-mix instrumental chain: loudnorm → gain."""
    return (
        f"[{in_label}]{loudnorm_filter},"
        f"volume={instrumental_gain_db}dB"
        f"[{out_label}]"
    )


def ducking_chain(
    *,
    vocal_label: str,
    instrumental_label: str,
    vocal_out: str,
    ducked_out: str,
    duck_db: float,
) -> str:
    """Split the vocal into mix + sidechain; duck the instrumental against it.

    Ratio mapping (1.0 + duck_db * 0.5) is a taste curve — real gain reduction
    depends on signal level, so duck_db is not a dB guarantee.
    """
    duck_ratio = max(1.0, 1.0 + duck_db * 0.5)
    return (
        f"[{vocal_label}]asplit=2[{vocal_out}][v_sc];"
        f"[{instrumental_label}][v_sc]sidechaincompress="
        f"threshold=0.05:ratio={duck_ratio:.3f}:attack=20:release=250"
        f"[{ducked_out}]"
    )


def bus_chain(
    *,
    vocal_label: str,
    instrumental_label: str,
    out_label: str,
    master_gain_db: float,
) -> str:
    """Final bus: amix → master gain → limiter."""
    return (
        f"[{vocal_label}][{instrumental_label}]"
        f"amix=inputs=2:duration=longest:dropout_transition=2,"
        f"volume={master_gain_db}dB,"
        f"alimiter=limit=0.94"
        f"[{out_label}]"
    )


def assemble(fragments: list[str]) -> str:
    """Join non-empty fragments with `;`."""
    return ";".join(f for f in fragments if f)


__all__ = [
    "vocal_chain",
    "reverb_chain",
    "instrumental_chain",
    "ducking_chain",
    "bus_chain",
    "assemble",
]
