# stages/record-mix — v2 research notes

Research captured 2026-04-23. Sources cited inline.

## TL;DR — three highest-leverage upgrades

1. **Time alignment via GCC-PHAT against `vocals.wav` from `stages/separate`.**
   Cross-correlate the user's recording against the isolated vocal stem (or
   instrumental if the user muted themselves). Whitened cross-spectrum, ±800
   ms search window. <10 ms precision, a few hundred ms of CPU. Fixes the
   dominant karaoke-app complaint. Every real karaoke app ships some
   version (Smule calls it the "Vocal Match Slider" — just manual).

2. **Loudness + sidechain ducking filter graph**, not raw `amix`.
   Two-pass `loudnorm` on vocal (`I=-16 LUFS, TP=-1.5, LRA=11`) and
   instrumental (`I=-14 LUFS`), then ffmpeg `sidechaincompress` so the
   instrumental ducks 3–6 dB during vocal phrases, `alimiter` at -0.5 dBTP
   on the bus. All native ffmpeg. This is what a "pro-sounding" karaoke
   mix actually is.

3. **Re-run Demucs on the user recording to strip speaker bleed.**
   Reuse `stages/separate`'s Demucs v4 — one extra pass, ~20–40 s CPU on
   a 3-min mono take. Materially better than any linear AEC you'd tune by
   hand; subsumes the WebRTC AEC3 problem that music breaks it. Add a
   `clean_bleed: bool` flag on the record-mix request. Re-run GCC-PHAT
   *after* separation for sample-accurate alignment (Demucs adds 1–3 ms
   phase shift).

---

## 1. Time alignment / sync

- **Problem.** `MediaRecorder` timestamps do not expose capture latency.
  Bluetooth adds 150–400 ms codec-round-trip delay. Raw mixing offsets the
  vocal noticeably.
- **Baseline.** Trust browser timestamps — broken.
- **SOTA.** Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
  whitens the cross-spectrum so the correlation peak is a delta regardless
  of spectral coloration — canonical for reverberant rooms.
  - [IEEE GCC-PHAT overview](https://ieeexplore.ieee.org/document/8949522/)
  - [Reference impl](https://github.com/MinAungThu/GCC-PHAT)
- **Recommendation.** GCC-PHAT vs `vocals.wav`. Fall back to correlating
  vs instrumental if peak SNR < 3 dB. Optionally emit a 20 ms 1 kHz
  pre-roll beep in the browser for a deterministic anchor.

## 2. Acoustic echo / bleed cancellation

- **Browser `echoCancellation: true`** runs WebRTC AEC3 but is tuned for
  speech double-talk; it downmixes to mono and notches stable vocal
  harmonics. Fine for voice chat, mediocre for music.
  [Switchboard AEC3 explainer](https://switchboard.audio/hub/how-webrtc-aec3-works/),
  [discuss-webrtc music thread](https://groups.google.com/g/discuss-webrtc/c/s-VwouG-Tco).
- **Classical AEC:** NLMS/RLS filters with instrumental as known reference
  (`speexdsp-python`, `adaptfilt`). Breaks on speaker non-linearity and
  moving sources.
- **Neural AEC:** ICASSP 2023 AEC Challenge winners
  ([arxiv 2309.12553](https://arxiv.org/abs/2309.12553)) — too much
  scope.
- **Recommendation.** Do not build an AEC stage. Expose a "headphones?"
  checkbox to the user; set `echoCancellation: false` when headphones,
  `true` otherwise. Handle bleed via re-separation (§3).

## 3. Vocal isolation on the recording (bleed cleanup)

- **Tooling.** Reuse existing Demucs v4 Hybrid Transformer. Keep
  `vocals` stem, discard `no_vocals`.
- **Cost.** One extra Demucs pass per record-along. ~20–40 s CPU on a
  3-minute mono recording on Cloud Run. Trivial code change (new
  invocation of existing model).
- **Single biggest quality win** after sync. Ship behind `clean_bleed`
  flag, default `true` unless user ticked "headphones".

## 4. Level / loudness management

- **EBU R128 / ITU-R BS.1770** loudness normalization via ffmpeg
  `loudnorm`. Two-pass is ~1 LU more accurate than single-pass.
  - [Loudnorm deep-dive](http://k.ylo.ph/2016/04/04/loudnorm.html)
  - [ffmpeg-normalize](https://github.com/slhck/ffmpeg-normalize)
- **Sidechain ducking** via ffmpeg `sidechaincompress`.
  - [Filter docs](https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Audio/sidechaincompress.html)
  - [Example](https://ffmpeg.org/pipermail/ffmpeg-user/2018-August/040933.html)
- **Recommended filter graph:**
  1. Two-pass loudnorm vocal → `I=-16 LUFS, TP=-1.5, LRA=11`
  2. Two-pass loudnorm instrumental → `I=-14 LUFS`
  3. `sidechaincompress` with vocal as side-chain, `threshold=0.1:ratio=4:attack=20:release=250`
  4. `amix=weights=1 0.9`
  5. `alimiter` at `-0.5 dBTP`

## 5. Pitch correction

### F0 estimation
- `librosa.pyin` — CPU-cheap, SP-based, noisy under bleed.
- **CREPE** — deep CNN, SOTA on clean monophonic; slow. [repo](https://github.com/marl/crepe)
- **SPICE** — self-supervised, noise-robust.
  [Google SPICE](https://research.google/blog/spice-self-supervised-pitch-estimation/)
- **RMVPE (2023)** — polyphonic-robust, beats pYIN/CREPE on bleed.
  [arxiv 2306.15412](https://arxiv.org/abs/2306.15412)
- **FCPE (2025)** — 5× faster than RMVPE, 77× faster than CREPE,
  comparable RPA. Current winner for real-time CPU.
  [arxiv 2509.15140](https://arxiv.org/abs/2509.15140),
  [CNChTu/FCPE](https://github.com/CNChTu/FCPE)

### Scale detection (for "snap")
- Krumhansl-Schmuckler on chromagram. Hand-roll in ~30 lines of librosa to
  avoid Essentia's AGPL (see §8).

### Re-synthesis
- **PSOLA** — good formants for small shifts.
- **Phase vocoder** — smears transients.
- **WORLD** — highest formant fidelity (analysis/synthesis with explicit
  F0/spectral envelope/aperiodicity).
  [MDPI WORLD paper](https://www.mdpi.com/2078-2489/13/3/103)

### Recommendation
- **"smooth"** — RubberBand (`pyrubberband` + `--formant`), pitch-smooth
  deviations < 50 cents toward centered-median F0.
- **"snap"** — FCPE → librosa/Krumhansl scale → snap only when deviation
  < 80 cents (avoid "dalek voice" on intentional runs/bends) → resynth
  via RubberBand. Alternative: `x42-autotune` / `fat1.lv2` via `jalv`
  subprocess (known-good DSP, accepts MIDI-defined scales).

## 6. Spatialization / "sitting in the mix"

- Dry vocal over produced music sounds amateur.
- **Fixed chain:** `highpass=f=80 → acompressor=threshold=-18dB:ratio=3 →
  equalizer=f=4000:w=1:g=2 → afir=<plate_ir.wav>:wet=0.12`
- Ship **one plate-reverb IR** as a stage asset (Studio Nord Bremen free
  EMT-plate IRs exist).
- `afir` is ffmpeg-native since 4.3 — no new dependencies.
  [IR.lv2 ref](https://github.com/Anchakor/ir.lv2)

## 7. Browser-side vs server-side

- `AudioWorkletProcessor` runs at 128-sample quanta (~2.7 ms @ 48 kHz),
  can host WASM-compiled DSP.
  [MDN AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet),
  [Emscripten Wasm Worklets](https://emscripten.org/docs/api_reference/wasm_audio_worklets.html)
- Real-time pitch shift in browser exists:
  [phaze](https://github.com/olvb/phaze).
- **Known latency floor** — 25–60 ms on desktop, worse on Android.
  Graph cannot reliably see output-hardware latency.
  [jefftk.com/p/browser-audio-latency](https://www.jefftk.com/p/browser-audio-latency)
- **Recommendation.** Do heavy DSP server-side (sync, bleed cleanup,
  pitch correction). Do light tunable FX browser-side (gain, EQ, reverb
  wet, ducking depth) via Web Audio — see "Interactive mixing" below.
- Serious low-latency monitoring is a native-app path (Oboe / AVAudioEngine).
  Smule's Oboe migration dropped average latency 109 ms → 39 ms.
  [Android Dev Blog](https://android-developers.googleblog.com/2022/02/smule-adopts-googles-oboe-to-improve.html)

## 8. What real karaoke apps ship

- **Smule** — Oboe for low-latency capture/playback; manual "Vocal Match
  Slider" for post-hoc alignment; server-side mix with reverb/EQ preset
  "Styles"; explicitly tells users Bluetooth breaks sync.
- **StarMaker** — user-tunable latency slider, capture-side FX, pitch
  correction toggle.
- **WeSing / Karafun / Singa** — scarce public material; reverse-
  engineering suggests the shared pattern: **dry capture + server-side
  mix with preset chains + manual sync-offset slider**.
- **Implication.** Auto-GCC-PHAT alignment + a manual offset slider +
  tunable preset chain already matches or beats what these ship for
  family-scale.

---

## Interactive mixing — v2 direction: parameterized server render

**Decision (2026-04-23):** v2 keeps **all DSP on the server** and exposes
tunable parameters through the request contract. UI sliders → debounced
POST to `/process` → server re-renders the full chain → frontend swaps
`<audio>` source. Accept the 2–10 s render cost for a 3-min song for
now.

Rationale: avoids owning a browser-side DSP stack; avoids Web Audio's
25–60 ms latency floor being audible on autotune monitoring; keeps one
code path for render logic.

**Tunable params to expose on `record_mix_request`:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `vocal_gain_db` | number | 0 | pre-mix vocal level |
| `instrumental_gain_db` | number | 0 | pre-mix instrumental level |
| `reverb_wet` | number 0–1 | 0.12 | convolution reverb mix |
| `duck_db` | number 0–12 | 4 | sidechain compressor depth |
| `presence_db` | number -3..+6 | 2 | EQ lift at ~4 kHz |
| `autotune` | `off` / `smooth` / `snap` | `off` | already in contract |
| `clean_bleed` | bool | true | run Demucs on recording first |

**Server filter graph per render** (all native ffmpeg + optional
Demucs/RubberBand prepass):
1. (optional) Demucs on recording → clean vocal stem.
2. GCC-PHAT align vocal vs `vocals.wav` → apply offset.
3. (optional) RubberBand smooth / x42-autotune snap on vocal.
4. Two-pass loudnorm on vocal (-16 LUFS) and instrumental (-14 LUFS).
5. `highpass=80, acompressor, equalizer@4k, afir=plate.wav:wet=<reverb_wet>`
   on vocal; `volume=<vocal_gain_db>dB` at end.
6. `sidechaincompress` with vocal as side-chain, ratio tied to `duck_db`.
7. `volume=<instrumental_gain_db>dB` on instrumental.
8. `amix`, `alimiter` at -0.5 dBTP, `libmp3lame -q:a 2`.

**Deferred — not v2:** browser-side Web Audio graph for real-time
parameter response, `OfflineAudioContext` export, AudioWorklet ducking.
Revisit once the server version ships and we have a UX signal.
Research notes on that path preserved in git history if needed
(see prior revision of this file).

---

## Concrete tool table

| Purpose | Tool | License |
|---|---|---|
| Sync | scipy GCC-PHAT (~30 LOC) | BSD |
| Bleed cleanup | Demucs v4 (already in `stages/separate`) | MIT |
| F0 | FCPE (torchfcpe on PyPI) | MIT |
| Key detection | librosa + hand-rolled Krumhansl | BSD |
| Pitch shift | RubberBand via pyrubberband | **GPLv2** ⚠ |
| Autotune (alt) | x42-autotune / fat1.lv2 via jalv | GPLv2 |
| Convolution reverb | ffmpeg `afir` (built-in) | LGPL |
| Loudness / ducking / limit | ffmpeg native filters | LGPL |
| AEC (if needed) | speexdsp-python | BSD |
| Browser mix | Web Audio API + AudioWorklet | — |

**License hazards:**
1. **RubberBand is GPLv2** — fine for server-side batch. Embedding in a
   mobile client requires commercial licence from Breakfast Quay.
   [license page](https://breakfastquay.com/rubberband/license.html)
2. **Essentia is AGPL-3.0** — network clause affects hosted services.
   Hand-roll KS from librosa instead.
3. LV2 autotune plugins are all GPL — fine as separate processes via
   subprocess; linkage issue only if embedded.

---

## Reference index

- Sync: [GCC-PHAT IEEE](https://ieeexplore.ieee.org/document/8949522/) · [ref impl](https://github.com/MinAungThu/GCC-PHAT)
- AEC: [Switchboard AEC3](https://switchboard.audio/hub/how-webrtc-aec3-works/) · [discuss-webrtc music](https://groups.google.com/g/discuss-webrtc/c/s-VwouG-Tco) · [ICASSP 2023 AEC Challenge](https://arxiv.org/abs/2309.12553)
- Separation: [Demucs](https://pypi.org/project/demucs/) · [UVR MDX discussion](https://github.com/Anjok07/ultimatevocalremovergui/discussions/444)
- Loudness: [loudnorm deep-dive](http://k.ylo.ph/2016/04/04/loudnorm.html) · [ffmpeg-normalize](https://github.com/slhck/ffmpeg-normalize)
- Ducking: [sidechaincompress](https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Audio/sidechaincompress.html) · [example](https://ffmpeg.org/pipermail/ffmpeg-user/2018-August/040933.html)
- F0: [CREPE](https://github.com/marl/crepe) · [SPICE](https://research.google/blog/spice-self-supervised-pitch-estimation/) · [RMVPE](https://arxiv.org/abs/2306.15412) · [FCPE](https://arxiv.org/abs/2509.15140) · [CNChTu/FCPE](https://github.com/CNChTu/FCPE)
- Resynth: [Stanford EE264 phase-vocoder/PSOLA](https://web.stanford.edu/class/ee264/projects/EE264_w2015_final_project_kong.pdf) · [WORLD](https://www.mdpi.com/2078-2489/13/3/103)
- Autotune: [x42/fat1.lv2](https://github.com/x42/fat1.lv2)
- Key detection: [Essentia Key (AGPL)](https://essentia.upf.edu/reference/streaming_Key.html) · [jackmcarthur/musical-key-finder](https://github.com/jackmcarthur/musical-key-finder)
- Reverb: [IR.lv2](https://github.com/Anchakor/ir.lv2) · [x42/convoLV2](https://github.com/x42/convoLV2)
- Browser: [AudioWorklet MDN](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet) · [phaze](https://github.com/olvb/phaze) · [Web Audio latency](https://www.jefftk.com/p/browser-audio-latency)
- Karaoke apps: [Smule + Oboe](https://android-developers.googleblog.com/2022/02/smule-adopts-googles-oboe-to-improve.html) · [Smule sync slider](https://smule.zendesk.com/hc/en-us/articles/360027766671-How-to-fix-sync-issues-on-Android)
- Licence: [RubberBand licence](https://breakfastquay.com/rubberband/license.html)
