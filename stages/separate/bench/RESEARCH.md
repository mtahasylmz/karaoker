# annemusic — Vocal Separation Models, State of the Art (Apr 2026)

**Scope.** 2-stem (vocals vs. instrumental), high-resolution audio (44.1 kHz stereo, full band). Target use: karaoke generation — clean instrumental for sing-over, vocal bleed penalized harder than reverb/harmony loss. Runtime budget ~30–120 s/track on a single consumer GPU (T4/L4) or CPU fallback.

## 1. Current SOTA landscape

The space bifurcated around 2023. The old MDX-Net / Demucs axis (CNN / hybrid CNN+Transformer, ~10M–40M params) has been overtaken on quality by a new **RoFormer** family (band-split transformers with RoPE) from ByteDance, plus the **SCNet** line. Everything interesting today is either a RoFormer variant or a band-split hybrid. Demucs is now the "fast baseline" rather than SOTA.

### Demucs family (Meta FAIR) — MIT
- **htdemucs** (v4 Hybrid Transformer): hybrid spectrogram + waveform U-Net with transformer cross-domain attention. ~41M params. 44.1 kHz stereo, 4 stems (vocals/drums/bass/other). Vocals SDR ≈ 8.3 dB (MUSDB18 alone); ~9.0 dB avg for the `_ft` fine-tuned variant on MUSDB-HQ ([Demucs repo](https://github.com/adefossez/demucs)).
- **htdemucs_ft**: same arch, fine-tuned per-stem; 4× slower inference.
- **htdemucs_6s**: adds piano + guitar; piano is unreliable per upstream notes.
- **Runtime**: real-time or faster on a T4 for a 3-min track. Very well-maintained `demucs` PyPI package, loads from PyTorch Hub cache.
- **Hard limit**: 7.8 s segment length — bounds VRAM but also a known quality ceiling.

### BS-RoFormer (ByteDance, Lu et al., ICASSP 2024) — MIT code, weights mixed
Band-Split RoPE Transformer: splits STFT into subbands, alternates intra-band and inter-band transformers ([arXiv 2309.02612](https://arxiv.org/abs/2309.02612)). 44.1 kHz stereo native.

- Small (L=6): 72M params, MUSDB18-HQ avg SDR 9.80 dB w/o extra data.
- Large (L=12): 93M, avg SDR 11.99 dB with extra training data (ByteDance internal corpus).
- Community checkpoints (via viperx / ZFTurbo) report vocal-stem SDR 12.9–13.0 on Multisong test (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`, `ep_368_sdr_12.9628.ckpt`). These numbers aren't strictly comparable to MUSDB-HQ SDR but are the de facto industry reference ([mvsep algo #34](https://mvsep.com/algorithms/34)).
- Inference for the L=6 variant fits in ~6–8 GB VRAM at 8 s chunks.
- **PyPI**: `BS-RoFormer` (lucidrains implementation, MIT). No official ByteDance weights; the usable weights come from community retraining — license is MIT in the code repos, but several of the checkpoint drops are silent on weight license. Treat as "practically usable, formally ambiguous."

### Mel-Band RoFormer (Wang et al., ByteDance; ISMIR 2023 LBD + [arXiv 2409.04702](https://arxiv.org/abs/2409.04702))
Same family as BS-RoFormer, but replaces the hand-designed band split with a **mel-scale projection** that overlaps subbands. Consistent +0.5 dB avg over BS-RoFormer, biggest gains on vocals.

- 44.1 kHz stereo main model: 105M params, vocals SDR **12.08 dB on MUSDB18-HQ alone**, 13.29 dB with extra data ([paper](https://arxiv.org/html/2409.04702v1)).
- 24 kHz mono small variant: 9.1M params, 11.01 dB vocals SDR.
- 24 kHz mono large: 50.7M params, 12.69 dB vocals SDR.
- **Kimberley Jensen's checkpoint** ([HF](https://huggingface.co/KimberleyJSN/melbandroformer), MIT) is the widely-used community vocals model — 44.1 kHz stereo, 8 s chunks (352,800 samples), performs *slightly better than the paper* because it was trained on more data. Intel also maintains an OpenVINO port.
- Inference API (raw torch): `model(torch.randn(B, 2, N_samples)) -> stems tensor`.

### SCNet (Tsinghua + Skywork AI, ICASSP 2024) — code open
Sparse Compression Network. Frequency-domain U-Net that splits spectrum into low/mid/high subbands with variable compression ratios ([arXiv 2401.13276](https://arxiv.org/abs/2401.13276)).

- 10.08M params — about ¼ of HT Demucs.
- MUSDB18-HQ: avg 9.0 dB, **vocals 9.89 dB**, drums 10.51, bass 8.82, other 6.76.
- SCNet-large: avg 9.69 dB.
- **CPU inference ~48% of HT Demucs wall time**. Best quality/compute ratio of any model here if you're CPU-bound.
- Community "SCNet XL IHF" checkpoint in ZFTurbo repo: avg SDR 10.09; "SCNet Masked XL IHF" 9.83.

### Apollo (Tsinghua, Li & Luo, ICASSP 2025, [arXiv 2409.08514](https://arxiv.org/abs/2409.08514))
Not a separation model per se — it's a band-sequence restoration model for lossy→lossless MP3 restoration. Relevant here because it can be chained after a separator to clean residual artifacts. Skip unless separator output has compression artifacts. Weights at [JusperLee/Apollo](https://huggingface.co/JusperLee/Apollo) (MIT).

### MDX-Net / MDX23 / KUIELAB-MDX-Net
Generation prior to RoFormer. Still competitive for specific use cases (e.g., `UVR-MDX-NET-Inst_HQ_4`, `MDX23C-8KFFT-InstVoc_HQ_2` — vocals SDR ~10–10.5 on Multisong). MDX23 pipeline is an ensemble of Demucs4 + MDX — effective but slow. No longer SOTA.

### ZFTurbo's Music-Source-Separation-Training (MIT code)
The practical hub. [Repo](https://github.com/ZFTurbo/Music-Source-Separation-Training) + [pretrained_models.md](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md). Supports `mdx23c`, `htdemucs`, `segm_models`, `bs_roformer`, `mel_band_roformer`, `swin_upernet`, `bandit(_v2)`, `scnet(_tran)`, `apollo`, `bs_mamba2`, `bs_conformer`, `bs_polarformer`. Top current vocal checkpoints on Multisong:

| Model (checkpoint) | Vocals SDR (Multisong) |
|---|---|
| BS PolarFormer (Mar 2025) | **11.00** |
| MelBand RoFormer (Kimberley) | 10.98 |
| BS RoFormer (viperx) | 10.87 |
| MDX23C | 10.17 |
| SegmModels ViTLarge23 | 9.77 |
| BS Mamba2 (Jan 2025) | 10.86 MUSDB / 8.83 Multisong |

(Multisong is harsher than MUSDB18-HQ test; subtract ~1–2 dB vs paper numbers.)

### UVR (Anjok07)
GUI wrapper. Current recommendation: RoFormer ensembles — typically BS-RoFormer + MelBand-RoFormer + SCNet XL IHF combined via [ZFTurbo ensemble scripts](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/ensemble.md). Achieves vocals SDR ~13.6 dB in community tests — but at 3× the cost.

### 2025–2026 entries worth knowing
- **Moises-Light** (WASPAA 2025, [arXiv 2510.06785](https://arxiv.org/abs/2510.06785)): band-split U-Net, ~2.2M params, avg SDR ~9.96 dB — matches HTDemucs at 1/20th the size.
- **Windowed Sink Attention Mel-RoFormer** ([arXiv 2510.25745](https://arxiv.org/abs/2510.25745)): replaces full temporal attention with localized windowed attention + attention sinks. Recovers 92% of Mel-RoFormer SDR at 44.5× fewer FLOPs. MIT. Cloud Run CPU candidate.
- **BS PolarFormer** (ZFTurbo, Mar 2025): polar coordinate positional embeddings, current top of the ZFTurbo vocal leaderboard (11.00 SDR).

## 2. Benchmarks

### MUSDB18-HQ vocals SDR — top results

| Model | Vocals SDR | Train data | Source |
|---|---|---|---|
| Mel-RoFormer (Wang) | **13.29** | MUSDB + extra | [paper](https://arxiv.org/html/2409.04702v1) |
| BS-RoFormer L=12 | ~12.8 (avg 11.99) | MUSDB + extra | [paper](https://arxiv.org/abs/2309.02612) |
| Mel-RoFormer (Wang) | **12.08** | MUSDB-HQ only | paper |
| BS-RoFormer L=6 | ~11.5 (avg 9.80) | MUSDB-HQ only | paper |
| SCNet-large | ~10.5 (avg 9.69) | MUSDB-HQ only | [paper](https://arxiv.org/abs/2401.13276) |
| SCNet | 9.89 (avg 9.0) | MUSDB-HQ only | paper |
| HT Demucs v4 ft | ~9.4 (avg 9.0) | MUSDB + 800 songs | Demucs repo |
| HT Demucs v4 | ~8.3 | MUSDB + 800 songs | Demucs repo |

Notes on metrics:
- **uSDR** (utterance-level SDR — a.k.a. "global SDR") is what most papers post-2023 report. **cSDR** (chunk-level) is the older MUSDB eval protocol (museval). SI-SDR de-weights scale; not used in MUSDB evals.
- SDR on MUSDB18-HQ is saturating — perceptual differences shrink above ~10 dB. ISMIR 2025 has a [paper arguing SDR averages hide meaningful perceptual errors](https://ismir2025program.ismir.net/poster_221.html).
- Newer benchmarks: **MoisesDB** (Moises.ai, 2023) and **MedleyDB** increasingly reported — SCNet generalizes MUSDB→MoisesDB at 10.33 dB.

## 3. Practical recommendations for annemusic

Given: 44.1 kHz stereo in, clean instrumental out, CPU or T4/L4 GPU, 30–120 s/track budget, MIT-like license required (consumer product).

**Ranked picks to bench.**

1. **MelBand RoFormer — Kimberley Jensen checkpoint** ([HF](https://huggingface.co/KimberleyJSN/melbandroformer))
   - *Why*: best quality/effort ratio. Vocals-specialist (not 4-stem — perfect for annemusic). MIT code, 44.1 kHz stereo native, 8 s chunks.
   - *Cost*: ~1:49 wall-clock for a 3-min track on M3 Max per nomadkaraoke benchmarks; on a T4 expect 30–50 s. ~6–8 GB VRAM at 8 s chunk, num_overlap=2.
   - *Risk*: Kimberley model card is empty — license inferred from related repos. Verify before shipping.

2. **BS-RoFormer — viperx checkpoint** (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`)
   - *Why*: battle-tested in UVR and MVSep, vocals SDR ~12.97 on Multisong. Slightly better than MelBand on bass-heavy genres, slightly worse on vocals in aggregate — very close.
   - *Risk*: weights released by a pseudonymous community author. Code MIT. Mirror to our own bucket.

3. **SCNet (XL IHF from ZFTurbo)** as a CPU fallback
   - *Why*: 10M params, CPU inference at ~half of HT Demucs, avg SDR 10.09 on Multisong. Graceful degradation path if GPU quotas fail.
   - *Not for primary bench* — SOTA quality is clearly at RoFormer tier.

**Keep htdemucs as the "guaranteed works" floor** — it's what the MVP ships; a RoFormer regression on some genre would still leave a usable fallback.

**Do not bench** BS-Mamba2 / BS-PolarFormer / BS-Conformer as first picks — less community validation, unclear license trail.

**Ensembles** (BS + Mel + SCNet) hit ~13.6 dB vocal SDR but take 3× the budget. "Premium render" tier only.

**License summary.** All MIT-family candidates. Watchout: ByteDance never released their own weights; every usable RoFormer checkpoint is community-retrained. For legal review, plan to retrain a vocals-only model on MUSDB18-HQ + MoisesDB (Moises dataset is CC-BY-NC-SA — OK for internal but not commercial training without license). Fallback: HT Demucs weights are Meta-released under MIT.

## 4. Integration notes

### `audio-separator` (nomadkaraoke, MIT) — [PyPI](https://pypi.org/project/audio-separator/) / [GitHub](https://github.com/nomadkaraoke/python-audio-separator)
Recommended integration path for `stages/separate`. Python ≥ 3.10, CUDA 11.8/12.2, CoreML on Apple Silicon, DirectML on Windows.

```python
from audio_separator.separator import Separator
s = Separator(output_dir="/tmp/out")
s.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
[vocals, instrumental] = s.separate("input.wav")
```

Takes file paths only — wrap with a tempfile if you have numpy. Auto-downloads UVR-hosted weights; cache to `/tmp` on Cloud Run or bake into Docker. Produces stereo 44.1 kHz WAV output.

### `demucs` PyPI — baseline
`python -m demucs --two-stems=vocals -n htdemucs file.wav`. Takes file paths; `demucs.apply.apply_model` accepts torch tensors if you want to go low-level. Weights auto-download from `dl.fbaipublicfiles.com` — network-gate in Cloud Run.

### Raw ZFTurbo training repo — custom inference
`inference.py --config_path configs/config_vocals_mel_band_roformer.yaml --model_path mel.ckpt --input_folder in --store_dir out`. More control but more glue. Use if `audio-separator` doesn't expose a knob you need.

### Gotchas
- **torch version pinning**. Demucs and the RoFormer repos all have tight torch pins. `audio-separator` brings its own pinned stack — build a dedicated venv per stage (annemusic already does via `uv sync --package annemusic-stage-separate`).
- **Weight hosting**. UVR-community weights are scattered across MEGA, Google Drive, HuggingFace, and SourceForge mirrors. `audio-separator` resolves via `models.json` URLs that **have changed hosts before** — mirror to GCS (`gs://annemusic-weights/separate/`) and override the download URL.
- **Chunk / overlap tuning**. MelBand RoFormer defaults to 8 s chunks; `num_overlap=4` is the sweet spot.
- **Cloud Run cold start**. Bake weights into the image (~600 MB for MelBand RoFormer).
- **Stereo preservation**. BS/Mel-RoFormer preserve stereo.
- **whisperx downstream**. All three recommended models output WAV 44.1 kHz stereo; whisperx resamples to 16 kHz — no prep needed.

---

## TL;DR — what to bench first

Bench **MelBand RoFormer (Kimberley)** and **BS-RoFormer (viperx ep_317)** via the `audio-separator` PyPI package, on 4 fixtures covering (pop, rock, electronic, rap). Measure SDR on MUSDB18-HQ dev + wall-clock on the target T4/L4. Keep **htdemucs** as the fallback code path; skip ensembles and Demucs-6s for MVP. If you end up CPU-bound in production, swap in **SCNet XL IHF** before falling back to htdemucs — it's the only model that matches htdemucs speed at RoFormer-ish quality. Revisit **Windowed Sink Attention Mel-RoFormer** in Q3 2026.

---

## Sources

- [paperswithcode — Music Source Separation on MUSDB18-HQ](https://paperswithcode.com/sota/music-source-separation-on-musdb18-hq)
- [SigSep — MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html)
- [arXiv 2309.02612 — BS-RoFormer](https://arxiv.org/abs/2309.02612)
- [arXiv 2310.01809 — Mel-Band RoFormer](https://arxiv.org/abs/2310.01809)
- [arXiv 2409.04702 — Mel-RoFormer vocal separation (Wang)](https://arxiv.org/abs/2409.04702)
- [arXiv 2401.13276 — SCNet](https://arxiv.org/abs/2401.13276)
- [arXiv 2409.08514 — Apollo](https://arxiv.org/abs/2409.08514)
- [arXiv 2510.06785 — Moises-Light](https://arxiv.org/abs/2510.06785)
- [arXiv 2510.25745 — Windowed Sink Attention Mel-RoFormer](https://arxiv.org/abs/2510.25745)
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [ZFTurbo — pretrained_models.md](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)
- [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
- [KimberleyJensen/Mel-Band-Roformer-Vocal-Model](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)
- [KimberleyJSN/melbandroformer (HF)](https://huggingface.co/KimberleyJSN/melbandroformer)
- [Intel OpenVINO Mel-RoFormer](https://huggingface.co/Intel/vocals_mel_band_roformer_kimberleyJSN_openvino)
- [JusperLee/Apollo (HF)](https://huggingface.co/JusperLee/Apollo)
- [facebookresearch/demucs](https://github.com/adefossez/demucs)
- [MVSep algo 34 — BS RoFormer](https://mvsep.com/algorithms/34)
- [audio-separator (PyPI)](https://pypi.org/project/audio-separator/)
- [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [anjok07/ultimatevocalremovergui](https://github.com/anjok07/ultimatevocalremovergui)
- [vocalremover.cloud UVR 2025 guide](https://vocalremover.cloud/blog/uvr-best-model-aug-2025)
- [ISMIR 2025 — Perceptual errors beyond SDR](https://ismir2025program.ismir.net/poster_221.html)
- [ISMIR 2025 — Spatial info preservation](https://ismir2025program.ismir.net/poster_300.html)
- [Moises — Research Innovations 2025](https://music.ai/blog/research/Moises-Research-Innovations-2025/)
