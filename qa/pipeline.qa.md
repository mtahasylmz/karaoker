# QA: pipeline

E2E through the CLI only. Needs a CUDA box (RunPod fine), `ANNEMUSIC_FIXTURE`
set to a real music video (~3-5 min, sung vocals), and its `lyrics.txt`.

## Procedure

1. `annemusic $ANNEMUSIC_FIXTURE -o /tmp/qa-run` — expect exit 0 in wall time
   under ~10 min on an RTX-class GPU.
2. Play `/tmp/qa-run/instrumental.wav`: vocals audibly removed, music intact.
3. Play `/tmp/qa-run/vocals.wav`: vocals audible, music suppressed.
4. Open `lyrics.ass` in a text editor: one Dialogue line per lyric line,
   timed; no `\kf` tags; the text is the actual song lyrics, not gibberish.
5. Overlay check: `ffplay -vf "ass=/tmp/qa-run/lyrics.ass" $ANNEMUSIC_FIXTURE`
   — each line appears/clears on the beat within ~0.5 s through verse and
   chorus; no stale line frozen on screen during instrumental breaks.
6. `manifest.json`: language matches the song; `vocal_activity` regions line
   up with audible instrumental breaks; word count plausible for the song.
7. Re-run with `--language <wrong-lang>` and confirm it still exits 0
   (hint honored, garbage-in tolerated, no crash).
8. Re-run into the same dir without `--force`: refused. With `--force`: clobbers.
9. `annemusic missing.mp4`: non-zero exit, readable one-line error, no traceback.

## Known-lyrics timing (LRCLIB)

10. Re-run with `--artist "$ANNEMUSIC_ARTIST" --title "$ANNEMUSIC_TITLE"`.
    `manifest.source` == `"lrclib"`. This path skips ASR + alignment, so it
    is *fast* (separation + VAD only). Overlay-check (`ffplay -vf ass=...`):
    **the whole point** — line onsets should now land on the beat noticeably
    better than the ASR run, since they come from human LRC timestamps.
    Line-level display (no word fill); judge whether the line changes read
    as karaoke.
11. Re-run with a nonsense `--artist`/`--title`: `manifest.source` == `"asr"`,
    all four artifacts still present (graceful fallback, no crash).
12. Re-run the real song with `--no-lyrics-fetch`: `source` == `"asr"` (flag
    wins over an available match).

Record: fixture used + its artist/title, GPU, wall time per run, LRCLIB
match hit/miss, and side-by-side feel of the ASR run (step 5) vs the LRCLIB
run (step 10) — which one earns "loved".
