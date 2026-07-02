# QA: pipeline

E2E through the CLI only. Needs a CUDA box (RunPod fine), `ANNEMUSIC_FIXTURE`
set to a real music video (~3-5 min, sung vocals), and its `lyrics.txt`.

## Procedure

1. `annemusic $ANNEMUSIC_FIXTURE -o /tmp/qa-run` — expect exit 0 in wall time
   under ~10 min on an RTX-class GPU.
2. Play `/tmp/qa-run/instrumental.wav`: vocals audibly removed, music intact.
3. Play `/tmp/qa-run/vocals.wav`: vocals audible, music suppressed.
4. Open `lyrics.ass` in a text editor: Dialogue lines carry `\kf` tags; words
   are the actual song lyrics, not gibberish.
5. Overlay check: `ffplay -vf "ass=/tmp/qa-run/lyrics.ass" $ANNEMUSIC_FIXTURE`
   — word fills track the singing within ~0.5 s through verse and chorus;
   no frozen highlight during instrumental breaks.
6. `manifest.json`: language matches the song; `vocal_activity` regions line
   up with audible instrumental breaks; word count plausible for the song.
7. Re-run with `--language <wrong-lang>` and confirm it still exits 0
   (hint honored, garbage-in tolerated, no crash).
8. Re-run into the same dir without `--force`: refused. With `--force`: clobbers.
9. `annemusic missing.mp4`: non-zero exit, readable one-line error, no traceback.

Record: fixture used, GPU, wall time per run, and any word-timing drift
observed in step 5.
