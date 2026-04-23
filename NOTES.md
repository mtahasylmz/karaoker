# annemusic — future work

## 1. Better transcription quality

Current: `small` whisper model, auto language detection, transcribing the demucs-vocals stem.
Known cheap wins, roughly in order of impact:

- **Bigger Whisper model.** `WHISPER_MODEL=medium` for ~2× better accuracy on non-English (esp.
  Turkish), `large-v3` for the best quality. Env-var only, no code change; just bump the worker
  config and accept the ~2–4× slower run. `large-v3` needs ~3 GB RAM, still fits our `--memory 8Gi`.
- **Pin the language.** Passing `language="tr"` (or letting user choose) to
  `fw_model.transcribe(...)` skips detection and avoids misclassifying Turkish as
  French/English/etc. The UI already knows who's uploading; could default from browser locale
  or add a dropdown.
- **Transcribe the full mix, not just the vocals stem.** Demucs adds artifacts that sometimes
  hurt recognition. Try the mix; fall back to vocals if SNR looks poor. Or transcribe both and
  pick the result with higher confidence.
- **`initial_prompt` with known lyrics snippet** — a few words of correct lyrics bias whisper
  strongly toward the right vocabulary.
- **`condition_on_previous_text=False`** is already good; consider `temperatures=[0.0]` to stop
  whisper from hallucinating on noisy stretches.
- **Post-process**: strip obvious repeats (whisper loves to loop), collapse duplicate adjacent
  words.

## 2. Known lyrics → better everything

If the user provides the lyrics up front, we can:

- Use them as `initial_prompt` (see above).
- **Skip transcription entirely** and only do forced alignment: feed the provided lyrics as
  "segments" to `whisperx.align(...)`. This is the exact function we already call; it accepts
  arbitrary text. Result is perfect lyrics with alignment-grade timings — the best karaoke.
- Source options, roughly fastest → most work:
  1. User pastes lyrics in a textarea at upload time. Simplest. Probably what to build first.
  2. Fetch from a lyrics API:
     - **Musixmatch** has synced lyrics (LRC format with `[mm:ss.xx]` timestamps) — already
       word-ish-timed. Would let us skip both whisper and align if we trust the timings. Paid.
     - **LRCLIB** (lrclib.net) — free, community LRCs, good coverage for popular songs.
     - **Genius** — lyrics only, no timestamps. Useful as `initial_prompt` but still need align.
  3. Audio fingerprinting to auto-identify the song (ACRCloud, AudD, Shazam), then fetch lyrics
     from one of (2). One extra step, but zero user input.

Recommended stack for v2: LRCLIB for auto-lookup by (title, artist) the user provides, falling
back to whisper if not found. Musixmatch only if we need coverage LRCLIB lacks.

## 3. Searchable video names

Current: objects are `videos/{sha256}.mp4`, Redis stores `video:{sha256}` hashes — meaningless
to humans. The user's "my karaokes" list shows hex prefixes.

Quick win (no new infra):
- Store a `title` field in `video:{sha256}` (from the uploaded filename, stripped of extension
  and common cruft). Already have `content_type`; easy to add the filename from the frontend's
  `file.name`. Render that in history instead of `sha256[:12]`.
- Add an optional "song title" / "artist" input on the upload form — user can set clean
  metadata if they care. Stored in same hash.

For actual search across users (find a song someone else karaoked already):
- **Upstash Search** is literally their full-text product (built on Redis). Index `title`,
  `artist`, `lyrics` (if we have them), `username`. `POST /search/index/{index}/documents`,
  query with `?query=...`. Cheap, in the same data plane as our existing Redis.
- Alternative: **Upstash Vector** if we go the semantic-search route (embed lyrics, find by
  meaning). Overkill for v1.
- Fallback without either: `SCAN video:* + MATCH` in Redis — fine at family scale.

## 4. Record-along (sing with the video)

Browser-side only, no pipeline impact:
- `navigator.mediaDevices.getUserMedia({audio: true})` → `MediaRecorder` producing webm/opus.
- UI: red "record" button next to the video player. Start on button click; record while the
  `<video>` plays; stop on pause/end.
- Sync: store the blob client-side, let the user download it, or PUT to GCS as
  `recordings/{sha}/{ts}.webm`.
- Mix (later): ffmpeg merge the user's recorded vocal with the `no_vocals.wav` instrumental →
  "your version" mp3. Done worker-side if we keep the instrumental in GCS.
- First cut: just record + download. Mix in v2.

Watch out for: iOS Safari's `getUserMedia` quirks on background tabs, and permission prompts
that newcomers won't know what to do with — show a clear explainer.

## Phase C issues to revisit

- **Duplicate "workflow started" log lines.** `@upstash/workflow`'s `serve()`
  handler re-enters the top of the workflow function once per `context.call`
  — any log statement before the first `context.run(...)` or `context.call()`
  prints 5× per pipeline run. Fix: wrap the initial log in
  `context.run("init", () => log.info(...))` so it runs exactly once. Trivial,
  left as a polish pass.
- **qstash-cli dev squats ports 8080 AND 8081** (the second is presumably its
  admin/metrics endpoint). API was pushed to 8082. Document this in every
  stage's `CLAUDE.md` so future sessions don't re-trip over it.
- **qstash-cli dev signing keys rotate per run** (`QSTASH_CURRENT_SIGNING_KEY`
  / `QSTASH_NEXT_SIGNING_KEY`). Restarting the CLI requires copying the new
  keys into `.env` and restarting every service that reads them. A small
  startup shim that exports these from the CLI's stdout would save a minute
  per restart; nice-to-have.
- **Vite binds IPv6-only on localhost** (`[::1]:5173`). `curl 127.0.0.1:5173`
  fails; `curl localhost:5173` works. Not a real bug, but worth remembering
  when scripting health checks.
- **`/dev/trigger` bypasses `objectExists()` in local mode.** The skip is
  gated on `NODE_ENV=local`; in prod the real GCS check fires. Keep the
  bypass narrow — don't let it leak into any other endpoint.
- **Log feed needed `overflow-wrap: anywhere` on `.msg`**. Long SHA strings
  were pushing the flex row wider than the panel and clipping the timestamp
  + stage columns off-screen. Fixed in styles.css; leave a note for any
  future components that render long opaque strings.

## 5. Incidental cleanups (not urgent)

- Drop `whisperx` from `requirements.txt` if we stop using `whisperx.align` in a future
  version. Right now we still call it; keep it pinned for cache hits.
- Delete the dormant `youtube-cookies` Secret Manager secret — unused since the upload pivot.
- Add a GCS lifecycle rule on `uploads/` to auto-delete sources after N days.
- Switch `/ping` to a proper Cloud Run startup+liveness probe config, now that we know the
  front-door caveat.
