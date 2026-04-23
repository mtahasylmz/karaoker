# annemusic

Upload a music video, get back a karaoke-overlaid playback (vocals removed, lyrics synced, sing along). Family-scale.

Full project docs, architecture, scars, and phase plan: **[CLAUDE.md](./CLAUDE.md)** and the plan at `~/.claude/plans/glowing-gathering-ocean.md`.

## Quick start (local)

```bash
cp .env.example .env
# Fill in UPSTASH + QSTASH creds from your Upstash dashboard
# and GOOGLE_APPLICATION_CREDENTIALS pointing at a GCP SA key.

pnpm install
pnpm contracts:build
pnpm dev            # docker compose up — brings up qstash dev + every stage
```

Then open http://localhost:5173 (web dev server — exists after Phase C).

## Per-stage development

Each stage has its own `CLAUDE.md` with its contract link, chosen model, local-test command, and bench harness. Work on a stage in its own agent session:

- `stages/separate/CLAUDE.md`
- `stages/transcribe/CLAUDE.md`
- `stages/align/CLAUDE.md`
- `stages/compose/CLAUDE.md`
- `stages/record-mix/CLAUDE.md`

## Deploy

```bash
bash infra/setup.sh            # one-time per GCP project
bash infra/deploy-stage.sh <stage-or-app-name>
```

## Useful tools

- `pnpm logs --stage <name> --follow` — live-tail one stage's Redis Stream
- `pnpm logs --job <id>` — reconstruct all stages of one job
- `bash infra/wipe.sh --yes` — nuke all dev state (Redis keys + GCS uploads/videos)
