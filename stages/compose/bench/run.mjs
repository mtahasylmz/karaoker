// Smoke run: load the bench ComposeRequest, validate it against the contract,
// build the .ass + lines + manifest, and print a small summary. Asserts the
// one invariant that matters here — the line count stays in lock-step with
// the number of Dialogue: events — so a broken port fails loudly on the
// command line without needing the full test suite.
//
// Run: pnpm --filter @annemusic/stage-compose bench

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { ComposeRequest, PlaybackManifest } from "@annemusic/contracts";

import { buildAss, buildLines } from "../src/ass.ts";
import { buildManifest } from "../src/manifest.ts";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "fixtures/fixture.json");
const raw = JSON.parse(readFileSync(fixturePath, "utf8"));

const req = ComposeRequest.parse(raw);

const ass = buildAss(req.words, req.style ?? {});
const lines = buildLines(req.words, req.style ?? {});
const manifest = buildManifest(req, "file:///tmp/compose-bench/lyrics.ass");

const dialogueCount = ass.split("\n").filter((l) => l.startsWith("Dialogue:")).length;
if (dialogueCount !== lines.length) {
  console.error(
    `lock-step invariant broken: ${dialogueCount} Dialogue: events vs ${lines.length} lines`,
  );
  process.exit(1);
}

const parsedManifest = PlaybackManifest.safeParse(manifest);
if (!parsedManifest.success) {
  console.error("manifest failed contract:", JSON.stringify(parsedManifest.error.issues, null, 2));
  process.exit(1);
}

console.log(JSON.stringify({
  job_id: req.job_id,
  words: req.words.length,
  lines: lines.length,
  dialogue_events: dialogueCount,
  vocal_activity: manifest.vocal_activity.length,
  ass_bytes: Buffer.byteLength(ass),
  duration: manifest.duration,
}, null, 2));
