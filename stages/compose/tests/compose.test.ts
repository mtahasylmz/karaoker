// Regression tests for stages/compose. `node:test` + `node:assert/strict`,
// executed via `tsx --test` (see package.json). These lock down the
// invariants that the TS typechecker cannot express:
//
//   - buildAss and buildLines stay in lock-step (one Dialogue: per line).
//   - buildManifest produces a contract-valid PlaybackManifest.
//   - Word-level filters (empty text, zero duration) match between the two.
//
// No snapshot tests — byte-for-byte output would trip on any cosmetic
// tweak to DEFAULT_STYLE. Assert structural invariants instead.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, test } from "node:test";
import assert from "node:assert/strict";

import {
  ComposeRequest,
  PlaybackManifest,
  type Word,
} from "@annemusic/contracts";

import { buildAss, buildLines } from "../src/ass.js";
import { buildManifest } from "../src/manifest.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../bench/fixtures/fixture.json");
const fixture = ComposeRequest.parse(JSON.parse(readFileSync(fixturePath, "utf8")));

function dialogueLines(ass: string): string[] {
  return ass.split("\n").filter((l) => l.startsWith("Dialogue:"));
}

describe("buildLines / buildAss lock-step", () => {
  test("fixture grouping produces exactly 3 lines", () => {
    // Fixture has two gaps > 1.5 s (4.0→6.0 and 7.3→10.0) so grouping must
    // split into three lines. Asserting the specific count catches any
    // change to break_gap or the grouping logic.
    const lines = buildLines(fixture.words, fixture.style ?? {});
    assert.equal(lines.length, 3);
  });

  test("every line has start <= end and at least one word", () => {
    const lines = buildLines(fixture.words, fixture.style ?? {});
    for (const line of lines) {
      assert.ok(line.words.length >= 1, "line is non-empty");
      assert.ok(line.start <= line.end, `${line.start} <= ${line.end}`);
      assert.equal(line.start, line.words[0]!.start);
      assert.equal(line.end, line.words[line.words.length - 1]!.end);
    }
  });

  test("Dialogue: event count matches lines.length", () => {
    // Lock-step: both helpers apply the same filter and groupLines with the
    // same params, so counts must match. They can drift only if every token
    // on a line collapses to empty after sanitize().trim() — the fixture
    // avoids that.
    const ass = buildAss(fixture.words, fixture.style ?? {});
    const lines = buildLines(fixture.words, fixture.style ?? {});
    assert.equal(dialogueLines(ass).length, lines.length);
  });
});

describe("buildManifest contract validity", () => {
  test("manifest assembled from fixture validates against PlaybackManifest", () => {
    const manifest = buildManifest(fixture, "file:///tmp/lyrics.ass", 1_700_000_000_000);
    const parsed = PlaybackManifest.safeParse(manifest);
    assert.ok(
      parsed.success,
      parsed.success ? "" : JSON.stringify(parsed.error.issues, null, 2),
    );
  });

  test("manifest.lines and manifest.vocal_activity are populated", () => {
    const manifest = buildManifest(fixture, "file:///tmp/lyrics.ass", 1_700_000_000_000);
    assert.equal(manifest.lines.length, 3);
    assert.equal(manifest.vocal_activity.length, 3);
    assert.equal(manifest.created_at, 1_700_000_000_000);
  });
});

describe("word coverage", () => {
  test("every input word appears in exactly one line, in order", () => {
    const lines = buildLines(fixture.words, fixture.style ?? {});
    const flat = lines.flatMap((l) => l.words);
    const cleaned = fixture.words.filter((w) => w.text && w.end > w.start);
    assert.equal(flat.length, cleaned.length);
    for (let i = 0; i < cleaned.length; i++) {
      assert.deepEqual(flat[i], cleaned[i]);
    }
  });
});

describe("ASS structure", () => {
  test("output starts with [Script Info] and contains required sections", () => {
    const ass = buildAss(fixture.words, fixture.style ?? {});
    assert.ok(ass.startsWith("[Script Info]\n"), "starts with Script Info header");
    assert.ok(ass.includes("\n[V4+ Styles]\n"), "contains V4+ Styles section");
    assert.ok(ass.includes("\n[Events]\n"), "contains Events section");
  });

  test("exactly one Dialogue: per line", () => {
    const ass = buildAss(fixture.words, fixture.style ?? {});
    const lines = buildLines(fixture.words, fixture.style ?? {});
    assert.equal(dialogueLines(ass).length, lines.length);
  });
});

describe("edge cases", () => {
  test("empty words: empty lines, still contract-valid manifest", () => {
    const emptyReq = { ...fixture, words: [] as Word[] };
    const ass = buildAss(emptyReq.words, {});
    const lines = buildLines(emptyReq.words, {});
    assert.equal(lines.length, 0);
    assert.equal(dialogueLines(ass).length, 0);

    const manifest = buildManifest(emptyReq, "file:///tmp/lyrics.ass");
    const parsed = PlaybackManifest.safeParse(manifest);
    assert.ok(
      parsed.success,
      parsed.success ? "" : JSON.stringify(parsed.error.issues, null, 2),
    );
    assert.equal(manifest.duration, undefined);
    assert.equal(manifest.lines.length, 0);
  });

  test("single word produces a single line with identical timings", () => {
    const oneWord: Word[] = [{ text: "hi", start: 1.0, end: 1.3 }];
    const lines = buildLines(oneWord, {});
    assert.equal(lines.length, 1);
    assert.equal(lines[0]!.start, 1.0);
    assert.equal(lines[0]!.end, 1.3);
    const ass = buildAss(oneWord, {});
    assert.equal(dialogueLines(ass).length, 1);
  });

  test("zero-duration and empty-text words are filtered", () => {
    const words: Word[] = [
      { text: "a", start: 1.0, end: 1.0 }, // zero duration — dropped
      { text: "", start: 1.1, end: 1.4 },  // empty text — dropped
      { text: "b", start: 1.2, end: 1.5 }, // keeper
    ];
    const lines = buildLines(words, {});
    assert.equal(lines.length, 1);
    assert.equal(lines[0]!.words.length, 1);
    assert.equal(lines[0]!.words[0]!.text, "b");
    const ass = buildAss(words, {});
    assert.equal(dialogueLines(ass).length, 1);
  });
});

describe("determinism", () => {
  test("same input produces identical .ass bytes and lines", () => {
    const a1 = buildAss(fixture.words, fixture.style ?? {});
    const a2 = buildAss(fixture.words, fixture.style ?? {});
    assert.equal(a1, a2);
    const l1 = buildLines(fixture.words, fixture.style ?? {});
    const l2 = buildLines(fixture.words, fixture.style ?? {});
    assert.deepEqual(l1, l2);
  });
});
