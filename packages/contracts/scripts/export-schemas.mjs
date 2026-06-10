// Exports every Zod schema from ../dist/index.js as a JSON Schema file
// under ../json-schema/. Consumed by Python stages via jsonschema.
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { zodToJsonSchema } from "zod-to-json-schema";
import * as schemas from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, "..", "json-schema");

// Clean rebuild so stale schemas don't linger.
rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const toSnake = (name) =>
  name
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2")
    .toLowerCase();

// Tolerant reader: zod-to-json-schema emits `additionalProperties: false`
// for every plain z.object, but the Zod consumers (compose, orchestrator)
// STRIP unknown keys rather than reject them — so the same wire payload
// was valid in TS and a 400 in Python, and adding even an optional field
// forced a stages-before-orchestrator lockstep deploy. Producers may add
// fields freely; consumers ignore what they don't know.
const stripAdditionalProperties = (node) => {
  if (Array.isArray(node)) {
    node.forEach(stripAdditionalProperties);
    return;
  }
  if (node && typeof node === "object") {
    if (node.additionalProperties === false) delete node.additionalProperties;
    Object.values(node).forEach(stripAdditionalProperties);
  }
};

const index = {};
let wrote = 0;
for (const [name, value] of Object.entries(schemas)) {
  if (!value || typeof value !== "object") continue;
  // Filter to actual Zod schemas (have _def + .parse).
  if (!("_def" in value) || typeof value.parse !== "function") continue;
  const filename = `${toSnake(name)}.json`;
  const json = zodToJsonSchema(value, { name, $refStrategy: "none" });
  stripAdditionalProperties(json);
  writeFileSync(resolve(outDir, filename), JSON.stringify(json, null, 2) + "\n");
  index[name] = `./${filename}`;
  wrote++;
}
writeFileSync(
  resolve(outDir, "index.json"),
  JSON.stringify(index, null, 2) + "\n",
);

// Routing-table snapshot for the shared-py parity test: flows.py mirrors
// flows.ts by hand, and this artifact is how a drift becomes a test failure
// instead of a silently different backend in one language's pipeline.
const flows = {
  qwen_align_langs: [...schemas.QWEN_ALIGN_LANGS].sort(),
  qwen_transcribe_langs: [...schemas.QWEN_TRANSCRIBE_LANGS].sort(),
  default_flow: schemas.DEFAULT_FLOW,
};
writeFileSync(resolve(outDir, "flows.json"), JSON.stringify(flows, null, 2) + "\n");

console.log(`contracts: wrote ${wrote} schemas + flows.json → ${outDir}`);
