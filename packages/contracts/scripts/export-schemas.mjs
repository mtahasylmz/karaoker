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

const index = {};
let wrote = 0;
for (const [name, value] of Object.entries(schemas)) {
  if (!value || typeof value !== "object") continue;
  // Filter to actual Zod schemas (have _def + .parse).
  if (!("_def" in value) || typeof value.parse !== "function") continue;
  const filename = `${toSnake(name)}.json`;
  const json = zodToJsonSchema(value, { name, $refStrategy: "none" });
  writeFileSync(resolve(outDir, filename), JSON.stringify(json, null, 2) + "\n");
  index[name] = `./${filename}`;
  wrote++;
}
writeFileSync(
  resolve(outDir, "index.json"),
  JSON.stringify(index, null, 2) + "\n",
);
console.log(`contracts: wrote ${wrote} schemas → ${outDir}`);
