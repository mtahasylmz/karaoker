import { serve as nodeServe } from "@hono/node-server";
import { Hono } from "hono";
import { annemusicWorkflow } from "./workflow.js";

const app = new Hono();

app.get("/ping", (c) => c.json({ ok: true, service: "orchestrator" }));
app.post("/workflow", annemusicWorkflow);

const port = Number(process.env.PORT ?? 8090);
nodeServe({ fetch: app.fetch, port });
console.log(JSON.stringify({
  ts: Date.now(),
  stage: "orchestrator",
  level: "info",
  msg: "listening",
  data: { port },
}));
