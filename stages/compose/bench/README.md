# compose bench

A realistic `ComposeRequest` (Rick Astley first verse — deliberately crosses
the 1.5 s `break_gap` twice so line grouping splits three ways) plus a small
runner that validates the request against the contract, builds the `.ass` +
manifest, and asserts the line-count lock-step invariant between `buildAss`
and `buildLines`.

```bash
pnpm --filter @annemusic/stage-compose bench
```

Output is a JSON summary on stdout; exit code is non-zero if the fixture
stops parsing as a `ComposeRequest`, if the assembled manifest stops
validating as a `PlaybackManifest`, or if the `.ass` `Dialogue:` count
drifts from the `lines.length`.

The fixture is reused by `tests/compose.test.ts` for regression assertions.
