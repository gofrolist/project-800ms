---
title: "Avoid sentinel-string comparisons in runtime sed-substituted bundles"
module: apps/web
date: 2026-05-02
category: docs/solutions/best-practices
problem_type: best_practice
component: tooling
severity: high
root_cause: logic_error
resolution_type: code_fix
applies_when:
  - "Vite (or any bundler) build with placeholder ARG values in a Dockerfile"
  - "Runtime sed / envsubst substitution on JS/CSS assets in a container entrypoint"
  - "Build-time placeholder string also appears as a literal in source comparisons"
  - "Build-once / deploy-anywhere Docker image model with deploy-time config injection"
tags:
  - sed
  - runtime-substitution
  - vite
  - docker
  - placeholder
  - sentinel
  - whitelist-validation
  - feature-flag
---

# Avoid sentinel-string comparisons in runtime sed-substituted bundles

## Context

This project ships a Vite SPA via the *publish-once, deploy-anywhere* pattern: the bundle is built once with placeholder env values baked in at build time, the GHCR image is pulled to any environment, and a one-shot busybox container rewrites the placeholders in-place at deploy time using `sed`. That keeps a single image promotable through staging and production with no rebuild.

```dockerfile
# apps/web/Dockerfile (excerpt)
ARG VITE_API_URL=__API_URL__
ARG VITE_API_KEY=__API_KEY__
ARG VITE_TTS_PRELOAD_ENGINES=__TTS_PRELOAD_ENGINES__
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_API_KEY=$VITE_API_KEY
ENV VITE_TTS_PRELOAD_ENGINES=$VITE_TTS_PRELOAD_ENGINES
RUN bun run build

# ... (later, at container start)
CMD ["/bin/sh", "-c", "set -e; \
  : \"${TTS_PRELOAD_ENGINES:?...}\"; \
  cp -r /dist-template/. /dist/; \
  grep -rl __TTS_PRELOAD_ENGINES__ /dist | xargs -r sed -i \
    \"s|__TTS_PRELOAD_ENGINES__|${TTS_PRELOAD_ENGINES}|g\"; \
"]
```

The pattern is sound. The friction point is when application code wants to distinguish between **"substitution has not run yet"** (dev mode, or pre-sed) and **"substitution ran and produced a real value"**. The intuitive solution — compare the runtime value against the placeholder literal — is a trap, because that comparison is itself a substitution target.

## Guidance

**Prefer detect-by-whitelist over detect-by-sentinel for runtime-substituted code.** Validate against an authoritative domain catalog. Anything that fails validation — unsubstituted placeholder, typo, empty string — collapses to the same safe fallback behavior. There is no literal sentinel string in the source for `sed` to collide with.

### Wrong — sentinel comparison

```ts
// apps/web/src/App.tsx (PR #67 — buggy)
const TTS_PRELOAD_ENGINES_RAW =
  import.meta.env.VITE_TTS_PRELOAD_ENGINES ?? "__TTS_PRELOAD_ENGINES__";

const TTS_PRELOAD_ENGINES_FILTER: ReadonlySet<string> | null = (() => {
  if (
    !TTS_PRELOAD_ENGINES_RAW ||
    TTS_PRELOAD_ENGINES_RAW === "__TTS_PRELOAD_ENGINES__"  // ← literal sentinel
  ) {
    return null;  // "show all engines" fallback for dev
  }
  // ... filter logic ...
})();
```

Vite inlines `import.meta.env.VITE_TTS_PRELOAD_ENGINES` at build time. With `ARG VITE_TTS_PRELOAD_ENGINES=__TTS_PRELOAD_ENGINES__`, the literal `__TTS_PRELOAD_ENGINES__` ends up in **two** places in the compiled bundle:

1. The variable assignment (the intended `sed` target).
2. The string literal inside the `=== "__TTS_PRELOAD_ENGINES__"` comparison (collateral).

When the container's `sed -i "s|__TTS_PRELOAD_ENGINES__|piper,silero,xtts|g"` runs at deploy time, it rewrites both occurrences. The runtime check becomes:

```ts
if (RAW === "piper,silero,xtts") return null;  // always true post-substitution
```

The filter always returns `null`. The feature flag is permanently disabled.

### Right — whitelist validation

```ts
// apps/web/src/App.tsx (PR #68 — fixed)
const TTS_PRELOAD_ENGINES_RAW = import.meta.env.VITE_TTS_PRELOAD_ENGINES;
// undefined in dev | '__TTS_PRELOAD_ENGINES__' pre-sed | 'piper,silero,xtts' post-sed

const VALID_ENGINE_IDS = new Set(TTS_ENGINES.map((e) => e.id));
//    ^? Set<TtsEngine> — narrow union from the authoritative catalog

const isValidEngineId = (s: string): s is TtsEngine =>
  (VALID_ENGINE_IDS as ReadonlySet<string>).has(s);

const TTS_PRELOAD_ENGINES_FILTER = (() => {
  if (!TTS_PRELOAD_ENGINES_RAW) return null;
  const ids = TTS_PRELOAD_ENGINES_RAW.split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(isValidEngineId);          // unrecognised tokens → filtered out
  return ids.length > 0 ? new Set(ids) : null;
})();
```

Three clean cases — none of them depend on a literal sentinel:

| Situation | `RAW` value | Result |
|---|---|---|
| `bun run dev` (no env) | `undefined` | `null` → all engines shown |
| Docker build, `sed` not yet run | `'__TTS_PRELOAD_ENGINES__'` | split → 0 valid ids → `null` → all engines shown (graceful) |
| Docker build, `sed` ran with real value | `'piper,silero,xtts'` | split → 3 valid ids → `Set(3)` → filter active |

No sentinel literal exists in the source. `sed` has exactly one occurrence to substitute.

## Why This Matters

### The bug is silent

The wrong path is the *safe* path. "Show all engines" is the same fallback used in dev mode, so a manual tester who doesn't already know which engines were configured can stare at the UI and see nothing alarming. Only a tester who specifically expects an engine to be hidden will catch it.

### Substitution confirms success, logic is still broken

`curl`-ing the live bundle and `grep`-ing for the placeholder pattern returns zero hits — `sed` did its job. The substituted value appears in the bundle in the right place. A build engineer reading the dist sees nothing wrong. The defect is not in the substitution; it's in the relationship between two source tokens that compile down to the same byte sequence.

### Debugging cost

In this incident the wrong-engine-set symptom led to suspicion of: wrong env var on the host, Docker layer cache serving stale dist, Caddy serving unprocessed assets, browser cache poisoning. All were investigated before bundle source inspection finally revealed the double occurrence. *(session history)* The root cause is unintuitive because it requires reasoning simultaneously about Vite inlining, `sed` global substitution, and string-literal identity in source.

### Why the original `__API_URL__` / `__API_KEY__` pattern didn't hit this

The pre-existing pattern (set up when the SPA was first containerized for GHCR) was structurally safe: those placeholder strings only appeared in the env-var assignment site, never in any comparison expression elsewhere in the source. Each placeholder appeared exactly once in the bundle, so `sed` had nothing to collide with. The implicit assumption — *"each placeholder appears exactly once"* — was never explicit, and it broke as soon as someone (correctly) wanted to detect the unsubstituted state at runtime. *(session history)*

## When to Apply

This guidance applies whenever a source file uses the **same byte sequence** as both the substitution target and a literal operand elsewhere:

- Vite / webpack / esbuild builds with `ARG`-injected placeholder env vars that `sed` rewrites at deploy time
- `envsubst` over HTML/JS/YAML config files where template variables (`${VAR}` or `__VAR__`) double as comparison operands
- Backend equivalents — Go/Python/Ruby config structs where a placeholder default (`"__DB_URL__"`) is compared in a startup health check; the same substitution will corrupt the check
- Any shell, Python, or Go template where the substitution marker doubles as a magic-value sentinel in application logic

The pattern generalizes: **if your substitution tool performs a global string replace** (which `sed -i`, `envsubst`, and most template engines do), **any occurrence of the target string in source — not just the intended one — will be replaced.**

## Related Traps and Brittle Workarounds

### Same bug class, different surface

```ts
// Still broken — sed rewrites the substring inside .includes()
if (RAW.includes("__TTS_PRELOAD_ENGINES__")) return null;

// Still broken — sed finds the full string regardless of JS expression form
if (RAW.startsWith("__")) return null;
// (and produces false positives for any other __VAR__ env var)
```

### Apparent workaround: split the literal at type-time

```ts
// FRAGILE — do not do this
const SENTINEL = "__TTS_PRELOAD" + "_ENGINES__";
if (RAW === SENTINEL) return null;
```

This appears to defeat `sed`'s pattern match because at the source level, the bytes are split. But:

- Minifiers (esbuild, terser, Vite's rollup) constant-fold string concatenations of literals — the merged literal reappears in the bundle and `sed` rewrites it again.
- TypeScript `const` inlining can do the same.
- Behavior is implementation-defined and silently regresses on a compiler / minifier upgrade.
- It's cargo-cult obscurantism: the next engineer who reads it won't understand why the string is split, and will "clean it up."

### The correct principle

Don't have a literal sentinel at all. Validate against the domain catalog. If a value is not a member of the known-good set, treat it as absent — whether it's `undefined`, an empty string, an unsubstituted placeholder, or a typo. This collapses three fragile branches (dev / pre-sed / post-sed) into one straightforward membership test with no string literals that can be contaminated.

## Prevention

### Code convention for runtime-substituted env reads

```ts
// VITE_TTS_PRELOAD_ENGINES is injected at build time as a placeholder
// and rewritten by sed at deploy time. Do NOT compare against the
// placeholder string — the comparison itself would be rewritten.
// Validate against the domain catalog instead.
const TTS_PRELOAD_ENGINES_RAW = import.meta.env.VITE_TTS_PRELOAD_ENGINES;
```

### Smoke-test invariant

After running `sed` (or `envsubst`) in a build or smoke-test script, assert that **zero** occurrences of any placeholder pattern remain in the dist:

```sh
remaining=$(grep -rl '__[A-Z_]*__' /app/dist/assets/*.js | wc -l)
if [ "$remaining" -gt 0 ]; then
  echo "ERROR: unsubstituted placeholders remain in dist" >&2
  grep -rn '__[A-Z_]*__' /app/dist/assets/*.js >&2
  exit 1
fi
```

This catches a different failure mode (env var unset, sed step skipped) but also catches a self-collision indirectly: if a sentinel comparison was rewritten, the bundle will still pass this check, but a *companion* assertion that the substituted value parses to a valid domain element will surface the bug. The current CI smoke test in `.github/workflows/ci.yml` only checks placeholder absence; pairing it with a runtime-behavior assertion (issue #69) would catch this class.

### Test coverage

The original bug shipped because the SPA's filter logic had no unit tests at all (issue #70). A small extracted pure function `buildEngineFilter(raw, validIds)` is straightforwardly testable against:

- `raw === undefined` → `null`
- `raw === "__TTS_PRELOAD_ENGINES__"` (sentinel collision pattern) → `null` (because zero ids match the whitelist)
- `raw === "piper,silero,xtts"` → `Set(3)`
- `raw === "qwn3,slero"` (typos) → `null`
- `raw === "  Piper , SILERO  "` (whitespace + case) → `Set(2)` after normalization

A test for the sentinel-collision case specifically prevents regression even if a future refactor reintroduces a literal-comparison check.

## Related

- **Companion hazard, same `sed` CMD:** issue [#71](https://github.com/gofrolist/project-800ms/issues/71) — `${TTS_PRELOAD_ENGINES}` flows unquoted into the `sed` replacement field, so `&` or `|` in the value (operator typo or supply-chain compromise) breaks substitution or silently corrupts the bundle.
- **Missing test coverage that would have caught this:** issues [#69](https://github.com/gofrolist/project-800ms/issues/69) (CI smoke-test asserts substitution presence, not filter behavior) and [#70](https://github.com/gofrolist/project-800ms/issues/70) (SPA filter has zero unit tests).
- **Source PRs:** [#67](https://github.com/gofrolist/project-800ms/pull/67) introduced the per-engine TTS feature flag with the buggy sentinel; [#68](https://github.com/gofrolist/project-800ms/pull/68) replaced it with whitelist validation; [#77](https://github.com/gofrolist/project-800ms/issues/77) was the review finding that prompted this doc.
- **Implementation files:** `apps/web/src/App.tsx` (filter logic), `apps/web/Dockerfile` (`sed` CMD), `infra/docker-compose.tls.yml` (env passthrough to the web one-shot).
