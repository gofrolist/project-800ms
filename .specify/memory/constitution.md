<!--
SYNC IMPACT REPORT
==================
Version change: (unratified template) → 1.0.0
Rationale: Initial ratification. File previously contained only template placeholders;
this commit replaces them with concrete, project-grounded content, which constitutes a
MAJOR bump from "no constitution" to 1.0.0.

Modified principles: none (initial set)
Added sections:
  - Core Principles: I. Latency Budget Is Law, II. Test-First Discipline,
    III. Grounded & Bounded NPC Behavior, IV. Security & Multi-Tenant Boundaries,
    V. Observable, Reproducible Pipelines
  - Performance & Latency Standards (non-negotiable numeric budgets)
  - Development Workflow & Quality Gates
  - Governance (amendment procedure, versioning policy)

Removed sections: none

Templates & docs requiring updates:
  - ✅ .specify/templates/plan-template.md — Constitution Check section already
    references the constitution file generically; no rewrite needed, but plans must now
    explicitly enumerate gate checks (see §Development Workflow below). No edit needed
    in v1.0.0; revisit if later amendments add/remove principles.
  - ✅ .specify/templates/spec-template.md — Success Criteria section must include a
    latency / scope-grounding metric for any voice-path feature; generic template stays,
    but /speckit-specify authors are now bound by Principle I and III.
  - ✅ .specify/templates/tasks-template.md — Polish phase already lists security
    hardening and docs; no edits needed in v1.0.0. If principles evolve, task categories
    may need to add e.g. "latency regression check" as a standing task type.
  - ✅ CLAUDE.md (project root) — already documents the <800ms target, rate limiting,
    XFF handling, GigaAM hallucination filter, and secrets flow; aligns with principles
    I, III, IV, and V. No edits required.
  - ⚠ README.md — references "<800ms" target but does not mention the NPC/RAG/Arizona RP
    scope; not a constitution concern, but a later docs pass is suggested.

Follow-up TODOs: none — all principles grounded in current repo state and active plans.
-->

# project-800ms Constitution

<!-- Real-time voice assistant: browser ↔ LiveKit ↔ Pipecat agent (VAD → GigaAM STT →
LLM → TTS) with a FastAPI token-minting backend, integrated with the Arizona RP game
(https://arizona-rp.com/) to provide voice-driven Helper/Guide NPCs backed by a curated
Russian Knowledge Base. -->

## Core Principles

### I. Latency Budget Is Law (NON-NEGOTIABLE)

First audio out MUST land within **800ms** of end-of-speech (EOS) under the documented
reference hardware (RTX 5080 / L4 / A10G / L40S, 16 GB+ VRAM). Every component carries
an explicit sub-budget; the sum of sub-budgets MUST NOT exceed the total.

- VAD → STT hand-off, STT emit, LLM first-token, TTS first-audio-out, and any retrieval
  step MUST each have a declared budget recorded in the feature plan's "Performance"
  section before implementation begins.
- RAG / retrieval sits inline on the turn in v1 (no speculative prefetch); its budget is
  **200–500ms** and counts against the 800ms total.
- Barge-in responsiveness MUST NOT regress. `asyncio.to_thread` work is uncancellable,
  so any blocking call in the Pipecat pipeline sets the barge-in floor — authors MUST
  state the new floor in the plan when adding such a call.
- Any PR that changes the hot audio path (VAD, STT, LLM call, TTS, FrameProcessor
  insertion) MUST include a before/after latency measurement in the PR description
  using the existing `tests/` harness or a documented manual run. A measurement absence
  is grounds to block merge.

**Rationale:** The product is defined by perceived response latency. Every drift is
visible to the user. Treating latency as a per-change contract — not a best-effort — is
the only way to keep the target defensible as surface area grows.

### II. Test-First Discipline (NON-NEGOTIABLE)

TDD is mandatory for every feature, bug fix, and refactor.

- Tests MUST be written before implementation, MUST fail first (RED), and only then
  MUST be made to pass (GREEN), then refactored.
- **Minimum coverage: 80%** for new or modified modules. Coverage is a floor, not a
  ceiling; security- and latency-sensitive paths warrant more.
- Any write path (database INSERT/UPDATE/DELETE, LiveKit webhook handler, ingestion
  upsert) MUST have a happy-path integration test that asserts **every writable
  column** — not only the status column. This rule is derived from the
  `delete-sessions-missing-audio-seconds-2026-04-23` regression.
- New HTTP or gRPC endpoints MUST ship with contract tests before client wiring.
- Tests MUST use real databases for integration scope (Postgres via Docker), not mocks,
  except when explicitly justified in the plan's complexity-tracking table.
- Fixes to flaky or failing tests MUST diagnose root cause first; silencing or
  retry-wrapping a test requires an explicit note in the PR and a follow-up issue.

**Rationale:** Mocked tests that passed while prod migrations broke are a documented
historical failure mode here. The cost of writing the test first is always lower than
the cost of a post-release regression on a real-time voice product.

### III. Grounded & Bounded NPC Behavior

Helper/Guide NPCs (and any future NPC archetype) MUST answer only from a curated,
tenant-scoped knowledge base, and MUST stay inside their declared persona and scope.

- NPC prompts MUST constrain the LLM to retrieved context for factual answers (game
  commands, prices, quest locations, level requirements). The LLM MUST NOT be allowed
  to answer from parametric memory on these classes of questions.
- An in-scope classifier MUST run before retrieval. Out-of-scope questions MUST route
  to a refusal or clarification path; they MUST NOT fall through to an ungrounded LLM
  answer.
- Persona lock is mandatory: the system prompt MUST instruct the model to refuse
  roleplay hijacks, prompt-injection framing, and requests to "ignore previous
  instructions". This mitigates OWASP LLM #1.
- **Timing-channel parity**: in-scope and out-of-scope branches MUST perform equal work
  (or synthetic equivalent) so that response latency does not leak KB membership. Any
  measured asymmetry >50ms MUST be padded or redesigned.
- Scope boundaries for v1: Russian only; Helper/Guide NPCs only; conversations only —
  no map geotagging, no writes back to the game client, no player-state queries.
  Expanding scope requires a constitution amendment or an explicit Complexity Tracking
  entry in the feature plan.

**Rationale:** The product's legitimacy with Arizona RP players depends on answers
being correct. Hallucinated commands, prices, or locations destroy trust faster than
any latency regression. Grounding is the product, not a feature.

### IV. Security & Multi-Tenant Boundaries

Every public surface, every tenant boundary, and every secret is treated as a
production attack surface.

- **No hardcoded secrets.** API keys, LiveKit JWTs, DB passwords, and model tokens MUST
  come from environment variables or a secret manager (Terraform → SSM → `user_data.sh`
  flow in prod). Validation at startup is required (`require_env()` in agent,
  Pydantic `Field(min_length=...)` in API settings).
- **Tenant isolation is a data invariant, not a convention.** Any new table that stores
  tenant-owned data MUST include `tenant_id NOT NULL` from the first migration. All
  reads MUST be tenant-scoped; all writes MUST assert tenant ownership.
- **Rate limiting** MUST be enforced on every authenticated `/v1/*` route (tenant
  bucket), every unauthenticated admin surface (IP bucket), and every webhook (IP
  bucket). Caches for each keyspace MUST be isolated — sharing the cache leaks
  cross-tenant state.
- **XFF trust boundary:** `X-Forwarded-For` MUST be honored only when the TCP peer
  falls inside `settings.trusted_proxy_cidrs`. Anywhere outside this boundary, the
  TCP peer IP is authoritative. This rule is derived from the `xff-spoof` incident.
- **Short-lived credentials.** LiveKit caller tokens MUST be ≤15 min TTL; agent-minted
  room tokens MUST be ≤30 min TTL. Longer TTLs require a written justification in the
  PR.
- **CORS** MUST be wildcard only in dev. Production deploys MUST derive allowed origins
  from configured domain list (`CORS_ALLOWED_ORIGINS`).
- Any change touching auth, token minting, rate limiting, user input handling, SQL
  query construction, or external API calls MUST be reviewed by the `security-reviewer`
  agent before merge.

**Rationale:** Multi-tenant voice infrastructure is a high-value target. The project
has already absorbed two security regressions (XFF spoof, rate-limit cache eviction);
the principles above codify the lessons so they are not re-learned.

### V. Observable, Reproducible Pipelines

The system MUST be debuggable from logs alone when a user reports a bad turn, and
rebuildable from source when an image rots.

- **Structured logging** via loguru with lazy `{name}` placeholders (not f-strings).
  Log lines in the hot path MUST include `room_id`, `session_id`, and `turn_id` where
  available.
- **Error redaction:** Pipecat `FrameProcessor` failures MUST be surfaced as
  `ErrorFrame` rather than raw exceptions; stack traces MUST NOT leak downstream to the
  user-facing transcript channel.
- **Heavy models** (STT, embedder, LLM client) MUST be preloaded via singleton at
  service boot — not lazily on first turn. See `services/agent/models.py::load_gigaam`
  for the reference pattern.
- **Docker image layout:** services with heavy deps (CUDA, PyTorch, sentence-
  transformers) MUST split into a `Dockerfile.base` (deps, rebuilt only on lockfile
  change) and a thin app `Dockerfile` (source COPY only). Base images MUST include a
  `python -c "import <critical_deps>"` smoke layer — lockfile clean does not imply
  runtime clean.
- **Model weights are never baked** into images. HF caches MUST be persisted via named
  volumes (`hf_cache_agent`, `hf_cache_retriever`, …) scoped per service UID.
- **Compound learnings** — every non-trivial production incident, build failure, or
  design decision with a blast radius MUST produce a doc under `docs/solutions/` with
  `title`, `module`, `tags`, and `problem_type` frontmatter. Future features consult
  these before implementing in the same area.

**Rationale:** At <800ms turn budgets, there is no time to re-derive context when a
turn goes wrong. Observability is how the team's knowledge compounds; without it, every
regression is a new investigation.

## Performance & Latency Standards

Reference hardware: RTX 5080 / L4 / A10G / L40S, 16 GB+ VRAM, NVIDIA Container Toolkit,
Docker Compose v2.

Non-negotiable numeric budgets:

| Stage                                   | Target        | Notes                            |
|-----------------------------------------|---------------|----------------------------------|
| First audio out after EOS               | **≤ 800 ms**  | End-to-end SLO                   |
| RAG retrieval (when on-turn)            | 200 – 500 ms  | Counts against the 800 ms total  |
| Barge-in floor                          | ≤ retrieval RTT | `asyncio.to_thread` uncancellable |
| GigaAM hallucination filter floor       | 300 ms / 2 tok | Below this → segment dropped     |
| User transcript debounce                | 1.0 s         | `_USER_TRANSCRIPT_DEBOUNCE_SECS` |
| LiveKit caller token TTL                | ≤ 15 min      |                                  |
| LiveKit agent-minted room token TTL     | ≤ 30 min      |                                  |
| Min test coverage on new/changed code   | 80 %          | Principle II                     |

Feature plans introducing a new stage on the audio path MUST add a row to this table
via a MINOR amendment (see Governance).

## Development Workflow & Quality Gates

Feature pipeline (mirrors `/speckit-*` commands and project CLAUDE.md guidance):

1. **Research & Reuse first.** Before new implementation, search GitHub code, primary
   library docs (Context7), and relevant package registries. Adopt or port proven
   approaches where they meet the requirement.
2. **Ideation → Plan.** Non-trivial features enter via `docs/ideation/` then graduate
   to `docs/plans/` with a requirements trace (`R1..Rn`), scope boundaries, and
   "Deferred to Separate Tasks" list. The plan's Constitution Check section MUST
   enumerate which principles are in tension and cite the Complexity Tracking entry if
   any principle is to be violated.
3. **Test-first implementation.** Per Principle II.
4. **Code review.** The `code-reviewer` agent runs after writing code; the
   `security-reviewer` agent is mandatory for changes touching Principle IV surfaces.
   CRITICAL and HIGH findings MUST be resolved before merge.
5. **Pre-commit gates.** `ruff check`, `ruff format`, `gitleaks`, file-hygiene hooks,
   and `actionlint` MUST pass locally before commit. CI re-runs them — CI failure
   blocks merge.
6. **Commit convention.** `type(scope): description` — types: `feat`, `fix`, `refactor`,
   `docs`, `test`, `chore`, `perf`, `ci`. Scopes: `api`, `agent`, `web`, `infra`,
   `retriever`, `ci`.
7. **Definition of Done for a feature:** all acceptance scenarios pass, ≥80% coverage,
   latency budget measured and recorded, security review signed off (if in scope),
   `docs/solutions/` updated if the feature resolved or exposed a class of issue.

Complexity gate: any plan that violates a principle MUST document the violation in its
Complexity Tracking table with the rationale and the simpler alternative rejected. A
plan with an unjustified violation is not approvable.

## Governance

This constitution supersedes ad-hoc practice. When it conflicts with CLAUDE.md, the
constitution wins; CLAUDE.md SHOULD be updated to align, or the constitution SHOULD be
amended — whichever reflects the team's current intent.

**Amendment procedure:**

1. Propose the amendment in a PR modifying `.specify/memory/constitution.md`.
2. Update the Sync Impact Report (HTML comment at top of file) with version delta,
   modified/added/removed principles, and template-update status.
3. Run `/speckit-constitution` (this command) so dependent templates
   (`plan-template.md`, `spec-template.md`, `tasks-template.md`) are re-validated.
4. Amendment lands only when at least one reviewer outside the author has approved.

**Versioning policy** (semantic):

- **MAJOR**: Backward-incompatible governance changes, principle removals, or material
  redefinitions that existing plans/specs would fail under.
- **MINOR**: New principle or section added; materially expanded guidance; new
  performance budget row.
- **PATCH**: Clarifications, wording fixes, typo corrections, non-semantic refinements.

**Compliance review:**

- Every PR description MUST state "Constitution compliance: ✅" or enumerate principles
  in tension with a link to the plan's Complexity Tracking entry.
- Quarterly (or on major scope expansion — e.g. second NPC archetype, multi-language,
  game-client callbacks), the constitution is reviewed against current reality; drift
  is either codified (amendment) or corrected (code change).

Runtime development guidance lives in `CLAUDE.md` (project root), `.specify/templates/`,
and language-specific rules under `~/.claude/rules/`. This constitution governs what
those documents MAY prescribe, not the day-to-day mechanics.

**Version**: 1.0.0 | **Ratified**: 2026-04-23 | **Last Amended**: 2026-04-23
