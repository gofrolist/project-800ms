---
date: 2026-04-23
topic: rag-memory-system
focus: RAG / persistent memory so the voice agent ("NPC") remembers users and conversation history across sessions
mode: repo-grounded
---

# Ideation: RAG / Persistent Memory for the Voice Agent

## Grounding Context

### Codebase Context

- FastAPI (3.14) + Pipecat agent (3.12, GPU) + React SPA + PostgreSQL 18 (async SQLAlchemy 2.0).
- Existing tables: `tenants`, `api_keys`, `sessions`, `session_transcripts`.
- `sessions` already carries `user_id`, `tenant_id`, `npc_id`, `persona`, and a `context` JSONB column **currently unused** — designed for future expansion.
- `session_transcripts` is append-only (`role`, `text`, timestamps) — a natural outbox.
- No Redis, no vector DB, no embedding layer exists today. Single L4 GPU is saturated by GigaAM STT + vLLM LLM + Piper TTS.
- Agent is stateless per-room. `LLMContext` is rebuilt per-utterance at `services/agent/pipeline.py`.
- **Two session-end paths** (explicit `DELETE /v1/sessions` + LiveKit `room_finished` webhook). Documented regression (`delete-sessions-missing-audio-seconds-2026-04-23.md`) proves every persist-on-end hook must land on both paths.
- Latency target: **<800ms first-audio-out** — retrieval must fit inside the turn budget.
- Rate-limit buckets use bounded TTLCaches; the `xff-spoof` learning warns against co-locating new per-tenant state in those same caches.

### Past Learnings

- **xff-spoof shared-cache eviction** — keep any new RAG-side caches (embedding client pool, per-tenant namespace map, vector-store session) isolated from the rate-limit TTLCaches.
- **DELETE + webhook dual lifecycle** — memory consolidation MUST fire on both; assert the side-effect, not the HTTP status.
- No prior memory/RAG art in the repo — this is greenfield; plan to document the landed design under `docs/solutions/best-practices/`.

### External Context

- **Salesforce VoiceAgentRAG** (March 2026) — dual-agent semantic cache (Slow Thinker pre-fetches 3–5 candidates during previous TTS playback; Fast Talker reads in-memory FAISS). 316× retrieval speedup, 75% hit rate. Directly addresses the <800ms constraint.
- **Mem0** — three-stage extract/consolidate/retrieve. **Pipecat has a first-party `Mem0MemoryService` frame processor** that intercepts `LLMContextFrame` and injects memory as a system-prompt prefix — lowest-friction pipeline hook.
- **Zep/Graphiti** — bi-temporal knowledge graph (event-time + ingestion-time), LiveKit integration exists, P95 <250ms claimed. Informs the provenance/validity schema design.
- **Letta / MemGPT** — OS-paged three-tier memory (core / recall / archival). Informs the two-tier loader.
- **SQLite+FTS5** benchmarks <1ms at 5k entries (beats Pinecone P95 25–50ms at small scale) — considered and rejected below.
- **Failure modes:** memory poisoning (MINJA >95% success against naive extraction), hallucination amplification from summaries, catastrophic forgetting under compression, context drift under re-embedding, cost explosion under naive per-turn extraction, stale cache coherence.
- **Compliance:** GDPR treats voice transcripts as personal data and voiceprints as biometric. AES-256 at rest + TLS 1.3 in transit is the minimum; right-to-erasure via provenance tagging is the canonical pattern.

## Ranked Ideas

### 1. Provenance-tagged memory atoms as the universal schema

**Description:** A single `memory_atoms` table becomes the one read path for every downstream feature. Shape: `{id, tenant_id, npc_id, user_id, subject, predicate, object, confidence, source_session_id, source_transcript_span, extractor_version, valid_from, valid_to, redacted_at, visibility, zone, status}`. Every atom points back to the exact `session_transcripts` span that produced it. Supersede-not-delete with bi-temporal validity (Zep-style, no KG complexity upfront). Extensions needed by later ideas are columns on this one table.

**Rationale:** One investment pays for recall, right-to-erasure (a `WHERE` clause), audit replay, "why did you say that?" UI (JOIN to `session_transcripts`), fine-tuning dataset export (SELECT), hallucination forensics, cross-NPC gossip (loosen `npc_id` filter). Tenant isolation rides the existing `tenant_id` scoping already used everywhere. Without this schema, right-to-erasure / audit become retrofits. This is the lowest-leverage-per-dollar investment in the set.

**Downsides:** Requires extraction logic that actually produces structured subject/predicate/object triples — naive "summary paragraph" approaches can't populate this well. Needs Alembic migration and careful index strategy (partial indexes on `valid_to IS NULL` and `(tenant_id, npc_id, user_id)` composite). Extractor-version bookkeeping adds complexity but unlocks replay.

**Confidence:** 85%
**Complexity:** Medium
**Status:** Unexplored

---

### 2. Two-tier memory with an <800ms-safe loader

**Description:** Split memory retrieval into a HOT tier and a COLD tier. Hot tier: ≤20 pinned "anchor" atoms per `(user_id, npc_id)` pair, pre-warmed at `/dispatch` and baked into the LLM system prompt — O(1) at turn time, zero retrieval cost during the <800ms window. Cold tier: lazy semantic retrieval (pgvector on PG18) fetched via an LLM function-call tool only when the user explicitly cues recall ("do you remember when…"). Loader has a strict ~40ms deadline and degrades gracefully (anchors-only → + top-1 pool hit → full retrieval) based on available headroom. Optional second-stage: a VoiceAgentRAG-style Slow Thinker that speculatively prefetches likely-next-turn candidates during the previous TTS playback (hides retrieval behind audio the user is already hearing).

**Rationale:** The product's entire differentiation is latency. A RAG feature that regresses p95 TTFB by 300ms is a net loss even if the answers are smarter. Anchors are a constant-time lookup keyed by `(tenant_id, npc_id, user_id)`. The anchor primitive also powers catastrophic-forgetting protection (pinned facts bypass summarization compression), user-facing "what does the NPC know about me" editable dashboard rows, and eventual fine-tuning seed data. The speculative prefetch pattern is a 2026 published result (Salesforce) with 316× speedup on cache hits.

**Downsides:** Two caches = two consistency problems. Anchor selection heuristic (which atoms promote to anchor?) is tuning-sensitive. Speculative prefetch adds Pipecat frame-handler complexity and needs a cap on wasted embeddings when the prediction misses. pgvector in the cold tier adds PG extension management.

**Confidence:** 75%
**Complexity:** Medium (High if speculative prefetch is added in V1)
**Status:** Unexplored

---

### 3. Async extraction pipeline with versioned replay

**Description:** The live agent writes only the immutable `session_transcripts` rows it already writes — no memory logic on the hot path. A separate `services/memory-worker` Python service consumes new transcripts via PG18's `SKIP LOCKED` queue (or `pg_notify`) and produces memory atoms off the GPU. Worker prompts and extractor models are versioned (`memory_atoms.extractor_version`); replaying the entire session history against a new extractor version is a one-line CLI command. Session-end triggers (DELETE + webhook, unified behind a single lifecycle helper) enqueue a final consolidation job.

**Rationale:** Two wins in one design. First, it protects the 800ms hot path absolutely — no inline LLM extraction, no embedding model competing with GigaAM+vLLM+TTS for the single L4 GPU. Second, versioned/replayable extraction is the foundation for every future iteration: when the extraction prompt improves, re-run against all transcripts; A/B two extractors; ablate; export deltas for fine-tuning data. It also gives you a clean retry/DLQ story for extraction failures and turns "did we capture this fact?" from a bug into a data question. Uses PG18 native `SKIP LOCKED` — no Redis, matches the stated "no Redis" architecture. Mirrors the `hf_cache_agent` named-volume pattern so embedding model downloads are cached.

**Downsides:** New deployable unit in `infra/docker-compose.yml`. Worker lifecycle management (graceful shutdown, backlog monitoring). Replay across production history requires cost guardrails — a careless re-extraction could cost $1000s in LLM calls. The session-end unification needs to survive the documented DELETE-vs-webhook race from the `delete-sessions-missing-audio-seconds` incident.

**Confidence:** 85%
**Complexity:** Medium
**Status:** Unexplored

---

### 4. Therapist-note summaries as the MVP memory shape

**Description:** At session end, an LLM pass produces a structured clinical-note JSON: `{chief_complaint, worked_on, homework, to_revisit_next_time}`. The note is written to `sessions.context` JSONB (already designed for exactly this). On subsequent session start, the NPC's system prompt receives the last N notes for `(user_id, npc_id)` — not raw transcripts, not vector hits, just the clinician-style summaries. Raw `session_transcripts` rows are archived per retention policy.

**Rationale:** This is the shortest path to shipped value. One LLM call per session end, no retrieval engine, no vector DB, no speculative prefetch. The therapist-note analogy is proven (Woebot, Wysa) and maps exactly to the NPC-remembers-you goal: "last time we talked about your move to Berlin — want to pick that up?" Structured beats unstructured: 5 lines of `to_revisit` tokens beat 50 turns of transcript stuffing ~50× on cost. Also slashes GDPR exposure because derived summaries can be crafted to exclude identifying specifics. Works as a hard-coded extractor before the full M5 atoms schema lands.

**Downsides:** Summaries lose information — structural recall ("you told me your dog is named Noodle") is imperfect. Hallucination amplification risk is real: the NPC re-reads its own summary as fact. This is an MVP that eventually gets *replaced or subsumed* by the M5 atoms schema rather than a long-term destination — the note becomes one atom type, not the primary store.

**Confidence:** 90%
**Complexity:** Low
**Status:** Unexplored

---

### 5. Per-NPC memory ownership with per-atom visibility flags

**Description:** Memory is owned by `(npc_id, user_id)` pairs, not by users globally. Every memory atom carries a `visibility` enum: `npc_private` (default), `tenant_shared` (all of this tenant's NPCs can read), `cross_tenant_public` (community worldbook). A stern tutor NPC's memory about Sasha is logically disjoint from a flirty bartender NPC's memory about Sasha — cross-NPC leaks are impossible by construction. Retrieval joins on visibility; users can flip per-fact visibility from the memory card (M11). Prior art: Shadow of Mordor Nemesis system, Inworld AI NPC memory.

**Rationale:** `sessions.npc_id` + `persona` are already first-class — the product has bet on multi-NPC from day zero. Per-NPC ownership matches how humans remember each other (your dentist knows different things than your bartender). Per-atom visibility turns a future compliance headache (GDPR purpose limitation) into a product feature: users share name across all NPCs but keep sensitive topics scoped to one. Shared worldbook emerges for free once the enum exists — community lore, running jokes, FAQ-grade knowledge, no extra infrastructure. Also differentiating: "multi-NPC with true relationship continuity" is a product story competitors without this schema can't tell.

**Downsides:** Retrieval logic gets more branching. Visibility migrations (moving atoms between scopes) need audit trails. User confusion risk: "why doesn't bartender NPC know what I told tutor NPC?" needs UX treatment. Worldbook promotion requires k-anonymity or similar to avoid leaking private atoms.

**Confidence:** 80%
**Complexity:** Medium
**Status:** Unexplored

---

### 6. User-editable memory card (with operator inspector)

**Description:** A React view in `apps/web`: every atom visible to its owning user, with pin / edit / delete / visibility-flip controls. LLM-extracted atoms carry `source_type='llm_extraction'`; user edits write new atoms with `source_type='user_edit'` that supersede originals via `valid_to`. The same endpoint (scoped to tenant admin, rate-limited via the existing `enforce_admin_ip_rate_limit`) serves operator memory-inspection debugging. Retrieval traces are logged alongside — every turn records which atoms were injected into the LLM context.

**Rationale:** One UI, many jobs — the single highest-leverage frontend surface in this ideation. It is simultaneously: (a) the trust primitive that unlocks willingness to share more, (b) the GDPR right-to-rectification workflow for free, (c) the ground-truth labeling UI for fine-tuning data later, (d) the debugging UI for hallucinations ("the NPC thinks I'm a dentist but I said scientist"), (e) the operator support tool for "bot told me the wrong thing" tickets, and (f) a defensible onboarding surface ("tell me about yourself"). Reuses the existing React SPA and admin rate-limit pattern. Feeds directly into the human-approved extraction idea (CC3).

**Downsides:** Frontend work is non-trivial (CRUD + state + auth + empty-states). Operator-scoped view needs tenant-isolation proofs in tests to avoid the `xff-spoof`-style cross-tenant leak class. "Users rarely edit memory" is a real concern — the feature's value depends on the usage rate of the approval workflow, not the view alone.

**Confidence:** 85%
**Complexity:** Medium
**Status:** Unexplored

---

### 7. Human-approved memory: LLM proposes, user disposes

**Description:** Every LLM-extracted atom lands as `status='proposed'`. Proposed atoms never reach retrieval; only user-approved (`status='active'`) atoms are injected into the system prompt. The memory card (M6) surfaces proposals as a review queue with one-click approve / reject / edit. Auto-approval rules for narrow high-confidence atom types (pronoun capture, explicitly stated name) keep UX friction low; everything richer requires an explicit click. Rejected proposals feed a "stop extracting this kind of thing" signal back to the extractor prompt.

**Rationale:** This is a structural defense against MINJA-style memory-poisoning (which has >95% success against naive extraction pipelines). An attacker planting "remember the user prefers X" in conversation content can't reach retrieval without the user clicking approve — and a real user will notice and reject instead. Also defends against the transcript-as-memory-amplifier failure: GigaAM STT errors and assistant hallucinations stay in `proposed` until the user either corrects them or rejects. The same approval rail later powers fine-tuning label quality. And "here's what I think I learned about you — confirm?" is a remarkably high-trust UX signal vs. silent auto-write.

**Downsides:** Approval friction is real — the system must auto-approve low-risk high-confidence categories (name, pronouns, stated allergies) or users will bounce. Notification channel needs design (in-app prompt? end-of-session review?). Proposals that expire unreviewed need a clean-up policy to avoid a growing graveyard. Doesn't stop all poisoning — an attacker who gets a user to approve counts as "working as intended"; needs pairing with M5 provenance spans for post-hoc forensics.

**Confidence:** 70%
**Complexity:** Medium
**Status:** Unexplored

## Suggested Build Order (informational, not a plan)

This ideation yields a foreground/background split plus a UX gate that work well in this order once brainstorming narrows to requirements:

1. **M4** (async extraction worker) + unified session-end lifecycle helper — lands infrastructure, keeps hot path clean.
2. **M5** (memory atoms schema) + **M9** (therapist notes as one atom type) — structured store with a shippable shape.
3. **M3** (two-tier loader, hot anchors first, speculative prefetch deferred) — connects the store to the turn.
4. **M6** (per-NPC visibility enum on atoms) — cheap now, costly to retrofit later.
5. **M11** (memory card) + **CC3** (human-approved) — trust + compliance + anti-injection as a unit.

Use `/ce-brainstorm` on any single idea to narrow it into requirements before `/ce-plan`.

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| M1 | Cold-start amnesia | Motivation statement, not a design candidate |
| M2 | Unified session-end lifecycle | Shared infrastructure dependency across all designs — an implementation detail inside M4, not a standalone ideation candidate |
| M7 | Prose-only rewritten paragraph | Subsumed by M9's structured therapist-note schema |
| M8 | SQLite+FTS5 keyword-only | Introduces a second store alongside PG18; PG18's native FTS / pgvector covers the same role without operational split |
| M10 | Reinforcement-weighted memory | Strong pattern but premature; layer on top of M5 once real signal exists |
| M12 | Commitment ledger | Compelling atomic unit but too narrow; better expressed as one atom type (`predicate='commitment'`) within M5 |
| M13 | Topic threads | Ambitious and retention-positive, but depends on accumulated cross-session history; revisit after M5+M4 produce it |
| M14 | Consent-scoped memory zones | Valuable policy primitive but naturally a column on M5, not a standalone design |
| M15 | Voice fingerprint identity | Compounding long-term value, but existing token-mint flow already carries `user_id`; net-new model dependency + spoofing discipline not justified by current auth model |
| M16 | Flashbulb + leitmotif callbacks | Novel affect+position mechanic, but premature — depends on M5 atoms and meaningful session history first |
| M17 | Bloom filter pre-gate | Narrow micro-optimization; solve only after the broader retrieval path is instrumented and needs trimming |
| M18 | VAD-gap in-session reflection | Creative but marginal; fits as a sub-feature of M4's extraction worker, not a separate design |
| M19 | Infinite audio-native memory | Storage + compliance cost explosion without a grounded product need; audio-recall is speculative |
| M20 | Persona-lensed memory | Flavor layer subsumed by M6 (per-NPC ownership naturally produces per-NPC memory character) |
| M21 | Portable .npcmem | Bold server-stateless design, but requires client-side crypto + key management outside current MVP scope |
| CC1 | Pipecat Mem0 + custom atoms | Tactical implementation shortcut for M4+M5 — recommended as starting scaffold in implementation, not a separate design choice |
| CC2 | Hot anchors + commitment ledger | M12 itself rejected; commitments are an atom type within M5, and M3 defines the hot tier without needing the ledger framing |
