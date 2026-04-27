# Feature Specification: Real-Time Voice Assistant Core

**Feature Branch**: `001-voice-assistant-core`
**Created**: 2026-04-23
**Status**: Draft
**Input**: User description: "Real-time voice assistant: LiveKit (WebRTC) ↔ Pipecat agent (VAD → GigaAM STT → LLM → TTS) with a FastAPI token-minting backend."

## Clarifications

### Session 2026-04-23

- Q: Which NPC archetype(s) are in scope for v1? → A: Helper/Guide NPCs only; other archetypes (tavern keeper, quest giver, etc.) deferred.
- Q: Which languages does v1 support? → A: Russian only; multilingual support deferred.
- Q: Does v1 write anything back to the game client (map geotagging, player-state queries, in-game actions)? → A: No — v1 is conversations-only; the voice assistant has no outbound integration with the game client.
- Q: What is v1's functional scope — full platform vs. minimal slice? → A: Minimal MVP — smallest vertical slice that delivers the voice loop for a Helper/Guide NPC in Russian.
- Q: Is v1 releasable to end-players on its own, or gated on a KB-grounding layer? → A: v1 is an early-stage development and internal-testing milestone. Acceptance targets a dev/test-quality Helper/Guide NPC conversation loop. Production release criteria (including grounding) will be decided in a later iteration and are out of scope for this spec. Any prior draft plans under `docs/plans/` are exploratory, not binding requirements.
- Q: Which user stories (US1–US5) stay in v1? → A: All five stay. "Minimal" means no *new* surface area beyond the current voice-loop conversation (no game-client integration, no additional features); it does not mean pruning already-implemented capabilities (end-call, barge-in, multi-tenant rate limiting, transcript display).
- Q: How is a session bound to the Helper/Guide persona at session-creation time? → A: Session-creation request accepts an optional `npc_id` field. If omitted, the server binds the session to `"helper_guide"`. If present, `"helper_guide"` is the only value accepted in v1; any other value is rejected. The `sessions.npc_id` column is populated either way.
- Q: Does a session track an individual player identity, or only the tenant? → A: Tenant-scoped anonymous callers in v1. Sessions and transcripts are linked to a tenant only; no player-id plumbing. Rate limiting stays per-tenant + per-IP. Per-player identity, if needed later, is a separate future spec.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Start a voice conversation and hear a low-latency reply (Priority: P1)

A player opens the web client, presses **Start Call**, and speaks. The system responds
with synthesized voice within the latency budget after the player stops speaking. The
conversation is audio-only, bidirectional, and feels like a natural phone call.

**Why this priority**: This is the product. Nothing else matters if the first turn does
not complete end-to-end within the promised response time. This story is the
measurable MVP — if it works, the platform has value; if it doesn't, nothing else
rescues it.

**Independent Test**: A tester can open the web client, click **Start Call**, speak a
Russian sentence, and confirm they hear an audible spoken response whose first audio
lands within 800 ms of the moment they stop talking, on the documented reference
hardware. Success is observable without the tester knowing anything about the
underlying pipeline.

**Acceptance Scenarios**:

1. **Given** a verified tenant caller with a fresh session request, **When** the caller
   clicks **Start Call** and speaks a short Russian utterance, **Then** the caller
   hears the first word of the assistant's reply within 800 ms of end-of-speech, and
   the reply audio continues without audible gaps until the assistant finishes.
2. **Given** an ongoing voice session, **When** the caller speaks a second utterance,
   **Then** the assistant responds to the second utterance within the same latency
   budget, and the assistant's reply reflects the caller's actual words (not generic
   filler).
3. **Given** the caller's utterance is very short or silent (e.g. <300 ms of voiced
   audio, or a cough), **When** the voice-activity detector marks end-of-speech,
   **Then** the system does not fabricate a transcription; either it prompts the
   caller to repeat or it remains silent, without producing a hallucinated reply.

---

### User Story 2 - End a call cleanly and record usage (Priority: P2)

A player finishes their conversation and clicks **End Call** (or closes the browser
tab). Both sides tear down immediately: the agent stops running, the room is cleaned
up, no further billing or audio frames flow, and the session's total duration is
recorded for the tenant's usage accounting.

**Why this priority**: Real-time voice is expensive per second (GPU time, TTS cycles).
A session that fails to terminate silently burns resources and skews usage numbers.
P1 proves the product works; P2 proves the product can be operated.

**Independent Test**: A tester starts a session, speaks once, confirms the reply, then
clicks **End Call**. Within a few seconds, the tester can verify (via an operator-
facing tool or database inspection) that the session is marked terminated, the
recorded audio-seconds matches the elapsed call duration within a small margin, and no
further agent activity is logged for that room.

**Acceptance Scenarios**:

1. **Given** an active voice session, **When** the caller clicks **End Call**,
   **Then** the agent stops processing within 5 seconds, the room is closed, and the
   session record reflects the final state (terminated, with duration).
2. **Given** an active voice session, **When** the caller closes the browser tab or
   loses network connectivity for longer than the grace window, **Then** the system
   detects the disconnect and tears down the session automatically, recording the
   duration up to the point of disconnect.
3. **Given** a caller has ended a session, **When** the same caller immediately
   starts a new session, **Then** the new session gets a fresh room; no audio, state,
   or transcript from the prior session is reachable from the new session.

---

### User Story 3 - Interrupt the assistant mid-reply (barge-in) (Priority: P2)

The assistant is speaking. The caller starts talking over it. The assistant stops its
current reply within a small, predictable window and begins processing the new
utterance. The caller does not have to wait for the assistant to finish.

**Why this priority**: Voice conversations are interruptible by nature. A one-way
voice bot that ignores the caller until it finishes feels broken. P1 delivers the turn
shape; P3 delivers the conversational shape.

**Independent Test**: A tester starts a session and asks a question that will produce a
long reply (e.g. a multi-sentence answer). Mid-reply, the tester starts speaking.
Within a short, predictable window, the assistant stops talking and begins responding
to the new input.

**Acceptance Scenarios**:

1. **Given** the assistant is actively speaking a reply, **When** the caller begins
   speaking, **Then** the assistant's audio output stops within the documented
   barge-in window, and the system begins processing the caller's new utterance.
2. **Given** barge-in has occurred, **When** the caller's new utterance completes,
   **Then** the system replies only to the new utterance — not a concatenation of the
   old (interrupted) reply plus the new one.

---

### User Story 4 - Tenant-scoped access and rate limiting (Priority: P2)

Only authorized tenants can start sessions. Each tenant has a fair-use ceiling so a
misbehaving client (or a compromised key) cannot starve other tenants of capacity.
Abusive traffic gets a clear "try again later" signal rather than silently degrading
the shared service.

**Why this priority**: The platform is multi-tenant from day one. Without this, a
single noisy tenant or a leaked key can take down the service for everyone.

**Independent Test**: A tester with no tenant credentials attempts to create a session
and is rejected. A tester with valid credentials who exceeds the configured per-tenant
session creation rate is rejected with a "retry after" signal. A third tenant's
requests continue to succeed throughout.

**Acceptance Scenarios**:

1. **Given** a request to start a session arriving without valid tenant credentials,
   **When** the request reaches the session-creation surface, **Then** the request is
   rejected with an authentication error, and no room or agent is allocated.
2. **Given** a tenant who has already exceeded their configured rate over the sliding
   window, **When** that tenant tries to start another session, **Then** the request
   is rejected with a clear "rate limited — retry after N seconds" signal.
3. **Given** a tenant at their rate limit, **When** a different (unaffected) tenant
   starts a session, **Then** that second tenant's session succeeds and is not
   degraded by the noisy tenant.

---

### User Story 5 - Real-time transcript visible during the call (Priority: P3)

As the caller speaks and the assistant replies, the caller sees a running transcript
in the web client: their own words (as finalized by the recognizer) and the
assistant's words. The transcript is for recall, readback, and accessibility — it does
not replace the voice channel.

**Why this priority**: Transcripts are a strong UX affordance (accessibility, noisy
environments, "what did it just say?") but are not required for the core voice loop to
work. P1–P4 deliver a usable voice product; P5 makes it better.

**Independent Test**: During a live session, a tester speaks a distinctive sentence
and verifies it appears in the client transcript view within a few seconds. The
assistant replies, and those words also appear attributed to the assistant. Closing
and reopening the session starts a fresh transcript.

**Acceptance Scenarios**:

1. **Given** an active session, **When** the caller finishes an utterance, **Then**
   the finalized transcript of that utterance appears in the web client within a few
   seconds, attributed to the caller.
2. **Given** an active session, **When** the assistant speaks a reply, **Then** the
   text of the reply appears in the web client transcript, attributed to the
   assistant, and stays visible for the remainder of the call.

---

### Edge Cases

- **Non-Russian input**: Caller speaks in a language other than Russian (e.g. CJK
  characters detected, or English). The system does not attempt to transcribe it as
  Russian; it responds with a polite "Russian only" message in Russian or prompts the
  caller to speak in Russian.
- **Very short utterance**: A cough, keyboard tap, or single phoneme. The system
  suppresses it rather than generating a hallucinated transcript and reply.
- **Concurrent session creation under load**: Multiple tenants start sessions in the
  same second. Each gets an isolated room; none can hear or be heard by any other.
- **Mid-call GPU contention**: Under peak load, the STT/LLM stage takes longer than
  budget. The system does not crash; it completes the turn late and records the
  latency overshoot for operator review. The caller experiences a longer pause, not a
  dropped call.
- **Token expiry mid-call**: A caller's join credential expires while still in the
  room. The system does not forcibly disconnect a caller mid-utterance; the session
  completes naturally or times out on inactivity.
- **Repeat start-call taps**: The caller double-clicks **Start Call** quickly. The
  system creates exactly one session for that intent; the duplicate request is
  rejected or merged, not used to provision a second agent.
- **Disconnect during the first turn**: The caller loses network between pressing
  **Start Call** and speaking. The provisioned room and agent are reclaimed
  automatically rather than hanging idle.
- **Assistant silent / dead-air**: The pipeline produces no audio for longer than a
  documented threshold. The caller sees either a polite fallback message or the
  session is ended with an error recorded for operator review, never an indefinite
  silent room.
- **Abuse via rapid reconnect**: A misbehaving client rapidly starts and ends
  sessions. Per-tenant and per-IP rate limiting absorbs this without affecting other
  tenants.

## Requirements *(mandatory)*

### Functional Requirements

**Session lifecycle**

- **FR-001**: The system MUST expose an authenticated session-creation surface that,
  for a valid tenant, returns a one-shot, short-lived join credential and a unique
  room identifier. The request MAY include an `npc_id` field; if omitted, the server
  MUST bind the session to `"helper_guide"`. In v1, `"helper_guide"` is the only
  accepted value; any other `npc_id` MUST be rejected. The bound persona MUST be
  recorded on the session record.
- **FR-002**: Each session MUST receive its own isolated room. Audio, transcripts, and
  agent state from one session MUST NOT be reachable from any other session.
- **FR-003**: On session creation, the system MUST provision a conversational agent
  dedicated to that room before the caller joins, so the first turn is not delayed by
  cold-start.
- **FR-004**: The system MUST provide an explicit end-session action that terminates
  the room, stops the agent, records final usage, and is idempotent (calling it twice
  is safe).
- **FR-005**: The system MUST detect client disconnects (closed tab, lost network
  beyond a grace window) and automatically terminate the session as if the caller had
  pressed **End Call**.

**Conversation loop**

- **FR-006**: The system MUST detect end-of-speech automatically — the caller does not
  press a button to indicate "I'm done talking".
- **FR-007**: The system MUST produce the first audio of the assistant's reply within
  **800 ms** of end-of-speech on documented reference hardware.
- **FR-008**: The system MUST suppress hallucinated replies on utterances that fall
  below documented voice-activity thresholds (too short, too few tokens, non-speech).
- **FR-009**: The system MUST allow the caller to interrupt (barge in on) the
  assistant's reply; on barge-in, the assistant's audio output MUST stop within the
  documented window, and processing MUST pivot to the new caller utterance.
- **FR-010**: The system MUST respond only to Russian utterances in v1. Non-Russian
  utterances MUST be handled per the "Non-Russian input" edge case above, not silently
  mistranscribed. Multilingual support is deferred to a separate spec.

**Transcripts (P3-gated)**

- **FR-011**: The system MUST deliver finalized caller transcripts to the web client
  during the call, attributed to the caller, with a short, bounded delay.
- **FR-012**: The system MUST deliver the assistant's reply text to the web client
  during the call, attributed to the assistant.

**Multi-tenant boundaries**

- **FR-013**: Every session-creation request MUST be authenticated against a tenant
  credential; unauthenticated requests MUST be rejected before any resources are
  allocated.
- **FR-014**: The system MUST enforce a per-tenant rate limit on session creation and
  a per-source-IP rate limit on unauthenticated admin and webhook surfaces. Rate-limit
  decisions for one tenant MUST NOT affect another tenant.
- **FR-015**: Over-limit requests MUST receive a clear, machine-readable "rate
  limited" response including a retry-after hint.
- **FR-016**: Only join credentials issued for the exact target room MUST be
  accepted; a credential for room A MUST NOT admit the bearer into room B.
- **FR-017**: Join credentials MUST expire within a short, documented TTL (≤15 min
  for callers, ≤30 min for agent-minted room credentials) so a leaked credential has
  a bounded blast radius.

**Observability & operations**

- **FR-018**: The system MUST record, for each session, at minimum: tenant, room id,
  start time, end time, total audio seconds, and terminal state. These records MUST
  be queryable by operators.
- **FR-019**: Pipeline failures (STT error, LLM failure, TTS failure) MUST surface as
  redacted error signals to the caller's client rather than raw stack traces, and
  MUST be logged on the server with enough context (room id, turn id, tenant) to
  reproduce.
- **FR-020**: The system MUST expose a health-check surface that reports whether core
  dependencies (token minting, agent dispatcher, media server, LLM endpoint) are
  reachable.

**Secrets & configuration**

- **FR-021**: No credential, API key, or model token may be present in source or
  container images; all such values MUST be read from environment or a secret store
  at start-up and the service MUST refuse to start if any required value is missing.

### Key Entities *(include if feature involves data)*

- **Tenant**: An authorized customer of the platform. Owns one or more API keys, has a
  configured rate-limit profile, and accumulates usage records. All sessions are
  scoped to exactly one tenant.
- **Session**: One end-to-end voice conversation from **Start Call** to termination.
  Belongs to a tenant, has a unique room id, carries start/end timestamps, total
  audio seconds, and a terminal state (completed, ended-by-caller, timed-out, errored).
- **Room**: The media-isolation boundary. One session = one room. A room has a bounded
  lifetime, a dedicated agent, and is unreachable after teardown.
- **Caller Join Credential**: A short-lived, room-specific token that lets a single
  web client join the room. Not reusable across rooms or after expiry.
- **Turn**: One caller utterance plus the resulting assistant reply within a session.
  Characterized by its end-of-speech timestamp, transcript, reply text, first-audio-
  out latency, and any pipeline errors.
- **Transcript Line**: An attributed piece of text (caller or assistant) associated
  with a session and (when known) a turn. Delivered to the client as the call
  progresses.
- **Usage Record**: The billable / operator-visible record of a session's resource
  consumption, primarily audio-seconds, grouped by tenant and time window.
- **NPC Persona**: The conversational character that wraps the language model. v1
  defines exactly one persona: **Helper/Guide**. A persona determines the system
  prompt, allowed tone, refusal-on-scope-miss behavior, and (when the dependent KB
  grounding lands) the knowledge-base binding. A session is associated with exactly
  one persona for its lifetime.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: **95%** of conversational turns produce first audio out within **800 ms**
  of end-of-speech on documented reference hardware under nominal (non-saturated)
  load. No single turn exceeds **1500 ms** under nominal load.
- **SC-002**: **99%** of session-creation requests from valid tenants under their
  rate-limit ceiling succeed within **2 seconds**, measured end-to-end from
  **Start Call** to the caller being joined in the room.
- **SC-003**: Barge-in is honored: when the caller starts speaking during an
  assistant reply, assistant audio stops within **250 ms** and the next turn begins
  processing the new utterance.
- **SC-004**: Zero cross-room leakage in any audit: across a representative load
  test running many concurrent sessions, **0** audio frames, transcript lines, or
  assistant outputs from one session are observed in another session.
- **SC-005**: Session teardown completes within **5 seconds** of the caller pressing
  **End Call** or the system detecting a disconnect; agent activity after teardown is
  **0**.
- **SC-006**: Usage records agree with observed call duration to within **±1 second**
  for **100%** of sessions (no silent under- or over-billing).
- **SC-007**: A malformed or unauthenticated session-creation request is rejected in
  under **200 ms** and never provisions a room or agent.
- **SC-008**: A misbehaving tenant pinned at their rate limit does not degrade a
  separate tenant's latency by more than **10%**; an unrelated tenant continues to hit
  SC-001 throughout the noisy-neighbor scenario.
- **SC-009**: In a 30-minute sustained run on reference hardware with continuous
  conversation, **zero** sessions terminate unexpectedly due to internal failure
  (memory leak, deadlock, lost connection to a dependency).
- **SC-010**: New operators, reading only operator-facing logs and session records,
  can reconstruct what happened in a failed turn in under **5 minutes** without
  consulting source code.

## Assumptions

- **Client is a modern browser.** The caller-side client is a modern desktop or
  mobile browser with WebRTC support; native mobile and telephony gateways are out of
  scope for this spec.
- **Russian only in v1.** All sessions are Russian-language. Multilingual turn-by-turn
  switching and non-Russian-language support are out of scope for v1.
- **Pre-warmed speech model.** The speech-recognition model is pre-loaded by the
  agent at boot time so per-session cold-start does not contribute to the latency
  budget. First-user-after-deploy may incur extra latency that is tolerated outside
  the SC-001 measurement window.
- **Reference hardware is documented in the constitution.** The 800 ms target
  assumes GPU-class hardware within the range stated in the project constitution's
  Performance & Latency Standards section. Cheaper hardware is a valid deployment
  target but is not bound by SC-001.
- **Operator dashboards are out of scope.** Operator-facing observability is served
  by logs and direct database inspection in v1. A UI for operators is a later
  feature; SC-010 must be satisfied without one.
- **Billing runs outside this feature.** This spec is responsible for accurate usage
  *records* (SC-006). Conversion of usage to invoices is a separate system.
- **Single NPC archetype in v1: Helper/Guide.** Other NPC archetypes (tavern keeper,
  quest giver, etc.) are out of scope for v1. Each session binds to exactly one NPC
  persona for its lifetime.
- **v1 is an early-stage development and internal-testing milestone.** Acceptance is
  defined against a dev/test-quality Helper/Guide conversation loop, not against a
  production release to end-players. Any exploratory drafts under `docs/plans/`,
  `docs/ideation/`, or `docs/brainstorms/` are non-binding — this spec does not
  depend on those documents and does not inherit requirements from them.
- **KB grounding is out of scope for v1.** Whether and how to ground the NPC on a
  curated knowledge base is a future iteration. v1's Helper/Guide uses whatever
  conversational behavior is configured for it at the time of testing; hallucinated
  game facts are an acknowledged early-stage limitation, not a v1 defect.
- **No game-client integration in v1.** The voice assistant does not call back into
  the game client, read player state, geotag on the game map, or trigger in-game
  actions. v1 is conversations-only.
- **External LLM availability.** The LLM endpoint (local or external OpenAI-
  compatible) is assumed available. LLM-endpoint outage is an edge case surfaced to
  operators; high-availability of the LLM is a separate infrastructure concern.
