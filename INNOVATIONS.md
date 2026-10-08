# Innovations in the Modernized GTM Agent

A summary of the design contributions in this repo (`gtm_agent_modernized_v1.0`),
which combines **three independently useful techniques** in one working agent:
skills-bound tool disclosure, per-query LLM routing, and turn-budget latency
engineering — all verifiable offline.

---

## 1. Binding tools to skills (progressive tool disclosure)

**Problem.** A production GTM agent needs dozens of tools (CRM, call
transcripts, LinkedIn, enrichment, scheduling…). Shipping every schema up
front wastes context and invites the model to call tools that belong to a
different workflow.

**Innovation.** A 50-tool simulated stack in which 47 tools are **deferred**
(`extras={"defer_loading": True}`). Each skill's `SKILL.md` frontmatter
declares the tools it owns via `metadata.include_tools`, and the deepagents
`SkillsMiddleware` discloses a tool's schema to the model **only after the
model reads the skill that lists it**. Before that, the tool call fails as
unknown.

Consequences:

- A lead follow-up run never sees the 38 tools owned by the other skills —
  context savings and capability gating scale with the tool count.
- One skill (`competitor-analysis`) is *methodology-only*: it binds zero
  tools and rides on the model's own knowledge, showing the spectrum from
  tool-bound to knowledge-only skills.
- The mechanism is **proven by an offline smoke test** (`tests/smoke_test_skill_tools.py`)
  using a scripted `GenericFakeChatModel`: it snapshots the bound tool list on
  every model call and asserts deferred schemas are disclosed only after the
  `SKILL.md` is read — no API key required.

## 2. Per-query LLM routing via middleware (`decision_harness`)

**Problem.** A single frontier model for every GTM workflow is wasteful:
lead lookups are mechanical, drafting and ambiguous escalations are not.

**Innovation.** [`decision_harness`](https://github.com/cwijayasundara/deep_agents_utils_v1.0)
is layered on the skills-bound agent as a **model-selection middleware**:

- **One-shot classification.** Each thread's first query is classified in a
  single typed-questions request: tier choice + confidence, complexity, and a
  planning predicate. (OpenAI Decisions API when configured.)
- **Graceful degradation.** Without any key, the exact same routing shape
  falls back to a deterministic offline heuristic (`heuristic+gtm`) — zero
  network, zero cost, fully demoable.
- **Sticky per thread.** The chosen model sticks for the thread's lifetime,
  keeping the provider prompt cache warm; classification happens once, not
  per turn.
- **Tier escalation on failure.** Provider errors escalate to the next tier
  rather than failing the run.
- **Catalog overlay pattern.** `gtm_agent/catalog_overlay.json` merges a
  project-local view (four models, three tiers, prices, capability and
  `intelligence` fields) over the harness defaults, so the routing policy is
  data, not code.

## 3. Full-loop cost & latency observability

**Innovation.** Routing decisions are only useful if they're measured. A
custom `CallTelemetryMiddleware` (outermost, so retry/escalation time is
included) times every model call and captures token details **including
prompt-cache read and cache-creation counts** — the number that actually
validates the sticky-model choice. A `RoutingTelemetry` collector joins:

- routing decisions (model, tier, reason, stickiness, classifier used,
  classification latency, escalation events)
- per-call latency, input/output tokens, cache hits
- **dollar cost** per call and per run

…into a printed summary plus an exported JSON ledger (`out/usage/ledger-*.json`),
turning model-selection claims into inspectable evidence.

## 4. Turn-budget discipline as prompt architecture

**Innovation.** Agent latency is dominated by model round trips, so the
system prompt and skills encode an explicit **turn economy**:

- target budget of 3–4 model turns per workflow
- turn 2 = *one* message with every independent lookup batched in parallel;
  turn 3 reserved only for lookups that depended on turn-2 results
- tool-call turns carry no prose; narrative is saved for one dense
  final summary (≤120 words)
- pinned `SKILL.md` content is never re-read

This treats prompt engineering as **latency engineering**, with the
telemetry ledger (§3) available to verify the budget is actually met.

## 5. Guardrails by workflow design (human-in-the-loop)

**Innovation.** The outbound workflow hard-codes an order —
*do-not-send checks → research → draft → rationale → queue for approval* —
and never sends anything autonomously:

- `check_contact_history` gates everything: open support tickets,
  `do_not_send` flags, or a teammate's recent outreach stop the draft
  entirely, with a human-touch recommendation instead
- drafts are queued to Slack with rationale **and sources** attached, for
  explicit rep approval
- every draft is explainable by construction (rationale for the chosen angle)

## 6. Offline-first, evidence-first engineering

**Innovation.** Every system of record (Salesforce, Gong, LinkedIn, Exa,
BigQuery, Zendesk, Slack) is mocked with in-memory fixtures, so:

- the full demo, including routing and the disclosure smoke test, runs with
  **zero credentials** (fake chat model for tests, heuristic classifier for
  routing)
- the same code path is exercised live when keys are present — mock/live is
  a runner flag, not a fork

---

## Honest-evidence note

The routing *mechanism* is measured (§3), but the catalog **contents** are
configured defaults: all prices in `catalog_overlay.json` are explicit
placeholders, and the `intelligence` scores are hand-assigned relative
rankings, not benchmark results. Validate tier→model assignments against
provider pricing pages and a small eval set before production.

## File map

| Concern | Where |
|---|---|
| Skills-bound agent, turn-budget prompt | `gtm_agent/agent.py`, `skills/*/SKILL.md` |
| 50-tool mock stack, deferred loading | `gtm_agent/tools.py` |
| Routing middleware, telemetry, ledger | `gtm_agent/routing.py` |
| Catalog overlay (data, not code) | `gtm_agent/catalog_overlay.json` |
| Disclosure smoke test (offline) | `tests/smoke_test_skill_tools.py` |
| CLI: live/routed runs, `route` inspector | `main.py` |
