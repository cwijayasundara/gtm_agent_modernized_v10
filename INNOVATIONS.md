# Innovations in the Modernized GTM Agent

A summary of the design contributions in this repo (`gtm_agent_modernized_v1.0`),
which combines **five independently useful selection/invocation layers** plus a
per-tool approval boundary in one working agent: skill-bound tool disclosure,
OpenAI Decisions intent routing, conversation-aware tool selection, read-only
programmatic tool calling, and defined invocation — plus per-query model routing
and turn-budget latency engineering, all verifiable offline.

The central claim is narrower than "all best practices": the repo is a working
reference for **how to resolve user intent to a workflow, and a workflow to a
small tool surface, without putting the entire catalog in the main model's
context**. Section 1.0 makes that selection stack explicit; §11 lists what the
pattern does *not* guarantee.

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
model reads the skill that lists it**.

Consequences:

- A lead follow-up run never sees the 35 tools owned by the other skills
  (context savings and capability gating scale with the tool count).
- One skill (`competitor-analysis`) is *methodology-only*: it binds zero
  tools and rides on the model's own knowledge, showing the spectrum from
  tool-bound to knowledge-only skills.
- The mechanism is **proven by an offline smoke test** (`tests/smoke_test_skill_tools.py`)
  using a scripted `GenericFakeChatModel`: it snapshots the bound tool list on
  every model call and asserts deferred schemas are disclosed only after the
  `SKILL.md` is read — no API key required.

### 1.0 Intent is matched to a skill, then to tools

The system prompt is not a 50-way dispatch table. `GTM_SYSTEM_PROMPT` is a
**policy prompt**: workflow order, safety gates, approval semantics, and turn
budget. It does not enumerate tool routes.

There are three ways a request reaches a skill:

1. **Explicit workflow** — `lead`, `briefing`, and `prep` pin the skill before
   the graph starts. The user already chose the workflow via the CLI.
2. **Intent router** — `auto` and `intent` call `IntentRouter` before the agent
   is built. It asks OpenAI Decisions for a `choice` over the runnable
   workflows plus a `needs_clarification` predicate. High confidence pins the
   skill; low confidence asks for a target instead of guessing.
3. **Model fallback** — when neither of the above applies, `SkillsMiddleware`
   still injects each skill's description and the model matches the request
   against it.

The full path is:

```
user request
  │
  ├─ explicit command ───────► pinned skill
  ├─ IntentRouter ───────────► skill / clarify / general
  └─ model fallback ─────────► match SKILL.md description
                                 │
                                 ▼
  read or pin SKILL.md ──► metadata.include_tools undefer only that skill's
                                 tools (SkillsMiddleware)
                                 │
                                 ▼
  model picks among disclosed schemas ──► normal tool call / PTC / direct
```

`outbound-followup` owns 12 tools, `account-intelligence` 15,
`customer-health` 12, and `meeting-prep` 8; `competitor-analysis` is
methodology-only. `metadata.include_tools` supplies the tool family once the
skill is selected.

What the system prompt *does* control is call shape: batch independent lookups,
do not re-read a pinned skill, gate before drafting, keep the final summary
short. That is §4's latency policy, not tool selection.

The five layers answer five different questions:

| Layer | Question | Mechanism | Where |
|---|---|---|---|
| 1. Skill binding | How does the chosen workflow's tool family become visible? | `include_tools` + deferred schemas + `SkillsMiddleware` | `skills/*/SKILL.md` |
| 2. Intent router | Which workflow is this request? | explicit pin or OpenAI Decisions `Choice` + clarification predicate | `main.py`, `gtm_agent/intent.py` |
| 3. Tool selection | Which few tools are relevant *now*? | conversation transcript + OpenAI Decisions | `gtm_agent/selection.py` |
| 4. Interpreter/PTC | How can many read-only calls be composed? | one JS program calling allowlisted `tools.*` | `gtm_agent/ptc.py` |
| 5. Defined workflow | Can the model be removed entirely? | authored tool order | `gtm_agent/direct.py` |

Only layer 1 is mandatory. Layers 2–5 are independent knobs and compose via
`build_routed_agent(..., extra_middleware=[…])`. Action tools are outside PTC
and gated by `HumanInTheLoopMiddleware` (`gtm_agent/hitl.py`). None of this is
an intent→tool switch inside the system prompt.

### 1.1 The two halves: a registry and a playbook

The feature is a contract between code and markdown, and neither half is
useful alone.

```python
# gtm_agent/tools.py — the registry says "this schema is withheld"
DEFERRED = {"defer_loading": True}

@tool(extras=DEFERRED)
def check_contact_history(lead_id: str) -> str:
    """Check whether a teammate already reached out to this lead/account recently..."""
```

```yaml
# skills/outbound-followup/SKILL.md — the playbook says "what I need"
metadata:
  include_tools: lookup_lead check_contact_history get_gong_transcripts
```

`defer_loading` is not a repo convention: it is a documented field of the
tool definition (it passes through `langchain_anthropic` to the provider), and
deepagents uses the same key in `extras` to decide what to withhold. The agent
is still built with the whole catalog — nothing is unregistered:

```python
# gtm_agent/agent.py
return create_deep_agent(
    model=_make_model(model),
    tools=tools.ALL_TOOLS,          # all 50 registered
    skills=["/skills/"],            # wires SkillsMiddleware
    backend=FilesystemBackend(root_dir=PROJECT_ROOT),
)
```

The withholding happens per **model call**, in middleware, not at
tool-registration time.

### 1.2 The flow, end to end

Build time — nothing is hidden yet; the flag is only carried:

```
  gtm_agent/tools.py                           skills/outbound-followup/SKILL.md
  ┌─────────────────────────────────────┐      ┌────────────────────────────────┐
  │ ALL_TOOLS (50)                      │      │ metadata:                      │
  │   3  always available               │      │    include_tools: lookup_lead  │
  │  47  deferred                       │      │      check_contact_history ... │
  │      extras={"defer_loading": True} │      │                                │
  └────────────────┬────────────────────┘      └──────────────┬─────────────────┘
                   │                                          │
                   └────────────────────┬─────────────────────┘
                                        ▼
          create_deep_agent(tools=tools.ALL_TOOLS, skills=["/skills/"])
                                        │
                                        ▼
          ┌────────────────────────────────────────────────────────────────┐
          │ all 50 tools stay registered on the agent                      │
          │ SkillsMiddleware is placed in the graph, before prompt caching │
          └────────────────────────────────────────────────────────────────┘
```

Per model call — the middleware decides what leaves the process:

```
   conversation state: messages + the 50 registered tools
                     │
                     ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ SkillsMiddleware.wrap_model_call                                           │
  │                                                                            │
  │  1 _find_skill_reads ──► did the model read a SKILL.md that lists tools?   │
  │       │                                                                    │
  │  2 _plan_disclosure ──► match each include_tools name to a registered      │
  │       │                 deferred tool; anchor it at the earliest such read │
  │  3 _bind_disclosures ─► disclosed: drop "defer_loading"                    │
  │       │                 others:   keep "defer_loading"                     │
  │  4 record {tool → include name} in `_skill_tools_disclosed`                │
  └──────────────────┬──────────────────────────┬──────────────────────────────┘
                     │                          │
                     no read: 47 deferred       read: +12 undeferred
                     ▼                          ▼
                     11 schemas shipped         23 schemas shipped
```

The run as the model experiences it:

```
  MODEL CALL 1                                    11 schemas shipped
  ┌────────────────────────────────────────────────────────────────┐
  │ user: "Follow up on lead-1042."                                │
  │ → read_file("/skills/outbound-followup/SKILL.md")              │
  └──────────────────────────────┬─────────────────────────────────┘
                                 ▼   ToolMessage: frontmatter + the ordered checklist

  ── middleware now finds the read → the 12 names are undeferred ──

  MODEL CALL 2                                    23 schemas shipped
  ┌────────────────────────────────────────────────────────────────┐
  │ → check_contact_history("lead-1042")     ← invisible on call 1 │
  └──────────────────────────────┬─────────────────────────────────┘
                                 ▼   ToolMessage: no do-not-send flag, no prior touch

  MODEL CALL 3                                    23 schemas shipped
  ┌────────────────────────────────────────────────────────────────┐
  │ → lookup_lead · web_search_company · draft_slack_followup …    │
  └────────────────────────────────────────────────────────────────┘
      (the 35 tools owned by intel / health / prep stay deferred throughout)
```

And the reverse — disclosure is anchored to a position in the transcript, so it
can be lost:

```
  compaction evicts /skills/outbound-followup/SKILL.md from the transcript
  │
  ▼
  _find_skill_reads finds no read → the 12 tools go dark again
  │
  ▼
  11 schemas shipped, 47 deferred
  (the playbook left the context, so the capability it carried left too)
```

### 1.3 What happens on each model call

`SkillsMiddleware.wrap_model_call` (deepagents 0.7.23, `middleware/skills.py`
→ `middleware/_skill_tools.py`) runs this pipeline before the provider is
called:

1. **Find skill reads.** `_find_skill_reads` scans the conversation for a
   *successful* `read_file` result whose path resolves to a discovered
   `SKILL.md` that declares tools. Any `offset`/`limit` counts, and so does
   content that was later truncated — only the call and the result's status
   are inspected. A *pinned* skill message counts too.
2. **Resolve include names.** Each name in the read skills'
   `metadata.include_tools` is matched against the tools bound to the request.
   In this repo every name matches, so disclosure goes through the *deferred*
   path and no middleware-side tool resolver is needed.
3. **Anchor the disclosure.** `_plan_disclosure` records the **earliest** read
   that produced each tool. Disclosure is *positional*, not global: a tool
   becomes callable from the point in the transcript where its skill was read.
4. **Undefer.** `_bind_disclosures` copies those tools with the
   `defer_loading` key removed, so their schemas are sent on this call onward.
   The other 35 deferred tools keep the key and stay dark.
5. **Record and gate.** The `{tool name → include name}` map is checkpointed in
   private state (`_skill_tools_disclosed`), so the tool node can resolve a
   call that arrived from an inline disclosure.

Placement matters and `create_deep_agent` handles it: the middleware sits
*after* compaction, model fallback and routing (so it sees the conversation the
model actually receives) and *before* prompt caching.

### 1.4 How the schema actually reaches the model

deepagents picks one of two transports per model:

| Model family | Transport |
|---|---|
| Anthropic `claude-opus-5`, `claude-fable-5`, `claude-mythos-5`, `claude-opus-4-8` | Mid-conversation `tool_addition` block carrying the full `tool_definition` |
| OpenAI `gpt-6-*`, `gpt-5.6-*` (Responses API only) | `additional_tools` input item |
| Anything else — including this repo's default `claude-haiku-5-5` | Fallback: the tool is bound in `tools` **without** `defer_loading`, so its schema simply appears in the tool list from that call on |

The fallback is why the feature is portable: an unsupported provider still
gets correct behaviour, it just pays the schema in `tools` instead of inline.
Two provider-specific guards apply on the inline path — a tool whose input
schema has a root `oneOf`/`anyOf`/`allOf` is *not* disclosed (Anthropic rejects
it) but logged as a warning, and the disclosure is inserted only after a
tool-result batch or a human turn so message ordering stays valid.

Because disclosure is anchored to a position in the transcript, it lives only
as long as that read stays in context: compaction can evict the `SKILL.md` and
the tool goes dark again. That is the mechanical reason the turn-budget prompt
(§4) forbids re-reading a pinned skill.

### 1.5 Tool inventory

| Bucket | Tools | Mode |
|---|---|---|
| deepagents built-ins | `read_file` `write_file` `edit_file` `ls` `glob` `grep` `delete` `task` | always visible |
| cross-workflow | `list_pending_drafts` `get_current_date` `search_internal_wiki` | always visible |
| `outbound-followup` | 12 (CRM, engagement, drafting) | deferred |
| `account-intelligence` | 15 (usage, hiring, funding, market) | deferred |
| `customer-health` | 12 (support, renewals, adoption) | deferred |
| `meeting-prep` | 8 (relationship context) | deferred |
| `competitor-analysis` | 0 — methodology only | n/a |
| **Total GTM** | **50** (47 deferred + 3 always) | |

`len(ALL_TOOLS) == 50` is asserted at import (`gtm_agent/tools.py`) and again
in `main.py`, so the inventory cannot drift silently.

### 1.6 Measured effect

From the smoke test's per-call snapshot (58 bound = 50 GTM + 8 built-ins):

| Model call | Bound | Still deferred | Model can see |
|---|---|---|---|
| 1 — before any skill read | 58 | 47 | 11 (8 built-ins + 3 cross-workflow) |
| 2 — after reading `outbound-followup/SKILL.md` | 58 | 35 | 23 (11 + the 12 outbound tools) |

The count of *bound* tools never changes — only the number of schemas actually
shipped. The 12 tools that appeared are exactly the ones the read skill lists;
the 35 owned by account-intelligence, customer-health and meeting-prep stayed
deferred, which the test asserts explicitly (`isdisjoint` for the newly
disclosed set, `issubset` for the unrelated set).

### 1.7 Boundary conditions

Worth stating plainly, because the marketing version of this feature
overclaims:

- **Disclosure is a visibility control, not an authorization control.** The
tool is still registered on the agent, so a model that hallucinated the exact
name and arguments *could* invoke it — the smoke test's scripted model does
exactly that and the call succeeds. What deferral really buys is that a
well-behaved provider cannot emit a well-formed call for a schema it was never
shown. For a hard capability boundary, register the tool with
`SkillsMiddleware(tools=...)` instead of `create_deep_agent(tools=...)`: that
is the *gated* path, where calls to an undisclosed tool fail as unknown.
- **Name matching is exact.** A typo in `include_tools` does not raise; it logs
  at debug and the tool silently never appears. `include_tools` must also be a
  space-separated *string*: a YAML list is warned about and stringified to
  `"['a', 'b']"`, whose names never match a tool — the skill then discloses
  nothing.
- **The GTM-specific floor is 11 tools.** Built-ins plus the cross-workflow
  three are visible on every call. That floor, not the 50-tool catalog, is what
  a follow-up run actually costs.

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
validates the sticky-model choice. `RoutingTelemetry` covers routing and cost:

- routing decisions (model, tier, reason, stickiness, classifier used,
  classification latency, escalation events)
- per-call latency, input/output tokens, cache hits
- **dollar cost** per call and per run

…into a printed summary plus an exported JSON ledger (`out/usage/ledger-*.json`).
The runner footer adds the selection-stack evidence that is not a model call:
intent (intent, skill, action, confidence, classifier, latency), tool selection
(selected tools, probabilities, events), the PTC allowlist names, and HITL
approve/reject decisions.

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
telemetry ledger (§3) available to verify the budget is actually met. It is
call-shape policy, not intent routing: by the time these rules apply, layer 1
(skill binding) and layer 3 (tool selection) have already narrowed the tool
surface.

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
- action tools are outside the PTC bridge; `--approve` pauses them through
  `HumanInTheLoopMiddleware`, and the runner records the approve/reject
  decision in the footer
- PTC itself is read-only, so a model-authored program cannot bypass the action
  boundary

## 6. Offline-first, evidence-first engineering

**Innovation.** Every system of record (Salesforce, Gong, LinkedIn, Exa,
BigQuery, Zendesk, Slack) is mocked with in-memory fixtures, so:

- the full demo, including routing, intent classification, tool selection, and
  the disclosure smoke test, runs with **zero credentials** (fake chat model
  for tests; deterministic fallbacks for routing, intent, and tool selection)
- the same code path is exercised live when keys are present — mock/live is
  a runner flag, not a fork

---

## 7. Defined tool invocation — no agent loop required

**Problem.** Even with skills-bound tool discovery, a workflow that needs a
known sequence (do-not-send gate → research → draft) still runs as a
model-driven chain of tool calls — and with dozens of tools the model must also
pick “which tool next?”.

**Innovation.** The agent stops being the orchestrator:

- `gtm_agent/direct.py` exposes deterministic runners that invoke the mocked
  tools directly in the defined order and enforce the same gates (do-not-send,
  no-recent-touch, draft queueing for rep approval). The CLI exposes them as
  `python main.py direct lead|intel|prep …`.
- The deep agents also ship two further shapes: the **read-only code
  interpreter** (`CodeInterpreterMiddleware` + skill-derived PTC allowlist, §9)
  and the **conversation-aware Decisions tool selector**
  (`ConversationToolSelectionMiddleware`, §10).

**Evidence.** `tests/smoke_test_defined_invocation.py` (gates + intel + prep run
with zero model turns) and the `defined invocation` scenario in
`python main.py test` both pass offline.

## 8. Intent routing with OpenAI Decisions

**Problem.** Skills describe when they apply, but nothing forces a request to a
skill before the agent starts. A bad skill guess costs a whole run.

**Innovation.** `IntentRouter` runs before the agent is built:

- one OpenAI Decisions request with a `choice` over the runnable workflows and a
  `needs_clarification` predicate;
- confidence floor `0.65` and clarification threshold `0.6`;
- high confidence → pin the skill;
- low confidence or clarification-positive → ask for a target instead of
  guessing;
- API failure → deterministic keyword fallback;
- `auto` adds a demo-fixture target resolver, so `"Follow up with lead-1042"` or
  `"Prep me for Priya Sharma"` can run end to end.

This is the hard intent layer the earlier version lacked. It is intentionally
pre-flight, not middleware: the chosen skill determines the interpreter's PTC
allowlist, and a clarification should stop before any model or tool call.

**Evidence.** `tests/test_intent.py` covers high confidence, low confidence,
API failure fallback, and target resolution. `python main.py intent "..."` and
`python main.py auto "..."` exercise the same layer offline.

## 9. Code interpreter: read-only PTC plus per-tool HITL

**Problem.** Serial model-mediated tool calls: every tool call costs a full model
round trip. Letting the interpreter call action tools makes the workflow cheaper
but bypasses per-tool approval.

**Innovation.** The interpreter is now a **read-only** execution surface:

- the allowlist is derived from the selected skill's `include_tools`, filtered
  by `side_effect` metadata (`gtm_agent/ptc.py`);
- `draft_slack_followup` and `queue_followup_sequence` are never bridged;
- the `eval` program gates, batches with `Promise.all`, and returns
  `{draft, rationale}`;
- the draft is then queued through the normal `draft_slack_followup` tool, where
  `HumanInTheLoopMiddleware` can pause it per tool.

The outbound trace is `eval → draft_slack_followup`, not a single eval that also
performs the side effect. That is the price of a real approval boundary. The
prep failure is gone because its allowlist now includes `get_person_dossier` and
`get_last_interaction`.

**Evidence.** `tests/test_interpreter_flow.py` asserts the drafted lead emits
`["eval", "draft_slack_followup"]`, the blocked lead emits `["eval"]`, and the
prep evaluator can call its tools. `tests/test_hitl.py` proves rejection prevents
execution.

## 10. Conversation-aware tool selection with OpenAI Decisions

**Problem.** The previous selector used the agent's main model and read only the
last human message. A follow-up like “do that for the other account” had almost
no context to select from.

**Innovation.** `ConversationToolSelectionMiddleware` subclasses
`decision_harness.ToolSelectionMiddleware`:

- renders recent user messages, assistant tool calls, and tool results;
- asks OpenAI Decisions which candidate tools are relevant;
- selects once per run, not on every model turn;
- always keeps deepagents built-ins;
- on timeout, refusal, circuit-open, malformed output, or “no tool clears the
  threshold”, keeps the full tool set.

`decision_harness.ModelSelectionMiddleware` remains unchanged for model routing;
the tool selector reuses its `ToolSelector` and `DecisionsClient` rather than
forking the harness.

**Evidence.** `tests/test_selection.py` proves prior tool results reach the
selector and failures keep all tools. `python main.py test` asserts the selected
tool is bound and an unrelated tool is not.

## 11. Scope check: a strong reference, not “all best practices”

The repo is a coherent reference implementation of the main techniques for
keeping a many-tool agent focused:

- resolve **intent to a skill** first, using a decision model rather than a
  giant dispatch prompt;
- disclose only that skill's **tool schemas**, rather than shipping the catalog;
- **retrieve a smaller tool subset** from the conversation before the loop;
- collapse read-only invocation into a **code program** to remove serial round
  trips;
- put **action tools behind per-tool HITL**;
- fall back to **deterministic invocation** when the workflow is stable.

It is not a claim that every best practice is implemented, nor that the
trade-offs are free. The remaining gaps:

| Gap | Consequence |
|---|---|
| Tool selection runs once per run | a mid-run intent change does not re-select tools |
| `auto` target resolution uses demo fixtures only | it is not production NLU |
| PTC cannot pause inside a running `eval` | action tools are kept outside PTC instead |
| Disclosure is not authorization | a hallucinated exact tool name can still be invoked; use `SkillsMiddleware(tools=...)` for a hard gate |
| The context floor is not zero | built-ins + cross-workflow tools are visible on every call |
| No end-to-end quality benchmark | selection is measured by visibility/call shape, not answer accuracy |

The architectural point is the **separation of concerns**: intent lives in the
intent router, exposure in `include_tools`, relevance in the conversation-aware
selector, invocation in PTC/direct, actions in HITL, and model choice in routing.
That is the reusable lesson; “all best practices” is not.

## Honest-evidence note

The routing *mechanism* is measured (§3), but the catalog **contents** are
configured defaults: all prices in `catalog_overlay.json` are explicit
placeholders, and the `intelligence` scores are hand-assigned relative
rankings, not benchmark results. Validate tier→model assignments against
provider pricing pages and a small eval set before production. The selection
stack is likewise a design pattern, not a quality benchmark: §1.6, §9, and §10
measure visibility and call shape, not end-to-end answer quality against a
baseline.

## File map

| Concern | Where |
|---|---|
| Skills-bound agent, turn-budget prompt | `gtm_agent/agent.py`, `skills/*/SKILL.md` |
| Intent → skill → tool selection semantics (§1.0, §8) | `gtm_agent/intent.py`, `skills/*/SKILL.md` |
| Conversation-aware tool selection (§10) | `gtm_agent/selection.py` |
| Read-only PTC allowlist + HITL (§9) | `gtm_agent/ptc.py`, `gtm_agent/hitl.py` |
| 50-tool mock stack, `DEFERRED` marker | `gtm_agent/tools.py` |
| Per-skill tool bindings (§1) | `skills/*/SKILL.md` (`metadata.include_tools`) |
| Disclosure mechanism (§1.2–1.4) | `deepagents/middleware/skills.py`, `middleware/_skill_tools.py` (installed, 0.7.23) |
| Decision backend + selector factory | `gtm_agent/decisions.py` |
| Skill frontmatter parser | `gtm_agent/skills.py` |
| Routing middleware, telemetry, ledger | `gtm_agent/routing.py` |
| Catalog overlay (data, not code) | `gtm_agent/catalog_overlay.json` |
| Disclosure smoke test (offline) | `tests/smoke_test_skill_tools.py` |
| Offline LLM simulator (scripted runs, GTM simulator) | `gtm_agent/simulator.py`, `main.py` |
| Defined tool invocation workflows (no agent loop) | `gtm_agent/direct.py`, `main.py` (`python main.py direct …`) |
| Interpreter / PTC agent builder | `gtm_agent/agent.py` (`build_interpreter_agent`) |
| Legacy LLM tool-selector builder (not on the CLI path) | `gtm_agent/agent.py` (`build_tool_selector_agent`) |
| Interpreter / defined / selector / intent / HITL scenarios | `main.py` (`GTMAgentTestRunner`), `tests/test_*.py` |
| CLI: live/routed runs, `intent`, `auto`, `--approve` | `main.py` |
