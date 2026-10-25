# OpenAI Decisions for intent, tool selection, PTC allowlists, and HITL

**Date:** 2026-10-25  
**Status:** Draft for review  
**Scope:** Local-first example app, OpenAI Decisions as the only live decision backend

## 1. Context

The GTM agent already demonstrates skills-bound tool disclosure, optional code
interpreter/PTC, defined invocation, and OpenAI Decisions-backed model routing
via `decision_harness`. Its own documentation names five remaining gaps:

1. no hard intent classifier;
2. the tool selector uses the agent's main model, not a cheap decision model;
3. tool selection reads only the last human message, not the conversation;
4. the PTC allowlist is hand-maintained and already misses prep tools;
5. PTC calls bypass per-tool HITL.

This design closes those gaps locally, without forking `decision_harness`.
OpenAI Decisions is the first live backend. A Jev adapter remains possible
behind the same seam later.

## 2. Goals

- Add a confidence-gated intent classifier that maps a request to a skill or
  to a clarification/general outcome.
- Reuse `decision_harness.ModelSelectionMiddleware` unchanged.
- Reuse `decision_harness.ToolSelectionMiddleware` through a local subclass
  that reads recent conversation history.
- Generate the interpreter's PTC allowlist from skill metadata and tool
  side-effect metadata.
- Enforce per-tool HITL at the action-tool boundary; PTC bridges read-only
  tools only.
- Keep the app fully runnable offline with a deterministic fallback and fake
  decisions.
- Make the new behavior observable: intent, confidence, selected tools,
  selection latency, and fallback events appear in the run footer.
- Add an `auto` command that uses the classifier to select a workflow for a
  natural-language request, with deterministic target resolution over the
  demo fixtures.

## 3. Non-goals

- A Jev backend implementation. The interface is kept open, but Jev is a
  follow-up.
- True per-tool pauses inside a running `eval` program. The supported safety
  model is “read-only PTC + HITL on normal action tools”.
- General-purpose entity extraction. The `auto` command resolves targets only
  from the known demo fixtures; it is not a production NLU layer.
- Adding new business workflows for `customer-health` or
  `competitor-analysis`. Their skills remain available; they are not intent
  targets until they have runners.
- Live-API tests. Tests use fakes/offline mode; live smoke tests are manual.

## 4. Key decisions

### 4.1 OpenAI Decisions first

Use the OpenAI Decisions API via the existing `decision_harness` client:

- `decision_harness.DecisionsClient` for transport, retries, circuit breaking,
  ZDR, and refusal handling.
- `decision_harness.ModelSelectionMiddleware` for model routing.
- `decision_harness.ToolSelector` and `ToolSelectionMiddleware` for tool
  selection.

`decision_harness` is currently imported by `routing.py` but missing from
`pyproject.toml`; add it as a dependency.

### 4.2 Local extensions, not a fork

New generic behavior lives in this repo first:

- `IntentRouter` (pre-flight service);
- `ConversationToolSelectionMiddleware` (subclass of
  `decision_harness.ToolSelectionMiddleware`);
- automatic PTC allowlist helpers;
- HITL/review plumbing.

If these prove out, the generic `IntentRouter` / conversation-aware selection
can be upstreamed into `decision_harness` later.

### 4.3 Intent classification is pre-flight, not middleware

Intent classification runs in `GTMAgentRunner` before the agent is built, not
inside the graph. Reasons:

- the selected skill determines the interpreter's PTC allowlist at build time;
- a low-confidence “clarify” outcome should stop before any model or tool call;
- explicit subcommands already know their skill, so they skip classification.

The classifier still uses OpenAI Decisions and is exposed through
`python main.py intent "..."` for inspection.

### 4.4 PTC is read-only; actions use normal HITL

`CodeInterpreterMiddleware` cannot pause per `tools.*` call inside one `eval`.
The safe boundary is:

- bridge only tools marked `side_effect=False`;
- keep `draft_slack_followup` and `queue_followup_sequence` outside PTC;
- run those through the normal tool path with `interrupt_on` when HITL is
  enabled.

The interpreter workflow returns a draft proposal from `eval`; queueing the
draft is a subsequent normal tool call.

## 5. Architecture

```text
CLI / runner
  │
  ├─ RunPlan
  │   ├─ explicit workflow or intent classification
  │   ├─ selected skill + PTC allowlist
  │   └─ HITL config
  │
  ├─ IntentRouter (pre-flight, OpenAI Decisions or offline)
  │
  └─ agent builder
      ├─ ModelSelectionMiddleware          reuse unchanged
      ├─ ConversationToolSelectionMiddleware
      │   └─ decision_harness.ToolSelector reuse
      ├─ CodeInterpreterMiddleware(ptc=...)
      └─ HumanInTheLoopMiddleware via interrupt_on
```

### 5.1 Middleware order

For the normal loop:

1. `CallTelemetryMiddleware` (outermost)
2. `ModelSelectionMiddleware` (existing `--route`)
3. `ConversationToolSelectionMiddleware`
4. deepagents `SkillsMiddleware`
5. `HumanInTheLoopMiddleware` (installed by `create_deep_agent` when
   `interrupt_on` is present)

For interpreter mode, insert `CodeInterpreterMiddleware` alongside the tool
selector. Intent remains pre-flight.

## 6. Components

### 6.1 `gtm_agent/intent.py`

```python
@dataclass(frozen=True)
class IntentDecision:
    intent: str
    skill: str | None
    action: Literal["route", "clarify", "general"]
    confidence: float
    probabilities: dict[str, float]
    clarify_probability: float
    classifier: str
    latency_ms: int
    events: list[str]
```

`IntentRouter`:

- `classify(prompt, conversation="") -> IntentDecision`
- default backend: OpenAI Decisions when `OPENAI_API_KEY` /
  `DECISIONS_API_KEY` is present, else offline.
- one request with:
  - `choice` `intent`: `outbound_followup`, `account_intelligence`,
    `meeting_prep`, `inbox_review`, `general`;
  - `predicate` `needs_clarification`.
- target resolution is deterministic and separate from classification:
  - outbound → a known `lead-*` id in the prompt;
  - account intelligence → a known fixture account name;
  - meeting prep → a known fixture contact name;
  - inbox → no target;
  - general → no target.
- mapping:

| Intent | Skill | Action |
|---|---|---|
| `outbound_followup` | `outbound-followup` | route |
| `account_intelligence` | `account-intelligence` | route |
| `meeting_prep` | `meeting-prep` | route |
| `inbox_review` | none | route to inbox handler |
| `general` | none | general |

- policy:
  - `needs_clarification >= 0.6` or `confidence < 0.65` → `clarify`
  - `confidence >= 0.65` → `route`
  - `general` only when the classifier selects it with sufficient confidence;
    otherwise `clarify`.
- on `SelectorUnavailable` → offline heuristic; event says so.
- the full probabilities and classifier source are kept for telemetry.

`HeuristicIntentRouter` uses keyword/description overlap and returns a lower
confidence. It is deterministic and used offline.

### 6.2 `gtm_agent/selection.py`

`render_conversation(messages, max_chars=12000)` renders the last N messages
with roles, assistant tool calls, and tool results. It preserves the most
recent content and truncates older content first.

`ConversationToolSelectionMiddleware` subclasses
`decision_harness.ToolSelectionMiddleware`:

- overrides `_select()` to pass the rendered conversation to
  `ToolSelector.select()`;
- candidate tools are the selected skill's `include_tools` when a skill is
  known, otherwise all GTM tools;
- `always_include` contains all deepagents built-ins (including `read_file`);
- `max_tools=6` (outbound needs gate + CRM + research + draft);
- `threshold=0.5`, `on_none="all"`;
- `per_call=False`: select once per runner invocation;
- records `last` selection for telemetry.

If the app later supports persistent multi-turn threads, add a turn key to
invalidate the cached selection when the latest human message changes.

### 6.3 `gtm_agent/ptc.py`

- `ptc_allowlist(skill: str | None) -> list[str]`
- parse the selected skill's `metadata.include_tools`;
- keep tools whose `side_effect` extra is false;
- include cross-workflow read-only tools only when the skill itself lists them;
- return `[]` for `skill=None`, disabling PTC bridging for general mode.

Tool metadata:

- add `side_effect: bool` to the tool extras;
- `draft_slack_followup` and `queue_followup_sequence` are side effects;
- all other mock tools default to read-only;
- `defer_loading` remains unchanged.

The interpreter builder receives the derived allowlist. No hand-maintained
`DEFAULT_INTERPRETER_PTC` remains in the live path; keep it only as a legacy
export if needed for compatibility.

### 6.4 HITL

`ACTION_TOOLS = {"draft_slack_followup", "queue_followup_sequence"}`.

When `--approve` is enabled:

- pass `interrupt_on={name: True for name in ACTION_TOOLS}` to
  `create_deep_agent`;
- the runner handles `__interrupt__` by showing the requested actions and
  collecting `approve` / `reject` decisions;
- tests use an auto-approver or auto-rejecter callback.

PTC programs cannot call action tools because they are absent from the bridge.
The interpreter system prompt tells the model to return `{draft, rationale}`
from `eval`, then queue the draft through the normal tool path.

### 6.5 Builder changes

- `build_agent`, `build_interpreter_agent`, `build_tool_selector_agent`, and
  `build_routed_agent` accept `extra_middleware` and `interrupt_on` where
  applicable.
- `build_routed_agent` continues to place `ModelSelectionMiddleware` first.
- `GTMAgentRunner` builds a `RunPlan` before invoking:
  - explicit subcommand → known skill;
  - `intent` command → classifier only;
  - `--selector` → attach `ConversationToolSelectionMiddleware`;
  - `--interpreter` → attach `CodeInterpreterMiddleware(ptc=auto_allowlist)`;
  - `--approve` → attach HITL config.

## 7. CLI and configuration

New commands/flags:

```bash
python main.py intent "Prep me for my meeting with Priya Sharma"
python main.py auto "Follow up with lead-1042"
python main.py --decisions lead lead-1042
python main.py --selector lead lead-1042
python main.py --interpreter --selector lead lead-1042
python main.py --approve lead lead-1042
```

Environment:

```bash
OPENAI_API_KEY=...
# or DECISIONS_API_KEY=...
GTM_DECISION_BACKEND=auto        # auto | openai | offline
GTM_DECISION_MODEL=gpt-6-luna
GTM_INTENT_CONFIDENCE_FLOOR=0.65
GTM_TOOL_SELECTION_MAX_TOOLS=6
GTM_TOOL_SELECTION_THRESHOLD=0.5
```

Behavior:

- `auto`: OpenAI Decisions if a key is present, else offline.
- `openai`: require a key; fail with a clear error.
- `offline`: never call the network.
- `--selector` is the new conversation-aware Decisions selector. The old
  `LLMToolSelectorMiddleware` remains available internally for comparison but
  is no longer the default example path.
- `--route` remains independent and reuses `ModelSelectionMiddleware`.

## 8. Error handling

| Failure | Behavior |
|---|---|
| Intent API unreachable / refusal | fall back to offline heuristic; record event |
| Intent confidence low | `clarify`; no agent call |
| Tool selector unavailable | keep all tools; record event |
| No tool clears threshold | `on_none="all"` keeps all tools |
| PTC allowlist empty | no bridge; normal tools still work |
| HITL rejection | tool not executed; error message returned to the model |
| Missing OpenAI key in `openai` mode | fail fast with configuration error |

Fail-open is deliberate for selection (never lose tools because a selector is
down). Fail-closed is deliberate for actions (no approval, no execution).

## 9. Observability

Add to the run footer/report:

- intent, confidence, action, classifier source, latency;
- selected tools and probabilities/events from `ToolSelectionMiddleware.last`;
- PTC allowlist size and names;
- HITL decisions (approved/rejected).

No new ledger is required for v1. The existing `CostLedger` continues to cover
model calls. Decision-call usage may be added later by extending
`decision_harness` to expose raw usage.

## 10. Testing

Offline tests:

1. **Intent routing** — fake Decisions answers route to the right skill;
   low-confidence answers produce `clarify`; API failure falls back to the
   heuristic.
2. **Conversation-aware selection** — fake client asserts the conversation
   contains a previous tool result; selected tools are filtered; selector
   failure keeps all tools.
3. **Automatic PTC allowlist** — outbound contains gate/research tools but not
   `draft_slack_followup`; prep contains `get_person_dossier` and
   `get_last_interaction`; `--interpreter prep` completes without the old
   `TypeError`.
4. **HITL** — `--approve` interrupts before `draft_slack_followup`; reject
   prevents the queue write; PTC cannot call the action tool.
5. **Model middleware reuse** — `--route` still records a sticky model and
   ledger entries.
6. **CLI** — `intent`, `--selector`, `--interpreter`, and `--approve` compose
   without breaking existing workflows.

All tests run without API keys; live checks are manual.

## 11. Migration and compatibility

- Existing explicit workflows remain unchanged when no new flags are used.
- `--selector` behavior changes from LLM-based to Decisions-based. README and
  INNOVATIONS.md must be updated explicitly.
- `DEFAULT_INTERPRETER_PTC` is no longer the live path; keep a compatibility
  alias if external imports depend on it.
- Interpreter outbound moves the draft side effect out of `eval`; update the
  simulator script and the measured “one eval” claim to “one eval + one
  HITL-gated action”.
- Add `decision_harness` to `pyproject.toml`.

## 12. Risks and trade-offs

- **More decision calls.** Intent adds one call on the intent path; tool
  selection adds one per run, not per model turn. This is the intended cost.
- **OpenAI dependency.** OpenAI-first adds a provider key; offline mode keeps
  tests and demos working.
- **PTC scope narrows.** Read-only PTC means drafts happen outside the program.
  That is the price of real per-tool HITL.
- **Classification quality.** Intent labels and tool descriptions need review;
  thresholds are configurable and tests pin behavior.
- **`decision_harness` coupling.** Local subclasses depend on its middleware
  internals (`_filter`, `_select`). Pin a compatible version and keep the
  subclass small.

## 13. Acceptance criteria

- `python main.py test` passes offline, including the new scenarios.
- `python main.py intent "..."` prints a confident intent/skill or a
  clarification request.
- `python main.py auto "..."` routes to the matching workflow when the intent
  and demo-fixture target resolve; otherwise it asks for a target.
- With `OPENAI_API_KEY`, `--selector` binds a conversation-aware subset and
  falls back to all tools on failure.
- `--interpreter prep` no longer fails with `TypeError: not a function`.
- `--approve` interrupts on action tools; rejection prevents the side effect.
- Existing lead/briefing/prep workflows and artifacts still work.
- README, `.env.example`, and INNOVATIONS.md document the new path and remove
  the “missing bits” claims.

## 14. File map

| Concern | Files |
|---|---|
| Intent classification + auto dispatch | `gtm_agent/intent.py`, `main.py` |
| Conversation-aware tool selection | `gtm_agent/selection.py` |
| Automatic PTC allowlist | `gtm_agent/ptc.py`, `gtm_agent/tools.py`, `gtm_agent/agent.py` |
| HITL | `gtm_agent/hitl.py`, `gtm_agent/agent.py`, `main.py` |
| Builders/runner wiring | `gtm_agent/agent.py`, `gtm_agent/routing.py`, `main.py` |
| Tests | `tests/smoke_test_*.py`, `main.py` scenario runner |
| Docs | `README.md`, `.env.example`, `INNOVATIONS.md` |
