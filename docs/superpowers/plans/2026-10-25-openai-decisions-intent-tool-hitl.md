# OpenAI Decisions Intent, Tool Selection, PTC, and HITL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a confidence-gated intent router, conversation-aware OpenAI Decisions tool selection, automatic read-only PTC allowlists, and per-tool HITL to the GTM example app.

**Architecture:** Keep `decision_harness`’s model-routing middleware unchanged. Add local modules for intent classification, conversation-aware tool selection, PTC policy, and HITL, then wire them into the existing runner and builders. OpenAI Decisions is the only live decision backend; offline and fake paths keep tests deterministic.

**Tech Stack:** Python 3.10+, LangChain 1.4, deepagents 0.7.23, decision_harness 0.1.0, OpenAI Decisions (`gpt-6-luna`), optional QuickJS/`langchain-quickjs`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-openai-decisions-intent-tool-hitl-design.md`

## Global Constraints

- Python `>=3.10`; this repo is developed on 3.13.
- Live decision model is `gpt-6-luna`; do not add another live decision backend in this plan.
- `GTM_DECISION_BACKEND` accepts `auto`, `openai`, or `offline`.
- Intent confidence floor is `0.65`; clarification predicate threshold is `0.6`.
- Tool-selection threshold is `0.5`; default `max_tools` is `6`.
- PTC is read-only. `draft_slack_followup` and `queue_followup_sequence` are side-effect tools and must never be bridged into PTC.
- No test may require an API key or make a network request.
- Existing lead, briefing, prep, inbox, direct, routed, and offline workflows must keep working.
- Add `decision_harness>=0.1.0` and `pyyaml>=6.0` to project dependencies.
- Do not implement the Jev adapter in this plan.

## Review Focus

- **Ambiguous request:** a low-confidence or clarification-positive intent must produce a clarification and must not call the agent or any tool.
- **Selector failure:** a timeout, refusal, circuit-open, or malformed response must leave the full tool set available.
- **Prep PTC:** the prep interpreter must be able to call `get_person_dossier` and `get_last_interaction`; it must not reproduce the current `TypeError: not a function`.
- **HITL rejection:** rejecting `draft_slack_followup` must leave `tools._SLACK_DRAFTS` empty; approving it must queue exactly one draft.
- **Conversation context:** tool selection must see prior assistant tool calls and tool results, not only the latest human message.

---

## File Structure

| File | Responsibility |
|---|---|
| `gtm_agent/skills.py` (new) | Parse `skills/*/SKILL.md` frontmatter into `SkillSpec`; expose skill descriptions and `include_tools`. |
| `gtm_agent/ptc.py` (new) | Derive a read-only PTC allowlist from a skill and tool side-effect metadata. |
| `gtm_agent/decisions.py` (new) | Resolve the decision backend, construct `DecisionsClient` and `ToolSelector`, read env thresholds. |
| `gtm_agent/intent.py` (new) | `IntentDecision`, OpenAI-backed `IntentRouter`, offline heuristic, demo target resolver. |
| `gtm_agent/selection.py` (new) | Render conversation context; `ConversationToolSelectionMiddleware` subclass. |
| `gtm_agent/hitl.py` (new) | Action-tool list, interrupt config, reviewers, `run_with_hitl`. |
| `gtm_agent/tools.py` | Add `side_effect` extras, `KNOWN_LEADS` / `KNOWN_ACCOUNTS` / `KNOWN_CONTACTS`. |
| `gtm_agent/agent.py` | Builder params for extra middleware, checkpointer, interrupts, skill-derived PTC; update interpreter prompt. |
| `gtm_agent/routing.py` | Pass `interrupt_on` and `checkpointer` through `build_routed_agent`. |
| `gtm_agent/simulator.py` | Move interpreter draft out of `eval`; script the normal draft tool call. |
| `main.py` | Runner integration, `intent` / `auto` commands, new flags, telemetry. |
| `tests/test_skills.py` (new) | Skill catalog and metadata tests. |
| `tests/test_ptc.py` (new) | Allowlist tests. |
| `tests/test_decisions.py` (new) | Backend selection and selector factory tests. |
| `tests/test_intent.py` (new) | Intent classification and target resolution tests. |
| `tests/test_selection.py` (new) | Conversation rendering and middleware tests. |
| `tests/test_builders.py` (new) | Builder pass-through and interpreter skill wiring. |
| `tests/test_hitl.py` (new) | Reviewer and HITL resume tests. |
| `tests/test_runner.py` (new) | Runner integration with stub selector, interpreter, and approval. |
| `tests/test_interpreter_flow.py` (new) | Interpreter read-only PTC and draft-outside-eval sequence. |
| `tests/test_cli.py` (new) | Offline `intent` and `auto` command behavior. |
| `README.md`, `INNOVATIONS.md`, `.env.example` | Document the new path and remove stale gap claims. |

---

### Task 1: Skill catalog and side-effect metadata

**Files:**
- Create: `gtm_agent/skills.py`
- Modify: `gtm_agent/tools.py`
- Modify: `pyproject.toml`
- Test: `tests/test_skills.py`

**Interfaces:**
- Produces: `SkillSpec(name: str, description: str, include_tools: tuple[str, ...], path: Path)`
- Produces: `load_skills(root: Path | None = None) -> dict[str, SkillSpec]`
- Produces: `skill_tool_names(skill: str | None, skills: dict[str, SkillSpec] | None = None) -> list[str]`
- Modifies: tools carry `extras["side_effect"] is True` only for write actions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skills.py
from gtm_agent import tools
from gtm_agent.skills import load_skills, skill_tool_names


def test_load_skills_reads_frontmatter_and_tools():
    skills = load_skills()
    assert set(skills) == {
        "account-intelligence",
        "competitor-analysis",
        "customer-health",
        "meeting-prep",
        "outbound-followup",
    }
    assert len(skills["outbound-followup"].include_tools) == 12
    assert len(skills["meeting-prep"].include_tools) == 8
    assert skills["competitor-analysis"].include_tools == ()
    assert "follow-up" in skills["outbound-followup"].description.lower()


def test_skill_tool_names_is_ordered_and_empty_for_unknown():
    assert skill_tool_names("outbound-followup")[:3] == [
        "lookup_lead",
        "check_contact_history",
        "get_gong_transcripts",
    ]
    assert skill_tool_names(None) == []
    assert skill_tool_names("does-not-exist") == []


def test_side_effect_metadata_is_explicit_and_deferred_count_unchanged():
    by_name = {tool.name: tool for tool in tools.ALL_TOOLS}
    assert by_name["draft_slack_followup"].extras["side_effect"] is True
    assert by_name["queue_followup_sequence"].extras["side_effect"] is True
    assert by_name["lookup_lead"].extras.get("side_effect") is not True
    deferred = [t for t in tools.ALL_TOOLS if (t.extras or {}).get("defer_loading")]
    assert len(deferred) == 47
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_skills.py -v`
Expected: FAIL because `gtm_agent.skills` does not exist and `side_effect` is absent.

- [ ] **Step 3: Implement the skill catalog and metadata**

Create `gtm_agent/skills.py` with `SkillSpec`, `load_skills`, and `skill_tool_names`. Parse YAML frontmatter with `yaml.safe_load`; accept `include_tools` as either a whitespace-separated string or a list. Return an empty tuple when absent.

In `gtm_agent/tools.py`:

- add `side_effect: bool = False` to `_make_tool`;
- merge `{"side_effect": True}` into extras only when requested;
- mark `draft_slack_followup` and `queue_followup_sequence` as side effects;
- keep `DEFERRED`/`defer_loading` unchanged.

Add `pyyaml>=6.0` to `pyproject.toml` dependencies.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_skills.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gtm_agent/skills.py gtm_agent/tools.py pyproject.toml tests/test_skills.py
git commit -m "feat: add skill catalog and tool side-effect metadata"
```

---

### Task 2: Automatic read-only PTC allowlist

**Files:**
- Create: `gtm_agent/ptc.py`
- Test: `tests/test_ptc.py`

**Interfaces:**
- Consumes: `skill_tool_names()` from Task 1; `tools.ALL_TOOLS` with `side_effect`.
- Produces: `ptc_allowlist(skill: str | None, *, root: Path | None = None) -> list[str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ptc.py
from gtm_agent import tools
from gtm_agent.ptc import ptc_allowlist


def test_outbound_allowlist_is_read_only():
    allow = ptc_allowlist("outbound-followup")
    assert "check_contact_history" in allow
    assert "lookup_lead" in allow
    assert "draft_slack_followup" not in allow
    assert "queue_followup_sequence" not in allow


def test_prep_allowlist_contains_the_tools_the_script_uses():
    assert {"get_person_dossier", "get_last_interaction"} <= set(
        ptc_allowlist("meeting-prep")
    )


def test_no_skill_means_no_ptc_bridge():
    assert ptc_allowlist(None) == []


def test_every_allowlisted_name_is_a_real_tool():
    names = {tool.name for tool in tools.ALL_TOOLS}
    for skill in ("outbound-followup", "account-intelligence", "meeting-prep"):
        assert set(ptc_allowlist(skill)) <= names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ptc.py -v`
Expected: FAIL because `gtm_agent.ptc` does not exist.

- [ ] **Step 3: Implement `ptc_allowlist`**

Build `TOOL_BY_NAME` from `tools.ALL_TOOLS`. For a skill name, read `skill_tool_names(skill)`, keep each tool whose extras do not set `side_effect` to `True`, and preserve skill order. Return `[]` for `None` or unknown skills.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ptc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gtm_agent/ptc.py tests/test_ptc.py
git commit -m "feat: derive read-only PTC allowlists from skills"
```

---

### Task 3: Decision backend helpers

**Files:**
- Create: `gtm_agent/decisions.py`
- Modify: `pyproject.toml`
- Test: `tests/test_decisions.py`

**Interfaces:**
- Produces: `resolve_backend(backend: str | None = None) -> Literal["openai", "offline"]`
- Produces: `make_decisions_client() -> DecisionsClient`
- Produces: `make_tool_selector(backend: str | None = None, *, client: Any | None = None) -> ToolSelector`
- Produces: `decision_env(name: str, default: str) -> str` for threshold/model env reads.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_decisions.py
import pytest
from gtm_agent.decisions import make_tool_selector, resolve_backend


def test_auto_without_key_is_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
    assert resolve_backend("auto") == "offline"


def test_auto_with_key_is_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert resolve_backend("auto") == "openai"
    assert resolve_backend("openai") == "openai"
    assert resolve_backend("offline") == "offline"


def test_explicit_openai_without_key_fails(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        resolve_backend("openai")


def test_offline_selector_keeps_every_tool():
    selector = make_tool_selector("offline")
    selection = selector.select(
        "Follow up with lead-1042",
        [("lookup_lead", "Look up a lead."), ("check_contact_history", "Check history.")],
        max_tools=None,
    )
    assert set(selection.selected) == {"lookup_lead", "check_contact_history"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_decisions.py -v`
Expected: FAIL because `gtm_agent.decisions` does not exist.

- [ ] **Step 3: Implement the helpers**

Use `decision_harness.DecisionsClient` and `decision_harness.ToolSelector`. `resolve_backend("auto")` chooses OpenAI when either key is non-empty, else offline. Explicit `openai` without a key raises `RuntimeError` naming `OPENAI_API_KEY` and `DECISIONS_API_KEY`.

`make_tool_selector` returns `ToolSelector(offline=True)` for offline, otherwise `ToolSelector(client=client or DecisionsClient(model=...))` with `threshold=float(decision_env("GTM_TOOL_SELECTION_THRESHOLD", "0.5"))`.

Add `decision_harness>=0.1.0` to `pyproject.toml`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_decisions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gtm_agent/decisions.py pyproject.toml tests/test_decisions.py
git commit -m "feat: add OpenAI Decisions backend helpers"
```

---

### Task 4: Intent router, offline fallback, and demo target resolver

**Files:**
- Create: `gtm_agent/intent.py`
- Modify: `gtm_agent/tools.py` (public `KNOWN_*` tuples)
- Test: `tests/test_intent.py`

**Interfaces:**
- Consumes: `load_skills`, `skill_tool_names`; `DecisionsClient.answers`.
- Produces: `IntentDecision(intent, skill, action, confidence, probabilities, clarify_probability, classifier, latency_ms, events)`
- Produces: `IntentRouter(backend=..., client=..., confidence_floor=..., clarify_floor=...).classify(prompt, conversation="") -> IntentDecision`
- Produces: `HeuristicIntentRouter().classify(prompt, conversation="") -> IntentDecision`
- Produces: `resolve_target(intent: str, prompt: str) -> str | None`
- Modifies: `tools.KNOWN_LEADS`, `tools.KNOWN_ACCOUNTS`, `tools.KNOWN_CONTACTS` are public tuples for the demo resolver.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_intent.py
from gtm_agent.intent import HeuristicIntentRouter, IntentRouter, resolve_target


class FakeClient:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def post(self, payload):
        self.calls.append(payload)
        return self.data


def _answers(intent="outbound_followup", confidence=0.91, clarify=0.05):
    return {
        "answers": [
            {
                "name": "intent",
                "type": "choice",
                "choice": intent,
                "confidence": confidence,
                "probabilities": [
                    {"value": intent, "probability": confidence},
                    {"value": "general", "probability": 1 - confidence},
                ],
            },
            {"name": "needs_clarification", "type": "predicate", "probability": clarify},
        ]
    }


def test_high_confidence_intent_routes_and_skill_is_mapped():
    router = IntentRouter(backend="openai", client=FakeClient(_answers()))
    decision = router.classify("Follow up with lead-1042")
    assert decision.action == "route"
    assert decision.intent == "outbound_followup"
    assert decision.skill == "outbound-followup"
    assert decision.confidence == 0.91


def test_low_confidence_or_clarification_asks():
    router = IntentRouter(
        backend="openai", client=FakeClient(_answers(confidence=0.4, clarify=0.9))
    )
    decision = router.classify("do the thing")
    assert decision.action == "clarify"
    assert decision.clarify_probability == 0.9


def test_api_failure_falls_back_to_heuristic():
    class Broken:
        def post(self, payload):
            raise RuntimeError("network down")

    decision = IntentRouter(backend="openai", client=Broken()).classify(
        "Prep me for my meeting with Priya Sharma"
    )
    assert decision.classifier == "heuristic"
    assert decision.skill == "meeting-prep"
    assert decision.action == "route"


def test_target_resolution_uses_demo_fixtures():
    assert resolve_target("outbound_followup", "Follow up with lead-1042") == "lead-1042"
    assert resolve_target("account_intelligence", "briefing for Helix Health") == "Helix Health"
    assert resolve_target("meeting_prep", "prep me for Priya Sharma") == "Priya Sharma"
    assert resolve_target("meeting_prep", "prep me for nobody") is None


def test_heuristic_offline_routes_strong_signal():
    decision = HeuristicIntentRouter().classify("Prepare a meeting prep for Priya Sharma")
    assert decision.action == "route"
    assert decision.skill == "meeting-prep"
    assert decision.classifier == "heuristic"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_intent.py -v`
Expected: FAIL because `gtm_agent.intent` does not exist.

- [ ] **Step 3: Implement the intent layer**

Create `gtm_agent/intent.py`:

- build a `choice` question over `outbound_followup`, `account_intelligence`, `meeting_prep`, `inbox_review`, `general`;
- add a `needs_clarification` predicate;
- parse answers with `DecisionsClient.answers(data, ["intent", "needs_clarification"])`;
- use the confidence floor from `GTM_INTENT_CONFIDENCE_FLOOR` (default `0.65`) and clarify threshold `0.6`;
- on any exception, call `HeuristicIntentRouter` and append a `decisions unavailable` event;
- use `classifier="decisions:gpt-6-luna"` or `"heuristic"`.

Add public fixture tuples to `tools.py`:

```python
KNOWN_LEADS = tuple(_LEADS)
KNOWN_ACCOUNTS = tuple(dict.fromkeys([
    *_USAGE_SIGNALS, *_HIRING, *_FUNDING,
    *(lead["company"] for lead in _LEADS.values()),
]))
KNOWN_CONTACTS = tuple(dict.fromkeys([
    *_DOSSIERS, *_LAST_INTERACTIONS,
    *(lead["contact"] for lead in _LEADS.values()),
]))
```

`resolve_target` matches known names case-insensitively, longest name first. For outbound, use `re.search(r"lead-\\d+", prompt, re.I)` and verify membership in `KNOWN_LEADS`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_intent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gtm_agent/intent.py gtm_agent/tools.py tests/test_intent.py
git commit -m "feat: add intent router and demo target resolution"
```

---

### Task 5: Conversation-aware tool selection middleware

**Files:**
- Create: `gtm_agent/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `ToolSelectionMiddleware` and `SelectorUnavailable` from `decision_harness`.
- Produces: `render_conversation(messages: Sequence[BaseMessage], max_chars: int = 12000) -> str`
- Produces: `ConversationToolSelectionMiddleware(selector=..., candidate_names=..., always_include=..., max_tools=6)`
- Reuses: `ToolSelectionMiddleware._filter()`; `middleware.last` remains the telemetry object.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_selection.py
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from decision_harness import SelectorUnavailable, ToolSelection
from gtm_agent import tools
from gtm_agent.selection import ConversationToolSelectionMiddleware, render_conversation


class StubSelector:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def select(self, task, tool_pairs, **kwargs):
        self.calls.append((task, tool_pairs, kwargs))
        if self.error:
            raise self.error
        return self.result


def _request(messages):
    return SimpleNamespace(messages=messages, tools=tools.ALL_TOOLS)


def test_render_conversation_includes_prior_tool_result():
    messages = [
        HumanMessage("Follow up with lead-1042"),
        AIMessage(
            "",
            tool_calls=[
                {"name": "check_contact_history", "args": {"lead_id": "lead-1042"}, "id": "c1", "type": "tool_call"}
            ],
        ),
        ToolMessage("No prior touch", name="check_contact_history", tool_call_id="c1"),
    ]
    text = render_conversation(messages)
    assert "USER: Follow up with lead-1042" in text
    assert "check_contact_history" in text
    assert "No prior touch" in text


def test_middleware_passes_conversation_and_filters_candidates():
    selector = StubSelector(
        ToolSelection(selected=["check_contact_history"], probabilities={"check_contact_history": 0.9})
    )
    middleware = ConversationToolSelectionMiddleware(
        selector=selector,
        candidate_names={"check_contact_history", "lookup_lead"},
        always_include=["read_file"],
        max_tools=2,
    )
    messages = [
        HumanMessage("Follow up with lead-1042"),
        ToolMessage("lead history", name="check_contact_history", tool_call_id="c1"),
    ]
    selected = middleware._select(_request(messages))
    assert selected == ["check_contact_history"]
    task, pairs, kwargs = selector.calls[0]
    assert "lead history" in task
    assert {name for name, _ in pairs} == {"check_contact_history", "lookup_lead"}
    assert kwargs["max_tools"] == 2
    assert kwargs["on_none"] == "all"


def test_selector_failure_keeps_all_tools():
    middleware = ConversationToolSelectionMiddleware(
        selector=StubSelector(error=SelectorUnavailable("down")),
        candidate_names={"lookup_lead"},
    )
    assert middleware._select(_request([HumanMessage("Follow up")])) is None


def test_no_tool_clears_threshold_keeps_all_tools():
    middleware = ConversationToolSelectionMiddleware(
        selector=StubSelector(
            ToolSelection(selected=["lookup_lead"], probabilities={"lookup_lead": 0.1},
                          events=["no tool cleared threshold 0.50 -> keeping all (1) tools"])
        ),
        candidate_names={"lookup_lead"},
    )
    assert middleware._select(_request([HumanMessage("Follow up")])) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_selection.py -v`
Expected: FAIL because `gtm_agent.selection` does not exist.

- [ ] **Step 3: Implement the middleware**

Create `gtm_agent/selection.py`:

- `render_conversation` renders `HumanMessage`, `AIMessage` (including tool call names/args), and `ToolMessage` (name/content). Truncate individual blocks and keep the most recent `max_chars`.
- Subclass `ToolSelectionMiddleware`; set `per_call=False`, `on_none="all"`.
- Override `_select(request)`:
  - build candidates from `request.tools` filtered by `candidate_names` (or all GTM tools if `None`);
  - return `None` if no candidates or no conversation text;
  - call `self.selector.select(conversation, pairs, max_tools=self.max_tools, always_include=self.always_include, on_none=self.on_none)`;
  - on `SelectorUnavailable`, set `self.last` with an event and return `None`;
  - if `selection.events` contains `no tool cleared threshold`, set `self.last` and return `None` so the unfiltered tool set survives.
- Keep `always_include` for `_filter`; do not include built-ins in the selector's candidate pairs.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gtm_agent/selection.py tests/test_selection.py
git commit -m "feat: add conversation-aware Decisions tool selection"
```

---

### Task 6: Builder support for middleware, HITL, checkpointer, and skill PTC

**Files:**
- Modify: `gtm_agent/agent.py`
- Modify: `gtm_agent/routing.py`
- Test: `tests/test_builders.py`

**Interfaces:**
- Produces: `build_agent(..., extra_middleware: list[Any] | None = None, interrupt_on: dict[str, Any] | None = None, checkpointer: Any | None = None)`
- Produces: `build_interpreter_agent(..., skill: str | None = None, ptc: list[str] | None = None, extra_middleware: ..., interrupt_on=..., checkpointer=...)`
- Produces: `build_routed_agent(..., interrupt_on=..., checkpointer=...)`
- `build_interpreter_agent` uses `ptc_allowlist(skill)` when `ptc` is omitted and `skill` is set.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_builders.py
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from gtm_agent.agent import build_interpreter_agent


class EvalModel(GenericFakeChatModel):
    def bind_tools(self, bound_tools, **kwargs):
        return self


def test_interpreter_builder_uses_skill_allowlist():
    script = (
        'const r = await tools.getPersonDossier({ query: "Priya Sharma" }); r;'
    )
    model = EvalModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "eval", "args": {"code": script}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="done"),
    ]))
    agent = build_interpreter_agent(model=model, skill="meeting-prep")
    result = agent.invoke(
        {"messages": [HumanMessage("prep")], "pinned_skills": ["meeting-prep"]},
        config={"recursion_limit": 20},
    )
    eval_msgs = [m for m in result["messages"] if getattr(m, "name", None) == "eval"]
    assert eval_msgs and "Priya Sharma" in str(eval_msgs[0].content)
    assert "not a function" not in str(eval_msgs[0].content)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_builders.py -v`
Expected: FAIL because `build_interpreter_agent` does not accept `skill`.

- [ ] **Step 3: Implement builder parameters**

In `gtm_agent/agent.py`:

- add `extra_middleware`, `interrupt_on`, `checkpointer` to builders;
- `build_interpreter_agent(skill=...)` resolves `ptc_allowlist(skill)` when `ptc is None`;
- keep `DEFAULT_INTERPRETER_PTC` as a legacy fallback but remove `draft_slack_followup` from it;
- pass `interrupt_on` and `checkpointer` to `create_deep_agent`;
- update `GTM_INTERPRETER_SYSTEM_PROMPT`: PTC is read-only; return `{draft, rationale}`; call `draft_slack_followup` as a normal tool; never call action tools inside `eval`.

In `gtm_agent/routing.py`, add `interrupt_on` and `checkpointer` parameters and pass them to `create_deep_agent`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_builders.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gtm_agent/agent.py gtm_agent/routing.py tests/test_builders.py
git commit -m "feat: wire skill PTC and HITL options into agent builders"
```

---

### Task 7: HITL helpers and resume loop

**Files:**
- Create: `gtm_agent/hitl.py`
- Test: `tests/test_hitl.py`

**Interfaces:**
- Produces: `ACTION_TOOLS = ("draft_slack_followup", "queue_followup_sequence")`
- Produces: `interrupt_config(enabled: bool) -> dict[str, bool] | None`
- Produces: `AutoApprover(approve: bool = True).review(action_requests, review_configs) -> list[dict]`
- Produces: `CliApprover().review(action_requests, review_configs) -> list[dict]`
- Produces: `run_with_hitl(agent, payload, config, reviewer) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hitl.py
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from gtm_agent.hitl import AutoApprover, interrupt_config, run_with_hitl


@tool
def write_thing(x: str) -> str:
    """Write a thing."""
    return f"wrote {x}"


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, bound_tools, **kwargs):
        return self


def make_agent():
    model = FakeModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_thing", "args": {"x": "a"}, "id": "1", "type": "tool_call"}]),
        AIMessage(content="done"),
    ]))
    return create_agent(
        model,
        tools=[write_thing],
        middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_config(True))],
        checkpointer=InMemorySaver(),
    )


def test_approve_executes_once():
    agent = make_agent()
    result = run_with_hitl(
        agent,
        {"messages": [HumanMessage("write a")]},
        {"configurable": {"thread_id": "approve"}},
        AutoApprover(True),
    )
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "wrote a"


def test_reject_prevents_execution():
    agent = make_agent()
    result = run_with_hitl(
        agent,
        {"messages": [HumanMessage("write a")]},
        {"configurable": {"thread_id": "reject"}},
        AutoApprover(False),
    )
    assert not [m for m in result["messages"] if m.type == "tool"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hitl.py -v`
Expected: FAIL because `gtm_agent.hitl` does not exist.

- [ ] **Step 3: Implement the HITL helpers**

Create `gtm_agent/hitl.py`:

- `interrupt_config(True)` returns `{name: True for name in ACTION_TOOLS}`; `False` returns `None`.
- `AutoApprover.review` returns `[{"type": "approve"}]` or `[{"type": "reject", "message": "..."}]` for every action request.
- `CliApprover.review` prints each action name/args/description and asks `approve? [y/N]`; default reject.
- `run_with_hitl` invokes the agent; while `result.get("__interrupt__")` exists, flatten `interrupt.value["action_requests"]`, get decisions from the reviewer, and resume with `Command(resume={"decisions": decisions})` using the same config.
- The agent must have a checkpointer; raise a clear `RuntimeError` if an interrupt occurs without one.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hitl.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gtm_agent/hitl.py tests/test_hitl.py
git commit -m "feat: add HITL reviewers and resume loop"
```

---

### Task 8: Runner integration

**Files:**
- Modify: `main.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Modifies: `GTMAgentRunner.__init__(..., decision_backend="auto", approve=False, reviewer=None, tool_selector=None)`
- Modifies: `_invoke()` builds selector/interpreter/HITL middleware and uses `run_with_hitl`.
- Produces: `last_tool_selection: ToolSelection | None`, `last_ptc_allowlist: list[str]`, `last_intent: IntentDecision | None` for footer telemetry.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runner.py
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from decision_harness import ToolSelection
from gtm_agent import tools
from gtm_agent.hitl import AutoApprover
from gtm_agent.simulator import GTMSimulator, ScriptedModel, _call
from main import GTMAgentRunner


class StubSelector:
    def select(self, task, tool_pairs, **kwargs):
        return ToolSelection(selected=["check_contact_history"], probabilities={"check_contact_history": 0.9})


class EvalModel(GenericFakeChatModel):
    def bind_tools(self, bound_tools, **kwargs):
        return self


def test_selector_runner_binds_only_selected_tool():
    model = ScriptedModel(messages=iter([
        _call("check_contact_history", {"lead_id": "lead-1042"}, "c1"),
        AIMessage(content="done"),
    ]))
    runner = GTMAgentRunner(selector=True, tool_selector=StubSelector(), decision_backend="offline")
    result = runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    assert "check_contact_history" in str(result["messages"])


def test_interpreter_runner_uses_skill_allowlist():
    model = EvalModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "eval",
            "args": {"code": 'const r = await tools.checkContactHistory({lead_id: "lead-1042"}); r;'},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="done"),
    ]))
    runner = GTMAgentRunner(interpreter=True, decision_backend="offline")
    runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    assert "draft_slack_followup" not in runner.last_ptc_allowlist
    assert "check_contact_history" in runner.last_ptc_allowlist


def test_approve_queues_once_and_reject_does_not():
    tools.reset_fixtures()
    model, _, _ = GTMSimulator().outbound_script("lead-1042", pinned=True)
    runner = GTMAgentRunner(
        approve=True, reviewer=AutoApprover(True), decision_backend="offline"
    )
    runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    assert len(tools._SLACK_DRAFTS) == 1

    tools.reset_fixtures()
    model, _, _ = GTMSimulator().outbound_script("lead-1042", pinned=True)
    runner = GTMAgentRunner(
        approve=True, reviewer=AutoApprover(False), decision_backend="offline"
    )
    runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    assert tools._SLACK_DRAFTS == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL because `GTMAgentRunner` lacks the new parameters/attributes and does not use HITL.

- [ ] **Step 3: Integrate the runner**

In `GTMAgentRunner`:

- resolve `self._tool_selector = tool_selector or make_tool_selector(decision_backend)`;
- in `_invoke`, build `extra_middleware`:
  - selector: `ConversationToolSelectionMiddleware(selector=self._tool_selector, candidate_names=set(skill_tool_names(pinned_skill)) or None, always_include=INFRASTRUCTURE_TOOLS, max_tools=6 or None for offline)`;
  - interpreter: `ptc = ptc_allowlist(pinned_skill)`, store it, and add `interpreter_middleware(ptc)` when non-empty;
- pass `interrupt_on=interrupt_config(self.approve)` and `checkpointer=InMemorySaver() if self.approve else None`;
- call `run_with_hitl` when approving, otherwise `agent.invoke`;
- store `last_tool_selection`, `last_ptc_allowlist`, and `last_intent`;
- update `_print_footer` to print intent, selected tools, and the actual PTC allowlist rather than `DEFAULT_INTERPRETER_PTC`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_runner.py
git commit -m "feat: integrate Decisions selection, PTC, and HITL into the runner"
```

---

### Task 9: Interpreter workflow moves drafts outside `eval`

**Files:**
- Modify: `gtm_agent/agent.py`
- Modify: `gtm_agent/simulator.py`
- Modify: `tests/smoke_test_interpreter_ptc.py`
- Modify: `main.py` scenario expectations
- Test: `tests/test_interpreter_flow.py` (new)

**Interfaces:**
- `eval` returns `{outcome, lead_id, draft, rationale, ...}` and does not call `draftSlackFollowup`.
- The scripted interpreter run emits `eval` then `draft_slack_followup` for a drafted lead; a blocked lead emits only `eval`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_interpreter_flow.py
from gtm_agent import tools
from gtm_agent.simulator import GTMSimulator
from main import GTMAgentRunner


def test_interpreter_draft_is_a_normal_tool_call():
    tools.reset_fixtures()
    model, outcome, _ = GTMSimulator().interpreter_outbound_script("lead-1042", pinned=True)
    runner = GTMAgentRunner(interpreter=True, decision_backend="offline")
    result = runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    calls = [c["name"] for m in result["messages"] if m.type == "ai" for c in m.tool_calls]
    assert outcome == "drafted"
    assert calls == ["eval", "draft_slack_followup"]
    assert len(tools._SLACK_DRAFTS) == 1


def test_blocked_lead_has_no_draft_tool_call():
    tools.reset_fixtures()
    model, outcome, _ = GTMSimulator().interpreter_outbound_script("lead-1044", pinned=True)
    runner = GTMAgentRunner(interpreter=True, decision_backend="offline")
    result = runner._invoke(model, "Follow up on lead-1044.", pinned_skill="outbound-followup")
    calls = [c["name"] for m in result["messages"] if m.type == "ai" for c in m.tool_calls]
    assert outcome == "blocked"
    assert calls == ["eval"]
    assert tools._SLACK_DRAFTS == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_interpreter_flow.py -v`
Expected: FAIL because the interpreter script currently calls `draftSlackFollowup` inside `eval` and the expected sequence is `["eval"]`.

- [ ] **Step 3: Move the side effect outside `eval`**

In `gtm_agent/simulator.py`, remove the `await tools.draftSlackFollowup(...)` line from `_INTERPRETER_OUTBOUND_JS`. In `interpreter_outbound_script`, after the `eval` call, append a normal `_call("draft_slack_followup", {...})` only when the fixture outcome is `drafted`, using `GTMSimulator._draft_for()` for exact args.

In `gtm_agent/agent.py`, update the interpreter prompt to say the bridge is read-only and that the draft must be queued through the normal tool.

Update `tests/smoke_test_interpreter_ptc.py` and the `interpreter PTC` scenario in `main.py` to expect `["eval", "draft_slack_followup"]` for drafted leads and `["eval"]` for blocked leads, intel, and prep.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_interpreter_flow.py tests/smoke_test_interpreter_ptc.py -v` and `python main.py test`
Expected: PASS; no `TypeError` in the prep eval.

- [ ] **Step 5: Commit**

```bash
git add gtm_agent/agent.py gtm_agent/simulator.py tests/smoke_test_interpreter_ptc.py tests/test_interpreter_flow.py main.py
git commit -m "feat: keep interpreter PTC read-only and queue drafts through HITL"
```

---

### Task 10: CLI `intent` and `auto` commands plus flags

**Files:**
- Modify: `main.py`
- Test: `tests/test_cli.py` (new)

**Interfaces:**
- Adds subcommands `intent PROMPT...` and `auto PROMPT...`.
- Adds global `--decision-backend {auto,openai,offline}`, `--decisions`, `--approve`, `--yes`.
- `auto` uses `IntentRouter`, prints the decision, resolves the demo target, and dispatches to `process_lead`, `briefing`, `prep`, or `inbox`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import sys
import main


def test_intent_command_offline(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--decision-backend", "offline", "intent", "Prep me for Priya Sharma"])
    assert main.main() == 0
    out = capsys.readouterr().out
    assert "meeting-prep" in out
    assert "heuristic" in out


def test_auto_command_dispatches_offline(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["main.py", "--decision-backend", "offline", "auto", "Follow up with lead-1042"])
    assert main.main() == 0
    out = capsys.readouterr().out
    assert "outbound-followup" in out
    assert "lead-1042" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL because the subcommands and flags do not exist.

- [ ] **Step 3: Implement the CLI**

Add the global flags and subparsers. For `intent`, construct `IntentRouter(backend=args.decision_backend)` and print `intent`, `skill`, `action`, `confidence`, `classifier`, and events; return `0`.

For `auto`, classify, print the decision, then:

- `outbound_followup` → `resolve_target`; call `runner.process_lead(target)`;
- `account_intelligence` → `runner.briefing(target)`;
- `meeting_prep` → `runner.prep(target)`;
- `inbox_review` → `runner.inbox()`;
- `clarify` or missing target → print the available demo examples and return `0`.

`--decisions` is an alias for `--decision-backend openai`. `--approve` enables HITL; `--yes` selects `AutoApprover(True)` instead of `CliApprover`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_cli.py
git commit -m "feat: add intent and auto CLI commands"
```

---

### Task 11: Documentation and final regression

**Files:**
- Modify: `README.md`
- Modify: `INNOVATIONS.md`
- Modify: `.env.example`
- Test: full suite

- [ ] **Step 1: Update documentation**

Document:

- OpenAI Decisions as the first live decision backend;
- `intent`, `auto`, `--selector`, `--interpreter`, `--approve`;
- the read-only PTC boundary;
- the new env vars and `OPENAI_API_KEY` / `DECISIONS_API_KEY`;
- the corrected interpreter trace (`eval` + normal draft call);
- removal of the “missing intent/selector/PTC/HITL” gaps in `INNOVATIONS.md`.

- [ ] **Step 2: Run the full offline suite**

Run: `pytest -q`
Expected: PASS.

Run: `python main.py test`
Expected: all scenarios pass, including intent, selection, auto PTC, and HITL.

Run: `python main.py --interpreter prep "Priya Sharma"`
Expected: prep completes without `TypeError: not a function`.

- [ ] **Step 3: Commit**

```bash
git add README.md INNOVATIONS.md .env.example
git commit -m "docs: document OpenAI Decisions intent, selection, PTC, and HITL"
```

---

## Self-Review Notes

- **Spec coverage:** intent classifier (Tasks 3–4, 10), model middleware reuse (Task 8), conversation-aware selection (Task 5), automatic PTC (Tasks 1–2, 6), per-tool HITL (Tasks 7–9), docs (Task 11).
- **Type consistency:** `IntentDecision`, `ptc_allowlist`, `ConversationToolSelectionMiddleware`, `interrupt_config`, `run_with_hitl`, and `GTMAgentRunner` attributes are defined once and consumed by name in later tasks.
- **Review Focus mapping:** ambiguous intent → Task 4/10; selector failure → Task 5; prep PTC → Tasks 2/6/9; HITL rejection → Tasks 7/8; conversation context → Task 5.
