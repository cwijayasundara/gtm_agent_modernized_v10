# GTM Agent — invoking the right tool, by design

A go-to-market agent modeled on
[How we built LangChain's GTM Agent](https://www.langchain.com/blog/how-we-built-langchains-gtm-agent),
built with [`deepagents`](https://github.com/langchain-ai/deepagents) and **Claude Haiku 5.5**
(`claude-haiku-5-5`).

It runs three real GTM workflows against a **50-tool simulated stack** (Salesforce, Gong, Exa,
BigQuery, Zendesk, Slack — all in-memory mocks):

1. **Inbound lead follow-up** — do-not-send checks → research → draft → queue for rep approval.
2. **Account intelligence** — usage, hiring and funding signals → expansion/risk flags.
3. **Meeting prep** — dossier, last interaction, priorities, agenda, style notes.

Every workflow writes artifacts to `out/` and **nothing is ever sent**: drafts wait for explicit
rep approval.

What makes this sample different is *how intent becomes a tool call*. With 50 tools in the
catalog, "which tool next?" becomes the agent's hardest — and most expensive — decision. This
repo implements a five-layer selection stack plus an explicit approval boundary, and moves
stable workflows all the way down to deterministic code.

---

## The problem: too many tools, one model decision

An agent loop asks the model, on every turn, "given this conversation, which tool do I call
next?" At small scale that is fine. At 50 tools it produces three distinct failure modes:

| Failure mode | What it looks like |
|---|---|
| **Context bloat** | Every tool schema ships on every request; token cost and latency grow with the catalog, not with the task. |
| **Wrong-workflow tools** | The model reaches for an analytics tool during an outbound task because nothing scopes the toolset to the playbook. |
| **Serial round trips** | Each tool call costs a full model turn: look up the lead → *model turn* → search the web → *model turn* → draft. Five calls, five round trips, five chances to drift. |

These are different problems, so one mechanism can't fix all three.

## The answer: five layers, each removing a decision

| # | Layer | Mechanism | Removes |
|---|---|---|---|
| 1 | **Skills bind tools** (progressive disclosure) | 47 of 50 tools are created with `extras={"defer_loading": True}`; a tool's schema is disclosed only after the model reads the `SKILL.md` that lists it in `metadata.include_tools` | *Discovery*: the model can't call, or even see, tools from a workflow it hasn't loaded |
| 2 | **Intent routing** | OpenAI Decisions `Choice` + `needs_clarification` predicate map a request to a skill, or ask instead of guessing | *Wrong workflow*: the agent starts with one playbook, not all five |
| 3 | **Conversation-aware tool selection** | `ConversationToolSelectionMiddleware` sends recent messages, tool calls, and tool results to OpenAI Decisions and binds only the chosen tools | *Relevance*: the loop sees a focused tool surface, not all 50 |
| 4 | **Code interpreter / PTC** | `CodeInterpreterMiddleware(ptc=[...])` exposes **read-only** tools as `tools.<camelCase>()` inside a JS REPL; the model writes one program per workflow | *Serial round trips*: gate, batch, and compose in one `eval` turn |
| 5 | **Defined workflows** | `gtm_agent/direct.py` invokes the same tools directly, in the playbook's authored order, with the same gates | *Everything*: no model in the loop at all — used for deterministic/offline execution |

Action tools are deliberately outside PTC: `draft_slack_followup` and
`queue_followup_sequence` run through the normal tool path with `--approve`, so a human
approves the side effect per tool. Layers 1–4 still let the model reason; layer 5 doesn't.

### Layer 1 — a skill scopes the toolset

```python
@tool(extras={"defer_loading": True})          # schema withheld until disclosed
def check_contact_history(lead_id: str) -> str: ...
```

```yaml
# skills/outbound-followup/SKILL.md
metadata:
  include_tools: lookup_lead check_contact_history get_gong_transcripts
                 web_search_company draft_slack_followup
```

A lead follow-up run only ever sees the 12 outbound GTM tools; the 15 analytics tools are not
in the request at all. Skill isolation is asserted in the test suite.

### Layer 2 — classify intent before the loop

```bash
python main.py intent "Prep me for my meeting with Priya Sharma"
python main.py auto "Follow up with lead-1042"
```

`IntentRouter` asks OpenAI Decisions for a `choice` over the runnable skills plus a
`needs_clarification` predicate. High confidence pins the matching skill; low confidence asks
for a target instead of guessing. `auto` also resolves demo-fixture targets, so the request
runs end to end. `GTM_DECISION_BACKEND=offline` keeps the same shape with a keyword fallback.

### Layer 3 — conversation-aware tool selection

```python
agent = build_agent(extra_middleware=[
    ConversationToolSelectionMiddleware(selector=ToolSelector(...), max_tools=6)
])
```

The selector sees recent user messages, assistant tool calls, and tool results — not just the
last line. It asks OpenAI Decisions which tools matter, then binds those plus the deepagents
built-ins. It selects once per run, not on every model turn. `--selector` enables it; the
offline fallback keeps all tools if the decision call fails.

### Layer 4 — let the model write code that calls the tools

```python
agent = build_interpreter_agent(skill="outbound-followup")   # automatic read-only PTC allowlist
```

The outbound playbook collapses to **one** `eval` program the model authors:

```javascript
(async () => {
  const leadId = "lead-1042";

  // Gate first: a do-not-send signal or a touch within 3 days ends the program.
  const history = JSON.parse(await tools.checkContactHistory({ lead_id: leadId }));
  if (history.do_not_send) return { outcome: "blocked", reason: history.summary };
  if ((history.prior_outreach || []).some((o) => o.days_ago <= 3))
    return { outcome: "blocked", reason: history.summary };

  // Batch the independent research calls in one Promise.all.
  const lead = JSON.parse(await tools.lookupLead({ lead_id: leadId }));
  const [research, gong] = await Promise.all([
    tools.webSearchCompany({ company: lead.company }),
    tools.getGongTranscripts({ lead_id: leadId }),
  ]);

  // The eval bridge is read-only. Return the proposal; queue it through the
  // normal HITL-gated draft tool after the program finishes.
  const draft = `Hi ${lead.contact.split(" ")[0]} — ...`;   // model-written copy
  const rationale = "Cold → brief, research-backed opener referencing their AI push.";
  return { outcome: "drafted", draft, rationale };
})()
```

Gating, batching, and dependent chaining happen inside the runtime; the model sees one
`<result>` instead of several tool-call turns. The follow-up `draft_slack_followup` call is a
normal tool call, so `--approve` can pause it.

### Guardrail — per-tool approval for actions

```bash
python main.py --approve --yes lead lead-1042   # auto-approve for a demo
python main.py --approve lead lead-1042         # interactive approve/reject
```

PTC cannot call action tools. `draft_slack_followup` and `queue_followup_sequence` are only
reachable through the normal tool path, where `HumanInTheLoopMiddleware` interrupts before
execution. Rejecting leaves the draft queue empty; approving queues exactly one draft.

### Layer 5 — or don't use a model at all

```python
from gtm_agent import direct

direct.run_outbound_followup("lead-1042")
# {"outcome": "drafted", "draft": "...", "rationale": "...", "draft_queued": True}
```

`direct.py` builds a `name → tool` registry over the same 50 tools and calls them in the
playbook's order, enforcing the same do-not-send / recent-touch gates and queueing into the
same `pending_approval` draft list the agent uses.

### Measured, offline (`python main.py test` fixtures)

| Outbound follow-up run | Model turns | Tool invocations |
|---|---|---|
| Agent loop, skill not pinned | 6 | 5 (incl. `read_file`) |
| Agent loop, skill pinned | 5 | 4 |
| Interpreter (PTC), skill pinned | 3 | 2 visible (`eval` + `draft_slack_followup`); 5 tool calls total |
| Defined workflow (`direct`) | **0** | 5 |

The interpreter run's queued draft is asserted **byte-identical** to the loop version's, so
the cheaper path does not change the outcome — only the number of round trips.

---

## Architecture

```
python main.py …                                  CLI: workflow × execution mode × model flags
      │
      ▼
GTMAgentRunner (main.py)                          picks the builder, pins the skill, saves artifacts
      │
      ├─ IntentRouter (pre-flight)                 → skill / clarify / general
      ├─ build_agent()                            ─┐
      ├─ build_interpreter_agent()                 │  gtm_agent/agent.py
      └─ build_routed_agent()                     ─┘  gtm_agent/routing.py
                │
                ▼
        create_deep_agent
          ├── model         Claude Haiku 5.5  (or the scripted fake model offline)
          ├── tools         50 mocked tools; 47 deferred
          ├── skills        /skills/*/SKILL.md  →  metadata.include_tools
          ├── backend       FilesystemBackend(root_dir=<repo>)
          └── middleware    SkillsMiddleware                  ← layer 1 (disclosure)
                            ConversationToolSelectionMiddleware ← layer 3 (--selector)
                            CodeInterpreterMiddleware           ← layer 4 (--interpreter)
                            ModelSelectionMiddleware            ← model routing (--route)
                            HumanInTheLoopMiddleware            ← --approve actions
                │
                ▼
        tool calls → in-memory Salesforce / Gong / Exa / BigQuery / Zendesk / Slack fixtures
                     (gtm_agent/tools.py; reset_fixtures() between runs)

python main.py direct …    ← bypasses the whole graph: gtm_agent/direct.py calls the same
                             tool functions directly, in the playbook's authored order (layer 5)
```

### Modules

| Module | Responsibility |
|---|---|
| `main.py` | CLI, `GTMAgentRunner` (workflow + mode wiring, artifacts in `out/`), `GTMAgentTestRunner` (8 offline scenarios) |
| `gtm_agent/agent.py` | The three agent builders, `GTM_SYSTEM_PROMPT`, `GTM_INTERPRETER_SYSTEM_PROMPT`, legacy PTC fallback |
| `gtm_agent/intent.py` | OpenAI Decisions intent router, offline fallback, demo target resolver |
| `gtm_agent/selection.py` | Conversation transcript renderer + `ConversationToolSelectionMiddleware` |
| `gtm_agent/ptc.py` | Read-only PTC allowlist derived from skill metadata |
| `gtm_agent/hitl.py` | Action-tool interrupt config, reviewers, HITL resume loop |
| `gtm_agent/decisions.py` | Decision backend resolution and `ToolSelector` factory |
| `gtm_agent/skills.py` | `SKILL.md` frontmatter parser |
| `gtm_agent/tools.py` | 50-tool mock stack: 3 always-on, 47 deferred; side-effect metadata; fixtures + `reset_fixtures()` |
| `gtm_agent/direct.py` | Layer 5: defined invocation runners (`run_outbound_followup`, `run_account_briefing`, `run_meeting_prep`) |
| `gtm_agent/simulator.py` | Offline LLM: scripted deepagents runs, the GTM playbook simulator, the JS the scripted model "authors", and the selector-capable fake |
| `gtm_agent/routing.py` | `decision_harness` model-selection middleware, telemetry ledger, `build_routed_agent(..., extra_middleware=...)` |
| `gtm_agent/catalog_overlay.json` | Routing catalog as data: 4 models, 3 tiers, prices, capability fields |
| `skills/*/SKILL.md` | Playbooks + the tools each one owns (`metadata.include_tools`) |
| `tests/smoke_test_*.py` | Standalone proofs: skills disclosure, interpreter PTC, defined invocation |

### Where the pieces meet

- **Skills ↔ tools**: `include_tools` in a `SKILL.md` frontmatter names deferred tools; the
  deepagents `SkillsMiddleware` discloses them when the skill is read *or pinned*
  (`payload["pinned_skills"]`, used by the CLI to skip the `read_file` turn).
- **Runner ↔ modes**: `GTMAgentRunner(live=…, routed=…, interpreter=…, selector=…). _invoke()`
  selects the builder; the flags are independent and compose.
- **Routing ↔ everything else**: `build_routed_agent(extra_middleware=[…], system_prompt=…)`
  inserts PTC/selector middleware alongside the routing middleware.

---

## Run it

### 1. Install

Requires Python ≥ 3.10 (developed/tested on 3.13). The repo uses `uv`:

```bash
uv venv .venv
uv pip install -e ".[dev,interpreter]"
```

- `dev` adds `pytest` for the unit suite.
- `interpreter` adds the optional QuickJS runtime for `--interpreter`.
- If you only want the basic agent loop, `uv pip install -e .` is enough;
  `--interpreter` will then print an install hint.

Optional: copy the environment template and fill in keys later.

```bash
cp .env.example .env
```

### 2. Quick start — no API key needed

```bash
.venv/bin/python main.py test       # 8 rule-based scenarios
.venv/bin/python main.py            # a full simulated GTM day
```

The offline runs use a scripted fake chat model (`gtm_agent/simulator.py`) but the **real**
deepagents graph: real tool-calling loop, real skills middleware, real filesystem backend.
Only the LLM is simulated.

Run one workflow at a time:

```bash
.venv/bin/python main.py lead lead-1042           # inbound lead follow-up
.venv/bin/python main.py briefing "Helix Health"  # account-intelligence briefing
.venv/bin/python main.py prep "Priya Sharma"      # meeting-prep brief
.venv/bin/python main.py inbox                    # drafts pending rep approval
```

Artifacts land in `out/drafts/`, `out/briefings/`, and `out/meeting-preps/`.
Nothing is sent: drafts are queued with `status: pending_approval`.

### 3. Try the selection layers

```bash
# Layer 1 — skills-scoped agent loop (default)
.venv/bin/python main.py lead lead-1042

# Layer 2 — classify a natural-language request
.venv/bin/python main.py intent "Prep me for my meeting with Priya Sharma"
.venv/bin/python main.py auto "Follow up with lead-1042"
.venv/bin/python main.py auto "Give me an account intelligence briefing for Helix Health"
.venv/bin/python main.py auto "Show me the approval inbox"

# Layer 3 — conversation-aware tool selection
.venv/bin/python main.py --selector lead lead-1042

# Layer 4 — read-only code interpreter / programmatic tool calling
.venv/bin/python main.py --interpreter lead lead-1042
.venv/bin/python main.py --interpreter briefing "Helix Health"
.venv/bin/python main.py --interpreter prep "Priya Sharma"

# Layer 5 — defined workflows: no agent loop, no model, no API key
.venv/bin/python main.py direct lead lead-1042
.venv/bin/python main.py direct intel "Helix Health"
.venv/bin/python main.py direct prep "Priya Sharma"

# Compose layers
.venv/bin/python main.py --route --interpreter --selector lead lead-1042
```

`intent` only classifies; `auto` classifies and dispatches. `auto` resolves targets from the
demo fixtures (`lead-1042`, `Helix Health`, `Priya Sharma`) and asks for a target when it cannot
find one.

`--selector` sends the recent conversation to OpenAI Decisions and binds only the selected
tools. Without an OpenAI key it falls back to keeping all tools. `--interpreter` derives a
read-only PTC allowlist from the selected skill; the draft is then queued through the normal
tool path.

### 4. Approval boundary for action tools

```bash
# Interactive: prints each action and asks approve/reject
.venv/bin/python main.py --approve lead lead-1042

# Non-interactive demo: auto-approve every requested action
.venv/bin/python main.py --approve --yes lead lead-1042

# Approval + interpreter + routing
.venv/bin/python main.py --approve --yes --route --interpreter lead lead-1042
```

`draft_slack_followup` and `queue_followup_sequence` are never bridged into PTC. They run as
normal tools behind `HumanInTheLoopMiddleware`. Rejecting leaves the draft queue empty.

### 5. Optional: live OpenAI Decisions

Intent routing, conversation-aware tool selection, and `--route` use the OpenAI Decisions API
when a key is available. Without a key they use deterministic offline fallbacks.

```bash
# Put a key in .env, or export it.
export OPENAI_API_KEY=sk-...
# DECISIONS_API_KEY is also accepted.

# Use Decisions for auto-intent and tool selection
.venv/bin/python main.py --decision-backend openai auto "Follow up with lead-1042"
.venv/bin/python main.py --decision-backend openai --selector lead lead-1042

# --decisions is a shortcut for --decision-backend openai
.venv/bin/python main.py --decisions intent "Prep me for Priya Sharma"

# Force offline behavior
.venv/bin/python main.py --decision-backend offline auto "Prep me for Priya Sharma"
```

`--route` remains independent: it uses `decision_harness` to choose the chat model
(`--decision-backend` controls intent/tool decisions).

### 6. Optional: live Claude model

```bash
cp .env.example .env               # set ANTHROPIC_API_KEY
.venv/bin/python main.py --live lead lead-1042
.venv/bin/python main.py --live briefing "Helix Health"
.venv/bin/python main.py --live prep "Priya Sharma"

# Live model + Decisions + approval
.venv/bin/python main.py --live --decision-backend openai --approve lead lead-1042
```

Example live prompts:

- `"New inbound lead: lead-1043 (Acme Robotics). Run the standard follow-up."`
- `"Give me an account intelligence briefing for Helix Health."`
- `"Position us against Pinecone and Weaviate."` (methodology-only skill)

### 7. Configuration

| Variable | Default | Used by |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | `--live` Claude calls |
| `OPENAI_API_KEY` / `DECISIONS_API_KEY` | — | intent, tool selection, model routing |
| `GTM_DECISION_BACKEND` | `auto` | `auto`, `openai`, or `offline` |
| `GTM_DECISION_MODEL` | `gpt-6-luna` | OpenAI Decisions model |
| `GTM_INTENT_CONFIDENCE_FLOOR` | `0.65` | intent route vs clarify |
| `GTM_TOOL_SELECTION_MAX_TOOLS` | `6` | live tool-selection cap |
| `GTM_TOOL_SELECTION_THRESHOLD` | `0.5` | live tool-selection probability floor |
| `GTM_MODEL_ID` | `claude-haiku-5-5` | default chat model |
| `FIREWORKS_API_KEY` | — | fast tiers for `--route` |
| `LANGSMITH_*` | — | optional tracing |

`auto` selects OpenAI Decisions when a key is present, otherwise offline. Explicit
`--decision-backend openai` fails fast if no key is configured.

### 8. Tests and proofs

```bash
.venv/bin/python -m pytest -q                    # 33 unit tests
.venv/bin/python main.py test -v                 # 8 offline scenarios, disclosure snapshots

.venv/bin/python tests/smoke_test_skill_tools.py        # deferred tools disclosed only after SKILL.md read
.venv/bin/python tests/smoke_test_defined_invocation.py # gates + intel + prep with zero model turns
.venv/bin/python tests/smoke_test_interpreter_ptc.py    # eval → allowlisted tool → result (needs [interpreter])
```

`main.py test` runs 8/8 with the interpreter extra installed. Without it, the interpreter
scenario is skipped with an install hint (`7/7 passed, 1 skipped`) rather than failing.

Expected `main.py test` output:

```
✅ tool universe        50 tools (47 deferred); prep run disclosed exactly its 8, hid 42.
✅ do-not-send gate     checked first, drafted nothing for blocked lead.
✅ outbound playbook    check → research → draft; queued as pending_approval.
✅ skill isolation      intel run disclosed its 15; outbound tools stayed hidden.
✅ model routing        query routed to openai/gpt-6-luna (balanced), sticky after call 1, 6 ledger entries.
✅ defined invocation   no agent loop: gates enforced, drafts queued, intel + prep produced.
✅ interpreter PTC      read-only eval + HITL-gated draft; same gates as the loop playbook.
✅ tool selector        Decisions selector bound 9 tools including built-ins.
8/8 scenarios passed
```

All of the above run **without credentials**. `--live` needs `ANTHROPIC_API_KEY`;
`--route`, `intent`, `auto`, and `--selector` use `OPENAI_API_KEY` /
`DECISIONS_API_KEY` when available and fall back offline otherwise.

---

## Tools ↔ skills map (50-tool simulated stack)

| Skill (`metadata.include_tools`) | Deferred tools bound |
|---|---|
| `outbound-followup` | 12 — CRM, engagement history, LinkedIn, drafting, follow-up sequences |
| `account-intelligence` | 15 — usage, installs, hiring, funding, NPS, churn/expansion scores |
| `customer-health` | 12 — tickets, escalations, renewals, credits, QBR notes |
| `meeting-prep` | 8 — last interaction, dossiers, threads, agendas |
| `competitor-analysis` | 0 — methodology-only (rides on model knowledge) |
| *(always available)* | 3 — `list_pending_drafts`, `get_current_date`, `search_internal_wiki` |

Bespoke tools (`lookup_lead`, `check_contact_history`, `draft_slack_followup`,
`query_product_usage`) carry real fixture data; the rest are generated mocks with a generic
`query` parameter standing in for the wider stack.

---

## Model routing: 4 LLMs, chosen per query (`--route`)

[`decision_harness`](https://github.com/cwijayasundara/deep_agents_utils_v1.0) is layered on
the skills-bound agent as model-selection middleware. Each thread's first query is classified
once (tier + confidence, complexity, planning) and routed to the cheapest capable model in
`gtm_agent/catalog_overlay.json`:

| Tier | Model | Provider |
|---|---|---|
| fast | glm-5.3-flash | fireworks |
| fast alt | deepseek-4-flash | fireworks |
| balanced | gpt-6-luna | openai |
| performance | claude-haiku-5-5 | anthropic |

- **Sticky per thread** (prompt cache stays warm); escalates a tier on provider failure; tiers
  without a configured key are pruned from the routable catalog.
- Every call is logged with tokens, **$ cost** (catalog prices — placeholders, verify before
  production) and latency; the ledger is exported to `out/usage/ledger-*.json`.
- Classification uses the Decisions API when `DECISIONS_API_KEY`/`OPENAI_API_KEY` is set, and
  degrades to an offline GTM-aware heuristic (`heuristic+gtm`) otherwise.

```bash
.venv/bin/python main.py route "Process lead-1042: run the standard follow-up"          # decision preview
.venv/bin/python main.py route "Review our renewal strategy for a churn-risk account"   # → performance
.venv/bin/python main.py --route --live lead lead-1042                                  # routed live run + report
```

---

<details>
<summary><b>Background: how deepagents binds tools to skills</b></summary>

deepagents ≥ 0.7 lets a skill **own its tools**. Mechanics (v0.7.23):

1. **Defer a tool** with `extras={"defer_loading": True}` — the provider withholds its schema
   at the start of the conversation; calling it before disclosure fails as an unknown tool.
2. **Bind it to a skill** by listing it under `metadata.include_tools` in the skill's
   `SKILL.md` frontmatter.
3. **Disclose on read** — when the model reads that `SKILL.md` (or the skill is *pinned*), the
   `SkillsMiddleware` discloses the bound tools. On inline-capable models (Claude Opus 5.x /
   Fable 5.x, GPT-5.6/6 Responses) the definition is injected mid-conversation as a
   provider-native `tool_addition` / `additional_tools` block; on other models (incl. Claude
   Haiku 5.5) the deferred tool is swapped for an un-deferred copy so the schema ships in the
   normal `tools` payload.
4. **Alternative wiring** — pass a `SkillToolResolver` to `SkillsMiddleware(tools=...)` to map
   an include name to tools known only at runtime (e.g. generated MCP tool names).

Why it matters: token savings plus a capability gate — the model only sees the toolset
relevant to the playbook it loaded, and an outbound run cannot touch analytics tools.

Reference architecture behind this sample: trigger on new Salesforce leads; strict checklist
(*do-not-send → research → draft → rationale → queue for approval*); human-in-the-loop with
per-rep learning; two workflows (lead follow-up, account intelligence); deepagents for spiky
tool outputs (filesystem offload) and native planning; evals + LangSmith traces in CI.

</details>

---

## Ideas to take it further

- **Promote a workflow down the layers.** Once a playbook's tool sequence stops changing, move
  it from the loop to `direct.py` (or PTC) and keep the agent for the ambiguous cases.
- **Add per-rep memory** (`memory=["/memory/AGENTS.md"]`) so rep edits shape future drafts.
- **Add compiled subagents** (e.g. a market researcher) with their own skills; `include_tools`
  works there too.
- **Swap mocks for real integrations via MCP**, using `SkillToolResolver` so one include name
  can stand for a whole MCP integration's tools.
- **Turn the PTC allowlist into policy**: `ptc=` already limits what a model-authored program
  can reach; add per-tenant allowlists for the same pattern.
