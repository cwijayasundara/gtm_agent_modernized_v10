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

What makes this sample different is *how tools get invoked*. With 50 tools in the catalog,
"which tool next?" becomes the agent's hardest — and most expensive — decision. This repo
implements four layers that answer it, two of which take the decision away from the model
entirely.

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

## The answer: four layers, each removing a decision

| # | Layer | Mechanism | Removes |
|---|---|---|---|
| 1 | **Skills bind tools** (progressive disclosure) | 47 of 50 tools are created with `extras={"defer_loading": True}`; a tool's schema is disclosed only after the model reads the `SKILL.md` that lists it in `metadata.include_tools` | *Discovery*: the model can't call, or even see, tools from a workflow it hasn't loaded |
| 2 | **LLM tool selection** | `LLMToolSelectorMiddleware` runs a cheap selection call before the loop and binds only the chosen names | *Relevance*: the loop starts with 4 tools instead of 50 |
| 3 | **Code interpreter / PTC** | `CodeInterpreterMiddleware(ptc=[...])` exposes allowlisted tools as `tools.<camelCase>()` inside a JS REPL; the model writes one program per workflow | *Serial round trips*: gate, batch, compose and queue in a single `eval` turn |
| 4 | **Defined workflows** | `gtm_agent/direct.py` invokes the same tools directly, in the playbook's authored order, with the same gates | *Everything*: no model in the loop at all — used for deterministic/offline execution |

Layers 1–3 still let the model reason; layer 4 doesn't. That is deliberate: the repo ships both
so you can move a workflow down the stack as it stabilizes.

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

### Layer 2 — pick the few relevant tools before the loop

```python
agent = build_tool_selector_agent(max_tools=3)   # LLMToolSelectorMiddleware
```

Offline the selector sees the entire catalog (the 50 GTM tools plus the deepagents built-ins)
and picks the relevant names; the agent loop then binds only those — `4` for outbound, `3` for
intel. The test scripts a one-tool selection and asserts the loop binds 1 of the 50 GTM tools.

### Layer 3 — let the model write code that calls the tools

```python
agent = build_interpreter_agent()   # CodeInterpreterMiddleware(ptc=DEFAULT_INTERPRETER_PTC)
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

  // Draft and queue for rep approval inside the same program.
  const draft = `Hi ${lead.contact.split(" ")[0]} — ...`;   // model-written copy
  const rationale = "Cold → brief, research-backed opener referencing their AI push.";
  await tools.draftSlackFollowup({ lead_id: leadId, draft, rationale });
  return { outcome: "drafted", draft, rationale };
})()
```

Gating, batching, dependent chaining and the side effect all happen inside the runtime; the
model sees one `<result>` instead of five tool-call turns.

### Layer 4 — or don't use a model at all

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
| Interpreter (PTC), skill pinned | 2 | 5, all inside **1** `eval` |
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
      ├─ build_agent()                            ─┐
      ├─ build_interpreter_agent()                 │  gtm_agent/agent.py
      ├─ build_tool_selector_agent()               │  create_deep_agent(...)
      └─ build_routed_agent()                     ─┘  gtm_agent/routing.py
                │
                ▼
        create_deep_agent
          ├── model         Claude Haiku 5.5  (or the scripted fake model offline)
          ├── tools         50 mocked tools; 47 deferred
          ├── skills        /skills/*/SKILL.md  →  metadata.include_tools
          ├── backend       FilesystemBackend(root_dir=<repo>)
          └── middleware    SkillsMiddleware            ← layer 1 (disclosure)
                            LLMToolSelectorMiddleware   ← layer 2 (--selector)
                            CodeInterpreterMiddleware   ← layer 3 (--interpreter)
                            ModelSelectionMiddleware    ← model routing (--route)
                │
                ▼
        tool calls → in-memory Salesforce / Gong / Exa / BigQuery / Zendesk / Slack fixtures
                     (gtm_agent/tools.py; reset_fixtures() between runs)

python main.py direct …    ← bypasses the whole graph: gtm_agent/direct.py calls the same
                             tool functions directly, in the playbook's authored order (layer 4)
```

### Modules

| Module | Responsibility |
|---|---|
| `main.py` | CLI, `GTMAgentRunner` (workflow + mode wiring, artifacts in `out/`), `GTMAgentTestRunner` (8 offline scenarios) |
| `gtm_agent/agent.py` | The three agent builders, `GTM_SYSTEM_PROMPT`, `GTM_INTERPRETER_SYSTEM_PROMPT`, `DEFAULT_INTERPRETER_PTC` |
| `gtm_agent/tools.py` | 50-tool mock stack: 3 always-on, 47 deferred; in-memory fixtures + `reset_fixtures()` |
| `gtm_agent/direct.py` | Layer 4: defined invocation runners (`run_outbound_followup`, `run_account_briefing`, `run_meeting_prep`) |
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

### Quick start — no API key needed

Requires Python ≥ 3.10 (this repo is developed/tested on 3.13).

```bash
uv venv .venv
uv pip install -e .                 # deepagents, langchain-anthropic, python-dotenv
.venv/bin/python main.py test       # 8/8 offline scenarios
.venv/bin/python main.py            # a full simulated GTM day (3 leads + briefing + prep + inbox)
```

The offline runs use a scripted fake chat model (`gtm_agent/simulator.py`) but the **real**
deepagents graph: real tool-calling loop, real skills middleware, real filesystem backend.
Only the LLM is simulated.

### Optional: enable the code interpreter (layer 3)

```bash
uv pip install -e ".[interpreter]"  # adds deepagents[quickjs] → langchain-quickjs
```

Without this extra the rest of the app works; `--interpreter` raises a clear install hint.

### The workflows

```bash
.venv/bin/python main.py                          # full simulated day
.venv/bin/python main.py lead lead-1042           # inbound lead follow-up
.venv/bin/python main.py briefing "Helix Health"  # account-intelligence briefing
.venv/bin/python main.py prep "Priya Sharma"      # meeting-prep brief
.venv/bin/python main.py inbox                    # drafts pending rep approval
```

Artifacts land in `out/drafts/`, `out/briefings/`, `out/meeting-preps/`; nothing is sent.

### Execution modes

```bash
# Layer 1 — skills-scoped agent loop (default)
.venv/bin/python main.py lead lead-1042

# Layer 2 — LLM tool selection narrows the toolset before the loop
.venv/bin/python main.py --selector lead lead-1042

# Layer 3 — code interpreter / programmatic tool calling (one eval per workflow)
.venv/bin/python main.py --interpreter lead lead-1042
.venv/bin/python main.py --interpreter briefing "Helix Health"
.venv/bin/python main.py --interpreter prep "Priya Sharma"

# Layer 4 — defined workflows: no agent loop, no model, no API key
.venv/bin/python main.py direct lead lead-1042
.venv/bin/python main.py direct intel "Helix Health"
.venv/bin/python main.py direct prep "Priya Sharma"

# Layers compose: routed + interpreter + selector
.venv/bin/python main.py --route --interpreter --selector lead lead-1042
# tools bound: 1 (eval) | tool trace: eval | routing: gpt-6-luna (sticky)
```

`--interpreter` prints its PTC footprint in the run footer:

```
  tool trace:  eval
  invocation:  code interpreter — 1 eval program(s), 8 tools on the PTC allowlist
```

### Live mode (Claude Haiku 5.5)

```bash
cp .env.example .env               # then set ANTHROPIC_API_KEY
.venv/bin/python main.py --live lead lead-1042
.venv/bin/python main.py --live briefing "Helix Health"
.venv/bin/python main.py --live prep "Priya Sharma"
```

Example live prompts (the model drives the loop itself):

- `"New inbound lead: lead-1043 (Acme Robotics). Run the standard follow-up."`
- `"Give me an account intelligence briefing for Helix Health."`
- `"Position us against Pinecone and Weaviate."` (methodology-only skill)

### Tests and proofs

```bash
.venv/bin/python main.py test -v                 # 8 scenarios, per-model-call disclosure snapshots

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
✅ interpreter PTC      1 eval per workflow; same gates + identical draft as the loop playbook.
✅ tool selector        LLM selector bound 1 of 50 tools (chosen before the loop).
8/8 scenarios passed
```

All of the above run **without credentials**; the only key the repo can use is
`ANTHROPIC_API_KEY` for `--live` (plus provider keys for `--route`).

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
