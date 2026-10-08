# Sample GTM Agent — deepagents "binding tools to skills"

A sample GTM agent modeled on [How we built LangChain's GTM Agent](https://www.langchain.com/blog/how-we-built-langchains-gtm-agent),
built with [`deepagents`](https://github.com/langchain-ai/deepagents) and
**Claude Haiku 5.5** (`claude-haiku-5-5`), demonstrating the new
**binding tools to skills** feature (see Sydney Runkle's [announcement](https://x.com/sydneyrunkle/status/2107906347199336864)).

---

## Research summary

### The feature: binding tools to skills

deepagents ≥ 0.7 lets a skill **own its tools**. The mechanics (from
`deepagents/middleware/_skill_tools.py` and `middleware/skills.py`, v0.7.23):

1. **Defer a tool** — create it with `extras={"defer_loading": True}`:
   ```python
   @tool(extras={"defer_loading": True})
   def lookup_lead(lead_id: str) -> str: ...
   ```
   The provider withholds its schema from the model at the start of the
   conversation. Calling it before disclosure fails as an unknown tool.

2. **Bind it to a skill** — list it in the skill's `SKILL.md` frontmatter
   (space-separated names under `metadata.include_tools`):
   ```yaml
   metadata:
     include_tools: lookup_lead check_contact_history
   ```

3. **Disclosure on read** — once the model uses `read_file` on that
   `SKILL.md` (or the skill is *pinned* via `pinned_skills`), the
   `SkillsMiddleware` discloses the bound tools for as long as the skill
   stays in context:
   - On inline-capable models (Claude Opus 5.x / Fable 5.x, GPT-5.6/6
     Responses), the tool definition is injected mid-conversation as a
     provider-native `tool_addition` / `additional_tools` block.
   - On other models (incl. Claude Haiku 5.5), the middleware swaps the
     deferred tool for an un-deferred copy so the schema ships in the
     normal `tools` payload.

4. **Alternative wiring** — instead of deferring agent tools, you can pass
   a `SkillToolResolver` to `SkillsMiddleware(tools=...)` directly. A
   resolver maps an include name to one or more tools known only at runtime
   (e.g. generated MCP tool names): one name can stand for a whole MCP
   integration.

**Why it matters** (from the announcement / docs): token savings and
focus — the model only sees the toolset relevant to the playbook it loaded,
and each skill acts as a capability gate: the outbound playbook can't touch
analytics tools it never asked for.

### The reference architecture (LangChain's GTM agent)

From the blog post:

- **Trigger** on new Salesforce leads; run a strict checklist:
  *do-not-send checks → research → draft → rationale → queue for approval*.
- **Human-in-the-loop**: nothing sends without rep approval; every draft
  carries reasoning + sources. Rep edits (send/edit/cancel) are logged and
  feed a per-rep memory/learning loop.
- **Two workflows**: inbound lead follow-up (relationship-aware drafting:
  customer / warm / cold) and weekly account intelligence (usage, hiring,
  funding signals → expansion/risk flags).
- **Subagents**: parallel compiled subagents with constrained toolsets and
  structured outputs act as contracts with the parent agent.
- **Built on deepagents** because inputs are "spiky" — large tool results
  offload to the virtual filesystem, and native planning keeps runs on
  checklist. Evals (rule-based + LLM judge) run in CI; every rep action
  attaches to the trace in LangSmith.

The repo even ships a `examples/deploy-gtm-agent` example (supervisor +
sync market-researcher subagent + skills), which this sample borrows from.

---

## What this sample implements

```
gtm_agent_modernized_v1.0/
├── main.py             # GTMAgentTestRunner: rule-based scenario tests + live mode
├── gtm_agent/
│   ├── agent.py        # create_deep_agent + Claude Haiku 5.5 + skills
│   ├── tools.py        # 50-tool simulated stack: 47 DEFERRED + 3 always-on
│   └── demo.py         # simple runnable inbound-lead demo
├── skills/
│   ├── outbound-followup/SKILL.md      # binds 12 tools via include_tools
│   ├── account-intelligence/SKILL.md   # binds 15 tools via include_tools
│   ├── customer-health/SKILL.md        # binds 12 tools via include_tools
│   ├── meeting-prep/SKILL.md           # binds 8 tools via include_tools
│   └── competitor-analysis/SKILL.md    # methodology-only (binds none)
└── tests/smoke_test_skill_tools.py     # minimal binding proof, no API key
```

- **Model**: `claude-haiku-5-5` via `langchain_anthropic.ChatAnthropic`
- **Harness**: `create_deep_agent(model, tools, skills=["/skills/"], backend=FilesystemBackend(root_dir=<project>))`
- **Data sources** (Salesforce, Gong, Exa, BigQuery, Slack) are mocked
  in-memory fixtures — no external credentials needed except
  `ANTHROPIC_API_KEY` for the live demo.

### Tools ↔ skills map (50-tool simulated stack)

| Skill (`metadata.include_tools`) | Deferred tools bound |
|---|---|
| `outbound-followup` | 12 — CRM, engagement history, LinkedIn, drafting, follow-up sequences |
| `account-intelligence` | 15 — usage, installs, hiring, funding, NPS, churn/expansion scores |
| `customer-health` | 12 — tickets, escalations, renewals, credits, QBR notes |
| `meeting-prep` | 8 — last interaction, dossiers, threads, agendas |
| `competitor-analysis` | 0 — methodology-only |
| *(always available)* | 3 — `list_pending_drafts`, `get_current_date`, `search_internal_wiki` |

So a lead follow-up run only ever sees 12 of 50 tools; an intel run only 15.
Highlights are bespoke mocked tools (`lookup_lead`, `check_contact_history`,
`draft_slack_followup`, `query_product_usage`, ...); the rest are generated
mocks with a generic `query` parameter standing in for a realistic stack
(Salesforce, Gong, LinkedIn, Exa, BigQuery, Zendesk, Slack...).

---

## Run it

Dependencies are declared in `pyproject.toml`; secrets live in `.env`
(see `.env.example`) and are loaded automatically via `python-dotenv`.

```bash
# 1. Environment
uv venv .venv
uv pip install -e .            # installs deps from pyproject.toml
cp .env.example .env           # then fill in ANTHROPIC_API_KEY

# 2. Minimal binding proof (no API key needed)
.venv/bin/python tests/smoke_test_skill_tools.py

# 3. Full scenario test suite — meaningful, rule-based, no API key needed
.venv/bin/python main.py test   # add -v for per-model-call disclosure snapshots
#   ✅ tool universe     50 tools (47 deferred, 3 always-on); run disclosed only its skill's tools
#   ✅ do-not-send gate  SEV-2 lead → check runs first, nothing drafted
#   ✅ outbound playbook check → research → draft, queued as pending_approval
#   ✅ skill isolation   intel run never sees outbound tools

# 4. Do something useful: run the agent's actual GTM workflows (offline simulator)
.venv/bin/python main.py                        # a full simulated day: 3 leads + briefing + prep + inbox
.venv/bin/python main.py lead lead-1042         # process one inbound lead
.venv/bin/python main.py briefing Helix Health  # account-intelligence briefing
.venv/bin/python main.py prep "Priya Sharma"    # meeting-prep brief
.venv/bin/python main.py inbox                  # drafts pending rep approval
# → artifacts (drafts, briefings, preps) land in out/; nothing ever sends
#   without rep approval — the human-in-the-loop step.

# 5. Same workflows on the real model (needs ANTHROPIC_API_KEY in .env)
.venv/bin/python main.py --live lead lead-1042
.venv/bin/python main.py --live briefing Helix Health
```

Example live prompts:

- `"New inbound lead: lead-1043 (Acme Robotics). Run the standard follow-up."`
- `"Give me an account intelligence briefing for Helix Health."`
- `"Position us against Pinecone and Weaviate."` (methodology-only skill)

## Model routing: 4 LLMs, chosen per query (`--route`)

`main.py --route ...` adds [decision_harness](https://github.com/cwijayasundara/deep_agents_utils_v1.0)
as a routing middleware on top of the skills-bound agent. Each thread's query
is classified once (tier choice + confidence, complexity, planning) and routed
to the cheapest capable model in a four-model catalog (`gtm_agent/catalog_overlay.json`):

| Tier | Model | Provider |
|---|---|---|
| fast | glm-5.3-flash | fireworks |
| fast alt | deepseek-4-flash | fireworks |
| balanced | gpt-6-luna | openai |
| performance | claude-haiku-5-5 | anthropic |

- The decision is **sticky per thread** (prompt cache stays warm) and
  escalates a tier on provider failure; tiers without a configured API key
  are pruned from the routable catalog.
- Every call is logged with tokens, **$ cost** (catalog prices — placeholders,
  verify before production), and **response time**; the full ledger is
  exported to `out/usage/ledger-*.json`.
- Classification uses the OpenAI Decisions API when `DECISIONS_API_KEY`/`OPENAI_API_KEY`
  is set; otherwise it degrades to an offline heuristic extended with GTM
  signals (`heuristic+gtm`).

```bash
python main.py route "Process lead-1042: run the standard follow-up"   # decision preview, offline
python main.py route "Review our renewal strategy for a churn-risk account"  # → performance
python main.py --route --live lead lead-1042   # routed live run + usage report
python main.py --route                          # routed simulated day + report
```

## Ideas to take it further

- Add a per-rep **memory** source (`memory=["/memory/AGENTS.md"]`) so rep
  edits shape future drafts — mirrors the blog's learning loop.
- Add **compiled subagents** (e.g. a market-researcher) with their own
  skills; `include_tools` works there too.
- Swap the mocks for real integrations via **MCP** and a
  `SkillToolResolver` so one include name can stand for an entire MCP
  integration's tools.
