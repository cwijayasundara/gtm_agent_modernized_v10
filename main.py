"""The sample GTM agent, made useful: a workflow runner + test harness.

Runs the agent's real workflows end to end against the simulated 50-tool
stack, and writes its artifacts (drafts, briefings, meeting preps) to
`out/`. Every run still goes through the full deepagents graph — real
tool-calling loop, real skills middleware, real filesystem — only the LLM
is simulated offline (a data-driven scripted model built from the mock
fixtures). With `--live`, the same workflows run on Claude Haiku 5.5.

Usage:
    python main.py                          # a full simulated GTM day
    python main.py lead lead-1042           # process one inbound lead
    python main.py briefing Helix Health    # account-intelligence briefing
    python main.py prep "Priya Sharma"      # meeting-prep brief
    python main.py inbox                    # drafts pending rep approval
    python main.py test [-v]                # rule-based scenario tests
    python main.py --live lead lead-1042    # any workflow, on real Claude Haiku 5.5
    python main.py direct lead lead-1042    # defined tool invocation (no agent loop)
    python main.py --interpreter lead lead-1042   # tools invoked via a code interpreter (PTC)
    python main.py --selector lead lead-1042      # LLM tool selection before the loop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from gtm_agent import direct, tools
from gtm_agent.agent import build_agent
from gtm_agent.routing import RoutingTelemetry
from gtm_agent.simulator import (  # noqa: F401  # re-exported for the scenario tests
    _BOUND,
    _disclosed_skill_tools,
    _snapshot,
    _script,
    _read_skill,
    _call,
    _tool_call_sequence,
    AIMessage,
    ScriptedModel,
    SelectorScriptedModel,
    as_selector_model,
    GTMSimulator,
)

OUT_DIR = Path(__file__).resolve().parent / "out"

SKILL_SIZES = {"outbound-followup": 12, "account-intelligence": 15, "customer-health": 12, "meeting-prep": 8}


# ---------------------------------------------------------------------------
# The runner: same graph offline and live, artifacts written to out/
# ---------------------------------------------------------------------------


class GTMAgentRunner:
    """Executes GTM workflows through the deepagents graph and persists artifacts.

    With routed=True, the model per thread is chosen from the catalog
    (glm-5.3-flash / deepseek-4-flash / gpt-6-luna / claude-haiku-5-5)
    by `decision_harness` based on the query; usage, $, and latency land
    in a ledger and are reported after the run.

    With interpreter=True the run uses the deep agent that invokes tools via
    the code interpreter (programmatic tool calling / PTC):
    `build_interpreter_agent` + `CodeInterpreterMiddleware(ptc=...)`.
    With selector=True it uses the deep agent whose tool universe is narrowed
    by `LLMToolSelectorMiddleware` via `build_tool_selector_agent`.
    """

    def __init__(self, live: bool = False, routed: bool = False, interpreter: bool = False, selector: bool = False) -> None:
        self.live = live
        self.routed = routed
        self.interpreter = interpreter
        self.selector = selector
        self.telemetry = RoutingTelemetry()
        self._thread_n = 0
        self.drafts: list[tuple[str, str, str]] = []  # (lead_id, draft, rationale)

    # -- shared plumbing ----------------------------------------------------

    def _invoke(self, model: Any, prompt: str, pinned_skill: str | None = None) -> dict[str, Any]:
        _BOUND.clear()
        tools.reset_fixtures()
        self._thread_n += 1
        thread_id = f"gtm-{self._thread_n}-{prompt.split(':')[0][:20].replace(' ', '-')[:20]}"
        # Pinning a skill injects its SKILL.md before the first model call and
        # discloses its tools from turn 1 — saves the read_file round trip.
        payload: dict[str, Any] = {"messages": [HumanMessage(content=prompt)]}
        if pinned_skill:
            payload["pinned_skills"] = [pinned_skill]
        if self.routed:
            from gtm_agent.routing import AVAILABLE_LABELS, build_routed_agent

            override = {label: model for label in AVAILABLE_LABELS} if model is not None else None
            extra: list[Any] = []
            prompt = None
            if self.interpreter:
                from gtm_agent.agent import GTM_INTERPRETER_SYSTEM_PROMPT, interpreter_middleware

                extra = [interpreter_middleware()]
                prompt = GTM_INTERPRETER_SYSTEM_PROMPT
            if self.selector:
                from langchain.agents.middleware import LLMToolSelectorMiddleware

                max_tools = len(model.selected) if isinstance(model, SelectorScriptedModel) else 3
                extra = [
                    *extra,
                    LLMToolSelectorMiddleware(model=model, max_tools=max_tools or None),
                ]
            agent = build_routed_agent(
                self.telemetry,
                models_override=override,
                extra_middleware=extra,
                system_prompt=prompt,
            )
            config = {"recursion_limit": 60, "configurable": {"thread_id": thread_id}}
            return agent.invoke(payload, config=config)
        if self.interpreter:
            from gtm_agent.agent import build_interpreter_agent

            agent = build_interpreter_agent(model=model)
        elif self.selector:
            from gtm_agent.agent import build_tool_selector_agent

            max_tools = len(model.selected) if isinstance(model, SelectorScriptedModel) else 3
            agent = build_tool_selector_agent(model=model, max_tools=max_tools or None)
        else:
            agent = build_agent(model=model) if model is not None else build_agent()
        return agent.invoke(payload, config={"recursion_limit": 40})

    def _maybe_selector(self, model: ScriptedModel) -> Any:
        """Wrap an offline script as a selector-capable model when --selector is on."""
        return as_selector_model(model) if self.selector else model

    def _print_routing(self) -> None:
        from gtm_agent.routing import friendly

        for d in self.telemetry.decisions[-1:]:
            conf = d["tier_confidence"]
            conf_s = f"{conf:.2f}" if isinstance(conf, float) else "—"
            print(
                f"  routing:    tier={d['tier']} conf={conf_s} "
                f"complexity={d['complexity']} → {friendly(d['model'])} [{d['reason']} via {d['classifier'] or 'sticky'}]"
            )

    @staticmethod
    def _text(content: Any) -> str:
        """Normalize a message's content (str or provider content blocks) to text."""
        if isinstance(content, str):
            return content
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    @staticmethod
    def _save(rel_path: str, content: str) -> Path:
        path = OUT_DIR / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    @staticmethod
    def _print_footer(result: dict[str, Any]) -> None:
        calls = _tool_call_sequence(result)
        if _BOUND:
            baseline = {n for n, d in _BOUND[0] if not d}
            skill_tools = {n for n, d in _BOUND[-1] if not d} - baseline
            print(f"  tools bound: {len(_BOUND[0])} | skill tools disclosed: {len(skill_tools)} | hidden: {50 - len(skill_tools)}")
        print(f"  tool trace:  {' → '.join(calls) if calls else '(none)'}")
        if "eval" in calls:
            from gtm_agent.agent import DEFAULT_INTERPRETER_PTC

            n_evals = sum(1 for c in calls if c == "eval")
            print(
                f"  invocation:  code interpreter — {n_evals} eval program(s), "
                f"{len(DEFAULT_INTERPRETER_PTC)} tools on the PTC allowlist"
            )

    # -- workflows ----------------------------------------------------------

    def process_lead(self, lead_id: str) -> str:
        """Inbound-lead follow-up: do-not-send checks → research → draft (for approval)."""
        print(f"\nProcessing lead {lead_id}{'  (live: ' + ('routed catalog' if self.routed else 'claude-haiku-5-5') + ')' if self.live else ''}")
        prompt = f"New inbound lead just landed in Salesforce: {lead_id}. Run the standard follow-up process for this lead."
        if self.live:
            result = self._invoke(None, prompt, pinned_skill="outbound-followup")
            print("  " + self._text(result["messages"][-1].content).replace("\n", "\n  "))
            self._print_footer(result)
            self._print_routing()
            return "live"

        model, outcome, final = (
            GTMSimulator().interpreter_outbound_script(lead_id, pinned=True)
            if self.interpreter
            else GTMSimulator().outbound_script(lead_id, pinned=True)
        )
        model = self._maybe_selector(model)
        result = self._invoke(model, prompt, pinned_skill="outbound-followup")
        print("  " + final.content.replace("\n", "\n  "))
        self._print_footer(result)
        self._print_routing()

        if outcome == "drafted" and tools._SLACK_DRAFTS:
            d = tools._SLACK_DRAFTS[-1]
            path = self._save(f"drafts/{lead_id}-pending.md", f"# Draft for {lead_id} (pending approval)\n\n## Draft\n\n{d['draft']}\n\n## Rationale\n\n{d['rationale']}\n")
            print(f"  artifact:   {path}")
        elif outcome == "blocked":
            path = self._save(f"drafts/{lead_id}-blocked.md", f"# {lead_id} — no outreach\n\n{final.content}\n")
            print(f"  artifact:   {path}")
        return outcome

    def briefing(self, account: str) -> None:
        """Account-intelligence briefing: usage, hiring, funding → flags and actions."""
        print(f"\nAccount briefing: {account}{'  (live: ' + ('routed catalog' if self.routed else 'claude-haiku-5-5') + ')' if self.live else ''}")
        prompt = f"Give me an account intelligence briefing for {account}."
        if self.live:
            result = self._invoke(None, prompt, pinned_skill="account-intelligence")
            content = self._text(result["messages"][-1].content)
            self._print_footer(result)
            self._print_routing()
        else:
            model, final = (
                GTMSimulator().interpreter_intel_script(account, pinned=True)
                if self.interpreter
                else GTMSimulator().intel_script(account, pinned=True)
            )
            model = self._maybe_selector(model)
            result = self._invoke(model, prompt, pinned_skill="account-intelligence")
            content = final.content
            self._print_footer(result)
        self._print_routing()
        path = self._save(f"briefings/{account.lower().replace(' ', '-')}.md", content)
        print(f"  artifact:   {path}")

    def prep(self, contact: str) -> None:
        """Meeting-prep brief: dossier, last interaction, agenda, style notes."""
        print(f"\nMeeting prep: {contact}{'  (live: ' + ('routed catalog' if self.routed else 'claude-haiku-5-5') + ')' if self.live else ''}")
        prompt = f"Prepare a one-page brief for my upcoming meeting with {contact}."
        if self.live:
            result = self._invoke(None, prompt, pinned_skill="meeting-prep")
            content = self._text(result["messages"][-1].content)
            self._print_footer(result)
            self._print_routing()
        else:
            model, final = (
                GTMSimulator().interpreter_prep_script(contact, pinned=True)
                if self.interpreter
                else GTMSimulator().prep_script(contact, pinned=True)
            )
            model = self._maybe_selector(model)
            result = self._invoke(model, prompt, pinned_skill="meeting-prep")
            content = final.content
            self._print_footer(result)
        self._print_routing()
        path = self._save(f"meeting-preps/{contact.lower().replace(' ', '-')}.md", content)
        print(f"  artifact:   {path}")

    def inbox(self) -> None:
        """Drafts pending rep approval (the human-in-the-loop step)."""
        print("\n📥 Drafts pending rep approval")
        drafts_dir = OUT_DIR / "drafts"
        saved = sorted(drafts_dir.glob("*-pending.md")) if drafts_dir.exists() else []
        if not saved:
            print("  (empty — run a lead workflow first)")
            return
        for p in saved:
            print(f"  • {p.stem}")

    # -- defined (no-agent-loop) workflows ----------------------------------

    def run_direct(self, workflow: str, target: str) -> str:
        """Invoke GTM tools in a *defined* way — no agent loop, no LLM in the loop.

        Thin CLI-facing wrapper over the deterministic runners in
        ``gtm_agent.direct`` (see that module's docstring for the design).
        """
        tools.reset_fixtures()
        if workflow == "lead":
            result = direct.run_outbound_followup(target)
            if result["outcome"] == "drafted":
                d = tools._SLACK_DRAFTS[-1]
                path = self._save(f"drafts/{target}-pending.md", f"# Draft for {target} (pending approval)\n\n## Draft\n\n{d['draft']}\n\n## Rationale\n\n{d['rationale']}\n")
                return f"✅ Defined outbound follow-up for {target}: draft queued -> {path.name}\n\n## Rationale\n\n{d['rationale']}"
            path = self._save(f"drafts/{target}-blocked.md", f"# {target} — no outreach\n\n{result['reason']}\n")
            return f"🚫 Defined outbound follow-up for {target}: blocked ({result['reason']})\n\nSaved: {path.name}"
        if workflow == "intel":
            result = direct.run_account_briefing(target)
            path = self._save(f"briefings/{target.lower().replace(' ', '-')}.md", result["briefing"])
            return f"📋 Defined account briefing for {target}: {result['outcome']}\n\n{result['briefing']}\n\nSaved: {path.name}"
        if workflow == "prep":
            result = direct.run_meeting_prep(target)
            path = self._save(f"meeting-preps/{target.lower().replace(' ', '-')}.md", result["briefing"])
            return f"🧑‍💼 Defined meeting prep for {target}: {result['outcome']}\n\n{result['briefing']}\n\nSaved: {path.name}"
        raise ValueError(f"unknown direct workflow: {workflow!r}")

    # -- full day -----------------------------------------------------------

    def run_day(self) -> int:
        """A simulated day: process the inbound queue, brief accounts, prep meetings."""
        print("☀️  GTM agent — simulated day " + ("(live: claude-haiku-5-5)" if self.live else "(offline simulator, real deepagents harness)"))
        for lead_id in ("lead-1042", "lead-1043", "lead-1044"):
            self.process_lead(lead_id)
        self.briefing("Helix Health")
        self.prep("Priya Sharma")
        self.inbox()
        print("\nAll artifacts written to out/. Nothing was sent — every draft waits for rep approval. ✅")
        self.telemetry.print_report()
        return 0


# ---------------------------------------------------------------------------
# Rule-based scenario tests (regression suite)
# ---------------------------------------------------------------------------


class GTMAgentTestRunner:
    """Rule-based scenario tests for the GTM agent, offline by default."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.results: list[tuple[str, bool, str]] = []
        self.skipped: list[tuple[str, str]] = []

    def test_tool_universe(self) -> None:
        """The agent simulates a 50-tool stack; nothing leaks before a skill is read."""
        assert len(tools.ALL_TOOLS) == 50, f"expected 50 tools, got {len(tools.ALL_TOOLS)}"
        deferred = {t.name for t in tools.ALL_TOOLS if (t.extras or {}).get("defer_loading")}
        assert len(deferred) == 47, f"expected 47 deferred tools, got {len(deferred)}"
        for skill, count in SKILL_SIZES.items():
            tools_in_skill = self._skill_tools(skill)
            assert len(tools_in_skill) == count, f"{skill} binds {len(tools_in_skill)} tools, expected {count}"
            assert tools_in_skill <= deferred, f"{skill} names non-deferred tools: {tools_in_skill - deferred}"

        tools.reset_fixtures()
        runner = GTMAgentRunner()
        model, final = GTMSimulator().prep_script("Priya Sharma")
        result = runner._invoke(model, "Prep me for a meeting with Priya Sharma.")
        disclosed = _disclosed_skill_tools()
        assert len(disclosed) == SKILL_SIZES["meeting-prep"], f"expected 8 disclosed, got {len(disclosed)}"
        assert not ({"lookup_lead", "get_open_tickets"} & disclosed)
        assert any(m.name == "get_person_dossier" for m in result["messages"] if m.type == "tool")
        self._record("tool universe", True, "50 tools (47 deferred); prep run disclosed exactly its 8, hid 42.")

    def test_do_not_send(self) -> None:
        """A lead with an open SEV-2 ticket: do-not-send check runs FIRST and NO draft is queued."""
        tools.reset_fixtures()
        model, _, _ = GTMSimulator().outbound_script("lead-1044")
        result = GTMAgentRunner()._invoke(model, "Follow up on lead-1044.")
        calls = _tool_call_sequence(result)
        assert calls[0] == "read_file" and calls[1] == "check_contact_history", f"do-not-send check not first: {calls}"
        assert "draft_slack_followup" not in calls, f"drafted for a do-not-send lead: {calls}"
        assert tools._SLACK_DRAFTS == [], "a draft was queued despite do-not-send"
        self._record("do-not-send gate", True, "checked first, drafted nothing for blocked lead.")

    def test_outbound_playbook_order(self) -> None:
        """Clean lead: do-not-send check → research → draft queued as pending_approval."""
        tools.reset_fixtures()
        model, _, _ = GTMSimulator().outbound_script("lead-1042")
        result = GTMAgentRunner()._invoke(model, "Follow up on lead-1042.")
        calls = _tool_call_sequence(result)
        assert calls == [
            "read_file",
            "check_contact_history",
            "lookup_lead",
            "web_search_company",
            "draft_slack_followup",
        ], f"playbook order violated: {calls}"
        assert len(tools._SLACK_DRAFTS) == 1 and tools._SLACK_DRAFTS[0]["status"] == "pending_approval"
        assert len(_disclosed_skill_tools()) == SKILL_SIZES["outbound-followup"]

        # Pinned skill: the same playbook without the read_file round trip.
        tools.reset_fixtures()
        model, _, _ = GTMSimulator().outbound_script("lead-1042", pinned=True)
        result = GTMAgentRunner()._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
        pinned_calls = _tool_call_sequence(result)
        assert pinned_calls == [
            "check_contact_history",
            "lookup_lead",
            "web_search_company",
            "draft_slack_followup",
        ], f"pinned playbook should skip read_file: {pinned_calls}"
        self._record("outbound playbook", True, "check → research → draft; queued as pending_approval.")

    def test_skill_isolation(self) -> None:
        """An account-intelligence run never discloses outbound-only tools."""
        tools.reset_fixtures()
        model, _ = GTMSimulator().intel_script("Helix Health")
        GTMAgentRunner()._invoke(model, "Account briefing for Helix Health.")
        disclosed = _disclosed_skill_tools()
        assert len(disclosed) == SKILL_SIZES["account-intelligence"], f"expected 15 disclosed, got {len(disclosed)}"
        outbound = {"lookup_lead", "check_contact_history", "draft_slack_followup", "get_open_tickets", "get_person_dossier"}
        assert not (outbound & disclosed), f"outbound/other-skill tools leaked: {outbound & disclosed}"
        self._record("skill isolation", True, "intel run disclosed its 15; outbound tools stayed hidden.")

    def test_routing_and_ledger(self) -> None:
        """Routed runs: query-based model choice, sticky per thread, ledger entries."""
        tools.reset_fixtures()
        runner = GTMAgentRunner(routed=True, live=False)
        model, _, _ = GTMSimulator().outbound_script("lead-1042")
        result = runner._invoke(model, "Follow up on lead-1042.")

        # Routing must not break the playbook...
        assert _tool_call_sequence(result) == [
            "read_file", "check_contact_history", "lookup_lead",
            "web_search_company", "draft_slack_followup",
        ], f"playbook changed under routing: {_tool_call_sequence(result)}"
        # ...the query was classified by the GTM heuristic and routed...
        classified = [d for d in runner.telemetry.decisions if not d["sticky"]]
        assert classified and classified[0]["classifier"].startswith("heuristic"), classified
        assert classified[0]["model"].startswith("openai/"), classified[0]  # balanced → gpt-6-luna
        # ...and every call landed in the ledger with tokens/cost/latency fields.
        ledger = runner.telemetry.ledger
        assert ledger is not None and len(ledger.entries) > 0
        entry = ledger.entries[0]
        assert entry.input_tokens == 0 and entry.cost == 0.0  # offline fake: no billed usage
        assert entry.latency_ms >= 0 and entry.reason in {"classified", "sticky"}
        totals = ledger.totals()
        assert totals["grand"]["calls"] == len(ledger.entries)
        self._record(
            "model routing",
            True,
            f"query routed to {classified[0]['model']} (balanced), sticky after call 1, {len(ledger.entries)} ledger entries.",
        )

    def test_defined_invocation(self) -> None:
        """Direct mode: the playbook runs with zero model turns and the same gates."""
        from gtm_agent import direct

        tools.reset_fixtures()
        drafted = direct.run_outbound_followup("lead-1042")
        assert drafted["outcome"] == "drafted" and drafted["draft_queued"] is True, drafted
        assert len(tools._SLACK_DRAFTS) == 1
        assert tools._SLACK_DRAFTS[0]["status"] == "pending_approval"

        tools.reset_fixtures()
        assert direct.run_outbound_followup("lead-1044")["outcome"] == "blocked"
        assert direct.run_outbound_followup("lead-1043")["outcome"] == "blocked"
        assert tools._SLACK_DRAFTS == [], "direct mode queued a draft for a blocked lead"

        assert "Account briefing" in direct.run_account_briefing("Helix Health")["briefing"]
        assert "Meeting prep" in direct.run_meeting_prep("Priya Sharma")["briefing"]
        self._record(
            "defined invocation",
            True,
            "no agent loop: gates enforced, drafts queued, intel + prep produced.",
        )

    def test_interpreter_playbook(self) -> None:
        """Interpreter mode: the same gates + side effects as one eval program."""
        tools.reset_fixtures()
        model, outcome, _ = GTMSimulator().interpreter_outbound_script("lead-1042", pinned=True)
        result = GTMAgentRunner(interpreter=True)._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
        calls = _tool_call_sequence(result)
        assert outcome == "drafted"
        assert calls == ["eval"], f"expected one eval program, got {calls}"
        assert len(tools._SLACK_DRAFTS) == 1, "eval program did not queue a draft"
        queued = tools._SLACK_DRAFTS[0]
        assert queued["status"] == "pending_approval"

        # The JS-authored draft must match the Python-side playbook draft —
        # proof the composition in code did not change the outcome.
        lead = tools._LEADS["lead-1042"]
        research = tools._WEB_RESEARCH[lead["company"]]
        expected, _ = GTMSimulator._draft_for(lead, research)
        assert queued["draft"] == expected, f"interpreter draft drifted:\n{queued['draft']!r}\n{expected!r}"

        # Blocked leads stay blocked inside the program (no draft, one eval).
        tools.reset_fixtures()
        model, outcome, _ = GTMSimulator().interpreter_outbound_script("lead-1044", pinned=True)
        result = GTMAgentRunner(interpreter=True)._invoke(model, "Follow up on lead-1044.", pinned_skill="outbound-followup")
        assert outcome == "blocked" and _tool_call_sequence(result) == ["eval"]
        assert tools._SLACK_DRAFTS == [], "interpreter drafted for a do-not-send lead"

        # Intel + prep also collapse to a single eval each.
        tools.reset_fixtures()
        model, _ = GTMSimulator().interpreter_intel_script("Helix Health", pinned=True)
        result = GTMAgentRunner(interpreter=True)._invoke(model, "Account briefing for Helix Health.", pinned_skill="account-intelligence")
        assert _tool_call_sequence(result) == ["eval"]

        tools.reset_fixtures()
        model, _ = GTMSimulator().interpreter_prep_script("Priya Sharma", pinned=True)
        result = GTMAgentRunner(interpreter=True)._invoke(model, "Prep me for a meeting with Priya Sharma.", pinned_skill="meeting-prep")
        assert _tool_call_sequence(result) == ["eval"]

        # Interpreter + routing compose: still one eval, routing still logged.
        tools.reset_fixtures()
        routed = GTMAgentRunner(routed=True, interpreter=True)
        model, _, _ = GTMSimulator().interpreter_outbound_script("lead-1042", pinned=True)
        result = routed._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
        assert _tool_call_sequence(result) == ["eval"]
        assert routed.telemetry.ledger is not None and routed.telemetry.ledger.entries

        self._record(
            "interpreter PTC",
            True,
            "1 eval per workflow; same gates + identical draft as the loop playbook.",
        )

    def test_tool_selector(self) -> None:
        """Tool selector: the agent loop only sees the tools the selector picked."""
        tools.reset_fixtures()
        model = SelectorScriptedModel(
            messages=iter(
                [
                    _call("check_contact_history", {"lead_id": "lead-1042"}, "c1"),
                    AIMessage(content="Checked history."),
                ]
            ),
            selected=["check_contact_history"],
        )
        result = GTMAgentRunner(selector=True)._invoke(model, "Follow up on lead-1042.")
        bound = {name for name, _deferred in _BOUND[-1]}
        assert bound == {"check_contact_history"}, f"selector did not narrow the toolset: {sorted(bound)}"
        assert _tool_call_sequence(result) == ["check_contact_history"]
        self._record(
            "tool selector",
            True,
            f"LLM selector bound {len(bound)} of 50 tools (chosen before the loop).",
        )

    @staticmethod
    def _skill_tools(skill: str) -> set[str]:
        frontmatter = open(f"skills/{skill}/SKILL.md").read().split("---")[1]
        for line in frontmatter.splitlines():
            if line.strip().startswith("include_tools:"):
                return set(line.split(":", 1)[1].split())
        raise AssertionError(f"no include_tools in skills/{skill}/SKILL.md")

    def _record_skipped(self, name: str, detail: str) -> None:
        self.skipped.append((name, detail))
        print(f"  ⏭ {name:20s} skipped — {detail}")

    def _record(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))
        mark = "✅" if passed else "❌"
        print(f"  {mark} {name:20s} {detail}")
        if self.verbose:
            for i, snapshot in enumerate(_BOUND, 1):
                hidden = sum(1 for _, d in snapshot if d)
                print(f"       model call {i}: {len(snapshot)} tools bound, {hidden} still hidden")

    def run_all(self) -> int:
        print("\nGTM agent — offline scenario tests (50-tool simulated stack)\n" + "=" * 64)
        for scenario in [
            self.test_tool_universe,
            self.test_do_not_send,
            self.test_outbound_playbook_order,
            self.test_skill_isolation,
            self.test_routing_and_ledger,
            self.test_defined_invocation,
            self.test_interpreter_playbook,
            self.test_tool_selector,
        ]:
            try:
                scenario()
            except ImportError as exc:  # optional extra missing (e.g. QuickJS)
                self._record_skipped(scenario.__name__.removeprefix("test_"), str(exc).splitlines()[0])
            except AssertionError as exc:
                self._record(scenario.__name__, False, str(exc)[:120])
        print("=" * 64)
        passed = sum(1 for _, ok, _ in self.results if ok)
        suffix = f" ({len(self.skipped)} skipped: optional extra not installed)" if self.skipped else ""
        print(f"{passed}/{len(self.results)} scenarios passed{suffix}\n")
        return 0 if passed == len(self.results) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="run on real models (keys from .env)")
    parser.add_argument("--route", action="store_true", help="route each run to the best model (glm-5.3-flash / deepseek-4-flash / gpt-6-luna / claude-haiku-5-5) via decision_harness; logs tokens, $ and latency")
    parser.add_argument("--interpreter", action="store_true", help="run the agent via the code interpreter (programmatic tool calling / PTC allowlist)")
    parser.add_argument("--selector", action="store_true", help="run the agent with the LLM tool selector middleware")
    sub = parser.add_subparsers(dest="cmd")

    route_p = sub.add_parser("route", help="show which model a query routes to (works offline via heuristic; add --decisions to use the Decisions API)")
    route_p.add_argument("query", nargs="+")
    route_p.add_argument("--decisions", action="store_true", help="force the Decisions API classifier (needs DECISIONS_API_KEY/OPENAI_API_KEY)")

    lead_p = sub.add_parser("lead", help="process one inbound lead (e.g. lead-1042)")
    lead_p.add_argument("lead_id")
    brief_p = sub.add_parser("briefing", help="account-intelligence briefing for an account")
    brief_p.add_argument("account", nargs="?", default="Helix Health")
    prep_p = sub.add_parser("prep", help='meeting-prep brief (e.g. "Priya Sharma")')
    prep_p.add_argument("contact")
    sub.add_parser("inbox", help="drafts pending rep approval")
    test_p = sub.add_parser("test", help="run the rule-based scenario tests")
    test_p.add_argument("-v", "--verbose", action="store_true")

    direct_p = sub.add_parser("direct", help="invoke GTM tools in a defined way (no agent loop)")
    direct_sub = direct_p.add_subparsers(dest="direct_cmd", required=True)

    dl = direct_sub.add_parser("lead", help="outbound follow-up via defined tool invocation")
    dl.add_argument("lead_id")

    di = direct_sub.add_parser("intel", help="account-intelligence briefing via defined tool invocation")
    di.add_argument("account", nargs="?", default="Helix Health")

    dp = direct_sub.add_parser("prep", help="meeting prep via defined tool invocation")
    dp.add_argument("contact")

    args = parser.parse_args()

    if args.cmd == "route":
        from gtm_agent.routing import classify_query, friendly

        query = " ".join(args.query)
        d = classify_query(query, use_decisions=True if args.decisions else None)
        c = d["tier_confidence"]
        print(f"\nquery:      {query}")
        print(f"routed to:  {friendly(d['chosen_model'])}  (tier={d['tier']}, reason={d['reason']})")
        print(f"classifier: {d['classifier']} in {d['classification_latency_ms']}ms")
        print(f"confidence: {c if c is None else round(c, 2)} | complexity: {d['complexity']} | needs-planning: {d['needs_planning']}")
        for e in d["events"]:
            print(f"  event: {e}")
        return 0

    runner = GTMAgentRunner(live=args.live, routed=args.route, interpreter=args.interpreter, selector=args.selector)

    if args.cmd == "lead":
        runner.process_lead(args.lead_id)
        runner.telemetry.print_report()
        return 0
    if args.cmd == "briefing":
        runner.briefing(args.account)
        runner.telemetry.print_report()
        return 0
    if args.cmd == "prep":
        runner.prep(args.contact)
        runner.telemetry.print_report()
        return 0
    if args.cmd == "inbox":
        runner.inbox()
        return 0
    if args.cmd == "test":
        return GTMAgentTestRunner(verbose=args.verbose).run_all()

    if args.cmd == "direct":
        workflow = args.direct_cmd
        target = getattr(args, "lead_id", None) or getattr(args, "account", None) or getattr(args, "contact", None)
        print(runner.run_direct(workflow, target))
        return 0

    runner.run_day()
    return 0


if __name__ == "__main__":
    sys.exit(main())
