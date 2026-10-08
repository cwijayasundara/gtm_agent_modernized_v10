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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from gtm_agent import tools
from gtm_agent.agent import build_agent
from gtm_agent.routing import RoutingTelemetry

OUT_DIR = Path(__file__).resolve().parent / "out"

SKILL_SIZES = {"outbound-followup": 12, "account-intelligence": 15, "customer-health": 12, "meeting-prep": 8}

# ---------------------------------------------------------------------------
# Instrumented models
# ---------------------------------------------------------------------------

_BOUND: list[list[tuple[str, bool]]] = []


def _snapshot(bound_tools: list[Any]) -> list[tuple[str, bool]]:
    out = []
    for t in bound_tools:
        if isinstance(t, dict):
            out.append((t.get("name", "?"), False))
        else:
            out.append((t.name, bool((t.extras or {}).get("defer_loading"))))
    return sorted(out)


class ScriptedModel(GenericFakeChatModel):
    """Scripted model that records which tools the harness binds per call."""

    def bind_tools(self, bound_tools, **kwargs):  # type: ignore[no-untyped-def]
        _BOUND.append(_snapshot(bound_tools))
        return self  # scripted model ignores tool schemas


def _script(*turns: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(list(turns)))


def _read_skill(skill: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"file_path": f"/skills/{skill}/SKILL.md"}, "id": call_id, "type": "tool_call"}],
    )


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _tool_call_sequence(result: dict[str, Any]) -> list[str]:
    return [
        call["name"]
        for message in result["messages"]
        if message.type == "ai"
        for call in message.tool_calls
    ]


def _disclosed_skill_tools() -> set[str]:
    """Skill-bound tools disclosed by the last model call (baseline removed)."""
    baseline = {name for name, deferred in _BOUND[0] if not deferred}
    return {name for name, deferred in _BOUND[-1] if not deferred} - baseline


# ---------------------------------------------------------------------------
# The offline "LLM": turns mock fixtures into scripted agent turns
# ---------------------------------------------------------------------------


class GTMSimulator:
    """Data-driven stand-in for Claude Haiku that follows each skill's playbook."""

    def outbound_script(self, lead_id: str, *, pinned: bool = False) -> tuple[ScriptedModel, str, AIMessage]:
        """Build the scripted run for the outbound-followup playbook."""
        history = tools._CONTACT_HISTORY.get(lead_id, {})
        lead = tools._LEADS.get(lead_id, {})
        reason = history.get("summary", "unknown lead")

        if history.get("do_not_send"):
            final = AIMessage(
                content=(
                    f"🚫 No draft for {lead_id} ({lead.get('contact', '?')} at {lead.get('company', '?')}):\n"
                    f"   {reason}\n"
                    "   Recommend a human touch instead of automated outreach."
                )
            )
            turns = [] if pinned else [_read_skill("outbound-followup", "c1")]
            turns.append(_call("check_contact_history", {"lead_id": lead_id}, "c2"))
            script = _script(*turns, final)
            return script, "blocked", final

        if any(o.get("days_ago", 99) <= 3 for o in history.get("prior_outreach", [])):
            final = AIMessage(
                content=(
                    f"🚫 No draft for {lead_id}: {reason}\n"
                    "   Coordinate with the teammate instead of a second touch."
                )
            )
            turns = [] if pinned else [_read_skill("outbound-followup", "c1")]
            turns.append(_call("check_contact_history", {"lead_id": lead_id}, "c2"))
            script = _script(*turns, final)
            return script, "blocked", final

        # Clear to draft: research, then a relationship-aware draft.
        company = lead.get("company", "")
        research = tools._WEB_RESEARCH.get(company, "")
        draft, rationale = self._draft_for(lead, research)
        final = AIMessage(
            content=(
                f"✅ Draft queued for {lead_id} ({lead.get('contact')} at {company}), "
                f"pending rep approval.\n\nAngle: {rationale}"
            )
        )
        script = _script(
            _read_skill("outbound-followup", "c1"),
            _call("check_contact_history", {"lead_id": lead_id}, "c2"),
            _call("lookup_lead", {"lead_id": lead_id}, "c3"),
            _call("web_search_company", {"company": company}, "c4"),
            _call("draft_slack_followup", {"lead_id": lead_id, "draft": draft, "rationale": rationale}, "c5"),
            final,
        )
        return script, "drafted", final

    @staticmethod
    def _draft_for(lead: dict[str, Any], research: str) -> tuple[str, str]:
        contact, company = lead.get("contact", "?"), lead.get("company", "?")
        if lead.get("account_type") == "customer":
            draft = (
                f"Hi {contact.split()[0]} — the AI clinical-notes launch looks great. "
                "Your usage has grown 38% MoM and audit logs keep coming up; "
                "worth 20 minutes to scope the Enterprise tier?"
            )
            rationale = "Customer → expansion angle, anchored on usage growth and stated needs."
        else:  # cold
            draft = (
                f"Hi {contact.split()[0]} — saw {company} is pushing into LLM-powered BI copilots "
                "and hiring AI engineers. Teams at that stage usually hit eval/tooling walls fast. "
                "We help with exactly that — open to a short note-exchange on your eval approach?"
            )
            rationale = "Cold → brief, research-backed opener referencing their AI push and hiring."
        return draft, rationale

    def intel_script(self, account: str, *, pinned: bool = False) -> tuple[ScriptedModel, AIMessage]:
        usage = tools._USAGE_SIGNALS.get(account, {})
        hiring = tools._HIRING.get(account, "No notable AI hiring activity.")
        funding = tools._FUNDING.get(account, "No recent funding news.")
        credits = usage.get("credits_remaining_pct", "n/a")
        final = AIMessage(
            content=(
                f"# Account briefing — {account}\n\n"
                f"## Signals\n"
                f"- Usage: {usage.get('weekly_active_users_trend', 'no data')} WAU trend, "
                f"{usage.get('sdk_installs_last_7d', '—')} SDK installs last 7d\n"
                f"- Hiring: {hiring}\n"
                f"- Funding: {funding}\n\n"
                f"## Flags\n"
                f"- ⚠️ Credits at {credits}% — pre-renewal expansion conversations should start now\n"
                f"- Renewal in {usage.get('renewal_in_days', '—')} days\n\n"
                f"## Recommended action\n"
                f"Reach out to the most engaged user about the Enterprise tier; "
                f"address credits before they run out."
            )
        )
        head = [] if pinned else [_read_skill("account-intelligence", "c1")]
        script = _script(
            *head,
            _call("query_product_usage", {"account": account}, "c2"),
            _call("get_hiring_signals", {"company": account}, "c3"),
            _call("get_funding_news", {"company": account}, "c4"),
            final,
        )
        return script, final

    def prep_script(self, contact: str, *, pinned: bool = False) -> tuple[ScriptedModel, AIMessage]:
        dossier = tools._DOSSIERS.get(contact, {})
        last = tools._LAST_INTERACTIONS.get(contact, {})
        priorities = ", ".join(dossier.get("priorities", [])) or "—"
        final = AIMessage(
            content=(
                f"# Meeting prep — {contact}\n\n"
                f"**Role**: {dossier.get('role', '?')}\n"
                f"**Background**: {dossier.get('background', '—')}\n"
                f"**Current priorities**: {priorities}\n"
                f"**Last interaction**: {last.get('type', '—')} {last.get('when', '')} — {last.get('summary', '')}\n\n"
                f"## Suggested agenda\n"
                f"1. Recap where we left off ({last.get('type', 'last touch')})\n"
                f"2. Their current priority: {priorities.split(',')[0] if priorities != '—' else 'open'}\n"
                f"3. One concrete next step\n\n"
                f"**Style note**: {dossier.get('style', '—')}"
            )
        )
        head = [] if pinned else [_read_skill("meeting-prep", "c1")]
        script = _script(
            *head,
            _call("get_person_dossier", {"query": contact}, "c2"),
            _call("get_last_interaction", {"query": contact}, "c3"),
            final,
        )
        return script, final


# ---------------------------------------------------------------------------
# The runner: same graph offline and live, artifacts written to out/
# ---------------------------------------------------------------------------


class GTMAgentRunner:
    """Executes GTM workflows through the deepagents graph and persists artifacts.

    With routed=True, the model per thread is chosen from the catalog
    (glm-5.3-flash / deepseek-4-flash / gpt-6-luna / claude-haiku-5-5)
    by `decision_harness` based on the query; usage, $, and latency land
    in a ledger and are reported after the run.
    """

    def __init__(self, live: bool = False, routed: bool = False) -> None:
        self.live = live
        self.routed = routed
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
            agent = build_routed_agent(self.telemetry, models_override=override)
            config = {"recursion_limit": 60, "configurable": {"thread_id": thread_id}}
            return agent.invoke(payload, config=config)
        agent = build_agent(model=model) if model is not None else build_agent()
        return agent.invoke(payload, config={"recursion_limit": 40})

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

        model, outcome, final = GTMSimulator().outbound_script(lead_id, pinned=True)
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
            model, final = GTMSimulator().intel_script(account, pinned=True)
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
            model, final = GTMSimulator().prep_script(contact, pinned=True)
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

    @staticmethod
    def _skill_tools(skill: str) -> set[str]:
        frontmatter = open(f"skills/{skill}/SKILL.md").read().split("---")[1]
        for line in frontmatter.splitlines():
            if line.strip().startswith("include_tools:"):
                return set(line.split(":", 1)[1].split())
        raise AssertionError(f"no include_tools in skills/{skill}/SKILL.md")

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
        ]:
            try:
                scenario()
            except AssertionError as exc:
                self._record(scenario.__name__, False, str(exc)[:120])
        print("=" * 64)
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"{passed}/{len(self.results)} scenarios passed\n")
        return 0 if passed == len(self.results) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="run on real models (keys from .env)")
    parser.add_argument("--route", action="store_true", help="route each run to the best model (glm-5.3-flash / deepseek-4-flash / gpt-6-luna / claude-haiku-5-5) via decision_harness; logs tokens, $ and latency")
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

    runner = GTMAgentRunner(live=args.live, routed=args.route)

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

    runner.run_day()
    return 0


if __name__ == "__main__":
    sys.exit(main())
