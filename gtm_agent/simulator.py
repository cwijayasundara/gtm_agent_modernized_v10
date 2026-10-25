"""The offline "LLM" harness: scripted deepagents runs + the data-driven simulator.

The sample GTM agent runs its real deepagents graph end to end; only the chat
model is simulated offline (a data-driven scripted model built from the mock
fixtures in ``gtm_agent.tools``).

Usage:
    from gtm_agent import simulator

    script = simulator.GTMSimulator().outbound_script("lead-1042", pinned=True)
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from . import tools

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
# Programmatic tool calling: the JS a real model authors for the interpreter
# ---------------------------------------------------------------------------
#
# With ``CodeInterpreterMiddleware(ptc=[...])`` the model writes one JS program
# per workflow step instead of firing a serial chain of tool calls. The mocked
# tools return JSON *strings*, so the script parses the structured ones; that
# is a property of this mock stack, not of PTC itself.

_INTERPRETER_OUTBOUND_JS = '''(async () => {
  const leadId = "__LEAD_ID__";

  // Gate first: never draft on top of a do-not-send signal or a fresh touch.
  const history = JSON.parse(await tools.checkContactHistory({ lead_id: leadId }));
  if (history.do_not_send === true) {
    return { outcome: "blocked", lead_id: leadId, reason: history.summary };
  }
  const touchedRecently = (history.prior_outreach || []).some((o) => o.days_ago <= 3);
  if (touchedRecently) {
    return { outcome: "blocked", lead_id: leadId, reason: history.summary };
  }

  // Research batch: one Promise.all instead of three model turns.
  const lead = JSON.parse(await tools.lookupLead({ lead_id: leadId }));
  const [research, gong] = await Promise.all([
    tools.webSearchCompany({ company: lead.company }),
    tools.getGongTranscripts({ lead_id: leadId }),
  ]);

  // Draft + queue for rep approval, all in the same program.
  const first = lead.contact.split(" ")[0];
  const company = lead.company;
  const isCustomer = lead.account_type === "customer";
  const draft = isCustomer
    ? "Hi " + first + " — the AI clinical-notes launch looks great. Your usage has grown 38% MoM and audit logs keep coming up; worth 20 minutes to scope the Enterprise tier?"
    : "Hi " + first + " — saw " + company + " is pushing into LLM-powered BI copilots and hiring AI engineers. Teams at that stage usually hit eval/tooling walls fast. We help with exactly that — open to a short note-exchange on your eval approach?";
  const rationale = isCustomer
    ? "Customer → expansion angle, anchored on usage growth and stated needs."
    : "Cold → brief, research-backed opener referencing their AI push and hiring.";

  // The eval bridge is read-only: return the proposal, then queue it through
  // the normal HITL-gated draft_slack_followup tool.
  return { outcome: "drafted", lead_id: leadId, draft: draft, rationale: rationale,
           research: research, gong: gong, contact: lead.contact, company: company };
})()'''

_INTERPRETER_INTEL_JS = '''(async () => {
  const account = "__ACCOUNT__";
  const results = await Promise.all([
    tools.queryProductUsage({ account: account }),
    tools.getHiringSignals({ company: account }),
    tools.getFundingNews({ company: account }),
  ]);
  return { account: account, usage: JSON.parse(results[0]), hiring: results[1], funding: results[2] };
})()'''

_INTERPRETER_PREP_JS = '''(async () => {
  const contact = "__CONTACT__";
  const results = await Promise.all([
    tools.getPersonDossier({ query: contact }),
    tools.getLastInteraction({ query: contact }),
  ]);
  return { contact: contact, dossier: JSON.parse(results[0]), last: JSON.parse(results[1]) };
})()'''


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
            *([] if pinned else [_read_skill("outbound-followup", "c1")]),
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

    # -- interpreter variants: one model-authored eval per workflow ----------
    #
    # These are the same playbooks as above, but expressed the way a model
    # writes them for `CodeInterpreterMiddleware(ptc=[...])`: a single `eval`
    # program that gates, batches, composes, and queues — instead of a serial
    # chain of tool calls. Only the tool invocation *shape* differs; the gates
    # and side effects are identical (asserted in the test suite).

    def interpreter_outbound_script(
        self, lead_id: str, *, pinned: bool = False
    ) -> tuple[ScriptedModel, str, AIMessage]:
        """Outbound follow-up as one read-only JS program plus a normal draft call."""
        _, outcome, final = self.outbound_script(lead_id, pinned=True)
        head = [] if pinned else [_read_skill("outbound-followup", "c1")]
        code = _INTERPRETER_OUTBOUND_JS.replace("__LEAD_ID__", lead_id)
        turns = [*head, _call("eval", {"code": code}, "c2")]
        if outcome == "drafted":
            lead = tools._LEADS[lead_id]
            research = tools._WEB_RESEARCH.get(lead["company"], "")
            draft, rationale = self._draft_for(lead, research)
            turns.append(
                _call(
                    "draft_slack_followup",
                    {"lead_id": lead_id, "draft": draft, "rationale": rationale},
                    "c3",
                )
            )
        turns.append(final)
        return _script(*turns), outcome, final

    def interpreter_intel_script(
        self, account: str, *, pinned: bool = False
    ) -> tuple[ScriptedModel, AIMessage]:
        """Account briefing as one JS program: three lookups in one Promise.all."""
        _, final = self.intel_script(account, pinned=True)
        head = [] if pinned else [_read_skill("account-intelligence", "c1")]
        code = _INTERPRETER_INTEL_JS.replace("__ACCOUNT__", account)
        script = _script(*head, _call("eval", {"code": code}, "c2"), final)
        return script, final

    def interpreter_prep_script(
        self, contact: str, *, pinned: bool = False
    ) -> tuple[ScriptedModel, AIMessage]:
        """Meeting prep as one JS program: dossier + last interaction in parallel."""
        _, final = self.prep_script(contact, pinned=True)
        head = [] if pinned else [_read_skill("meeting-prep", "c1")]
        code = _INTERPRETER_PREP_JS.replace("__CONTACT__", contact)
        script = _script(*head, _call("eval", {"code": code}, "c2"), final)
        return script, final


class SelectorScriptedModel(ScriptedModel):
    """Scripted model that also answers the tool selector's structured call.

    ``LLMToolSelectorMiddleware`` asks a model (the same one by default) to pick
    relevant tool names before the agent loop runs. A real model does that with
    structured output; this stand-in returns a fixed selection so the offline
    tests can assert which tools the agent loop actually sees.
    """

    selected: list[str] = []  # type: ignore[assignment]

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        from langchain_core.runnables import RunnableLambda

        selection = list(self.selected)
        return RunnableLambda(lambda _input: {"tools": selection})


def as_selector_model(script: ScriptedModel, selected: list[str] | None = None) -> SelectorScriptedModel:
    """Wrap a scripted run as a selector-capable model for offline `--selector` runs.

    The selection defaults to the tools the script actually calls (minus
    ``read_file``, which the CLI pins away), so the offline demo shows the
    selector narrowing the catalog to exactly the workflow's tool surface.
    """
    turns = list(script.messages)
    if selected is None:
        selected = sorted(
            {
                call["name"]
                for message in turns
                if message.type == "ai"
                for call in message.tool_calls
                if call["name"] != "read_file"
            }
        )
    return SelectorScriptedModel(messages=iter(turns), selected=selected)
