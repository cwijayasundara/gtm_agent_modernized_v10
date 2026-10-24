"""Defined tool invocation for the GTM workflows — no agent loop required.

When an agent carries many tools, the *agent loop* makes the model guess which
tool to use next (and when to stop). The deepagents "binding tools to skills"
story (LangChain blog: *Revamping Skills in Deep Agents*) keeps every skill's
toolset hidden until the skill is read, which solves discovery — but it still
means each workflow runs as a model-driven sequence of tool calls.

This module is the other half of the answer, and the one this sample ships as
an **example**: the application owns the orchestration.

Two shapes of "invoke tools in a defined way" live here:

1. **Deterministic workflows** (`run_outbound_followup`, `run_account_briefing`,
   `run_meeting_prep`) — the playbook steps are authored as code and the mocked
   tools are invoked directly, in the defined order, with the same gates the
   agent loop enforces. No LLM in the loop; no model deciding anything.

2. **Agent-side, code-composed tool calls** — the deep agent can be given a
   code interpreter (`CodeInterpreterMiddleware` + programmatic tool calling,
   LangChain blog: *Give Your Agents an Interpreter*) so the model writes JS like
   `const r = await tools.lookupLead({ lead_id }); r;` and the harness bridges
   the allowlisted tools. Tool calls become code-level composition instead of
   serial model-mediated steps.

The deterministic runners below are what the offline simulator and the live
agent both fold into, so "direct mode" and "agent mode" can't drift apart.

Run the deterministic workflows from the CLI:

    python main.py direct outbound lead-1042
    python main.py direct intel "Helix Health"
    python main.py direct prep "Priya Sharma"
"""

from __future__ import annotations

import json
from typing import Any

from . import tools
from .simulator import GTMSimulator

# Tool registry for defined invocations: name -> bound BaseTool.
TOOL_BY_NAME: dict[str, Any] = {t.name: t for t in tools.ALL_TOOLS}


def _invoke(name: str, **kwargs: Any) -> Any:
    """Invoke one tool directly, in a defined (code-authored) way."""
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        raise KeyError(f"unknown tool named {name!r} in defined playbook")
    return tool.invoke(kwargs)


def _as_json(result: Any) -> dict[str, Any]:
    """Normalize a tool result to a dict (mock tools return JSON strings)."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _text(result: Any) -> str:
    """Normalize a tool result to a printable string."""
    if isinstance(result, str):
        return result
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Outbound follow-up: do-not-send checks -> research -> draft -> queue approval
# ---------------------------------------------------------------------------


def run_outbound_followup(lead_id: str) -> dict[str, Any]:
    """Run the outbound-followup playbook by invoking tools in a defined order.

    Sequence (mirrors `skills/outbound-followup/SKILL.md`):
        check_contact_history -> lookup_lead, get_gong_transcripts, web_search_company
        -> draft_slack_followup (only if the do-not-send gate passes)

    Returns:
        {"outcome": "drafted" | "blocked", ...} — never sends; drafts are queued
        for rep approval via `draft_slack_followup`.
    """
    history = _as_json(_invoke("check_contact_history", lead_id=lead_id))
    reason = str(history.get("summary") or "unknown reason")

    if history.get("do_not_send") or any(
        o.get("days_ago", 99) <= 3 for o in history.get("prior_outreach") or []
    ):
        return {"outcome": "blocked", "lead_id": lead_id, "reason": reason, "draft": None}

    # Clear to draft: the research batch runs only after the gate passes.
    lead = _as_json(_invoke("lookup_lead", lead_id=lead_id))
    company = str(lead.get("company") or "")
    research = _text(_invoke("web_search_company", company=company))
    gong = _text(_invoke("get_gong_transcripts", lead_id=lead_id))

    draft, rationale = GTMSimulator._draft_for(lead, research)
    _invoke(
        "draft_slack_followup",
        lead_id=lead_id,
        draft=draft,
        rationale=rationale,
    )
    return {
        "outcome": "drafted",
        "lead_id": lead_id,
        "contact": lead.get("contact"),
        "company": company,
        "draft": draft,
        "rationale": rationale,
        "research": research,
        "gong": gong,
        "draft_queued": True,
    }


# ---------------------------------------------------------------------------
# Account intelligence briefing: usage, hiring, funding -> flags + action
# ---------------------------------------------------------------------------


def run_account_briefing(account: str) -> dict[str, Any]:
    """Run the account-intelligence playbook by invoking tools in a defined order.

    Sequence: query_product_usage -> get_hiring_signals -> get_funding_news.
    """
    usage = _as_json(_invoke("query_product_usage", account=account))
    hiring = _text(_invoke("get_hiring_signals", company=account))
    funding = _text(_invoke("get_funding_news", company=account))
    credits = usage.get("credits_remaining_pct", "n/a")
    renewal = usage.get("renewal_in_days", "—")

    briefing = (
        f"# Account briefing — {account}\n\n"
        f"## Signals\n"
        f"- Usage: {usage.get('weekly_active_users_trend', 'no data')} WAU trend, "
        f"{usage.get('sdk_installs_last_7d', '—')} SDK installs last 7d\n"
        f"- Hiring: {hiring}\n"
        f"- Funding: {funding}\n\n"
        f"## Flags\n"
        f"- ⚠️ Credits at {credits}% — pre-renewal expansion conversations should start now\n"
        f"- Renewal in {renewal} days\n\n"
        f"## Recommended action\n"
        f"Reach out to the most engaged user about the Enterprise tier; "
        f"address credits before they run out."
    )
    return {
        "outcome": "ok",
        "account": account,
        "usage": usage,
        "hiring": hiring,
        "funding": funding,
        "briefing": briefing,
    }


# ---------------------------------------------------------------------------
# Meeting prep: dossier, last interaction, agenda, style notes
# ---------------------------------------------------------------------------


def run_meeting_prep(contact: str) -> dict[str, Any]:
    """Run the meeting-prep playbook by invoking tools in a defined order.

    Sequence: get_person_dossier -> get_last_interaction.
    """
    dossier = _as_json(_invoke("get_person_dossier", query=contact))
    last = _as_json(_invoke("get_last_interaction", query=contact))
    priorities = ", ".join(dossier.get("priorities", [])) or "—"
    first_priority = priorities.split(",")[0] if priorities != "—" else "open"

    prep = (
        f"# Meeting prep — {contact}\n\n"
        f"**Role**: {dossier.get('role', '?')}\n"
        f"**Background**: {dossier.get('background', '—')}\n"
        f"**Current priorities**: {priorities}\n"
        f"**Last interaction**: {last.get('type', '—')} {last.get('when', '')} — {last.get('summary', '')}\n\n"
        f"## Suggested agenda\n"
        f"1. Recap where we left off ({last.get('type', 'last touch')})\n"
        f"2. Their current priority: {first_priority}\n"
        f"3. One concrete next step\n\n"
        f"**Style note**: {dossier.get('style', '—')}"
    )
    return {"outcome": "ok", "contact": contact, "dossier": dossier, "last_interaction": last, "briefing": prep}