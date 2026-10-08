"""Mocked data-source tools for the sample GTM agent — a 50-tool simulated stack.

Every tool that touches a "system of record" (Salesforce, Gong, LinkedIn,
Exa, BigQuery, Zendesk, Slack...) is mocked with in-memory fixtures so the
sample runs without external credentials.

The GTM-specific tools are **deferred** (`extras={"defer_loading": True}`).
Their schemas are NOT sent to the model up front. Each skill declares the
tools it needs via `metadata.include_tools` in its `SKILL.md` frontmatter,
and the deepagents `SkillsMiddleware` discloses a tool's schema to the model
only once the skill that lists it has been read. That is the "binding tools
to skills" feature this sample demonstrates — at 50 tools, the context
savings and capability gating become very visible: a lead follow-up run
never sees the 38 tools owned by other skills.

Tool inventory (50):
-  3 always available  : list_pending_drafts, get_current_date, search_internal_wiki
- 12 outbound-followup : CRM + engagement + drafting (bound by `outbound-followup`)
- 15 account intel     : usage, hiring, funding, market signals (bound by `account-intelligence`)
- 12 customer health   : support, renewals, adoption (bound by `customer-health`)
-  8 meeting prep      : relationship context before calls (bound by `meeting-prep`)
- (competitor-analysis is a methodology-only skill and binds no tools)
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

DEFERRED = {"defer_loading": True}

# ---------------------------------------------------------------------------
# In-memory fixtures (stand-ins for Salesforce / Gong / Exa / BigQuery / Slack)
# ---------------------------------------------------------------------------

_LEADS: dict[str, dict[str, Any]] = {
    "lead-1042": {
        "lead_id": "lead-1042",
        "contact": "Priya Sharma",
        "title": "VP Engineering",
        "company": "Northwind Analytics",
        "account_type": "cold",  # cold | warm prospect | customer
        "intent_score": 72,
        "source": "webinar: 'Evals in production'",
        "notes": "Asked about LangSmith pricing in webinar Q&A.",
    },
    "lead-1043": {
        "lead_id": "lead-1043",
        "contact": "Marcus Lee",
        "title": "Head of Platform",
        "company": "Acme Robotics",
        "account_type": "warm prospect",
        "intent_score": 88,
        "source": "inbound demo request",
        "notes": "Trialing agent tracing for their fleet robots.",
    },
    "lead-1044": {
        "lead_id": "lead-1044",
        "contact": "Dana Kim",
        "title": "CTO",
        "company": "Helix Health",
        "account_type": "customer",
        "intent_score": 64,
        "source": "pricing page visit",
        "notes": "Existing customer exploring an expansion.",
    },
}

_CONTACT_HISTORY: dict[str, dict[str, Any]] = {
    "lead-1042": {
        "prior_outreach": [],
        "open_support_ticket": False,
        "do_not_send": False,
        "summary": "No prior contact from anyone on the team.",
    },
    "lead-1043": {
        "prior_outreach": [
            {"rep": "Jess", "days_ago": 1, "channel": "email", "topic": "demo follow-up"}
        ],
        "open_support_ticket": False,
        "do_not_send": False,
        "summary": "Jess emailed yesterday about the demo; a second touch so soon risks being spammy.",
    },
    "lead-1044": {
        "prior_outreach": [{"rep": "Sam", "days_ago": 30, "channel": "call", "topic": "QBR"}],
        "open_support_ticket": True,
        "do_not_send": True,
        "summary": "Open SEV-2 support ticket since yesterday. Do not send marketing outreach.",
    },
}

_GONG_CALLS: dict[str, list[dict[str, str]]] = {
    "lead-1044": [
        {
            "title": "QBR with Helix Health",
            "days_ago": "30",
            "highlight": "Dana mentioned their platform team wants audit logs 'yesterday' — strong expansion signal for the Enterprise tier.",
        }
    ],
}

_WEB_RESEARCH: dict[str, str] = {
    "Northwind Analytics": "Northwind Analytics (Series B, ~120 employees) just announced a push into LLM-powered BI copilots and is hiring 4 AI engineers.",
    "Acme Robotics": "Acme Robotics ships warehouse robots with on-device agents; their eng blog details LangGraph-style orchestration for fleet planning.",
    "Helix Health": "Helix Health, an existing customer, publicly launched an AI clinical-notes feature last week.",
}

_USAGE_SIGNALS: dict[str, dict[str, Any]] = {
    "Helix Health": {
        "weekly_active_users_trend": "+38% MoM",
        "sdk_installs_last_7d": 4120,
        "credits_remaining_pct": 18,
        "renewal_in_days": 75,
    },
}

_HIRING: dict[str, str] = {
    "Helix Health": "3 open roles for 'AI Platform Engineers' posted this week.",
}

_FUNDING: dict[str, str] = {
    "Northwind Analytics": "Raised $40M Series B two weeks ago (TechCrunch).",
}

_SUPPORT_TICKETS: dict[str, list[dict[str, Any]]] = {
    "Helix Health": [
        {"id": "TCK-8841", "severity": "SEV-2", "opened_days_ago": 1, "status": "open",
         "summary": "Trace export failing for EU region workloads."},
    ],
}

_SLACK_DRAFTS: list[dict[str, str]] = []

_DOSSIERS: dict[str, dict[str, Any]] = {
    "Priya Sharma": {
        "role": "VP Engineering, Northwind Analytics",
        "background": "Scaled data platform from 10 to 80 engineers; leads the BI copilot initiative.",
        "priorities": ["evaluation tooling for LLM features", "team enablement", "cost control"],
        "style": "data-driven, brief and direct, no fluff",
    },
    "Dana Kim": {
        "role": "CTO, Helix Health",
        "background": "Clinical-tech veteran; sponsors the AI clinical-notes initiative.",
        "priorities": ["audit logs", "EU data residency", "platform reliability"],
        "style": "technical, wants specifics and dates",
    },
}

_LAST_INTERACTIONS: dict[str, dict[str, str]] = {
    "Priya Sharma": {
        "type": "webinar Q&A",
        "when": "6 days ago",
        "summary": "Asked about LangSmith pricing in the 'Evals in production' webinar Q&A.",
    },
    "Dana Kim": {
        "type": "QBR call",
        "when": "30 days ago",
        "summary": "Wants audit logs 'yesterday' — strong Enterprise expansion signal.",
    },
}


# ---------------------------------------------------------------------------
# Bespoke deferred, skill-bound tools (used by the offline test scenarios)
# ---------------------------------------------------------------------------


@tool(extras=DEFERRED)
def lookup_lead(lead_id: str) -> str:
    """Look up a lead's full CRM record: contact, title, company, account type (customer / warm prospect / cold), intent score, and source."""
    return json.dumps(_LEADS.get(lead_id, {"error": f"unknown lead {lead_id}"}))


@tool(extras=DEFERRED)
def check_contact_history(lead_id: str) -> str:
    """Check whether a teammate already reached out to this lead/account recently and whether any do-not-send signal exists (e.g. open support ticket). Always run this BEFORE drafting."""
    return json.dumps(_CONTACT_HISTORY.get(lead_id, {"error": f"unknown lead {lead_id}"}))


@tool(extras=DEFERRED)
def get_gong_transcripts(lead_id: str) -> str:
    """Retrieve highlights from recent sales calls (Gong) with this lead's account, if any."""
    return json.dumps(_GONG_CALLS.get(lead_id, []))


@tool(extras=DEFERRED)
def web_search_company(company: str) -> str:
    """Search the web (Exa) for what the company is currently doing with AI: launches, funding, hiring."""
    return _WEB_RESEARCH.get(company, f"No recent public AI activity found for {company}.")


@tool(extras=DEFERRED)
def draft_slack_followup(lead_id: str, draft: str, rationale: str) -> str:
    """Queue a personalized outreach draft as a Slack DM for rep approval. NEVER sends directly; requires explicit rep approval. Include the draft body and the reasoning behind the chosen angle."""
    _SLACK_DRAFTS.append({"lead_id": lead_id, "draft": draft, "rationale": rationale, "status": "pending_approval"})
    return f"Draft queued for lead {lead_id} and sent to the rep for approval (status: pending_approval)."


@tool(extras=DEFERRED)
def query_product_usage(account: str) -> str:
    """Pull product usage analytics (BigQuery) for an account: active users trend, SDK installs, credits remaining, renewal date."""
    return json.dumps(_USAGE_SIGNALS.get(account, {"error": f"no usage data for {account}"}))


@tool(extras=DEFERRED)
def get_hiring_signals(company: str) -> str:
    """Check hiring signals for a company (e.g. AI engineer job postings) — a strong expansion signal."""
    return _HIRING.get(company, f"No notable AI hiring activity found for {company}.")


@tool(extras=DEFERRED)
def get_funding_news(company: str) -> str:
    """Check recent funding news for a company — budget availability signal."""
    return _FUNDING.get(company, f"No recent funding news for {company}.")


# ---------------------------------------------------------------------------
# Generated mock tools (simulating the rest of a realistic GTM stack)
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str, fixture_fn: Callable[[str], Any] | None = None) -> BaseTool:
    """Create a deferred mock tool with a generic `query` parameter."""

    def _impl(query: str = "") -> str:
        if fixture_fn is not None:
            return json.dumps(fixture_fn(query))
        return json.dumps({
            "tool": name,
            "query": query,
            "result": f"[simulated] No notable findings for {query!r} — mock data source.",
        })

    return tool(name, description=description, extras=DEFERRED)(_impl)


# --- outbound-followup additions (4) ---
search_linkedin_profile = _make_tool(
    "search_linkedin_profile",
    "Look up a contact's LinkedIn profile: role history, recent posts, and shared connections.",
)
get_email_thread_history = _make_tool(
    "get_email_thread_history",
    "Retrieve past email threads between our team and this contact's company.",
)
get_marketing_touches = _make_tool(
    "get_marketing_touches",
    "List this lead's marketing touchpoints: webinar attendance, content downloads, event visits.",
)
queue_followup_sequence = _make_tool(
    "queue_followup_sequence",
    "Optionally enroll an approved contact into a follow-up email sequence.",
)
get_contacts_at_account = _make_tool(
    "get_contacts_at_account",
    "List all known contacts at an account with roles and engagement recency.",
)
get_calendar_availability = _make_tool(
    "get_calendar_availability",
    "Check the owning rep's calendar availability to propose meeting slots.",
)
check_domain_reputation = _make_tool(
    "check_domain_reputation",
    "Screen the company's email domain for spam/bounce risk before sending outreach.",
)

# --- account-intelligence additions (12) ---
get_sdk_installs = _make_tool(
    "get_sdk_installs",
    "Weekly SDK/package install counts for an account's org — developer ecosystem signal.",
    lambda q: {"account": q, "installs_last_7d": 4120, "trend": "+12% WoW"},
)
get_web_traffic = _make_tool(
    "get_web_traffic",
    "Web traffic to our docs/site attributed to an account's domain.",
)
get_community_activity = _make_tool(
    "get_community_activity",
    "Community and GitHub activity from an account's engineers (issues, discussions, stars).",
)
get_expansion_score = _make_tool(
    "get_expansion_score",
    "Composite 0-100 expansion-likelihood score for an account.",
    lambda q: {"account": q, "expansion_score": 81, "drivers": ["usage growth", "AI hiring"]},
)
get_churn_risk = _make_tool(
    "get_churn_risk",
    "Composite 0-100 churn-risk score for an account with contributing factors.",
)
get_nps_score = _make_tool(
    "get_nps_score",
    "Latest NPS survey responses from an account's users.",
)
get_news_mentions = _make_tool(
    "get_news_mentions",
    "Recent news mentions for a company: launches, partnerships, executive moves.",
)
get_funding_history = _make_tool(
    "get_funding_history",
    "Full funding history for a company: rounds, amounts, dates, investors.",
    lambda q: {"company": q, "rounds": [{"round": "Series B", "amount": "$40M", "when": "2 weeks ago"}]},
)
get_competitor_mentions = _make_tool(
    "get_competitor_mentions",
    "Public mentions of competitors by a company's team — competitive-risk signal.",
)
get_product_adoption_stage = _make_tool(
    "get_product_adoption_stage",
    "Where an account sits in the adoption funnel: evaluating / piloting / scaling / mature.",
)
get_exec_changes = _make_tool(
    "get_exec_changes",
    "Recent executive changes at a company — new leaders often re-evaluate vendors.",
)
get_integrations_usage = _make_tool(
    "get_integrations_usage",
    "Which of our integrations an account uses and how actively.",
)

# --- customer-health (12) ---
get_open_tickets = _make_tool(
    "get_open_tickets",
    "List open support tickets for an account with severity and age.",
    lambda q: {"account": q, "tickets": _SUPPORT_TICKETS.get(q, [])},
)
get_support_sla = _make_tool(
    "get_support_sla",
    "Support plan tier and SLA compliance for an account.",
)
get_recent_calls_summary = _make_tool(
    "get_recent_calls_summary",
    "Summaries of recent support/success calls with an account.",
)
get_renewal_date = _make_tool(
    "get_renewal_date",
    "Contract renewal date and auto-renewal status for an account.",
    lambda q: {"account": q, "renewal_in_days": 75, "auto_renew": True},
)
get_credit_balance = _make_tool(
    "get_credit_balance",
    "Remaining platform credits and projected depletion date for an account.",
    lambda q: {"account": q, "credits_remaining_pct": 18, "projected_empty": "in 3 weeks"},
)
get_license_utilization = _make_tool(
    "get_license_utilization",
    "Seats purchased vs. actively used for an account.",
)
get_onboarding_status = _make_tool(
    "get_onboarding_status",
    "Onboarding progress and blockers for a new account.",
)
get_open_action_items = _make_tool(
    "get_open_action_items",
    "Open action items promised to an account during recent calls.",
)
get_feature_requests = _make_tool(
    "get_feature_requests",
    "Feature requests submitted by an account, ranked by asker influence.",
)
get_uptime_status = _make_tool(
    "get_uptime_status",
    "Platform uptime/incident history affecting an account over the last 30 days.",
)
get_escalations = _make_tool(
    "get_escalations",
    "Active executive escalations for an account, if any.",
)
get_qbr_notes = _make_tool(
    "get_qbr_notes",
    "Notes and outcomes from the account's most recent quarterly business review.",
)

# --- meeting-prep (8) ---
get_last_interaction = _make_tool(
    "get_last_interaction",
    "Details of the most recent interaction with a contact: channel, date, outcome.",
    lambda q: _LAST_INTERACTIONS.get(q, {"contact": q, "result": "[simulated] No recorded interactions."}),
)
get_open_threads = _make_tool(
    "get_open_threads",
    "Unresolved conversation threads with a contact across email and Slack.",
)
get_mutual_connections = _make_tool(
    "get_mutual_connections",
    "Mutual connections between our team and a contact — useful for warm intros.",
)
get_account_playbook = _make_tool(
    "get_account_playbook",
    "The agreed account plan/playbook for an account: goals, stakeholders, next steps.",
)
get_person_dossier = _make_tool(
    "get_person_dossier",
    "Background dossier on a contact: role, priorities, past interactions, preferences.",
    lambda q: _DOSSIERS.get(q, {"contact": q, "result": "[simulated] No dossier on file."}),
)
get_upcoming_events = _make_tool(
    "get_upcoming_events",
    "Upcoming events (conferences, webinars) a contact or account registered for.",
)
get_pitch_deck_latest = _make_tool(
    "get_pitch_deck_latest",
    "The latest pitch/deck shared with an account and whether it was viewed.",
)
get_agenda_template = _make_tool(
    "get_agenda_template",
    "Best-practice agenda template for a given meeting type (discovery, QBR, renewal).",
)

# ---------------------------------------------------------------------------
# Always-available tools (NOT deferred — these stay in the model's tool list)
# ---------------------------------------------------------------------------


@tool
def list_pending_drafts() -> str:
    """List all outreach drafts currently pending rep approval."""
    if not _SLACK_DRAFTS:
        return json.dumps([])
    return json.dumps(_SLACK_DRAFTS)


@tool
def get_current_date() -> str:
    """Get today's date, for SLAs and recency checks."""
    import datetime

    return datetime.date.today().isoformat()


@tool
def search_internal_wiki(query: str) -> str:
    """Search the internal wiki for GTM processes, pricing, and product FAQs."""
    return f"[simulated] No internal wiki pages match {query!r} — mock data source."


ALL_TOOLS: list[BaseTool] = [
    # deferred — bound to skills via `metadata.include_tools`
    lookup_lead,
    check_contact_history,
    get_gong_transcripts,
    web_search_company,
    draft_slack_followup,
    get_contacts_at_account,
    search_linkedin_profile,
    get_email_thread_history,
    get_marketing_touches,
    get_calendar_availability,
    check_domain_reputation,
    queue_followup_sequence,
    query_product_usage,
    get_hiring_signals,
    get_funding_news,
    get_sdk_installs,
    get_web_traffic,
    get_community_activity,
    get_expansion_score,
    get_churn_risk,
    get_nps_score,
    get_news_mentions,
    get_funding_history,
    get_competitor_mentions,
    get_product_adoption_stage,
    get_exec_changes,
    get_integrations_usage,
    get_open_tickets,
    get_support_sla,
    get_recent_calls_summary,
    get_renewal_date,
    get_credit_balance,
    get_license_utilization,
    get_onboarding_status,
    get_open_action_items,
    get_feature_requests,
    get_uptime_status,
    get_escalations,
    get_qbr_notes,
    get_last_interaction,
    get_open_threads,
    get_mutual_connections,
    get_account_playbook,
    get_person_dossier,
    get_upcoming_events,
    get_pitch_deck_latest,
    get_agenda_template,
    # always available
    list_pending_drafts,
    get_current_date,
    search_internal_wiki,
]

assert len(ALL_TOOLS) == 50, f"expected 50 tools, got {len(ALL_TOOLS)}"


def reset_fixtures() -> None:
    """Clear mutable state between test scenarios."""
    _SLACK_DRAFTS.clear()
