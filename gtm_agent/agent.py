"""The sample GTM agent, built on deepagents.

Demonstrates the "binding tools to skills" feature:

- GTM tools are created with ``extras={"defer_loading": True}`` so their
  schemas are withheld from the model at the start of a conversation.
- Each skill lists the tools it needs via ``metadata.include_tools`` in its
  ``SKILL.md`` frontmatter.
- ``create_deep_agent(skills=["/skills/"])`` wires a ``SkillsMiddleware``
  that discloses a deferred tool's schema only after the model reads the
  ``SKILL.md`` of a skill that lists it. Before that, calling the tool
  fails as an unknown tool.

Model: ``claude-haiku-5-5`` (Claude Haiku 5.5).

Three agent shapes are provided for the same GTM workflow:

- :func:`build_agent` — deep agent with skills-backed tool binding (default).
- :func:`build_interpreter_agent` — deep agent whose tools are invoked via the
  code interpreter / programmatic tool calling (``CodeInterpreterMiddleware``
  with a ``ptc`` allowlist) instead of serial model-mediated tool calls.
- :func:`build_tool_selector_agent` — deep agent whose tool universe is narrowed
  with the LLM tool selector middleware before the agent loop starts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import LLMToolSelectorMiddleware

from . import tools
from .ptc import ptc_allowlist, read_only_names

# Load API keys / settings from a `.env` file at the project root (if present).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "claude-haiku-5-5"

GTM_SYSTEM_PROMPT = """You are LangChain's GTM agent. You help sales reps run outbound and account intelligence end to end.

Your workflow for lead follow-up is strictly:
do-not-send checks -> research -> draft -> rationale -> queue for rep approval.

Rules:
- Load the matching skill before acting. Use the `skills` or `read_file` tool
  to read a skill's SKILL.md before using any tools it describes.
- Nothing is ever sent without explicit rep approval: drafts are queued to
  Slack with your reasoning and sources attached.
- Be cautious: if there is any reason not to reach out (open support ticket,
  teammate already engaged, recent contact), say so and draft nothing.
- Be explainable: every draft comes with a short rationale for the chosen
  angle so the rep can refine it.

Speed rules (these matter — every model round trip costs seconds):
- A skill's SKILL.md may already be in context (pinned) — if so, do NOT call
  read_file on it again; start using its tools immediately.
- Target budget: at most 3-4 model turns per workflow. Turn 1: read the
  skill (only if not already pinned). Turn 2: ONE message with EVERY
  independent lookup batched together (do-not-send check, CRM record, Gong,
  marketing touches — all at once). Turn 3: anything that depended on
  turn-2 results (e.g. web_search_company with the company name).
  Turn 4: draft + queue. NEVER one tool per turn.
- Call only tools whose output you will actually use. Skip marginal lookups.
- Tool-call turns carry no prose; save all narrative for the final summary.
- Final summary: at most 120 words, dense, no restating the checklist.
"""

# Extra guidance for the interpreter (programmatic tool calling) agent. The
# middleware already documents the `tools.<camelCase>(…)` API; these rules say
# *how to structure a GTM workflow* as one program.
GTM_INTERPRETER_SYSTEM_PROMPT = GTM_SYSTEM_PROMPT + """
Interpreter rules (overriding the turn-budget rules above):
- Do the workflow in code. Write ONE `eval` program that gates, batches, and
  composes — not a sequence of tool-call turns.
- The `eval` bridge is READ-ONLY. Never call `draftSlackFollowup` or
  `queueFollowupSequence` inside the program; those actions stay on the normal
  tool path so a rep can approve them.
- Gate first. `checkContactHistory` runs before any drafting; if it reports a
  do-not-send signal or a teammate touch within 3 days, return a `blocked`
  result and stop — do not draft.
- Batch independent lookups with `Promise.all` inside the same program.
- Chain dependent calls in the same program (use the CRM record's company name
  as the research input) instead of returning intermediate values.
- Return a small object from the program (`{ outcome, draft, rationale, ... }`)
  and keep the final prose summary under 120 words.
- After the program returns a draft, call `draft_slack_followup` as a normal
  tool to queue it for rep approval. Never send anything directly.
"""

# Tools the interpreter is allowed to bridge into JavaScript
# (`globalThis.tools[<camelCase>] = ...; tools.<camelCase>(...)`).
# Mirrors the full GTM tool surface used by the defined workflows in
# ``gtm_agent.direct``.
# Legacy fallback for callers that do not pass a skill. It is read-only;
# action tools always stay on the normal HITL-gated tool path.
DEFAULT_INTERPRETER_PTC: list[str] = [
    "lookup_lead",
    "check_contact_history",
    "get_gong_transcripts",
    "web_search_company",
    "query_product_usage",
    "get_hiring_signals",
    "get_funding_news",
]

DEFAULT_TOOL_SELECTOR_MAX_TOOLS = 3


def _make_model(model: BaseChatModel | None = None) -> BaseChatModel:
    """Resolve the chat model (explicit override, else ``GTM_MODEL_ID``)."""
    model_id = os.getenv("GTM_MODEL_ID", DEFAULT_MODEL_ID)
    return model or ChatAnthropic(model=model_id, max_tokens=8000)


def interpreter_middleware(ptc: list[str] | None = None) -> Any:
    """Build the code-interpreter middleware (imported lazily: optional extra).

    Raises:
        ImportError: with install instructions when the QuickJS runtime is absent.
    """
    try:
        from langchain_quickjs import CodeInterpreterMiddleware
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The interpreter needs the optional QuickJS runtime. Install it with:\n"
            '    uv pip install -e ".[interpreter]"\n'
            "or: uv pip install \"deepagents[quickjs]\""
        ) from exc
    return CodeInterpreterMiddleware(
        ptc=ptc if ptc is not None else DEFAULT_INTERPRETER_PTC
    )


def build_agent(
    model: BaseChatModel | None = None,
    *,
    extra_middleware: list[Any] | None = None,
    interrupt_on: dict[str, Any] | None = None,
    checkpointer: Any | None = None,
):
    """Build the GTM deep agent with skills-backed tool binding."""
    return create_deep_agent(
        model=_make_model(model),
        tools=tools.ALL_TOOLS,
        system_prompt=GTM_SYSTEM_PROMPT,
        skills=["/skills/"],
        backend=FilesystemBackend(root_dir=PROJECT_ROOT),
        middleware=[*(extra_middleware or [])],
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
    )


def build_interpreter_agent(
    model: BaseChatModel | None = None,
    *,
    skill: str | None = None,
    ptc: list[str] | None = None,
    system_prompt: str | None = None,
    skills: list[str] | None = None,
    backend: Any | None = None,
    extra_middleware: list[Any] | None = None,
    interrupt_on: dict[str, Any] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build the GTM deep agent that invokes tools through a code interpreter.

    Instead of letting the model fire a serial chain of tool calls, the agent is
    given a programmatic tool calling (PTC) surface: the model writes a small JS
    program (via the interpreter's ``eval`` tool), and the harness bridges only
    the allowlisted tools into the ``tools`` namespace before executing the program.

    This is the "agent writes code that calls the tools" pattern from the
    LangChain deep-agents interpreter post. Tool invocations become code-level
    composition (one eval per logical step, batched data gathering) rather than
    model-mediated round trips inside the loop.

    Args:
        model: Chat model override (defaults to ``GTM_MODEL_ID``).
        skill: Selected skill. When ``ptc`` is omitted, the allowlist is derived
            from this skill's ``include_tools``.
        ptc: Allowlisted tool names the interpreter may bridge. ``None`` uses the
            skill-derived list, falling back to :data:`DEFAULT_INTERPRETER_PTC`.
        system_prompt: Override the GTM system prompt (optional).
        skills: Skill globs for ``create_deep_agent`` (optional).
        backend: Filesystem backend override (optional).
        extra_middleware: Additional middleware placed after the interpreter.
        interrupt_on: HITL configuration for action tools.
        checkpointer: LangGraph checkpointer required for HITL resume.
    """
    if ptc is None:
        ptc = ptc_allowlist(skill) if skill is not None else DEFAULT_INTERPRETER_PTC
    ptc = read_only_names(ptc)
    return create_deep_agent(
        model=_make_model(model),
        tools=tools.ALL_TOOLS,
        system_prompt=system_prompt or GTM_INTERPRETER_SYSTEM_PROMPT,
        skills=skills or ["/skills/"],
        backend=backend or FilesystemBackend(root_dir=PROJECT_ROOT),
        middleware=[interpreter_middleware(ptc), *(extra_middleware or [])],
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
    )


def build_tool_selector_agent(
    model: BaseChatModel | None = None,
    *,
    max_tools: int = DEFAULT_TOOL_SELECTOR_MAX_TOOLS,
    always_include: list[str] | None = None,
    system_prompt: str | None = None,
    skills: list[str] | None = None,
    backend: Any | None = None,
    extra_middleware: list[Any] | None = None,
    interrupt_on: dict[str, Any] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build the GTM deep agent whose tool set is narrowed by an LLM selector.

    The ``LLMToolSelectorMiddleware`` (LangChain) uses a (cheap) selection model to
    pick the most relevant tools from the full GTM toolset before the agent loop
    runs. Useful when the agent carries dozens of tools and the selector keeps the
    active tool surface small.
    """
    return create_deep_agent(
        model=_make_model(model),
        tools=tools.ALL_TOOLS,
        system_prompt=system_prompt or GTM_SYSTEM_PROMPT,
        skills=skills or ["/skills/"],
        backend=backend or FilesystemBackend(root_dir=PROJECT_ROOT),
        middleware=[
            LLMToolSelectorMiddleware(
                model=_make_model(model),
                max_tools=max_tools,
                always_include=always_include or [],
            ),
            *(extra_middleware or []),
        ],
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
    )
