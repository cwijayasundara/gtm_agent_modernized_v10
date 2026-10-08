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
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from . import tools

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


def build_agent(model: BaseChatModel | None = None):
    """Build the GTM deep agent with skills-backed tool binding."""
    model_id = os.getenv("GTM_MODEL_ID", DEFAULT_MODEL_ID)
    model = model or ChatAnthropic(model=model_id, max_tokens=8000)
    return create_deep_agent(
        model=model,
        tools=tools.ALL_TOOLS,
        system_prompt=GTM_SYSTEM_PROMPT,
        skills=["/skills/"],
        backend=FilesystemBackend(root_dir=PROJECT_ROOT),
    )
