"""Conversation-aware tool selection using the Decisions API."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from typing import Any

from decision_harness import SelectorUnavailable, ToolSelection
from decision_harness.middleware.tool_selection import ToolSelectionMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from . import tools

# Deepagents built-ins are always available; they are not selection candidates.
INFRASTRUCTURE_TOOLS = (
    "read_file",
    "write_file",
    "edit_file",
    "ls",
    "glob",
    "grep",
    "delete",
    "task",
    "write_todos",
)
GTM_TOOL_NAMES = frozenset(tool.name for tool in tools.ALL_TOOLS)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def _clip(text: str, limit: int = 1200) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def render_conversation(
    messages: Sequence[BaseMessage],
    max_chars: int = 12000,
) -> str:
    """Render recent conversation messages, including tool calls and results."""

    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"USER: {_clip(_text(message.content))}")
        elif isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                args = json.dumps(call.get("args", {}), sort_keys=True)
                lines.append(f"ASSISTANT TOOL CALL: {call.get('name')} {_clip(args)}")
            content = _clip(_text(message.content))
            if content:
                lines.append(f"ASSISTANT: {content}")
        elif isinstance(message, ToolMessage):
            name = getattr(message, "name", None) or "tool"
            lines.append(f"TOOL RESULT ({name}): {_clip(_text(message.content))}")

    rendered = "\n".join(line for line in lines if line.strip())
    if len(rendered) <= max_chars:
        return rendered
    return rendered[-max_chars:]


class ConversationToolSelectionMiddleware(ToolSelectionMiddleware):
    """Filter tools using a conversation transcript and a Decisions selector."""

    def __init__(
        self,
        *,
        selector: Any,
        candidate_names: Collection[str] | None = None,
        always_include: list[str] | None = None,
        max_tools: int | None = 6,
    ) -> None:
        super().__init__(
            selector=selector,
            always_include=always_include or [],
            max_tools=max_tools,
            per_call=False,
            on_none="all",
        )
        self.candidate_names = (
            frozenset(candidate_names) if candidate_names is not None else GTM_TOOL_NAMES
        )

    def _select(self, request: Any) -> list[str] | None:
        candidates = [
            tool
            for tool in (request.tools or [])
            if not isinstance(tool, dict) and tool.name in self.candidate_names
        ]
        if not candidates:
            return None

        conversation = render_conversation(request.messages or [])
        if not conversation:
            return None

        pairs = [(tool.name, tool.description or "") for tool in candidates]
        try:
            selection: ToolSelection = self.selector.select(
                conversation,
                pairs,
                max_tools=self.max_tools,
                always_include=self.always_include,
                on_none=self.on_none,
            )
        except SelectorUnavailable as exc:
            self.last = ToolSelection(
                selected=[],
                events=[f"tool selector unavailable, keeping all tools: {exc}"],
            )
            return None

        self.last = selection
        if not selection.selected or any(
            "no tool cleared threshold" in event for event in selection.events
        ):
            return None
        return selection.selected
