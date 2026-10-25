"""Automatic, read-only PTC allowlists derived from skill metadata."""

from __future__ import annotations

from pathlib import Path

from . import tools
from .skills import load_skills, skill_tool_names

TOOL_BY_NAME = {tool.name: tool for tool in tools.ALL_TOOLS}


def ptc_allowlist(skill: str | None, *, root: Path | None = None) -> list[str]:
    """Return the skill's tools that are safe to bridge into the code interpreter.

    Side-effect tools stay on the normal tool path so they can be gated by HITL.
    """

    if skill is None:
        return []
    names = skill_tool_names(skill, load_skills(root))
    allowed: list[str] = []
    for name in names:
        tool = TOOL_BY_NAME.get(name)
        if tool is None:
            continue
        if (tool.extras or {}).get("side_effect") is True:
            continue
        allowed.append(name)
    return allowed
