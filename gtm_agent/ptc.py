"""Automatic, read-only PTC allowlists derived from skill metadata."""

from __future__ import annotations

from pathlib import Path

from . import tools
from .skills import load_skills, skill_tool_names

TOOL_BY_NAME = {tool.name: tool for tool in tools.ALL_TOOLS}


def read_only_names(names: list[str]) -> list[str]:
    """Drop known side-effect tools from an explicit PTC allowlist."""

    return [
        name
        for name in names
        if (TOOL_BY_NAME.get(name) is None)
        or (TOOL_BY_NAME[name].extras or {}).get("side_effect") is not True
    ]


def ptc_allowlist(skill: str | None, *, root: Path | None = None) -> list[str]:
    """Return the skill's tools that are safe to bridge into the code interpreter.

    Side-effect tools stay on the normal tool path so they can be gated by HITL.
    """

    if skill is None:
        return []
    return read_only_names(skill_tool_names(skill, load_skills(root)))
