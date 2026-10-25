"""Skill catalog parsed from ``skills/*/SKILL.md`` frontmatter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """One skill's metadata and the tools it owns."""

    name: str
    description: str
    include_tools: tuple[str, ...]
    path: Path


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    _, frontmatter, _ = text.split("---", 2)
    data = yaml.safe_load(frontmatter) or {}
    return data if isinstance(data, dict) else {}


def _include_tools(metadata: dict) -> tuple[str, ...]:
    raw = metadata.get("include_tools")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(raw.split())
    if isinstance(raw, list):
        return tuple(str(name) for name in raw)
    return ()


def load_skills(root: Path | None = None) -> dict[str, SkillSpec]:
    """Return every skill keyed by its declared name, in directory order."""

    skills_dir = Path(root) if root is not None else SKILLS_DIR
    skills: dict[str, SkillSpec] = {}
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        data = _frontmatter(skill_md)
        name = str(data.get("name") or skill_md.parent.name)
        description = str(data.get("description") or "").strip()
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        skills[name] = SkillSpec(
            name=name,
            description=description,
            include_tools=_include_tools(metadata),
            path=skill_md,
        )
    return skills


def skill_tool_names(
    skill: str | None,
    skills: dict[str, SkillSpec] | None = None,
) -> list[str]:
    """Return the ordered ``include_tools`` list for a skill, or ``[]``."""

    if skill is None:
        return []
    catalog = skills if skills is not None else load_skills()
    spec = catalog.get(skill)
    return list(spec.include_tools) if spec is not None else []
