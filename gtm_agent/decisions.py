"""OpenAI Decisions backend resolution and selector construction."""

from __future__ import annotations

import os
from typing import Any, Literal

from decision_harness import DecisionsClient, ToolSelector

Backend = Literal["openai", "offline"]


def decision_env(name: str, default: str) -> str:
    """Read a decision-setting environment variable with a default."""

    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _has_key() -> bool:
    return bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("DECISIONS_API_KEY") or "").strip()
    )


def resolve_backend(backend: str | None = None) -> Backend:
    """Resolve ``auto`` to a concrete live/offline backend."""

    requested = backend or decision_env("GTM_DECISION_BACKEND", "auto")
    if requested == "auto":
        return "openai" if _has_key() else "offline"
    if requested == "offline":
        return "offline"
    if requested == "openai":
        if not _has_key():
            raise RuntimeError(
                "GTM_DECISION_BACKEND=openai requires OPENAI_API_KEY or DECISIONS_API_KEY"
            )
        return "openai"
    raise ValueError(
        f"unknown decision backend {requested!r}; expected auto, openai, or offline"
    )


def make_decisions_client() -> DecisionsClient:
    """Build the OpenAI Decisions wire client."""

    return DecisionsClient(model=decision_env("GTM_DECISION_MODEL", "gpt-6-luna"))


def make_tool_selector(
    backend: str | None = None,
    *,
    client: Any | None = None,
) -> ToolSelector:
    """Build a ``ToolSelector`` for the selected backend."""

    resolved = resolve_backend(backend)
    if resolved == "offline":
        return ToolSelector(offline=True)
    threshold = float(decision_env("GTM_TOOL_SELECTION_THRESHOLD", "0.5"))
    return ToolSelector(
        client=client if client is not None else make_decisions_client(),
        threshold=threshold,
    )
