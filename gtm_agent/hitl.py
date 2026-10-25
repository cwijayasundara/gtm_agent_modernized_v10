"""Human-in-the-loop helpers for action tools."""

from __future__ import annotations

import json
from typing import Any, Protocol

from langgraph.types import Command

ACTION_TOOLS = ("draft_slack_followup", "queue_followup_sequence")


def interrupt_config(enabled: bool) -> dict[str, bool] | None:
    """Return the deepagents ``interrupt_on`` mapping when HITL is enabled."""

    if not enabled:
        return None
    return {name: True for name in ACTION_TOOLS}


class Reviewer(Protocol):
    def review(
        self,
        action_requests: list[dict[str, Any]],
        review_configs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


class AutoApprover:
    """Deterministic reviewer for tests and non-interactive demos."""

    def __init__(self, approve: bool = True) -> None:
        self.approve = approve

    def review(
        self,
        action_requests: list[dict[str, Any]],
        review_configs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.approve:
            return [{"type": "approve"} for _ in action_requests]
        return [
            {"type": "reject", "message": "Rejected by AutoApprover."}
            for _ in action_requests
        ]


class CliApprover:
    """Interactive approver used by ``--approve``."""

    def review(
        self,
        action_requests: list[dict[str, Any]],
        review_configs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        for request in action_requests:
            print(f"\nApproval required: {request.get('name')}")
            args = json.dumps(request.get("args", {}), indent=2, sort_keys=True)
            print(args)
            answer = input("Approve? [y/N] ").strip().lower()
            if answer in {"y", "yes", "approve"}:
                decisions.append({"type": "approve"})
            else:
                decisions.append(
                    {"type": "reject", "message": "Rejected by reviewer."}
                )
        return decisions


def _interrupt_payloads(result: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    action_requests: list[dict] = []
    review_configs: list[dict] = []
    for interrupt in result.get("__interrupt__") or []:
        value = getattr(interrupt, "value", interrupt)
        if not isinstance(value, dict):
            continue
        action_requests.extend(value.get("action_requests") or [])
        review_configs.extend(value.get("review_configs") or [])
    return action_requests, review_configs


def run_with_hitl(
    agent: Any,
    payload: dict[str, Any],
    config: dict[str, Any],
    reviewer: Reviewer,
) -> dict[str, Any]:
    """Invoke an agent and resume every human approval interrupt."""

    result = agent.invoke(payload, config=config)
    while result.get("__interrupt__"):
        if getattr(agent, "checkpointer", None) is None:
            raise RuntimeError("HITL requires an agent with a checkpointer")
        action_requests, review_configs = _interrupt_payloads(result)
        decisions = reviewer.review(action_requests, review_configs)
        if len(decisions) != len(action_requests):
            raise ValueError(
                f"reviewer returned {len(decisions)} decisions for "
                f"{len(action_requests)} actions"
            )
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
        )
    return result
