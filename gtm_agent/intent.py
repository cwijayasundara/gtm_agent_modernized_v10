"""Confidence-gated intent classification and demo target resolution."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from decision_harness import DecisionsClient

from . import tools
from .decisions import decision_env, make_decisions_client, resolve_backend

Action = Literal["route", "clarify", "general"]

INTENT_TO_SKILL: dict[str, str | None] = {
    "outbound_followup": "outbound-followup",
    "account_intelligence": "account-intelligence",
    "meeting_prep": "meeting-prep",
    "inbox_review": None,
    "general": None,
}

INTENT_CHOICES: dict[str, str] = {
    "outbound_followup": "Qualify an inbound lead, follow up with a lead, or draft outreach.",
    "account_intelligence": "Account review, usage/hiring/funding signals, expansion, or renewal briefing.",
    "meeting_prep": "Prepare for an upcoming meeting or call with a contact or account.",
    "inbox_review": "Review drafts that are waiting for rep approval.",
    "general": "Anything else, including small talk or requests outside the available workflows.",
}

_HEURISTIC_SIGNALS: dict[str, tuple[str, ...]] = {
    "outbound_followup": (
        "lead", "follow up", "follow-up", "outreach", "inbound", "prospect",
        "sdr", "draft an email", "email draft",
    ),
    "account_intelligence": (
        "account intelligence", "account review", "briefing", "expansion",
        "usage", "hiring", "funding", "renewal", "pipeline",
    ),
    "meeting_prep": (
        "meeting prep", "prep me", "prepare me", "prepare a meeting",
        "upcoming meeting", "discovery call", "qbr prep", "agenda",
    ),
    "inbox_review": ("inbox", "pending drafts", "waiting for approval"),
}


@dataclass(slots=True)
class IntentDecision:
    intent: str
    skill: str | None
    action: Action
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    clarify_probability: float = 0.0
    classifier: str = ""
    latency_ms: int = 0
    events: list[str] = field(default_factory=list)


def _intent_payload(prompt: str, conversation: str = "") -> dict[str, Any]:
    state = prompt if not conversation else f"{conversation}\n\nUSER: {prompt}"
    return {
        "model": decision_env("GTM_DECISION_MODEL", "gpt-6-luna"),
        "input": state,
        "questions": [
            {
                "type": "choice",
                "name": "intent",
                "instructions": "Which GTM workflow should handle this request?",
                "choices": [
                    {"value": name, "description": description}
                    for name, description in INTENT_CHOICES.items()
                ],
            },
            {
                "type": "predicate",
                "name": "needs_clarification",
                "instructions": (
                    "Does this request lack enough information to choose a workflow?"
                ),
            },
        ],
    }


class HeuristicIntentRouter:
    """Deterministic keyword fallback used offline or after API failure."""

    def classify(self, prompt: str, conversation: str = "") -> IntentDecision:
        started = time.perf_counter()
        text = f"{conversation}\n{prompt}".lower()
        scores = {
            intent: sum(1 for signal in signals if signal in text)
            for intent, signals in _HEURISTIC_SIGNALS.items()
        }
        intent, score = max(scores.items(), key=lambda item: item[1])
        if score == 0:
            intent, score = "general", 0

        confidence = 0.0 if score == 0 else (0.8 if score >= 2 else 0.7)
        action: Action = "route" if confidence >= 0.65 else "clarify"
        skill = INTENT_TO_SKILL.get(intent)
        return IntentDecision(
            intent=intent,
            skill=skill,
            action=action,
            confidence=confidence,
            probabilities={intent: confidence} if score else {},
            clarify_probability=0.0 if score else 1.0,
            classifier="heuristic",
            latency_ms=int((time.perf_counter() - started) * 1000),
            events=[] if score else ["no intent signal matched"],
        )


class IntentRouter:
    """OpenAI Decisions-backed intent router with an offline fallback."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        client: Any | None = None,
        confidence_floor: float | None = None,
        clarify_floor: float = 0.6,
    ) -> None:
        if client is not None:
            self.offline = False
        else:
            self.offline = resolve_backend(backend) == "offline"
            if not self.offline:
                client = make_decisions_client()
        self.client = client
        self.confidence_floor = (
            float(decision_env("GTM_INTENT_CONFIDENCE_FLOOR", "0.65"))
            if confidence_floor is None
            else confidence_floor
        )
        self.clarify_floor = clarify_floor
        self.model = decision_env("GTM_DECISION_MODEL", "gpt-6-luna")
        self._fallback = HeuristicIntentRouter()

    def classify(self, prompt: str, conversation: str = "") -> IntentDecision:
        started = time.perf_counter()
        if self.offline:
            return self._fallback.classify(prompt, conversation)
        try:
            data = self.client.post(_intent_payload(prompt, conversation))
            return self._from_response(data, started)
        except Exception as exc:  # noqa: BLE001 - fallback is the contract
            decision = self._fallback.classify(prompt, conversation)
            return replace(
                decision,
                latency_ms=int((time.perf_counter() - started) * 1000),
                events=[*decision.events, f"decisions unavailable: {exc}"],
            )

    def _from_response(self, data: dict[str, Any], started: float) -> IntentDecision:
        answers = DecisionsClient.answers(data, ["intent", "needs_clarification"])
        answer = answers["intent"]
        intent = str(answer.get("choice") or "general")
        probabilities = {
            str(item.get("value")): float(item.get("probability", 0.0) or 0.0)
            for item in answer.get("probabilities") or []
            if isinstance(item, dict) and item.get("value") is not None
        }
        confidence = float(answer.get("confidence") or probabilities.get(intent, 0.0) or 0.0)
        clarify = float(answers["needs_clarification"].get("probability", 0.0) or 0.0)

        if clarify >= self.clarify_floor or confidence < self.confidence_floor:
            action: Action = "clarify"
        elif intent == "general":
            action = "general"
        else:
            action = "route"

        return IntentDecision(
            intent=intent,
            skill=INTENT_TO_SKILL.get(intent),
            action=action,
            confidence=confidence,
            probabilities=probabilities,
            clarify_probability=clarify,
            classifier=f"decisions:{self.model}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _match_known(prompt: str, names: tuple[str, ...]) -> str | None:
    text = prompt.lower()
    for name in sorted(names, key=len, reverse=True):
        if name.lower() in text:
            return name
    return None


def resolve_target(intent: str, prompt: str) -> str | None:
    """Resolve a demo-fixture target from a natural-language prompt."""

    if intent == "outbound_followup":
        match = re.search(r"lead-\d+", prompt, re.IGNORECASE)
        if match is None:
            return None
        lead_id = match.group(0).lower()
        return lead_id if lead_id in tools.KNOWN_LEADS else None
    if intent == "account_intelligence":
        return _match_known(prompt, tools.KNOWN_ACCOUNTS)
    if intent == "meeting_prep":
        return _match_known(prompt, tools.KNOWN_CONTACTS)
    return None
