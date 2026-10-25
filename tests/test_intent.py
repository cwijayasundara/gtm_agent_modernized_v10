from gtm_agent.intent import HeuristicIntentRouter, IntentRouter, resolve_target


class FakeClient:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def post(self, payload):
        self.calls.append(payload)
        return self.data


def _answers(intent="outbound_followup", confidence=0.91, clarify=0.05):
    return {
        "answers": [
            {
                "name": "intent",
                "type": "choice",
                "choice": intent,
                "confidence": confidence,
                "probabilities": [
                    {"value": intent, "probability": confidence},
                    {"value": "general", "probability": 1 - confidence},
                ],
            },
            {"name": "needs_clarification", "type": "predicate", "probability": clarify},
        ]
    }


def test_high_confidence_intent_routes_and_skill_is_mapped():
    router = IntentRouter(backend="openai", client=FakeClient(_answers()))
    decision = router.classify("Follow up with lead-1042")
    assert decision.action == "route"
    assert decision.intent == "outbound_followup"
    assert decision.skill == "outbound-followup"
    assert decision.confidence == 0.91


def test_low_confidence_or_clarification_asks():
    router = IntentRouter(
        backend="openai", client=FakeClient(_answers(confidence=0.4, clarify=0.9))
    )
    decision = router.classify("do the thing")
    assert decision.action == "clarify"
    assert decision.clarify_probability == 0.9


def test_api_failure_falls_back_to_heuristic():
    class Broken:
        def post(self, payload):
            raise RuntimeError("network down")

    decision = IntentRouter(backend="openai", client=Broken()).classify(
        "Prep me for my meeting with Priya Sharma"
    )
    assert decision.classifier == "heuristic"
    assert decision.skill == "meeting-prep"
    assert decision.action == "route"


def test_target_resolution_uses_demo_fixtures():
    assert resolve_target("outbound_followup", "Follow up with lead-1042") == "lead-1042"
    assert resolve_target("account_intelligence", "briefing for Helix Health") == "Helix Health"
    assert resolve_target("meeting_prep", "prep me for Priya Sharma") == "Priya Sharma"
    assert resolve_target("meeting_prep", "prep me for nobody") is None


def test_heuristic_offline_routes_strong_signal():
    decision = HeuristicIntentRouter().classify("Prepare a meeting prep for Priya Sharma")
    assert decision.action == "route"
    assert decision.skill == "meeting-prep"
    assert decision.classifier == "heuristic"
