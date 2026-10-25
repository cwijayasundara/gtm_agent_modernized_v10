import sys

import main


def test_intent_command_offline(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--decision-backend",
            "offline",
            "intent",
            "Prep me for Priya Sharma",
        ],
    )
    assert main.main() == 0
    out = capsys.readouterr().out
    assert "meeting-prep" in out
    assert "heuristic" in out


def test_auto_command_dispatches_offline(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--decision-backend",
            "offline",
            "auto",
            "Follow up with lead-1042",
        ],
    )
    assert main.main() == 0
    out = capsys.readouterr().out
    assert "outbound-followup" in out
    assert "lead-1042" in out


def test_route_decisions_flag_is_not_shadowed_by_subparser(monkeypatch, capsys):
    import gtm_agent.routing as routing

    seen = {}

    def fake_classify(query, *, use_decisions=None):
        seen["use_decisions"] = use_decisions
        return {
            "chosen_model": "openai/gpt-6-luna",
            "tier": "balanced",
            "reason": "classified",
            "tier_confidence": 0.9,
            "complexity": 1.0,
            "needs_planning": 0.0,
            "classifier": "fake",
            "classification_latency_ms": 1,
            "events": [],
        }

    monkeypatch.setattr(routing, "classify_query", fake_classify)
    monkeypatch.setattr(sys, "argv", ["main.py", "--decisions", "route", "hello"])
    assert main.main() == 0
    assert seen["use_decisions"] is True
