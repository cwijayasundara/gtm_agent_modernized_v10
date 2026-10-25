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
