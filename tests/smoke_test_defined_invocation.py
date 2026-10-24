"""Smoke test: defined tool invocation (no agent loop) for the GTM workflows.

The GTM agent no longer has to discover every tool on its own. These runners
invoke the mocked tools directly, in the playbook's defined order, and assert
the same gates the agent loop enforces (do-not-send, no-recent-touch, draft
queueing for rep approval).

Run:  .venv/bin/python tests/smoke_test_defined_invocation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gtm_agent import tools  # noqa: E402
from gtm_agent.direct import run_account_briefing, run_meeting_prep, run_outbound_followup  # noqa: E402


def main() -> None:
    # --- outbound follow-up: clear lead -> research -> draft -> queue -------
    tools.reset_fixtures()
    result = run_outbound_followup("lead-1042")

    assert result["outcome"] == "drafted", result
    assert result["lead_id"] == "lead-1042"
    assert result["draft_queued"] is True
    assert "Priya" in result["draft"], result["draft"]
    assert len(tools._SLACK_DRAFTS) == 1
    assert tools._SLACK_DRAFTS[0]["status"] == "pending_approval"

    # --- outbound follow-up: recent touch -> blocked, nothing drafted -------
    tools.reset_fixtures()
    result = run_outbound_followup("lead-1043")
    assert result["outcome"] == "blocked", result
    assert result["draft"] is None
    assert len(tools._SLACK_DRAFTS) == 0

    # --- outbound follow-up: do-not-send -> blocked -----------------------
    tools.reset_fixtures()
    result = run_outbound_followup("lead-1044")
    assert result["outcome"] == "blocked", result
    assert result["draft"] is None
    assert len(tools._SLACK_DRAFTS) == 0

    # --- account intelligence briefing ------------------------------------
    result = run_account_briefing("Helix Health")
    assert result["outcome"] == "ok", result
    assert "Account briefing" in result["briefing"]

    # --- meeting prep -------------------------------------------------------
    result = run_meeting_prep("Priya Sharma")
    assert result["outcome"] == "ok", result
    assert "Meeting prep" in result["briefing"]

    print("✅ defined invocation: outbound gates + intel + prep workflows all ran without an agent loop")


if __name__ == "__main__":
    main()