from gtm_agent import tools
from gtm_agent.simulator import GTMSimulator
from main import GTMAgentRunner


def test_interpreter_draft_is_a_normal_tool_call():
    tools.reset_fixtures()
    model, outcome, _ = GTMSimulator().interpreter_outbound_script(
        "lead-1042", pinned=True
    )
    runner = GTMAgentRunner(interpreter=True, decision_backend="offline")
    result = runner._invoke(
        model, "Follow up on lead-1042.", pinned_skill="outbound-followup"
    )
    calls = [
        c["name"]
        for m in result["messages"]
        if m.type == "ai"
        for c in m.tool_calls
    ]
    assert outcome == "drafted"
    assert calls == ["eval", "draft_slack_followup"]
    assert len(tools._SLACK_DRAFTS) == 1


def test_blocked_lead_has_no_draft_tool_call():
    tools.reset_fixtures()
    model, outcome, _ = GTMSimulator().interpreter_outbound_script(
        "lead-1044", pinned=True
    )
    runner = GTMAgentRunner(interpreter=True, decision_backend="offline")
    result = runner._invoke(
        model, "Follow up on lead-1044.", pinned_skill="outbound-followup"
    )
    calls = [
        c["name"]
        for m in result["messages"]
        if m.type == "ai"
        for c in m.tool_calls
    ]
    assert outcome == "blocked"
    assert calls == ["eval"]
    assert tools._SLACK_DRAFTS == []
