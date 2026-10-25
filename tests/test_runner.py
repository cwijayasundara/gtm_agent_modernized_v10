from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from decision_harness import ToolSelection
from gtm_agent import tools
from gtm_agent.hitl import AutoApprover
from gtm_agent.simulator import GTMSimulator, ScriptedModel, _call
from main import GTMAgentRunner


class StubSelector:
    def select(self, task, tool_pairs, **kwargs):
        return ToolSelection(
            selected=["check_contact_history"],
            probabilities={"check_contact_history": 0.9},
        )


class EvalModel(GenericFakeChatModel):
    def bind_tools(self, bound_tools, **kwargs):
        return self


def test_selector_runner_binds_only_selected_tool():
    model = ScriptedModel(
        messages=iter(
            [
                _call("check_contact_history", {"lead_id": "lead-1042"}, "c1"),
                AIMessage(content="done"),
            ]
        )
    )
    runner = GTMAgentRunner(
        selector=True, tool_selector=StubSelector(), decision_backend="offline"
    )
    result = runner._invoke(
        model, "Follow up on lead-1042.", pinned_skill="outbound-followup"
    )
    assert "check_contact_history" in str(result["messages"])


def test_interpreter_runner_uses_skill_allowlist():
    model = EvalModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "eval",
                            "args": {
                                "code": 'const r = await tools.checkContactHistory({lead_id: "lead-1042"}); r;'
                            },
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    runner = GTMAgentRunner(interpreter=True, decision_backend="offline")
    runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    assert "draft_slack_followup" not in runner.last_ptc_allowlist
    assert "check_contact_history" in runner.last_ptc_allowlist


def test_approve_queues_once_and_reject_does_not():
    tools.reset_fixtures()
    model, _, _ = GTMSimulator().outbound_script("lead-1042", pinned=True)
    runner = GTMAgentRunner(
        approve=True, reviewer=AutoApprover(True), decision_backend="offline"
    )
    runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    assert len(tools._SLACK_DRAFTS) == 1

    tools.reset_fixtures()
    model, _, _ = GTMSimulator().outbound_script("lead-1042", pinned=True)
    runner = GTMAgentRunner(
        approve=True, reviewer=AutoApprover(False), decision_backend="offline"
    )
    runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    assert tools._SLACK_DRAFTS == []


def test_hitl_decisions_are_recorded():
    tools.reset_fixtures()
    model, _, _ = GTMSimulator().outbound_script("lead-1042", pinned=True)
    runner = GTMAgentRunner(
        approve=True, reviewer=AutoApprover(True), decision_backend="offline"
    )
    runner._invoke(model, "Follow up on lead-1042.", pinned_skill="outbound-followup")
    assert [decision["type"] for decision in runner.last_hitl_decisions] == ["approve"]
