from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from gtm_agent.hitl import AutoApprover, interrupt_config, run_with_hitl


@tool
def draft_slack_followup(x: str) -> str:
    """Write a thing."""
    return f"wrote {x}"


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, bound_tools, **kwargs):
        return self


def make_agent():
    model = FakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "draft_slack_followup",
                            "args": {"x": "a"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    return create_agent(
        model,
        tools=[draft_slack_followup],
        middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_config(True))],
        checkpointer=InMemorySaver(),
    )


def test_approve_executes_once():
    agent = make_agent()
    result = run_with_hitl(
        agent,
        {"messages": [HumanMessage("write a")]},
        {"configurable": {"thread_id": "approve"}},
        AutoApprover(True),
    )
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "wrote a"


def test_reject_prevents_execution():
    agent = make_agent()
    result = run_with_hitl(
        agent,
        {"messages": [HumanMessage("write a")]},
        {"configurable": {"thread_id": "reject"}},
        AutoApprover(False),
    )
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert not [m for m in tool_messages if getattr(m, "status", None) != "error"]
    assert "wrote a" not in " ".join(str(m.content) for m in tool_messages)
