from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from decision_harness import SelectorUnavailable, ToolSelection
from gtm_agent import tools
from gtm_agent.selection import ConversationToolSelectionMiddleware, render_conversation


class StubSelector:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def select(self, task, tool_pairs, **kwargs):
        self.calls.append((task, tool_pairs, kwargs))
        if self.error:
            raise self.error
        return self.result


def _request(messages):
    return SimpleNamespace(messages=messages, tools=tools.ALL_TOOLS)


def test_render_conversation_includes_prior_tool_result():
    messages = [
        HumanMessage("Follow up with lead-1042"),
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "check_contact_history",
                    "args": {"lead_id": "lead-1042"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage("No prior touch", name="check_contact_history", tool_call_id="c1"),
    ]
    text = render_conversation(messages)
    assert "USER: Follow up with lead-1042" in text
    assert "check_contact_history" in text
    assert "No prior touch" in text


def test_middleware_passes_conversation_and_filters_candidates():
    selector = StubSelector(
        ToolSelection(
            selected=["check_contact_history"],
            probabilities={"check_contact_history": 0.9},
        )
    )
    middleware = ConversationToolSelectionMiddleware(
        selector=selector,
        candidate_names={"check_contact_history", "lookup_lead"},
        always_include=["read_file"],
        max_tools=2,
    )
    messages = [
        HumanMessage("Follow up with lead-1042"),
        ToolMessage("lead history", name="check_contact_history", tool_call_id="c1"),
    ]
    selected = middleware._select(_request(messages))
    assert selected == ["check_contact_history"]
    task, pairs, kwargs = selector.calls[0]
    assert "lead history" in task
    assert {name for name, _ in pairs} == {"check_contact_history", "lookup_lead"}
    assert kwargs["max_tools"] == 2
    assert kwargs["on_none"] == "all"


def test_selector_failure_keeps_all_tools():
    middleware = ConversationToolSelectionMiddleware(
        selector=StubSelector(error=SelectorUnavailable("down")),
        candidate_names={"lookup_lead"},
    )
    assert middleware._select(_request([HumanMessage("Follow up")])) is None


def test_no_tool_clears_threshold_keeps_all_tools():
    middleware = ConversationToolSelectionMiddleware(
        selector=StubSelector(
            ToolSelection(
                selected=["lookup_lead"],
                probabilities={"lookup_lead": 0.1},
                events=["no tool cleared threshold 0.50 -> keeping all (1) tools"],
            )
        ),
        candidate_names={"lookup_lead"},
    )
    assert middleware._select(_request([HumanMessage("Follow up")])) is None
