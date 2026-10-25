from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from gtm_agent.agent import build_interpreter_agent


class EvalModel(GenericFakeChatModel):
    def bind_tools(self, bound_tools, **kwargs):
        return self


def test_interpreter_builder_uses_skill_allowlist():
    script = 'const r = await tools.getPersonDossier({ query: "Priya Sharma" }); r;'
    model = EvalModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "eval",
                            "args": {"code": script},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    agent = build_interpreter_agent(model=model, skill="meeting-prep")
    result = agent.invoke(
        {"messages": [HumanMessage("prep")], "pinned_skills": ["meeting-prep"]},
        config={"recursion_limit": 20},
    )
    eval_msgs = [m for m in result["messages"] if getattr(m, "name", None) == "eval"]
    assert eval_msgs and "VP Engineering" in str(eval_msgs[0].content)
    assert "not a function" not in str(eval_msgs[0].content)
