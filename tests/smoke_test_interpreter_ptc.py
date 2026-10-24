"""Smoke test: deep agent that invokes tools through the code interpreter (PTC).

Instead of the agent loop handing the model a serial chain of tool calls, the
agent is given a programmatic tool calling surface:

    build_interpreter_agent() -> create_deep_agent(..., middleware=[CodeInterpreterMiddleware(ptc=[...])])

The model writes a small JS program via the interpreter's `eval` tool; the
harness bridges the allowlisted tools into the `tools` namespace and executes
the program.

Run:  .venv/bin/python tests/smoke_test_interpreter_ptc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gtm_agent.agent import build_interpreter_agent  # noqa: E402


EVAL_SCRIPT = 'const r = await tools.lookupLead({ lead_id: "lead-1042" }); r;'


class EvalScriptedModel(GenericFakeChatModel):
    """Fake model that first writes (and runs) an eval call, then answers."""

    def __init__(self) -> None:
        super().__init__(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "eval", "args": {"code": EVAL_SCRIPT}, "id": "c1", "type": "tool_call"}
                        ],
                    ),
                    AIMessage(content="Done — I looked up the lead via the interpreter."),
                ]
            )
        )

    def bind_tools(self, bound_tools, **kwargs):  # type: ignore[no-untyped-def]
        return self  # scripted model ignores tool schemas


def main() -> None:
    agent = build_interpreter_agent(model=EvalScriptedModel())
    result = agent.invoke(
        {"messages": [HumanMessage(content="Follow up on lead-1042.")]},
        config={"recursion_limit": 25},
        context={"run_id": "smoke-interpreter-ptc"},
    )

    msgs = result["messages"]
    eval_msgs = [m for m in msgs if getattr(m, "name", None) == "eval"]
    assert len(eval_msgs) == 1, f"expected exactly one eval tool call, got {[m.type for m in msgs]}"

    text = str(eval_msgs[0].content or "")
    assert text.startswith("<result>") and text.rstrip("</result>").strip(), f"unexpected eval result: {text!r}"

    payload = json.loads(text.strip().strip("</result>").strip())
    assert payload.get("lead_id") == "lead-1042", payload
    assert payload.get("contact") == "Priya Sharma", payload

    ai_messages = [m for m in msgs if m.type == "ai"]
    assert ai_messages, "expected a final model turn after the interpreter result"

    print("✅ interpreter PTC: model-authored eval bridged an allowlisted tool and got the lead back")


if __name__ == "__main__":
    main()