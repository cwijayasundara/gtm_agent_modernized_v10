"""Smoke test: prove that "binding tools to skills" works, without an API key.

A scripted fake chat model (1) reads the outbound-followup SKILL.md, then
(2) follows the playbook by calling `check_contact_history`. We record the
tool list the harness binds on every model call and assert that deferred
skill tools are only disclosed (un-deferred, schema sent to the provider)
AFTER the skill that lists them is read.

Run:  .venv/bin/python tests/smoke_test_skill_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gtm_agent.agent import build_agent  # noqa: E402

# (tool name, is still deferred) recorded per model call
BOUND: list[list[tuple[str, bool]]] = []


def _snapshot(tools) -> list[tuple[str, bool]]:  # type: ignore[no-untyped-def]
    out = []
    for t in tools:
        name = t.get("name", "?") if isinstance(t, dict) else t.name
        deferred = False if isinstance(t, dict) else bool((t.extras or {}).get("defer_loading"))
        out.append((name, deferred))
    return sorted(out)


class RecordingFakeModel(GenericFakeChatModel):
    """Fake model that records the tools bound on each call."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        BOUND.append(_snapshot(tools))
        return self  # scripted model ignores tool schemas


OUTBOUND_SKILL_TOOLS = {"lookup_lead", "check_contact_history", "get_gong_transcripts", "web_search_company", "draft_slack_followup"}
INTEL_SKILL_TOOLS = {"query_product_usage", "get_hiring_signals", "get_funding_news"}


def main() -> None:
    scripted = [
        # Turn 1: model loads the outbound-followup skill...
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"file_path": "/skills/outbound-followup/SKILL.md"}, "id": "call_1", "type": "tool_call"}
            ],
        ),
        # ...then, following the playbook, runs the do-not-send check FIRST.
        AIMessage(
            content="",
            tool_calls=[
                {"name": "check_contact_history", "args": {"lead_id": "lead-1042"}, "id": "call_2", "type": "tool_call"}
            ],
        ),
        AIMessage(content="Playbook followed: no do-not-send flags, no prior contact. Draft queued for rep approval."),
    ]
    model = RecordingFakeModel(messages=iter(scripted))
    agent = build_agent(model=model)

    result = agent.invoke(
        {"messages": [HumanMessage(content="Follow up on lead-1042.")]},
        config={"recursion_limit": 15},
    )

    names_by_call = [{n for n, _ in call} for call in BOUND]
    deferred_by_call = [{n for n, d in call if d} for call in BOUND]

    # 1. Built-ins (read_file, task, ...) are always available.
    assert "read_file" in names_by_call[0], names_by_call[0]
    # 2. Before the skill is read, ALL deferred GTM tools are withheld from the provider.
    assert OUTBOUND_SKILL_TOOLS | INTEL_SKILL_TOOLS <= deferred_by_call[0], deferred_by_call[0]
    # 3. After reading the SKILL.md, the outbound skill's tools are disclosed (schemas sent)...
    assert OUTBOUND_SKILL_TOOLS.isdisjoint(deferred_by_call[1]), f"still deferred after skill read: {deferred_by_call[1]}"
    # 4. ...while the account-intelligence skill's tools stay withheld.
    assert INTEL_SKILL_TOOLS <= deferred_by_call[1], f"unrelated skill tools leaked: {INTEL_SKILL_TOOLS - deferred_by_call[1]}"
    # 5. The previously-deferred tool call actually executed.
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert any(m.name == "check_contact_history" for m in tool_messages), "deferred tool call never executed"

    print("model call 1 (before skill read)  ->", len(BOUND[0]), "tools,", len(deferred_by_call[0]), "still deferred")
    print("model call 2 (after SKILL.md read) ->", len(BOUND[1]), "tools,", len(deferred_by_call[1]), "still deferred")
    print("\n✅ PASS: deferred tools disclosed only after reading the SKILL.md that lists them;" 
          " other skills' tools stay hidden.")


if __name__ == "__main__":
    main()
