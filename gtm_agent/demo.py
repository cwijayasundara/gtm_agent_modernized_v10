"""Run the sample GTM agent on a mock inbound-lead trigger.

Requires ANTHROPIC_API_KEY in the environment (Claude Haiku 5.5).

    python -m gtm_agent.demo
"""

from __future__ import annotations

from .agent import build_agent


def main() -> None:
    agent = build_agent()

    trigger = (
        "New inbound lead just landed in Salesforce: lead-1042 "
        "(Priya Sharma, VP Engineering at Northwind Analytics, webinar sign-up, "
        "intent score 72). Run the standard follow-up process for this lead."
    )
    print(f"Trigger: {trigger}\n{'-' * 70}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": trigger}]},
        config={"recursion_limit": 40},
    )
    print("-" * 70)
    print("Agent's final message:\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
