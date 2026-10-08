---
name: account-intelligence
description: >-
  Aggregate account-level signals (product usage, hiring, funding, expansion
  and risk flags) to show reps where to focus. Trigger on: account review,
  account intelligence, expansion opportunity, deal risk, renewal, weekly
  report, pipeline.
metadata:
  include_tools: query_product_usage get_hiring_signals get_funding_news get_sdk_installs get_web_traffic get_community_activity get_expansion_score get_churn_risk get_nps_score get_news_mentions get_funding_history get_competitor_mentions get_product_adoption_stage get_exec_changes get_integrations_usage
---

# Account Intelligence Playbook

For each account in scope, aggregate signals and produce a prioritized
briefing. Tailor the report to the audience:

## Sales briefing (default)

- Pull `query_product_usage` for adoption trends, SDK installs, credits
  remaining, and renewal timing.
- Pull `get_hiring_signals` — companies hiring AI engineers are strong
  expansion candidates.
- Pull `get_funding_news` — fresh funding usually means budget.

Surface, in priority order:

1. **Expansion opportunities** — rising usage + AI hiring + new launches.
2. **Deal risks** — credits running low, usage declining, renewal < 90 days.
3. **Competitive moves** — any public signal of evaluating alternatives.
4. **Who to contact** — the most engaged person and the suggested next touch.

## Format

Rank accounts by urgency. For each: one-line signal summary, the supporting
data points, and one recommended action. Flag anything that needs a human to
step in this week.
