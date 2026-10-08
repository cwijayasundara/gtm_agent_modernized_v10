---
name: customer-health
description: >-
  Account health briefing for deployed engineers and customer success: usage
  trends, open tickets, escalations, renewals, credit levels. Trigger on:
  account health, customer health, support tickets, renewal risk, QBR,
  escalation, credits, deployed engineer briefing.
metadata:
  include_tools: get_open_tickets get_support_sla get_recent_calls_summary get_renewal_date get_credit_balance get_license_utilization get_onboarding_status get_open_action_items get_feature_requests get_uptime_status get_escalations get_qbr_notes
---

# Customer Health Playbook

Produce an account-health briefing that flags what actually needs a human
to step in, so nobody spends Sunday evenings digging through dashboards.

1. Pull `get_open_tickets` and `get_escalations` first — open incidents
   change everything else (no commercial asks while a SEV is open).
2. Pull `get_renewal_date` and `get_credit_balance` — renewals < 90 days or
   credits < 25% are action items.
3. Pull `get_recent_calls_summary`, `get_qbr_notes`, and
   `get_open_action_items` — unresolved promises are silent churn risk.
4. Optionally add `get_license_utilization`, `get_onboarding_status`,
   `get_feature_requests`, `get_uptime_status`, `get_support_sla`.

## Output format

- **Needs a human this week** — incidents, escalations, broken promises.
- **Renewal outlook** — date, risk level, and pre-renewal to-dos.
- **Adoption signals** — utilization, onboarding, requests worth productizing.
