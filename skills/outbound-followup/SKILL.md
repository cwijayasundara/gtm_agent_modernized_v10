---
name: outbound-followup
description: >-
  Playbook for qualifying and following up on inbound leads and writing
  personalized outreach drafts. Trigger on: new lead, inbound lead,
  follow-up, outreach, email draft, SDR, prospect.
metadata:
  include_tools: lookup_lead check_contact_history get_gong_transcripts web_search_company draft_slack_followup get_contacts_at_account search_linkedin_profile get_email_thread_history get_marketing_touches get_calendar_availability check_domain_reputation queue_followup_sequence
---

# Outbound Follow-up Playbook

Follow this exact checklist for every lead, in order. Do not skip the
do-not-send checks — a single poorly timed email can undo months of
relationship building.

## 1. Do-not-send checks (ALWAYS first)

- Call `check_contact_history` before anything else.
- If `do_not_send` is true (e.g. open support ticket), STOP. Report why you
  are not drafting and recommend a human touch instead.
- If a teammate reached out within the last 3 days, STOP. Draft nothing;
  suggest the rep coordinate with the teammate instead.

## 2. Research (batched — one turn)

Fire all independent lookups in a single parallel batch: `lookup_lead`,
`get_gong_transcripts`, `get_marketing_touches`, and `web_search_company`
(once `lookup_lead` gives you the company name). Do not spread these across
separate turns.

Use in the draft only the lookups that returned useful data; ignore mock or
empty results rather than re-querying.

## 3. Draft (relationship-aware)

The draft must reflect the account state:

- **Customer** → expansion angle. Reference product usage, recent calls, and
  outcomes. Warm, specific, outcome-first.
- **Warm prospect** → momentum angle. Reference their trial or prior
  conversation and propose one concrete next step.
- **Cold** → brief, research-backed opener. Two to three sentences, one
  specific observation about *their* business, one soft ask. No fluff.

## 4. Deliver for approval (human-in-the-loop)

- Call `draft_slack_followup` with the draft and a short `rationale`
  explaining why this angle was chosen, so the rep can refine it.
- NEVER send anything directly. Every draft requires explicit rep approval.
