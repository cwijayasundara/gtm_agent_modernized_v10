---
name: meeting-prep
description: >-
  Prepare for an upcoming call or meeting with a contact or account: who
  they are, where things left off, and what to ask. Trigger on: meeting
  prep, prepare for a call, upcoming meeting, discovery call, QBR prep,
  agenda, briefing before a meeting.
metadata:
  include_tools: get_last_interaction get_open_threads get_mutual_connections get_account_playbook get_person_dossier get_upcoming_events get_pitch_deck_latest get_agenda_template
---

# Meeting Prep Playbook

Build a one-page brief for the rep before a customer touchpoint.

1. Pull `get_last_interaction` and `get_open_threads` — know exactly where
   the conversation left off and what is still unanswered.
2. Pull `get_person_dossier` for each attendee; `get_mutual_connections`
   for warm rapport options.
3. Pull `get_account_playbook` to align the talking points with the account
   plan, and `get_pitch_deck_latest` to see which materials they've seen.
4. Close with a proposed agenda from `get_agenda_template` and 3 sharp
   questions tailored to the contact's role and priorities.
