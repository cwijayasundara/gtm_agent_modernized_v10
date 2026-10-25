from gtm_agent import tools
from gtm_agent.ptc import ptc_allowlist


def test_outbound_allowlist_is_read_only():
    allow = ptc_allowlist("outbound-followup")
    assert "check_contact_history" in allow
    assert "lookup_lead" in allow
    assert "draft_slack_followup" not in allow
    assert "queue_followup_sequence" not in allow


def test_prep_allowlist_contains_the_tools_the_script_uses():
    assert {"get_person_dossier", "get_last_interaction"} <= set(
        ptc_allowlist("meeting-prep")
    )


def test_no_skill_means_no_ptc_bridge():
    assert ptc_allowlist(None) == []


def test_every_allowlisted_name_is_a_real_tool():
    names = {tool.name for tool in tools.ALL_TOOLS}
    for skill in ("outbound-followup", "account-intelligence", "meeting-prep"):
        assert set(ptc_allowlist(skill)) <= names
