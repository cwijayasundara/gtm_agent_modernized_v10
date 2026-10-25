from gtm_agent import tools
from gtm_agent.skills import load_skills, skill_tool_names


def test_load_skills_reads_frontmatter_and_tools():
    skills = load_skills()
    assert set(skills) == {
        "account-intelligence",
        "competitor-analysis",
        "customer-health",
        "meeting-prep",
        "outbound-followup",
    }
    assert len(skills["outbound-followup"].include_tools) == 12
    assert len(skills["meeting-prep"].include_tools) == 8
    assert skills["competitor-analysis"].include_tools == ()
    assert "follow-up" in skills["outbound-followup"].description.lower()


def test_skill_tool_names_is_ordered_and_empty_for_unknown():
    assert skill_tool_names("outbound-followup")[:3] == [
        "lookup_lead",
        "check_contact_history",
        "get_gong_transcripts",
    ]
    assert skill_tool_names(None) == []
    assert skill_tool_names("does-not-exist") == []


def test_side_effect_metadata_is_explicit_and_deferred_count_unchanged():
    by_name = {tool.name: tool for tool in tools.ALL_TOOLS}
    assert by_name["draft_slack_followup"].extras["side_effect"] is True
    assert by_name["queue_followup_sequence"].extras["side_effect"] is True
    assert by_name["lookup_lead"].extras.get("side_effect") is not True
    deferred = [t for t in tools.ALL_TOOLS if (t.extras or {}).get("defer_loading")]
    assert len(deferred) == 47
