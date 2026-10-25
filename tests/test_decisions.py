import pytest

from gtm_agent.decisions import make_tool_selector, resolve_backend


def test_auto_without_key_is_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
    assert resolve_backend("auto") == "offline"


def test_auto_with_key_is_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert resolve_backend("auto") == "openai"
    assert resolve_backend("openai") == "openai"
    assert resolve_backend("offline") == "offline"


def test_explicit_openai_without_key_fails(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        resolve_backend("openai")


def test_offline_selector_keeps_every_tool():
    selector = make_tool_selector("offline")
    selection = selector.select(
        "Follow up with lead-1042",
        [("lookup_lead", "Look up a lead."), ("check_contact_history", "Check history.")],
        max_tools=None,
    )
    assert set(selection.selected) == {"lookup_lead", "check_contact_history"}
