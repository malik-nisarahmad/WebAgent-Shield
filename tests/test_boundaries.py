import asyncio
import socket
from unittest.mock import AsyncMock

import pytest
from webagent_shield.browser import Policy
from webagent_shield.extraction import extract, evidence, inspect
from webagent_shield.agent import Gateway


def test_hidden_comments_and_scripts_do_not_enter_evidence():
    passages = extract('<p>Useful fact</p><!-- comment --><div hidden>hidden</div><script>script</script>', 'https://example.com', 'rendered')
    scan = {"id": "scan", "url": "https://example.com", "coverage": {}, **inspect(passages)}
    assert [p["text"] for p in evidence(scan)["passages"]] == ["Useful fact"]
    assert any(p["kind"] == "comment" for p in passages)
    assert all(p["location"] for p in passages)
    assert scan["detector_status"] == "not_trained"


def test_private_address_denied(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args: [(2, 1, 6, '', ('127.0.0.1', 443))])
    with pytest.raises(ValueError):
        asyncio.run(Policy({"example.com"}).validate("https://example.com"))


@pytest.mark.parametrize("url", ["file:///secret", "https://evil.test", "https://example.com:9000", "https://user:pass@example.com"])
def test_scope_denied(url):
    with pytest.raises(ValueError):
        asyncio.run(Policy({"example.com"}).validate(url))


def test_unknown_tool_never_executes():
    client = AsyncMock()
    gateway = Gateway(Policy({"example.com"}), "test", client)
    result = asyncio.run(gateway.execute("read_file", {"path": "secret"}))
    assert "error" in result
    client.post.assert_not_called()
    assert gateway.events[0]["decision"] == "denied"


def test_evidence_truncation_is_reported():
    passages = extract("<p>Fact</p>" * 100, "https://example.com", "rendered")
    packet = evidence({"id": "scan", "url": "https://example.com", "coverage": {}, **inspect(passages)})
    assert len(packet["passages"]) == 80
    assert packet["coverage"]["partial"]


def test_attribute_locations_distinguish_elements():
    passages = extract('<img alt="first"><img alt="second">', "https://example.com", "raw")
    attributes = [p for p in passages if p["kind"] == "attribute"]
    assert len({p["location"] for p in attributes}) == 2


def test_gateway_returns_only_inspected_eligible_evidence(monkeypatch):
    import webagent_shield.agent as agent_module
    passages = extract('<p>Useful evidence</p><!-- diagnostic only -->', 'https://example.com', 'rendered')
    scan = {"id": "scan", "url": "https://example.com", "coverage": {}, **inspect(passages)}
    monkeypatch.setattr(agent_module, "scan_page", AsyncMock(return_value=scan))
    policy = AsyncMock()
    gateway = Gateway(policy, "test", AsyncMock())
    result = asyncio.run(gateway.execute("open_page", {"url": "https://example.com"}))
    assert [p["text"] for p in result["passages"]] == ["Useful evidence"]
    assert gateway.scans[0] == scan
    policy.validate.assert_awaited_once()


def test_api_health_and_missing_run():
    import httpx
    from webagent_shield.api import app

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/health")).json()["phase"] == 2
            assert (await client.get("/v1/agent-runs/nonexistent-test-run")).status_code == 404

    asyncio.run(check())


def test_rate_limit_retry(monkeypatch):
    import httpx
    import webagent_shield.agent as agent_module
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "test")
    monkeypatch.setattr(agent_module.asyncio, "sleep", AsyncMock())
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    client = AsyncMock()
    client.post.side_effect = [httpx.Response(429, request=request), httpx.Response(200, request=request)]
    response = asyncio.run(agent_module.request_completion(client, []))
    assert response.status_code == 200
    assert client.post.await_count == 2


def test_malformed_provider_tool_call_is_retried(monkeypatch):
    import httpx
    import webagent_shield.agent as agent_module
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "test")
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    client = AsyncMock()
    client.post.side_effect = [
        httpx.Response(400, request=request, json={"error": {"code": "tool_use_failed"}}),
        httpx.Response(200, request=request),
    ]
    response = asyncio.run(agent_module.request_completion(client, []))
    assert response.status_code == 200
    assert client.post.await_count == 2


def test_answer_is_forced_after_two_scans(monkeypatch):
    import webagent_shield.agent as agent_module
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "test")
    request = type("Client", (), {"post": AsyncMock(return_value=type("Response", (), {
        "status_code": 200, "raise_for_status": lambda self: None})())})()
    asyncio.run(agent_module.request_completion(request, [], require_answer=True))
    assert "tools" not in request.post.await_args.kwargs["json"]


def test_research_step_requires_a_tool(monkeypatch):
    import webagent_shield.agent as agent_module
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "test")
    request = type("Client", (), {"post": AsyncMock(return_value=type("Response", (), {
        "status_code": 200, "raise_for_status": lambda self: None})())})()
    asyncio.run(agent_module.request_completion(request, []))
    assert request.post.await_args.kwargs["json"]["tool_choice"] == "required"


def test_final_answer_context_has_sources_without_tool_history():
    import webagent_shield.agent as agent_module
    passage = {"id": "p", "text": "A useful fact", "url": "https://example.com",
               "channel": "rendered", "eligible": True}
    scan = {"id": "scan", "url": "https://example.com", "coverage": {},
            "detector_status": "not_trained", "passages": [passage]}
    messages = agent_module.final_answer_messages("Find the fact", [scan])
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "https://example.com" in messages[1]["content"]
    assert "A useful fact" in messages[1]["content"]


def test_missing_source_links_are_appended_once():
    import webagent_shield.agent as agent_module
    source = "https://example.com/page"
    answer = agent_module.add_source_links("Answer", [source])
    assert source in answer
    assert agent_module.add_source_links(answer, [source]).count(source) == 1


def test_final_evidence_is_relevant_and_balanced_across_sources():
    import json
    import webagent_shield.agent as agent_module

    def scan(url, texts):
        passages = [{"id": str(index), "text": text, "url": url, "channel": "rendered",
                     "eligible": True} for index, text in enumerate(texts)]
        return {"id": url, "url": url, "coverage": {}, "detector_status": "not_trained",
                "passages": passages}

    messages = agent_module.final_answer_messages(
        "Compare lists and tuples", [
            scan("https://one.test", ["navigation", "Lists are mutable"]),
            scan("https://two.test", ["navigation", "Tuples are immutable"]),
        ])
    documents = json.loads(messages[1]["content"])["documents"]
    assert documents[0]["excerpts"][0] == "Lists are mutable"
    assert documents[1]["excerpts"][0] == "Tuples are immutable"


def test_gateway_rejects_same_document_with_another_fragment(monkeypatch):
    import webagent_shield.agent as agent_module
    passage = {"id": "p", "text": "evidence", "url": "https://example.com/page",
               "channel": "rendered", "eligible": True}
    scan = {"id": "scan", "url": "https://example.com/page", "coverage": {},
            "detector_status": "not_trained", "passages": [passage]}
    monkeypatch.setattr(agent_module, "scan_page", AsyncMock(return_value=scan))
    policy = AsyncMock()
    gateway = Gateway(policy, "test", AsyncMock())
    first = asyncio.run(gateway.execute("open_page", {"url": "https://example.com/page#lists"}))
    second = asyncio.run(gateway.execute("open_page", {"url": "https://example.com/page#tuples"}))
    assert "passages" in first
    assert "error" in second
    assert "different document" in second["error"]
    assert gateway.pages == 1
