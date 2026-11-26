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
