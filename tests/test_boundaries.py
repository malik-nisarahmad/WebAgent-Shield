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
