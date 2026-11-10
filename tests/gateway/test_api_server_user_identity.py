"""OpenAI ``user`` field on /v1/chat/completions: validated, threaded to
AIAgent(user_id=...) for per-user memory, and echoed in X-Hermes-User."""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import (
    _create_app,
    _make_adapter,
    _make_routing_adapter,
    _patch_create_agent_runtime,
)


def _fake_create_agent(captured):
    def _create(**kwargs):
        captured.update(kwargs)
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "ok", "messages": []}
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        return agent
    return _create


async def _post(adapter, body, captured):
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent(captured)):
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-secret"},
                json={"messages": [{"role": "user", "content": "hi"}], **body},
            )
            await resp.read()
            return resp


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_valid_user_threads_to_agent_and_is_echoed(stream):
    captured = {}
    resp = await _post(_make_adapter(api_key="sk-secret"), {"user": "hugo", "stream": stream}, captured)
    assert resp.status == 200
    assert captured["user_id"] == "hugo"
    assert resp.headers.get("X-Hermes-User") == "hugo"


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["Hugo", "a b", "a" * 65, 7, ""])
async def test_invalid_user_is_rejected(user):
    captured = {}
    resp = await _post(_make_adapter(api_key="sk-secret"), {"user": user}, captured)
    assert resp.status == 400
    assert captured == {}


@pytest.mark.asyncio
async def test_without_user_no_user_id_and_no_header():
    captured = {}
    resp = await _post(_make_adapter(api_key="sk-secret"), {}, captured)
    assert resp.status == 200
    assert captured["user_id"] is None
    assert "X-Hermes-User" not in resp.headers


def test_create_agent_passes_user_id_to_aiagent(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
    adapter = _make_routing_adapter({})
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

    adapter._create_agent(session_id="s1", user_id="hugo")

    assert captured["user_id"] == "hugo"
