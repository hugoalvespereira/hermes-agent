"""Focused contract for the bounded Sol context-variant backport."""
from unittest.mock import Mock
import pytest
from agent import model_metadata as mm
from agent.transports.codex import ResponsesApiTransport


@pytest.mark.parametrize('slug,wire', [
    ('gpt-6-sol-900k', 'gpt-6-sol'),
    ('openai/gpt-6-sol-900k', 'openai/gpt-6-sol'),
    ('gpt-6-sol-2026-09-22-900k', 'gpt-6-sol-2026-09-22'),
    ('gpt-6-sol', 'gpt-6-sol'),
    ('gpt-6-sol-pro-900k', 'gpt-6-sol-pro-900k'),
    ('gpt-5.5-900k', 'gpt-5.5-900k'),
])
def test_wire_alias(slug, wire):
    assert mm.strip_codex_context_variant_suffix(slug) == wire


@pytest.mark.parametrize('slug,expected', [
    ('gpt-6-sol-900k', 900000), ('gpt-6-sol', 272000),
    ('openai/gpt-6-sol-900k', 900000),
    ('gpt-5.6-sol', 900000), ('gpt-5.5', 272000),
])
def test_fallback_and_existing_behavior(slug, expected):
    assert mm._resolve_codex_oauth_context_length(slug) == expected


@pytest.mark.parametrize('advertised,maximum,expected', [
    (272000, None, 900000), (272000, 500000, 500000),
    (272000, 1050000, 900000), (372000, 1050000, 372000),
])
def test_real_catalog_parser_and_cap(monkeypatch, advertised, maximum, expected):
    mm._codex_oauth_context_cache.clear()
    mm._codex_oauth_max_context_cache.clear()
    response = Mock(status_code=200)
    response.json.return_value = {'models': [{'slug': 'gpt-6-sol', 'context_window': advertised, 'max_context_window': maximum}]}
    monkeypatch.setattr(mm, '_ensure_requests', lambda: None)
    monkeypatch.setattr(mm, 'requests', Mock(get=Mock(return_value=response)))
    assert mm._resolve_codex_oauth_context_length('gpt-6-sol-900k', 'fixture') == expected
    assert mm._resolve_codex_oauth_context_length('gpt-6-sol-900k', 'fixture') == expected
    assert mm.requests.get.call_count == 1


@pytest.mark.parametrize('codex,expected', [(True, 'gpt-6-sol'), (False, 'gpt-6-sol-900k')])
def test_main_transport_and_override(codex, expected):
    request = ResponsesApiTransport().build_kwargs(
        model='gpt-6-sol-900k', messages=[{'role':'user','content':'ping'}],
        is_codex_backend=codex, reasoning_config={'effort':'medium'},
        request_overrides={'model':'gpt-6-sol-900k'},
    )
    assert request['model'] == expected
    assert request['reasoning']['effort'] == 'medium'
