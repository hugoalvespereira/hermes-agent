"""Initial page attach: a live default page is fully set up before ``start()`` reports ready, while a background
tab that answers browser-side CDP commands but never ``Page.enable`` must not fail the start. No real Chrome."""

from __future__ import annotations

import asyncio

import pytest

from tests.waiting import async_wait_until
from tools import browser_supervisor as bs

SETUP = ["Page.enable", "Runtime.enable", "Target.setAutoAttach",
         "Page.addScriptToEvaluateOnNewDocument", "Fetch.enable", "Runtime.evaluate"]


def _supervisor(on_page_enable=None):
    sup = bs.CDPSupervisor(task_id="t", cdp_url="ws://127.0.0.1:1/devtools/browser/x")
    calls = []

    async def cdp(method, params=None, *, session_id=None, timeout=10.0):
        calls.append(method)
        await asyncio.sleep(0)  # each reply arrives on a later loop turn, as from the reader
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"type": "page", "targetId": "T1", "url": "https://exn.sharepoint.com/"}]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "page-1"}}
        if method == "Page.enable" and on_page_enable:
            await on_page_enable()
        return {"result": {}}

    sup._cdp = cdp
    return sup, calls


def test_initial_attach_completes_live_page_setup_before_ready():
    sup, calls = _supervisor()
    asyncio.run(sup._attach_initial_page())
    assert calls == ["Target.getTargets", "Target.attachToTarget", *SETUP]
    assert sup._page_session_id == "page-1"


def test_initial_attach_setup_error_still_fails_start():
    async def fail():
        raise RuntimeError("CDP error on id=3: {'message': 'Page.enable failed'}")

    sup, _ = _supervisor(fail)
    with pytest.raises(RuntimeError, match="Page.enable failed"):
        asyncio.run(sup._attach_initial_page())


def test_initial_attach_returns_after_bound_on_a_stalled_page(monkeypatch):
    monkeypatch.setattr(bs, "INITIAL_PAGE_SETUP_TIMEOUT_S", 0.05)

    async def run():
        answered = asyncio.Event()
        sup, calls = _supervisor(answered.wait)
        await sup._attach_initial_page()
        assert calls == ["Target.getTargets", "Target.attachToTarget", "Page.enable"]
        assert sup._page_session_id == "page-1"
        answered.set()  # the tab answers later: the setup still completes
        await async_wait_until(lambda: len(calls) == 2 + len(SETUP), what="late initial page setup")
        assert calls[2:] == SETUP

    asyncio.run(run())
