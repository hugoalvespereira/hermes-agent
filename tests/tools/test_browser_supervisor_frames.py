"""Frame lookup that lets the vault fill a form inside a cross-origin iframe, in-process or OOPIF."""

import asyncio
import threading

from tools.browser_supervisor import CDPSupervisor
from tools.browser_supervisor_frames import FrameInfo, FrameTarget


def _oopif(frame_id, sid, parent_sid, url):
    return FrameInfo(frame_id=frame_id, url=url, origin="", parent_frame_id=None, is_oopif=True,
                     cdp_session_id=sid, parent_session_id=parent_sid)


def test_oopif_sessions_returns_only_live_frames_rooted_at_the_page():
    sup = CDPSupervisor(task_id="t", cdp_url="ws://127.0.0.1:1/devtools/browser/x")
    for f in (
        _oopif("f1", "child-a", "page-A", "https://csfe.bankid.no/login"),
        _oopif("f2", "grandchild-a", "child-a", "https://nested.example/"),
        _oopif("f3", "child-b", "page-B", "https://other.example/"),
        _oopif("f4", None, "page-A", "https://detached.example/"),
        _oopif("f5", "orphan", "gone-session", "https://orphan.example/"),
    ):
        sup._frames[f.frame_id] = f

    assert sorted(sup.oopif_sessions("page-A")) == [
        ("child-a", "https://csfe.bankid.no/login"),
        ("grandchild-a", "https://nested.example/"),
    ]


def test_target_attach_records_the_embedder_session():
    """``Page.frameAttached`` is not replayed on a late attach, so the auto-attach event's own session is the parent link."""
    sup = CDPSupervisor(task_id="t", cdp_url="ws://127.0.0.1:1/devtools/browser/x")
    sup._enable_child_domains = lambda sid: asyncio.sleep(0)

    async def run():
        await sup._on_target_attached({"sessionId": "child-1", "targetInfo": {
            "type": "iframe", "targetId": "f1", "url": "https://csfe.bankid.no/"}}, "page-A")
        sup._on_frame_navigated({"frame": {"id": "f1", "url": "https://csfe.bankid.no/login"}}, "child-1")

    asyncio.run(run())
    assert sup.oopif_sessions("page-A") == [("child-1", "https://csfe.bankid.no/login")]


def test_focus_page_resolves_a_same_site_iframe_that_has_no_session_of_its_own():
    """login.bankid.no embeds csfe.bankid.no: same site, so Chrome renders the iframe in the tab's process and
    it never gets a CDP session. focus_page must still find it and hand back a context bound to that frame."""
    sup = CDPSupervisor(task_id="t", cdp_url="ws://127.0.0.1:1/devtools/browser/x")
    contexts = {"top": 1, "child": 2}
    probes = []

    async def cdp(method, params=None, *, session_id=None, timeout=10.0):
        params = params or {}
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"type": "page", "targetId": "T1", "url": "https://login.bankid.no/"}]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "page-A"}}
        if method == "Page.getFrameTree":
            return {"result": {"frameTree": {
                "frame": {"id": "top", "url": "https://login.bankid.no/"},
                "childFrames": [{"frame": {"id": "child", "url": "https://csfe.bankid.no/login", "parentId": "top"}}]}}}
        if method == "Page.createIsolatedWorld":
            return {"result": {"executionContextId": contexts[params["frameId"]]}}
        if method == "Runtime.evaluate" and "contextId" in params:
            probes.append(params["contextId"])
            return {"result": {"result": {"value": params["contextId"] == contexts["child"]}}}
        return {"result": {}}

    sup._cdp = cdp
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    sup._loop = loop
    try:
        out = sup.focus_page("https://csfe.bankid.no", accept="!!document.querySelector('input[type=password]')")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)

    assert out == {"ok": True, "url": "https://csfe.bankid.no/login", "frame": FrameTarget("page-A", contexts["child"])}
    assert probes == [contexts["child"]]  # the origin-mismatched top document was never probed
    assert sup._page_session_id == "page-A"
