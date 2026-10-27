"""Frame lookup that lets the vault fill a form inside a cross-origin iframe, in-process or OOPIF."""

import asyncio
import threading

from tools.browser_supervisor import CDPSupervisor
from tools.browser_supervisor_frames import FrameInfo, FrameTarget, _may_share_site


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


LOGIN_TAB, CSFE_OOPIF, X_TAB = "248F3859AF8ED00AFA4F73E7AC19817C", "FC1BEB49C3887778EDA05508644F9406", "CA6F9DF808A61707491DE8252987C50D"
ACCEPT = "!!document.querySelector('input[type=password]')"


def _page(target_id, url):
    return {"type": "page", "targetId": target_id, "url": url}


def _iframe(target_id, url, parent):
    return {"type": "iframe", "targetId": target_id, "url": url, "parentFrameId": parent}


def _focus(sup, cdp, origin, *, timeout=10.0):
    """Run ``focus_page`` against the fake ``cdp`` on a real supervisor loop thread."""
    sup._cdp = cdp
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    sup._loop = loop
    try:
        return sup.focus_page(origin, accept=ACCEPT, timeout=timeout)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)


def test_may_share_site_never_rejects_a_same_site_url():
    assert _may_share_site("https://login.bankid.no/x", "https://csfe.bankid.no")
    assert _may_share_site("https://bankid.no:8443/", "https://csfe.bankid.no")
    assert not _may_share_site("http://login.bankid.no/", "https://csfe.bankid.no")
    assert not _may_share_site("https://x.com/home", "https://csfe.bankid.no")
    assert not _may_share_site("https://exn.sharepoint.com/", "https://csfe.bankid.no")
    assert _may_share_site("https://a.co.uk/", "https://b.co.uk")  # an extra candidate, never a missed one
    assert _may_share_site("http://10.0.0.1:8080/", "http://10.0.0.1")
    assert not _may_share_site("http://10.0.0.2/", "http://10.0.0.1")
    assert _may_share_site("http://localhost:3000/", "http://localhost")
    assert not _may_share_site("http://intranet/", "http://localhost")


def test_candidate_pages_links_an_oopif_to_its_tab_by_parent_frame_id():
    targets = [
        _page("SP", "https://exn.sharepoint.com/sites/x"),
        _page(X_TAB, "https://x.com/home"),
        _page(LOGIN_TAB, "https://login.bankid.no/"),
        _iframe(CSFE_OOPIF, "https://csfe.bankid.no/login", LOGIN_TAB),
    ]
    assert CDPSupervisor._candidate_pages(targets, "https://csfe.bankid.no") == [LOGIN_TAB]


def test_candidate_pages_keeps_same_site_tabs_for_in_process_frames():
    targets = [
        _page("SP", "https://exn.sharepoint.com/"),
        _page("SAME_SITE", "https://login.bankid.no/"),
        _page("HOST", "https://shop.example/"),
        _iframe("OOPIF", "https://csfe.bankid.no/login", "HOST"),
        _page(X_TAB, "https://x.com/"),
    ]
    assert CDPSupervisor._candidate_pages(targets, "https://csfe.bankid.no") == ["HOST", "SAME_SITE"]


def test_candidate_pages_keeps_every_tab_when_a_matching_oopif_cannot_be_placed():
    targets = [
        _page("SP", "https://exn.sharepoint.com/"),
        _page("SAME_SITE", "https://www.bankid.no/"),
        _iframe("OOPIF", "https://csfe.bankid.no/login", "in-process-subframe"),
        _page(X_TAB, "https://x.com/"),
        _page("BLANK", "about:blank"),
    ]
    assert CDPSupervisor._candidate_pages(targets, "https://csfe.bankid.no") == ["SAME_SITE", "SP", X_TAB]


def test_candidate_pages_any_origin_keeps_every_http_tab():
    targets = [_page("A", "https://x.com/"), _page("B", "chrome://newtab/"), _page("C", "http://localhost/"),
               _iframe("F", "https://ads.example/", "A")]
    assert CDPSupervisor._candidate_pages(targets, "") == ["A", "C"]


def test_focus_page_never_touches_unrelated_stalled_tabs():
    """160 tabs; every tab but the BankID one (index 57) answers browser-side commands only, as background tabs do."""
    sup = CDPSupervisor(task_id="t", cdp_url="ws://127.0.0.1:1/devtools/browser/x")
    pages = [_page(f"TAB{i:03d}", f"https://site{i}.example/") for i in range(160)]
    pages[0] = _page("SP", "https://exn.sharepoint.com/")
    pages[12] = _page(X_TAB, "https://x.com/home")
    pages[57] = _page(LOGIN_TAB, "https://login.bankid.no/")
    targets = pages + [_iframe(CSFE_OOPIF, "https://csfe.bankid.no/login", LOGIN_TAB)]
    trees = {"login-s": ("top", "https://login.bankid.no/"), "csfe-s": (CSFE_OOPIF, "https://csfe.bankid.no/login")}
    attached = []

    async def cdp(method, params=None, *, session_id=None, timeout=10.0):
        params = params or {}
        if method == "Target.getTargets":
            return {"result": {"targetInfos": targets}}
        if method == "Target.attachToTarget":
            attached.append(params["targetId"])
            return {"result": {"sessionId": "login-s" if params["targetId"] == LOGIN_TAB else "stalled-s"}}
        if session_id == "stalled-s":
            await asyncio.Event().wait()  # a background tab never answers a renderer command
        if method == "Target.setAutoAttach":  # Chrome reports the tab's existing OOPIF before replying
            sup._frames[CSFE_OOPIF] = _oopif(CSFE_OOPIF, "csfe-s", "login-s", "https://csfe.bankid.no/login")
        if method == "Page.getFrameTree":
            frame_id, url = trees[session_id]
            return {"result": {"frameTree": {"frame": {"id": frame_id, "url": url}}}}
        if method == "Page.createIsolatedWorld":
            return {"result": {"executionContextId": 7}}
        if method == "Runtime.evaluate" and "contextId" in params:
            return {"result": {"result": {"value": session_id == "csfe-s"}}}
        return {"result": {}}

    out = _focus(sup, cdp, "https://csfe.bankid.no", timeout=2.0)

    assert out == {"ok": True, "url": "https://csfe.bankid.no/login", "frame": FrameTarget("csfe-s", 7)}
    assert attached == [LOGIN_TAB]
    assert sup._page_session_id == "login-s"


def test_focus_page_skips_a_stalled_candidate_tab():
    """Two same-site candidates; the first never answers Page.enable. It is skipped and detached, not fatal."""
    sup = CDPSupervisor(task_id="t", cdp_url="ws://127.0.0.1:1/devtools/browser/x")
    targets = [_page("STALLED", "https://www.bankid.no/"), _page("LIVE", "https://login.bankid.no/")]
    detached = []

    async def cdp(method, params=None, *, session_id=None, timeout=10.0):
        params = params or {}
        if method == "Target.getTargets":
            return {"result": {"targetInfos": targets}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": f"s-{params['targetId']}"}}
        if method == "Target.detachFromTarget":
            detached.append(params["sessionId"])
            return {"result": {}}
        if session_id == "s-STALLED":
            await asyncio.Event().wait()
        if method == "Page.getFrameTree":
            return {"result": {"frameTree": {
                "frame": {"id": "top", "url": "https://login.bankid.no/"},
                "childFrames": [{"frame": {"id": "child", "url": "https://csfe.bankid.no/login", "parentId": "top"}}]}}}
        if method == "Page.createIsolatedWorld":
            return {"result": {"executionContextId": 3}}
        if method == "Runtime.evaluate" and "contextId" in params:
            return {"result": {"result": {"value": True}}}
        return {"result": {}}

    out = _focus(sup, cdp, "https://csfe.bankid.no", timeout=0.5)

    assert out == {"ok": True, "url": "https://csfe.bankid.no/login", "frame": FrameTarget("s-LIVE", 3)}
    assert detached == ["s-STALLED"]
    assert sup._page_session_id == "s-LIVE"
