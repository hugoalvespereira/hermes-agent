"""OOPIF session lookup that lets the vault fill a form inside a cross-origin iframe."""

import asyncio

from tools.browser_supervisor import CDPSupervisor
from tools.browser_supervisor_frames import FrameInfo


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
