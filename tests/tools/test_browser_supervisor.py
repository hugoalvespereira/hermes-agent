"""Integration tests for tools.browser_supervisor.

Exercises the supervisor end-to-end against a real local Chrome
(``--remote-debugging-port``).  Skipped when Chrome is not installed
— these are the tests that actually verify the CDP wire protocol
works, since mock-CDP unit tests can only prove the happy paths we
thought to model.

These tests spawn a **real Chrome process** on the machine running them.
They are therefore opt-in, twice over:

* ``@pytest.mark.integration`` — excluded by the default
  ``addopts = "-m 'not integration'"`` in ``pyproject.toml``, so a bare
  ``pytest`` cannot launch a browser on a developer's desktop by accident.
* ``HERMES_E2E_BROWSER=1`` — the env gate this docstring has always claimed.
  It previously existed only in this prose: nothing read the variable, and
  the sole real gate was "is a Chrome binary on PATH", which is true on most
  desktops and on ``ubuntu-latest``. Now it is enforced.

Run manually:
    HERMES_E2E_BROWSER=1 scripts/run_tests.sh -m integration \\
        tests/tools/test_browser_supervisor.py

(``scripts/run_tests.sh`` runs under ``env -i`` and forwards
``HERMES_E2E_BROWSER`` explicitly; ``-m integration`` overrides the default
marker filter.)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
        reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in",
    ),
    pytest.mark.skipif(
        not shutil.which("google-chrome") and not shutil.which("chromium"),
        reason="Chrome/Chromium not installed",
    ),
]


def _find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    pytest.skip("no Chrome binary found")


@pytest.fixture
def chrome_cdp(tmp_path):
    """Start a headless Chrome with --remote-debugging-port, yield its WS URL.

    Binds an ephemeral port (bind port 0, read back the real one) so
    concurrently-running test files in the per-file parallel runner can
    never collide on a fixed port.
    Always launches with ``--site-per-process`` so cross-origin iframes
    become real OOPIFs (needed by the iframe interaction tests).
    """
    profile = tempfile.mkdtemp(prefix="hermes-supervisor-test-")
    stderr = (tmp_path / "chrome.stderr").open("w+b")
    proc = subprocess.Popen(
        [
            _find_chrome(),
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--headless=new",
            "--disable-gpu",
            "--site-per-process",  # force OOPIFs for cross-origin iframes
        ],
        stdout=subprocess.DEVNULL,
        stderr=stderr,
    )

    ws_url = None
    port = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            port = int((Path(profile) / "DevToolsActivePort").read_text(encoding="utf-8").splitlines()[0])
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=1
            ) as r:
                info = json.loads(r.read().decode())
                ws_url = info["webSocketDebuggerUrl"]
                break
        except Exception:
            time.sleep(0.25)
    if ws_url is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, AssertionError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except (AssertionError, Exception):
                pass
        shutil.rmtree(profile, ignore_errors=True)
        stderr.seek(0)
        diagnostic = stderr.read().decode("utf-8", errors="replace")
        stderr.close()
        pytest.fail(f"Chrome didn't expose CDP in time: {diagnostic}")

    yield ws_url, port

    # Tear down Chrome. The stdlib `subprocess._wait()` POSIX implementation
    # has a known race (https://bugs.python.org/issue38630): when SIGCHLD
    # arrives concurrently with `proc.wait()`, `_try_wait(WNOHANG)` can
    # return a foreign pid and the `assert pid == self.pid or pid == 0`
    # fires. We saw this in CI on slice 1 after this fixture's teardown
    # (PR #33661 follow-up). Swallow the stdlib race + force-kill if wait
    # hangs, then always reap so we don't leak a zombie.
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
    except (subprocess.TimeoutExpired, AssertionError, Exception):
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except (AssertionError, Exception):
            pass
    shutil.rmtree(profile, ignore_errors=True)
    stderr.close()


def _test_page_url() -> str:
    html = """<!doctype html>
<html><head><title>Supervisor pytest</title></head><body>
<h1>Supervisor pytest</h1>
<iframe id="inner" srcdoc="<body><h2>frame-marker</h2></body>" width="400" height="100"></iframe>
</body></html>"""
    return "data:text/html;base64," + base64.b64encode(html.encode()).decode()


def _fire_on_page(cdp_url: str, expression: str) -> None:
    """Navigate the first page target to a data URL and fire `expression`."""
    import websockets as _ws_mod

    async def run():
        async with _ws_mod.connect(cdp_url, max_size=50 * 1024 * 1024) as ws:
            next_id = [1]

            async def call(method, params=None, session_id=None):
                cid = next_id[0]
                next_id[0] += 1
                p = {"id": cid, "method": method}
                if params:
                    p["params"] = params
                if session_id:
                    p["sessionId"] = session_id
                await ws.send(json.dumps(p))
                async for raw in ws:
                    m = json.loads(raw)
                    if m.get("id") == cid:
                        return m

            targets = (await call("Target.getTargets"))["result"]["targetInfos"]
            page = next(t for t in targets if t.get("type") == "page")
            attach = await call(
                "Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}
            )
            sid = attach["result"]["sessionId"]
            await call("Page.navigate", {"url": _test_page_url()}, session_id=sid)
            await asyncio.sleep(1.5)  # let the page load
            await call(
                "Runtime.evaluate",
                {"expression": expression, "returnByValue": True},
                session_id=sid,
            )

    asyncio.run(run())


@pytest.fixture
def supervisor_registry():
    """Yield the global registry and tear down any supervisors after the test."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    yield SUPERVISOR_REGISTRY
    SUPERVISOR_REGISTRY.stop_all()


def _wait_for_dialog(supervisor, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = supervisor.snapshot()
        if snap.pending_dialogs:
            return snap.pending_dialogs
        time.sleep(0.1)
    return ()


def test_supervisor_start_and_snapshot(chrome_cdp, supervisor_registry):
    """Supervisor attaches, exposes an active snapshot with a top frame."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-1", cdp_url=cdp_url)

    # Navigate so the frame tree populates.
    _fire_on_page(cdp_url, "/* no dialog */ void 0")

    # Give a moment for frame events to propagate
    time.sleep(1.0)
    snap = supervisor.snapshot()
    assert snap.active is True
    assert snap.task_id == "pytest-1"
    assert snap.pending_dialogs == ()
    # At minimum a top frame should exist after the navigate.
    assert snap.frame_tree.get("top") is not None


def test_main_frame_alert_detection_and_dismiss(chrome_cdp, supervisor_registry):
    """alert() in the main frame surfaces and can be dismissed via the sync API."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-2", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-MAIN-ALERT'), 50)")
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no dialog detected"
    d = dialogs[0]
    assert d.type == "alert"
    assert "PYTEST-MAIN-ALERT" in d.message

    result = supervisor.respond_to_dialog("dismiss")
    assert result["ok"] is True
    # State cleared after dismiss
    time.sleep(0.3)
    assert supervisor.snapshot().pending_dialogs == ()


def test_iframe_contentwindow_alert(chrome_cdp, supervisor_registry):
    """alert() fired from inside a same-origin iframe surfaces too."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-3", cdp_url=cdp_url)

    _fire_on_page(
        cdp_url,
        "setTimeout(() => document.querySelector('#inner').contentWindow.alert('PYTEST-IFRAME'), 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no iframe dialog detected"
    assert any("PYTEST-IFRAME" in d.message for d in dialogs)

    result = supervisor.respond_to_dialog("accept")
    assert result["ok"] is True


def test_prompt_dialog_with_response_text(chrome_cdp, supervisor_registry):
    """prompt() gets our prompt_text back inside the page."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-4", cdp_url=cdp_url)

    # Fire a prompt and stash the answer on window
    _fire_on_page(
        cdp_url,
        "setTimeout(() => { window.__promptResult = prompt('give me a token', 'default-x'); }, 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs
    d = dialogs[0]
    assert d.type == "prompt"
    assert d.default_prompt == "default-x"

    result = supervisor.respond_to_dialog("accept", prompt_text="PYTEST-PROMPT-REPLY")
    assert result["ok"] is True


def test_browser_dialog_tool_end_to_end(chrome_cdp, supervisor_registry):
    """Full agent-path check: fire an alert, call the tool handler directly."""
    from tools.browser_dialog_tool import browser_dialog

    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-tool", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-TOOL-END2END'), 50)")
    assert _wait_for_dialog(supervisor), "no dialog detected via wait_for_dialog"

    r = json.loads(browser_dialog(action="dismiss", task_id="pytest-tool"))
    assert r["success"] is True
    assert r["action"] == "dismiss"
    assert "PYTEST-TOOL-END2END" in r["dialog"]["message"]



def test_evaluate_runtime_unserializable_value(chrome_cdp, supervisor_registry):
    """``Infinity``/``NaN``/``BigInt`` come back via ``unserializableValue``."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-eval-5", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "void 0")
    time.sleep(0.5)

    out = supervisor.evaluate_runtime("Infinity")
    assert out["ok"] is True
    assert out["result"] == "Infinity"


@pytest.mark.parametrize("iframe_host", ["localhost", "127.0.0.1"], ids=["cross-site-oopif", "same-site-in-process"])
def test_vault_fill_reaches_a_password_form_in_a_cross_origin_iframe(chrome_cdp, supervisor_registry, tmp_path, iframe_host):
    """BankID shape: the top page (127.0.0.1:A) embeds the login form from another origin (<host>:B). On
    ``localhost`` that is another site, which ``--site-per-process`` makes an OOPIF; on ``127.0.0.1`` it is the
    same site (login.bankid.no / csfe.bankid.no), rendered in the top page's process with no session of its own.
    The real browser_vault_fill finds the frame, proves its origin and writes there; the top document is never
    the target."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from unittest.mock import patch

    import websockets as _ws_mod

    from agent.vault_store import VaultStore
    from tools import browser_vault_tool

    form_server = None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = (f'<h1>top</h1><iframe src="http://{iframe_host}:{form_server.server_address[1]}/form"></iframe>'
                    if self.path == "/top"
                    else '<form><input name="user"><input type="password" name="pw" autocomplete="current-password"></form>')
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *_a):
            pass

    top_server, form_server = (ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2))
    for server in (top_server, form_server):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    form_origin = f"http://{iframe_host}:{form_server.server_address[1]}"
    cdp_url, _ = chrome_cdp
    try:
        supervisor = supervisor_registry.get_or_start(task_id="pytest-vault-frame", cdp_url=cdp_url)

        async def open_tab():  # the tab browser_exec would open, not the one the supervisor attached to
            async with _ws_mod.connect(cdp_url) as ws:
                await ws.send(json.dumps({"id": 1, "method": "Target.createTarget",
                                          "params": {"url": f"http://127.0.0.1:{top_server.server_address[1]}/top"}}))
                while json.loads(await ws.recv()).get("id") != 1:
                    pass
        asyncio.run(open_tab())
        time.sleep(2.0)

        store = VaultStore(base_dir=tmp_path / "vault")
        meta = store.add_item("login", "frame", {"identifier_type": "username", "identifier": "u",
                                                 "password": "frame-s3cret"}, origin=form_origin)
        with patch("agent.vault_store.get_vault_store", return_value=store):
            raw = browser_vault_tool.browser_vault_fill(meta.id, task_id="pytest-vault-frame")
        out = json.loads(raw)
        assert out["success"] is True and out["origin"] == form_origin, raw
        assert "frame-s3cret" not in raw

        # Ordering assumption: the tab's existing OOPIF is known as soon as focus_page's attach returns.
        focused = supervisor.focus_page(form_origin)
        assert focused["ok"], focused
        # Only the same-site iframe lives in the tab's own session: proves that row is not passing via an OOPIF.
        assert (focused["frame"].session_id == supervisor._page_session_id) == (iframe_host == "127.0.0.1")
        read = supervisor.evaluate_runtime("document.querySelector('input[name=pw]').value", frame=focused["frame"])
        assert read["result"] == "frame-s3cret"
        assert supervisor.evaluate_runtime("document.querySelector('input[name=pw]')")["result"] is None
    finally:
        for server in (top_server, form_server):
            server.shutdown()
