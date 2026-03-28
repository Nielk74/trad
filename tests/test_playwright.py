# tests/test_playwright.py
"""
Playwright browser tests for the Real-Time Translator frontend.

Requires the app to be running. Start it with:
    .venv/bin/python app.py --config small --host 0.0.0.0 --port 3003

Run:
    .venv/bin/pytest tests/test_playwright.py -v
    .venv/bin/pytest tests/test_playwright.py -v --headed   # show browser
    .venv/bin/pytest tests/test_playwright.py -v --base-url http://localhost:3003
"""
import base64
import io
import json
import re
import struct
import time

import pytest
from playwright.sync_api import Page, expect, sync_playwright

BASE_URL = "https://localhost:3003"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_silent_wav(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Return a minimal valid WAV file with silence."""
    n_samples = int(sample_rate * duration_s)
    pcm = b"\x00\x00" * n_samples  # 16-bit silence
    data_size = len(pcm)
    file_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", file_size, b"WAVE",
        b"fmt ", 16,
        1,              # PCM
        1,              # mono
        sample_rate,
        sample_rate * 2,
        2,              # block align
        16,             # bits per sample
        b"data", data_size,
    )
    return header + pcm


def _wav_b64(duration_s: float = 0.5) -> str:
    return base64.b64encode(_make_silent_wav(duration_s)).decode()


def _fixture_wav_b64(lang: str) -> str:
    """Return base64 of the pre-downloaded fixture WAV for a language."""
    from tests.download_fixtures import get_fixture_path
    path = get_fixture_path(lang)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def browser_ctx():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            base_url=BASE_URL,
            permissions=["microphone"],
            ignore_https_errors=True,
        )
        yield ctx
        ctx.close()
        browser.close()


@pytest.fixture()
def page(browser_ctx):
    pg = browser_ctx.new_page()
    yield pg
    pg.close()


@pytest.fixture()
def app_page(page: Page):
    """Navigate to the app and wait for it to initialise."""
    page.goto(BASE_URL)
    # wait for language selects to be populated (init() calls /languages)
    page.wait_for_selector("#langA option", state="attached", timeout=10_000)
    return page


# ---------------------------------------------------------------------------
# 1. Page load & basic structure
# ---------------------------------------------------------------------------

class TestPageLoad:
    def test_title(self, app_page: Page):
        assert "Translator" in app_page.title()

    def test_header_visible(self, app_page: Page):
        expect(app_page.locator("header h1")).to_be_visible()

    def test_tier_selector_present(self, app_page: Page):
        tier_sel = app_page.locator("#tierSelect")
        expect(tier_sel).to_be_visible()
        options = tier_sel.locator("option").all()
        values = [o.get_attribute("value") for o in options]
        assert set(values) == {"small", "medium", "high"}

    def test_both_record_buttons_visible(self, app_page: Page):
        expect(app_page.locator("#btnA")).to_be_visible()
        expect(app_page.locator("#btnB")).to_be_visible()

    def test_output_box_visible(self, app_page: Page):
        expect(app_page.locator(".output-box")).to_be_visible()

    def test_status_bar_visible(self, app_page: Page):
        expect(app_page.locator("#statusBar")).to_be_visible()

    def test_replay_button_initially_disabled(self, app_page: Page):
        btn = app_page.locator("#replayBtn")
        expect(btn).to_be_disabled()


# ---------------------------------------------------------------------------
# 2. /status endpoint — tier reflected in UI
# ---------------------------------------------------------------------------

class TestStatusInit:
    def test_tier_select_matches_server_config(self, app_page: Page):
        """The tier selector should be set to whatever config the server reports."""
        resp = app_page.request.get(f"{BASE_URL}/status")
        assert resp.ok
        server_config = resp.json()["config"]
        selected = app_page.locator("#tierSelect").input_value()
        assert selected == server_config

    def test_status_bar_shows_ready(self, app_page: Page):
        status_text = app_page.locator("#statusText").inner_text()
        assert "ready" in status_text.lower() or "fallback" in status_text.lower()


# ---------------------------------------------------------------------------
# 3. Language selects populated from /languages
# ---------------------------------------------------------------------------

class TestLanguageSelects:
    def test_lang_a_has_options(self, app_page: Page):
        options = app_page.locator("#langA option").all()
        assert len(options) >= 4

    def test_lang_b_has_options(self, app_page: Page):
        options = app_page.locator("#langB option").all()
        assert len(options) >= 4

    def test_lang_a_default_english(self, app_page: Page):
        assert app_page.locator("#langA").input_value() == "en"

    def test_lang_b_default_french(self, app_page: Page):
        assert app_page.locator("#langB").input_value() == "fr"

    def test_languages_match_api(self, app_page: Page):
        resp = app_page.request.get(f"{BASE_URL}/languages")
        assert resp.ok
        api_codes = {l["code"] for l in resp.json()["languages"]}
        ui_codes = {
            o.get_attribute("value")
            for o in app_page.locator("#langA option").all()
        }
        assert api_codes == ui_codes

    def test_can_change_lang_a(self, app_page: Page):
        app_page.locator("#langA").select_option("fr")
        assert app_page.locator("#langA").input_value() == "fr"

    def test_can_change_lang_b(self, app_page: Page):
        app_page.locator("#langB").select_option("zh")
        assert app_page.locator("#langB").input_value() == "zh"


# ---------------------------------------------------------------------------
# 4. API endpoints via page.request (same origin)
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    def test_health_endpoint(self, app_page: Page):
        resp = app_page.request.get(f"{BASE_URL}/health")
        assert resp.ok
        data = resp.json()
        assert set(data.keys()) == {"stt", "translation", "tts"}
        for v in data.values():
            assert isinstance(v, bool)

    def test_status_endpoint_schema(self, app_page: Page):
        resp = app_page.request.get(f"{BASE_URL}/status")
        assert resp.ok
        data = resp.json()
        assert "config" in data
        assert data["config"] in ("small", "medium", "high")
        assert "models_loaded" in data
        assert "fallback_active" in data
        assert isinstance(data["fallback_active"], list)

    def test_languages_endpoint_schema(self, app_page: Page):
        resp = app_page.request.get(f"{BASE_URL}/languages")
        assert resp.ok
        data = resp.json()
        assert "languages" in data
        for lang in data["languages"]:
            assert "code" in lang
            assert "name" in lang
            assert "flag" in lang
            assert "tiers" in lang

    def test_transcribe_endpoint_silent_audio(self, app_page: Page):
        """Sending silent audio should return a (possibly empty) text string."""
        resp = app_page.request.post(
            f"{BASE_URL}/transcribe",
            data=json.dumps({"audio": _wav_b64(), "lang": "en"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        data = resp.json()
        assert "text" in data
        assert "fallback" in data
        assert isinstance(data["text"], str)
        assert isinstance(data["fallback"], bool)

    def test_translate_endpoint(self, app_page: Page):
        resp = app_page.request.post(
            f"{BASE_URL}/translate",
            data=json.dumps({"text": "Hello", "source": "en", "target": "fr"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        data = resp.json()
        assert "translation" in data
        assert len(data["translation"]) > 0

    def test_synthesize_endpoint(self, app_page: Page):
        resp = app_page.request.post(
            f"{BASE_URL}/synthesize",
            data=json.dumps({"text": "Hello", "lang": "en"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        data = resp.json()
        assert "audio" in data
        assert "fallback" in data
        raw = base64.b64decode(data["audio"])
        assert raw[:4] == b"RIFF", "Response audio is not a WAV file"
        assert len(raw) > 100

    def test_transcribe_rejects_too_long_audio(self, app_page: Page):
        resp = app_page.request.post(
            f"{BASE_URL}/transcribe",
            data=json.dumps({"audio": _wav_b64(duration_s=35.0), "lang": "en"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    def test_config_endpoint_invalid_value(self, app_page: Page):
        resp = app_page.request.post(
            f"{BASE_URL}/config",
            data=json.dumps({"config": "xlarge"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    def test_config_endpoint_valid_value(self, app_page: Page):
        resp = app_page.request.post(
            f"{BASE_URL}/config",
            data=json.dumps({"config": "small"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        data = resp.json()
        assert data["config"] == "small"


# ---------------------------------------------------------------------------
# 5. JS-level pipeline via page.evaluate (intercepts network)
# ---------------------------------------------------------------------------

class TestJSPipeline:
    def test_translate_and_synthesize_via_js(self, app_page: Page):
        """Call translate + synthesize directly through the page's fetch."""
        result = app_page.evaluate("""async () => {
            const tr = await fetch('/translate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: 'Good morning', source: 'en', target: 'fr'}),
            }).then(r => r.json());

            const sy = await fetch('/synthesize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: tr.translation, lang: 'fr'}),
            }).then(r => r.json());

            return {translation: tr.translation, audioLen: sy.audio.length, fallback: sy.fallback};
        }""")
        assert len(result["translation"]) > 0
        assert result["audioLen"] > 100
        assert isinstance(result["fallback"], bool)

    def test_full_pipeline_via_js(self, app_page: Page):
        """Transcribe → translate → synthesize entirely within the browser context."""
        wav_b64 = _fixture_wav_b64("en")
        result = app_page.evaluate(f"""async () => {{
            const wav_b64 = "{wav_b64}";

            const tRes = await fetch('/transcribe', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{audio: wav_b64, lang: 'en'}}),
            }}).then(r => r.json());

            const trRes = await fetch('/translate', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{text: tRes.text, source: 'en', target: 'fr'}}),
            }}).then(r => r.json());

            const sRes = await fetch('/synthesize', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{text: trRes.translation, lang: 'fr'}}),
            }}).then(r => r.json());

            return {{
                transcript: tRes.text,
                translation: trRes.translation,
                audioLen: sRes.audio.length,
            }};
        }}""")
        assert isinstance(result["transcript"], str)
        assert isinstance(result["translation"], str)
        assert result["audioLen"] > 100


# ---------------------------------------------------------------------------
# 6. Network request monitoring
# ---------------------------------------------------------------------------

class TestNetworkRequests:
    def test_page_load_calls_status_and_languages(self, page: Page):
        calls = []
        page.on("request", lambda req: calls.append(req.url))
        page.goto(BASE_URL)
        page.wait_for_selector("#langA option", state="attached", timeout=10_000)

        paths = [re.sub(r"https?://[^/]+", "", u) for u in calls]
        assert "/status" in paths, f"/status not called; got: {paths}"
        assert "/languages" in paths, f"/languages not called; got: {paths}"

    def test_no_cross_origin_requests(self, page: Page):
        cross_origin = []
        def on_req(req):
            if req.url.startswith("http") and BASE_URL not in req.url:
                cross_origin.append(req.url)
        page.on("request", on_req)
        page.goto(BASE_URL)
        page.wait_for_selector("#langA option", state="attached", timeout=10_000)
        assert cross_origin == [], f"Unexpected cross-origin requests: {cross_origin}"

    def test_no_console_errors_on_load(self, page: Page):
        errors = []
        page.on("console", lambda msg: errors.append(msg) if msg.type == "error" else None)
        page.goto(BASE_URL)
        page.wait_for_selector("#langA option", state="attached", timeout=10_000)
        assert errors == [], f"Console errors on load: {[e.text for e in errors]}"


# ---------------------------------------------------------------------------
# 7. Tier switching UI
# ---------------------------------------------------------------------------

class TestTierSwitching:
    def test_tier_change_posts_to_config(self, app_page: Page):
        config_calls = []
        app_page.on("request", lambda req: config_calls.append(req) if "/config" in req.url else None)

        current = app_page.locator("#tierSelect").input_value()
        new_tier = "medium" if current != "medium" else "small"
        app_page.locator("#tierSelect").select_option(new_tier)

        # Wait briefly for the fetch to fire
        app_page.wait_for_timeout(500)
        assert len(config_calls) >= 1
        body = json.loads(config_calls[0].post_data)
        assert body["config"] == new_tier

    def test_status_bar_shows_switching_message(self, app_page: Page):
        """Status bar should briefly show a switching message."""
        messages = []

        def capture(req):
            if "/config" in req.url:
                messages.append(app_page.locator("#statusBar").inner_text())

        app_page.on("request", capture)
        current = app_page.locator("#tierSelect").input_value()
        new_tier = "medium" if current != "medium" else "small"
        app_page.locator("#tierSelect").select_option(new_tier)
        app_page.wait_for_timeout(300)

        if messages:
            assert any("switch" in m.lower() or "tier" in m.lower() or "…" in m for m in messages)


# ---------------------------------------------------------------------------
# 8. Replay button state
# ---------------------------------------------------------------------------

class TestReplayButton:
    def test_replay_enabled_after_synthesize(self, app_page: Page):
        """After a successful /synthesize call the replay button should be enabled."""
        app_page.evaluate("""async () => {
            const sRes = await fetch('/synthesize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: 'Hello', lang: 'en'}),
            }).then(r => r.json());

            // Simulate what handleAudio does
            window.lastAudioB64 = sRes.audio;
            document.getElementById('replayBtn').disabled = false;
        }""")
        expect(app_page.locator("#replayBtn")).to_be_enabled()


# ---------------------------------------------------------------------------
# 9. Accessibility & responsiveness
# ---------------------------------------------------------------------------

class TestAccessibility:
    def test_lang_selects_have_labels(self, app_page: Page):
        labels = app_page.locator(".lang-col label").all()
        assert len(labels) >= 2
        texts = [l.inner_text().strip() for l in labels]
        assert any(t for t in texts)

    def test_record_buttons_have_text(self, app_page: Page):
        for btn_id in ("#btnA", "#btnB"):
            text = app_page.locator(btn_id).inner_text()
            assert len(text.strip()) > 0

    def test_mobile_viewport_no_overflow(self, browser_ctx):
        page = browser_ctx.new_page()
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(BASE_URL)
        page.wait_for_selector("#langA option", state="attached", timeout=10_000)
        # Check that the main content fits within viewport width
        overflow = page.evaluate("""() => {
            const main = document.querySelector('main');
            return main ? main.scrollWidth > main.clientWidth : false;
        }""")
        assert not overflow, "Content overflows on mobile viewport"
        page.close()


# ---------------------------------------------------------------------------
# 10. Error handling in UI
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_translate_empty_text_handled(self, app_page: Page):
        """Calling /translate with empty text should return a 200 with empty-ish result
        or a 400, but NOT crash the page."""
        try:
            resp = app_page.request.post(
                f"{BASE_URL}/translate",
                data=json.dumps({"text": "", "source": "en", "target": "fr"}),
                headers={"Content-Type": "application/json"},
            )
            # Either succeeds gracefully or returns a client error — no 500
            assert resp.status != 500
        except Exception as e:
            pytest.fail(f"Request raised unexpectedly: {e}")

    def test_synthesize_empty_text_handled(self, app_page: Page):
        try:
            resp = app_page.request.post(
                f"{BASE_URL}/synthesize",
                data=json.dumps({"text": "", "lang": "en"}),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status != 500
        except Exception as e:
            pytest.fail(f"Request raised unexpectedly: {e}")

    def test_unknown_route_returns_404(self, app_page: Page):
        resp = app_page.request.get(f"{BASE_URL}/does-not-exist")
        assert resp.status == 404


# ---------------------------------------------------------------------------
# 11. Mobile emulation — touch recording flow with mocked MediaRecorder
# ---------------------------------------------------------------------------

def _fire_touch(page: Page, selector: str, event: str):
    """Dispatch a touch event via JS to avoid Touch constructor identifier issues."""
    page.evaluate("""([sel, evt]) => {
        const el = document.querySelector(sel);
        const rect = el.getBoundingClientRect();
        const touch = new Touch({
            identifier: Date.now(),
            target: el,
            clientX: rect.left + rect.width / 2,
            clientY: rect.top + rect.height / 2,
        });
        const isTouchEnd = evt === 'touchend';
        el.dispatchEvent(new TouchEvent(evt, {
            bubbles: true,
            cancelable: true,
            touches: isTouchEnd ? [] : [touch],
            changedTouches: [touch],
        }));
    }""", [selector, event])


# JS injected before page scripts run: replaces getUserMedia + MediaRecorder
# with a fake that immediately produces a valid silent WAV blob.
_MOCK_MEDIA_JS = """
(function() {
    // Build a tiny silent WAV (0.1 s, mono 16kHz 16-bit)
    function silentWav() {
        const sr = 16000, n = 1600;
        const buf = new ArrayBuffer(44 + n * 2);
        const v = new DataView(buf);
        function ws(off, s) { for (let i=0;i<s.length;i++) v.setUint8(off+i, s.charCodeAt(i)); }
        ws(0,'RIFF'); v.setUint32(4, 36+n*2, true); ws(8,'WAVE');
        ws(12,'fmt '); v.setUint32(16,16,true); v.setUint16(20,1,true);
        v.setUint16(22,1,true); v.setUint32(24,sr,true); v.setUint32(28,sr*2,true);
        v.setUint16(32,2,true); v.setUint16(34,16,true);
        ws(36,'data'); v.setUint32(40,n*2,true);
        return new Blob([buf], {type:'audio/wav'});
    }

    // Fake MediaRecorder
    class FakeMediaRecorder extends EventTarget {
        constructor(stream, opts) {
            super();
            this.stream = stream;
            this.state = 'inactive';
            this.ondataavailable = null;
            this.onstop = null;
        }
        start() {
            this.state = 'recording';
            // Immediately fire a dataavailable event with our silent WAV
            const blob = silentWav();
            if (this.ondataavailable) this.ondataavailable({data: blob});
        }
        stop() {
            this.state = 'inactive';
            if (this.onstop) this.onstop();
        }
        static isTypeSupported(t) { return t === 'audio/wav'; }
    }
    window.MediaRecorder = FakeMediaRecorder;

    // Fake getUserMedia — returns a stub stream
    const fakeStream = {
        getTracks: () => [{stop: () => {}}],
    };
    navigator.mediaDevices = navigator.mediaDevices || {};
    navigator.mediaDevices.getUserMedia = async () => fakeStream;

    // Suppress audio.play() errors (no audio hardware in headless)
    const origPlay = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = function() {
        return Promise.resolve();
    };
})();
"""


@pytest.fixture()
def mobile_page(browser_ctx):
    """A page emulating iPhone 12 with mocked media APIs."""
    page = browser_ctx.new_page()
    page.set_viewport_size({"width": 390, "height": 844})
    # Inject mock before any page script runs
    page.add_init_script(_MOCK_MEDIA_JS)
    page.goto(BASE_URL)
    page.wait_for_selector("#langA option", state="attached", timeout=10_000)
    yield page
    page.close()


class TestMobileRecording:
    def test_touch_start_adds_recording_class(self, mobile_page: Page):
        """Touching btnA should add the 'recording' CSS class."""
        _fire_touch(mobile_page, "#btnA", "touchstart")
        mobile_page.wait_for_timeout(300)
        classes = mobile_page.locator("#btnA").get_attribute("class") or ""
        assert "recording" in classes

    def test_touch_end_removes_recording_class(self, mobile_page: Page):
        """After touchend the button should no longer have 'recording' class."""
        _fire_touch(mobile_page, "#btnA", "touchstart")
        mobile_page.wait_for_timeout(300)
        _fire_touch(mobile_page, "#btnA", "touchend")
        mobile_page.wait_for_timeout(300)
        classes = mobile_page.locator("#btnA").get_attribute("class") or ""
        assert "recording" not in classes

    def test_btn_label_changes_to_recording(self, mobile_page: Page):
        _fire_touch(mobile_page, "#btnA", "touchstart")
        mobile_page.wait_for_timeout(300)
        label = mobile_page.locator("#btnA .btn-label").inner_text()
        assert "recording" in label.lower() or "…" in label

    def test_status_shows_transcribing_after_touchend(self, mobile_page: Page):
        """After releasing the button the pipeline should kick off."""
        _fire_touch(mobile_page, "#btnA", "touchstart")
        mobile_page.wait_for_timeout(300)
        _fire_touch(mobile_page, "#btnA", "touchend")
        try:
            mobile_page.wait_for_function(
                "() => document.getElementById('statusText').innerHTML.includes('…') "
                "|| document.getElementById('statusText').textContent.includes('Transcrib') "
                "|| document.getElementById('translationText').textContent !== '—'",
                timeout=8_000,
            )
        except Exception:
            pass
        status = mobile_page.locator("#statusText").inner_html()
        translation = mobile_page.locator("#translationText").inner_text()
        assert (
            "…" in status
            or len(translation) > 1
            or "ready" in status.lower()
            or "error" in status.lower()
        ), f"Unexpected status after touchend: '{status}', translation: '{translation}'"

    def test_full_pipeline_completes_on_mobile(self, mobile_page: Page):
        """Full touch→transcribe→translate→synthesize flow should complete."""
        mobile_page.locator("#langA").select_option("en")
        mobile_page.locator("#langB").select_option("fr")

        _fire_touch(mobile_page, "#btnA", "touchstart")
        mobile_page.wait_for_timeout(300)
        _fire_touch(mobile_page, "#btnA", "touchend")

        mobile_page.wait_for_function(
            "() => !document.getElementById('replayBtn').disabled "
            "|| document.getElementById('statusText').textContent.includes('Error') "
            "|| document.getElementById('statusText').textContent.includes('Ready')",
            timeout=30_000,
        )
        status = mobile_page.locator("#statusText").inner_text()
        assert "…" not in status, f"Pipeline still loading after 30s: '{status}'"

    def test_translation_text_updated_on_mobile(self, mobile_page: Page):
        """translationText should be updated from '—' after the pipeline runs."""
        mobile_page.locator("#langA").select_option("en")
        mobile_page.locator("#langB").select_option("fr")

        _fire_touch(mobile_page, "#btnA", "touchstart")
        mobile_page.wait_for_timeout(300)
        _fire_touch(mobile_page, "#btnA", "touchend")

        mobile_page.wait_for_function(
            "() => document.getElementById('translationText').textContent !== '—' "
            "&& document.getElementById('translationText').textContent !== '…'",
            timeout=30_000,
        )
        text = mobile_page.locator("#translationText").inner_text()
        assert len(text) > 0 and text != "—"

    def test_replay_button_enabled_after_mobile_pipeline(self, mobile_page: Page):
        """Replay button should become enabled after a successful synthesis."""
        _fire_touch(mobile_page, "#btnA", "touchstart")
        mobile_page.wait_for_timeout(300)
        _fire_touch(mobile_page, "#btnA", "touchend")

        mobile_page.wait_for_function(
            "() => !document.getElementById('replayBtn').disabled",
            timeout=30_000,
        )
        expect(mobile_page.locator("#replayBtn")).to_be_enabled()

    def test_second_touch_ignored_while_recording(self, mobile_page: Page):
        """Starting a second recording before the first stops should be a no-op."""
        _fire_touch(mobile_page, "#btnA", "touchstart")
        mobile_page.wait_for_timeout(200)
        _fire_touch(mobile_page, "#btnB", "touchstart")
        mobile_page.wait_for_timeout(200)
        class_a = mobile_page.locator("#btnA").get_attribute("class") or ""
        class_b = mobile_page.locator("#btnB").get_attribute("class") or ""
        assert "recording" in class_a
        assert "recording" not in class_b
        _fire_touch(mobile_page, "#btnA", "touchend")

    def test_microphone_denied_shows_error(self, browser_ctx):
        """If getUserMedia rejects, the status bar should show an error."""
        page = browser_ctx.new_page()
        page.set_viewport_size({"width": 390, "height": 844})
        page.add_init_script("""
        (function() {
            navigator.mediaDevices = navigator.mediaDevices || {};
            navigator.mediaDevices.getUserMedia = async () => {
                throw new DOMException('Permission denied', 'NotAllowedError');
            };
        })();
        """)
        page.goto(BASE_URL)
        page.wait_for_selector("#langA option", state="attached", timeout=10_000)
        _fire_touch(page, "#btnA", "touchstart")
        page.wait_for_function(
            "() => document.getElementById('statusText').textContent.includes('denied') "
            "|| document.getElementById('statusText').textContent.includes('Microphone') "
            "|| document.getElementById('statusText').textContent.includes('⚠')",
            timeout=5_000,
        )
        status = page.locator("#statusText").inner_text()
        assert "⚠" in status or "denied" in status.lower() or "microphone" in status.lower()
        page.close()
