"""Speed layer: disk cache, LLM connection reuse + caching + fail-fast, STT grace window, fast paths, server stream."""
import http.server
import json
import os
import threading
import time
from pathlib import Path

import pytest

from voiceops.agent import Agent, fast_opener, only_filler
from voiceops.cache import DiskCache, make_key
from fake_llm import FakeLLM


# ------------------------------------------------------------------ cache
def test_disk_cache_roundtrip_and_prune(tmp_path):
    c = DiskCache(tmp_path, ".txt", max_bytes=300)
    assert c.get_text("a") is None
    c.put_text("a", "hello")
    assert c.get_text("a") == "hello" and c.hits == 1
    for i in range(60):
        c.put_text(f"k{i}", "x" * 40)
    assert sum(f.stat().st_size for f in tmp_path.glob("*.txt")) <= 300 + 40 * 25   # pruned (runs every 25 writes)


def test_disk_cache_copy_file(tmp_path):
    c = DiskCache(tmp_path / "c", ".wav")
    src = tmp_path / "a.wav"; src.write_bytes(b"R" * 100)
    c.put_file("k", src)
    dst = tmp_path / "b.wav"
    assert c.copy_to("k", dst) and dst.read_bytes() == src.read_bytes()
    assert not c.copy_to("missing", tmp_path / "z.wav")


def test_unwritable_cache_is_a_noop():
    c = DiskCache("/proc/forbidden/dir", ".txt")
    assert not c.enabled and c.get_text("a") is None
    c.put_text("a", "x")


# ------------------------------------------------------------------ LLM client against a local server
class _H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    conns = 0; posts = 0; mode = "ok"
    def setup(self):
        type(self).conns += 1; super().setup()
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("content-length", 0)); self.rfile.read(n)
        type(self).posts += 1
        if _H.mode == "401":
            body, code = b'{"error":"bad key"}', 401
        else:
            txt = "not json at all" if _H.mode == "bad" else '{"intent": "x"}'
            body, code = json.dumps({"choices": [{"message": {"content": txt}}]}).encode(), 200
        self.send_response(code); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)


@pytest.fixture
def server(monkeypatch, tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _H.conns = _H.posts = 0; _H.mode = "ok"
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k"); monkeypatch.setenv("LLM_PROVIDER", "openai"); monkeypatch.setenv("LLM_MODEL", "m")
    import voiceops.llm as L
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path); monkeypatch.setattr(L, "_SESSION", None)
    yield srv
    srv.shutdown()


def test_connections_are_reused(server, monkeypatch):
    monkeypatch.setenv("LLM_CACHE", "0")
    from voiceops.llm import LLM
    llm = LLM()
    for i in range(4):
        llm.complete("s", f"u{i}")
    assert _H.posts == 4 and _H.conns == 1            # one TCP/TLS connection for all four calls


def test_json_cache_hit_skips_network(server):
    from voiceops.llm import LLM
    llm = LLM()
    assert llm.json("s", "u") == {"intent": "x"} and _H.posts == 1
    assert llm.json("s", "u") == {"intent": "x"} and _H.posts == 1 and llm.cache_hits == 1
    assert llm.json("s", "different") == {"intent": "x"} and _H.posts == 2


def test_malformed_json_is_never_cached(server):
    from voiceops.llm import LLM, LLMError
    _H.mode = "bad"
    llm = LLM()
    with pytest.raises(LLMError):
        llm.json("s", "u")
    _H.mode = "ok"
    assert llm.json("s", "u") == {"intent": "x"}      # not poisoned by the earlier bad answer


def test_bad_key_fails_fast(server, monkeypatch):
    monkeypatch.setenv("LLM_CACHE", "0")
    from voiceops.llm import LLM, LLMUnavailable
    _H.mode = "401"
    t = time.time()
    with pytest.raises(LLMUnavailable):
        LLM().complete("s", "u")
    assert _H.posts == 1 and time.time() - t < 1.0    # no 3x retry with sleeps


# ------------------------------------------------------------------ STT grace window
class _Stt:
    def __init__(self, delays, texts): self.delays, self.texts = delays, texts
    def transcribe(self, path, language_code, format=None):
        time.sleep(self.delays[language_code]); return {"transcript": self.texts[language_code]}


def _speech(delays, texts):
    from voiceops.speech import Speech
    s = object.__new__(Speech); s.stt = _Stt(delays, texts); s.tts_cache = None
    return s


def test_stt_does_not_wait_for_slow_second_language(tmp_path):
    a = tmp_path / "a.mp3"; a.write_bytes(b"x")
    s = _speech({"en-IN": .05, "hi-IN": 3.0}, {"en-IN": "how long does the refund take", "hi-IN": "x"})
    t = time.time(); out = s.transcribe(a, ["en-IN", "hi-IN"], grace_s=.2)
    assert time.time() - t < 1.0 and out["en-IN"] and out["hi-IN"] == ""


def test_stt_waits_for_everyone_when_first_answer_is_weak(tmp_path):
    a = tmp_path / "a.mp3"; a.write_bytes(b"x")
    s = _speech({"en-IN": .05, "hi-IN": .5}, {"en-IN": "hmm", "hi-IN": "मेरा रिफंड कब आएगा"})
    out = s.transcribe(a, ["en-IN", "hi-IN"], grace_s=.1)
    assert out["hi-IN"].startswith("मेरा")


def test_long_wav_split_uses_portable_tmpdir(tmp_path):
    import wave
    from voiceops.speech import _wav_chunks
    p = tmp_path / "long.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\0\0" * 8000 * 120)
    parts, temps = _wav_chunks(p, tmp_path)
    assert len(parts) == 3 and all(x.exists() for x in temps)


def test_tts_cache_skips_second_synthesis(tmp_path):
    from voiceops.speech import Speech
    s = object.__new__(Speech); s.tts_cache = DiskCache(tmp_path / "t", ".wav"); calls = []
    class T:
        def synthesize(self, text, voice, model, language, output_file): calls.append(text); Path(output_file).write_bytes(b"RIFF" + b"0" * 100)
    s.tts = T()
    s.synthesize("hello there", "en", tmp_path / "1.wav"); s.synthesize("hello there", "en", tmp_path / "2.wav")
    assert len(calls) == 1 and (tmp_path / "2.wav").stat().st_size > 44


# ------------------------------------------------------------------ fast paths
@pytest.mark.parametrize("text,kind", [
    ("Hello, mujhe order ke baare mein help chahiye.", "ask_order_id"), ("Hi, I need some help with an order.", "ask_order_id"),
    ("नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।", "ask_order_id"), ("Hi", "greeting"), ("Namaste", "greeting"), ("can you help me", "greeting")])
def test_fast_opener_positive(text, kind):
    assert fast_opener(text)[0] == kind


@pytest.mark.parametrize("text", ["Hi, my order hasn't arrived yet. Where is it?", "Hello I want to cancel my order", "hi I need help with a refund of 500",
                                  "I need help with order 89782837", "Hi, I need help with an order and I want a human", "How long does a UPI refund take?"])
def test_fast_opener_never_swallows_a_real_request(text):
    assert fast_opener(text) is None


def test_only_filler():
    assert only_filler("Order number hai") and only_filler("ऑर्डर नंबर है ।")
    assert not only_filler("where is my refund") and not only_filler("my order id is, I want a human")


def test_opener_and_order_id_turns_use_no_llm(kb):
    llm = FakeLLM(); ag = Agent(llm=llm, retriever=kb)
    r = ag.respond("Hi, I need some help with an order.")
    assert llm.calls == 0 and "AI" in r.text and "order ID" in r.text and r.decision == "clarify"
    r = ag.respond("My order number is 89782837")
    assert llm.calls == 0 and ag.s.awaiting_verification


def test_stage_hook_reports_progress(kb):
    ag = Agent(llm=FakeLLM(), retriever=kb); seen = []
    ag.on_stage = seen.append
    ag.respond("How long does a UPI refund take?")
    assert seen == ["understanding", "retrieving", "writing"]


# ------------------------------------------------------------------ server stream
def test_server_streams_events_in_order(kb):
    from fastapi.testclient import TestClient
    from voiceops.server import create_app
    app = create_app(lambda: Agent(llm=FakeLLM(), retriever=kb), None, "no voice", None, ())
    c = TestClient(app)
    assert c.get("/api/config").json()["voice"]["available"] is False
    with c.stream("POST", "/api/chat", json={"sid": "session-0001", "message": "How long does a UPI refund take?"}) as r:
        ev = [json.loads(l) for l in r.iter_lines() if l]
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "stage" and "reply" in kinds and kinds[-1] == "done" and kinds.index("reply") < kinds.index("done")
    rep = next(e for e in ev if e["type"] == "reply")["reply"]
    assert rep["sources"] and rep["session"]["turn"] == 1
    assert c.post("/api/chat", json={"sid": "x", "message": "hi"}).status_code == 400
    assert c.post("/api/voice", data={"sid": "session-0001"}, files={"audio": ("a.wav", b"RIFF", "audio/wav")}).status_code == 503
    assert c.get("/api/audio/" + "0" * 32 + ".wav").status_code == 404
    assert "ZentroMart" in c.get("/").text
