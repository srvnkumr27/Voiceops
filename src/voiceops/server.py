"""Web server for the VoiceOps UI: FastAPI + streaming (NDJSON) so the page can show progress as a turn happens.

  POST /api/chat   {sid, message, speak}          -> stream of events
  POST /api/voice  multipart: sid, audio, speak   -> stream of events
  GET  /api/audio/<id>.wav                        -> a synthesised reply
  POST /api/reset  {sid}                          GET /api/config
Events (one JSON object per line): stage, heard, reply, audio, done, error.
The browser records raw audio only; speech recognition and synthesis happen exclusively through Gnani on the server.
"""
import contextlib
import json
import queue
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .pipeline import run_turn_events

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
TICKET_SUFFIX = re.compile(r"\s*\(Ticket:[^)]*\)")
AUDIO_NAME = re.compile(r"^[a-f0-9]{32}\.wav$")
MAX_UPLOAD = 25 * 1024 * 1024
EXAMPLES = ["My order still hasn't arrived. Can I cancel it?", "How long does a UPI refund take?",
            "Can I return a laptop after 9 days?", "I'd like to talk to a person"]


def serialize_reply(rep, agent) -> dict:
    s = agent.s
    sources = []
    for r in rep.retrieved or []:
        head, _, body = (r.get("text") or "").partition("\n")
        sources.append({"id": r["id"], "score": r.get("score"), "doc": r.get("doc"), "status": r.get("status"),
                        "doc_type": r.get("doc_type"), "title": head.split(")", 1)[-1].strip(" )"), "snippet": body.strip()[:700]})
    t = rep.ticket if (rep.ticket and rep.ticket.get("created")) else None
    a = rep.analysis or {}
    return {
        "text": TICKET_SUFFIX.sub("", rep.text).strip(), "language": rep.language, "decision": rep.decision,
        "latency_s": rep.latency_s, "used": list(rep.used_sections or []), "sources": sources,
        "tools": list(rep.tool_calls or []), "reasons": list(rep.escalation_reasons or []),
        "ticket": {"id": t["ticket_id"], "duplicate": bool(t.get("duplicate"))} if t else None,
        "analysis": {"intent": a.get("intent"), "standalone": a.get("standalone_query")},
        "session": {"order_id": s.order_id, "verified": s.verified, "awaiting": s.awaiting_verification, "locked": s.locked,
                    "language": s.language, "turn": s.turn},
    }


class Entry:
    def __init__(self, agent):
        self.agent, self.lock, self.seen = agent, threading.Lock(), time.time()


class Sessions:
    def __init__(self, make_agent, cap=100, ttl=2 * 3600):
        self.make, self.cap, self.ttl, self.items, self.mu = make_agent, cap, ttl, OrderedDict(), threading.Lock()

    def get(self, sid: str) -> Entry:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", sid or ""):
            raise HTTPException(400, "bad session id")
        with self.mu:
            now = time.time()
            for k in [k for k, e in self.items.items() if now - e.seen > self.ttl]:
                self.items.pop(k, None)
            e = self.items.get(sid)
            if e is None:
                e = self.items[sid] = Entry(self.make())
                while len(self.items) > self.cap:
                    self.items.popitem(last=False)
            e.seen = now
            self.items.move_to_end(sid)
            return e

    def drop(self, sid):
        with self.mu:
            self.items.pop(sid, None)


def _stream(work):
    """Run work(emit) in a thread and yield its events as NDJSON lines."""
    q: queue.Queue = queue.Queue()

    def runner():
        try:
            work(q.put)
        except Exception as exc:  # reported to the page, never crashes the server
            import logging
            logging.getLogger("voiceops").exception("turn failed")
            q.put({"type": "error", "message": _friendly(exc)})
        finally:
            q.put(None)

    threading.Thread(target=runner, daemon=True).start()
    while True:
        ev = q.get()
        if ev is None:
            return
        yield json.dumps(ev, ensure_ascii=False) + "\n"


def _friendly(exc) -> str:
    m = str(exc)
    if "TTS failed" in m:
        return "The reply was written but the voice could not be generated. Check your Gnani key and connection."
    if "LLM" in type(exc).__name__ or "LLM call failed" in m:
        return "The language model did not respond. Check your LLM key and connection, then try again."
    return "Something went wrong while handling that. Please try again."


def create_app(make_agent, speech=None, speech_error=None, llm=None, prewarm_langs=("en",)) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_app):                                # best-effort warm-ups, in the background
        warm = getattr(llm, "warmup", None)
        if warm:
            threading.Thread(target=warm, daemon=True).start()
        if speech is not None and prewarm_langs and hasattr(speech, "prewarm"):
            threading.Thread(target=speech.prewarm, args=(prewarm_langs,), daemon=True).start()
        yield

    app = FastAPI(title="ZentroMart voice support", docs_url=None, redoc_url=None, lifespan=lifespan)
    sessions = Sessions(make_agent)
    audio_dir = Path(tempfile.mkdtemp(prefix="voiceops-audio-"))

    def sweep_audio():
        cutoff = time.time() - 3600
        for f in audio_dir.glob("*.wav"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"})

    @app.get("/api/config")
    def config():
        from .config import TTS_MODEL, TTS_VOICE
        return {"voice": {"available": speech is not None, "reason": None if speech else (speech_error or "Voice is not configured.")},
                "llm": f"{getattr(llm, 'provider', '')}/{getattr(llm, 'model', '')}".strip("/"),
                "tts": {"model": TTS_MODEL, "voice": TTS_VOICE}, "examples": EXAMPLES}

    @app.get("/api/audio/{name}")
    def audio(name: str):
        if not AUDIO_NAME.match(name) or not (audio_dir / name).is_file():
            raise HTTPException(404)
        return FileResponse(audio_dir / name, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.post("/api/reset")
    def reset(body: dict):
        sessions.drop(str(body.get("sid", "")))
        return {"ok": True}

    def respond_stream(entry, **kw):
        out_name = f"{uuid.uuid4().hex}.wav"

        def work(emit):
            sweep_audio()
            with entry.lock:                                 # one turn at a time per call
                def ev(e):
                    if e["type"] == "reply":
                        e = {**e, "reply": serialize_reply(e["reply"], entry.agent)}
                    elif e["type"] == "audio":
                        e = {**e, "url": f"/api/audio/{out_name}"}
                    emit(e)
                run_turn_events(entry.agent, ev, speech=speech, out_audio=audio_dir / out_name, **kw)

        return StreamingResponse(_stream(work), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/api/chat")
    def chat(body: dict):
        msg = str(body.get("message", "")).strip()
        if not msg:
            raise HTTPException(400, "empty message")
        speak = bool(body.get("speak")) and speech is not None
        return respond_stream(sessions.get(str(body.get("sid", ""))), text=msg[:2000], speak=speak)

    @app.post("/api/voice")
    async def voice(sid: str = Form(...), audio: UploadFile = File(...)):
        if speech is None:
            return JSONResponse({"error": speech_error or "Voice is not configured."}, status_code=503)
        data = await audio.read()
        if not data:
            raise HTTPException(400, "empty audio")
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "audio too large")
        entry = sessions.get(sid)
        suffix = Path(audio.filename or "").suffix.lower()
        suffix = suffix if suffix in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm") else ".wav"
        src = audio_dir / f"in-{uuid.uuid4().hex}{suffix}"
        src.write_bytes(data)

        def cleanup_later(e=None):
            threading.Timer(120, lambda: src.unlink(missing_ok=True)).start()
        cleanup_later()
        return respond_stream(entry, audio_path=src)

    return app


def build_default_app():
    """Wire the real components. Voice failing to start never blocks text chat."""
    from .agent import Agent
    from .kb import Retriever
    from .llm import LLM
    kb, llm = Retriever(), LLM()
    Agent(llm=llm, retriever=kb)                             # fail fast, with a clear message, if the dataset is missing
    speech, err = None, None
    try:
        from .speech import Speech
        speech = Speech()
    except Exception as exc:
        err = f"Voice is off: {exc}. Set GNANI_API_KEY in .env and restart. Typing still works."
        print(f"[voice] {err}")
    from .config import env_flag
    langs = tuple(x for x in __import__("os").getenv("PREWARM_LANGS", "en").split(",") if x.strip()) if env_flag("PREWARM", True) else ()
    return create_app(lambda: Agent(llm=llm, retriever=kb), speech, err, llm, langs)
