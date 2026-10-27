"""One turn: audio -> Gnani STT -> agent (LLM + RAG + tools) -> Gnani TTS.

voice_turn()      simple blocking call (used by run.py and the legacy UI).
run_turn_events() same work, but reports progress through emit(event) so a UI can show the transcript and the reply text
                  BEFORE the audio is synthesised (text first, voice second = the customer sees the answer sooner).
"""
import time
from pathlib import Path

from . import templates as TPL
from .speech import Speech, pick_transcript


def _unclear_reply(agent):
    from .agent import Reply
    from .textnorm import speakable
    lang = agent.s.language
    text = TPL.get("didnt_catch", lang) or TPL.get("didnt_catch", "en")
    return Reply(text, speakable(text, lang), lang, "clarify")


def voice_turn(audio_path, agent, speech: Speech, out_audio, languages=None):
    cands = speech.transcribe(audio_path, languages)
    heard, alts = pick_transcript(cands)
    if not heard.strip():                                  # unclear / silent audio: recover gracefully
        rep = _unclear_reply(agent)
    else:
        rep = agent.respond(heard, alternatives=alts or None)
    speech.synthesize(rep.spoken or rep.text, rep.language, out_audio)
    return heard, rep, cands


def run_turn_events(agent, emit, *, audio_path=None, text=None, speech=None, out_audio=None, speak=False, languages=None):
    """Run a voice turn (audio_path) or a text turn (text). Events: stage, heard, reply, audio, done.
    Raises on failure after emitting nothing further; the caller decides how to report it."""
    t0 = time.perf_counter()
    ms = lambda a, b: int(round((b - a) * 1000))
    timings = {}
    agent.on_stage = lambda name: emit({"type": "stage", "stage": name})
    try:
        if audio_path is not None:
            emit({"type": "stage", "stage": "transcribing"})
            t = time.perf_counter()
            cands = speech.transcribe(audio_path, languages)
            heard, alts = pick_transcript(cands)
            timings["stt_ms"] = ms(t, time.perf_counter())
            emit({"type": "heard", "text": heard, "alternatives": alts, "ms": timings["stt_ms"]})
            speak = True                                     # a spoken question gets a spoken answer
        else:
            heard, alts = text, []
        t = time.perf_counter()
        rep = _unclear_reply(agent) if not (heard or "").strip() else agent.respond(heard, alternatives=alts or None)
        timings["agent_ms"] = ms(t, time.perf_counter())
        emit({"type": "reply", "reply": rep, "timings": dict(timings)})
        if speak and speech is not None:
            emit({"type": "stage", "stage": "speaking"})
            t = time.perf_counter()
            speech.synthesize(rep.spoken or rep.text, rep.language, out_audio)
            timings["tts_ms"] = ms(t, time.perf_counter())
            emit({"type": "audio", "ms": timings["tts_ms"]})
        timings["total_ms"] = ms(t0, time.perf_counter())
        emit({"type": "done", "timings": timings})
        return heard, rep
    finally:
        agent.on_stage = None
