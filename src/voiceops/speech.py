"""Gnani speech layer. STT and TTS use ONLY the official gnani-vachana SDK (brief rule: no other speech service).

Speed notes
  * STT tries several languages in parallel; we use the first one's answer and wait only STT_GRACE_S more for the rest.
  * TTS audio is cached on disk (scripted lines repeat in almost every call); prewarm() fills it before the first call.
"""
import concurrent.futures as cf
import os
import shutil
import tempfile
import time
import uuid
import wave
from pathlib import Path

from .cache import DiskCache, make_key
from .config import TTS_MODEL, TTS_VOICE, STT_CANDIDATES, STT_GRACE_S, CACHE_DIR, env_flag

TTS_LANG = {"en": "en-IN", "hi": "hi-IN", "hi-en": "hi-en", "mr": "mr-IN", "ta": "ta-IN", "te": "te-IN",
            "kn": "kn-IN", "ml": "ml-IN", "gu": "gu-IN", "bn": "bn-IN", "pa": "pa-IN"}
# timbre-v3.0: 13 voices, each speaks all 10 languages -> used for non-English replies
V30_VOICE = "Shivani"
MAX_CLIP_S = 55  # REST STT accepts clips up to 60 s


def _wav_chunks(path: Path, tmpdir: Path):
    """Split long WAVs into <=55 s pieces (other formats are passed through untouched).
    Returns (parts, temp_files_to_delete)."""
    if path.suffix.lower() != ".wav":
        return [path], []
    with wave.open(str(path), "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        if n / rate <= MAX_CLIP_S:
            return [path], []
        params, step, out, i = w.getparams(), int(MAX_CLIP_S * rate), [], 0
        tag = uuid.uuid4().hex[:8]
        while True:
            frames = w.readframes(step)
            if not frames:
                break
            p = tmpdir / f"{path.stem}_{tag}_part{i}.wav"
            with wave.open(str(p), "wb") as o:
                o.setparams(params); o.writeframes(frames)
            out.append(p); i += 1
        return out, list(out)


class Speech:
    def __init__(self):
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        self.stt = GnaniSTTClient()   # reads GNANI_API_KEY from the environment
        self.tts = GnaniTTSClient()
        self.tts_cache = None
        if env_flag("TTS_CACHE", True):
            c = DiskCache(CACHE_DIR / "tts", ".wav", max_bytes=120 * 1024 * 1024)
            self.tts_cache = c if c.enabled else None

    # ---------------------------------------------------------------- listen
    def _transcribe_one(self, path: Path, lang: str) -> str:
        fmt = "transcribe" if lang in ("en-IN", "hi-IN") else "verbatim"   # ITN: digits/dates in written form
        r = self.stt.transcribe(str(path), language_code=lang, format=fmt)
        return (r.get("transcript") or "").strip()

    def transcribe(self, audio_path, languages=None, tmpdir=None, grace_s=None):
        """Returns {lang_code: transcript}. en-IN also covers English mixed with Hindi.

        All languages run in parallel. Once the FIRST language returns a usable transcript (>= 3 words) we wait at most
        `grace_s` for the others instead of for the slowest one; an empty/short first answer waits for everyone."""
        languages = languages or STT_CANDIDATES
        grace = STT_GRACE_S if grace_s is None else grace_s
        path = Path(audio_path)
        tmp = Path(tmpdir) if tmpdir else Path(tempfile.gettempdir())
        parts, temp_files = _wav_chunks(path, tmp)

        def run(lang):
            try:
                return lang, " ".join(t for t in (self._transcribe_one(p, lang) for p in parts) if t)
            except Exception:  # one failing language must not kill the turn
                return lang, ""

        ex = cf.ThreadPoolExecutor(max_workers=len(languages))
        results = {lang: "" for lang in languages}
        try:
            pending = {ex.submit(run, lang) for lang in languages}
            deadline = None
            while pending:
                timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                done, pending = cf.wait(pending, timeout=timeout, return_when=cf.FIRST_COMPLETED)
                if not done:
                    break                                   # grace window over: proceed without the stragglers
                for f in done:
                    lang, text = f.result()
                    results[lang] = text
                    if lang == languages[0] and len(text.split()) >= 3 and deadline is None:
                        deadline = time.monotonic() + grace
        finally:
            ex.shutdown(wait=False, cancel_futures=True)    # never block the call on a slow straggler
            if temp_files:                                  # stragglers may still read these: clean up a bit later
                import threading
                threading.Timer(90, lambda: [Path(f).unlink(missing_ok=True) for f in temp_files]).start()
        return results

    # ---------------------------------------------------------------- speak
    def synthesize(self, text: str, lang: str, out_file):
        """Write reply audio. Always names the model (the SDK's old default is no longer served)."""
        code = TTS_LANG.get(lang, "en-IN")
        model, voice = TTS_MODEL, TTS_VOICE
        if lang != "en":                       # Hindi / regional: v3.0 voices speak every language
            model, voice = "timbre-v3.0", os.getenv("GNANI_TTS_VOICE_V3", V30_VOICE)
        attempts = [(code, model, voice)]
        if code == "hi-en":                    # SDK 0.8.x validator may reject 'hi-en' -> fall back to hi-IN
            attempts.append(("hi-IN", model, voice))
        attempts.append(("en-IN", TTS_MODEL, TTS_VOICE))
        last = None
        for c, m, v in attempts:
            ck = make_key("tts", text, c, m, v) if self.tts_cache else None
            if ck and self.tts_cache.copy_to(ck, out_file):
                return str(out_file)
            try:
                self.tts.synthesize(text, voice=v, model=m, language=c, output_file=str(out_file))
                if ck:
                    self.tts_cache.put_file(ck, out_file)
                return str(out_file)
            except Exception as e:
                last = e
        raise RuntimeError(f"TTS failed: {last}")

    def prewarm(self, langs=("en",)):
        """Synthesize the scripted lines once (best effort, background) so the first call has instant audio."""
        from . import templates as TPL
        from .textnorm import speakable
        tmp = Path(tempfile.gettempdir()) / f"voiceops-prewarm-{uuid.uuid4().hex[:6]}.wav"
        try:
            for lang in langs:
                intro = TPL.intro(lang) or ""
                for kind in ("ask_order_id", "ask_verification", "greeting", "verified_ok", "didnt_catch", "retry_verification"):
                    body = TPL.get(kind, lang)
                    if not body:
                        continue
                    variants = [body] + ([f"{intro} {body}"] if intro and kind in ("ask_order_id", "greeting") else [])
                    for v in variants:
                        try:
                            self.synthesize(speakable(v, lang), lang, tmp)
                        except Exception:
                            return                           # no key / offline: stop quietly
        finally:
            tmp.unlink(missing_ok=True)


def pick_transcript(cands: dict):
    """Choose the primary hypothesis; all hypotheses are passed to the agent's analyzer as alternatives."""
    nonempty = {k: v for k, v in cands.items() if v}
    if not nonempty:
        return "", []
    primary = nonempty.get("en-IN") or next(iter(nonempty.values()))
    # a Devanagari hi-IN transcript wins when en-IN returned (near-)nothing useful
    hi = nonempty.get("hi-IN", "")
    if hi and any("\u0900" <= ch <= "\u097F" for ch in hi) and len(primary.split()) < max(2, len(hi.split()) // 2):
        primary = hi
    return primary, [v for v in nonempty.values() if v != primary]
