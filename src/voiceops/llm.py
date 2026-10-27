"""Thin, provider-agnostic LLM client (Anthropic Messages API or any OpenAI-compatible endpoint).

Speed notes
  * One shared keep-alive connection pool: the 2-3 sequential calls in a turn reuse one TLS connection.
  * Optional disk cache (LLM_CACHE=0 turns it off). Temperature is 0, so an identical prompt gives an identical
    answer; only *validated* JSON is cached, so a malformed reply is never replayed.
  * Non-transient errors (400/401/403/404) fail immediately instead of sleeping through three retries.
  * LLM_ANALYZER_MODEL (optional) lets the cheap "understanding" call use a faster model than the reply call.
  * LLM_REASONING_EFFORT (optional) overrides the Gemini "thinking" level (e.g. minimal / low / off).
"""
import json as _json
import os
import re
import sys
import threading
import time

import requests
from requests.adapters import HTTPAdapter

from .cache import DiskCache, make_key
from .config import CACHE_DIR, env_flag


class LLMError(RuntimeError):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


class LLMUnavailable(LLMError):
    """Network / auth / quota failure (as opposed to malformed model output)."""


NON_TRANSIENT = {400, 401, 403, 404}

_SESSION = None
_SESSION_LOCK = threading.Lock()


def http() -> requests.Session:
    """Process-wide keep-alive session (urllib3's pool is thread-safe)."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                adapter = HTTPAdapter(pool_connections=4, pool_maxsize=16, max_retries=0)
                s.mount("https://", adapter)
                s.mount("http://", adapter)
                _SESSION = s
    return _SESSION


def _timeout():
    return (float(os.getenv("LLM_CONNECT_TIMEOUT", "8")), int(os.getenv("LLM_TIMEOUT", "120")))


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    return _json.loads(text[start:i + 1])
                except _json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    raise LLMError(f"No JSON object in model output: {text[:200]!r}")


class LLM:
    def __init__(self, provider=None, model=None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        model = model or os.getenv("LLM_MODEL")
        # Forgiving setup: only an OpenAI-compatible key (e.g. Gemini/Groq) is present -> use that provider
        if self.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
            print("[llm] No ANTHROPIC_API_KEY found; using the OpenAI-compatible provider from OPENAI_* settings.", file=sys.stderr)
            if not model or model.lower().startswith("claude"):
                model = "gemini-3.8-flash" if "googleapis" in os.getenv("OPENAI_BASE_URL", "") else "gpt-4o-mini"
        self.model = model or ("claude-sonnet-5-5" if self.provider == "anthropic" else "gpt-4o-mini")
        print(f"[llm] provider={self.provider} model={self.model}", file=sys.stderr)
        self.calls = 0
        self.cache_hits = 0
        self.cache = None
        if env_flag("LLM_CACHE", True):
            c = DiskCache(CACHE_DIR / "llm", ".txt", max_bytes=50 * 1024 * 1024)
            self.cache = c if c.enabled else None

    # ------------------------------------------------------------------ public
    def complete(self, system: str, user: str, max_tokens: int = 700, temperature: float = 0.0,
                 history: list | None = None, model: str | None = None, use_cache: bool = True) -> str:
        msgs = list(history or []) + [{"role": "user", "content": user}]
        model = model or self.model
        ck = None
        if use_cache and self.cache and temperature == 0:
            ck = make_key("complete", self.provider, model, system, _json.dumps(msgs, ensure_ascii=False),
                          max_tokens, os.getenv("LLM_REASONING_EFFORT", ""))
            hit = self.cache.get_text(ck)
            if hit is not None:
                self.cache_hits += 1
                return hit
        last = None
        for attempt in range(3):
            try:
                self.calls += 1
                out = self._anthropic(system, msgs, max_tokens, temperature, model) if self.provider == "anthropic" \
                    else self._openai(system, msgs, max_tokens, temperature, model)
                if ck and out.strip():
                    self.cache.put_text(ck, out)
                return out
            except Exception as e:  # network / 429 / 5xx
                last = e
                print(f"[llm] attempt {attempt + 1}/3 failed: {str(e)[:400]}", file=sys.stderr)
                if getattr(e, "status", None) in NON_TRANSIENT:
                    break                                   # retrying a bad key / bad request only wastes time
                time.sleep(1.2 * (attempt + 1))
        raise LLMUnavailable(f"LLM call failed: {last}", status=getattr(last, "status", None))

    def json(self, system: str, user: str, max_tokens: int = 700, model: str | None = None) -> dict:
        ck = None
        if self.cache:
            ck = make_key("json", self.provider, model or self.model, system, user, max_tokens,
                          os.getenv("LLM_REASONING_EFFORT", ""))
            hit = self.cache.get_text(ck)
            if hit is not None:
                try:
                    data = _json.loads(hit)
                    self.cache_hits += 1
                    return data
                except ValueError:
                    pass
        data = extract_json(self.complete(system, user, max_tokens=max_tokens, model=model, use_cache=False))
        if ck:
            self.cache.put_text(ck, _json.dumps(data, ensure_ascii=False))
        return data

    def warmup(self):
        """Open the TLS connection before the first turn so the first answer doesn't pay for the handshake."""
        try:
            if self.provider == "anthropic":
                key = os.getenv("ANTHROPIC_API_KEY")
                if key:
                    http().get(os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/models?limit=1",
                               headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=(5, 8))
            else:
                key = os.getenv("OPENAI_API_KEY")
                if key:
                    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                    http().get(base + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=(5, 8))
        except Exception:
            pass                                            # warm-up is best effort only

    # ------------------------------------------------------------------ providers
    def _anthropic(self, system, msgs, max_tokens, temperature, model):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set", status=401)
        r = http().post(
            os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": max_tokens, "temperature": temperature,
                  "system": system, "messages": msgs}, timeout=_timeout())
        if r.status_code != 200:
            raise LLMError(f"{r.status_code}: {r.text[:300]}", status=r.status_code)
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")

    def _openai(self, system, msgs, max_tokens, temperature, model):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set", status=401)
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, *msgs]}
        if "generativelanguage.googleapis.com" in base:
            body["max_tokens"] = max(max_tokens, 1500)
            # hidden "thinking" is slow and can eat the token budget: keep it minimal (LLM_REASONING_EFFORT=off skips the field)
            effort = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
            if effort != "off":
                body["reasoning_effort"] = effort or ("none" if "2.5" in model else "low")
        headers = {"Authorization": f"Bearer {key}"}
        r = http().post(base + "/chat/completions", headers=headers, json=body, timeout=_timeout())
        if r.status_code == 400 and "reasoning_effort" in body:      # model does not accept that option: retry without it
            body.pop("reasoning_effort")
            r = http().post(base + "/chat/completions", headers=headers, json=body, timeout=_timeout())
        if r.status_code != 200:
            raise LLMError(f"{r.status_code}: {r.text[:300]}", status=r.status_code)
        content = (r.json()["choices"][0]["message"] or {}).get("content") or ""
        if not content.strip():
            raise LLMError("empty reply from model (try LLM_MODEL=gemini-2.0-flash)")
        return content
