"""Tiny persistent cache (the storage layer for speed). Content-addressed, atomic writes, size-capped.

Used for two things only:
  * LLM results   (temperature is 0, so an identical prompt gives an identical answer)
  * TTS audio     (scripted lines such as the intro and the order-ID prompt repeat in almost every call)
A cache problem must never break a call: every failure degrades to "miss".
"""
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def make_key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


class DiskCache:
    def __init__(self, root, suffix=".txt", max_bytes=0):
        self.suffix, self.max_bytes, self.hits, self.misses = suffix, max_bytes, 0, 0
        self._puts = 0
        try:
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:                       # read-only disk etc.: behave as an always-empty cache
            self.root = None

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def path(self, key: str):
        return (self.root / f"{key}{self.suffix}") if self.root else None

    # ---------------------------------------------------------------- text
    def get_text(self, key: str):
        p = self.path(key)
        try:
            if p and p.is_file():
                text = p.read_text(encoding="utf-8")
                self.hits += 1
                os.utime(p, None)             # keep recently used entries when pruning
                return text
        except OSError:
            pass
        self.misses += 1
        return None

    def put_text(self, key: str, text: str):
        self._write(key, lambda f: f.write(text.encode("utf-8")))

    # ---------------------------------------------------------------- files (audio)
    def copy_to(self, key: str, dest) -> bool:
        p = self.path(key)
        try:
            if p and p.is_file() and p.stat().st_size > 44:
                shutil.copyfile(p, dest)
                self.hits += 1
                os.utime(p, None)
                return True
        except OSError:
            pass
        self.misses += 1
        return False

    def put_file(self, key: str, src):
        def w(f):
            with open(src, "rb") as s:
                shutil.copyfileobj(s, f)
        self._write(key, w)

    # ---------------------------------------------------------------- internals
    def _write(self, key, writer):
        p = self.path(key)
        if not p:
            return
        try:
            fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".part")
            with os.fdopen(fd, "wb") as f:
                writer(f)
            os.replace(tmp, p)                # atomic: readers never see half a file
            self._puts += 1
            if self.max_bytes and self._puts % 25 == 0:
                self._prune()
        except OSError:
            pass

    def _prune(self):
        try:
            files = [(f.stat().st_mtime, f.stat().st_size, f) for f in self.root.glob(f"*{self.suffix}")]
            total = sum(s for _, s, _ in files)
            for _, size, f in sorted(files):  # oldest first
                if total <= self.max_bytes:
                    break
                f.unlink(missing_ok=True)
                total -= size
        except OSError:
            pass
