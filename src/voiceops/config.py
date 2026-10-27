"""Paths, environment and constants. Nothing secret is stored here."""
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 10, 20)  # fixed dataset reference date (KB-00.2)


def _load_dotenv():
    env = ROOT / ".env"
    if not env.exists():
        return
    text = env.read_text(encoding="utf-8-sig", errors="ignore")   # tolerate Windows BOM
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'"):
            v = v.strip(v[0])
        else:
            v = v.split(" #")[0].strip()
        if v:                       # a non-empty value in .env overrides stale shell variables
            os.environ[k.strip()] = v


_load_dotenv()


def dataset_dir() -> Path:
    cands = []
    if os.getenv("DATASET_DIR"):
        cands.append(Path(os.environ["DATASET_DIR"]).expanduser())
    name = "Capstone_Project_Dataset_FTTT_2026"
    cands += [ROOT / name, ROOT.parent / name, Path.cwd() / name, Path.cwd() / "dataset"]
    for c in cands:
        if (c / "kb").is_dir() and (c / "data").is_dir():
            return c.resolve()
    raise FileNotFoundError(
        "Dataset folder not found. Extract Capstone_Project_Dataset_FTTT_2026.zip and set "
        "DATASET_DIR in .env (folder containing kb/, data/, tools/, practice/).")


def import_tools():
    """Import the organizers' mock tools (lookup_order, create_ticket) from the dataset."""
    d = str(dataset_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    from tools import lookup_order, create_ticket  # noqa: E402
    return lookup_order, create_ticket


TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.getenv("GNANI_TTS_VOICE", "Kaveri")
STT_CANDIDATES = [c.strip() for c in os.getenv("GNANI_STT_CANDIDATES", "en-IN,hi-IN").split(",") if c.strip()]
TOP_K = int(os.getenv("TOP_K", "6"))

# ---- speed knobs (all optional; defaults are safe) -------------------------------------------------------------
CACHE_DIR = Path(os.getenv("VOICEOPS_CACHE_DIR", str(ROOT / ".cache"))).expanduser()
STT_GRACE_S = float(os.getenv("STT_GRACE_S", "1.0"))   # after the first STT language answers, wait at most this long for the others


def env_flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip().lower() not in ("0", "false", "no", "off")
