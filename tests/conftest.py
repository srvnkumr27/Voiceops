import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("RETRIEVER", "bm25")


import pytest


@pytest.fixture(scope="session")
def kb():
    from voiceops.kb import Retriever
    return Retriever()
