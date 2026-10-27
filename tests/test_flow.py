"""Replays the practice data through the real agent code with a stub LLM (plumbing check)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evaluate as E
from fake_llm import FakeLLM
from voiceops.agent import Agent
from voiceops.kb import Retriever
from voiceops.evalkit import read_csv, score_single

KB = Retriever()
make = lambda: Agent(llm=FakeLLM(), retriever=KB)


def test_conversations_tools_and_decisions():
    res = E.run_step3(make)
    bad = [r for r in res if not (r["tool_ok"] and r["decision_ok"] and r["not_ok"])]
    for r in bad:
        print(r["conversation_id"], r["turn"], r["expected"], r["decision"], r["tool_ok"], r["tools"], r["forbidden_hit"])
    assert not bad
