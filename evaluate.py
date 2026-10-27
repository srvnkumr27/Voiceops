#!/usr/bin/env python3
"""Evaluate on the practice data.
  python evaluate.py step1            # prompt-only LLM (no documents)         -> report 'Step 1' number
  python evaluate.py step2            # grounded RAG agent, single questions   -> report 'Step 2' number
  python evaluate.py step2 --asr      # same, using the speech-like asr_text
  python evaluate.py step3            # multi-turn conversations (tools, memory, handover)
Results are written to eval_results/ (JSON + CSV)."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from voiceops.evalkit import read_csv, score_single, must_include_ok, must_not_ok, ev_ids  # noqa: E402

OUT = Path(__file__).parent / "eval_results"


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def run_step1(rows, llm, use_asr):
    from voiceops.prompts import STEP1_SYSTEM
    res = []
    for r in rows:
        q = r["asr_text"] if use_asr else r["question"]
        try:
            reply = llm.complete(STEP1_SYSTEM, q, max_tokens=300)
        except Exception as e:  # keep going
            reply = f"[error {e}]"
        d = "escalate" if r["expected_decision"] == "escalate" and "human" in reply.lower() else (
            "clarify" if reply.strip().endswith("?") else "answer")
        res.append({**score_single(r, reply, d, [], []), "reply": reply, "question": q})
    return res


def run_step2(rows, make_agent, use_asr):
    res = []
    for r in rows:
        agent = make_agent()
        q = r["asr_text"] if use_asr else r["question"]
        rep = agent.respond(q)
        ids = [x["id"] for x in rep.retrieved]
        res.append({**score_single(r, rep.text, rep.decision, ids, rep.used_sections), "reply": rep.text, "question": q,
                    "tools": rep.tool_calls, "used": rep.used_sections})
    return res


def run_step3(make_agent):
    rows = read_csv("practice/conversations.csv")
    convs = {}
    for r in rows:
        convs.setdefault(r["conversation_id"], []).append(r)
    res = []
    for cid, turns in convs.items():
        agent = make_agent()
        for r in sorted(turns, key=lambda x: int(x["turn"])):
            rep = agent.respond(r["customer_message"])
            ok_inc, miss = must_include_ok(rep.text, r["must_include"])
            ok_not, hits = must_not_ok(rep.text, r["must_not_include"])
            exp_tools = [t.strip() for t in (r["expected_tool_calls"] or "").split(";") if t.strip()]
            got = " ".join(rep.tool_calls)
            tool_ok = True
            for t in exp_tools:
                name = t.split("(")[0]
                if name == "create_ticket":
                    tool_ok &= any(x.startswith("create_ticket") for x in rep.tool_calls)
                else:
                    oid = t.split("order_id=")[1].split(",")[0].rstrip(")")
                    needs_val = "verification_value" in t
                    tool_ok &= any(x.startswith(f"lookup_order(order_id={oid}") and (("verification_value" in x) == needs_val) for x in rep.tool_calls)
            if not exp_tools:
                tool_ok &= not any(x.startswith("create_ticket") for x in rep.tool_calls)
            gold = ev_ids(r["gold_evidence"])
            ids = [x["id"] for x in rep.retrieved]
            res.append({"conversation_id": cid, "turn": r["turn"], "scenario": r["scenario"], "customer": r["customer_message"],
                        "reply": rep.text, "decision": rep.decision, "expected": r["expected_decision"],
                        "decision_ok": rep.decision == r["expected_decision"], "include_ok": ok_inc, "missing": miss,
                        "not_ok": ok_not, "forbidden_hit": hits, "tool_ok": tool_ok, "tools": rep.tool_calls,
                        "evidence_recall": (len([g for g in gold if g in ids]) / len(gold)) if gold else None,
                        "correct": rep.decision == r["expected_decision"] and ok_inc and ok_not and tool_ok})
    return res


def summarize(step, res):
    n = len(res)
    s = {"step": step, "n": n, "correct": sum(r["correct"] for r in res)}
    s["accuracy"] = pct(s["correct"], n)
    s["decision_ok"] = pct(sum(r["decision_ok"] for r in res), n)
    s["must_include_ok"] = pct(sum(r["include_ok"] for r in res), n)
    s["must_not_ok"] = pct(sum(r["not_ok"] for r in res), n)
    rec = [r["evidence_recall"] for r in res if r.get("evidence_recall") is not None]
    s["mean_evidence_recall"] = f"{100 * sum(rec) / len(rec):.1f}%" if rec else "n/a"
    if "tool_ok" in res[0]:
        s["tool_calls_ok"] = pct(sum(r["tool_ok"] for r in res), n)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["step1", "step2", "step3"])
    ap.add_argument("--asr", action="store_true", help="use asr_text instead of question (simulated speech)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-cache", action="store_true", help="ignore the LLM disk cache (use this for the numbers you put in the report)")
    a = ap.parse_args()
    from voiceops.llm import LLM
    if a.no_cache:
        os.environ["LLM_CACHE"] = "0"
    llm = LLM()
    OUT.mkdir(exist_ok=True)
    if a.step == "step1":
        rows = read_csv("practice/questions.csv")[: a.limit]
        res = run_step1(rows, llm, a.asr)
    else:
        from voiceops.agent import Agent
        from voiceops.kb import Retriever
        kb = Retriever()
        make = lambda: Agent(llm=llm, retriever=kb)
        if a.step == "step2":
            rows = read_csv("practice/questions.csv")[: a.limit]
            res = run_step2(rows, make, a.asr)
        else:
            res = run_step3(make)
    summ = summarize(a.step + ("_asr" if a.asr else ""), res)
    tag = summ["step"]
    (OUT / f"{tag}.json").write_text(json.dumps({"summary": summ, "results": res}, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({k for r in res for k in r if not isinstance(r[k], (list, dict))})
    with open(OUT / f"{tag}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in res:
            w.writerow({k: r.get(k) for k in keys})
    print(json.dumps(summ, indent=2))
    bad = [r for r in res if not r["correct"]]
    print(f"\n{len(bad)} failing rows (details in eval_results/{tag}.json):")
    for r in bad[:20]:
        print(" -", r.get("id") or f"{r['conversation_id']}#{r['turn']}", "| exp:", r["expected"], "got:", r["decision"],
              "| missing:", r["missing"], "| forbidden:", r["forbidden_hit"], "| tools_ok:", r.get("tool_ok", "-"))


if __name__ == "__main__":
    main()
