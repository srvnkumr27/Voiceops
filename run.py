#!/usr/bin/env python3
"""Batch run: a folder of customer audio files -> replies as text + audio + the KB chunks used.

  python run.py --audio-dir ./test_audio --out ./out

Each file is an independent call, UNLESS names look like  <conversation>_t<turn>.wav  (e.g. PC002_t3.wav):
those are grouped into one conversation and played in turn order. Add --text-mode to feed .txt files
(one customer message per file) instead of audio, e.g. for debugging without Gnani."""
import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
GROUP = re.compile(r"^(?P<conv>.+?)[_-]t(?:urn)?(?P<turn>\d+)$", re.I)


def plan(files):
    groups, singles = {}, []
    for f in files:
        m = GROUP.match(f.stem)
        if m:
            groups.setdefault(m["conv"], []).append((int(m["turn"]), f))
        else:
            singles.append([f])
    return [[f for _, f in sorted(v)] for v in groups.values()] + singles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--languages", help="comma list of STT languages to try, default from GNANI_STT_CANDIDATES")
    ap.add_argument("--text-mode", action="store_true", help="read .txt files instead of audio (no Gnani calls)")
    a = ap.parse_args()

    from voiceops.agent import Agent
    from voiceops.kb import Retriever
    from voiceops.llm import LLM
    in_dir, out = Path(a.audio_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exts = {".txt"} if a.text_mode else AUDIO_EXT
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        sys.exit(f"No {'text' if a.text_mode else 'audio'} files found in {in_dir}")
    kb, llm = Retriever(), LLM()
    speech = None
    if not a.text_mode:
        from voiceops.pipeline import voice_turn
        from voiceops.speech import Speech
        speech = Speech()
    langs = [x.strip() for x in a.languages.split(",")] if a.languages else None
    rows = []
    for convo in plan(files):
        agent = Agent(llm=llm, retriever=kb)   # fresh session per call
        for f in convo:
            rec = {"file": f.name}
            try:
                if a.text_mode:
                    heard, rep, cands = f.read_text(encoding="utf-8").strip(), None, {}
                    rep = agent.respond(heard)
                else:
                    heard, rep, cands = voice_turn(f, agent, speech, out / f"{f.stem}_reply.wav", langs)
                rec.update({"transcript": heard, "stt_candidates": cands, "reply_text": rep.text, "spoken_text": rep.spoken,
                            "language": rep.language, "decision": rep.decision, "tool_calls": rep.tool_calls,
                            "escalation_reasons": rep.escalation_reasons, "ticket": rep.ticket,
                            "sources_used": rep.used_sections,
                            "chunks": [{"id": c["id"], "score": c["score"], "text": c["text"]} for c in rep.retrieved],
                            "reply_audio": None if a.text_mode else f"{f.stem}_reply.wav", "latency_s": rep.latency_s})
                (out / f"{f.stem}.txt").write_text(rep.text + "\n", encoding="utf-8")
            except Exception as e:  # never abort the whole batch on one bad file
                rec["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            (out / f"{f.stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            rows.append(rec)
            print(f"[{f.name}] {rec.get('decision', 'ERROR')}: {rec.get('reply_text', rec.get('error'))[:110]}")
    with open(out / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "transcript", "reply_text", "decision", "sources_used", "retrieved_ids", "reply_audio", "error"])
        for r in rows:
            w.writerow([r["file"], r.get("transcript", ""), r.get("reply_text", ""), r.get("decision", ""),
                        " ; ".join(r.get("sources_used", [])), " ; ".join(c["id"] for c in r.get("chunks", [])),
                        r.get("reply_audio", ""), r.get("error", "")])
    print(f"\nDone: {len(rows)} file(s) -> {out}/ (JSON per file, results.csv, *_reply.wav)")


if __name__ == "__main__":
    main()
