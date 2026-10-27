"""Scoring that follows DATA_DICTIONARY.md: ' ; ' = all required, ' / ' = alternatives, case-insensitive,
equivalent dates / spoken numbers / native digits accepted."""
import csv
import re
import unicodedata
from datetime import date

from .config import dataset_dir
from .textnorm import ascii_digits, MONTHS

_HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", ascii_digits(str(s or ""))).lower()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)                      # 10,000 -> 10000
    s = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", s)           # 27th -> 27
    s = re.sub(r"\b(\d{1,2}) of ([a-z]+)", r"\1 \2", s)           # 27 of october -> 27 october
    s = re.sub(r"\b(" + "|".join(_WORDS) + r")\b", lambda m: _WORDS[m.group(1)], s)
    s = s.replace("₹", "rs ").replace("rupees", "rs")
    return re.sub(r"\s+", " ", s)


def _dates_in(text: str):
    """All dates mentioned in a reply, as ISO strings (accepts ISO, '27 October 2026', 'October 27, 2026')."""
    out = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    for d, mo, y in re.findall(r"(\d{1,2}) ([a-z]+),? (\d{4})", text):
        if mo in MONTHS:
            out.add(f"{y}-{MONTHS.index(mo)+1:02d}-{int(d):02d}")
    for mo, d, y in re.findall(r"([a-z]+) (\d{1,2}),? (\d{4})", text):
        if mo in MONTHS:
            out.add(f"{y}-{MONTHS.index(mo)+1:02d}-{int(d):02d}")
    return out


def _alt_hit(alt: str, reply_n: str, reply_dates: set) -> bool:
    a = norm(alt).strip()
    if not a:
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a):
        return a in reply_dates or a in reply_n
    return a in reply_n


def must_include_ok(reply: str, spec) -> tuple[bool, list]:
    if not spec or str(spec).lower() == "nan":
        return True, []
    rn = norm(reply)
    dates = _dates_in(rn)
    missing = []
    for fact in str(spec).split(" ; "):
        alts = [x for x in fact.split(" / ")]
        if not any(_alt_hit(a, rn, dates) for a in alts):
            missing.append(fact)
    return (not missing), missing


def must_not_ok(reply: str, spec) -> tuple[bool, list]:
    if not spec or str(spec).lower() == "nan":
        return True, []
    rn = norm(reply)
    hits = [p for p in str(spec).split(" ; ") if p.strip() and norm(p).strip() in rn]
    return (not hits), hits


def read_csv(rel):
    with open(dataset_dir() / rel, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ev_ids(spec):
    return [x.strip() for x in str(spec or "").split(" ; ") if x.strip() and not x.strip().startswith("products.csv")]


def score_single(row, reply: str, decision: str, retrieved_ids, used_ids):
    ok_inc, miss = must_include_ok(reply, row.get("must_include"))
    ok_not, hits = must_not_ok(reply, row.get("must_not_include"))
    accept = {x.strip() for x in (row.get("acceptable_decisions") or row.get("expected_decision") or "").split(";") if x.strip()} \
        or {row["expected_decision"]}
    accept.add(row["expected_decision"])
    ok_dec = decision in accept
    gold = ev_ids(row.get("gold_evidence"))
    rec = (len([g for g in gold if g in retrieved_ids]) / len(gold)) if gold else None
    return {"id": row["id"], "decision": decision, "expected": row["expected_decision"], "decision_ok": ok_dec,
            "include_ok": ok_inc, "missing": miss, "not_ok": ok_not, "forbidden_hit": hits,
            "correct": ok_dec and ok_inc and ok_not, "evidence_recall": rec, "retrieved": list(retrieved_ids)}
