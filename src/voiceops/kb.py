"""Knowledge base: section-level chunking, hybrid retrieval, precedence & supersession handling."""
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import dataset_dir, TOP_K

PRECEDENCE = {"policy": 1.0, "agent_guide": 0.92, "help": 0.85, "faq": 0.7, "marketing": 0.45, "meta": 0.75}

# Romanised-Hindi / Devanagari -> English retrieval terms (query expansion; documents are mostly English)
LEXICON = {
    "wapas": "return", "vapas": "return", "waapas": "return", "लौटाने": "return", "लौटा": "return", "वापस": "return",
    "लौटना": "return", "रिटर्न": "return", "bhejna": "return send back", "लौटाना": "return",
    "paisa": "refund money", "paise": "refund money", "पैसे": "refund money", "रिफ़ंड": "refund", "रिफंड": "refund",
    "kab": "when how long", "कब": "when how long", "कितने": "how many long", "kitne": "how many long", "din": "days", "दिन": "days",
    "कैंसिल": "cancel", "cancel": "cancel", "रद्द": "cancel", "mana": "refuse", "मना": "refuse",
    "शिप": "shipped", "ship": "shipped", "डिलीवरी": "delivery", "deliver": "delivery", "delivered": "delivery",
    "खराब": "damaged defective", "टूटा": "damaged", "kharab": "damaged defective", "toota": "damaged",
    "गलत": "wrong item", "galat": "wrong item", "वारंटी": "warranty", "warranty": "warranty",
    "सील": "seal unopened", "खोल": "opened", "khol": "opened", "खुला": "opened",
    "कूपन": "coupon", "coupon": "coupon", "कैश": "cash", "ऑन": "on", "cod": "cash on delivery cod",
    "पिकअप": "pickup reverse pickup", "pickup": "reverse pickup", "फ़ीस": "fee", "fees": "fee", "फीस": "fee",
    "प्लस": "plus membership", "plus": "plus membership", "मुफ़्त": "free", "free": "free", "muft": "free",
    "बैंक": "bank transfer", "वॉलेट": "wallet", "wallet": "wallet", "तुरंत": "instant", "turant": "instant",
    "टिकट": "ticket", "इंसान": "human agent", "insaan": "human agent", "कोर्ट": "legal consumer court",
    "kaise": "how", "कैसे": "how", "kya": "", "hai": "", "है": "", "क्या": "", "mera": "my", "मेरा": "my",
    "order": "order", "ऑर्डर": "order", "sakta": "can", "sakti": "can", "सकता": "can", "सकती": "can",
    "karna": "do", "karein": "do", "करें": "do", "hoga": "", "होगा": "",
    "express": "express", "एक्सप्रेस": "express", "late": "late delay", "देर": "late delay",
    "आया": "arrived", "aaya": "arrived", "aa": "arrive", "jaana": "arrive", "chahiye": "",
    "दुकान": "store shop", "dukaan": "store shop", "स्टोर": "store",
}

STOP = set("the a an of to is are was were be been it its in on for and or if my me i you your we our can do does did will "
           "what how when where which who whom this that these those with at by from as not no than then there so but about "
           "into over after before up down out off again only own same too very just also have has had having".split())

# retrieval routing priors: analyzer intent -> sections that usually hold the answer
INTENT_PRIORS = {
    "refund_timeline": ["REF-2", "KB-00.3", "KB-00.1", "REF-1"],
    "refund_status": ["REF-1", "REF-2", "RET-7", "REF-6", "ESC-1"],
    "return_window": ["RET-2", "RET-3", "RET-8"],
    "return_eligibility": ["RET-2", "RET-4", "RET-5", "RET-6", "RET-8"],
    "cancel": ["CAN-1", "CAN-2", "CAN-3", "REF-1", "FAQH-1"],
    "delivery_status": ["TRK-1", "DEL-3", "DEL-1"],
    "late_delivery": ["DEL-4", "DEL-1"],
    "compensation": ["DEL-4", "ESC-4"],
    "damage_report": ["DWM-1", "DWM-2", "DWM-3", "DWM-4", "ESC-1"],
    "warranty": ["WAR-1", "WAR-2", "MKS-3", "WAR-3"],
    "troubleshooting": ["TRB-1", "TRB-2", "TRB-3", "TRB-4"],
    "coupon": ["CPN-1", "CPN-2", "REF-4"],
    "payment_cod": ["PAY-2", "PAY-4", "REF-3"],
    "cod_refund": ["REF-3", "REF-2", "PAY-4"],
    "plus": ["PLS-2", "PLS-1", "RET-6", "RET-7"],
    "human_request": ["ESC-1", "ESC-2", "FAQ-4"],
    "legal": ["ESC-1", "ESC-2"],
    "safety": ["DWM-5", "ESC-1"],
    "unauthorized_activity": ["ESC-1", "ESC-2"],
    "privacy": ["ESC-4", "IDV-4"],
    "delivery_policy": ["DEL-1", "DEL-2", "DEL-5", "DEL-6"],
}


def tokenize(text: str):
    text = text.lower()
    toks = re.findall(r"[\w\u0900-\u097F\u0980-\u0DFF]+", text)
    out = []
    for t in toks:
        if t in LEXICON:
            out += [w for w in LEXICON[t].split()]
            if t.isascii() and t not in STOP:
                out.append(t)
        elif t not in STOP:
            out.append(t)
    return out


@dataclass
class Chunk:
    id: str
    doc_id: str
    file: str
    title: str
    section_title: str
    text: str
    doc_type: str
    status: str
    effective_from: str
    language: str
    tokens: list = field(default_factory=list, repr=False)

    def render(self) -> str:
        sup = " [SUPERSEDED: applies only to cases the successor document says]" if self.status == "superseded" else ""
        return (f"[{self.id}] ({self.doc_type}, status={self.status}, effective_from={self.effective_from}){sup} "
                f"{self.title} / {self.section_title}\n{self.text}")


def _parse_front_matter(raw: str):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    meta, body = {}, raw
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return meta, body


def load_chunks():
    chunks = []
    for path in sorted((dataset_dir() / "kb").glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        parts = re.split(r"^##\s+", body, flags=re.M)
        for part in parts[1:]:
            head, _, rest = part.partition("\n")
            m = re.match(r"\[([^\]]+)\]\s*(.*)", head.strip())
            if not m:
                continue
            sid, stitle = m.group(1), m.group(2)
            text = rest.strip()
            c = Chunk(sid, meta.get("doc_id", ""), path.name, meta.get("title", ""), stitle, text,
                      meta.get("doc_type", ""), meta.get("status", "current"), meta.get("effective_from", ""),
                      meta.get("language", "en"))
            # title weighted x2 for lexical match
            c.tokens = tokenize(f"{meta.get('title','')} {stitle} {stitle} {text}")
            chunks.append(c)
    return chunks


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avg = sum(len(d) for d in docs) / max(1, self.N)
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, q):
        out = []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d)
            for t in set(q):
                f = self.tf[i].get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25([c.tokens for c in self.chunks])
        self.dense = None
        if os.getenv("RETRIEVER", "hybrid") == "hybrid":
            self._try_dense()

    def _try_dense(self):
        try:  # optional: multilingual dense embeddings (Hindi/Hinglish friendly)
            import numpy as np
            from sentence_transformers import SentenceTransformer
            name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
            self._st = SentenceTransformer(name)
            self._np = np
            self.dense = self._st.encode([f"passage: {c.section_title}. {c.text}" for c in self.chunks],
                                         normalize_embeddings=True)
        except Exception as e:  # pragma: no cover
            print(f"[kb] dense retrieval disabled ({type(e).__name__}); using BM25 only.")
            self.dense = None

    def search(self, query: str, intent: str | None = None, top_k: int = TOP_K,
               include_superseded: bool = False, force_ids=(), extra_queries=()):
        q = " ".join([query, *extra_queries])
        qt = tokenize(q)
        bm = self.bm25.scores(qt)
        mx = max(bm) or 1.0
        score = [b / mx for b in bm]
        if self.dense is not None:
            qv = self._st.encode([f"query: {q}"], normalize_embeddings=True)[0]
            ds = self.dense @ qv
            lo, hi = float(ds.min()), float(ds.max())
            ds = (ds - lo) / ((hi - lo) or 1.0)
            score = [0.5 * a + 0.5 * float(b) for a, b in zip(score, ds)]
        prior = INTENT_PRIORS.get(intent or "", [])
        res = []
        for i, c in enumerate(self.chunks):
            s = score[i] * PRECEDENCE.get(c.doc_type, 0.7)
            if c.status == "superseded":
                s *= 1.0 if include_superseded else 0.25
            if c.id in prior:
                s += 0.5 - 0.04 * prior.index(c.id)
            if c.id in force_ids:
                s += 2.0
            res.append((s, c))
        res.sort(key=lambda x: -x[0])
        forced = [(s_, c) for s_, c in res if c.id in force_ids]
        top = res[0][0] if res else 0.0
        cand = [(s_, c) for s_, c in res if c.id not in force_ids and s_ >= max(0.12, 0.35 * top)][: top_k + 3]
        kept = self._drop_shadowed(cand + forced)
        conflict = len(kept) < len(cand) + len(forced)           # a FAQ/marketing line lost to a policy
        kept_forced = [x for x in kept if x[1].id in force_ids]
        kept_rest = [x for x in kept if x[1].id not in force_ids][:top_k]
        out = sorted(kept_rest + kept_forced, key=lambda x: -x[0])
        if conflict and "KB-00.1" in self.by_id and all(c.id != "KB-00.1" for _, c in out):
            out.append((0.3, self.by_id["KB-00.1"]))              # cite the precedence rule that resolved the conflict
        return [{"id": c.id, "score": round(s, 3), "doc": c.file, "status": c.status,
                 "doc_type": c.doc_type, "text": c.render()} for s, c in out]

    @staticmethod
    def _drop_shadowed(scored):
        """KB-00.1 precedence at retrieval time: a FAQ / marketing line is dropped when a retrieved
        policy section already covers (>=50% of) its content words, because the policy wins on conflict
        (e.g. FAQ-1 '5 to 7 days' vs REF-2's per-method table)."""
        body = lambda c: set(tokenize(f"{c.section_title} {c.text}"))
        pol = [body(c) for _, c in scored if c.doc_type == "policy" and c.status == "current"]
        keep = []
        for s_, c in scored:
            if c.doc_type in ("faq", "marketing") and pol:
                t = body(c)
                if t and max(len(t & p) / len(t) for p in pol) >= 0.5:
                    continue
            keep.append((s_, c))
        return keep

    def toc(self) -> str:
        return "\n".join(f"{c.id}: {c.section_title}" for c in self.chunks if c.status == "current" or c.id.startswith("RETv3"))
