"""ZentroMart support agent (Steps 2 + 3).

Design: the LLM *understands* and *phrases*; code does everything that must be exactly right.
  1. understand  -> LLM analyzer returns JSON (language, intent, English standalone query, flags, entities)
  2. state       -> code: order-ID capture, identity check (max one retry), memory of order/item/pending question
  3. rules       -> code: ESC-1 handover triggers (value, repeat contact, overdue refund, legal, safety ...)
  4. facts       -> code: business-day / return-window / refund-ETA arithmetic from the verified order + catalog
  5. retrieve    -> hybrid retrieval over KB sections (keeps [SECTION-ID], status, precedence)
  6. respond     -> LLM writes a short spoken reply from ONLY the retrieved text + computed facts
  7. guard       -> code: no false claims of actions, AI disclosure, no ticket claims without created=True
"""
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import dates as D
from . import facts as F
from . import templates as TPL
from .config import TODAY, import_tools, TOP_K
from .datastore import get_store
from .kb import Retriever
from .llm import LLM, LLMError, LLMUnavailable
from .prompts import ANALYZER_SYSTEM, RESPONDER_SYSTEM
from .textnorm import (clean_asr, extract_order_id, extract_verification, digit_run_lengths, ascii_digits,
                       speakable)

ORDER_INTENTS = {"refund_status", "delivery_status", "late_delivery", "return_eligibility", "cancel", "damage_report"}
SHIPPED_LIKE = {"shipped", "out_for_delivery", "delivery_attempted"}

# ------------------------------------------------------------------ safety-net regexes (en / Hinglish / Hindi)
_R = lambda p: re.compile(p, re.I)
RX_HUMAN = _R(r"\b(human|real person|live agent|representative|speak (to|with) (a |an )?(person|agent|someone|executive)|talk (to|with) (a |an )?(person|agent|someone|executive)|customer care executive|manager)\b"
              r"|इंसान|इन्सान|किसी (व्यक्ति|आदमी) से|ह्यूमन|एजेंट से|insaan se|kisi insaan|human se|agent se baat|manager se")
RX_LEGAL = _R(r"\b(lawyer|legal action|legal notice|court|consumer (forum|court|commission)|police|sue you|advocate)\b|कोर्ट|पुलिस|वकील|उपभोक्ता (फोरम|आयोग|अदालत)|कानूनी")
RX_SAFETY = _R(r"swell|bulg|burning smell|smell of burn|burnt smell|spark|smok(e|ing)|caught fire|on fire|overheat|explod|blast|got hurt|injur"
               r"|सूज|जलने की|चिंगारी|धुआँ|धुआं|आग लग|jalne ki|dhuan|chingari")
RX_REPEAT = _R(r"third time|3rd time|again and again|several times|teesri baar|तीसरी बार|baar baar|बार बार")
RX_COMP = _R(r"compensat|voucher|मुआवज|वाउचर|goodwill")
RX_DAMAGE = _R(r"damag|broken|cracked|crack|torn|defect|wrong item|wrong product|missing|expired|टूटा|टूट|खराब|क्रैक|गलत सामान|kharab|toota|galat")

BAD_ACTION = [
    _R(r"\b(i|i've|ive|i have)\s+(have\s+)?(cancel+ed)\b"), _R(r"cancel+ed (it|your order) for you"), _R(r"cancel kar diya"), _R(r"कैंसिल कर दिया"),
    _R(r"\b(has|have) been (approved|credited|issued|added)\b"), _R(r"voucher (has been |is )?added"),
    _R(r"\b(i|i've|ive|i have)\s+(have\s+)?(approved|credited|issued|processed your refund)\b"), _R(r"approve kar diya"),
]
RX_TICKET_CLAIM = _R(r"ticket.{0,40}(created|raised|logged|bana diya|बना दिया|बनाया)|(created|raised|logged|bana diya|बना दिया|बनाया).{0,40}ticket|टिकट (बना|बनाया)|ticket bana")

# ------------------------------------------------------------------ deterministic fast paths (no LLM call needed)
# A turn made ONLY of these words is a generic greeting / "I need help with an order" opener. Any other word at all
# (a product, a complaint, a date...) sends the turn to the normal LLM analyzer, so nothing is ever guessed.
_GREET_WORDS = set("hi hii hiii hello hey hallo namaste namaskar good morning afternoon evening there ji jee team support "
                   "zentromart sir madam नमस्ते नमस्कार हेलो हैलो हाय जी".split())
_HELP_WORDS = set("i need some help with an a my the order orders about regarding can you me please pls mujhe mujhko apne apna "
                  "ek ke baare bare mein me par madad chahiye chahiy hai kuch ki मुझे अपने अपना ऑर्डर आर्डर के बारे में मदद हेल्प कुछ एक की चाहिए".split())
_HELP_CORE = {"help", "madad", "मदद", "हेल्प"}
_ORDER_WORDS = {"order", "orders", "ऑर्डर", "आर्डर"}
_HINGLISH_MARK = set("mujhe mujhko chahiye madad baare bare mein apne apna namaste namaskar kuch".split())
_FILLER = set("order orders id number no num is it its it's my the this that hai hain hoon ka ki ke mera meri mere yeh ye woh wo "
              "please pls yes yeah haan ji ok okay sure so and aur bata bataiye dijiye hmm uh um ऑर्डर आर्डर आईडी आइडी नंबर नम्बर "
              "है हैं मेरा मेरी मेरे का की के यह ये जी हाँ हां कृपया".split())


def _words(text: str):
    return re.findall(r"[\w\u0900-\u0963\u0966-\u097F']+", (text or "").lower())   # excludes the danda


def fast_opener(text: str):
    """('greeting' | 'ask_order_id', language) for a pure greeting / generic help opener, else None."""
    toks = _words(text)
    if not toks or len(toks) > 14 or any(ch.isdigit() for ch in text):
        return None
    allowed = _GREET_WORDS | _HELP_WORDS
    if not all(t in allowed for t in toks):
        return None
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        lang = "hi"
    elif any(t in _HINGLISH_MARK for t in toks):
        lang = "hi-en"
    else:
        lang = "en"
    if all(t in _GREET_WORDS for t in toks):
        return "greeting", lang
    if not (set(toks) & _HELP_CORE):
        return None
    return ("ask_order_id" if set(toks) & _ORDER_WORDS else "greeting"), lang


def only_filler(rest: str) -> bool:
    """True when what is left of an utterance after removing the order ID carries no request ('order number hai ...')."""
    toks = _words(rest)
    return bool(toks) and all(t in _FILLER for t in toks)


@dataclass
class Session:
    history: list = field(default_factory=list)       # [{"role": "user"/"assistant", "content": str}] (secrets redacted)
    language: str = "en"
    order_id: Optional[str] = None
    awaiting_verification: bool = False
    verify_attempts: int = 0
    verified: bool = False
    locked: bool = False
    order: Optional[dict] = None
    pending_question: Optional[str] = None
    last_skus: list = field(default_factory=list)
    last_intent: Optional[str] = None
    greeted: bool = False
    tickets: list = field(default_factory=list)       # [{"reason": .., "ticket_id": ..}]
    turn: int = 0

    def reset_order(self):
        self.order_id, self.order = None, None
        self.awaiting_verification = self.verified = self.locked = False
        self.verify_attempts = 0


@dataclass
class Reply:
    text: str
    spoken: str
    language: str
    decision: str
    retrieved: list = field(default_factory=list)      # [{"id","score","doc","status",...}]
    used_sections: list = field(default_factory=list)  # IDs the reply relied on
    tool_calls: list = field(default_factory=list)
    ticket: Optional[dict] = None
    analysis: dict = field(default_factory=dict)
    latency_s: float = 0.0
    escalation_reasons: list = field(default_factory=list)

    def to_dict(self, with_text=True):
        d = asdict(self)
        if not with_text:
            for r in d["retrieved"]:
                r.pop("text", None)
        return d


def _detect_script_lang(text: str, fallback="en") -> str:
    if re.search(r"[\u0900-\u097F]", text): return "hi"
    if re.search(r"[\u0B80-\u0BFF]", text): return "ta"
    if re.search(r"[\u0C00-\u0C7F]", text): return "te"
    if re.search(r"[\u0C80-\u0CFF]", text): return "kn"
    if re.search(r"[\u0D00-\u0D7F]", text): return "ml"
    if re.search(r"[\u0A80-\u0AFF]", text): return "gu"
    if re.search(r"[\u0980-\u09FF]", text): return "bn"
    if re.search(r"[\u0A00-\u0A7F]", text): return "pa"
    if re.search(r"\b(kya|hai|hain|nahi|mujhe|mera|meri|kab|kitne|kaise|karna|karein|wala|chahiye|aap|aur|hoon|tha|ko|pe|mein|ka|ki)\b", text, re.I):
        return "hi-en"
    return fallback if not text.strip() else "en"


class Agent:
    def __init__(self, llm: Optional[LLM] = None, retriever: Optional[Retriever] = None, today=TODAY):
        self.llm = llm or LLM()
        self.kb = retriever or Retriever()
        self.store = get_store()
        self.lookup_order, self.create_ticket = import_tools()
        self.today = today
        self.s = Session()
        self._toc = self.kb.toc()
        self.on_stage = None          # optional callback(name) so a UI can show progress: understanding / retrieving / writing

    def _stage(self, name):
        cb = self.on_stage
        if cb:
            try:
                cb(name)
            except Exception:
                pass

    # ================================================================ public API
    def reset(self):
        self.s = Session()

    def respond(self, user_text: str, alternatives=None) -> Reply:
        t0 = time.time()
        s = self.s
        s.turn += 1
        text = clean_asr(user_text or "", self.store.sku_prefixes)
        if not text.strip():
            lang = s.language
            msg = TPL.get("didnt_catch", lang) or TPL.get("didnt_catch", "en")
            return self._finish(Reply(msg, "", lang, "clarify"), user_text, t0, record=False)

        tool_calls: list = []
        self._secrets = set()
        new_oid, rest = extract_order_id(text)
        code_guess = extract_verification(rest if new_oid else text) if (s.awaiting_verification or new_oid) else None
        pure_code = bool(s.awaiting_verification and code_guess and len(text) <= 70 and not new_oid)
        pure_id = bool(new_oid and only_filler(rest))          # "order number hai 8978 2837": nothing else to understand

        fast = None if (new_oid or s.order_id or s.awaiting_verification or s.verified) else fast_opener(text)
        if fast:                                               # greeting / generic opener: answered by a template, no LLM
            kind, flang = fast
            s.language = flang
            a0 = self._default_analysis(text)
            a0.update(language=flang, intent="greeting" if kind == "greeting" else "order_help", needs_order=(kind == "ask_order_id"))
            return self._scripted(kind, flang, user_text, t0, tool_calls, a0, intro=not s.greeted)

        analysis = self._default_analysis(text)
        if not (pure_code or pure_id):
            self._stage("understanding")
            analysis = self._analyze(text, alternatives)
        script_lang = _detect_script_lang(text, "")
        non_latin = script_lang not in ("", "en", "hi-en")
        if pure_code or pure_id:
            lang = script_lang if non_latin else s.language
        else:
            lang = analysis["language"]
            if lang == "en" and non_latin:
                lang = script_lang
            if s.language == "mr" and lang == "hi":
                lang = "mr"
        s.language = lang
        analysis["language"] = lang

        # ---------------------------------------------------------- identity flow (code, not LLM)
        needs_order = self._needs_order(analysis)
        generic = analysis["intent"] in ("order_help", "verification_reply", "chitchat", "greeting")
        just_verified = False
        verify_msg = None
        if new_oid and new_oid != s.order_id:
            s.reset_order()
            s.order_id, s.awaiting_verification = new_oid, True
            self._lookup(new_oid, None, tool_calls)
            verify_msg = "ask_verification"
        elif (not new_oid and s.order_id is None and len(text) <= 80 and s.turn <= 4 and generic
              and any(5 <= n <= 12 and n != 8 for n in digit_run_lengths(text))):
            verify_msg = "order_id_unclear"

        if s.awaiting_verification and not s.locked:
            code = code_guess or (re.sub(r"\D", "", str(analysis.get("verification_value") or "")) or None)
            if code and len(code) in (4, 6):
                self._secrets.add(code)
                res = self._lookup(s.order_id, code, tool_calls)
                if res.get("found") and res.get("verified"):
                    s.verified, s.order, s.awaiting_verification, just_verified = True, res, False, True
                    verify_msg = None
                else:
                    s.verify_attempts += 1
                    if s.verify_attempts >= 2:
                        s.locked, s.awaiting_verification = True, False
                        verify_msg = "verification_failed"
                    else:
                        verify_msg = "retry_verification"
            elif verify_msg is None and (needs_order or generic):
                verify_msg = "ask_verification"   # general policy questions are answered without verification

        # ---------------------------------------------------------- effective question (memory)
        question = text
        if just_verified and s.pending_question:
            question = s.pending_question
            analysis = self._analyze(question, None, context_note="The customer has just been verified; this is their earlier request.")
            analysis["language"] = lang
            needs_order = self._needs_order(analysis)
        elif just_verified:
            analysis["intent"] = "verification_reply"

        # ---------------------------------------------------------- flags (LLM OR regex safety net)
        flags = self._flags(question, analysis)

        # ---------------------------------------------------------- decide: scripted ID-check replies
        if not just_verified and not flags["escalate_now"]:
            if verify_msg:
                if needs_order and not generic:
                    s.pending_question = analysis["standalone_query"] or text
                return self._scripted(verify_msg, lang, user_text, t0, tool_calls, analysis, intro=not s.greeted)
            if needs_order and not s.verified:
                if s.locked:
                    return self._scripted("verification_failed", lang, user_text, t0, tool_calls, analysis, intro=not s.greeted)
                if not generic:
                    s.pending_question = analysis["standalone_query"] or text
                kind = "ask_verification" if s.awaiting_verification else "ask_order_id"
                return self._scripted(kind, lang, user_text, t0, tool_calls, analysis, intro=not s.greeted)
        if just_verified and analysis["intent"] == "verification_reply" and not s.pending_question:
            return self._scripted("verified_ok", lang, user_text, t0, tool_calls, analysis, intro=False)
        if just_verified:
            s.pending_question = None

        # ---------------------------------------------------------- entities & facts
        prods = self.store.find_products(question + " " + (analysis.get("standalone_query") or ""), analysis.get("products") or [])
        item_rows, order = [], s.order if s.verified else None
        if order:
            item_rows = F.match_order_items(order, prods, question)
            if not item_rows and not prods and s.last_skus:
                item_rows = [it for it in order["items"] if it["sku"] in s.last_skus]
            if not item_rows and len(order["items"]) == 1:
                item_rows = list(order["items"])
        if not prods and not item_rows and s.last_skus and analysis["intent"] in ("return_window", "return_eligibility", "warranty", "refund_timeline") and analysis.get("is_follow_up", True):
            prods = [self.store.product(k) for k in s.last_skus if self.store.product(k)]
        if item_rows:
            s.last_skus = list(dict.fromkeys(it["sku"] for it in item_rows))
        elif prods:
            s.last_skus = [p["sku"] for p in prods[:3]]
        s.last_intent = analysis["intent"]

        facts, force_ids = self._build_facts(analysis, order, item_rows, prods, question)

        # ---------------------------------------------------------- ESC-1 rules (code)
        reasons = self._escalation_reasons(flags, analysis, order, item_rows, prods, question, just_verified)
        ticket, actions = None, []
        if reasons:
            force_ids += ["ESC-1", "ESC-2", "ESC-3"]
            force_ids += {"high_value_damage": ["DWM-3"], "safety_risk": ["DWM-5"], "refund_overdue": ["REF-1", "REF-6"],
                          "human_requested": ["FAQ-4"]}.get(reasons[0][0], [])
            ticket = self._handover(reasons, analysis, question, order, tool_calls)
            if any(r[0] == "refund_overdue" for r in reasons):
                actions.append("The refund is overdue: tell the customer the refund reference number (RRN) from ORDER FACTS so their bank can trace it, and that a colleague will follow up. Do not promise a date.")
            if ticket and ticket.get("created"):
                actions.append(f"TICKET CREATED successfully (reason: {', '.join(r[0] for r in reasons)}). A human colleague will contact the customer. "
                               "Do NOT read out the ticket ID. Do not promise any outcome, amount or time.")
            else:
                actions.append("TICKET CREATION FAILED. Apologise, say you could not hand over just now and ask the customer to try again shortly. Do NOT say a ticket exists.")
        if flags["compensation"]:
            force_ids += ["DEL-4", "ESC-4"]
            actions.append("The customer is asking for compensation/voucher/credit. Politely refuse unless a policy in CONTEXT allows exactly that; "
                           "apologise briefly, state what policy says for their case, and offer a human colleague only if they would like one.")
        if analysis["intent"] == "privacy" or analysis.get("asks_other_customer_info"):
            force_ids += ["ESC-4", "IDV-4"]
            actions.append("The customer wants another person's details. Refuse; you never share another customer's details.")
        if analysis.get("missing_detail") and not reasons:
            actions.append(f"A needed detail is missing: {analysis['missing_detail']}. Ask exactly ONE short clarifying question and do not guess.")
        if analysis.get("covered_by_docs") is False and not needs_order and not reasons:
            actions.append("The knowledge base probably does NOT cover this. If the excerpts truly don't answer it, say you don't have that information and offer a human colleague.")
        if not s.greeted:
            actions.append("This is your FIRST reply: begin by introducing yourself as ZentroMart's AI support assistant (one short clause).")
        if just_verified:
            actions.append("The customer has just passed identity verification. Now answer their earlier request using ORDER FACTS.")

        # ---------------------------------------------------------- retrieval
        include_sup = any("delivered before 1 August 2026" in f for f in facts)
        q = analysis.get("standalone_query") or question
        extra = [p["title"] for p in prods[:2]] + ([it["title"] for it in item_rows[:2]])
        intent_key = "cod_refund" if analysis["payment_method"] == "cod" and analysis["intent"] in ("refund_timeline", "refund_status") else analysis["intent"]
        self._stage("retrieving")
        retrieved = self.kb.search(q, intent=intent_key, top_k=TOP_K, include_superseded=include_sup,
                                   force_ids=tuple(force_ids), extra_queries=[" ".join(extra)] if extra else [])
        if include_sup and not any(r["id"] == "RETv3-2" for r in retrieved):
            retrieved += [{"id": c.id, "score": 0.5, "doc": c.file, "status": c.status, "doc_type": c.doc_type, "text": c.render()}
                          for c in self.kb.chunks if c.id == "RETv3-2"]

        # ---------------------------------------------------------- respond
        self._stage("writing")
        reply_obj = self._generate(question, lang, retrieved, facts, actions, analysis, ticket)
        reply_obj.retrieved, reply_obj.tool_calls, reply_obj.analysis = retrieved, tool_calls, analysis
        reply_obj.ticket, reply_obj.escalation_reasons = ticket, [r[0] for r in reasons]

        # decision hygiene
        if ticket and ticket.get("created"):
            reply_obj.decision = "escalate"
        elif reasons:
            reply_obj.decision = "answer" if reply_obj.decision == "escalate" else reply_obj.decision
        elif flags["compensation"] or analysis["intent"] == "privacy" or analysis.get("asks_other_customer_info"):
            reply_obj.decision = "refuse"
        elif analysis.get("asks_cancel") and order and order.get("status") in SHIPPED_LIKE:
            reply_obj.decision = "refuse"
        elif reply_obj.decision == "escalate":
            reply_obj.decision = "answer"
        return self._finish(reply_obj, user_text, t0)

    # ================================================================ internals
    def _lookup(self, oid, value, log):
        res = self.lookup_order(oid, value) if value else self.lookup_order(oid)
        shown = f"lookup_order(order_id={oid}" + (f", verification_value={'*' * len(value)})" if value else ")")
        log.append(shown)
        return res

    @staticmethod
    def _default_analysis(text):
        return {"language": _detect_script_lang(text), "intent": "other", "standalone_query": text, "needs_order": False,
                "wants_human": False, "legal": False, "safety_risk": False, "unauthorized_activity": False,
                "asks_other_customer_info": False, "damage_reported": False, "compensation_request": False,
                "asks_cancel": False, "products": [], "payment_method": None, "refund_initiated_date": None,
                "delivered_date": None, "order_placed_date": None, "covered_by_docs": None, "missing_detail": None,
                "verification_value": None, "is_follow_up": True}

    def _analyze(self, text, alternatives=None, context_note=""):
        s = self.s
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in s.history[-6:]) or "(start of conversation)"
        state = (f"order_id_given={bool(s.order_id)}; verified={s.verified}; awaiting_verification={s.awaiting_verification}; "
                 f"known_products={s.last_skus}; last_intent={s.last_intent}; pending_request={s.pending_question!r}")
        user = f"CONVERSATION SO FAR:\n{hist}\n\nSESSION STATE: {state}\n{context_note}\n"
        if alternatives:
            user += "ASR HYPOTHESES for the same audio (different language modes; use the one that makes sense):\n" + "\n".join(f"- {a}" for a in alternatives) + "\n"
        user += f"\nCUSTOMER'S LATEST MESSAGE:\n{text}"
        out = self._default_analysis(text)
        try:
            am = os.getenv("LLM_ANALYZER_MODEL")       # optional: a faster model for this cheap extraction call
            data = self.llm.json(ANALYZER_SYSTEM.format(today=self.today.isoformat(), toc=self._toc), user, max_tokens=700,
                                 **({"model": am} if am else {}))
            for k, v in data.items():
                if k in out and v is not None:
                    out[k] = v
            out["products"] = [str(p) for p in (out["products"] or [])]
        except (LLMError, ValueError, KeyError) as e:
            out["_analyzer_error"] = str(e)[:120]
        if out["language"] not in {"en", "hi", "hi-en", "mr", "ta", "te", "kn", "ml", "gu", "bn", "pa"}:
            out["language"] = _detect_script_lang(text, "en")
        if out["language"] == "en" and _detect_script_lang(text, "") not in ("", "en") and not out.get("_analyzer_error") is None:
            out["language"] = _detect_script_lang(text, "en")
        return out

    @staticmethod
    def _needs_order(a):
        # Only an explicit need for THEIR order record triggers the ID check; policy questions never do.
        return bool(a["needs_order"]) or a["intent"] == "order_help"

    @staticmethod
    def _flags(question, a):
        human = bool(a["wants_human"] or RX_HUMAN.search(question))
        legal = bool(a["legal"] or RX_LEGAL.search(question))
        safety = bool(a["safety_risk"] or RX_SAFETY.search(question))
        unauth = bool(a["unauthorized_activity"])
        comp = bool(a["compensation_request"] or RX_COMP.search(question))
        return {"human": human, "legal": legal, "safety": safety, "unauth": unauth, "compensation": comp,
                "repeat_claim": bool(RX_REPEAT.search(question)),
                "damage": bool(a["damage_reported"] or RX_DAMAGE.search(question)),
                "escalate_now": human or legal or safety or unauth}

    # ------------------------------------------------------------------ facts
    def _build_facts(self, a, order, item_rows, prods, question):
        L, force = [f"TODAY: {D.fmt(self.today)} (business days = Monday to Saturday, excluding holidays: "
                    + ", ".join(D.fmt(h) for h in sorted(self.store.holidays)) + ")."], []
        hol = self.store.holidays
        ctx = {"delivered_date": a.get("delivered_date"), "order_placed_date": a.get("order_placed_date")}
        if order:
            L += F.order_lines(order, hol)
            if a["intent"] in ("return_window", "return_eligibility", "refund_timeline", "refund_status", "damage_report", "plus", "other") or item_rows:
                L += F.window_lines_for_order(order, item_rows or order["items"][:2], hol)
            if a["intent"] in ("damage_report",) or a["damage_reported"]:
                dd = D.parse_date(order.get("delivered_on"))
                if dd:
                    L.append(f"DAMAGE TIMING: delivered {D.fmt(dd)}; today is {D.fmt(self.today)} ({(self.today - dd).days} day(s) later). "
                             "The 48-hour reporting window (DWM-1) counts from delivery.")
        if prods and not item_rows:
            L += F.product_lines(prods, None, ctx)
        elif item_rows and prods:
            for p in prods[:2]:
                L.append("PRODUCT: " + self.store.describe_product(p))
        method = a.get("payment_method")
        init = D.parse_date(a.get("refund_initiated_date"))
        if method and (init or a["intent"] in ("refund_timeline", "cod_refund", "refund_status", "other")):
            if method == "cod":
                L += F.refund_eta_lines("cod_wallet", init, hol) + F.refund_eta_lines("cod_bank", init, hol)
            else:
                L += F.refund_eta_lines(method, init, hol)
            force += ["REF-2", "KB-00.3"]
        elif order and a["intent"] in ("refund_timeline", "refund_status") and not F.refunds_of(order, hol):
            m = order.get("return_refund_method") or ("cod_wallet" if order["payment_method"] == "cod" else order["payment_method"])
            if order["payment_method"] == "cod":
                L += F.refund_eta_lines("cod_wallet", None, hol) + F.refund_eta_lines("cod_bank", None, hol)
            else:
                L += F.refund_eta_lines(m, None, hol)
            if order.get("is_plus"):
                L.append("PLUS MEMBER: return refunds are initiated at pickup, without waiting for QC (PLS-2).")
            force += ["REF-2"]
        if prods and any(p["return_category"] == "electronics" for p in prods) and (D.parse_date(ctx["delivered_date"]) or (order and D.parse_date(order.get("delivered_on")))):
            force += ["RET-3"]
        pins = re.findall(r"(?<!\d)\d{6}(?!\d)", ascii_digits(question))
        if pins and a["intent"] in ("delivery_policy", "payment_cod", "other"):
            for pc in pins[:1]:
                row = self.store.pincode(pc)
                L.append(f"PINCODE {pc}: " + (f"{row['city']}, {row['state']}; serviceable={row['serviceable']}; COD={row['cod_available']}; Express={row['express_available']}." if row else "not in the serviceable pincode table (ZentroMart does not deliver there)."))
        return L, force

    # ------------------------------------------------------------------ escalation
    def _escalation_reasons(self, flags, a, order, item_rows, prods, question, just_verified):
        R = []
        if flags["human"]: R.append(("human_requested", "Customer asked for a human agent."))
        if flags["unauth"]: R.append(("unauthorized_activity", "Customer reports an order, payment or account activity they did not make."))
        if flags["legal"]: R.append(("legal_threat", "Customer mentions legal action, consumer court or a police complaint."))
        if flags["safety"]: R.append(("safety_risk", "Product safety risk (swelling battery, burning smell, sparks, injury)."))
        if order:
            order_q = a["needs_order"] or a["intent"] in ORDER_INTENTS or flags["repeat_claim"]
            if order_q and int(order.get("contacts_last_7_days") or 0) >= 2:
                R.append(("repeat_contact", f"Third or later contact about this issue within 7 days (earlier contacts: {order.get('contacts_last_7_days')})."))
            if a["intent"] in ("refund_status", "refund_timeline") or "refund" in (a.get("standalone_query") or "").lower():
                for r in F.refunds_of(order, self.store.holidays):
                    if r["overdue"].get("overdue"):
                        R.append(("refund_overdue", f"{r['kind']} refund initiated {D.fmt(r['initiated'])} is {r['overdue']['late_business_days']} business days past its maximum timeline (RRN {r['rrn']})."))
                        break
        if flags["damage"] or a["intent"] == "damage_report":
            cands = []
            if order:
                cands = [it for it in item_rows] or ([it for it in order["items"] if it["sku"] == order.get("damage_sku")] if order.get("damage_sku") else [])
                prices = [int(it["unit_price_inr"]) for it in cands]
            else:
                prices = [int(p["price_inr"]) for p in prods]
            if prices and max(prices) > 10000:
                R.append(("high_value_damage", f"Damaged/defective/wrong item claim for an item priced above Rs 10,000 (Rs {max(prices)}); Priority Resolution must decide (DWM-3)."))
        return R

    def _handover(self, reasons, a, question, order, log):
        s = self.s
        codes = tuple(r[0] for r in reasons)
        for t in s.tickets:  # do not spam duplicates for the same reason set in one session
            if set(codes) <= set(t["reasons"]):
                return {"created": True, "ticket_id": t["ticket_id"], "duplicate": True}
        redact = self._redact
        earlier = " | ".join(m["content"] for m in s.history if m["role"] == "user")[-300:] or "nothing earlier"
        checks = []
        if s.order_id:
            checks.append(f"order {s.order_id} looked up; identity {'verified' if s.verified else 'NOT verified'}")
            if order:
                checks.append(f"order status {order['status']}, payment {order['payment_method']}, earlier contacts {order.get('contacts_last_7_days')}")
        else:
            checks.append("no order ID provided yet")
        summary = (f"Customer issue: {redact(question)[:300]}. Detected intent: {a['intent']}. Order ID: {s.order_id or 'none'}. "
                   f"Customer already told us: {earlier}. We already checked: {'; '.join(checks)}. "
                   f"Reason for escalation: {' '.join(r[1] for r in reasons)}")
        args = {"summary": summary, "reason": ",".join(codes), "intent": a["intent"]}
        if s.order_id:
            args["order_id"] = s.order_id
        res = self.create_ticket(**args)
        log.append(f"create_ticket(order_id={s.order_id}, reason={','.join(codes)})")
        if res.get("created"):
            s.tickets.append({"reasons": list(codes), "ticket_id": res["ticket_id"]})
        return res

    def _redact(self, text):
        """Remove verification values the customer just spoke (never log/echo them)."""
        t = ascii_digits(text or "")
        for sec in getattr(self, "_secrets", ()):
            t = re.sub(r"(?<!\d)" + r"[\s-]?".join(sec) + r"(?!\d)", "[verification value removed]", t)
        return t

    # ------------------------------------------------------------------ replies
    def _scripted(self, kind, lang, user_text, t0, tool_calls, analysis, intro=False):
        sections = {"ask_order_id": ["STY-1", "IDV-2"], "ask_verification": ["IDV-1", "IDV-2"], "retry_verification": ["IDV-3"],
                    "verification_failed": ["IDV-3"], "verified_ok": ["IDV-2"], "order_id_unclear": ["IDV-2"],
                    "greeting": ["STY-1"]}[kind]
        body = TPL.get(kind, lang)
        if body is None:  # language without a template -> LLM phrases a fixed directive
            body = self._llm_say(kind, lang)
        if intro and not TPL.AI_PATTERN.search(body):
            lead = TPL.intro(lang) or self._llm_say("intro", lang) or TPL.intro("en")
            body = f"{lead} {body}"
        retrieved = [{"id": i, "score": 1.0, "doc": "", "status": "current", "doc_type": "agent_guide", "text": self.kb.by_id[i].render()}
                     for i in dict.fromkeys((["STY-1"] if intro else []) + sections) if i in self.kb.by_id]
        r = Reply(body, "", lang, "refuse" if kind == "verification_failed" else ("answer" if kind == "verified_ok" else "clarify"),
                  retrieved=retrieved, used_sections=sections, tool_calls=tool_calls, analysis=analysis)
        return self._finish(r, user_text, t0)

    def _llm_say(self, kind, lang):
        directive = {
            "intro": "Say only: 'Hi, I'm ZentroMart's AI support assistant.'",
            "greeting": "Ask how you can help the customer today.",
            "ask_order_id": "Ask the customer for their order ID.",
            "order_id_unclear": "Say you didn't catch the full 8-digit order ID and ask them to say it again.",
            "ask_verification": "Thank them and ask for the last four digits of their registered phone number or the delivery pincode, to keep their order details safe.",
            "retry_verification": "Say the details don't match our records and ask them to check and try once more (last four digits of registered phone, or delivery pincode).",
            "verification_failed": "Say you could not verify them so you cannot share order details; suggest contacting support from the registered phone number; offer help with general policy questions.",
            "verified_ok": "Say thank you, they're verified, and ask how you can help with this order.",
            "escalated": "Apologise, say a ticket was created and a human colleague will contact them shortly, with a short summary passed on.",
        }[kind]
        try:
            out = self.llm.complete(f"You write ONE short spoken sentence or two for a customer support call in the language with code '{lang}' (use its own script). No markdown.",
                                    directive, max_tokens=120)
            return out.strip()
        except LLMError:
            return TPL.get(kind if kind != "intro" else "ask_order_id", "en") if kind != "intro" else TPL.intro("en")

    def _generate(self, question, lang, retrieved, facts, actions, analysis, ticket) -> Reply:
        s = self.s
        ctx = "\n\n".join(r["text"] for r in retrieved)
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in s.history[-8:]) or "(first message)"
        user = (f"CUSTOMER LANGUAGE CODE: {lang}\n\nACTIONS / SITUATION:\n" + ("\n".join(f"- {a}" for a in actions) or "- none") +
                f"\n\nCONTEXT (policy excerpts):\n{ctx}\n\nCOMPUTED FACTS AND ORDER FACTS:\n" + "\n".join(f"- {f}" for f in facts) +
                f"\n\nCONVERSATION SO FAR:\n{hist}\n\nCUSTOMER NOW SAYS (resolve follow-ups using the conversation): {question}\n\nReturn the JSON object.")
        data, bad, reply = None, None, ""
        for attempt in range(2):
            try:
                warn = f"\n\nYour previous draft was rejected because {bad}. Rewrite it without that." if bad else ""
                data = self.llm.json(RESPONDER_SYSTEM, user + warn, max_tokens=500)
                reply = str(data.get("reply", "")).strip()
            except LLMUnavailable:
                data = None
                break
            except LLMError:      # malformed JSON: try once more
                data = None
                continue
            bad = self._violation(reply, ticket)
            if reply and not bad:
                break
        if not data or not reply or bad:
            if ticket and ticket.get("created"):
                text = TPL.get("escalated", lang) or self._llm_say("escalated", lang)
                return Reply(text, "", lang, "escalate", used_sections=["ESC-1", "ESC-2"])
            if ticket is not None:
                text = TPL.get("escalation_failed", lang) or TPL.get("escalation_failed", "en")
                return Reply(text, "", lang, "clarify")
            text = TPL.get("llm_down", lang) or TPL.get("llm_down", "en")
            return Reply(text, "", lang, "clarify")
        reply = str(data["reply"]).strip()
        if not s.greeted and not TPL.AI_PATTERN.search(reply):
            lead = TPL.intro(lang) or TPL.intro("en")
            reply = f"{lead} {reply}"
        used = [x for x in (data.get("used_sections") or []) if isinstance(x, str)]
        used = [u.strip("[] ") for u in used if u.strip("[] ") in self.kb.by_id]
        dec = data.get("decision") if data.get("decision") in ("answer", "clarify", "refuse", "escalate") else "answer"
        return Reply(reply, "", lang, dec, used_sections=used)

    def _violation(self, reply, ticket):
        for rx in BAD_ACTION:
            if rx.search(reply):
                return "it claims an action (cancellation/approval/credit) that your tools did not perform"
        if RX_TICKET_CLAIM.search(reply) and not (ticket and ticket.get("created")) and not self.s.tickets:
            return "it claims a ticket exists but no ticket was created"
        return None

    # ------------------------------------------------------------------ finish
    def _finish(self, r: Reply, user_text, t0, record=True) -> Reply:
        s = self.s
        r.language = r.language or s.language
        speak_src = re.sub(r"\(Ticket:[^)]*\)", "", r.text)
        r.spoken = speakable(speak_src, r.language)
        if r.ticket and r.ticket.get("created") and not r.ticket.get("duplicate"):
            r.text = f"{r.text} (Ticket: {r.ticket['ticket_id']})"
        r.latency_s = round(time.time() - t0, 2)
        if record:
            s.history.append({"role": "user", "content": self._redact(user_text)})
            s.history.append({"role": "assistant", "content": re.sub(r"\(Ticket:[^)]*\)", "", r.text).strip()})
            s.greeted = True
        return r
