"""Offline stand-in for the LLM, used ONLY to test plumbing (state machine, tools, facts, retrieval).
Analyzer: tiny keyword heuristics. Responder: echoes its own context, so 'must_include' passing here
means 'the right facts reached the model' (context coverage) - NOT answer quality."""
import json
import re

from voiceops.agent import _detect_script_lang


class FakeLLM:
    provider, model, calls = "fake", "fake", 0

    def complete(self, system, user, **kw):
        self.calls += 1
        if system.startswith("You write ONE short"):
            return "Hello."
        return json.dumps(self._analyze(user) if "understanding module" in system else self._respond(user), ensure_ascii=False)

    def json(self, system, user, max_tokens=0, **kw):
        return json.loads(self.complete(system, user))

    # ------------------------------------------------------------------
    def _analyze(self, user):
        msg = user.split("CUSTOMER'S LATEST MESSAGE:\n", 1)[1].strip()
        low = msg.lower()
        a = {"language": _detect_script_lang(msg), "intent": "other", "standalone_query": msg, "needs_order": False,
             "products": [], "payment_method": None, "covered_by_docs": True}
        has = lambda *w: any(x in low for x in w)
        if has("human", "person", "insaan", "इंसान"): a.update(intent="human_request", wants_human=True)
        elif has("third time", "teesri baar", "तीसरी"): a.update(intent="other", needs_order=True)
        elif has("ship hua", "when was it shipped", "when was my order", "hasn't arrived", "where is it"): a.update(intent="delivery_status", needs_order=True)
        elif has("refund") and has("kahan", "where", "कहाँ"): a.update(intent="refund_status", needs_order=True)
        elif has("voucher", "compensation", "मुआवज"): a.update(intent="compensation", compensation_request=True)
        elif has("cancel", "कैंसिल", "stop it"): a.update(intent="cancel", asks_cancel=True, needs_order=not has("on its way", "shipped", "ship हो", "ship ho"))
        elif has("coupon", "कूपन"): a.update(intent="coupon")
        elif has("damage", "टूटा", "crack", "torn"): a.update(intent="damage_report", damage_reported=True)
        elif has("express") and has("late"): a.update(intent="late_delivery", needs_order=True)
        elif has("not at home", "not home", "not at "): a.update(intent="delivery_policy")
        elif has("refund") and has("kitne din", "how long", "kab", "when", "कितने दिन", "कब", "how soon", "by which date", "how fast"): a.update(intent="refund_timeline")
        elif has("return", "लौटा", "वापस", "send it back", "send back"): a.update(intent="return_eligibility")
        elif has("never placed", "did not place", "didn't place"): a.update(intent="unauthorized_activity", unauthorized_activity=True)
        elif has("order ke baare", "help with an order", "ऑर्डर के बारे में", "help with order"): a.update(intent="order_help", needs_order=True)
        elif re.fullmatch(r"[\d\s.,।a-zA-Z\u0900-\u097F]*", msg) and re.search(r"\d{4}", msg) and len(msg) < 60: a.update(intent="verification_reply")
        for k, v in (("credit card", "card"), ("upi", "upi"), ("net banking", "netbanking"), ("wallet", "wallet"), ("cash on delivery", "cod")):
            if k in low: a["payment_method"] = v
        m = re.search(r"(\d{1,2}) ([a-z]+) (\d{4})", low)
        if m and "refund" in low:
            from voiceops.textnorm import MONTHS
            mo = [i for i, x in enumerate(MONTHS) if x.startswith(m.group(2)[:3])]
            if mo: a["refund_initiated_date"] = f"{m.group(3)}-{mo[0]+1:02d}-{int(m.group(1)):02d}"
        return a

    def _respond(self, user):
        actions = user.split("ACTIONS / SITUATION:", 1)[1].split("CONTEXT (policy excerpts):", 1)[0]
        ctx = user.split("CONTEXT (policy excerpts):", 1)[1].split("CONVERSATION SO FAR:", 1)[0]
        dec = "answer"
        if "TICKET CREATED" in actions: dec = "escalate"
        elif "compensation" in actions: dec = "refuse"
        ids = re.findall(r"^\[([A-Za-z0-9.\-]+)\]", ctx, re.M)
        text = "Hi, I'm ZentroMart's AI assistant. sorry. ticket colleague team. " + ctx
        return {"reply": text, "decision": dec, "used_sections": ids[:3]}
