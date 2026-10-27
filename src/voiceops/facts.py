"""Turn a *verified* order + catalog rows into explicit, pre-computed facts for the responder.
All arithmetic (business days, return deadlines, refund ETAs, lateness) happens here, in code."""
from datetime import date, timedelta
from typing import Optional

from . import dates as D
from .config import TODAY
from .datastore import get_store


def _i(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def refunds_of(order: dict, hol: set):
    """All refund records on an order: cancellation / rto / return."""
    out = []
    for kind, pre in (("cancellation", "cancellation_refund"), ("rto", "rto_refund"), ("return", "return_refund")):
        init = D.parse_date(order.get(f"{pre}_initiated_on"))
        if not init:
            continue
        method = order.get(f"{pre}_method")
        eta = D.refund_eta(method, init, hol)
        completed = D.parse_date(order.get("return_refund_completed_on")) if kind == "return" else None
        od = D.refund_overdue(method, init, hol) if not completed else {"overdue": False}
        out.append({"kind": kind, "method": method, "initiated": init,
                    "amount": order.get(f"{pre}_amount_inr"),
                    "rrn": order.get({"cancellation": "cancellation_rrn", "rto": "rto_rrn", "return": "return_rrn"}[kind]),
                    "eta": eta, "completed": completed, "overdue": od})
    return out


def item_window(item_or_prod: dict, order: Optional[dict]):
    cat = item_or_prod.get("return_category")
    delivered = D.parse_date(order.get("delivered_on")) if order else None
    placed = D.parse_date(order.get("placed_at")) if order else None
    w = D.return_window(cat, delivered, placed)
    if w["days"] and delivered:
        w["deadline"] = delivered + timedelta(days=w["days"])
        w["open"] = TODAY <= w["deadline"]
    return w


def product_lines(prods, order=None, ctx: dict | None = None):
    """Catalog + policy facts for each product mentioned. ctx may carry user-stated dates."""
    st = get_store()
    ctx = ctx or {}
    lines = []
    for p in prods[:4]:
        lines.append("PRODUCT: " + st.describe_product(p))
        delivered = D.parse_date(order.get("delivered_on")) if order else D.parse_date(ctx.get("delivered_date"))
        placed = D.parse_date(order.get("placed_at")) if order else D.parse_date(ctx.get("order_placed_date"))
        w = D.return_window(p["return_category"], delivered, placed)
        if not w["returnable"]:
            lines.append(f"RETURN WINDOW for {p['title']}: NOT returnable for change of mind ({w['rule']}); "
                         "if damaged/expired/wrong/missing, report within 48 hours of delivery with photos (DWM-1).")
        else:
            s = f"RETURN WINDOW for {p['title']}: {w['days']} calendar days from delivery [{w['rule']}]."
            if delivered:
                dl = delivered + timedelta(days=w["days"])
                s += f" Delivered {D.fmt(delivered)}, so the last return day is {D.fmt(dl)}" \
                     f" ({'still open' if TODAY <= dl else 'already closed'} as of today, {D.fmt(TODAY)})."
            elif p["return_category"] == "electronics":
                s += " (If it was delivered before 1 August 2026 the window is 10 days instead.)"
            if p["return_category"] == "fashion" and not (placed and date(2026, 10, 1) <= placed <= date(2026, 10, 15)):
                s += " (Orders placed 1-15 October 2026 get 30 days: RET-8.)"
            lines.append(s)
        if p["return_category"] == "fashion":
            lines.append("REVERSE PICKUP FEE: Rs 49 on change-of-mind returns of clothing/footwear/bags; waived for "
                         "ZentroMart Plus members and for damaged/defective/wrong items (RET-6).")
        if p["return_category"] == "beauty_sealed":
            lines.append("CONDITION: beauty/personal-care items are returnable only if unopened with the seal intact (RET-4).")
        lines.append(f"HIGH-VALUE FLAG: item price Rs {p['price_inr']} -> "
                     f"{'ABOVE Rs 10,000: a damage/defect/wrong claim must go to a human (DWM-3)' if _i(p['price_inr']) > 10000 else 'not above Rs 10,000'}.")
    return lines


def refund_eta_lines(method, initiated, hol):
    """General refund timeline facts for a method (and an exact date range if initiation date is known)."""
    m = D.norm_method(method)
    if not m:
        return []
    lo, hi = D.REFUND_BD[m]
    label = {"upi": "UPI 1 to 3 business days", "card": "credit/debit card 5 to 7 business days",
             "netbanking": "net banking 3 to 5 business days", "wallet": "ZentroMart Wallet instant, within 2 hours",
             "cod_wallet": "ZentroMart Wallet (COD refund) instant, within 2 hours",
             "cod_bank": "bank transfer (NEFT, COD order) 5 to 7 business days after bank details are verified"}[m]
    lines = [f"REFUND TIMELINE ({m}): {label}, counted from the day the refund is initiated; business days = Mon-Sat excluding holidays."]
    if initiated:
        eta = D.refund_eta(m, initiated, hol)
        if eta["instant"]:
            lines.append(f"COMPUTED: refund initiated {D.fmt(initiated)} to Wallet -> instant (within 2 hours).")
        else:
            lines.append(f"COMPUTED: refund initiated {D.fmt(initiated)} via {m}: expected between {D.fmt(eta['earliest'])} "
                         f"and {D.fmt(eta['latest'])}; it should reach the customer by {D.fmt(eta['latest'])} at the latest.")
    return lines


def order_lines(order: dict, hol: set):
    """Facts about the verified order. Never includes phone/email/address."""
    L = []
    items = order.get("items", [])
    L.append(f"ORDER {order['order_id']}: status={order['status']}; payment={order['payment_method']}; "
             f"delivery_type={order['delivery_type']}; total=Rs {order['total_inr']}; customer first name {order.get('customer_first_name')}; "
             f"ZentroMart Plus member={'YES' if order.get('is_plus') else 'NO'}.")
    L.append(f"ORDER DATES: placed {D.fmt(D.parse_date(order.get('placed_at')))}; promised {D.fmt(D.parse_date(order.get('promised_date')))}; "
             f"shipped {D.fmt(D.parse_date(order.get('shipped_on'))) or 'not yet'}; delivered {D.fmt(D.parse_date(order.get('delivered_on'))) or 'not yet'}; "
             f"delivery attempts {order.get('delivery_attempts')}.")
    for it in items:
        L.append(f"ITEM line {it['line_no']}: {it['title']} ({it['sku']}), qty {it['qty']}, Rs {it['unit_price_inr']} each, category {it['return_category']}.")
    promised, shipped, delivered = (D.parse_date(order.get(k)) for k in ("promised_date", "shipped_on", "delivered_on"))
    if delivered and promised and delivered > promised:
        L.append(f"LATENESS: delivered {(delivered - promised).days} day(s) after the promised date. "
                 + ("Delivery type is EXPRESS: Rs 79 express fee is credited to Wallet within 48 hours (DEL-4)."
                    if order["delivery_type"] == "express" else
                    "Standard/slot delivery: NO compensation for lateness (DEL-4)."))
    elif not delivered and promised:
        d = (promised - TODAY).days
        L.append(f"DELIVERY STATUS: not delivered yet; promised {D.fmt(promised)} ({d} day(s) from today; "
                 f"{'on track' if d >= 0 else f'{-d} day(s) past the promised date'}).")
    if order.get("express_late_credit_inr"):
        L.append(f"EXPRESS LATE CREDIT: Rs {order['express_late_credit_inr']} credited to Wallet on {D.fmt(D.parse_date(order.get('express_late_credit_on')))}.")
    for r in refunds_of(order, hol):
        s = (f"REFUND ({r['kind']}): method={r['method']}; initiated {D.fmt(r['initiated'])}; amount Rs {r['amount']}; RRN {r['rrn']}.")
        if r["eta"]:
            s += (" Instant (Wallet)." if r["eta"]["instant"] else
                  f" Expected by {D.fmt(r['eta']['latest'])} at the latest (earliest {D.fmt(r['eta']['earliest'])}).")
        if r["completed"]:
            s += f" Completed {D.fmt(r['completed'])}."
        if r["overdue"].get("overdue"):
            s += (f" OVERDUE: {r['overdue']['late_business_days']} business days past the maximum timeline (more than 3) -> "
                  "ESC-1 requires handover to a human; share the RRN so the bank can trace it (REF-6).")
        L.append(s)
    if order.get("return_sku"):
        L.append(f"RETURN: item {order['return_sku']} qty {order.get('return_qty')} reason={order.get('return_reason')}; status={order.get('return_status')}; "
                 f"requested {D.fmt(D.parse_date(order.get('return_requested_on'))) or '-'}; picked up {D.fmt(D.parse_date(order.get('return_pickup_on'))) or 'not yet'}; "
                 f"QC passed {D.fmt(D.parse_date(order.get('return_qc_passed_on'))) or 'not yet'}; refund method {order.get('return_refund_method')}; "
                 f"pickup fee Rs {order.get('return_reverse_pickup_fee_inr') or 0}.")
        if order.get("return_pickup_on") and not order.get("return_refund_initiated_on"):
            pk = D.parse_date(order["return_pickup_on"])
            if order.get("is_plus"):
                L.append("RETURN REFUND: Plus member -> refund is initiated at pickup (RET-7/PLS-2).")
            else:
                L.append(f"RETURN REFUND: picked up {D.fmt(pk)}; item must pass quality check (QC); refund is initiated within 2 business days of pickup "
                         f"(by {D.fmt(D.add_business_days(pk, 2, hol))}) once QC passes; then the method's timeline applies (RET-7, REF-1, REF-2).")
    if order.get("damage_sku"):
        L.append(f"DAMAGE REPORT on file: sku {order['damage_sku']}, issue {order.get('damage_issue')}, reported {order.get('damage_hours_after_delivery')} hours after delivery, "
                 f"photos attached={order.get('damage_photos_attached')}, status={order.get('damage_status')}.")
    L.append(f"EARLIER CONTACTS about this order in last 7 days (excluding this one): {order.get('contacts_last_7_days')}.")
    if order["payment_method"] == "cod":
        L.append("PAYMENT NOTE: COD order - refunds go to ZentroMart Wallet (instant) or bank transfer (5 to 7 business days after bank verification) (REF-3).")
    return L


def match_order_items(order: dict, prods, text: str):
    """Which order line(s) is the customer talking about?"""
    items = order.get("items", [])
    skus = {p["sku"] for p in prods}
    hit = [it for it in items if it["sku"] in skus]
    if hit:
        return hit
    low = (text or "").lower()
    hit = [it for it in items if it["title"].split(",")[0].lower() in low]
    return hit


def window_lines_for_order(order, items, hol):
    L = []
    for it in items[:3]:
        w = item_window(it, order)
        t = it["title"]
        if not w["returnable"]:
            L.append(f"RETURN WINDOW for {t}: NOT returnable for change of mind ({w['rule']}).")
        elif "deadline" in w:
            L.append(f"RETURN WINDOW for {t}: {w['days']} days [{w['rule']}]; delivered {D.fmt(D.parse_date(order['delivered_on']))}; "
                     f"last return day {D.fmt(w['deadline'])} ({'still open' if w['open'] else 'CLOSED'} as of today {D.fmt(TODAY)}).")
        else:
            L.append(f"RETURN WINDOW for {t}: {w['days']} days from delivery [{w['rule']}] (not delivered yet, so the clock has not started).")
        if it["return_category"] == "fashion":
            fee = "waived (Plus member)" if order.get("is_plus") else "Rs 49 for change-of-mind (waived if damaged/defective/wrong)"
            L.append(f"REVERSE PICKUP FEE for {t}: {fee} (RET-6).")
    return L
