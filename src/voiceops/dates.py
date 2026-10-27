"""Deterministic date logic. The LLM never does calendar arithmetic."""
from datetime import date, datetime, timedelta
from typing import Optional

from .config import TODAY

# (min, max) business days after the refund is initiated (REF-2)
REFUND_BD = {
    "upi": (1, 3),
    "card": (5, 7),
    "netbanking": (3, 5),
    "wallet": (0, 0),       # instant, within 2 hours
    "cod_wallet": (0, 0),
    "cod_bank": (5, 7),
}
METHOD_ALIASES = {
    "credit card": "card", "debit card": "card", "credit_card": "card", "debit_card": "card",
    "net banking": "netbanking", "net_banking": "netbanking", "internet banking": "netbanking",
    "bank transfer": "cod_bank", "neft": "cod_bank", "cod bank": "cod_bank",
}
V4_EFFECTIVE = date(2026, 8, 1)
NOT_RETURNABLE = {"grocery", "innerwear", "digital"}
BASE_WINDOW = {"fashion": 15, "home_kitchen": 10, "beauty_sealed": 7, "books_media": 10}


def parse_date(s) -> Optional[date]:
    if not s:
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    try:
        return datetime.strptime(str(s).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def is_business_day(d: date, holidays: set) -> bool:
    return d.weekday() != 6 and d not in holidays  # Mon-Sat, minus holidays


def add_business_days(start: date, n: int, holidays: set) -> date:
    """The initiation day itself is day 0; day 1 is the next business day."""
    d, count = start, 0
    while count < n:
        d += timedelta(days=1)
        if is_business_day(d, holidays):
            count += 1
    return d


def business_days_between(start: date, end: date, holidays: set) -> int:
    """Business days elapsed in (start, end]."""
    if end <= start:
        return 0
    d, c = start, 0
    while d < end:
        d += timedelta(days=1)
        if is_business_day(d, holidays):
            c += 1
    return c


def norm_method(m) -> Optional[str]:
    if not m:
        return None
    m = str(m).strip().lower()
    m = METHOD_ALIASES.get(m, m)
    return m if m in REFUND_BD else None


def refund_eta(method, initiated: date, holidays: set):
    """Returns dict(earliest, latest, instant, method) or None."""
    m = norm_method(method)
    if not m or not initiated:
        return None
    lo, hi = REFUND_BD[m]
    return {"method": m, "instant": hi == 0, "earliest": add_business_days(initiated, lo, holidays),
            "latest": add_business_days(initiated, hi, holidays), "min_bd": lo, "max_bd": hi}


def refund_overdue(method, initiated: date, holidays: set, today: date = TODAY) -> dict:
    """ESC-1.5: escalate when more than 3 business days past the maximum timeline."""
    eta = refund_eta(method, initiated, holidays)
    if not eta:
        return {"overdue": False}
    late_bd = business_days_between(eta["latest"], today, holidays)
    return {"overdue": late_bd > 3, "late_business_days": late_bd, "latest": eta["latest"]}


def return_window(category: str, delivered_on: Optional[date], placed_on: Optional[date]) -> dict:
    """RET-2, RET-3 (electronics delivered before 1 Aug 2026 keep 10 days), RET-8 (festive fashion)."""
    if category in NOT_RETURNABLE:
        return {"returnable": False, "days": None, "rule": "RET-2/RET-5"}
    if category == "electronics":
        if delivered_on and delivered_on < V4_EFFECTIVE:
            return {"returnable": True, "days": 10, "rule": "RET-3 / RETv3-2 (delivered before 1 August 2026)"}
        return {"returnable": True, "days": 7, "rule": "RET-2 / RET-3"}
    if category == "fashion":
        if placed_on and date(2026, 10, 1) <= placed_on <= date(2026, 10, 15):
            return {"returnable": True, "days": 30, "rule": "RET-8 (festive sale, placed 1-15 October 2026)"}
        return {"returnable": True, "days": 15, "rule": "RET-2"}
    return {"returnable": True, "days": BASE_WINDOW.get(category, 10), "rule": "RET-2"}


def fmt(d: Optional[date]) -> str:
    """'27 October 2026' (no zero padding)."""
    return f"{d.day} {d.strftime('%B %Y')}" if d else ""
