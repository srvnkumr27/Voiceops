"""Text normalisation: ASR clean-up (spelled-out digits) and speech-friendly rewriting."""
import re
import unicodedata
from datetime import date

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

_UNAMBIG = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9",
            # romanised Hindi
            "shunya": "0", "ek": "1", "teen": "3", "char": "4", "paanch": "5", "panch": "5",
            "chhe": "6", "chheh": "6", "saat": "7", "aath": "8", "nau": "9"}
_AMBIG = {"to": "2", "too": "2", "for": "4", "do": "2"}
_MULT = {"double": 2, "triple": 3}


def ascii_digits(s: str) -> str:
    """Convert Devanagari / Bengali / Tamil ... digits to ASCII."""
    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() and not ch.isascii() else ch for ch in s)


def asr_to_digits(text: str) -> str:
    """'eight nine seven eight two eight three seven' -> '89782837'; 'ele zero zero one to' -> 'ele 0012'.
    Single number words ('one coupon') are left alone: a run needs >=2 unambiguous digit words."""
    toks = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]|\s+", text)
    out, i = [], 0

    def kind(t):
        tl = t.lower()
        if tl in _UNAMBIG: return "u"
        if tl in _AMBIG: return "a"
        if tl in _MULT: return "m"
        return None

    while i < len(toks):
        if kind(toks[i]) is None:
            out.append(toks[i]); i += 1; continue
        j, run = i, []
        while j < len(toks):
            if toks[j].isspace() or toks[j] == "-":
                # whitespace/hyphen allowed inside a run only if next non-space token continues it
                k = j
                while k < len(toks) and (toks[k].isspace() or toks[k] == "-"):
                    k += 1
                if k < len(toks) and (kind(toks[k]) or toks[k].isdigit() and len(toks[k]) == 1) and run:
                    j = k; continue
                break
            if kind(toks[j]) or (toks[j].isdigit() and len(toks[j]) == 1 and run):
                run.append(toks[j]); j += 1
            else:
                break
        unamb = sum(1 for t in run if kind(t) == "u" or t.isdigit())
        if unamb >= 2:
            while run and kind(run[0]) == "a":
                out.append(run.pop(0)); out.append(" ")
            digits, m = "", 1
            for t in run:
                tl = t.lower()
                if tl in _MULT: m = _MULT[tl]; continue
                d = t if t.isdigit() else (_UNAMBIG.get(tl) or _AMBIG.get(tl))
                digits += d * m; m = 1
            out.append(digits)
            i = j
        else:
            out.append(toks[i]); i += 1
    return "".join(out)


def clean_asr(text: str, sku_prefixes=()) -> str:
    """Normalise a speech transcript before the agent reads it."""
    t = ascii_digits(text or "")
    t = asr_to_digits(t)
    # 'ele-0012' / 'ele 0012' -> 'ELE-0012'
    if sku_prefixes:
        pat = r"\b(" + "|".join(sku_prefixes) + r")[\s-]*(\d{4})\b"
        t = re.sub(pat, lambda m: f"{m.group(1).upper()}-{m.group(2)}", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def extract_order_id(text: str):
    """8 digits, possibly written as 'xxxx xxxx' or with hyphens/spaces. Returns (id|None, text_without_it)."""
    t = ascii_digits(text)
    m = re.search(r"(?<![\d-])(\d)[\s-]?(\d)[\s-]?(\d)[\s-]?(\d)[\s-]?(\d)[\s-]?(\d)[\s-]?(\d)[\s-]?(\d)(?![\d])", t)
    if not m:
        return None, t
    return "".join(m.groups()), (t[:m.start()] + " " + t[m.end():])


_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + MONTH_RE + r"\w*\.?,?\s*(?:\d{4})?",
    r"\b" + MONTH_RE + r"\w*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*(?:\d{4})?",
]


def strip_dates(text: str) -> str:
    for p in _DATE_PATTERNS:
        text = re.sub(p, " ", text, flags=re.I)
    return text


def extract_verification(text: str):
    """Return a 4-digit phone suffix or 6-digit pincode from a short reply, else None.
    Dates (e.g. '17 October 2026') are removed first so years are never mistaken for a code."""
    t = strip_dates(ascii_digits(text))
    t = re.sub(r"(?<!\d)((?:\d[\s-]){3,5}\d)(?!\d)", lambda m: re.sub(r"[\s-]", "", m.group(1)), t)
    sixes = re.findall(r"(?<!\d)\d{6}(?!\d)", t)
    if sixes:
        return sixes[-1]
    fours = re.findall(r"(?<!\d)\d{4}(?!\d)", t)
    return fours[-1] if fours else None


def digit_run_lengths(text: str):
    t = ascii_digits(text)
    t = re.sub(r"(?<=\d)[\s-](?=\d)", "", t)
    return [len(x) for x in re.findall(r"\d+", t)]


# ---------------------------------------------------------------- speech-friendly rewriting
_ORD = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
_RUPEE_UNIT = {"en": "rupees", "hi-en": "rupees", "hi": "रुपये"}


def speakable(text: str, lang: str = "en") -> str:
    """Write for the ear (Gnani text-normalisation guide + STY-3): amounts in rupees, order IDs in
    groups of four digits, dates with month name and year, no markdown."""
    t = re.sub(r"[*_`#>]+", "", text)
    t = re.sub(r"\(\s*ticket[^)]*\)", "", t, flags=re.I)
    unit = _RUPEE_UNIT.get(lang, "rupees")
    t = re.sub(r"(?:₹|Rs\.?\s*|INR\s*)\s*([\d,]+(?:\.\d+)?)", lambda m: f"{m.group(1).replace(',', '')} {unit}", t)
    t = re.sub(r"(?<!\d)(\d{4})(\d{4})(?!\d)", r"\1 \2", t)  # 8-digit order IDs -> '8978 2837'

    def iso(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            date(y, mo, d)
        except ValueError:
            return m.group(0)
        name = MONTHS[mo - 1].capitalize()
        return f"{_ORD(d)} of {name} {y}" if lang in ("en", "hi-en") else f"{d} {name} {y}"
    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", iso, t)
    if lang in ("en", "hi-en"):
        t = re.sub(r"\b(\d{1,2}) (" + "|".join(m.capitalize() for m in MONTHS) + r") (\d{4})\b",
                   lambda m: f"{_ORD(int(m.group(1)))} of {m.group(2)} {m.group(3)}", t)
    return re.sub(r"\s+", " ", t).strip()
