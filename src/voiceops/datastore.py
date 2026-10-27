"""Read-only reference data: products, pincodes, holidays (orders go through the verified tool only)."""
import csv
import re
from datetime import date
from functools import lru_cache

from .config import dataset_dir
from .dates import parse_date


def _read(name):
    with open(dataset_dir() / "data" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


class Store:
    def __init__(self):
        self.products = {r["sku"]: r for r in _read("products.csv")}
        self.pincodes = {r["pincode"]: r for r in _read("pincodes.csv")}
        self.holidays = {parse_date(r["date"]) for r in _read("holidays.csv")}
        self.sku_prefixes = sorted({s.split("-")[0].lower() for s in self.products})
        self._base = {}
        for sku, p in self.products.items():
            self._base.setdefault(_norm(p["title"].split(",")[0]), []).append(sku)
        self._full = {_norm(p["title"]): sku for sku, p in self.products.items()}

    # ------------------------------------------------------------------ products
    def product(self, sku):
        return self.products.get(sku)

    def find_products(self, text: str, extra_names=()):
        """Find catalog products mentioned in free text (SKU, full title, or title without colour).
        Returns a list of products (variants of one base title are all returned)."""
        found, seen = [], set()

        def add(sku):
            if sku not in seen:
                seen.add(sku); found.append(self.products[sku])

        for src in [text, *extra_names]:
            if not src:
                continue
            for m in re.finditer(r"\b([A-Za-z]{3})-?(\d{4})\b", src):
                sku = f"{m.group(1).upper()}-{m.group(2)}"
                if sku in self.products:
                    add(sku)
        for src in [text, *extra_names]:
            n = _norm(src)
            if not n:
                continue
            padded = f" {n} "
            hit = [sku for full, sku in self._full.items() if f" {full} " in padded]
            if hit:
                for s in hit: add(s)
                continue
            for base, skus in self._base.items():
                if f" {base} " in padded:
                    for s in skus: add(s)
        if found:
            return found
        # fuzzy fallback on analyzer-provided names only (typos from ASR)
        import difflib
        for name in extra_names:
            n = _norm(name)
            if len(n) < 6:
                continue
            cands = difflib.get_close_matches(n, list(self._base) + list(self._full), n=1, cutoff=0.82)
            if cands:
                c = cands[0]
                for s in self._base.get(c, [self._full.get(c)]):
                    if s: add(s)
        return found

    @staticmethod
    def effective_assured(p) -> bool:
        """MKS-3: Assured never applies to marketplace-seller items, whatever the listing says."""
        return p["assured"] == "True" and p["fulfilled_by"] == "ZentroMart"

    def describe_product(self, p) -> str:
        eff = self.effective_assured(p)
        note = ""
        if p["assured"] == "True" and not eff:
            note = " (listing says Assured but item is fulfilled by a marketplace seller, so ZentroMart Assured does NOT apply: MKS-3)"
        return (f"{p['sku']} | {p['title']} | brand {p['brand']} | category {p['return_category']} | "
                f"price Rs {p['price_inr']} | fulfilled_by {p['fulfilled_by']} | catalog assured={p['assured']}"
                f" -> ZentroMart Assured applies: {'YES (12 months from delivery, in addition to brand warranty)' if eff else 'NO'}{note} | "
                f"brand warranty {p['warranty_months']} months")

    # ------------------------------------------------------------------ pincodes
    def pincode(self, pin):
        return self.pincodes.get(pin)


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
