from datetime import date
from voiceops import dates as D
from voiceops.datastore import get_store
from voiceops.textnorm import clean_asr, extract_order_id, extract_verification, speakable
from voiceops.evalkit import must_include_ok

HOL = get_store().holidays


def test_business_day_refund_eta_matches_dataset_gold():
    # gold answers from practice/questions.csv
    cases = [("card", date(2026, 10, 17), "2026-10-27"), ("upi", date(2026, 10, 12), "2026-10-15"),
             ("netbanking", date(2026, 10, 14), "2026-10-21"), ("card", date(2026, 10, 12), "2026-10-21"),
             ("netbanking", date(2026, 10, 12), "2026-10-17"), ("netbanking", date(2026, 10, 17), "2026-10-24"),
             ("upi", date(2026, 10, 17), "2026-10-22")]
    for m, start, gold in cases:
        assert D.refund_eta(m, start, HOL)["latest"].isoformat() == gold, (m, start)


def test_return_windows():
    assert D.return_window("electronics", date(2026, 7, 20), None)["days"] == 10      # PQ0019
    assert D.return_window("electronics", date(2026, 10, 18), None)["days"] == 7
    assert D.return_window("fashion", date(2026, 10, 14), date(2026, 10, 6))["days"] == 30   # RET-8
    assert D.return_window("fashion", date(2026, 10, 14), date(2026, 9, 30))["days"] == 15
    assert D.return_window("grocery", None, None)["returnable"] is False


def test_overdue_rule():
    assert D.refund_overdue("upi", date(2026, 7, 20), HOL)["overdue"]                    # PC010
    assert not D.refund_overdue("upi", date(2026, 10, 18), HOL)["overdue"]


def test_asr_and_ids():
    assert clean_asr("so sku ele-zero zero one to pe", ["ele"]) == "so sku ELE-0012 pe"
    assert clean_asr("17 october two zero two six") == "17 october 2026"
    assert extract_order_id("order number is eight nine seven eight two eight three seven"[0:0] + clean_asr("order number is eight nine seven eight two eight three seven"))[0] == "89782837"
    assert extract_order_id("Order number hai 8978 2837.")[0] == "89782837"
    assert extract_verification("आख़िरी चार अंक 4533 हैं।") == "4533"
    assert extract_verification("Pincode 440010 hai.") == "440010"
    assert extract_verification("refund started 17 October 2026") is None


def test_marketplace_assured_rule():
    st = get_store()
    assert st.effective_assured(st.product("ELE-0012")) is True
    assert st.effective_assured(st.product("ELE-0015")) is False   # Assured badge but marketplace seller


def test_speakable():
    t = speakable("Refund of ₹2,480 on order 89782837 by 27 October 2026.", "en")
    assert "2480 rupees" in t and "8978 2837" in t and "27th of October 2026" in t


def test_must_include_semantics():
    assert must_include_ok("It arrives by 27th of October 2026", "2026-10-27 / 27 October 2026")[0]
    assert must_include_ok("12 months, raise it in My Orders", "12 ; My Orders / app")[0]
    assert not must_include_ok("5 to 7 days", "instant / 2 hours")[0]
