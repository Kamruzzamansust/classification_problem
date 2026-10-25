"""Offline tests of the generic fallback (amounts, dates, times, links, phone numbers) that reads messages no template regex covers."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json, re, time
from sms_profiling.generic_extractors import GENERIC_PATTERN, extract_generic as g


def test_a_banking_message():
    m = "TK 18,500.00 withdrawn from A/C#10000**6001 on 06-06-26 @08:34 PM from SERPUR ATM. Balance TK 2,949.31. Helpline 16216"
    assert g(m) == {"debited_amount": 18500.0, "balance": 2949.31, "currency": "BDT", "date": "2026-06-06", "time": "20:34"}, g(m)
    assert list(g(m)) == ["debited_amount", "balance", "currency", "date", "time"], "amounts in reading order, then date, time"
    assert GENERIC_PATTERN == "generic-fallback"
    print("OK  a typical banking SMS: debited amount, balance, currency, date, time (the account number and the helpline are not read)")


def test_amounts_need_a_currency_or_a_formatted_number():
    assert g("Your OTP is 482915. Valid for 5 minutes. Order 12345 for 3 items.") is None, "plain integers are OTPs, ids, counts - never amounts"
    assert g("Get 2.5 GB data, 30% off, 1.5 km, 99.9 percent, 10.50 minutes") is None, "numbers with a unit are not money"
    assert g("Tk 500") == {"amount": 500.0, "currency": "BDT"} and g("500 Tk") == {"amount": 500.0, "currency": "BDT"}
    assert g("BDT 1,250.50 received")["credited_amount"] == 1250.5 and g("Rs. 99 only")["currency"] == "INR" and g("pay $ 20")["currency"] == "USD"
    assert g("৳ ১,২৫০ জমা") == {"credited_amount": 1250.0, "currency": "BDT"}, "Bengali digits and symbol"
    assert g("আপনার ৫০০ টাকা জমা হয়েছে")["credited_amount"] == 500.0
    assert g("Fee 2,000/- only")["fee"] == 2000.0, "'/-' marks an amount"
    assert g("Pay Tk. 1,000, now") == {"amount": 1000.0, "currency": "BDT"}, "a trailing comma or full stop is not part of the number"
    assert g("Tk 1,00,000.00 credited")["credited_amount"] == 100000.0, "Indian digit grouping"
    assert "amount" not in (g("Tk 99999999999 big") or {}), "absurd values are skipped"
    assert g("Balance 500.00")["balance"] == 500.0 and "currency" not in g("Balance 500.00"), "a formatted number is an amount; no currency is assumed"
    assert g("Tk 500, Tk 700") == {"amount": 500.0, "amount_2": 700.0, "currency": "BDT"}, "a second amount of the same kind gets _2"
    assert g("Tk 500 paid. Tk 500 paid.") == {"paid_amount": 500.0, "currency": "BDT"}, "the same value twice is kept once"
    print("OK  amounts: need a currency or a formatted number; units, OTPs, ids and counts are not money; Bengali digits, '/-', Indian grouping work")


def test_amounts_are_labelled_by_the_words_around_them():
    cases = {
        "Available balance Tk 1,200.00": "balance", "Your current bal: BDT 50.25": "balance", "বর্তমান ব্যালেন্স ৳ ৫০০": "balance",
        "Tk 500 has been credited to your account": "credited_amount", "Tk 500 successfully debited": "debited_amount",
        "BDT 1,000 withdrawn from ATM": "debited_amount", "Total Tk 800": "total_amount", "Your bill Tk 700 is overdue": "bill_amount",
        "Amount due Tk 450": "due_amount", "Service charge Tk 5.00": "fee", "VAT Tk 2.50": "fee", "Recharge of Tk 100 done": "recharge_amount",
        "Tk 300 paid to Pathao": "paid_amount", "Cashback Tk 20": "reward_amount", "Fare ৳ 450": "price", "Daily limit Tk 25,000": "limit_amount",
        "Tk 10 xyz": "amount", "Your A/C has been Credited by Tk 500": "credited_amount", "Debited by BDT 20": "debited_amount",
        "Tk 3.75 Vat on SMS Txn Charge": "fee", "Tk 150 SMS Txn Charge": "fee", "Tk 10,000 Deposit for A/C": "credited_amount",
        "ATM Txn Tk 5,000 Withdrawal": "debited_amount", "[Pathao] COD Tk 1,550": "cod_amount", "your txn token to debit Tk 5,075": "debited_amount",
    }
    assert "due_amount" not in g("Transaction of Tk 5,668 has been declined due to Expired Card"), "'declined due to' is not a due amount"
    assert g("Inv.X1, Amt.1234.50 by A/C") == {"amount": 1234.5}, "'Amt.' / 'Amount:' say a number is money, even without a currency"
    assert g("Transaction amount of 5,668.00 declined")["amount"] == 5668.0 and g("Amount: 500")["amount"] == 500.0
    assert g("Dear 500 amounts") is None
    for text, key in cases.items():
        got = g(text)
        assert got and key in got, (text, key, got)
    two = g("Tk 500 withdrawn. Balance Tk 200")
    assert two["debited_amount"] == 500.0 and two["balance"] == 200.0, "each amount is labelled by ITS words, not its neighbour's"
    print("OK  amounts get a meaningful key (balance / credited / debited / due / fee / paid / recharge / reward / price / limit / total) from the words around them")


def test_real_world_quirks_found_on_the_full_data():
    breb = "Vending AMT:1,580.Successful TXnID:AWD-s4yxc on 06-06-2026 20:56:34.Breakup:Seq:-2~9,EnergyCost:1,346.05,MeterRent:40,DemandCharge:126,Vat:75.24,Rebate:-7.29,Arrears:N/A."
    got = g(breb)
    assert got["amount"] == 1580.0 and got["amount_2"] == 1346.05 and got["fee"] == 75.24 and got["amount_3"] == -7.29, got      # decimals kept, labels stop at the neighbouring number, sign kept
    assert g("Balance 500.00.")["balance"] == 500.0 and g("Balance 1,250.50, thanks")["balance"] == 1250.5, "a full stop or comma after the number is not part of it"
    assert g("MYCash debited TK.1,000 Bal.TK.3,283.907,TNX:AWD-s4yq")["balance"] == 3283.907, "three decimals are kept"
    assert g("Tk 500 withdrawn from A/C 6001, Balance Tk 7") == {"debited_amount": 500.0, "balance": 7.0, "currency": "BDT"}, "the account number is not read"
    assert g("Pay 10-20 Tk") == {"amount": 20.0, "currency": "BDT"}, "a range is not a negative amount"
    assert g("net -5.50 due") == {"amount": -5.5}
    print("OK  real-data quirks: decimals before a comma, labels bounded by the neighbouring number, negative amounts, three-decimal balances")


def test_dates_and_times():
    for text, iso in [("on 06-06-26", "2026-06-06"), ("on 06/06/2026", "2026-06-06"), ("on 2026-06-06", "2026-06-06"), ("on 6 Jun 2026", "2026-06-06"),
                      ("on 06-Jun-26", "2026-06-06"), ("on 6th June, 2026", "2026-06-06"), ("on Jun 6, 2026", "2026-06-06"), ("on 07 Jun, 2026 at", "2026-06-07"),
                      ("তারিখ ০৬-০৬-২৬", "2026-06-06"), ("on 15.06.2026", "2026-06-15")]:
        assert g(text) == {"date": iso}, (text, g(text))
    assert g("expires on 15-Jun-2026, due by 20/06/2026") == {"expiry_date": "2026-06-15", "due_date": "2026-06-20"}, "words say which date it is"
    assert g("মেয়াদ ১৫-০৬-২৬ পর্যন্ত") == {"expiry_date": "2026-06-15"}
    for bad in ("on 31-02-26", "on 45-13-2026", "on 01-01-1999", "on 05/06/2099"):
        assert g(bad) is None, (bad, g(bad))
    assert g("at 08:34 PM") == {"time": "20:34"} and g("at 20:15") == {"time": "20:15"} and g("9:05:30 pm") == {"time": "21:05:30"} and g("@09:05AM") == {"time": "09:05"}
    assert g("at 25:99") is None and g("ratio 1:2") is None
    assert g("on 06-06-26 @08:34 PM") == {"date": "2026-06-06", "time": "20:34"}, "a time is not an amount and a date is not read twice"
    print("OK  dates (day first, many formats, Bengali digits) -> ISO; expiry / due labels; times -> 24h; impossible, old or far-future dates are not read")


def test_links_keep_only_the_domain():
    got = g("visit https://www.labaiddiagnostics.com/feedback?token=abc123XYZ&id=998877 now")
    assert got == {"link_domain": "labaiddiagnostics.com"} and "abc123" not in json.dumps(got) and "998877" not in json.dumps(got), got
    assert g("see bkash.com/app and www.brac.net")["link_domain"] == "bkash.com" and g("see bkash.com/app and www.brac.net")["link_domain_2"] == "brac.net"
    assert g("https://abc.bd https://abc.bd/x") == {"link_domain": "abc.bd"}, "the same domain once"
    assert g("mail help@brac.com for help") is None, "an e-mail address is personal data: skipped, and not mistaken for a link"
    assert g("Dear Md.Co and Dr.Me") is None, "a title and a name are not a link"
    assert g("http://user:pw@host.example.org:8080/p")["link_domain"] == "host.example.org", "credentials in a URL never leak"
    print("OK  links: only the domain (no path, query, token, credentials); e-mails and 'Md.Co' are not links")


def test_phone_numbers_are_masked():
    assert g("call 01712345678") == {"phone": "****5678"} and g("call +8801912345678")["phone"] == "****5678" and g("call 017-1234-5678")["phone"] == "****5678"
    assert g("call 01712345678", keep_identifiers=True) == {"phone": "01712345678"}, "keep_identifiers keeps the number, normalised"
    assert g("call +88 01912345678", keep_identifiers=True)["phone"] == "01912345678"
    assert g("call 01712345678 or 01898765432") == {"phone": "****5678", "phone_2": "****5432"}
    assert g("hotline 16216 or 10606") is None, "a short code is a service number, not a person"
    assert g("ref 02712345678") is None and g("ref 8801712345678901") is None, "not a mobile number / inside a longer number"
    assert g("আমাদের নম্বর ০১৭১২৩৪৫৬৭৮") == {"phone": "****5678"}, "Bengali digits"
    print("OK  phones: Bangladeshi mobile numbers only, reduced to the last four digits (keep_identifiers keeps them); hotlines and long ids are not phones")


def test_nothing_sensitive_is_ever_returned():
    msg = ("Your OTP is 482915, PIN 4821, password Pw839201. Account 123456789012 card 4111111111111111 invoice INV20260607123456 "
           "Tk 1,250.00 sent to 01712345678 on 06-06-26 visit https://x.example.com/reset?token=998877665544 mail me@x.com")
    out = g(msg)
    text = json.dumps(out)
    assert out["debited_amount"] == 1250.0 and out["phone"] == "****5678" and out["link_domain"] == "x.example.com"
    assert not re.search(r"\d{6,}", text.replace("1250.0", "")), text
    for secret in ("482915", "4821", "Pw839201", "123456789012", "4111111111111111", "998877665544", "me@x.com", "token"):
        assert secret not in text, (secret, text)
    print("OK  OTP, PIN, password, account / card numbers, long ids, URL tokens and e-mail addresses are never returned; phones are masked")


def test_robustness_and_speed():
    for bad in (None, float("nan"), 12345, "", "   ", b"x", ["a"]):
        assert g(bad) is None
    assert g("no numbers or links in this message at all") is None
    assert g("Tk 500 " * 3000) == {"amount": 500.0, "currency": "BDT"}
    msg = "TK 18,500.00 withdrawn from A/C#10000**6001 on 06-06-26 @08:34 PM from SERPUR ATM. Balance TK 2,949.31. Visit https://x.example.com/a call 01712345678"
    assert g(msg) == g(msg), "deterministic"
    t0 = time.time()
    for _ in range(3000):
        g(msg)
    per_s = 3000 / (time.time() - t0)
    assert per_s > 1500, f"too slow: {per_s:.0f} messages/s"
    print(f"OK  robust to None / NaN / numbers / empty / huge input; deterministic; {per_s:,.0f} messages/s")


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        fn()
    print("\nALL GENERIC EXTRACTOR TESTS PASSED")
