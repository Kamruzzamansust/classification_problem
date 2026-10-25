"""Offline tests of completeness.py: what a regex leaves out of a message, and the fixed facts that close the gap."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import regex
from sms_profiling import completeness as C

MSG = "Dear Rina Das , thanks for choosing Popular Diagnostic Centre (Narayanganj) as your healthcare  partner. Your Registration ID is 40000091 ."
RX = regex.compile(r"Dear (?P<name>.+?)\s*,.*Registration ID is (?P<registration_id>\d+)")
FACTS = {"organisation": "Popular Diagnostic Centre", "branch": "Narayanganj"}


def kinds(text, kind):
    return [p.text for p in C.salient_parts(text) if p.kind == kind]


def test_the_branch_is_reported_as_left_out_until_it_is_declared():
    m = RX.search(MSG)
    left = C.uncovered_parts(MSG, m)
    assert [p.text for p in left if p.kind == "word"] == ["Popular", "Diagnostic", "Narayanganj"], left      # 'Centre', 'Dear', 'Your' are ordinary wording
    assert not [p for p in left if p.kind == "number"], "the registration id is captured by a group"
    assert not C.uncovered_parts(MSG, m, FACTS), "declared facts cover their text"
    meas = C.measure([(MSG, m)], {})
    assert C.gaps(meas) and list(meas["words"]) == ["Popular", "Diagnostic", "Narayanganj"]
    text = C.describe_gaps(meas)
    assert "Narayanganj" in text and "fixed_facts" in text and text.startswith("incomplete")
    assert not C.gaps(C.measure([(MSG, m)], FACTS)) and C.leftover(C.measure([(MSG, m)], FACTS)) == {}
    print("OK  the branch / organisation are reported as left out, and stop being reported once declared as facts")


def test_what_counts_as_information():
    assert kinds("Tk 2,949.31 withdrawn on 06-06-26 at 08:34 PM", "number") == ["2,949.31", "06-06-26", "08:34"], "amounts, dates, times stay whole"
    assert kinds("আপনার অ্যাকাউন্টে ৫০০ টাকা জমা হয়েছে www.brac.net", "number") == ["৫০০"], "Bengali digits are numbers"
    assert kinds("আপনার অ্যাকাউন্টে ৫০০ টাকা জমা হয়েছে www.brac.net", "url") == ["www.brac.net"]
    assert [p.text for p in C.salient_parts("আপনার অ্যাকাউন্টে ৫০০ টাকা জমা হয়েছে")] == ["৫০০"], "Bengali has no capitals: only digits and links are checked"
    assert kinds("mail help@brac.com or visit https://www.labaid.com/feedback now", "url") == ["help@brac.com", "https://www.labaid.com/feedback"]
    assert kinds("see https://www.Labaid.com/Feedback", "word") == [] and kinds("see https://www.Labaid.com/Feedback", "number") == []
    assert kinds("LABAID sent this. Please wait. Congratulations Rina", "word") == ["LABAID", "Rina"], "ALL-CAPS always counts; a capital at the start of a sentence does not"
    assert kinds("Your OTP is 5 minutes valid. Thanks, BRAC Bank", "word") == ["BRAC"]
    print("OK  numbers (also Bengali digits), links / e-mails, proper-noun-like words are found; ordinary wording and sentence-initial capitals are not")


def test_a_group_that_swallows_the_message_is_free_text():
    m = regex.compile(r"Dear (?P<rest>.+)").search(MSG)
    meas = C.measure([(MSG, m)], {})
    assert meas["free_text_groups"] == ["rest"], meas
    ok = C.measure([(MSG, RX.search(MSG))], FACTS)
    assert ok["free_text_groups"] == []
    print("OK  a group covering most of the message is flagged (it would hide everything else)")


def test_left_out_must_be_systematic():
    rx = regex.compile(r"Dear (?P<name>\w+) .*ID is (?P<id>\d+)")
    msgs = [f"Dear Ali{c} welcome to Popular Diagnostic. Your ID is 4000009{i}" for i, c in enumerate("abcdefghij")]
    msgs[3] = "Dear Ali welcome. Your ID is 40000093"            # one message without the organisation
    meas = C.measure([(t, rx.search(t)) for t in msgs], {})
    assert set(meas["words"]) == {"Popular", "Diagnostic"} and meas["words"]["Popular"] == 0.9, meas["words"]
    one_off = [f"Dear Ali welcome. Your ID is 4000009{i}" for i in range(9)] + ["Dear Ali welcome Popular. Your ID is 40000090"]
    assert C.measure([(t, rx.search(t)) for t in one_off], {})["words"] == {}, "a word left out in a few messages only is not systematic"
    assert C.measure([(t, rx.search(t)) for t in msgs], {"o": "Popular Diagnostic"})["words"] == {}
    print("OK  only parts left out in at least 30% of the matched messages are reported")


def test_a_varying_word_is_caught_although_each_value_is_rare():
    rx = regex.compile(r"Dear (?P<name>\w+) .*ID is (?P<id>\d+)")
    msgs = [f"Dear Ali{c} welcome to Centre {b}. Your ID is 4000009{i}" for i, (c, b) in enumerate(zip("abcdefghij", ["Gazipur", "Khulna", "Savar", "Kushtia", "Cumilla", "Bogura", "Jashore", "Rangpur", "Sylhet", "Pabna"]))]
    meas = C.measure([(t, rx.search(t)) for t in msgs], {})
    assert C.gaps(meas) and len(meas["words"]) == 10 and all(v == 0.1 for v in meas["words"].values()), meas["words"]
    covered = regex.compile(r"Dear (?P<name>\w+) .*Centre (?P<branch>.+?)\. Your ID is (?P<id>\d+)")
    assert not C.gaps(C.measure([(t, covered.search(t)) for t in msgs], {})), "captured: nothing left out"
    one_off = msgs[:1] + [f"Dear Ali welcome. Your ID is 4000009{i}" for i in range(9)]
    assert not C.gaps(C.measure([(t, rx.search(t)) for t in one_off], {})), "a word in a single message of ten is not systematic"
    print("OK  a varying word (branch, merchant) is caught even though each value is rare: judged per message")


def test_check_facts():
    texts = [MSG, MSG.replace("Rina", "Tania")]
    ok, problems = C.check_facts(FACTS, texts, {"name", "registration_id"})
    assert ok == FACTS and not problems
    assert C.check_facts(None, texts, set()) == ({}, []) and C.check_facts({}, texts, set()) == ({}, [])
    ok, problems = C.check_facts({"branch": "Gazipur"}, texts, set())
    assert ok == {} and "appears in only 0%" in problems[0], problems
    ok, problems = C.check_facts({"branch": "Narayanganj"}, texts + ["no branch here"] * 2, set())
    assert ok == {} and "appears in only 50%" in problems[0], problems
    for bad, why in [({"name": "Narayanganj"}, "same name as a regex group"), ({"login_pin": "Narayanganj"}, "credentials"), ({"Branch Name": "Narayanganj"}, "snake_case"),
                     ({"branch": "N"}, "2-80"), ({"branch": "x" * 90}, "2-80"), ({"branch": 5}, "text copied"), ([("branch", "x")], "dict")]:
        ok, problems = C.check_facts(bad, texts, {"name"})
        assert ok == {} and problems and why in problems[0], (bad, problems)
    ok, problems = C.check_facts({f"f{i}": "Popular" for i in range(9)}, texts, set())
    assert any("max 8" in p for p in problems)
    ok, problems = C.check_facts({"shipping_fee": "Popular"}, texts, set())
    assert ok == {"shipping_fee": "Popular"} and not problems, "'shipping' is not the secret word 'pin'"
    print("OK  facts: must occur verbatim in the messages; no duplicates of groups, no secrets, snake_case, at most 8")


def test_facts_must_be_fixed_wording_of_the_template():
    tpl = "[VAR] thanks for choosing Popular Diagnostic Centre Narayanganj as your healthcare partner Your Registration ID is [NUM]"
    texts = [MSG, MSG.replace("Rina", "Tania")]
    ok, problems = C.check_facts(FACTS, texts, set(), tpl)
    assert ok == FACTS and not problems, problems
    # dates, numbers and names that merely happen to be constant in a short sample are NOT facts: they vary in real life
    for bad in ({"date": "06/06/2026"}, {"hotline": "16263"}, {"patient": "Rina"}, {"id": "40000091"}, {"x": "Popular Diagnostic Gazipur"}):
        ok, problems = C.check_facts(bad, texts, set(), tpl)
        assert ok == {} and "not fixed wording" in problems[0], (bad, problems)
    # a bracketed brand is a word of the template, a masked slot is not
    assert C._literal_words("[Pathao] [VAR] [NUM] [CODE] পণ্য") == {"Pathao", "পণ্য"}
    assert C.check_facts({"brand": "Pathao"}, ["[Pathao] x"], set(), "[Pathao] [VAR] x")[0] == {"brand": "Pathao"}
    print("OK  facts must be words that are fixed in the template: a constant date / number / name is rejected")


def test_secret_values_need_not_be_captured():
    msg = "Your report is ready. User name: H79609008757 and Password is 586791. Verification code 4821 for Parkview"
    m = regex.compile(r"report is ready").search(msg)
    left = C.uncovered_parts(msg, m)
    assert [p.text for p in left] == ["Parkview"], left                      # the credentials are never output, so nobody has to account for them
    msg2 = "Dear user, your balance is Tk 500 and pin 1234 is secret. spin 7"
    left2 = [p.text for p in C.uncovered_parts(msg2, regex.compile("Dear").search(msg2))]
    assert "500" in left2 and "1234" not in left2 and "7" in left2, left2         # only a value right after a credential word is exempt
    print("OK  values right after password / PIN / OTP / user name are exempt from the completeness check; other numbers are not")


def test_constant_numbers_and_slot_words_are_not_demanded():
    rx = regex.compile(r"Dear (?P<name>\w+) .*ID is (?P<id>\d+)")
    def run(msgs, template=None, facts=None):
        return C.measure([(t, rx.search(t)) for t in msgs], facts, template)
    hot = [f"Dear Ali{c} please call 10606 for help. Your ID is 4000009{i}" for i, c in enumerate("abcdefghij")]
    assert run(hot)["numbers"] == 0, "a hotline that is the same in every message needs neither a group nor a fact"
    varying = [f"Dear Ali{c} you got {100 + i} points. Your ID is 4000009{i}" for i, c in enumerate("abcdefghij")]
    assert run(varying)["numbers"] == 1.0 and C.gaps(run(varying)), "a number that changes must be captured"
    dated = [f"Dear Ali{c} your pack ends on 06-Jun-26. Your ID is 4000009{i}" for i, c in enumerate("abcdefghij")]
    assert run(dated)["numbers"] == 1.0, "a date that is constant in a one-day sample still varies in real life: it must be captured"
    timed = [f"Dear Ali{c} we call at 09:30 am. Your ID is 4000009{i}" for i, c in enumerate("abcdefghij")]
    assert run(timed)["numbers"] == 1.0
    assert run(varying[:3])["numbers"] == 0, "with fewer than 5 distinct messages nobody can tell whether a number varies"
    # a word that is constant but sits in a masked slot of the template (a name that happens to repeat) cannot be declared a fact
    slot = [f"Dear Ali{c} welcome Rahman. Your ID is 4000009{i}" for i, c in enumerate("abcdefghij")]
    assert list(run(slot)["words"]) == ["Rahman"], "without the template nothing is exempt"
    assert run(slot, "[VAR] welcome [VAR] Your ID is [NUM]")["words"] == {}, "constant word in a [VAR] slot: not demanded"
    assert list(run(slot, "[VAR] welcome Rahman Your ID is [NUM]")["words"]) == ["Rahman"], "fixed wording of the template must be declared"
    print("OK  constant numbers / words in masked slots are not demanded; constant dates, times and fixed template wording are")


def test_facts_ignore_spacing_and_line_breaks():
    text = "Thanks from Epic Health\n Care  Dhaka"
    m = regex.compile(r"Thanks").search(text)
    assert C.check_facts({"org": "Epic Health Care"}, [text], set())[0] == {"org": "Epic Health Care"}
    assert not C.uncovered_parts(text, m, {"org": "Epic Health Care", "city": "Dhaka"})
    assert [p.text for p in C.uncovered_parts(text, m)] == ["Epic", "Dhaka"]
    print("OK  a fact matches however the message spaces or breaks its lines")


def test_a_number_cut_into_pieces_is_caught():
    msgs = [f"Your repayment TK{a}.0 has been paid successfully. Please open the app." for a in (2613, 900, 3016, 2427, 1230, 450, 1800, 2200)]
    bad = regex.compile(r"Your repayment\s+TK(?P<repayment_code>\d+)\.(?P<amount>\d+)")                 # the agent's real mistake: TK2613.0 -> code 2613 + amount 0
    meas = C.measure([(t, bad.search(t)) for t in msgs])
    assert meas["split_amounts"]["share"] == 1.0 and meas["split_amounts"]["groups"] == ["amount", "repayment_code"], meas["split_amounts"]
    msg = C.split_amount_problem(meas)
    assert "cut into pieces" in msg and "TK(?P<amount>" in msg and "[CODE] [NUM]" in msg, msg
    for good in (r"Your repayment\s+TK(?P<amount>\d+(?:\.\d+)?)", r"Your repayment\s+(?P<amount>TK\d+\.\d+)", r"Your repayment\s+(?P<anything>.+?) has been"):
        assert C.split_amount_problem(C.measure([(t, regex.compile(good).search(t)) for t in msgs])) is None, good
    partial = [f"Balance Tk 2,949.{i}1 on 06-06-26" for i in range(8)]
    cut = regex.compile(r"Balance Tk (?P<balance>[\d,]+)")                                                  # stops at the decimal point
    assert "2949.01" in C.split_amount_problem(C.measure([(t, cut.search(t)) for t in partial]))
    untouched = regex.compile(r"(?P<w>Balance)")                                                           # no group touches the amount: that is 'left out', a different check
    assert C.split_amount_problem(C.measure([(t, untouched.search(t)) for t in partial])) is None
    print("OK  an amount cut into two groups (TK2613.0 -> code 2613 + amount 0), or cut at its decimals, is rejected; whole captures are fine")


def test_money_fields_must_hold_amounts():
    zero = ["0"] * 10
    assert "0 in almost every message" in C.money_problem("amount", zero) and "whole number" in C.money_problem("emi_amount", zero)
    assert C.money_problem("amount", ["2613", "900", "3016", "2427", "1230", "TK 450"]) is None, "currency words around the number are fine"
    assert "not amounts" in C.money_problem("balance", ["abc", "def", "ghi", "jkl", "mno", "pqr"])
    assert C.money_problem("repayment_code", zero) is None and C.money_problem("installment_no", zero) is None and C.money_problem("tenure_months", zero) is None, "not money"
    assert C.money_problem("amount", ["0", "0"]) is None, "too few values to judge"
    assert C.money_problem("amount", [0.0] * 6) is not None and C.money_problem("amount", [0.0, 5.0, 7.5, 9.0, 0.0, 12.0]) is None
    print("OK  a field named like money (amount, balance, fee, emi...) must hold amounts that are not all zero; codes, numbers and counts are not judged")


def test_field_labels_are_not_asked_for():
    msg = "Dear, M/S. Zara Steel. Challan No:700000163 Trip No 6000073 Qty:1.00, Amount:9450.00 Taka. Vehicle No.Dhaka Metro NA Driver- ZAMAN ; Cell-0171"
    words = kinds(msg, "word")
    assert words == ["Steel", "Metro", "ZAMAN"], words     # labels (Challan, Trip, Qty, Vehicle, Driver, Cell) are wording; values are not ("Zara", "Dhaka" follow a full stop: treated as sentence starts)
    assert kinds("PARTY NAME- Rahim, MAT CODE- X1", "word") == ["Rahim"]
    # but an entity is still asked for: followed by a bracket, a space, a comma or a full stop
    assert kinds("thanks for choosing Popular Diagnostic Centre (Narayanganj) as your partner", "word") == ["Popular", "Diagnostic", "Narayanganj"]
    assert kinds("see us at Centre Narayanganj. Thanks", "word") == ["Narayanganj"] and kinds("branch Gazipur, open", "word") == ["Gazipur"]
    print("OK  field labels (Qty:, Challan No:, Trip No 123, Driver-) are wording and are not asked for; names and places still are")


def test_check_event():
    assert all(C.check_event(e) is None for e in ("registration_confirmation", "atm_withdrawal", "otp", "report_ready", "a_b_c_d_e"))
    assert all(C.check_event(e) for e in ("", "Registration", "pack expired", "a_b_c_d_e_f", "x" * 60, "1abc", None, 5))
    print("OK  event labels: short snake_case")


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        fn()
    print("\nALL COMPLETENESS TESTS PASSED")
