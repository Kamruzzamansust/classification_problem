"""Offline tests of script 2 (extract_information.py): columns, typed values, privacy, empty rows, filters, robustness, CLI."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json, os, subprocess, sys, tempfile
import pandas as pd
from sms_profiling.extract_information import INFO_COL, REGEX_COL, RegexExtractor, extract_information, summary
from sms_profiling.templates import TemplateModel, generate_all

TEL, BANK, OTP = "Telecom / Operator Service", "Transactional (Banking)", "OTP / Verification Code"

TEL_RX = r"(?P<account_id>\d+)\s+আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে\s+(?P<expiry_date>\d{1,2}-\w{3}-\d{2})\s*।?\s*বর্তমান ব্যালেন্স\s+(?P<current_balance>[\d,]+(?:\.\d+)?)"
BANK_RX = (r"^TK\s*(?P<amount>[\d,]+(?:\.\d{1,2})?)\s+withdrawn\s+from\s+A/C#(?P<account_reference>[\d*]+)\s+on\s+(?P<transaction_date>\d{1,2}-\d{1,2}-\d{2,4})"
           r"\s+@(?P<transaction_time>\d{1,2}:\d{2}\s*[AP]M)\s+from\s+(?P<location>.+?)\s*\.\s*Balance\s+TK\s*(?P<remaining_balance>[\d,]+(?:\.\d{1,2})?)")
OTP_RX = r"Your (?P<service_name>\w+) verification code is (?P<otp_code>\d{6})\.? Do not share"


def tel(i):  return f"আপনার 01712{i:06d} আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে 0{i}-Jun-26 । বর্তমান ব্যালেন্স {100 + i * 7},{i}00.50 ।"
def bank(i): return f"TK {i}5,000.00 withdrawn from A/C#12345**67{i}{i} on 0{i}-06-26 @09:0{i} AM from MIRPUR 10 ATM DHAKA BD. Balance TK {i}7,949.31. BRAC Bank."
def otp(i):  return f"Your Daraz verification code is 48291{i}. Do not share it with anyone."


def build():
    rows = [(1000 + i, tel(i), TEL) for i in range(1, 8)] + [(2000 + i, bank(i), BANK) for i in range(1, 8)] + [(3000 + i, otp(i), OTP) for i in range(1, 4)]
    df = pd.DataFrame(rows, columns=["rid", "message_decoded", "Predicted_Category"])
    _, _, model = generate_all(df, "message_decoded", "Predicted_Category", "rid")
    tid = lambda text, cat: model.template_id(text, cat)
    reg = {tid(tel(1), TEL): {"category": TEL, "pattern": "p", "expanded_pattern": TEL_RX, "fields": {}},
           tid(bank(1), BANK): {"category": BANK, "pattern": "p", "expanded_pattern": BANK_RX, "fields": {}},
           tid(otp(1), OTP): {"category": OTP, "pattern": "p", "expanded_pattern": OTP_RX, "fields": {}}}
    return df, model, reg


def test_two_columns_typed_values_and_pattern():
    df, model, reg = build()
    before = df.copy(deep=True)
    out = extract_information(df, RegexExtractor(model, reg))
    assert df.equals(before) and list(df.columns) == ["rid", "message_decoded", "Predicted_Category"], "input is not modified"
    assert list(out.columns) == list(df.columns) + [INFO_COL, REGEX_COL], list(out.columns)       # exactly two new columns
    assert (out.index == df.index).all() and (out["rid"] == df["rid"]).all()
    t = json.loads(out.loc[0, INFO_COL])
    assert t == {"account_id": "****0001", "expiry_date": "2026-06-01", "current_balance": 107100.5}, t
    assert out.loc[0, REGEX_COL] == TEL_RX
    b = json.loads(out.loc[7, INFO_COL])
    assert b == {"amount": 15000.0, "account_reference": "****6711", "transaction_date": "2026-06-01", "transaction_time": "09:01",
                 "location": "MIRPUR 10 ATM DHAKA BD", "remaining_balance": 17949.31}, b
    assert out.loc[7, REGEX_COL] == BANK_RX
    assert all(json.loads(s) for s in out[INFO_COL].dropna()), "every information cell is valid, non-empty JSON"
    assert isinstance(json.loads(out.loc[0, INFO_COL])["current_balance"], float)
    print("OK  exactly two columns added; values typed (amount -> number, date -> ISO, time -> 24h); regex_pattern is the regex used")


def test_secrets_never_output_and_identifiers_masked():
    df, model, reg = build()
    out = extract_information(df, RegexExtractor(model, reg))
    o = json.loads(out.loc[14, INFO_COL])
    assert o == {"service_name": "Daraz"}, o                                    # otp_code was captured by the regex but is never output
    assert "482911" not in " ".join(out[INFO_COL].dropna()), "no OTP anywhere"
    full = extract_information(df, RegexExtractor(model, reg, keep_identifiers=True))
    assert json.loads(full.loc[7, INFO_COL])["account_reference"] == "12345**6711" and json.loads(full.loc[0, INFO_COL])["account_id"] == "01712000001"
    print("OK  OTP / PIN / password groups are dropped even if the regex captures them; account ids reduced to last 4 digits (keep_identifiers=True keeps them)")


def test_which_fields_are_masked():
    x = RegexExtractor(TemplateModel(), {})
    out = x.clean({"user_id": "99995683", "contact_phone": "+8801799995238", "phone": "01799995238", "account_id": "10000**6001", "customer_number": "998877",
                   "order_id": "ORD123456", "order_num": "778899", "merchant": "Farmio", "tracking_id": "TRK98765432", "amount": "৳450",
                   "pin": "1234", "otp": "482915", "verification_code": "112233", "short_id": None, "user_no": "12"})
    assert out == {"user_id": "****5683", "contact_phone": "****5238", "phone": "****5238", "account_id": "****6001", "customer_number": "****8877",
                   "order_id": "ORD123456", "order_num": "778899", "merchant": "Farmio", "tracking_id": "TRK98765432", "amount": 450.0}, out
    med = x.clean({"uhid": "999915691", "patient_id": "P-99995796", "patient_name": "RINA DEVI", "doctor_name": "DR. ARUN SEN",
                   "lab_no": "9992588", "invoice_no": "NA9999993685", "uhid_number": "999990071", "mrn": "55", "password": "386836", "user_name": "H99999995052"})
    assert med == {"uhid": "****5691", "patient_id": "****5796", "patient_name": "RINA DEVI", "doctor_name": "DR. ARUN SEN",
                   "lab_no": "****2588", "invoice_no": "NA9999993685", "uhid_number": "****0071"}, med      # user_name (portal login) and password are dropped
    assert x.clean({"label_id": "123456", "lab_number": "99887766", "username": "H1", "login": "x", "user_id": "12345678", "tablet_no": "1234"}) == \
        {"label_id": "123456", "lab_number": "****7766", "user_id": "****5678", "tablet_no": "1234"}, "word boundaries: 'label' / 'tablet' are not lab or user ids"
    bare = x.clean({"id": "99990273", "user": "u1234567", "account": "123456789", "order_id": "ORD123456", "id_card_type": "NID", "valid": "yes"})
    assert bare == {"id": "****0273", "user": "****4567", "account": "****6789", "order_id": "ORD123456", "id_card_type": "NID", "valid": "yes"}, bare
    reg = x.clean({"registration_id": "999997674", "reg_id": "999956691", "lab": "9992588", "admission_id": "A-2020-1234", "lab_name": "Central Lab",
                   "region_id": "12345678", "registration_date": "2026-06-01", "label_id": "123456", "registration": "77889900"})
    assert reg == {"registration_id": "****7674", "reg_id": "****6691", "lab": "****2588", "admission_id": "****1234", "lab_name": "Central Lab",
                   "region_id": "12345678", "registration_date": "2026-06-01", "label_id": "123456", "registration": "****9900"}, reg
    keep = RegexExtractor(TemplateModel(), {}, keep_identifiers=True).clean({"user_id": "99995683", "phone": "01799995238", "pin": "1234"})
    assert keep == {"user_id": "99995683", "phone": "01799995238"}, "keep_identifiers keeps ids, but secrets are never output"
    print("OK  masking: user / customer / account ids and phone numbers -> '****1234'; order ids, merchants, amounts kept; secrets dropped; <4 digits dropped")


def test_fixed_facts_and_event_are_added_to_information():
    df, model, reg = build()
    tid = next(k for k, v in reg.items() if v["category"] == TEL)
    reg[tid] = dict(reg[tid], facts={"service": "Akash", "package": "Standard", "support_phone": "01711122233", "pin": "1234", "account_id": "ignored"},
                    event="pack_expired")
    out = extract_information(df, RegexExtractor(model, reg))
    t = json.loads(out.loc[0, INFO_COL])
    assert list(t)[0] == "event" and t["event"] == "pack_expired", t                                         # what happened comes first
    assert t["service"] == "Akash" and t["package"] == "Standard", t                                         # fixed wording kept as information
    assert t["support_phone"] == "****2233", "a fixed phone number is an identifier-like value: masked like every other"
    assert "pin" not in t, "a fact named like a secret is never output"
    assert t["account_id"] == "****0001", "a value read from the message wins over a fact with the same name"
    assert t["expiry_date"] == "2026-06-01" and t["current_balance"] == 107100.5
    plain = json.loads(extract_information(df, RegexExtractor(model, build()[2])).loc[0, INFO_COL])
    assert "event" not in plain and "service" not in plain, "registries written before facts existed still work, unchanged"
    # a regex that matched but whose values are all dropped (an OTP-only template) still yields its event and facts
    otp_id = next(k for k, v in reg.items() if v["category"] == OTP)
    reg[otp_id] = dict(reg[otp_id], expanded_pattern=r"verification code is (?P<otp_code>\d{6})", event="otp_sent", facts={"organisation": "Daraz"})
    o = json.loads(extract_information(df, RegexExtractor(model, reg)).loc[14, INFO_COL])
    assert o == {"event": "otp_sent", "organisation": "Daraz"}, o
    print("OK  facts + event: added to every matched row (event first); secrets dropped, phones masked, read values win; old registries unchanged")


PRE, DEC = "\u09df", "\u09af\u09bc"            # Bengali য় as one character, and as two: the SAME letter, two encodings


def test_unicode_variants_of_one_message_get_one_template_and_both_extract():
    import unicodedata
    base = tel(3)
    a, b = base.replace(DEC, PRE), base.replace(PRE, DEC)
    assert a != b and unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b), "same visible text, different code points"
    df, model, reg = build()
    ex = RegexExtractor(model, reg)
    ia, pa, ra = ex.extract_one(a, TEL); ib, pb, rb = ex.extract_one(b, TEL)
    assert ra == rb == "extracted" and ia == ib and pa == pb == TEL_RX, (ra, rb)
    # a registry keyed by the template id of the UN-normalised text (built before the fix) is still found
    old_key = {model.template_id(a, TEL): v for v in [reg[model.template_id(unicodedata.normalize("NFC", a), TEL)]]}
    assert model.template_id(a, TEL) != model.template_id(unicodedata.normalize("NFC", a), TEL), "the two spellings really gave two template ids"
    r2 = RegexExtractor(model, old_key).extract_one(a, TEL)
    assert r2[2] == "extracted", r2
    print("OK  the same Bengali message in both encodings: one template, both extract (also against a registry built from un-normalized text)")


def test_templates_stage_merges_unicode_variants():
    import io
    from contextlib import redirect_stdout
    import pipeline as PL
    rows = [(i, tel(i % 7 + 1).replace(DEC, PRE if i % 2 else DEC), TEL) for i in range(40)]          # half stored one way, half the other
    d = tempfile.mkdtemp()
    pd.DataFrame(rows, columns=["rid", "message_decoded", "Predicted_Category"]).to_csv(os.path.join(d, "classified.csv"), index=False)
    with redirect_stdout(io.StringIO()):
        PL.stage_templates(PL.build_parser().parse_args(["templates", "--out-dir", d]))
    reg = pd.read_csv(os.path.join(d, "templates_registry.csv")); mt = pd.read_csv(os.path.join(d, "message_templates.csv"))
    assert len(reg) == 1 and int(reg.message_count.iloc[0]) == 40, reg[["template_id", "message_count"]]
    assert all(PRE not in s for s in mt.message_decoded), "the stored texts are normalized"
    print("OK  templates stage: messages stored with either encoding of the same letter become ONE template")


def test_invalid_stored_regex_is_reported_not_hidden():
    df, model, reg = build()
    tid = next(k for k, v in reg.items() if v["category"] == TEL)
    reg[tid] = dict(reg[tid], expanded_pattern="(unclosed")
    assert RegexExtractor(model, reg).extract_one(tel(1), TEL)[2] == "stored regex is invalid"
    print("OK  a stored regex that cannot compile has its own reason (it is no longer reported as 'no regex yet')")


def test_rows_without_a_regex_stay_empty_and_nothing_is_guessed():
    df, model, reg = build()
    reg.pop(next(k for k, v in reg.items() if v["category"] == OTP))             # no regex for this template
    extra = pd.DataFrame([(9001, None, TEL), (9002, "   ", TEL), (9003, tel(1), "Unknown Category"), (9004, tel(1), None), (9005, "Akash something else entirely", TEL),
                          (9006, tel(2).replace("Akash Standard", "Akash Lite"), TEL), (9007, 12345, TEL)], columns=df.columns)
    out = extract_information(pd.concat([df, extra], ignore_index=True), RegexExtractor(model, reg), fallback=False)
    st = out.attrs["extraction_stats"]
    assert out.loc[14:16, INFO_COL].isna().all() and out.loc[14:16, REGEX_COL].isna().all(), "OTP rows: template has no regex yet -> empty"
    assert out.loc[17:, INFO_COL].isna().all() and out.loc[17:, REGEX_COL].isna().all(), "blank / unknown category / different wording / non-text -> empty"
    assert st["extracted"] == 14 and st["empty text"] == 3 and st["category not modelled"] == 2 and st["template has no regex yet"] >= 3 + 2, st
    assert out[INFO_COL].notna().sum() == 14 and out[REGEX_COL].notna().sum() == 14
    assert "extracted" in summary(out) and "rows: 24" in summary(out)
    print("OK  no regex / blank text / unknown category / different wording / non-text -> both columns empty; reasons are counted")


def test_regex_that_does_not_match_leaves_the_row_empty():
    df, model, reg = build()
    tid = next(k for k, v in reg.items() if v["category"] == TEL)
    reg[tid] = dict(reg[tid], expanded_pattern=r"(?P<x>NEVER_MATCHES_\d+)")
    out = extract_information(df, RegexExtractor(model, reg), fallback=False)
    assert out.loc[:6, INFO_COL].isna().all() and out.attrs["extraction_stats"]["regex did not match"] == 7
    print("OK  a stored regex that does not match its message leaves the row empty (never a guess)")


def test_category_filter():
    df, model, reg = build()
    out = extract_information(df, RegexExtractor(model, reg), categories=[BANK])
    assert out[INFO_COL].notna().sum() == 7 and out.loc[:6, INFO_COL].isna().all() and out.attrs["extraction_stats"]["skipped (other category)"] == 10
    out = extract_information(df, RegexExtractor(model, reg), categories=[BANK, TEL])
    assert out[INFO_COL].notna().sum() == 14
    print("OK  --category filter: only the chosen categories are processed")


def test_broken_and_slow_regexes_are_contained():
    import time
    df, model, reg = build()
    tel_id = next(k for k, v in reg.items() if v["category"] == TEL)
    bank_id = next(k for k, v in reg.items() if v["category"] == BANK)
    reg[tel_id] = dict(reg[tel_id], expanded_pattern="(unclosed")                                   # does not compile
    reg[bank_id] = dict(reg[bank_id], expanded_pattern=r"^(a|aa)+$")                                # exponential backtracking on 'aaaa...b'
    df.loc[7, "message_decoded"] = "a" * 60 + "b"
    t0 = time.time()
    out = extract_information(df, RegexExtractor(model, reg), fallback=False)
    assert time.time() - t0 < 10, "a slow regex must be cut off by the time limit"
    assert out.loc[:6, INFO_COL].isna().all(), "a broken pattern disables only its own template"
    assert out.loc[10:, INFO_COL].notna().sum() == 3, "other templates keep working"
    print("OK  a pattern that does not compile, and one that backtracks forever, affect only their own template")


def test_errors_and_missing_files():
    df, model, reg = build()
    for bad, exc in [([1, 2], TypeError), (df.drop(columns=["message_decoded"]), KeyError), (df.drop(columns=["Predicted_Category"]), KeyError)]:
        try:
            extract_information(bad, RegexExtractor(model, reg)); raise AssertionError("no error")
        except exc:
            pass
    try:
        RegexExtractor.load(tempfile.mkdtemp()); raise AssertionError("no error")
    except FileNotFoundError as e:
        assert "templates" in str(e)
    assert extract_information(df.iloc[0:0], RegexExtractor(model, reg)).shape == (0, 5)
    print("OK  clear errors for a non-DataFrame, missing columns, missing model / registry files; empty DataFrame works")


def test_generic_fallback_fills_rows_without_a_regex():
    from sms_profiling.generic_extractors import GENERIC_PATTERN
    df, model, reg = build()
    reg.pop(next(k for k, v in reg.items() if v["category"] == TEL))                  # the Telecom template has no regex
    extra = pd.DataFrame([(9001, "Tk 500 credited. Balance Tk 1,200.00 on 06-06-26", "Unknown Category"), (9002, "Hello there", TEL), (9003, None, TEL),
                          (9004, "Call 01712345678", None)], columns=df.columns)
    full = pd.concat([df, extra], ignore_index=True)
    out = extract_information(full, RegexExtractor(model, reg))
    tel1 = json.loads(out.loc[0, INFO_COL])
    assert tel1 == {"balance": 107100.5, "date": "2026-06-01", "phone": "****0001"}, tel1                       # the generic reading, phone masked
    assert out.loc[0, REGEX_COL] == GENERIC_PATTERN == "generic-fallback"
    assert out.loc[7, REGEX_COL] == BANK_RX and "amount" in json.loads(out.loc[7, INFO_COL]), "rows with a regex keep the regex result"
    assert json.loads(out.loc[17, INFO_COL]) == {"credited_amount": 500.0, "balance": 1200.0, "currency": "BDT", "date": "2026-06-06"}, "category not modelled: still read"
    assert pd.isna(out.loc[18, INFO_COL]) and pd.isna(out.loc[19, INFO_COL]), "nothing generic in the text / no text: stays empty"
    assert json.loads(out.loc[20, INFO_COL]) == {"phone": "****5678"} and out.loc[20, REGEX_COL] == GENERIC_PATTERN, "no category at all: still read"
    st = out.attrs["extraction_stats"]
    assert st["extracted"] == 10 and st["generic fallback (a subset of the rows above)"] == 9, st            # 7 Telecom + 2 extra rows
    assert "generic fallback" in summary(out)
    off = extract_information(full, RegexExtractor(model, reg), fallback=False)
    assert off.loc[[0, 17, 20], INFO_COL].isna().all() and off.loc[7, INFO_COL] == out.loc[7, INFO_COL] and "generic fallback (a subset of the rows above)" not in off.attrs["extraction_stats"]
    only_bank = extract_information(full, RegexExtractor(model, reg), categories=[BANK])
    assert only_bank.loc[:6, INFO_COL].isna().all() and only_bank[INFO_COL].notna().sum() == 7, "rows of categories that were not asked for are never touched"
    keep = extract_information(full, RegexExtractor(model, reg, keep_identifiers=True))
    assert json.loads(keep.loc[0, INFO_COL])["phone"] == "01712000001", "keep_identifiers applies to the fallback too"
    print("OK  generic fallback: rows without a regex get amounts / dates / phones (regex_pattern 'generic-fallback'); regex rows are untouched; fallback=False leaves them empty")


def test_second_regexes_for_what_the_main_regex_misses():
    df, model, reg = build()
    tid = next(k for k, v in reg.items() if v["category"] == TEL)
    VAR_RX = r"(?P<account_id>\d+)\s+আইডির.*?বর্তমান ব্যালেন্স\s+(?P<current_balance>[\d,]+(?:\.\d+)?)"
    reg[tid + "~1"] = {"category": TEL, "pattern": "p", "expanded_pattern": VAR_RX, "fields": {}, "parent": tid, "event": "other_layout"}
    ex = RegexExtractor(model, reg)
    info, pat, why = ex.extract_one(tel(2), TEL)
    assert why == "extracted" and pat == TEL_RX and "event" not in info, "the main regex is tried first and wins"
    odd = tel(2).replace("।", "!")                                    # the same template (punctuation is ignored) but the main regex does not accept the '!'
    assert RegexExtractor(model, {tid: reg[tid]}).extract_one(odd, TEL)[2] == "regex did not match"
    info, pat, why = ex.extract_one(odd, TEL)
    assert why == "extracted" and pat == VAR_RX and info["event"] == "other_layout" and info["account_id"] == "****0002", (info, pat, why)
    neither = tel(2).replace("বর্তমান ব্যালেন্স", "বর্তমান  ব্যালেন্স")      # a double space: both regexes need a single one
    assert ex.extract_one(neither, TEL)[2] == "regex did not match"
    reg[tid] = dict(reg[tid], expanded_pattern="(unclosed")
    assert RegexExtractor(model, reg).extract_one(odd, TEL)[2] == "extracted", "a broken main regex does not hide a working second one"
    out = extract_information(df, ex)
    assert out[INFO_COL].notna().sum() == 17
    print("OK  second regexes: tried after the main regex, in order; the main one wins where it matches; a broken main regex does not hide them")


def test_command_line_end_to_end():
    df, model, reg = build()
    d = tempfile.mkdtemp()
    df.to_csv(os.path.join(d, "classified.csv"), index=False)
    model.save(os.path.join(d, "template_model.json"))
    json.dump(reg, open(os.path.join(d, "regex_registry.json"), "w", encoding="utf-8"), ensure_ascii=False)
    r = subprocess.run([sys.executable, "-m", "sms_profiling.extract_information", "--out-dir", d], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-400:]
    res = pd.read_csv(os.path.join(d, "extracted.csv"))
    assert list(res.columns)[-2:] == [INFO_COL, REGEX_COL] and res[INFO_COL].notna().sum() == 17 and "with information: 17" in r.stdout
    assert json.loads(res.loc[0, INFO_COL])["current_balance"] == 107100.5
    r = subprocess.run([sys.executable, "-m", "sms_profiling.extract_information", "--out-dir", d, "--category", BANK, "--out", os.path.join(d, "bank.csv")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and pd.read_csv(os.path.join(d, "bank.csv"))[INFO_COL].notna().sum() == 7
    reg.pop(next(k for k, v in reg.items() if v["category"] == TEL))
    json.dump(reg, open(os.path.join(d, "regex_registry.json"), "w", encoding="utf-8"), ensure_ascii=False)
    r1 = subprocess.run([sys.executable, "-m", "sms_profiling.extract_information", "--out-dir", d, "--out", os.path.join(d, "a.csv")], capture_output=True, text=True, cwd=ROOT)
    r2 = subprocess.run([sys.executable, "-m", "sms_profiling.extract_information", "--out-dir", d, "--no-fallback", "--out", os.path.join(d, "b.csv")], capture_output=True, text=True, cwd=ROOT)
    assert "with information: 17" in r1.stdout and "with information: 10" in r2.stdout, (r1.stdout[:200], r2.stdout[:200])
    assert (pd.read_csv(os.path.join(d, "a.csv"))[REGEX_COL] == "generic-fallback").sum() == 7
    pr = subprocess.run([sys.executable, os.path.join(ROOT, "pipeline.py"), "extract", "--out-dir", d, "--no-fallback", "--out", os.path.join(d, "c.csv")], capture_output=True, text=True)
    assert pr.returncode == 0 and "with information: 10" in pr.stdout, pr.stderr[-300:]
    print("OK  runs as a command: classified.csv -> extracted.csv with the two columns; --category filter works")


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        fn()
    print("\nALL EXTRACTION TESTS PASSED")
