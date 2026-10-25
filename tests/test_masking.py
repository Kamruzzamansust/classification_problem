"""Masked text: what is masked, that messages of one template collapse to one text, and that classification runs once per template."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import time
import pandas as pd
from sms_profiling.masking import mask_text
from sms_profiling.preprocess import add_masked_text, prepare_messages
from sms_profiling.classifier import SMSClassifier, classify

WITHDRAW = "TK 18,500.00 withdrawn from A/C#10000**6001 on 06-06-26 @08:34 PM from SERPUR SKB ATM SHERPUR BD. Balance TK 2,949.31. BRAC Bank."


def test_what_is_masked():
    cases = {
        WITHDRAW: "[AMOUNT] withdrawn from A/C#[NUM]**[NUM] on [DATE] @[TIME] from SERPUR SKB ATM SHERPUR BD. Balance [AMOUNT]. BRAC Bank.",
        "Tk 1,000.00 has been transferred from your BBL A/C: 6**6001 to BKASH wallet: 01712345678. Available balance is Tk 1,949.31. Queries: call 16221":
            "[AMOUNT] has been transferred from your BBL A/C: [NUM]**[NUM] to BKASH wallet: [PHONE]. Available balance is [AMOUNT]. Queries: call [NUM]",
        "Your OTP is 482915. Do not share it with anyone. Valid for 5 minutes.": "Your OTP is [NUM]. Do not share it with anyone. Valid for [NUM] minutes.",
        "Dear RINA DEVI, your appointment with DR. ARUN SEN is confirmed on  07 Jun, 2026 at 04:04 PM SL- 12":
            "Dear [NAME], your appointment with DR. [NAME] is confirmed on [DATE] at [TIME] SL- [NUM]",
        "Dear Sir/Madam, Reports of UHID No 999915691 and Lab No. 9992588 are Ready. Download from https://akhil.populardiagnostic.com/":
            "Dear Sir/Madam, Reports of UHID No [NUM] and Lab No. [NUM] are Ready. Download from [URL]",
        "Dear Customer, your electricity bill of Tk. 1250/- is due on 15/06/2026. Meter 55012345. Pay at www.desco.org.bd or call +8801712345678":
            "Dear Customer, your electricity bill of [AMOUNT] is due on [DATE]. Meter [NUM]. Pay at [URL] or call [PHONE]",
        "আপনার 01712345678 আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে 12-Jun-26 । বর্তমান ব্যালেন্স ৳ ২৫০.৫০ ।":
            "আপনার [PHONE] আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে [DATE] । বর্তমান ব্যালেন্স [AMOUNT] ।",
        "[Pathao] COD ৳450 হোল্ড পার্সেল রিসিভ করতে যোগাযোগ 01799995238": "[Pathao] COD [AMOUNT] হোল্ড পার্সেল রিসিভ করতে যোগাযোগ [PHONE]",
        "Your Daraz order #DZ8A93KD72 will be delivered on 08-Jun-26. Contact: rider@daraz.com.bd": "Your Daraz order #[REF] will be delivered on [DATE]. Contact: [EMAIL]",
        "Dear Mr.Anil Kumar Welcome to LABAID LIMITED (DIAGNOSTICS), SYLHET. Your UHID is 999990071.": "Dear Mr. [NAME] Welcome to LABAID LIMITED (DIAGNOSTICS), SYLHET. Your UHID is [NUM].",
        "Seat: L F2 Coach: 12 Rs-450 Hotline: 16216 Date: 2026-06-10 07:30 AM": "Seat: L F[NUM] Coach: [NUM] [AMOUNT] Hotline: [NUM] Date: [DATE] [TIME]",
        "Hello Rahim, you have received 500 taka from 01812345678. TrxID 9G7H2K1L3M": "Hello [NAME], you have received [AMOUNT] from [PHONE]. TrxID [REF]",
        "you received 500 taka and 1250/- and 450 Tk today": "you received [AMOUNT] and [AMOUNT] and [AMOUNT] today",
        "Dear Customer, Dear Sir/Madam, Dear Valued Member, hello world": "Dear Customer, Dear Sir/Madam, Dear Valued Member, hello world",
        "Dear Prof. Dr. Md. Karim Uddin, Thank you": "Dear Prof. Dr. Md. [NAME], Thank you",
    }
    for src, want in cases.items():
        assert mask_text(src) == want, f"\n  src : {src}\n  got : {mask_text(src)}\n  want: {want}"
    print("OK  masking: amounts, dates, times, numbers, phones, ids, links, e-mails, names; fixed wording kept (", len(cases), "cases )")


def test_messages_of_one_template_get_one_masked_text():
    a = "TK 5,000.00 withdrawn from A/C#12345**6789 on 07-06-26 @09:05 AM from MIRPUR 10 ATM DHAKA BD. Balance TK 7,949.31. BRAC Bank."
    b = "TK 18,500.00 withdrawn from A/C#10000**6001 on 06-06-26 @08:34 PM from MIRPUR 10 ATM DHAKA BD. Balance TK 2,949.31. BRAC Bank."
    assert mask_text(a) == mask_text(b), "same template, different values -> one masked text"
    assert mask_text(WITHDRAW) != mask_text(WITHDRAW.replace("withdrawn from", "credited to")), "different wording stays different"
    assert mask_text(a) != mask_text(a.replace("MIRPUR 10 ATM", "UTTARA ATM")), "places are not masked (they are wording): a different ATM is a different text"
    print("OK  values that change collapse to one text; different wording / places stay different")


def test_edge_cases_and_stability():
    for blank in ["", "   ", "\n\t", None, 12345, float("nan"), ["x"]]:
        assert mask_text(blank) == "", repr(blank)
    assert mask_text("[NUM] and [AMOUNT] already masked") == "[NUM] and [AMOUNT] already masked", "placeholders are never re-masked"
    samples = [WITHDRAW, "Dear Mr.Anil Kumar Welcome to LABAID", "Dear RINA DEVI, your appointment with DR. ARUN SEN", "Hello Rahim, got 500 taka from 01812345678 TrxID 9G7H2K1L3M",
               "Dear Prof. Dr. Md. Karim Uddin, Thank you", "Dear Mr. Sir", "Seat: L F2 Coach: 12 Rs-450", "আপনার ০১৭১২৩৪৫৬৭৮ আইডির ৳ ২৫০.৫০"]
    assert all(mask_text(mask_text(s)) == mask_text(s) for s in samples), "masking a masked text changes nothing"
    assert mask_text("৳ ২৫০.৫০ ও ১২৩৪৫৬") == "[AMOUNT] ও [NUM]", "Bangla digits are converted before masking"
    assert "OTP" in mask_text("Your OTP is 123456") and "bill" in mask_text("Your bill 500 is due") and "withdrawn" in mask_text(WITHDRAW), "category words survive"
    for junk in ["\x00\x01\x02", "ŧ" * 500, "9" * 5000, "a1" * 3000, "[" * 200 + "]" * 200, "1 " * 3000, ("word1 " * 2000)]:
        t0 = time.time(); r = mask_text(junk); assert isinstance(r, str) and time.time() - t0 < 2, repr(junk[:20])
    t0 = time.time(); [mask_text(f"TK {i},500.00 on 0{i % 9 + 1}-06-26 @0{i % 9 + 1}:15 PM Balance {i}.50 A/C#{i}0012**60{i % 99:02d}") for i in range(20000)]
    assert time.time() - t0 < 15, "20,000 messages in a few seconds"
    print("OK  blank / non-text -> '', no re-masking, idempotent, Bangla digits, category words kept, no hang on junk, fast")


def test_prepare_messages_creates_the_masked_text():
    hexed = "TK 500.00 sent to 01712345678".encode("utf-16-be").hex().upper()
    df = pd.DataFrame({"rid": [1, 2, 3, 4, 5], "msg_body": [WITHDRAW, hexed, None, "  ", "Dear Customer, bill 500 due"]})
    before = df.copy(deep=True)
    out = prepare_messages(df, "msg_body")
    assert df.equals(before), "input not modified"
    assert list(out.columns) == ["rid", "msg_body", "message_decoded", "original_language_flag", "message_masked"]
    assert out.loc[1, "message_decoded"] == "TK 500.00 sent to 01712345678" and out.loc[1, "message_masked"] == "[AMOUNT] sent to [PHONE]", "hex is decoded first, then masked"
    assert out.loc[0, "message_masked"] == mask_text(WITHDRAW) and out.loc[0, "msg_body"] == WITHDRAW
    assert pd.isna(out.loc[2, "message_masked"]) and pd.isna(out.loc[3, "message_masked"]), "blank messages have no masked text"
    assert "message_masked" not in prepare_messages(df, "msg_body", mask=False).columns
    assert add_masked_text(out.drop(columns="message_masked")).equals(out)
    print("OK  prepare_messages: decode first, then masked text; blanks stay empty; mask=False skips it")


class CountingModel(SMSClassifier):
    """Stands in for the LLM and records every text it is asked to classify."""
    asked: list = []
    def _first_token_logprobs(self, texts):
        CountingModel.asked.extend(texts)
        k = {c: l for l, c in self.prompt.key_to_cat.items()}
        return [({k["Transactional (Banking)"] if "withdrawn" in t else k["OTP / Verification Code"]: -0.01}, "") for t in texts]


def test_the_model_classifies_each_distinct_template_once():
    rows = [WITHDRAW.replace("18,500.00", f"{i},000.00").replace("6001", f"60{i:02d}").replace("06-06-26", f"0{i % 9 + 1}-06-26") for i in range(1, 501)]
    rows += [f"Your OTP is {400000 + i}. Do not share." for i in range(1, 301)]
    df = pd.DataFrame({"rid": range(len(rows)), "msg_body": rows})
    assert df.msg_body.nunique() == 800, "all 800 raw texts are different"
    CountingModel.asked = []
    prepared = prepare_messages(df, "msg_body")
    assert prepared.message_masked.nunique() == 2
    res = classify(prepared, "message_masked", classifier=CountingModel(), show_progress=False)
    assert len(CountingModel.asked) == 2, f"the model must see 2 templates, not 800: {len(CountingModel.asked)}"
    assert (res.loc[:499, "Predicted_Category"] == "Transactional (Banking)").all() and (res.loc[500:, "Predicted_Category"] == "OTP / Verification Code").all()
    CountingModel.asked = []
    classify(prepared, "message_decoded", classifier=CountingModel(), show_progress=False)
    assert len(CountingModel.asked) == 800, "classifying the raw text costs one model call per distinct text"
    print("OK  800 messages of 2 templates -> the model is called 2 times (raw text: 800); every message gets its template's category")


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        fn()
    print("\nALL MASKING TESTS PASSED")
