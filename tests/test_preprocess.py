import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import os, re, tempfile
import numpy as np, pandas as pd
import time as _time0
_REAL_SLEEP = _time0.sleep        # keep the real one for the throttle test
from sms_profiling import preprocess as T

T.time.sleep = lambda s: None            # no real waiting in the retry loop
import warnings

def hexenc(s): return s.encode("utf-16-be").hex().upper()

BN1 = "আপনার 01712345678 নম্বরে ৳ 5000 জমা হয়েছে। TrxID AB12CD34EF https://example.com/AbC123 মেইল user@mail.com"
BN2 = "প্যাকের মেয়াদ শেষ হয়েছে। বর্তমান ব্যালেন্স"
MIXED = "[Pathao] Product collected থেকে পণ্য সংগ্রহ হয়েছে"
EN = "Your OTP is 482915. Do not share."

calls = []
def fake(text):
    calls.append(text); return "EN<" + text + ">"

df = pd.DataFrame({"msg_body": [hexenc(BN1), EN, hexenc(BN1), None, "   ", BN2, hexenc("Hello from hex 1234"), MIXED, hexenc(BN2), "12345678", 7],
                   "keep": range(11)}, index=[3, 3, 4, 4, 1, 2, 2, 8, 9, 9, 0])
orig = df.copy(deep=True)
r = T.add_english_translation(df, "msg_body", translator=fake, show_progress=False, max_workers=3)

assert df.equals(orig), "input modified"
assert list(r.index) == list(df.index) and list(r["keep"]) == list(df["keep"]) and r["msg_body"].equals(df["msg_body"]), "rows/original column changed"
nn = lambda xs: [None if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in xs]   # pandas 3 stores missing as NaN
fl = nn(r["original_language_flag"].tolist()); st = nn(r["translation_status"].tolist()); en = nn(r["message_english"].tolist())
print("flags :", fl); print("status:", st)
assert fl == ["bangla", "english", "bangla", None, None, "bangla", "english", "bangla", "bangla", "english", "english"], fl
assert st == ["translated", "not_needed", "translated", None, None, "translated", "not_needed", "translated", "translated", "not_needed", "not_needed"], st
assert en[1] == EN and en[6] == "Hello from hex 1234", "English rows (incl. decoded hex English) must be unchanged"
assert en[9] == "12345678" and en[10] == "7", "digit-only strings are not hex text"
assert en[3] is None and en[4] is None
assert en[0] == en[2], "duplicate rows share one translation"
assert en[5] == en[8].replace("\n", "") or en[5] == en[8], "plain Bangla and the same text sent as hex give the same result"
print("OK  hex decoding, language flags, status, English untouched, original column untouched")

sent = "\n".join(calls)
assert len(calls) == 3, f"expected 3 distinct Bangla texts sent (BN1, BN2, MIXED), got {len(calls)}: {calls}"
assert not re.search(r"[0-9০-৯]{4,}", sent), "long digit run leaked to the translator"
assert "01712345678" not in sent and "AB12CD34EF" not in sent and "user@mail.com" not in sent and "example.com" not in sent
assert all(x in sent for x in ("[NUM]", "[CODE]", "[URL]", "[EMAIL]")), sent
assert "482915" not in sent and "Hello from hex" not in sent, "English rows must never be sent"
print("OK  masking: translator never saw digits>=4 / codes / links / e-mails; English rows never sent;", len(calls), "distinct texts sent for", int((np.array(fl) == 'bangla').sum()), "Bangla rows")

# ---- mask=False sends the text as it is (explicit opt-out)
calls.clear()
T.add_english_translation(df.iloc[[0]], "msg_body", translator=fake, mask=False, show_progress=False)
assert "01712345678" in calls[0] and "https://example.com/AbC123" in calls[0]
print("OK  mask=False sends unmasked text (only when you ask for it)")

# ---- failures: translator raises / returns empty -> status failed, Bangla kept, other rows still translated
def flaky(text):
    if "প্যাকের" in text: raise RuntimeError("429 too many requests")
    return "EN<" + text + ">"
rf = T.add_english_translation(df, "msg_body", translator=flaky, show_progress=False)
assert rf["translation_status"].tolist()[5] == "failed" and rf["message_english"].tolist()[5] == BN2, "failed row keeps its Bangla"
assert rf["translation_status"].tolist()[0] == "translated", "other rows unaffected by one failure"
re_ = T.add_english_translation(df.iloc[[5]], "msg_body", translator=lambda t: "   ", show_progress=False)
assert re_["translation_status"].iloc[0] == "failed", "empty translation counts as failure"
attempts = []
def count(t): attempts.append(1); raise RuntimeError("x")
T.add_english_translation(df.iloc[[5]], "msg_body", translator=count, show_progress=False)
assert len(attempts) == 3, f"3 attempts expected, got {len(attempts)}"
print("OK  failure handling: retried 3x, row marked failed and keeps its Bangla, the rest still translated")

# ---- checkpoint: second run sends nothing; the file holds no original digits / codes
with tempfile.TemporaryDirectory() as d:
    cp = os.path.join(d, "tr.csv"); calls.clear()
    a = T.add_english_translation(df, "msg_body", translator=fake, checkpoint_path=cp, show_progress=False)
    n1 = len(calls); calls.clear()
    b = T.add_english_translation(df, "msg_body", translator=fake, checkpoint_path=cp, show_progress=False)
    assert n1 == 3 and len(calls) == 0, (n1, len(calls))
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    txt = open(cp, encoding="utf-8").read()
    assert "01712345678" not in txt and "AB12CD34EF" not in txt and "user@mail.com" not in txt
print("OK  checkpoint resume (0 requests on the 2nd run); checkpoint holds only masked text")

# ---- decode_hex_sms edge cases
d = T.decode_hex_sms
assert d("12345678") == "12345678", "digit string is not text"
assert d("0123456789abcdef") == "0123456789abcdef", "random hex is not text"
assert d("ABC") == "ABC" and d("Hello world") == "Hello world", "odd length / not hex"
assert d(hexenc("Hello")) == "Hello" and d(hexenc(BN2)) == BN2
assert d(hexenc("1234")) == hexenc("1234"), "UCS-2 digits only has no letter: left as is (ambiguous)"
assert d("00000000") == "00000000"
assert d(hexenc("Bonus \U0001F389 offer")) == "Bonus \U0001F389 offer", "emoji (surrogate pairs) decode"
print("OK  hex edge cases (digits, random hex, odd length, emoji)")

# ---- other
assert T.add_english_translation(df.iloc[0:0], "msg_body", translator=fake, show_progress=False).shape[0] == 0
for bad, exc in [(pd.DataFrame({"a": [1]}), KeyError), ([1], TypeError)]:
    try: T.add_english_translation(bad, "msg_body", translator=fake); raise SystemExit("no error")
    except exc: pass
assert T.language_flag("প্যাকের pack") == "bangla" and T.language_flag("Tk 500 debited") == "english"
print("OK  empty df, clear errors, mixed Bangla+English counts as Bangla")

# ---- rate limiting: stop switch + long backoff, and no hammering
class TooManyRequests(Exception): pass
many = pd.DataFrame({"msg_body": [f"বার্তা নম্বর {c} ব্যালেন্স" for c in "ক খ গ ঘ ঙ চ ছ জ ঝ ঞ".split()]})
n_calls = []
def refused(t): n_calls.append(t); raise TooManyRequests("Server Error: You made too many requests to the server")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    rr = T.add_english_translation(many, "msg_body", translator=refused, show_progress=False, max_workers=1)
assert (rr["translation_status"] == "failed").all(), "all failed"
assert len(n_calls) == T.GIVE_UP_AFTER * 3, f"stop switch: {T.GIVE_UP_AFTER} texts x 3 attempts only, got {len(n_calls)}"
assert any("rate limiting" in str(x.message) for x in w), "user must be told why it stopped"
print(f"OK  rate limited: gave up after {len(n_calls)} requests instead of {len(many) * 3}, warned, rows kept as failed")

ok_after = {"n": 0}
def recovers(t):
    ok_after["n"] += 1
    if ok_after["n"] <= 2: raise TooManyRequests("429 too many requests")
    return "EN<" + t + ">"
rc = T.add_english_translation(many.iloc[:3], "msg_body", translator=recovers, show_progress=False, max_workers=1)
assert (rc["translation_status"] == "translated").all(), "a temporary 429 is retried and succeeds"
print("OK  a temporary 429 is retried (with long waits) and the translation succeeds")

sleeps = []
T.time.sleep = sleeps.append
try: T._translate_with_retry(refused, "x")
except T.RateLimited: pass
T.time.sleep = lambda s: None
assert sleeps == [15.0, 30.0, 45.0], sleeps
print("OK  backoff after 429: waits 15s, 30s, 45s")

# throttle: 2 requests/second shared by all threads
import time as _t, threading as _th
T.time.sleep = _REAL_SLEEP
th = T._Throttle(50); starts = []
def hit(): th.wait(); starts.append(_t.monotonic())
ts = [_th.Thread(target=hit) for _ in range(6)]; [x.start() for x in ts]; [x.join() for x in ts]
starts.sort(); gaps = [b - a for a, b in zip(starts, starts[1:])]
assert min(gaps) >= 0.015, gaps          # 50/s -> >= ~20ms apart even from 6 threads at once
print("OK  throttle spaces requests apart across threads")
# ---- decode-only is the default and never touches the translation service
def must_not_be_called(text): raise AssertionError("translation service was called in decode-only mode")
orig_g = T.google_bn_to_en; T.google_bn_to_en = must_not_be_called
dd = T.prepare_messages(df, "msg_body")
T.google_bn_to_en = orig_g
assert "message_english" not in dd.columns and "translation_status" not in dd.columns, "decode-only adds no translation columns"
assert list(dd.index) == list(df.index) and df.equals(orig), "index/input changed"
dec = nn(dd["message_decoded"].tolist()); fl2 = nn(dd["original_language_flag"].tolist())
assert dec[0] == BN1 and dec[2] == BN1 and dec[5] == BN2 and dec[8] == BN2 and dec[6] == "Hello from hex 1234", dec
assert dec[3] is None and dec[4] is None and dec[9] == "12345678" and dec[10] == "7"
assert fl2 == ["bangla", "english", "bangla", None, None, "bangla", "english", "bangla", "bangla", "english", "english"], fl2
print("OK  prepare_messages default = decode + flag only; translation service never called; blank rows empty")
tr = T.prepare_messages(df, "msg_body", translate=True, translator=fake, show_progress=False)
assert {"message_decoded", "original_language_flag", "message_english", "translation_status"} <= set(tr.columns)
print("OK  prepare_messages(translate=True) adds the optional translation columns")
assert T.decode_messages(df.iloc[0:0], "msg_body").shape[0] == 0
for bad, exc in [(pd.DataFrame({"a": [1]}), KeyError), ([1], TypeError)]:
    try: T.prepare_messages(bad, "msg_body"); raise SystemExit("no error")
    except exc: pass
print("OK  decode-only: empty df and clear errors")
# ---- decoded text is Unicode-normalized: Bengali য় as U+09DF and as U+09AF+U+09BC must become the same text
PRE, DEC = "\u09df", "\u09af\u09bc"
pair = pd.DataFrame({"msg_body": ["মেয" + PRE.replace(PRE, "") + "াদ " + PRE, hexenc("মে" + PRE + "াদ"), "মে" + DEC + "াদ", "মে" + PRE + "াদ"]})
out = T.decode_messages(pair, "msg_body")
dec = [x for x in out["message_decoded"]]
assert dec[1] == dec[2] == dec[3], [[hex(ord(c)) for c in x] for x in dec[1:]]
assert out["msg_body"].tolist() == pair["msg_body"].tolist(), "the original text is never changed"
assert PRE not in dec[1] and DEC in dec[1]
print("OK  message_decoded is NFC-normalized: the two encodings of Bengali য় (hex-encoded or plain) give the same text; msg_body is untouched")
print("\nALL TRANSLATOR TESTS PASSED")
