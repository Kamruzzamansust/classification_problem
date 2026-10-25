import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import math, os, tempfile
import numpy as np, pandas as pd
from sms_profiling import classifier as C

class Fake(C.SMSClassifier):
    """Stands in for vLLM: picks an option letter from keywords in the message."""
    calls = 0
    def _first_token_logprobs(self, texts):
        Fake.calls += len(texts)
        k = {c: l for l, c in self.prompt.key_to_cat.items()}
        out = []
        for t in texts:
            if "otp" in t.lower():   raw = {k["OTP / Verification Code"]: -0.01, k["Spam / Fraud / Scam"]: -6.0}
            elif "bank" in t.lower(): raw = {k["Transactional (Banking)"]: -0.05, k["Mobile Financial Services (MFS)"]: -3.5}
            elif "unsure" in t.lower(): raw = {l: -3.0 for l in list(self.prompt.key_to_cat)[:5]}   # flat -> low prob
            elif "none" in t.lower(): raw = {}                                                      # no valid letter
            else: raw = {k["Other"]: -0.02}
            out.append((raw, ""))
        return out

clf = Fake()
df = pd.DataFrame({
    "msg_body": ["Your OTP is 1234", "Bank: Tk 500 debited", "Your OTP is 1234", None, "   ", "unsure thing", "none here", "hello", 42, "  Bank: Tk 500 debited  "],
    "other": list(range(10)),
}, index=[5, 5, 7, 7, 1, 2, 3, 3, 9, 9])          # NON-unique index on purpose
orig = df.copy(deep=True)
res = C.classify(df, "msg_body", classifier=clf, show_progress=False)

assert df.equals(orig), "input df was modified"
assert len(res) == len(df) and list(res.index) == list(df.index), "rows/index changed"
assert list(res["other"]) == list(df["other"]), "order changed"
pc = res["Predicted_Category"].tolist()
print("categories:", pc)
assert pc[0] == "OTP / Verification Code" and pc[2] == pc[0], "duplicate text must get the same label"
assert pc[1] == "Transactional (Banking)" and pc[9] == pc[1], "strip() should make ' Bank..' identical to 'Bank..'"
assert pd.isna(pc[3]) and pd.isna(pc[4]), "None / blank text -> empty (None or NaN, depending on pandas version)"
assert pc[5] == "Other", "flat distribution (<0.40) must become Other"
assert res.loc[res.index[5], "Model_Choice"] != "Other", "Model_Choice keeps the raw argmax"
assert pc[6] == "Other", "no valid letter -> Other"
assert pc[7] == "Other" and pc[8] == "Other", "other + numeric text"
assert np.isnan(res["Predicted_Probability"].iloc[3]) and not np.isnan(res["Predicted_Probability"].iloc[0])
probs = res[[c for c in res.columns if c.startswith("Prob_")]]
assert probs.shape[1] == 21, probs.shape
s = probs.dropna().sum(axis=1); assert ((s - 1).abs() < 0.01).all(), s
assert Fake.calls == 6, f"unique messages classified once each, got {Fake.calls}"   # otp, bank, unsure, none, hello, 42
print("OK  duplicates/blank/NaN/index/Other rules, unique-once ({} model calls for {} rows)".format(Fake.calls, len(df)))

r2 = C.classify(df, "msg_body", classifier=clf, probabilities=False, show_progress=False)
assert not any(c.startswith("Prob_") for c in r2.columns); print("OK  probabilities=False")
assert C.classify(df.iloc[0:0], "msg_body", classifier=clf, show_progress=False).shape[0] == 0; print("OK  empty df")
for bad, exc in [(pd.DataFrame({"a": [1]}), KeyError), ([1, 2], TypeError)]:
    try: C.classify(bad, "msg_body", classifier=clf); raise SystemExit("no error")
    except exc as e: pass
print("OK  clear errors for missing column / non-DataFrame")

# checkpoint: second run must not call the model at all, and store no message text
with tempfile.TemporaryDirectory() as d:
    cp = os.path.join(d, "cp.csv"); Fake.calls = 0
    a = C.classify(df, "msg_body", classifier=clf, checkpoint_path=cp, chunk_size=2, show_progress=False)
    first_calls = Fake.calls; Fake.calls = 0
    b = C.classify(df, "msg_body", classifier=clf, checkpoint_path=cp, chunk_size=2, show_progress=False)
    assert first_calls == 6 and Fake.calls == 0, (first_calls, Fake.calls)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    raw = open(cp, encoding="utf-8").read()
    assert "OTP is" not in raw and "Bank:" not in raw, "checkpoint must not contain message text"
print("OK  checkpoint resume (0 model calls on 2nd run) and no message text in the checkpoint")

# long-message shortening (fake tokenizer: 1 token per 2 chars + 300 fixed)
class Tok:
    def apply_chat_template(self, msgs, **kw): return [1] * (len(msgs[0]["content"]) // 2)
c2 = C.SMSClassifier(max_model_len=1024)
ids = c2._build_ids("x" * 20000, tokenizer=Tok()); assert len(ids) <= 1016, len(ids)
short = c2._build_ids("hello", tokenizer=Tok()); assert len(short) > 100
print(f"OK  long message shortened to {len(ids)} tokens (limit 1016); short one untouched ({len(short)})")

# scoring parity with python_v5: same letters for the same categories
p = C.CategoryPrompt(C.DEFAULT_CATEGORIES)
assert p.key_to_cat["A"] == "E-commerce / Delivery" and p.key_to_cat["R"] == "Transactional (Banking)" and p.key_to_cat["U"] == "Other"
assert len(C.DEFAULT_CATEGORIES) == 20 and len(p.categories) == 21
print("OK  letters match python_v5 (A=E-commerce ... R=Transactional, U=Other)")
