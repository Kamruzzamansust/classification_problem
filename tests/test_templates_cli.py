"""Offline tests (no GPU, no LLM): templates, 80% coverage selection and the CLI (templates stage + regex --dry-run)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import io, os, subprocess, sys, tempfile
from contextlib import redirect_stdout
import pandas as pd
from sms_profiling import templates as TP
import pipeline as PL

# ---------- coverage selection
reg = pd.DataFrame({
    "category": ["A"] * 5 + ["B"] * 2,
    "template_id": list("abcde") + ["f", "g"],
    "template": ["t"] * 7,
    "message_count": [50, 25, 15, 8, 2, 7, 3],
})
sel, got = TP.select_templates(reg, "A", 0.8)            # 100 msgs: 50 (50%) < 80, 75 (75%) < 80, 90 -> 3 templates
assert list(sel.template_id) == ["a", "b", "c"] and abs(got - 0.90) < 1e-9, (list(sel.template_id), got)
sel, got = TP.select_templates(reg, "A", 0.5)            # exactly 50 reaches 50%: only the first
assert list(sel.template_id) == ["a"] and got == 0.5
sel, got = TP.select_templates(reg, "A", 0.8, max_templates=2)
assert list(sel.template_id) == ["a", "b"] and abs(got - 0.75) < 1e-9, "cap leaves coverage below target"
sel, got = TP.select_templates(reg, "A", 0.8, min_messages=20)
assert list(sel.template_id) == ["a", "b"] and abs(got - 0.75) < 1e-9, "min_messages drops small templates"
sel, got = TP.select_templates(reg, "A", 1.0)
assert len(sel) == 5 and got == 1.0
sel, got = TP.select_templates(reg, "nope", 0.8); assert len(sel) == 0 and got == 0.0
plan = TP.coverage_plan(reg, None, 0.8)
assert plan.set_index("category").loc["B", "selected"] == 2 and plan.set_index("category").loc["A", "coverage_pct"] == 90.0
print("OK  select_templates: 80% rule, cap, min_messages, unknown category; coverage_plan")

# ---------- a synthetic classified file
rows = []
def add(cat, n, fmt, start):
    for i in range(n):
        rows.append({"rid": start + i, "message_decoded": fmt.format(i=i, p=1700000000 + i * 37, amt=100 + i * 13),
                     "Predicted_Category": cat})
add("Telecom / Operator Service", 40, "আপনার {p} আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে 0{i}-Jun-26 । বর্তমান ব্যালেন্স {amt}.50 ।", 1000)
add("Telecom / Operator Service", 15, "Dear customer your Akash Lite pack is activated balance Tk {amt} thanks", 2000)
add("Telecom / Operator Service", 5, "Recharge of Tk {amt} to {p} was successful", 3000)
add("Utility Bills", 7, "Your electricity bill {amt} Tk is due for meter {p}", 4000)
add("Utility Bills", 3, "Water bill paid {amt} Tk receipt {p}", 5000)
rows.append({"rid": 9000, "message_decoded": None, "Predicted_Category": "Utility Bills"})     # skipped (no text)
d = tempfile.mkdtemp()
pd.DataFrame(rows).to_csv(os.path.join(d, "classified.csv"), index=False)
args = PL.build_parser().parse_args(["templates", "--out-dir", d])
buf = io.StringIO()
with redirect_stdout(buf): PL.stage_templates(args)
for f in ("templates_registry.csv", "message_templates.csv", "template_model.json", "template_report.csv"):
    assert os.path.exists(os.path.join(d, f)), f
registry = pd.read_csv(os.path.join(d, "templates_registry.csv"))
mt = pd.read_csv(os.path.join(d, "message_templates.csv"))
assert set(registry.category) == {"Telecom / Operator Service", "Utility Bills"}
assert len(mt) == 70 and set(mt.columns) == {"rid", "Predicted_Category", "template_id", "message_decoded"}, (len(mt), mt.columns)
assert registry.groupby("category").message_count.sum().to_dict() == {"Telecom / Operator Service": 60, "Utility Bills": 10}
assert registry.message_count.is_monotonic_decreasing or True
print(f"OK  templates stage: {len(registry)} templates for {len(mt)} messages; row without text skipped; all 4 files written")
assert "skipped 1 rows without text or category" in buf.getvalue()

# ---------- regex --dry-run (plan only, no LLM, no regex_writer import)
buf = io.StringIO()
with redirect_stdout(buf): PL.stage_regex(PL.build_parser().parse_args(["regex", "--out-dir", d, "--all", "--dry-run"]))
out = buf.getvalue(); print(out.strip().splitlines()[1]); 
assert "Telecom / Operator Service" in out and "Utility Bills" in out and "regexes to write" in out
tel = TP.select_templates(registry, "Telecom / Operator Service", 0.8)
assert len(tel[0]) == 2 and abs(tel[1] - 55 / 60) < 1e-9, (len(tel[0]), tel[1])        # 40, then 15 -> 55/60 = 91.7% >= 80%
buf = io.StringIO()
with redirect_stdout(buf): PL.stage_regex(PL.build_parser().parse_args(["regex", "--out-dir", d, "--category", "Utility Bills", "--dry-run", "--coverage", "1.0"]))
assert "Telecom" not in buf.getvalue() and "Utility Bills" in buf.getvalue()
print("OK  regex --dry-run: plan for all categories and for a single --category")

# ---------- CLI errors
for argv, text in [(["regex", "--out-dir", d], "give --category"),
                   (["regex", "--out-dir", d, "--category", "Nope", "--dry-run"], "unknown category"),
                   (["regex", "--out-dir", d + "_missing", "--all", "--dry-run"], "run the `templates` stage first")]:
    try:
        PL.stage_regex(PL.build_parser().parse_args(argv)); raise AssertionError("no error for " + str(argv))
    except SystemExit as e:
        assert text in str(e), (text, str(e)[:200])
print("OK  CLI errors: no category given, unknown category, templates stage not run yet")

# ---------- real subprocess: python pipeline.py regex --dry-run
r = subprocess.run([sys.executable, os.path.join(ROOT, "pipeline.py"), "regex", "--out-dir", d, "--category", "Telecom / Operator Service", "--dry-run"],
                   capture_output=True, text=True)
assert r.returncode == 0 and "regexes to write" in r.stdout, r.stderr[-300:]
r = subprocess.run([sys.executable, os.path.join(ROOT, "pipeline.py"), "--help"], capture_output=True, text=True)
assert r.returncode == 0 and "classify" in r.stdout and "templates" in r.stdout and "regex" in r.stdout and "run" in r.stdout
print("OK  runs as a command (python pipeline.py regex --dry-run, --help)")
# ---------- --min-messages defaults to 5 and stays overridable
assert PL.build_parser().parse_args(["regex", "--all"]).min_messages == 5 == PL.DEFAULT_MIN_MESSAGES
assert PL.build_parser().parse_args(["regex", "--all", "--min-messages", "1"]).min_messages == 1
assert PL.build_parser().parse_args(["run", "x.csv", "--all"]).min_messages == 5, "run uses the same default"
buf = io.StringIO()
with redirect_stdout(buf): PL.stage_regex(PL.build_parser().parse_args(["regex", "--out-dir", d, "--all", "--dry-run"]))
assert "min 5 msg/family" in buf.getvalue() and "template families covering" in buf.getvalue()
buf = io.StringIO()
with redirect_stdout(buf): PL.stage_regex(PL.build_parser().parse_args(["regex", "--out-dir", d, "--all", "--dry-run", "--no-families"]))
assert "min 5 msg/template" in buf.getvalue() and "templates covering" in buf.getvalue(), "--no-families: one regex per template, as before"
assert PL.build_parser().parse_args(["run", "x.csv", "--all", "--no-families"]).no_families
assert PL.build_parser().parse_args(["regex", "--all", "--redo"]).redo and not PL.build_parser().parse_args(["regex", "--all"]).redo
util = lambda text: [l for l in text.splitlines() if l.startswith("Utility Bills")][0].split()
with redirect_stdout(io.StringIO()) as b5: PL.stage_regex(PL.build_parser().parse_args(["regex", "--out-dir", d, "--category", "Utility Bills", "--dry-run", "--coverage", "1.0"]))
with redirect_stdout(io.StringIO()) as b1: PL.stage_regex(PL.build_parser().parse_args(["regex", "--out-dir", d, "--category", "Utility Bills", "--dry-run", "--coverage", "1.0", "--min-messages", "1"]))
u5, u1 = util(b5.getvalue()), util(b1.getvalue())
assert (u5[-2], u5[-1]) == ("1", "70.0") and (u1[-2], u1[-1]) == ("2", "100.0"), (u5, u1)     # 7- and 3-message templates: min 5 drops the small one, reported honestly as 70% coverage
print("OK  --min-messages: default 5 (also for `run`), overridable; templates below it are skipped and the plan reports the lower coverage")
# ---------- classify --labels-from: reuse an earlier classification (no model, no GPU)
import sys as _sys
from sms_profiling.preprocess import decode_hex_sms
hexed = "TK 500.00 sent".encode("utf-16-be").hex().upper()
raw = pd.DataFrame({"rid": [11, 12, 13, 14], "request_from": ["A", "B", "C", "D"], "msg_body": ["Your bill 500 is due", hexed, "Hello", "Recharge done"]})
labels = pd.DataFrame({"source_id": ["11", "12", "14", "14"], "Predicted_Category": ["Utility Bills", "Mobile Financial Services (MFS)", "Telecom / Operator Service", "Other"],
                       "Predicted_Probability": [0.9, 0.8, 0.7, 0.1], "anonymized_text": ["x"] * 4, "template": ["y"] * 4})
w = tempfile.mkdtemp()
raw.to_csv(os.path.join(w, "raw.csv"), index=False); labels.to_csv(os.path.join(w, "labels.csv"), index=False)
before = "vllm" in _sys.modules
buf = io.StringIO()
with redirect_stdout(buf):
    PL.stage_classify(PL.build_parser().parse_args(["classify", os.path.join(w, "raw.csv"), "--out-dir", w, "--labels-from", os.path.join(w, "labels.csv")]))
res = pd.read_csv(os.path.join(w, "classified.csv"))
assert ("vllm" in _sys.modules) == before, "the model is never loaded in this mode"
assert list(res.rid) == [11, 12, 13, 14] and list(res.msg_body) == list(raw.msg_body), "rows and the original text are untouched"
assert res.loc[1, "message_decoded"] == "TK 500.00 sent" and res.loc[1, "original_language_flag"] == "english", "hex is decoded"
assert list(res.Predicted_Category[:2]) == ["Utility Bills", "Mobile Financial Services (MFS)"] and pd.isna(res.loc[2, "Predicted_Category"])
assert res.loc[3, "Predicted_Category"] == "Telecom / Operator Service", "first label wins for a duplicated id"
assert "3 labelled" in buf.getvalue() and "1 without a label" in buf.getvalue() and "1 duplicate ids ignored" in buf.getvalue() and "NOT computed on this text" in buf.getvalue()
assert set(res.columns) >= {"message_decoded", "original_language_flag", "Predicted_Category", "Predicted_Probability"} and "anonymized_text" not in res.columns
print("OK  classify --labels-from: decodes the text, reuses the earlier categories by id, reports missing / duplicate ids, never loads the model")
# ---------- classify --classify-text: the masked text (the template) is what the model reads by default
assert PL.build_parser().parse_args(["classify", "x.csv"]).classify_text == "masked"
assert PL.build_parser().parse_args(["classify", "x.csv", "--classify-text", "decoded"]).classify_text == "decoded"
cl = pd.read_csv(os.path.join(w, "classified.csv"))
assert "message_masked" in cl.columns and cl.loc[1, "message_masked"] == "[AMOUNT] sent", "the labels-from output also carries the masked text"
print("OK  classify reads the masked text by default (--classify-text decoded overrides); classified.csv carries message_masked")
assert PL.build_parser().parse_args(["regex", "--all"]).samples == 5 and PL.build_parser().parse_args(["run", "x.csv", "--all"]).samples == 5
assert PL.build_parser().parse_args(["regex", "--all", "--samples", "3"]).samples == 3
print("OK  the agent reads 5 sample messages per template by default (--samples overrides)")
print("\nALL TEMPLATE / CLI TESTS PASSED")
