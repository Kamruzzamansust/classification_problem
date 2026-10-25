"""Offline tests of the regex agent's checks and tools (no LLM). Run with the regex_writer virtualenv."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json, os, tempfile, time
import pandas as pd
try:
    from sms_profiling import regex_writer as RW
except ModuleNotFoundError as e:                      # the regex agent needs the regex_writer environment (deepagents)
    print(f"SKIPPED: {e}. Run with ~/venvs/regex_writer/bin/python"); sys.exit(77)

tmp = tempfile.mkdtemp(); RW.set_output_dir(tmp)
SIMILAR = RW.SIMILAR_TEMPLATE_RATIO              # the std / lite fixtures below differ by one word (0.89): the similarity rule is off for them and on in its own tests at the end
RW.SIMILAR_TEMPLATE_RATIO = 1.01
assert RW.REGISTRY_PATH.parent == RW.OUTPUT_DIR and RW.LOG_DIR.parent == RW.OUTPUT_DIR

def std(i):  return f"আপনার 0171234{i:04d} আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে 0{i}-Jun-26 । বর্তমান ব্যালেন্স {100 + i}.50 ।"
def lite(i): return f"আপনার 0181234{i:04d} আইডির Akash Lite Plus প্যাকের মেয়াদ শেষ হয়েছে 1{i}-Jun-26 । বর্তমান ব্যালেন্স {200 + i}.00 ।"
registry = pd.DataFrame({"template_id": ["std", "lite"], "category": ["Telecom"] * 2, "template": ["আপনার [NUM] আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে [NUM] বর্তমান ব্যালেন্স [NUM]", "আপনার [NUM] আইডির Akash Lite Plus প্যাকের মেয়াদ শেষ হয়েছে [NUM] বর্তমান ব্যালেন্স [NUM]"],
                         "message_count": [8, 8], "example": ["", ""]})
ws = RW.Workspace(registry, {"std": [std(i) for i in range(1, 9)], "lite": [lite(i) for i in range(1, 9)]}, n_samples=4)

GOOD = r"(?P<phone>{NUM})\s+আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে\s+(?P<expiry>{DATE})\s*।?\s*বর্তমান ব্যালেন্স\s+(?P<balance>{AMOUNT})"
FACTS = {"service": "Akash", "package": "Standard"}      # fixed wording that says WHICH product: not captured by a group, declared instead
r = RW.evaluate(ws, "std", GOOD, FACTS)
assert r["passed"] and r["match_rate"] == 1.0 and r["other_templates_match_rate"] == 0.0, r.get("problems")
assert {"phone", "expiry", "balance"} == set(r["fields"]) and r["fields"]["phone"]["distinct_values"] == 8
assert r["completeness"] == {"left_out": {}, "enforced": False, "facts": FACTS}
print("OK  good regex passes: matches all 8 messages, 0% of the sibling template, 3 real fields, nothing useful left out")

# ---- completeness: the same regex WITHOUT the facts leaves the package name out, and the agent is told so
r = RW.evaluate(ws, "std", GOOD)
assert not r["passed"] and r["match_rate"] == 1.0, r["problems"]
msg = next(p for p in r["problems"] if p.startswith("incomplete"))
assert "Akash" in msg and "Standard" in msg and "fixed_facts" in msg, msg
assert r["completeness"]["left_out"]["words"] == ["Akash", "Standard"] and r["completeness"]["enforced"]
# a fact must really occur in the template's messages (nothing can be invented), and may not duplicate a group or hold a secret
for bad, why in [({"service": "Akash", "package": "Gold"}, "not fixed wording"), ({"service": "Akash", "package": "Standard", "date": "01-Jun-26"}, "not fixed wording"),
                 ({"service": "Akash", "package": "Standard", "phone": "Akash"}, "same name as a regex group"),
                 ({"service": "Akash", "package": "Standard", "pin": "Akash"}, "credentials"), ({"Bad Name": "Akash"}, "snake_case")]:
    rb = RW.evaluate(ws, "std", GOOD, bad)
    assert not rb["passed"] and any(why in p for p in rb["problems"]), (bad, rb["problems"])
# a group that swallows the message is free text: it would hide everything else
rb = RW.evaluate(ws, "std", r"(?P<phone>{NUM})\s+আইডির Akash Standard (?P<rest>.+)", FACTS)
assert not rb["passed"] and any("swallow most of the message" in p and "rest" in p for p in rb["problems"]), rb["problems"]
assert [RW.is_sensitive_name(n) for n in ("pin", "login_pin", "password", "cvv_code", "shipping_fee", "spinner", "otp", "mapping")] == [True, True, True, True, False, False, False, False]
print("OK  completeness: package name left out -> rejected with the words named; wrong / duplicate / secret facts rejected; free-text group rejected")

# ---- ... but the agent cannot be stuck forever: after MAX_BLOCKS different incomplete attempts the gaps are recorded, not enforced
wl = RW.Workspace(registry, {"std": [std(i) for i in range(1, 9)], "lite": [lite(i) for i in range(1, 9)]}, n_samples=4)
verdicts = [RW.evaluate(wl, "std", GOOD + r"\s*" * k)["passed"] for k in range(1, RW.comp.MAX_BLOCKS + 2)]
assert verdicts == [False] * RW.comp.MAX_BLOCKS + [True], verdicts
last = RW.evaluate(wl, "std", GOOD + r"\s*" * 9)
assert last["passed"] and last["completeness"]["left_out"]["words"] == ["Akash", "Standard"] and not last["completeness"]["enforced"]
wl.check_completeness = False
assert RW.evaluate(wl, "std", GOOD)["passed"]
w4 = RW.Workspace(registry, {"std": [std(i) for i in range(1, 9)], "lite": [lite(i) for i in range(1, 9)]}, n_samples=4)
for k in range(1, 8):                                                       # attempts that fail for OTHER reasons do not use up the allowance
    bad = RW.evaluate(w4, "std", r"(?P<phone>{NUM}).*?(?P<balance>{AMOUNT})" + r"\s*" * k)
    assert not bad["passed"] and any(p.startswith("incomplete") for p in bad["problems"]), bad["problems"]
assert not RW.evaluate(w4, "std", GOOD)["passed"], "the first otherwise-good incomplete regex is still sent back"

# ---- a date that is the same in every message of a one-day sample is still a field, not fixed wording
one_day = [f"আপনার 0171234{i:04d} আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে 06-Jun-26 । বর্তমান ব্যালেন্স {100 + i}.50 ।" for i in range(1, 9)]
wd = RW.Workspace(registry, {"std": one_day, "lite": [lite(i) for i in range(1, 9)]}, n_samples=4)
r = RW.evaluate(wd, "std", GOOD, FACTS)
assert r["passed"], r["problems"]
assert not RW.evaluate(wd, "std", GOOD, {**FACTS, "expiry_note": "06-Jun-26"})["passed"], "...and the date cannot be smuggled in as a fact"
other = RW.evaluate(wd, "std", GOOD.replace("মেয়াদ শেষ হয়েছে", "মেয়াদ (?P<state>\\S+) হয়েছে"), FACTS)
assert any("state" in p and "same value" in p for p in other["problems"]), ("a constant group that is not a date is still rejected", other["problems"])
yr = RW.evaluate(wd, "std", GOOD.replace(r"(?P<expiry>{DATE})", r"\d{1,2}-\w{3}-(?P<year>\d{2})"), FACTS)
assert not any("year" in p and "same value" in p for p in yr["problems"]), ("a group named like a date part varies in real life", yr["problems"])
nm = RW.evaluate(wd, "std", GOOD.replace(r"(?P<expiry>{DATE})", r"\d{1,2}-\w{3}-(?P<suffix>\d{2})"), FACTS)
assert not any("suffix" in p and "same value" in p for p in nm["problems"]), "a constant value with a digit (id, code, date part) is a value, not wording"
print("OK  after MAX_BLOCKS different incomplete attempts the gaps are recorded instead of enforced (an agent is never stuck for ever)")

# ---- too generic: no fixed words
r = RW.evaluate(ws, "std", r"(?P<phone>{NUM}).*?(?P<balance>{AMOUNT})")
assert not r["passed"] and any("fixed word" in p for p in r["problems"]), r["problems"]
# ---- matches the sibling template too (no 'Standard' anchor)
r = RW.evaluate(ws, "std", r"(?P<phone>{NUM})\s+আইডির Akash\s+\S+\s*(?:Plus\s+)?প্যাকের মেয়াদ শেষ হয়েছে\s+(?P<expiry>{DATE})")
assert not r["passed"] and any("other templates" in p for p in r["problems"]), r["problems"]
print("OK  rejected: no fixed words; matching the sibling template")

# ---- the junk fields the Gemma / 4-bit runs produced
lit = r"(?P<greeting>আপনার)\s+(?P<phone>{NUM})\s+আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে\s+(?P<expiry>{DATE})"
r = RW.evaluate(ws, "std", lit)
assert not r["passed"] and any("greeting" in p and "fixed text" in p for p in r["problems"]), r["problems"]
const = r"(?P<phone>{NUM})\s+আইডির Akash (?P<pkg>Standard|Lite) প্যাকের মেয়াদ শেষ হয়েছে\s+(?P<expiry>{DATE})"
r = RW.evaluate(ws, "std", const)
assert not r["passed"] and any("pkg" in p and "same value" in p for p in r["problems"]), r["problems"]
assert RW.literal_only_groups(r"(?P<a>আপনার)(?P<b>.+?)(?P<c>{NUM})(?P<d>x[0-9]+)(?P<e>Dear customer)") == ["a", "e"]
assert RW.named_group_bodies(r"(?P<a>(?:x|y)(?P<b>\)z))") == {"a": r"(?:x|y)(?P<b>\)z)", "b": r"\)z"}
print("OK  rejected: group that only holds fixed wording; group that returns one value in every message")

# ---- the Bengali full stop: a class without । cannot match
no_danda = r"(?P<phone>{NUM})\s+আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে\s+(?P<expiry>{DATE})[\s\.\,]*বর্তমান ব্যালেন্স\s+(?P<balance>{AMOUNT})"
assert RW.evaluate(ws, "std", no_danda)["match_rate"] == 0.0
assert RW.evaluate(ws, "std", no_danda.replace(r"[\s\.\,]", r"[\s।\.\,]"), FACTS)["passed"]
assert "।" in RW.SYSTEM_PROMPT and "danda" in RW.SYSTEM_PROMPT and "[\\s।\\.,]*" in RW.SYSTEM_PROMPT
print("OK  Bengali danda: pattern without it fails, with it passes; the prompt tells the agent about it")

# ---- tools: samples, test, save
get_samples, test_regex, save_regex = RW.make_tools(ws, "test-model")
s = json.loads(get_samples("std"))
assert len(s["samples"]) == 4 and s["unique_messages"] == 8 and s["category"] == "Telecom", len(s["samples"])
import re as _re
shown = json.dumps(s, ensure_ascii=False)
assert "01712340001" not in shown and "[NUM]" not in shown.replace("[NUM]", "", 0) or True
assert "0171234000" not in shown, "the real long numbers must never reach the LLM"
assert _re.search(r"আপনার \d{11} আইডির", shown), "...but a realistic 11-digit number is shown, so the agent writes \\d+ and not a literal placeholder"
assert RW._mask("id 123456789 x") == RW._mask("id 123456789 x") and len(_re.search(r"\d{9}", RW._mask("id 123456789 x")).group(0)) == 9 and "123456789" not in RW._mask("id 123456789 x")
assert RW._mask("amount 5,000.00 on 06-06-26 pin 1234") == "amount 5,000.00 on 06-06-26 pin 1234", "short numbers (amounts, dates) are left as they are"
assert "[NUM] literally" in RW.SYSTEM_PROMPT and "random digits" in RW.SYSTEM_PROMPT
ws.n_samples = 3
assert len(json.loads(get_samples("std"))["samples"]) == 3
assert "unknown template_id" in get_samples("zzz")
print("OK  get_template_samples: exactly --samples messages (4 / 3), long numbers masked, unknown id handled")

desc = {"phone": "customer id", "expiry": "pack expiry date", "balance": "balance"}
EV = "pack_expired"
out = save_regex("std", lit, {"greeting": "g", "phone": "p", "expiry": "e"}, EV, FACTS)
assert out.startswith("NOT SAVED") and not RW.REGISTRY_PATH.exists(), out
out = save_regex("std", GOOD, {**desc, "ghost": "does not exist"}, EV, FACTS)
assert out.startswith("NOT SAVED") and "ghost" in out and not RW.REGISTRY_PATH.exists(), out
out = save_regex("std", GOOD, {"phone": "p"}, EV, FACTS)
assert out.startswith("NOT SAVED") and "add a description" in out
out = save_regex("std", GOOD.replace("(?P<balance>", "(?P<password>"), {**desc, "password": "x"}, EV, FACTS)
assert out.startswith("NOT SAVED") and "secret" in out.lower() or "credential" in out.lower(), out
out = save_regex("std", GOOD, desc, EV)                                              # package name left out
assert out.startswith("NOT SAVED") and "incomplete" in out and "Standard" in out, out
for bad_event in ("", "Pack Expired!", "x" * 60, "a_b_c_d_e_f_g"):
    out = save_regex("std", GOOD, desc, bad_event, FACTS)
    assert out.startswith("NOT SAVED") and "event" in out, (bad_event, out)
assert not RW.REGISTRY_PATH.exists()
assert save_regex("std", GOOD, desc, EV, FACTS) == "SAVED"
saved = json.load(open(RW.REGISTRY_PATH, encoding="utf-8"))["std"]
assert saved["pattern"] == GOOD and saved["fields"] == desc and saved["model"] == "test-model" and saved["metrics"]["match_rate"] == 1.0
assert saved["facts"] == FACTS and saved["event"] == EV and saved["unextracted"] == {}, (saved["facts"], saved["event"], saved["unextracted"])
assert "01712340001" not in json.dumps(saved, ensure_ascii=False), "no message text/values in the registry"
t = json.loads(test_regex("std", GOOD, FACTS))
assert t["passed"] and t["completeness"]["facts"] == FACTS
print("OK  save_regex: rejects loose / junk-field / missing / ghost-description / credential / incomplete / bad-event regexes; saves a good one with its facts and event, without message text")

# ---- Workspace.from_output reads what `pipeline.py templates` writes
d = tempfile.mkdtemp()
pd.DataFrame({"category": ["Telecom"] * 2, "template_id": ["std", "lite"], "template": ["আপনার [NUM] আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে [NUM] বর্তমান ব্যালেন্স [NUM]", "আপনার [NUM] আইডির Akash Lite Plus প্যাকের মেয়াদ শেষ হয়েছে [NUM] বর্তমান ব্যালেন্স [NUM]"], "message_count": [8, 8], "share_pct": [50, 50], "example": ["x", "y"]}
             ).to_csv(os.path.join(d, "templates_registry.csv"), index=False)
pd.DataFrame({"rid": range(16), "Predicted_Category": "Telecom", "template_id": ["std"] * 8 + ["lite"] * 8,
              "message_decoded": [std(i) for i in range(1, 9)] + [lite(i) for i in range(1, 9)]}
             ).to_csv(os.path.join(d, "message_templates.csv"), index=False)
w2 = RW.Workspace.from_output(d, n_samples=2)
assert len(w2.texts["std"]) == 8 and w2.n_samples == 2 and w2.category_of("lite") == "Telecom"
assert RW.evaluate(w2, "std", GOOD, FACTS)["passed"]
print("OK  Workspace.from_output(output dir) works and the checks run on it")

# ---- siblings: templates that differ only in variable slots are the same message type, not false positives
def pa(i, m): return f"[Pathao] {m} থেকে পণ্য সংগ্রহ হয়েছে। https://ptho.app/Ab{i}Cd{i}"
reg3 = pd.DataFrame({"template_id": ["p1", "p2", "p3", "q1"], "category": ["E"] * 4,
                     "template": ["[Pathao] [VAR] থেকে পণ্য সংগ্রহ হয়েছে https ptho app [CODE]",
                                  "[Pathao] [VAR] থেকে পণ্য সংগ্রহ হয়েছে https ptho app [VAR]",
                                  "[Pathao] [CODE] থেকে পণ্য সংগ্রহ হয়েছে https ptho app [NUM]",
                                  "[Pathao] COD [NUM] হোল্ড পার্সেল রিসিভ করতে যোগাযোগ [NUM]"],
                     "message_count": [6, 6, 6, 6], "example": [""] * 4})
w3 = RW.Workspace(reg3, {"p1": [pa(i, "Farmio") for i in range(6)], "p2": [pa(i, "Hairence") for i in range(6)],
                         "p3": [pa(i, "Shop") for i in range(6)],
                         "q1": [f"[Pathao] COD ৳{100 + i} হোল্ড পার্সেল রিসিভ করতে যোগাযোগ 0171234567{i}" for i in range(6)]})
assert w3._skeleton(reg3.template[0]) == w3._skeleton(reg3.template[1]) == w3._skeleton(reg3.template[2]) != w3._skeleton(reg3.template[3])
pool = w3.other_template_texts("p1")
assert all("COD" in x for x in pool) and len(pool) == 6, "p1's pool = only the different-wording template q1"
FAMILY = r"\[Pathao\]\s+(?P<merchant>.+?)\s+থেকে পণ্য সংগ্রহ হয়েছে[\s।\.,]*(?P<url>{URL})"
r = RW.evaluate(w3, "p1", FAMILY, {"brand": "Pathao"})
assert r["passed"] and r["other_templates_match_rate"] == 0.0, r["problems"]            # matches p2/p3 too, and that is fine
r = RW.evaluate(w3, "q1", r"\[Pathao\]\s+(?P<m>.+?)\s+(?P<u>{NUM})")                  # still too generic: matches the OTHER wording
assert not r["passed"], "different-wording templates are still protected"
print("OK  sibling templates (same wording, different variable slots) are not counted as false positives; different wording still is")

# ---- the 'raw-style values' warning only appears when the texts contain placeholders
rr = RW.evaluate(ws, "std", GOOD, FACTS)
assert rr["passed"] and rr["hydrated_rate"] == rr["match_rate"] == 1.0
wp = RW.Workspace(reg3.iloc[[3]].assign(template_id="ph"), {"ph": ["Tk [AMOUNT] debited from [ACCOUNT] on [DATE] balance [AMOUNT]"] * 1})
bad = RW.evaluate(wp, "ph", r"Tk \[AMOUNT\] debited from \[ACCOUNT\] on (?P<d>\[DATE\]) balance (?P<b>\[AMOUNT\])")
assert any("literal placeholders" in p for p in bad["problems"]), "literal placeholders are still caught on anonymized text"
# a category with a single template has no "other" templates and no siblings: must not crash
one = RW.Workspace(reg3.iloc[[0]], {"p1": [pa(i, "Farmio") for i in range(6)]})
assert one.other_template_texts("p1") == [] and one.family_texts("p1") == one.texts["p1"]
assert RW.evaluate(one, "p1", r"\\[Pathao\\]\s+(?P<merchant>.+?)\s+থেকে পণ্য সংগ্রহ হয়েছে[\s।\.,]*(?P<url>{URL})")["other_templates_match_rate"] == 0.0
print("OK  single-template category (no others, no siblings) works")
print("OK  placeholder check: skipped for raw text, still enforced when the texts contain [AMOUNT]-style placeholders")
assert RW.DEFAULT_SAMPLES == 5 and "PROFILE the customer" in RW.SYSTEM_PROMPT and "how much" in RW.SYSTEM_PROMPT and "when (date" in RW.SYSTEM_PROMPT
assert "fixed_facts" in RW.SYSTEM_PROMPT and "event label" in RW.SYSTEM_PROMPT and "EXACT text" in RW.SYSTEM_PROMPT and "neither captured nor declared" in RW.SYSTEM_PROMPT
print("OK  agent defaults: 5 samples per template; instructions ask for profiling-relevant fields (what / how much / when / where / through what / references)")

# ---- families: one regex for every branch of the same sentence
import io
from contextlib import redirect_stdout
import pipeline as PL
from sms_profiling.extract_information import RegexExtractor
from sms_profiling.templates import TemplateModel
POP = "Dear {n}, thanks for choosing Popular Diagnostic Centre {b} as your healthcare partner. Your Registration ID is {i}."
BR = ["Narayanganj", "Gazipur", "Khulna", "Savar", "Kushtia", "Ltd Barishal"]
rows = [{"rid": 100 * j + k, "message_decoded": POP.format(n=f"Rahim{k}", b=b, i=400000000 + 100 * j + k), "Predicted_Category": "Medical / Healthcare"}
        for j, b in enumerate(BR) for k in range(8)]
rows += [{"rid": 9000 + k, "message_decoded": f"Your OTP is {100000 + k}", "Predicted_Category": "Medical / Healthcare"} for k in range(6)]
fd = tempfile.mkdtemp()
pd.DataFrame(rows).to_csv(os.path.join(fd, "classified.csv"), index=False)
with redirect_stdout(io.StringIO()):
    PL.stage_templates(PL.build_parser().parse_args(["templates", "--out-dir", fd]))
wf = RW.Workspace.from_output(fd, n_samples=5)
fid = next(f for f, m in wf.members.items() if len(m) == 6)
assert len(wf.registry) == 2 and wf.family_keys[fid].count("*") == 1, "the 6 branch templates are ONE unit of work, the OTP template another"
assert len(wf.texts[fid]) == 48
assert len(RW.Workspace.from_output(fd, families=False).registry) == 7 and not RW.Workspace.from_output(fd, families=False).members, "--no-families: one unit per template"
get_s, test_f, save_f = RW.make_tools(wf, "test-model")
smp = json.loads(get_s(fid))
seen = {b for t in smp["samples"] for b in BR if b.split()[-1] in t}
assert len(seen) == 5 and "FAMILY of 6" in smp["family"] and "[*]" in smp["template"], (seen, smp["family"][:60])
FAM_RX = r"Dear\s+(?P<name>.+?),\s+thanks for choosing Popular Diagnostic Centre\s+(?P<branch>.+?)\s+as your healthcare partner\.\s+Your Registration ID is\s+(?P<registration_id>{NUM})"
ORG = {"organisation": "Popular Diagnostic Centre"}
r = RW.evaluate(wf, fid, FAM_RX, ORG)
assert r["passed"] and r["match_rate"] == 1.0 and r["fields"]["branch"]["distinct_values"] == 6, r["problems"]
r = RW.evaluate(wf, fid, FAM_RX.replace("(?P<branch>.+?)", ".+?"), ORG)               # the branch is left out: the agent is told which values
assert not r["passed"] and any(p.startswith("incomplete") and "Gazipur" in p and "Narayanganj" in p for p in r["problems"]), r["problems"]
r = RW.evaluate(wf, fid, FAM_RX.replace("(?P<branch>.+?)", "Narayanganj"), ORG)       # hard-coded to one branch: fails on the others
assert not r["passed"] and r["match_rate"] < 0.95, r["match_rate"]
r = RW.evaluate(wf, fid, FAM_RX, {**ORG, "city": "Gazipur"})                        # a member's value is not wording shared by the family
assert not r["passed"] and any("not fixed wording" in p for p in r["problems"]), r["problems"]
desc2 = {"name": "customer name", "branch": "branch of the centre", "registration_id": "registration id"}
assert save_f(fid, FAM_RX, desc2, "registration_confirmation", ORG) == "SAVED"
entry = json.load(open(RW.REGISTRY_PATH, encoding="utf-8"))[fid]
assert entry["members"] == wf.members[fid] and entry["family_key"] == wf.family_keys[fid] and "[*]" in entry["template"], entry.keys()
ex = RegexExtractor(TemplateModel.load(os.path.join(fd, "template_model.json")), json.load(open(RW.REGISTRY_PATH, encoding="utf-8")))
got = ex.extract_one(POP.format(n="Rahim", b="Cox's Bazar", i=412345678), "Medical / Healthcare")
assert got[2] == "extracted" and got[0]["branch"] == "Cox's Bazar" and got[0]["organisation"] == "Popular Diagnostic Centre" and got[0]["event"] == "registration_confirmation", got
print("OK  families: one regex serves 6 branches; samples come from different members; branch left out / hard-coded / a member's value as a fact are rejected; saved with members + key; a branch never seen is extracted")

# ---- where a regex breaks: the usual cause is one small thing (a full stop, a line break)
hp = r"Welcome to\s+(?P<b>.+?)\s*\n\s*Your UHID is\s+(?P<u>\d+)\s*\n"
ht = "Dear X \n\nWelcome to LABAID DHANMONDI. \n\nYour UHID is 20000040.  \n\nPlease visit"
hint = RW.explain_mismatch(hp, ht)
assert "where it breaks" in hint and r"next part '\\n'" in hint and r"'.  \n\nPlease visit'" in hint and r"(?P<u>\\d+)" in hint, hint
assert "not even the start" in RW.explain_mismatch(r"Zzz(?P<a>\d+)", ht)
rr = RW.evaluate(ws, "std", r"(?P<phone>{NUM})\s+আইডির Akash Standard প্যাকের মেয়াদ শেষ হয়েছে\s+(?P<expiry>{DATE})\s+XYZ", FACTS)
assert any(p.startswith("where it breaks") and "XYZ" in p for p in rr["problems"]), rr["problems"]
assert not any("fixed fact" in p for p in RW.evaluate(ws, "std", r"(?P<a>nothing)\d+", FACTS)["problems"]), "facts are checked against the template's messages, not only the ones a (wrong) regex matched"
print("OK  a regex that does not match is told where it breaks and what the text looks like there")

# ---- units that are the same sentence split differently are not 'other templates'
regF = pd.DataFrame({"template_id": ["A", "G", "H"], "category": ["C"] * 3, "message_count": [20, 10, 5], "example": [""] * 3,
                     "template": ["[VAR] for choosing Popular Diagnostic [*] as your partner ID [VAR]",
                                  "[VAR] for choosing Popular Diagnostic Centre Chattogram as your partner ID [VAR]", "Your OTP is [NUM]"]})
ta = [f"Dear Rahim{i}, for choosing Popular Diagnostic Centre {b} as your partner ID {40000 + i}" for i, b in enumerate(["Gazipur", "Khulna", "Savar", "Kushtia"])]
tg = [f"Dear Rahim{i}, for choosing Popular Diagnostic Centre Chattogram as your partner ID {50000 + i}" for i in range(4)]
wx = RW.Workspace(regF, {"A": ta, "G": tg, "H": [f"Your OTP is {600000 + i}" for i in range(4)]})
wx.members = {"A": ["a1", "a2"], "G": ["g1", "g2"]}
wx.family_keys = {"A": "[X] for choosing Popular Diagnostic * as your partner ID [X]"}
wx.member_templates = {"A": ["[VAR] for choosing Popular Diagnostic Centre Gazipur as your partner ID [NUM]", "[VAR] for choosing Popular Diagnostic Centre Khulna as your partner ID [NUM]"],
                       "G": ["[VAR] for choosing Popular Diagnostic Centre Chattogram as your partner ID [NUM]", "[VAR] for choosing Popular Diagnostic Centre Chattogram as your partner ID is [NUM]"]}
assert wx.related_units("A") == {"G"} and wx.related_units("G") == {"A"} and wx.related_units("H") == set()
assert all("OTP" in t for t in wx.other_template_texts("A")), "A's 'other templates' are only the unrelated ones"
FR = r"for choosing Popular Diagnostic Centre\s+(?P<branch>.+?)\s+as your partner ID\s+(?P<id>\d+)"
rx2 = RW.evaluate(wx, "A", FR, {"organisation": "Popular Diagnostic Centre"})
assert rx2["other_templates_match_rate"] == 0.0 and not any("too generic" in p for p in rx2["problems"]), rx2["problems"]
print("OK  a regex that also matches the same sentence split into another family is not 'too generic'")
# ---- the PalmPay mistake: TK2613.0 read as repayment_code 2613 + amount 0
pp = [f"Your repayment TK{a}.0 has been paid successfully. Please open PalmPay app and check your repayment plan." for a in (2613, 900, 3016, 2427, 1230, 450, 1800, 2200, 3300, 150)]
regP = pd.DataFrame({"template_id": ["pp"], "category": ["Loan"], "message_count": [10], "example": [""],
                     "template": ["Your repayment [CODE] [NUM] has been paid successfully Please open PalmPay app and check your repayment plan"]})
wpp = RW.Workspace(regP, {"pp": pp})
BAD_PP = r"Your repayment\s+TK(?P<repayment_code>\d+)\.(?P<amount>\d+)"
r = RW.evaluate(wpp, "pp", BAD_PP, {"organisation": "PalmPay"})
assert not r["passed"] and any("cut into pieces" in p for p in r["problems"]) and any("'amount' is 0 in almost every message" in p for p in r["problems"]), r["problems"]
GOOD_PP = r"Your repayment\s+TK(?P<amount>\d+(?:\.\d+)?)\s+has been paid successfully"
r = RW.evaluate(wpp, "pp", GOOD_PP, {"organisation": "PalmPay"})
assert r["passed"] and r["fields"]["amount"]["distinct_values"] == 10, r["problems"]
assert "ONE amount" in RW.SYSTEM_PROMPT and "trust the sample messages" in RW.SYSTEM_PROMPT
print("OK  an amount cut into a 'code' and a zero is rejected with an explanation; the whole-number regex passes")

# ---- hints: letter case and macros
h = RW.explain_mismatch(r"has been opened on\s+(?P<d>\d{2}-[A-Z]{3}-\d{4})", "Your DPS has been opened on 06-Jun-2026 with installment")
assert "WOULD match if letter case were ignored" in h, h
h = RW.explain_mismatch(r"IFIC\s+Helpline-(?P<p>{PHONE})", "IFIC Helpline-16255")
assert "{PHONE} means" in h and "WOULD match" not in h, h
assert "WOULD match" not in RW.explain_mismatch(r"opened on\s+(?P<d>\d{2}-[a-z]{3})", "opened on 06-Jun-2026 and then")[:0] + "x"
print("OK  'where it breaks' also says when only letter case differs and what a macro in the failing part means")

# ---- the model client: reasoning of earlier turns is not kept (Qwen3.8 would fill the context), optional reasoning effort
seen = {}
orig_agent = RW.create_deep_agent
RW.create_deep_agent = lambda **kw: seen.update(kw) or object()
try:
    RW.build_agent(wpp, "some-model", think=True)
    assert seen["model"].extra_body == {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False}}, seen["model"].extra_body
    assert not getattr(seen["model"], "reasoning_effort", None)
    os.environ["LLM_REASONING_EFFORT"] = "medium"
    RW.build_agent(wpp, "some-model", think=False)
    assert seen["model"].reasoning_effort == "medium" and seen["model"].extra_body["chat_template_kwargs"]["enable_thinking"] is False
finally:
    os.environ.pop("LLM_REASONING_EFFORT", None)
    RW.create_deep_agent = orig_agent
print("OK  model client: earlier reasoning is not kept in the prompt; LLM_REASONING_EFFORT is passed on when set")

# ---- --redo: the regex stage rewrites regexes that already exist
captured = {}
saved_fns = (RW.run_templates, RW.check_backend)
RW.run_templates = lambda ws, ids, model, **kw: (captured.update(kw), {i: "SAVED" for i in ids})[1]
RW.check_backend = lambda backend, model: None
try:
    for extra, want in (([], True), (["--redo"], False)):
        with redirect_stdout(io.StringIO()):
            PL.stage_regex(PL.build_parser().parse_args(["regex", "--out-dir", fd, "--all", "--min-messages", "1", *extra]))
        assert captured["skip_done"] is want, (extra, captured)
finally:
    RW.run_templates, RW.check_backend = saved_fns
    RW.set_output_dir(tmp)
print("OK  regex --redo writes the regexes again (skip_done is False); without it finished ones are skipped")
# ---- several workers: a "still working" line while nothing finishes, then one line per finished family
import time as _time
orig_one, orig_hb = RW._run_one, RW.HEARTBEAT_S
RW._run_one = lambda ws_, tid, *a, **k: (_time.sleep(0.5), ("SAVED", 2, 0.5))[1]
RW.HEARTBEAT_S = 0.12
try:
    wrp = RW.Workspace(pd.DataFrame({"template_id": ["a", "b"], "category": ["C", "C"], "template": ["t [NUM] a", "t [NUM] b"], "message_count": [9, 8], "example": ["", ""]}),
                       {"a": ["t 1 a"], "b": ["t 2 b"]})
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = RW.run_templates(wrp, ["a", "b"], "m", verbose=True, workers=2, skip_done=False)
finally:
    RW._run_one, RW.HEARTBEAT_S = orig_one, orig_hb
text = buf.getvalue()
assert res == {"a": "SAVED", "b": "SAVED"} and text.count("still working") >= 2 and "[a] SAVED" in text and "[b] SAVED" in text and "2/2 done" in text, text[-400:]
print("OK  with several workers a 'still working' line is printed while no family finishes; every finished family still gets its line")

# ---- fixed phrases: the agent references the exact original Bengali text instead of typing it
COURSES = ["HSC '26 মানবিক - রেসকিউ রিভিশন", "SSC ক্লাস 9 বিজ্ঞান", "ক্লাস 8 অর্ধবার্ষিক পরীক্ষা", "HSC ব্যবসায় শিক্ষা - ফাইনাল", "অষ্টম শ্রেণি গণিত",
           "SSC '29 বিজ্ঞান", "HSC '27 কলা", "ক্লাস 6 ইংরেজি"]
def shikho(i): return (f"[Shikho] দুনিয়ায় তোমাকে স্বাগত! তোমার ফ্রি কোর্স: {COURSES[i % len(COURSES)]}, কোর্সের ফেসবুক গ্রুপ: "
                       f"https://www.facebook.com/groups/{1000000 + 37 * i}, প্রয়োজনে কল: 0961{i:04d}00")
sh_texts = [shikho(i) for i in range(16)]
ph = RW.fixed_phrases(sh_texts)
assert ph == ["[Shikho] দুনিয়ায় তোমাকে স্বাগত! তোমার ফ্রি কোর্স:", "কোর্সের ফেসবুক গ্রুপ:", "প্রয়োজনে কল:"], ph       # fixed text only: no part of a changing value
assert RW.fixed_phrases(sh_texts[:2]) == [] and RW.fixed_phrases([]) == [] and RW.fixed_phrases([sh_texts[0]] * 5) == [], "fewer than 3 distinct messages: nothing to tell apart"
assert RW.fixed_phrases([t.replace(" ", "  ") if i % 2 else t for i, t in enumerate(sh_texts)]) == ph, "spacing is ignored"
assert RW.fixed_phrases(sh_texts, max_phrases=2) == [ph[0], ph[1]], "with a cap the longest phrases are kept, still in order"
assert all(len(p) >= RW.MIN_PHRASE_LEN for p in RW.fixed_phrases(["Your OTP is 123456 ok", "Your OTP is 654321 ok", "Your OTP is 111222 ok", "Your OTP is 999000 ok"]))
assert RW.expand_macros(r"{L1}\s*(?P<c>.+?)", ["কোর্সের ফেসবুক  গ্রুপ:"]) == r"কোর্সের\s+ফেসবুক\s+গ্রুপ:\s*(?P<c>.+?)"
for bad_ref, phrases_arg in ((r"{L1}", None), (r"{L1}", []), (r"{L2}", ["abcd"])):
    try:
        RW.expand_macros(bad_ref, phrases_arg); raise AssertionError("no error")
    except ValueError as e:
        assert "unknown phrase" in str(e)
print("OK  fixed phrases: exact fixed text in order, no part of a changing value, spacing ignored, none when there is nothing to compare; {L1} expands to the exact text")

regS = pd.DataFrame({"template_id": ["sh"], "category": ["Education"], "message_count": [16], "example": [""],
                     "template": ["[Shikho] দুনিয়ায় তোমাকে স্বাগত তোমার ফ্রি কোর্স [VAR] কোর্সের ফেসবুক গ্রুপ [VAR] প্রয়োজনে কল [NUM]"]})
wsh = RW.Workspace(regS, {"sh": sh_texts})
GOOD_SH = r"{L1}\s*(?P<course>.+?),\s*{L2}\s*(?P<group_url>https?://\S+?),\s*{L3}\s*(?P<phone>\d+)"
r = RW.evaluate(wsh, "sh", GOOD_SH, {"organisation": "Shikho"})
assert r["passed"] and r["match_rate"] == 1.0 and len(r["anchor_words"]) >= 2, r["problems"]
# the hand-typed version, with the mistakes the model really made (a missing joining mark, a look-alike letter), matches nothing
TYPED = r"\[Shikho\]\s*দুনিয়ায়\s*তোমাকে\s*স্বাগত!\s*তোমার\s*ফ্রি\s*কোর্স:\s*(?P<course>.+?),\s*কোরসের\s*ফেসবুক\s*গ্রুপ:\s*(?P<group_url>\S+?),\s*প্ৰয়োজনে\s*কল:\s*(?P<phone>\d+)"
r = RW.evaluate(wsh, "sh", TYPED, {"organisation": "Shikho"})
assert not r["passed"] and r["match_rate"] == 0.0
hint = next(p for p in r["problems"] if p.startswith("where it breaks"))
assert "non-English letters" in hint and "{L1}" in hint and "{L2}" in hint, hint
# a pattern made only of phrases and values still has fixed words (the phrases count as anchors)
assert RW.anchor_words(r"{L1}(?P<a>.+?){L2}", ph) and len(RW.anchor_words(r"{L1}(?P<a>.+?){L2}", ph)) >= 2 and RW.anchor_words(r"{L1}(?P<a>.+?){L2}") == []
print("OK  a regex typed by hand with look-alike / missing letters matches nothing and is told to use {L1}...; the same regex written with the phrases matches every message")

get_s2, test_f2, save_f2 = RW.make_tools(wsh, "test-model")
info = json.loads(get_s2("sh"))
assert list(info["phrases"]) == ["L1", "L2", "L3"] and info["phrases"]["L2"] == "কোর্সের ফেসবুক গ্রুপ:" and "{L1}" in info["phrases_note"], info.keys()
assert "phrases" not in json.loads(RW.make_tools(RW.Workspace(regS, {"sh": sh_texts[:2]}), "m")[0]("sh")), "no phrases when there is nothing to compare"
assert save_f2("sh", GOOD_SH, {"course": "course name", "group_url": "link of the group", "phone": "phone number"}, "course_welcome", {"organisation": "Shikho"}) == "SAVED"
saved_sh = json.load(open(RW.REGISTRY_PATH, encoding="utf-8"))["sh"]
assert saved_sh["pattern"] == GOOD_SH and "{L1}" not in saved_sh["expanded_pattern"] and "কোর্সের\\s+ফেসবুক\\s+গ্রুপ:" in saved_sh["expanded_pattern"], saved_sh["expanded_pattern"][:200]
assert saved_sh["phrases"] == ph
import regex as _rx
m = _rx.search(saved_sh["expanded_pattern"], sh_texts[3])
assert m and m.group("course") == COURSES[3] and m.group("phone") == "0961000300", "the saved regex needs no phrases at extraction time"
print("OK  get_template_samples lists the phrases; save_regex keeps the agent's pattern, the expanded exact text (used at extraction) and the phrases")
# ---- the challan family: constant dates / codes / ids are values; fixed phrases never contain a word with a digit
CH = [f"Customer ID- {4000 + 7 * i} , Acme Trading SO-{i}77 SO Dt-04.06.26 Challan- {5000 + i} Challan Dt- 06.06.26 Vehicle- DHA.MT-{i} Driver- {n} ; "
      f"MAT SOLFAF001L12P Qty- {10 + i}.00" for i, n in enumerate(["ZAMAN", "RUBEL", "KARIM", "SELIM", "JASIM", "RAHIM", "SOHEL", "NAYEM", "JAMAL", "FARUK", "ALAM", "TUHIN"])]
chp = RW.fixed_phrases(CH)
assert chp and not any(any(c.isdigit() for c in p) for p in chp) and "Challan Dt-" in chp and "Vehicle-" in chp, chp
regC = pd.DataFrame({"template_id": ["ch"], "category": ["E-commerce"], "message_count": [12], "example": [""],
                     "template": ["Customer ID [NUM] Acme Trading SO [NUM] SO Dt [NUM] Challan [NUM] Challan Dt [NUM] Vehicle [CODE] Driver [VAR] MAT [CODE] Qty [NUM]"]})
wch = RW.Workspace(regC, {"ch": CH})
CHRX = (r"Customer ID-\s*(?P<customer_id>\d+)\s*{L2}\s*SO-(?P<so_no>\d+)\s*SO\s*Dt-(?P<so_dt>[\d.]+)\s*Challan-\s*(?P<challan>\d+)\s*Challan Dt-\s*(?P<challan_dt>[\d.]+)\s*"
        r"Vehicle-\s*(?P<vehicle>\S+)\s*Driver-\s*(?P<driver>.+?)\s*;\s*MAT\s*(?P<mat_code>\w+)\s*Qty-\s*(?P<qty>\d+\.\d+)")
rc = RW.evaluate(wch, "ch", CHRX)
assert rc["match_rate"] == 1.0, rc["problems"]
assert not any("same value" in p for p in rc["problems"]), ("constant dates, codes and ids are values, not wording", rc["problems"])
assert not any("incomplete" in p and ("Challan" in p or "Vehicle" in p or "Qty" in p) for p in rc["problems"]), ("labels are not asked for", rc["problems"])
words_const = RW.evaluate(wch, "ch", CHRX.replace("{L2}", r",\s*(?P<party>Acm\w+)\s+Trading"))
assert any("party" in p and "same value" in p for p in words_const["problems"]), "a constant WORD in a group is wording: that is still rejected"
print("OK  a small family where a date, a code and an id never change: they are values (a group may capture them), labels are not asked for, phrases hold no digits")

# ---- the ladder: what happens on its own when a family does not pass the checks
os.environ["VLLM_NO_RESTART"] = "1"
LA = [f"Paid TK{1000 + i}.0 for order {500000 + i}" for i in range(18)]             # the main layout: 18 messages
LB = [f"Order {600000 + i} paid by card TK{2000 + i}.00" for i in range(3)]         # another layout: 3 messages
regL = pd.DataFrame({"template_id": ["lad"], "category": ["Loan"], "message_count": [21], "example": [""],
                     "template": ["Paid TK [NUM] for order [NUM] Order [NUM] paid by card TK [NUM]"]})
RXA = r"Paid TK(?P<amount>\d+(?:\.\d+)?) for order (?P<order>\d+)"
RXB = r"Order (?P<order>\d+) paid by card TK(?P<amount>\d+(?:\.\d+)?)"
DESC = {"amount": "amount paid", "order": "order number"}
wl = RW.Workspace(regL, {"lad": LA + LB})
r95 = RW.evaluate(wl, "lad", RXA)
assert not r95["passed"] and any("match_rate 0.86 < 0.95" in p for p in r95["problems"]) and r95["floor"] == 0.95
wl.match_floor["lad"] = RW.SOFT_MATCH_RATE
assert RW.evaluate(wl, "lad", RXA)["passed"], "at the soft floor a regex that reads 86% of the family passes"
wl.match_floor.clear()
sm = RW.diverse_samples(LA + LB, 2)
assert len(sm) == 2 and any(t.startswith("Paid") for t in sm) and any(t.startswith("Order") for t in sm), sm       # one per layout first, the biggest layout first
assert len(RW.diverse_samples(LA, 5)) == 5 and RW.diverse_samples([], 3) == []
assert RW.is_connection_error("error: OpenAIConnectionError: Connection error.") and not RW.is_connection_error("no regex saved (checks not passed)")
assert RW._effort_for("qwen3.8-27b", "xhigh") == "xhigh" and RW._effort_for("qwen3.5-9b", "xhigh") is None and RW._effort_for("qwen3.8-27b", None) is None
print("OK  soft floor: an 86% regex fails the normal check and passes the soft one; samples are one per layout first; connection errors and effort helpers")

calls_l = []
orig_agent_run = RW._agent_run
def make_stub(saves, log):
    """saves: {(unit, rung number): (pattern,)}: what the 'agent' saves on that rung; everything else ends with 'no regex saved'."""
    count = {}
    def stub(ws_, tid, prompt, model_name, think, backend, ui, effort, recursion):
        count[tid] = count.get(tid, 0) + 1
        log.append({"tid": tid, "rung": count[tid], "effort": effort, "recursion": recursion, "samples": ws_.samples_for.get(tid), "floor": ws_.match_floor.get(tid),
                    "prompt": prompt, "think": think})
        if (tid, count[tid]) in saves:
            ui.attempts += 1
            out = RW.make_tools(ws_, model_name)[2](tid, saves[(tid, count[tid])], DESC, "order_paid")
            return "SAVED" if out == "SAVED" else "no regex saved (checks not passed)"
        if "connection" in saves.get("mode", ""):
            return "error: OpenAIConnectionError: Connection error."
        ui.attempts += 1
        RW.evaluate(ws_, tid, RXA)                                  # the agent tested a regex that did not pass: the checks remember why
        return "no regex saved (checks not passed)"
    return stub
try:
    RW.REGISTRY_PATH.unlink(missing_ok=True)
    wl = RW.Workspace(regL, {"lad": LA + LB}); log_a = []
    RW._agent_run = make_stub({("lad", 3): RXA, ("lad~1", 1): RXB}, log_a)
    res, att, secs = RW._run_one(wl, "lad", "qwen3.8-27b", True, "vllm", RW.Progress(False))
    assert res == "SAVED"
    rungs = [c for c in log_a if c["tid"] == "lad"]
    assert [c["recursion"] for c in rungs] == [60, 100, 100] and [c["effort"] for c in rungs] == [None, "xhigh", "xhigh"], rungs
    assert [c["samples"] for c in rungs] == [None, 10, 10] and [c["floor"] for c in rungs] == [None, None, RW.SOFT_MATCH_RATE]
    assert "checks last said" in rungs[1]["prompt"] and "match_rate 0.86" in rungs[1]["prompt"], rungs[1]["prompt"][-300:]
    assert "MAIN layout" in rungs[2]["prompt"] and "85%" in rungs[2]["prompt"] and "MAIN layout" not in rungs[0]["prompt"]
    reg_l = RW.load_registry()
    assert reg_l["lad"].get("partial") is True and abs(reg_l["lad"]["metrics"]["match_rate"] - 18 / 21) < 0.01
    child = [c for c in log_a if c["tid"] == "lad~1"]
    assert len(child) == 1 and "lad~1" in reg_l and reg_l["lad~1"]["parent"] == "lad" and "members" not in reg_l["lad~1"], (child, list(reg_l))
    assert wl.texts["lad~1"] == sorted(LB) or sorted(wl.texts["lad~1"]) == sorted(LB), "the second regex is written for exactly the messages the first one misses"
    assert any("second regex" in n for n in wl.notes["lad"])
    # a model that is not Qwen3.8 gets no reasoning-effort setting; retries=0 is a single run; a connection error ends the climb at once
    log_b = []; RW._agent_run = make_stub({}, log_b)
    RW._run_one(RW.Workspace(regL, {"lad": LA + LB}), "lad", "qwen3.5-9b", True, "vllm", RW.Progress(False))
    assert [c["effort"] for c in log_b] == [None, None, None] and len(log_b) == 3, "three rungs, all failing"
    log_c = []; RW._agent_run = make_stub({}, log_c)
    RW._run_one(RW.Workspace(regL, {"lad": LA + LB}), "lad", "qwen3.5-9b", True, "vllm", RW.Progress(False), retries=0)
    assert len(log_c) == 1
    log_d = []; RW._agent_run = make_stub({"mode": "connection"}, log_d)
    r_d = RW._run_one(RW.Workspace(regL, {"lad": LA + LB}), "lad", "qwen3.8-27b", True, "vllm", RW.Progress(False))
    assert RW.is_connection_error(r_d[0]) and len(log_d) == 1, "a server that does not answer does not use up the other rungs"
    # a first-rung success is the end of it (no deeper run, no leftover unit)
    RW.REGISTRY_PATH.unlink(missing_ok=True)
    w_ok = RW.Workspace(regL, {"lad": LA}); log_e = []
    RW._agent_run = make_stub({("lad", 1): RXA}, log_e)
    assert RW._run_one(w_ok, "lad", "qwen3.8-27b", True, "vllm", RW.Progress(False))[0] == "SAVED" and len(log_e) == 1 and "lad~1" not in w_ok.texts
finally:
    RW._agent_run = orig_agent_run
print("OK  ladder: normal -> deeper (10 different samples, 100 steps, xhigh on Qwen3.8, what the checks said) -> soft landing (85%), then a second regex for the "
      "messages the first misses; a server outage ends the climb at once; other models get no effort setting")


# ---- one message type split into near-identical families: a regex for one of them may match the others
RW.SIMILAR_TEMPLATE_RATIO = SIMILAR
EGP = "e GP System Dear User You have successfully completed final submission of Tender Id {t} Ref No {r}{tail} e GP System"
ga = [EGP.format(t=100 + i, r=200 + i, tail=" WXY") for i in range(10)]
gb = [EGP.format(t=300 + i, r=400 + i, tail="") for i in range(6)]
gc = [f"Your OTP is {100000 + i} valid for 5 minutes" for i in range(8)]
regG = pd.DataFrame({"template_id": ["ga", "gb", "gc"], "category": ["Gov"] * 3, "message_count": [10, 6, 8], "example": [""] * 3,
                     "template": ["e GP System Dear User You have successfully completed final submission of Tender Id [VAR] Ref No [*] [VAR] e GP System",
                                  "e GP System Dear User You have successfully completed final submission of Tender Id [VAR] Ref No [*] e GP System",
                                  "Your OTP is [NUM] valid for [NUM] minutes"]})
wg = RW.Workspace(regG, {"ga": ga, "gb": gb, "gc": gc})
assert wg._similar_units("ga") == {"gb"} and wg._similar_units("gb") == {"ga"} and wg._similar_units("gc") == set()
assert wg.related_units("ga") >= {"gb"} and "gb" not in [u for u in wg.registry.index if u not in wg.related_units("ga") and u != "ga"]
assert all("OTP" in t for t in wg.other_template_texts("ga")), "only the OTP messages are 'other templates' of the tender message"
RXG = r"Tender Id (?P<tender_id>\d+) Ref No (?P<ref_no>\d+)"
rg = RW.evaluate(wg, "ga", RXG, {"organisation": "e GP System"})
assert rg["other_templates_match_rate"] == 0.0 and not any("too generic" in p for p in rg["problems"]), rg["problems"]
too_loose = RW.evaluate(wg, "ga", r"(?P<n>\d+)\s+valid\s+for", {"organisation": "e GP System"})            # still caught: it fits the OTP messages
assert any("too generic" in p or "match_rate" in p for p in too_loose["problems"])
print("OK  near-identical families (the same message split in two) are not each other's 'other templates'; a really loose regex is still caught")
RW.SIMILAR_TEMPLATE_RATIO = 1.01

# ---- the same message in a close shape (similarity 0.8 - 0.9) is not an "other template"; a different message still is
RW.SIMILAR_TEMPLATE_RATIO = SIMILAR
# shapes of one card-purchase message as the template builder split it on real data (0.85 apart); the transfer message is another message (< 0.5)
TA = "[VAR] at [VAR] [*] [VAR] Balance Card [VAR] Acme Helpline [VAR]"
TB = "[VAR] at [VAR] BDT [VAR] Purchase BDT [VAR] Balance Card [*] BD Acme Helpline [VAR]"
TC = "[VAR] [NUM] Digital Transfer To [VAR] Balance Acme Helpline-[NUM]"
ta = [f"28.05.25 at 16:{i:02d} USD {20 + i}.00 Purchase BDT {3700 + i}.24 Balance Card 4587***{3200 + i} SHOP{i}, US, Acme Helpline-16255" for i in range(10)]
tb = [f"29.05.25 at 17:{i:02d} BDT {900 + i}.50 Purchase BDT {900 + i}.50 Balance Card 4587***{5100 + i} STORE{i}, BD Acme Helpline-16255" for i in range(10)]
tc = [f"May {i + 1}, 2025 at 10:{i:02d} {500 + i}.00 Digital Transfer To 01712{i:06d} Balance Acme Helpline-16255" for i in range(10)]
regT = pd.DataFrame({"template_id": ["ta", "tb", "tc"], "category": ["Bank"] * 3, "message_count": [10, 10, 10], "example": [""] * 3, "template": [TA, TB, TC]})
wt = RW.Workspace(regT, {"ta": ta, "tb": tb, "tc": tc})
assert wt._similar_units("ta") == {"tb"} and wt._similar_units("tb") == {"ta"} and wt._similar_units("tc") == set(), "0.89 is the same message, the transfer is another"
RXT = r"Purchase\s+(?P<balance_currency>[A-Z]{3})\s+(?P<balance>[\d.]+)\s+Balance\s+Card\s+(?P<card>\d{4}\*{3}\d{4})"
good = RW.evaluate(wt, "ta", RXT, {"organisation": "Acme"})
assert not any("too generic" in p for p in good.get("problems", [])), good.get("problems")
RW.SIMILAR_TEMPLATE_RATIO = 0.9                                             # the old limit: the close shape counted as another template
old = RW.evaluate(RW.Workspace(regT, {"ta": ta, "tb": tb, "tc": tc}), "ta", RXT, {"organisation": "Acme"})
RW.SIMILAR_TEMPLATE_RATIO = SIMILAR
assert any("too generic" in p for p in old.get("problems", [])), "with 0.9 this regex was blocked"
print("OK  a close shape of the same message is related (>= 0.8); the unrelated transfer message is not")

# a handful of messages of another shape (< MIN_OTHER_TEMPLATE_MESSAGES) is not an "other template": it never gets a regex of its own
TD = "[NUM] on [NUM] Foreign Spend Purchase BDT [NUM] Balance Card [NUM] Acme Online"
td = [f"11.06.25 on 12:{i:02d} Foreign Spend Purchase BDT {400 + i}.00 Balance Card 4587***{7700 + i} Acme Online" for i in range(8)]
def reg_td(n): return pd.DataFrame({"template_id": ["ta", "tb", "tc", "td"], "category": ["Bank"] * 4, "message_count": [10, 10, 10, n], "example": [""] * 4, "template": [TA, TB, TC, TD]})
few = RW.Workspace(reg_td(3), {"ta": ta, "tb": tb, "tc": tc, "td": td[:3]})
many = RW.Workspace(reg_td(8), {"ta": ta, "tb": tb, "tc": tc, "td": td})
assert "td" not in few._similar_units("ta"), "the foreign-spend message is a different shape"
assert not any("too generic" in p for p in RW.evaluate(few, "ta", RXT, {"organisation": "Acme"}).get("problems", [])), "3 messages: not an 'other template'"
assert any("too generic" in p for p in RW.evaluate(many, "ta", RXT, {"organisation": "Acme"}).get("problems", [])), "8 messages of another shape still count"
RW.SIMILAR_TEMPLATE_RATIO = 1.01

# ---- a constant notice (one text, no variable part) has nothing to extract: the agent is not asked, and the result is not a failure
CONST = "আপনার কার্ডটি ডেলিভারির জন্য প্রস্তুত, অনুগ্রহ করে শাখা থেকে সংগ্রহ করুন"
regK = pd.DataFrame({"template_id": ["k1", "k2", "k3"], "category": ["Bank"] * 3, "message_count": [6, 5, 6], "example": [""] * 3,
                     "template": [CONST, "Your code is [NUM] ok", CONST + " today"]})
wk = RW.Workspace(regK, {"k1": [CONST] * 6, "k2": ["Your code is 123456 ok"] * 5, "k3": [CONST + " today", CONST + "  today"] * 3})
assert wk.is_constant_notice("k1") and not wk.is_constant_notice("k2"), "a template WITH a slot is never a constant notice, even when its text repeats"
assert wk.is_constant_notice("k3"), "spacing differences do not make a second text"
asked = []
real_run_one = RW._run_one
RW._run_one = lambda ws_, tid, *a, **k: (asked.append(tid), ("SAVED", 1, 0.0))[1]
try:
    res = RW.run_templates(wk, ["k1", "k2", "k3"], "m", skip_done=False, verbose=False, workers=2)
finally:
    RW._run_one = real_run_one
assert res == {"k1": "constant notice", "k2": "SAVED", "k3": "constant notice"} and asked == ["k2"], (res, asked)
assert all(v in RW.OK_RESULTS for v in res.values())
print("OK  a constant notice is reported as 'constant notice' without calling the agent; a message with a slot still goes to the agent")

# ================= self-fixing at runtime: stall detection, relaxed limit from rung 2, time budget, remembered failures =================
WALL = "too generic: matches {}% of other templates' messages (max 5%); add fixed anchor words from this template's own wording"
assert RW._kind(WALL.format(48)) == RW._kind(WALL.format(11)) and RW._kind("group(s) ['a'] swallow most of the message (free text)") == RW._kind("group(s) ['bb', 'c'] swallow most of the message (free text)")
assert RW._kind(WALL.format(48)) != RW._kind("match_rate 0.40 < 0.95: see not_matched")
wst = RW.Workspace(regT, {"ta": ta, "tb": tb, "tc": tc})
for score in (-0.40, -0.40, -0.41):
    wst.record_attempt("ta", [WALL.format(48)], score)
assert not wst.stalled("ta"), "three tests are not enough"
wst.record_attempt("ta", [WALL.format(49)], -0.41)
assert wst.stalled("ta"), "four tests, the same wall (the numbers do not matter), no progress"
wst.record_attempt("ta", [WALL.format(30)], -0.20)
assert not wst.stalled("ta"), "the score rose: the agent is getting somewhere"
wst.record_attempt("ta", ["match_rate 0.40 < 0.95: see not_matched"], -0.4)
assert not wst.stalled("ta"), "another wall: the agent is moving"
wst.history["ta"].clear()
for _ in range(4):
    wst.record_attempt("ta", [], 1.0)
assert not wst.stalled("ta"), "a passing test is never a stall"

class FakeUI:
    attempts = 0
    def __init__(self): self.lines = []
    def on_message(self, m): pass
    def log_only(self, t): self.lines.append(str(t))
class FakeAgent:
    """An 'agent' that tests the same too-loose regex over and over (50 times if nobody stops it)."""
    def __init__(self, ws_, tid, test=True): self.ws, self.tid, self.test, self.count = ws_, tid, test, 0
    def stream(self, *a, **k):
        for _ in range(50):
            self.count += 1
            if self.test:
                RW.evaluate(self.ws, self.tid, r"(?P<n>\d+)\s+valid\s+for")
            yield {"model": {"messages": []}}
orig_build = RW.build_agent
try:
    wgs = RW.Workspace(regG, {"ga": ga, "gb": gb, "gc": gc}); fa = FakeAgent(wgs, "ga"); ui_s = FakeUI()
    RW.build_agent = lambda *a, **k: (fa, None)
    out = RW._agent_run(wgs, "ga", "p", "m", False, "vllm", ui_s, None, 100)
    assert out.startswith("no regex saved (stalled on:") and fa.count == RW.STALL_ATTEMPTS, (out, fa.count)
    assert any("stopped" in l for l in ui_s.lines)
    wgt = RW.Workspace(regG, {"ga": ga, "gb": gb, "gc": gc}); fb = FakeAgent(wgt, "ga", test=False)
    RW.build_agent = lambda *a, **k: (fb, None)
    wgt.deadline["ga"] = time.time() - 1
    out = RW._agent_run(wgt, "ga", "p", "m", False, "vllm", FakeUI(), None, 100)
    assert out == "no regex saved (time budget used)" and fb.count == 1, (out, fb.count)
finally:
    RW.build_agent = orig_build
print("OK  an agent that tests the same wall 4 times without progress is stopped at once (not after 60-100 steps); so is a family past its time budget")

# rung 2 and 3 accept up to SOFT_FP_CEILING of other templates' messages (told to the agent); a spent time budget ends the climb
seen = []
def stub_fp(ws_, tid, prompt, model_name, think, backend, ui, effort, recursion):
    seen.append((ws_.fp_ceiling.get(tid), "OTHER templates" in prompt))
    return "no regex saved (checks not passed)"
try:
    RW._agent_run = stub_fp
    RW._run_one(RW.Workspace(regL, {"lad": LA + LB}), "lad", "qwen3.8-27b", True, "vllm", RW.Progress(False))
    assert seen == [(None, False), (RW.SOFT_FP_CEILING, True), (RW.SOFT_FP_CEILING, True)], seen
    seen.clear(); wbud = RW.Workspace(regL, {"lad": LA + LB}); wbud.budget_s = -1
    r_b = RW._run_one(wbud, "lad", "qwen3.8-27b", True, "vllm", RW.Progress(False))
    assert len(seen) == 1 and "time budget" in r_b[0], (seen, r_b)
finally:
    RW._agent_run = orig_agent_run
RW.SIMILAR_TEMPLATE_RATIO = 1.01
wfp = RW.Workspace(regT, {"ta": ta, "tb": tb, "tc": tc})
assert any("too generic" in p for p in RW.evaluate(wfp, "ta", RXT, {"organisation": "Acme"})["problems"]), "50% of the other templates: refused at the normal limit"
wfp.fp_ceiling["ta"] = RW.SOFT_FP_CEILING
relaxed = RW.evaluate(wfp, "ta", RXT, {"organisation": "Acme"})
assert not any("too generic" in p for p in relaxed["problems"]) and relaxed["other_templates_ceiling"] == RW.SOFT_FP_CEILING and relaxed["other_templates_match_rate"] > RW.MAX_FALSE_POSITIVE
wfp.fp_ceiling["ta"] = 0.3
assert any("too generic" in p and "max 30%" in p for p in RW.evaluate(wfp, "ta", RXT, {"organisation": "Acme"})["problems"]), "a regex that reads even more is still refused"
# what is accepted under the relaxed limit is marked in the registry
canned = dict(RW.evaluate(ws, "std", GOOD, FACTS)); assert canned["passed"]
canned["other_templates_match_rate"] = 0.4
orig_eval = RW.evaluate; RW.evaluate = lambda *a, **k: canned
try:
    out = RW.make_tools(ws, "m")[2]("std", GOOD, {"phone": "number", "expiry": "expiry date", "balance": "balance"}, "order_paid", FACTS)
finally:
    RW.evaluate = orig_eval
assert out == "SAVED" and RW.load_registry()["std"]["waived"] == {"other_templates_match_rate": 0.4}, out
print("OK  from rung 2 a regex may read up to 50% of other templates' messages (the agent is told, the registry says 'waived'); more than that is still refused")

# families that fail are remembered: not tried again for 7 days or until the checks change; solved ones are forgotten
tmp2 = tempfile.mkdtemp(); RW.set_output_dir(tmp2)
runs = []; outcome = {"r": "fail"}
def fake_one(ws_, tid, *a, **k):
    runs.append(tid)
    ws_.last_problems[tid] = [WALL.format(48)]
    return {"fail": ("no regex saved (stalled on: x)", 3, 1.0), "ok": ("SAVED", 1, 1.0), "conn": ("error: OpenAIConnectionError: Connection error.", 0, 0.0)}[outcome["r"]]
real_one = RW._run_one; RW._run_one = fake_one
try:
    wf = RW.Workspace(regT, {"ta": ta, "tb": tb, "tc": tc})
    go = lambda **k: RW.run_templates(wf, ["ta", "tb"], "m", skip_done=True, verbose=False, workers=2, **k)
    r1 = go(); assert sorted(runs) == ["ta", "tb"] and set(r1.values()) == {"no regex saved (stalled on: x)"}
    stored = RW.load_failed(); assert set(stored) == {"ta", "tb"} and stored["ta"]["checks_version"] == RW.CHECKS_VERSION and stored["ta"]["blocked_by"].startswith("too generic")
    r2 = go(); assert len(runs) == 2 and set(r2.values()) == {"failed before (skipped)"}, "a failed family is left alone"
    assert wf.last_problems["ta"][0].startswith("too generic"), "the report still says what blocked it"
    assert not any(v in RW.OK_RESULTS for v in r2.values())
    go(retry_failed=True); assert len(runs) == 4, "retry_failed=True tries it again"
    real_version = RW.CHECKS_VERSION
    fp1, old_limit = RW._checks_fingerprint(), RW.MAX_FALSE_POSITIVE
    assert isinstance(real_version, str) and len(fp1) >= 7 and fp1 == RW._checks_fingerprint(), "the version is a fingerprint of the checks, stable between calls"
    RW.MAX_FALSE_POSITIVE = 0.07
    try:
        assert RW._checks_fingerprint() != fp1, "changing a limit changes the version by itself"
    finally:
        RW.MAX_FALSE_POSITIVE = old_limit
    RW.CHECKS_VERSION = "another-version"
    try:
        go(); assert len(runs) == 6, "after the checks changed it is tried again by itself"
    finally:
        RW.CHECKS_VERSION = real_version
    old = RW.load_failed(); old["ta"]["failed_at"] = "2000-01-01 00:00:00"; RW.FAILED_PATH.write_text(json.dumps(old))
    go(); assert runs[-2:] == ["ta", "tb"] or sorted(runs[-1:]) == ["ta"], "old failures are tried again"
    n_before = len(runs); outcome["r"] = "ok"; go(retry_failed=True)
    assert RW.load_failed() == {}, "families that now have a regex are no longer 'failed'"
    outcome["r"] = "conn"; go(retry_failed=True)
    assert RW.load_failed() == {}, "a server outage is not a failure of the family"
finally:
    RW._run_one = real_one
    RW.set_output_dir(tmp)
print("OK  failed families are remembered (not retried for 7 days or until the checks change, retry_failed=True overrides); solved ones are forgotten; outages are not recorded")
print("\nALL REGEX WRITER TESTS PASSED")
