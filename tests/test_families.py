"""Offline tests of template FAMILIES: templates that differ only in one varying part (a branch, an ATM, a merchant) share one regex."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import io, json, tempfile
from contextlib import redirect_stdout
import pandas as pd
from sms_profiling import templates as T
from sms_profiling.extract_information import INFO_COL, RegexExtractor, extract_information
import pipeline as PL

BRANCHES = ["Narayanganj", "Gazipur", "Khulna", "Savar", "Kushtia", "Ltd Barishal"]
POP = "Dear {n}, thanks for choosing Popular Diagnostic Centre {b} as your healthcare partner. Your Registration ID is {i}."
CAT = "Medical / Healthcare"


def pop(b, k): return POP.format(n=f"Rahim{k}", b=b, i=400000000 + k)
def trow(tid, template, n, cat="Med"): return dict(category=cat, template_id=tid, template=template, message_count=n, share_pct=1.0, example="e")
T_POP = "[VAR] thanks for choosing Popular Diagnostic Centre {b} as your healthcare partner Your Registration ID is [NUM]"


def test_family_keys():
    k = T.family_keys("Dear [VAR] your OTP is [NUM] valid for [NUM] minutes")
    assert k[0] == "Dear [X] your OTP is [X] valid for [X] minutes", "the first key is the exact normalised template (masked slots all [X])"
    assert "Dear [X] your OTP is * valid for [X] minutes" in k and "Dear [X] your OTP is [X] valid * minutes" in k
    assert T.family_keys("Your OTP is [NUM]") == ["Your OTP is [X]"], "fewer than 4 fixed words: no wildcard keys, short generic templates stay apart"
    assert T.family_keys("a [NUM] [VAR] [CODE] b")[0] == "a [X] b", "consecutive masked slots collapse"
    assert T._norm_tokens("[Pathao] [VAR] x") == ["[Pathao]", "[X]", "x"], "a bracketed brand is a word, a masked slot is not"
    print("OK  family keys: exact form + each 1-4 token run as a wildcard; masked slots unified; short templates get no wildcard")


def test_families_group_the_same_sentence_with_one_varying_part():
    rows = [trow(f"p{i}", T_POP.format(b=b), 50 - i) for i, b in enumerate(BRANCHES)]
    rows += [trow("pv", "[VAR] [NUM] [VAR] thanks for choosing Popular Diagnostic Centre Dhaka as your healthcare partner Your Registration ID is [NUM]", 5),
             trow("o1", "Your OTP is [NUM]", 100), trow("o2", "Your PIN is [NUM]", 100),
             trow("r1", "Your report is ready please collect it from the delivery counter of Popular Diagnostic Centre", 40),
             trow("r2", "Your invoice is ready please collect it from the delivery counter of Popular Diagnostic Centre", 30)]
    reg = T.add_families(pd.DataFrame(rows))
    fam = reg.set_index("template_id").family_id
    pops = ["p0", "p1", "p2", "p3", "p4", "p5", "pv"]
    assert len({fam[t] for t in pops}) == 1 and fam["p0"] != "p0", "all branches (also a 2-word one and one with a different name slot) are ONE family"
    key = reg.set_index("template_id").family_key["p0"]
    assert key == "[X] thanks for choosing Popular Diagnostic Centre * as your healthcare partner Your Registration ID is [X]", key
    assert fam["o1"] == "o1" and fam["o2"] == "o2", "short generic templates are not merged"
    assert fam["r1"] == fam["r2"] != "r1", "report / invoice differ in one word: same sentence, one family"
    assert fam["r1"] != fam["p0"]
    again = T.add_families(pd.DataFrame(rows[::-1]))
    assert dict(zip(again.template_id, again.family_id)) == dict(fam), "deterministic, whatever the row order"
    more = T.add_families(pd.DataFrame(rows + [trow("p9", T_POP.format(b="Rajshahi"), 3)]))
    assert more.set_index("template_id").family_id["p9"] == fam["p0"] and more.set_index("template_id").family_id["p0"] == fam["p0"], "a new branch joins; the family id does not change"
    # categories never mix
    two = T.add_families(pd.DataFrame(rows[:2] + [trow("x0", T_POP.format(b="Narayanganj"), 5, "Other"), trow("x1", T_POP.format(b="Gazipur"), 5, "Other")]))
    f2 = two.set_index("template_id").family_id
    assert f2["p0"] == f2["p1"] and f2["x0"] == f2["x1"] and f2["p0"] != f2["x0"]
    print("OK  families: branches (also multi-word / other name slot) merge; OTP vs PIN stay apart; deterministic; a new branch keeps the family id; categories never mix")


def test_family_registry_and_selection():
    rows = [trow(f"p{i}", T_POP.format(b=b), 50 - i) for i, b in enumerate(BRANCHES)] + [trow("o1", "Your OTP is [NUM]", 100)]
    reg = T.add_families(pd.DataFrame(rows))
    fr = T.family_registry(reg)
    assert len(fr) == 2 and set(fr.columns) >= {"category", "template_id", "template", "message_count", "members", "family_key"}
    big = fr[fr.template_id != "o1"].iloc[0]
    assert big.message_count == sum(50 - i for i in range(6)) and big.members == [f"p{i}" for i in range(6)], "biggest member first"
    assert "[*]" in big.template and "Narayanganj" not in big.template and "Popular Diagnostic Centre [*] as your" in big.template
    # a registry written before families existed (no family columns): every template is a family of one
    plain = T.family_registry(pd.DataFrame(rows))
    assert len(plain) == 7 and all(len(m) == 1 for m in plain.members)
    by_template, got_t = T.select_templates(reg, "Med", 0.8)                    # one regex per template: 4 of them
    by_family, got_f = T.select_templates(fr, "Med", 0.8)                       # one regex per family: 2 of them
    assert len(by_family) < len(by_template) and got_f >= got_t, (len(by_template), len(by_family), got_t, got_f)
    print("OK  family registry: members, summed counts, [*] in the displayed template; old registries work; selection needs fewer regexes for the same coverage")


def synthetic_classified():
    rows, k = [], 0
    for b in BRANCHES:
        for _ in range(8):
            k += 1; rows.append({"rid": k, "message_decoded": pop(b, k), "Predicted_Category": CAT})
    for _ in range(6):
        k += 1; rows.append({"rid": k, "message_decoded": f"Your OTP is {100000 + k}", "Predicted_Category": CAT})
    return pd.DataFrame(rows)


def run_templates_stage():
    d = tempfile.mkdtemp()
    synthetic_classified().to_csv(os.path.join(d, "classified.csv"), index=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        PL.stage_templates(PL.build_parser().parse_args(["templates", "--out-dir", d]))
    return d, buf.getvalue()


def test_templates_stage_and_regex_plan():
    d, out = run_templates_stage()
    reg = pd.read_csv(os.path.join(d, "templates_registry.csv"))
    assert {"family_id", "family_key"} <= set(reg.columns) and reg.family_id.nunique() == 2, reg[["template_id", "family_id"]]
    assert "families" in out and "7 templates in 2 families" in out, out[-300:]
    rep = pd.read_csv(os.path.join(d, "template_report.csv"))
    assert rep.loc[0, "families"] == 2 and "families_for_80pct_msgs" in rep.columns
    mt = pd.read_csv(os.path.join(d, "message_templates.csv"))
    assert set(mt.columns) == {"rid", "Predicted_Category", "template_id", "message_decoded"}, "the message file keeps its columns"
    def plan(*extra):
        buf = io.StringIO()
        with redirect_stdout(buf):
            PL.stage_regex(PL.build_parser().parse_args(["regex", "--out-dir", d, "--all", "--dry-run", "--min-messages", "1", "--coverage", "1.0", *extra]))
        return buf.getvalue()
    fam_out, tpl_out = plan(), plan("--no-families")
    assert "total: 2 regexes to write" in fam_out and "total: 7 regexes to write" in tpl_out, (fam_out[-120:], tpl_out[-120:])
    print("OK  templates stage writes family_id / family_key and a family count; regex plan: 2 regexes with families, 7 with --no-families")


def test_extractor_finds_the_family_regex_also_for_a_branch_never_seen():
    d, _ = run_templates_stage()
    from sms_profiling.templates import TemplateModel
    model = TemplateModel.load(os.path.join(d, "template_model.json"))
    reg = pd.read_csv(os.path.join(d, "templates_registry.csv"))
    fam = reg[reg.family_key.notna() & (reg.family_key != "")]
    fid, key = fam.family_id.iloc[0], fam.family_key.iloc[0]
    members = fam.sort_values("message_count", ascending=False).template_id.tolist()
    RX = r"Dear (?P<name>.+?),\s+thanks for choosing Popular Diagnostic Centre\s+(?P<branch>.+?)\s+as your healthcare partner\.\s+Your Registration ID is\s+(?P<registration_id>\d+)"
    registry = {fid: {"category": CAT, "pattern": "p", "expanded_pattern": RX, "fields": {}, "members": members, "family_key": key,
                      "facts": {"organisation": "Popular Diagnostic Centre"}, "event": "registration_confirmation"}}
    ex = RegexExtractor(model, registry)
    info, rx, why = ex.extract_one(pop("Gazipur", 7), CAT)                      # a member of the family
    assert why == "extracted" and info["branch"] == "Gazipur" and info["organisation"] == "Popular Diagnostic Centre" and info["registration_id"] == "****0007", info
    info, rx, why = ex.extract_one(pop("Ltd Barishal", 9), CAT)
    assert info["branch"] == "Ltd Barishal", info
    new = ex.extract_one(pop("Cox's Bazar", 11), CAT)                           # a branch no template has ever seen
    assert new[2] == "extracted" and new[0]["branch"] == "Cox's Bazar" and new[0]["event"] == "registration_confirmation", new
    assert ex.extract_one(pop("Chattogram", 12), CAT)[0]["branch"] == "Chattogram"
    tid = model.template_id(pop("Chattogram", 12), CAT)
    assert tid not in registry and ex.family_of(tid, pop("Chattogram", 12), CAT) == fid and (CAT, tid) in ex._family_cache, "found by the family key, then cached"
    other = ex.extract_one("Your OTP is 123456", CAT)
    assert other[2] == "template has no regex yet", "a different sentence is not caught by the family"
    stranger = ex.extract_one("Dear X, thanks for choosing Popular Diagnostic Centre Y as your healthcare partner. Your invoice is 5.", CAT)
    assert stranger[2] == "template has no regex yet"
    out = extract_information(synthetic_classified(), ex)
    assert out[INFO_COL].notna().sum() == 48 and {json.loads(s)["branch"] for s in out[INFO_COL].dropna()} == set(BRANCHES)
    print("OK  extractor: a family's regex serves every member and a branch that no template has ever seen; other sentences are left alone")


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        fn()
    print("\nALL FAMILY TESTS PASSED")
