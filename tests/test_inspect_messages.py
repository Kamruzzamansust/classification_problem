"""Offline tests of inspect_messages.py: category list, sampling filters, rendering, read-only CLI."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import hashlib, json, os, subprocess, sys, tempfile
import pandas as pd
from sms_profiling.extract_information import RegexExtractor
from sms_profiling.inspect_messages import list_categories, list_regexes, render, render_regexes, sample_category, show_regexes
from test_extract_information import BANK, BANK_RX, OTP, TEL, TEL_RX, build



def setup():
    df, model, reg = build()
    reg.pop(next(k for k, v in reg.items() if v["category"] == OTP))          # OTP: templates exist, no regex yet
    return df, RegexExtractor(model, reg), reg, model


def test_generic_fallback_is_shown_for_rows_without_a_regex():
    df, ex, reg, model = setup()
    bare = RegexExtractor(model, {})                                       # no template regex at all
    items = sample_category(df, bare, TEL, n=2, seed=1)
    assert all(i["information"] is None and i["generic"] and "balance" in i["generic"] for i in items), items[0]
    text = render(items, TEL)
    assert "generic fallback" in text and "[generic fallback]" in text and "balance" in text and "no regex has been written" in text, text[:600]
    assert sample_category(df, bare, TEL, n=2, only="matched") == [] and len(sample_category(df, bare, TEL, n=2, only="unmatched")) == 2
    t = list_categories(df, bare).set_index("category")
    assert t.loc[TEL, "messages_with_information_%"] == 0.0 and t.loc[TEL, "plus_generic_fallback_%"] == 100.0
    print("OK  inspect: rows without a template regex show what the generic fallback reads, marked as such; the category list counts it separately")


def test_category_list():
    df, ex, reg, _ = setup()
    t = list_categories(df, ex).set_index("category")
    assert list(t.columns) == ["messages", "regexes", "messages_with_information_%", "plus_generic_fallback_%"]
    assert t.loc[TEL, "messages"] == 7 and t.loc[TEL, "regexes"] == 1 and t.loc[TEL, "messages_with_information_%"] == 100.0
    assert t.loc[BANK, "messages_with_information_%"] == 100.0 and t.loc[OTP, "regexes"] == 0 and t.loc[OTP, "messages_with_information_%"] == 0.0
    print("OK  category list: messages, regexes written, share of messages that get information")


def test_sampling_and_filters():
    df, ex, reg, model = setup()
    items = sample_category(df, ex, TEL, n=3, seed=1)
    assert len(items) == 3 and all(i["row"]["Predicted_Category"] == TEL for i in items)
    for i in items:
        assert i["information"] and i["regex"] == TEL_RX and i["template"] and len(i["template_id"]) == 12 and i["reason"] == "extracted"
        assert i["template_id"] == model.template_id(i["row"]["message_decoded"], TEL) and i["template"] == model.transform(i["row"]["message_decoded"], TEL)
    assert [i["row"]["rid"] for i in sample_category(df, ex, TEL, 3, seed=1)] == [i["row"]["rid"] for i in items], "same seed -> same sample"
    assert [i["row"]["rid"] for i in sample_category(df, ex, TEL, 3, seed=2)] != [i["row"]["rid"] for i in items], "another seed -> another sample"
    un = sample_category(df, ex, OTP, n=5, only="unmatched")
    assert len(un) == 3 and all(i["information"] is None and i["regex"] is None and i["reason"] == "template has no regex yet" and i["template"] for i in un)
    assert sample_category(df, ex, OTP, n=5, only="matched") == [], "no matched rows in a category without regexes"
    assert len(sample_category(df, ex, TEL, n=50, only="matched")) == 7, "asking for more than exist returns what exists"
    assert sample_category(df, ex, TEL, n=5, only="unmatched") == []
    try:
        sample_category(df, ex, "Nope"); raise AssertionError("no error")
    except ValueError as e:
        assert "Categories:" in str(e)
    # a message whose wording differs: template exists for the category but this message has no regex / does not match
    odd = pd.concat([df, pd.DataFrame([(9, "Akash something totally different", TEL)], columns=df.columns)], ignore_index=True)
    u = sample_category(odd, ex, TEL, n=5, only="unmatched")
    assert len(u) == 1 and u[0]["reason"] == "template has no regex yet" and u[0]["information"] is None
    print("OK  sampling: reproducible by seed, filters all / matched / unmatched, template id + text correct, unknown category explained")


def test_rendering():
    df, ex, reg, _ = setup()
    txt = render(sample_category(df, ex, TEL, 2), TEL)
    assert "EXTRACTED" in txt and "message" in txt and "template" in txt and "(id " in txt and "regex" in txt and "information" in txt and "[1/2]" in txt
    first = sample_category(df, ex, TEL, 1)[0]
    assert json.dumps(first["information"], ensure_ascii=False) in render([first], TEL).replace("\n" + " " * 16, "")
    assert "(?P<account_id>" in render([first], TEL).replace("\n" + " " * 16, "")
    empty = render(sample_category(df, ex, OTP, 1, only="unmatched"), OTP)
    assert "empty" in empty and "(none)" in empty and "no regex has been written for this template yet" in empty
    masked = render(sample_category(df, ex, TEL, 1), TEL, mask=True)
    assert "01712" not in masked.split("template")[0] or "[NUM]" in masked.split("template")[0], masked[:300]
    unmasked = render(sample_category(df, ex, TEL, 1), TEL, mask=False)
    assert "০" not in "" and "01712" in unmasked
    print("OK  rendering: message / template / regex / information shown; empty rows say why; --mask hides long numbers")


def test_cli_is_read_only():
    df, ex, reg, model = setup()
    d = tempfile.mkdtemp()
    df.to_csv(os.path.join(d, "classified.csv"), index=False); model.save(os.path.join(d, "template_model.json"))
    json.dump(reg, open(os.path.join(d, "regex_registry.json"), "w", encoding="utf-8"), ensure_ascii=False)
    digest = lambda: {f: hashlib.sha1(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d))}
    before = digest()
    run = lambda *args: subprocess.run([sys.executable, "-m", "sms_profiling.inspect_messages", "--out-dir", d, *args], capture_output=True, text=True, cwd=ROOT)
    r = run()
    assert r.returncode == 0 and TEL in r.stdout and BANK in r.stdout and "messages_with_information_%" in r.stdout and "--category" in r.stdout, r.stderr[-300:]
    r = run("--category", TEL, "-n", "2")
    assert r.returncode == 0 and r.stdout.count("EXTRACTED") == 2 and "2 of 2 shown messages have information" in r.stdout, r.stdout[-400:]
    r = run("--category", OTP, "--only", "unmatched", "-n", "1")
    assert r.returncode == 0 and "empty" in r.stdout and "no regex has been written" in r.stdout
    r = run("--category", "Nope")
    assert r.returncode != 0 and "no rows for category" in r.stderr
    r = run("--category", TEL, "--only", "unmatched")
    assert r.returncode == 0 and "no unmatched messages found" in r.stdout
    assert digest() == before, "the inspector must not change any file"
    print("OK  runs as a command (list, one category, unmatched with reasons, bad category); no file is modified")


def test_regex_overview_shows_the_fields_each_regex_extracts():
    from sms_profiling.templates import generate_all
    df, ex, reg, model = setup()
    for e in reg.values():                                        # field descriptions as the agent saves them
        e["fields"] = {"account_id": "the customer's account id", "amount": "amount in BDT", "service_name": "service the code is for"}
    reg_otp = {k: v for k, v in reg.items() if v["category"] == OTP}
    assert reg_otp == {}, "setup removed the OTP regex"
    templates, _, _ = generate_all(df, "message_decoded", "Predicted_Category", "rid")
    items = list_regexes(ex, templates)
    assert {i["category"] for i in items} == {TEL, BANK} and len(items) == 2
    tel = next(i for i in items if i["category"] == TEL)
    names = {f["name"]: f for f in tel["fields"]}
    assert set(names) == {"account_id", "expiry_date", "current_balance"} and tel["messages"] == 7 and "Akash" in tel["template"]
    assert names["account_id"]["treatment"] == "masked to last 4 digits" and names["account_id"]["description"] == "the customer's account id"
    assert names["current_balance"]["treatment"] == "" and names["current_balance"]["description"] == ""
    bank = next(i for i in items if i["category"] == BANK)
    assert {f["name"]: f["treatment"] for f in bank["fields"]}["account_reference"] == "masked to last 4 digits"
    # a regex that captures a secret: it is listed, flagged as never output
    reg2 = dict(reg); k = next(iter(reg2)); reg2[k] = dict(reg2[k], expanded_pattern=r"(?P<service_name>\w+) code (?P<otp_code>\d{6})")
    flagged = {f["name"]: f["treatment"] for i in list_regexes(RegexExtractor(model, reg2), templates) for f in i["fields"]}
    assert flagged["otp_code"] == "never output (secret)"
    txt = render_regexes(items, {TEL: 60, BANK: 70})
    assert f"{TEL}: 1 regex(es) cover 7 messages of 60 (12%)" in txt and "[masked to last 4 digits]" in txt and "the customer's account id" in txt, txt[:600]
    assert "no regexes saved yet" in show_regexes_empty()
    print("OK  --regexes: per category the saved regexes, messages covered, each field with its description and privacy treatment")


def test_regex_self_check_reports_how_many_messages_it_really_extracts_from():
    from sms_profiling.templates import generate_all
    df, ex, reg, model = setup()
    templates, assignments, _ = generate_all(df, "message_decoded", "Predicted_Category", "rid")
    messages = assignments.rename(columns={})[["template_id", "message_decoded"]]
    ok = {i["category"]: i for i in list_regexes(ex, templates, None, messages)}
    assert ok[TEL]["works_on"] == 1.0 and ok[BANK]["works_on"] == 1.0
    assert "extracts from 100% of a sample" in render_regexes(list(ok.values()))
    tid = next(k for k, v in reg.items() if v["category"] == TEL)
    reg2 = dict(reg); reg2[tid] = dict(reg2[tid], expanded_pattern=r"(?P<x>NEVER_MATCHES_\d+)")           # a regex that is saved but never fires
    bad = {i["category"]: i for i in list_regexes(RegexExtractor(model, reg2), templates, None, messages)}
    assert bad[TEL]["works_on"] == 0.0 and bad[BANK]["works_on"] == 1.0
    txt = render_regexes(list(bad.values()))
    assert "WARNING extracts from only 0% of a sample" in txt and txt.count("WARNING") == 1
    assert list_regexes(ex, templates)[0]["works_on"] is None, "without the message file there is no self-check, and no false claim"
    print("OK  --regexes self-check: each regex reports the share of its template's messages it really extracts from; a dead regex gets a WARNING")


def test_regex_overview_warns_about_a_money_field_that_is_always_zero():
    from sms_profiling.templates import generate_all
    df, ex, reg, model = setup()
    templates, assignments, _ = generate_all(df, "message_decoded", "Predicted_Category", "rid")
    messages = assignments[["template_id", "message_decoded"]]
    assert all(i["warnings"] == [] for i in list_regexes(ex, templates, None, messages)), "good regexes raise nothing"
    tid = next(k for k, v in reg.items() if v["category"] == TEL)
    reg2 = dict(reg); reg2[tid] = dict(reg2[tid], expanded_pattern=r"হয়েছে\s+(?P<amount>0)\d-Jun-26")        # 'amount' = the 0 of a date: always zero
    items = {i["category"]: i for i in list_regexes(RegexExtractor(model, reg2), templates, None, messages)}
    assert items[TEL]["warnings"] and "0 in almost every message" in items[TEL]["warnings"][0] and items[BANK]["warnings"] == []
    assert "WARNING: field 'amount' is 0 in almost every message" in render_regexes(list(items.values()))
    print("OK  --regexes warns about a money field that is zero in (almost) every message: the regex reads the wrong part of the number")


def show_regexes_empty():
    d = tempfile.mkdtemp()
    df, ex, reg, model = setup()
    df.to_csv(os.path.join(d, "classified.csv"), index=False); model.save(os.path.join(d, "template_model.json"))
    json.dump({}, open(os.path.join(d, "regex_registry.json"), "w"))
    from pathlib import Path
    return show_regexes(Path(d), Path(d) / "classified.csv")


def test_regex_overview_command_line():
    df, ex, reg, model = setup()
    from sms_profiling.templates import generate_all
    templates, _, _ = generate_all(df, "message_decoded", "Predicted_Category", "rid")
    d = tempfile.mkdtemp()
    df.to_csv(os.path.join(d, "classified.csv"), index=False); model.save(os.path.join(d, "template_model.json")); templates.to_csv(os.path.join(d, "templates_registry.csv"), index=False)
    json.dump(reg, open(os.path.join(d, "regex_registry.json"), "w", encoding="utf-8"), ensure_ascii=False)
    before = {f: hashlib.sha1(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d))}
    for cmd in ([sys.executable, "-m", "sms_profiling.inspect_messages"], [sys.executable, os.path.join(ROOT, "pipeline.py"), "inspect"]):
        r = subprocess.run(cmd + ["--out-dir", d, "--regexes"], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0 and TEL in r.stdout and BANK in r.stdout and "account_id" in r.stdout and "[masked to last 4 digits]" in r.stdout, r.stderr[-300:]
        r = subprocess.run(cmd + ["--out-dir", d, "--regexes", "--category", BANK], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0 and BANK in r.stdout and TEL not in r.stdout
    assert {f: hashlib.sha1(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d))} == before, "read-only"
    print("OK  inspect --regexes works as a command (module and pipeline.py), narrows by --category, changes no file")


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        fn()
    print("\nALL INSPECT TESTS PASSED")
