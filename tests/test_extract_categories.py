"""Offline tests of extract_categories.py (DataFrame + category names in, information out). The model is replaced by a stub that writes the regex."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json, tempfile
import pandas as pd
try:
    from sms_profiling import regex_writer as RW
except ModuleNotFoundError as e:
    print(f"SKIPPED: {e}. Run with ~/venvs/regex_writer/bin/python"); sys.exit(77)
import extract_categories as EC
from sms_profiling import model_server as ms
os.environ["VLLM_NO_RESTART"] = "1"          # no test may ever start or stop a real server

MED, EDU, LOAN = "Medical / Healthcare", "Education", "Loan / EMI / Installment"
POP = "Dear {n}, thanks for choosing Popular Diagnostic Centre {b} as your healthcare partner. Your Registration ID is {i}."
BR = ["Narayanganj", "Gazipur", "Khulna", "Savar", "Kushtia", "Ltd Barishal"]
FAM_RX = r"Dear\s+(?P<name>.+?),\s+thanks for choosing Popular Diagnostic Centre\s+(?P<branch>.+?)\s+as your healthcare partner\.\s+Your Registration ID is\s+(?P<registration_id>\d+)"


def build_df():
    rows = [(f"Rahim{k}", POP.format(n=f"Rahim{k}", b=b, i=400000000 + 100 * j + k), MED) for j, b in enumerate(BR) for k in range(8)]
    rows += [(None, f"Your admission fee Tk {1000 + 50 * k}.00 is due on 06-06-26", EDU) for k in range(12)]
    rows += [(None, f"Tk {500 + k} paid", LOAN) for k in range(5)]
    rows += [(None, None, MED)]                                                    # a row without text
    d = pd.DataFrame(rows, columns=["who", "message_decoded", "Predicted_Category"])
    d["rid"] = range(len(d))
    return d.sample(frac=1, random_state=3)                                        # shuffled: the index is not 0..n-1


calls = []
env_seen = []
budgets = []


def fake_run_templates(ws, ids, model, **kw):
    """What the agent would do: write FAM_RX for the Popular family, fail for anything else (respecting skip_done like the real one)."""
    calls.append({"ids": list(ids), "model": model, **kw})
    env_seen.append(os.environ.get("LLM_REASONING_EFFORT"))
    budgets.append(ws.budget_s)
    reg, res = RW.load_registry(), {}
    for i in ids:
        if kw["skip_done"] and i in reg:
            res[i] = "already saved"; continue
        if "Popular" not in ws.registry.at[i, "template"]:
            ws.last_problems[i] = ["match_rate 0.40 < 0.95: see not_matched"]
            res[i] = "no regex saved (checks not passed)"; continue
        reg[i] = {"category": ws.category_of(i), "template": ws.registry.at[i, "template"], "pattern": FAM_RX, "expanded_pattern": FAM_RX,
                  "fields": {}, "members": ws.members.get(i, [i]), "family_key": ws.family_keys.get(i, ""),
                  "facts": {"organisation": "Popular Diagnostic Centre"}, "event": "registration_confirmation", "unextracted": {}}
        RW.REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
        res[i] = "SAVED"
    return res


def with_stubs(fn):
    saved = (RW.run_templates, RW.check_backend)
    RW.run_templates, RW.check_backend = fake_run_templates, lambda backend, model: None
    calls.clear(); env_seen.clear()
    try:
        return fn()
    finally:
        RW.run_templates, RW.check_backend = saved


def run(df, cats, work, **kw):
    kw.setdefault("model", "test-model"); kw.setdefault("verbose", False)
    return with_stubs(lambda: EC.extract_categories(df, cats, work_dir=work, **kw))


def test_one_category_end_to_end():
    df = build_df(); before = df.copy(deep=True); work = tempfile.mkdtemp()
    out = run(df, MED, work)
    assert df.equals(before) and "information" not in df.columns, "the input DataFrame is not modified"
    assert len(out) == len(df) and (out.index == df.index).all() and list(out.columns) == list(df.columns) + ["information", "regex_pattern"]
    med = out[out.Predicted_Category == MED]
    ok = med[med.message_decoded.notna()]
    assert len(ok) == 48 and ok.information.notna().all() and med[med.message_decoded.isna()].information.isna().all(), "every Medical message with text is read"
    info = ok.information.map(json.loads)
    assert {i["branch"] for i in info} == set(BR) and all(i["organisation"] == "Popular Diagnostic Centre" and i["event"] == "registration_confirmation" for i in info)
    assert all(i["registration_id"].startswith("****") for i in info), "the registration id is masked like everywhere else"
    assert (ok.regex_pattern == FAM_RX).all()
    assert out[out.Predicted_Category != MED][["information", "regex_pattern"]].isna().all().all(), "other categories are not touched"
    assert len(calls) == 1 and calls[0]["model"] == "test-model" and calls[0]["workers"] == 3 and calls[0]["think"] is True and calls[0]["skip_done"] is True
    s = out.attrs["regex_summary"]
    assert list(s.category) == [MED] and int(s.saved.iloc[0]) == 1 and int(s.failed.iloc[0]) == 0
    reg = json.load(open(os.path.join(work, "regex_registry.json"), encoding="utf-8"))
    assert len(reg) == 1 and len(next(iter(reg.values()))["members"]) == 6, "one regex for the family of 6 branch templates"
    assert sorted(os.listdir(work)) == ["regex_registry.json"], "only the regex registry is written to work_dir"
    print("OK  one category: all rows kept in order, Medical read through ONE family regex, other categories empty, input untouched, only the registry written")


def test_second_call_reuses_the_regexes_and_redo_rewrites_them():
    df, work = build_df(), tempfile.mkdtemp()
    a = run(df, MED, work)
    n_before = len(json.load(open(os.path.join(work, "regex_registry.json"), encoding="utf-8")))
    b = run(df, MED, work)
    assert calls[0]["skip_done"] is True and a[["information", "regex_pattern"]].equals(b[["information", "regex_pattern"]])
    assert len(json.load(open(os.path.join(work, "regex_registry.json"), encoding="utf-8"))) == n_before
    assert int(b.attrs["regex_summary"].saved.iloc[0]) == 1, "'already saved' counts as saved"
    run(df, MED, work, redo=True)
    assert calls[0]["skip_done"] is False
    print("OK  a second call finds the regexes already saved; redo=True writes them again")


def test_list_of_categories_fallback_and_options():
    df, work = build_df(), tempfile.mkdtemp()
    out = run(df, [MED, EDU], work)
    edu = out[out.Predicted_Category == EDU]
    assert len(edu) == 12 and edu.information.notna().all() and (edu.regex_pattern == "generic-fallback").all(), "no regex for Education: the generic fallback reads it"
    assert all(set(json.loads(x)) == {"fee", "currency", "due_date"} for x in edu.information), "the fee, its currency and the due date, from the words around them"
    assert out[out.Predicted_Category == LOAN].information.isna().all(), "a category that was not named is untouched"
    s = out.attrs["regex_summary"].set_index("category")
    assert s.loc[MED, "saved"] == 1 and s.loc[EDU, "failed"] == 1, s
    off = run(df, [MED, EDU], work, fallback=False)
    assert off[off.Predicted_Category == EDU].information.isna().all() and off[off.Predicted_Category == MED].information.notna().sum() == 48
    sel = run(df, [MED, EDU], work, only_selected=True)
    assert set(sel.Predicted_Category) == {MED, EDU} and len(sel) == 48 + 12
    one = run(df, (c for c in [EDU]), work)                                         # any iterable of names
    assert one[one.Predicted_Category == EDU].information.notna().all()
    print("OK  a list of categories; categories with no regex fall back to the generic reading; fallback=False, only_selected=True and iterables work")


def test_reasoning_effort_and_model_detection():
    df, work = build_df(), tempfile.mkdtemp()
    os.environ.pop("LLM_REASONING_EFFORT", None)
    run(df, MED, work, model="qwen3.8-27b")
    assert env_seen == ["medium"] and "LLM_REASONING_EFFORT" not in os.environ, (env_seen, "restored after the call")
    run(df, MED, work, model="qwen3.5-9b", redo=True)
    assert env_seen == [None]
    run(df, MED, work, model="qwen3.5-9b", redo=True, reasoning_effort="low")
    assert env_seen == ["low"]
    # the model the server serves is found by itself
    class R:
        def __init__(self, names): self.names = names
        def json(self): return {"data": [{"id": n} for n in self.names]}
    real = RW.requests.get
    try:
        RW.requests.get = lambda url, timeout=0: R(["only-model"])
        assert EC._served_model(RW) == "only-model"
        RW.requests.get = lambda url, timeout=0: R(["a", "b"])
        try:
            EC._served_model(RW); raise AssertionError("no error")
        except RuntimeError as e:
            assert "model=" in str(e)
        def down(url, timeout=0): raise ConnectionError("refused")
        RW.requests.get = down
        try:
            EC._served_model(RW); raise AssertionError("no error")
        except RuntimeError as e:
            assert "start_vllm.sh" in str(e)
    finally:
        RW.requests.get = real
    saved = RW.run_templates, RW.check_backend, RW.requests.get
    try:
        RW.run_templates = fake_run_templates
        RW.requests.get = lambda url, timeout=0: R(["served-name"])
        RW.check_backend = lambda backend, model: calls.append(("check", model))
        calls.clear()
        EC.extract_categories(df, MED, work_dir=tempfile.mkdtemp(), verbose=False)
        assert calls[0] == ("check", "served-name") and calls[1]["model"] == "served-name", calls
        def refuse(backend, model): raise SystemExit("Model 'x' not served")
        RW.check_backend = refuse
        try:
            EC.extract_categories(df, MED, work_dir=tempfile.mkdtemp(), verbose=False, model="x", auto_restart=False); raise AssertionError("no error")
        except RuntimeError as e:
            assert "not served" in str(e), "the command-line SystemExit becomes a normal exception"
    finally:
        RW.run_templates, RW.check_backend, RW.requests.get = saved
    print("OK  reasoning effort 'medium' for Qwen3.8 (restored after the call); the served model is found; server problems raise clear errors")


def test_a_server_that_goes_away_stops_the_run_loudly():
    df, work = build_df(), tempfile.mkdtemp()
    saved = RW.run_templates, RW.check_backend
    RW.check_backend = lambda backend, model: None
    RW.run_templates = lambda ws, ids, model, **kw: {i: "error: OpenAIConnectionError: Connection error." for i in ids}
    try:
        try:
            EC.extract_categories(df, [MED, EDU], work_dir=work, model="m", verbose=False); raise AssertionError("no error")
        except RuntimeError as e:
            assert "stopped answering" in str(e) and "regex_registry.json" in str(e) and "same arguments" in str(e), str(e)
    finally:
        RW.run_templates, RW.check_backend = saved
    # an ordinary failure (the checks did not pass) is not a reason to stop: that family just has no regex, and the user is warned
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = with_stubs(lambda: EC.extract_categories(df, [MED, EDU], work_dir=work, model="m", verbose=True))
    assert "WARNING: Education: 1 of 1 families have no regex" in buf.getvalue() and out[out.Predicted_Category == EDU].information.notna().all()
    print("OK  a server that stops answering aborts the call with a clear message (saved regexes are kept); an ordinary failed family only gives a warning")


def _restart_harness(stay_down=False):
    state = {"down": False, "runs": 0, "ensure": []}
    def check(backend, model):
        if state["down"]:
            raise SystemExit("Cannot reach vLLM")
    def run(ws, ids, model, **kw):
        state["runs"] += 1
        if state["runs"] == 1 or stay_down:
            state["down"] = True
            return {i: "error: OpenAIConnectionError: Connection error." for i in ids}
        return fake_run_templates(ws, ids, model, **kw)
    def ensure(url, want, recipe=None, wait_s=0, log=print):
        state["ensure"].append(want)
        state["down"] = stay_down
        return want or "m"
    return state, check, run, ensure


def test_the_server_is_restarted_and_the_run_carries_on():
    df, work = build_df(), tempfile.mkdtemp()
    state, check, run, ensure = _restart_harness()
    saved = RW.run_templates, RW.check_backend, ms.ensure_server
    RW.run_templates, RW.check_backend, ms.ensure_server = run, check, ensure
    try:
        out = EC.extract_categories(df, MED, work_dir=work, model="m", verbose=False)
    finally:
        RW.run_templates, RW.check_backend, ms.ensure_server = saved
    assert state["ensure"] == ["m"] and out.attrs["server_restarts"] == 1 and state["runs"] == 2, state
    med = out[(out.Predicted_Category == MED) & out.message_decoded.notna()]
    assert med.information.notna().all() and (med.regex_pattern == FAM_RX).all(), "after the restart the family got its regex"
    # the restarts are limited: a server that never comes back ends the call with a clear message
    state, check, run, ensure = _restart_harness(stay_down=True)
    RW.run_templates, RW.check_backend, ms.ensure_server = run, check, ensure
    try:
        try:
            EC.extract_categories(df, MED, work_dir=tempfile.mkdtemp(), model="m", verbose=False, max_restarts=2); raise AssertionError("no error")
        except RuntimeError as e:
            assert "2 automatic restart(s) used" in str(e) and "stopped answering" in str(e), str(e)
        assert len(state["ensure"]) == 2
        state, check, run, ensure = _restart_harness()
        RW.run_templates, RW.check_backend, ms.ensure_server = run, check, ensure
        try:
            EC.extract_categories(df, MED, work_dir=tempfile.mkdtemp(), model="m", verbose=False, auto_restart=False); raise AssertionError("no error")
        except RuntimeError as e:
            assert "0 automatic restart(s) used" in str(e) and state["ensure"] == [], "auto_restart=False never starts anything"
        # the server is down already at the start: it is started before anything is written (and a server with another model is not touched)
        state = {"ensure": []}
        def check_down(backend, model): raise SystemExit("Cannot reach vLLM")
        RW.run_templates, RW.check_backend = fake_run_templates, check_down
        ms.ensure_server = lambda url, want, recipe=None, wait_s=0, log=print: state["ensure"].append((want, recipe)) or "m"
        out = EC.extract_categories(df, MED, work_dir=tempfile.mkdtemp(), model="m", verbose=False, start_recipe={"SERVED": "m"})
        assert state["ensure"] == [("m", {"SERVED": "m"})] and out.information.notna().sum() == 48
    finally:
        RW.run_templates, RW.check_backend, ms.ensure_server = saved
    print("OK  a server that stops (or is down at the start) is started again and the run carries on; restarts are limited; auto_restart=False starts nothing")


def test_a_budget_and_the_report_of_what_still_has_no_regex():
    df, work = build_df(), tempfile.mkdtemp()
    seen = []
    def run(ws, ids, model, **kw):
        seen.append(list(ids))
        return fake_run_templates(ws, ids, model, **kw)
    saved = RW.run_templates, RW.check_backend
    RW.run_templates, RW.check_backend = run, lambda b, m: None
    try:
        out1 = EC.extract_categories(df, [MED, EDU], work_dir=work, model="m", verbose=False, max_new=1)
        assert len(seen) == 1 and len(seen[0]) == 1, "a budget of one new regex: the second category is not written in this call"
        assert out1[out1.Predicted_Category == EDU].information.notna().all(), "...its messages are read by the generic fallback meanwhile"
        seen.clear()
        EC.extract_categories(df, [MED, EDU], work_dir=work, model="m", verbose=False, max_new=1)
        assert len(seen) == 1 and RW.load_registry() and all("Popular" not in str(i) for i in seen[0]), "the next call spends the budget on what is NEW"
        seen.clear()
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = EC.extract_categories(df, [MED, EDU], work_dir=work, model="m", verbose=True)
    finally:
        RW.run_templates, RW.check_backend = saved
    ff = out.attrs["failed_families"]
    assert list(ff.columns) == ["category", "family_id", "messages", "result", "blocked_by", "template"] and len(ff) == 1 and ff.category.iloc[0] == EDU, ff
    assert ff.messages.iloc[0] == 12 and ff.blocked_by.iloc[0].startswith("match_rate 0.40"), ff.iloc[0].to_dict()
    assert "families still without a regex" in buf.getvalue() and ff.family_id.iloc[0] in buf.getvalue() and "match_rate 0.40" in buf.getvalue()
    assert out.attrs["regex_summary"].set_index("category").loc[EDU, "failed"] == 1 and "partial" in out.attrs["regex_summary"].columns
    print("OK  max_new spends a time budget on the biggest NEW families only; the report lists the families still without a regex and the check that blocked them")


def test_no_model_needed_when_nothing_is_to_be_written():
    df = build_df()
    saved = RW.run_templates, RW.check_backend, RW.requests.get
    def boom(*a, **k): raise AssertionError("the server must not be contacted")
    try:
        RW.run_templates, RW.check_backend, RW.requests.get = boom, boom, boom
        out = EC.extract_categories(df, EDU, work_dir=tempfile.mkdtemp(), min_messages=1000, verbose=False)     # no family is big enough: nothing to write
    finally:
        RW.run_templates, RW.check_backend, RW.requests.get = saved
    assert out[out.Predicted_Category == EDU].information.notna().all()
    print("OK  when no regex has to be written the vLLM server is not needed (generic fallback only)")


def test_errors_and_csv_input():
    df, work = build_df(), tempfile.mkdtemp()
    for bad, exc, text in [(([1, 2], MED), TypeError, "DataFrame"), ((df.drop(columns=["message_decoded"]), MED), KeyError, "message_decoded"),
                           ((df.drop(columns=["Predicted_Category"]), MED), KeyError, "Predicted_Category"), ((df, []), ValueError, "at least one"),
                           ((df, "No Such Category"), ValueError, "Medical / Healthcare  (49 rows)"), ((df, [MED, "Nope"]), ValueError, "'Nope'")]:
        try:
            with_stubs(lambda: EC.extract_categories(bad[0], bad[1], work_dir=work, model="m", verbose=False)); raise AssertionError(f"no error for {text}")
        except exc as e:
            assert text in str(e), (text, str(e)[:200])
    path = os.path.join(tempfile.mkdtemp(), "classified.csv"); df.to_csv(path, index=False)
    out = run(path, MED, work)                                                         # a CSV path instead of a DataFrame
    assert out.information.notna().sum() == 48
    print("OK  clear errors (not a DataFrame, missing column, unknown category lists the real ones with row counts, no category); a CSV path works")


def test_field_summary():
    df, work = build_df(), tempfile.mkdtemp()
    out = run(df, [MED, EDU], work)
    s = EC.field_summary(out, MED).set_index("field")
    assert s.loc["branch", "rows"] == 48 and s.loc["branch", "from_regex"] == 48 and s.loc["branch", "from_generic"] == 0 and s.loc["branch", "share_%"] == 100.0
    e = EC.field_summary(out, EDU)
    assert (e.from_regex == 0).all() and (e.from_generic > 0).all() and set(e.field) == {"fee", "currency", "due_date"}
    assert {"branch", "due_date"} <= set(EC.field_summary(out).field), "without a category: every row"
    assert EC.field_summary(out, LOAN).empty
    print("OK  field_summary: which fields were extracted, how often, from the regex or from the generic fallback")


def test_time_budget_and_retry_failed_reach_the_writer():
    df = build_df()
    with_stubs(lambda: EC.extract_categories(df, MED, work_dir=tempfile.mkdtemp(), model="m", verbose=False))
    assert budgets[-1] == RW.FAMILY_BUDGET_S and calls[-1]["retry_failed"] is False, "defaults: 20 minutes per family, failed families are left alone"
    out = with_stubs(lambda: EC.extract_categories(df, MED, work_dir=tempfile.mkdtemp(), model="m", verbose=False, family_seconds=77, retry_failed=True))
    assert budgets[-1] == 77 and calls[-1]["retry_failed"] is True
    assert {"partial", "waived"} <= set(out.attrs["regex_summary"].columns)
    print("OK  family_seconds and retry_failed reach the regex writer; the summary counts the regexes accepted under the relaxed limit (waived)")


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        fn()
    print("\nALL EXTRACT_CATEGORIES TESTS PASSED")
