"""extract_categories.py - give a classified DataFrame and the categories you want, get the information back. No command line.

    import pandas as pd
    from extract_categories import extract_categories, field_summary

    df = pd.read_csv("classify_data/classified.csv")
    result = extract_categories(df, "Loan / EMI / Installment")                          # one category
    result = extract_categories(df, ["Loan / EMI / Installment", "Education"])            # or a list
    result[["message_decoded", "information", "regex_pattern"]].head()
    field_summary(result, "Loan / EMI / Installment")                                     # which fields were extracted, and how often

What happens, for the categories you name (and only those):
  1. their messages are reduced to templates, and templates that differ only in one varying part (a branch, an ATM) are grouped into families;
  2. the families that cover 80% of the category get one regex each, written and tested by the Qwen agent: the vLLM server must be running
     (./start_vllm.sh, or the Qwen3.8-27B recipe in its header). Finished regexes are kept in work_dir/regex_registry.json, so a second call for
     the same categories does not call the model again;
  3. every message is read: by the regex of its template / family, else by the generic fallback (amounts, dates, times, link domains, phones).
It returns your DataFrame (all rows, same order, same columns) plus `information` (JSON text) and `regex_pattern`. Rows of other categories stay
empty (or pass only_selected=True to get just the rows of the chosen categories).

Run it with the regex_writer environment (~/venvs/regex_writer/bin/python, or a notebook kernel of that environment): it needs deepagents.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEFAULT_WORK_DIR = HERE / "output"


def _served_model(rw) -> str:
    """The model the vLLM server serves (when it serves exactly one), so its name does not have to be typed."""
    try:
        names = [m["id"] for m in rw.requests.get(f"{rw.VLLM_URL}/models", timeout=10).json()["data"]]
    except Exception as e:
        raise RuntimeError(f"Cannot reach the vLLM server at {rw.VLLM_URL} ({e}). Start it first: ./start_vllm.sh") from e
    if len(names) == 1:
        return names[0]
    if os.environ.get("LLM_MODEL") in names:
        return os.environ["LLM_MODEL"]
    raise RuntimeError(f"The vLLM server serves {names or 'no model'}: pass model=\"<name>\"")


def extract_categories(df: Union[pd.DataFrame, str, Path], categories: Union[str, Iterable[str]], *,
                       work_dir: Union[str, Path, None] = None, model: Optional[str] = None, workers: int = 3, think: bool = True,
                       reasoning_effort: Optional[str] = None, samples: int = 5, coverage: float = 0.8, min_messages: int = 5, retries: int = 2,
                       redo: bool = False, families: bool = True, fallback: bool = True, keep_identifiers: bool = False,
                       only_selected: bool = False, auto_restart: bool = True, max_restarts: int = 3, max_new: Optional[int] = None,
                       start_recipe: Optional[dict] = None, verbose: bool = True,
                       text_col: str = "message_decoded", category_col: str = "Predicted_Category") -> pd.DataFrame:
    """Write the regexes for `categories` (if they are not written yet) and extract the information of their messages.

    df              the classified DataFrame (or the path of its CSV): needs the message text and the category columns
    categories      one category name or a list of names
    work_dir        where regex_registry.json and the agent transcripts (logs/) are kept (default: output/ of this project). Pass
                    ".../classify_data" to share the regexes with the command-line pipeline. Only those two things are written there.
    model           name of the model the vLLM server serves (default: the one it serves)
    workers         families written in parallel (3 suits Qwen3.8-27B on the 24 GB card; 5 suits the 9B)
    think           let the model reason (slower, better regexes)
    reasoning_effort  "low" / "medium" / "xhigh" for Qwen3.8 (default: "medium" for Qwen3.8, the server's default for other models)
    coverage, min_messages  the families covering `coverage` of the category's messages get a regex; a family with fewer messages than
                    `min_messages` is skipped (a regex from 1-4 examples is usually overfitted)
    retries         what a family that does not pass the checks gets on its own: 1 = a deeper run (more and more different sample messages, more
                    steps, more reasoning), 2 = also a soft-landing run that accepts a regex reading 85% of the family and writes a second regex
                    for the messages it misses (default 2; 0 = a single run)
    auto_restart    start the vLLM server when it does not answer (at the beginning or in the middle of the run) and carry on, up to
                    `max_restarts` times; only on the machine that hosts the server. False = stop with a clear message instead.
    max_new         write at most this many new regexes in one call (the biggest families first): a time budget for a daily run; the rest is
                    read by the generic fallback until a later call
    redo            write the regexes again even where one is already saved
    families        False = one regex per template instead of one per family
    fallback        False = leave messages without a template regex empty instead of reading amounts, dates, times, links, phones
    keep_identifiers  True = keep full phone / account / user ids instead of the last four digits
    only_selected   True = return only the rows of `categories`
    """
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame or a CSV path, got {type(df).__name__}")
    for c in (text_col, category_col):
        if c not in df.columns:
            raise KeyError(f"column {c!r} not in the DataFrame; columns are {list(df.columns)}")
    cats = [categories] if isinstance(categories, str) else list(categories)
    if not cats:
        raise ValueError("give at least one category")
    known = df[category_col].value_counts()
    unknown = [c for c in cats if c not in known.index]
    if unknown:
        raise ValueError(f"unknown category {unknown}. Categories in this DataFrame:\n" + "\n".join(f"  {c}  ({n:,} rows)" for c, n in known.items()))

    try:
        from sms_profiling import regex_writer as rw
    except ImportError as e:
        raise ImportError(f"{e}\nRun this with the regex_writer environment: ~/venvs/regex_writer/bin/python") from e
    from sms_profiling.extract_information import INFO_COL, REGEX_COL, RegexExtractor, extract_information
    from sms_profiling import model_server as ms
    from sms_profiling.templates import CAT_COL, TEXT_COL, add_families, coverage_plan, family_registry, generate_all, select_templates

    # ---- 1. templates and families, for the chosen categories only (each category's vocabulary is fitted on its own messages)
    mask = (df[category_col].isin(cats) & df[text_col].notna()).to_numpy()
    positions = mask.nonzero()[0]
    sub = df.loc[mask, [text_col, category_col]].rename(columns={text_col: TEXT_COL, category_col: CAT_COL}).reset_index(drop=True)
    sub[TEXT_COL] = sub[TEXT_COL].astype(str).map(lambda s: unicodedata.normalize("NFC", s))   # one encoding per Bengali letter
    sub["_row"] = range(len(sub))
    templates, assignments, tmodel = generate_all(sub, TEXT_COL, CAT_COL, "_row")
    templates = add_families(templates)
    units = family_registry(templates) if families else templates
    plan = coverage_plan(units, cats, coverage, min_messages)
    if verbose:
        print(f"Plan: {'template families' if families else 'templates'} covering {coverage:.0%} of each category's messages (min {min_messages} msg each):")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(plan.to_string(index=False))
        print(f"total: {int(plan['selected'].sum()):,} regexes (those already saved are not written again)\n")

    # ---- 2. the regexes (only if there is something to write)
    work_dir = Path(work_dir) if work_dir else DEFAULT_WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    rw.set_output_dir(work_dir)
    summary, failed, restarts = [], [], 0
    if int(plan["selected"].sum()):
        def say(msg):
            if verbose:
                print(msg)

        def ready(want):
            """The model to use. When nothing answers, the server is started (again) first, if auto_restart is on."""
            try:
                name = want or _served_model(rw)
                rw.check_backend("vllm", name)                      # SystemExit when nothing answers or the model is not served
                return name
            except (RuntimeError, SystemExit) as e:
                if not auto_restart or "pass model=" in str(e):     # several models are served: the user has to choose
                    raise RuntimeError(str(e)) from e
                return ms.ensure_server(rw.VLLM_URL, want, start_recipe, log=say)

        model = ready(model)
        effort_before = os.environ.get("LLM_REASONING_EFFORT")
        effort = reasoning_effort or (None if "3.8" not in model else "medium")
        if effort:
            os.environ["LLM_REASONING_EFFORT"] = effort
        try:
            ws = rw.Workspace.from_frames(templates, assignments[["template_id", TEXT_COL]], n_samples=samples, families=families)
            budget = max_new
            for cat in cats:
                sel, got = select_templates(units, cat, coverage, min_messages)
                if sel.empty:
                    continue
                ids = sel["template_id"].tolist()
                if budget is not None:                                # a time budget: only the biggest NEW families
                    have = set(rw.load_registry())
                    ids = [i for i in ids if redo or i not in have][:budget]
                    budget -= len(ids)
                    if not ids:
                        continue
                say(f"\n=== {cat}: {len(ids)} {'template families' if families else 'templates'}, {got:.0%} of its messages ===")
                res, pending, first = {}, ids, True
                while True:
                    part = rw.run_templates(ws, pending, model, verbose=verbose, think=think, backend="vllm", workers=workers, retries=retries,
                                            skip_done=first and not redo)
                    res.update(part)
                    down = [i for i in pending if rw.is_connection_error(part.get(i))]
                    if not down:
                        break
                    if not auto_restart or restarts >= max_restarts:    # the server is gone for good: do not return fallback-only results as if all was well
                        raise RuntimeError(
                            f"The vLLM server stopped answering while the regexes of {cat!r} were written ({len(down)} of {len(pending)} families failed "
                            f"with connection errors; {restarts} automatic restart(s) used). Regexes that are already saved are kept in "
                            f"{rw.REGISTRY_PATH}. Start the server again and call extract_categories again with the same arguments: only the missing "
                            "regexes are written.")
                    restarts += 1
                    say(f"\nthe vLLM server stopped answering: restart {restarts} of {max_restarts} at most")
                    model = ready(model)
                    pending, first = down, False
                saved = sum(v in ("SAVED", "already saved") for v in res.values())
                reg_now = rw.load_registry()
                summary.append({"category": cat, "regexes_wanted": len(res), "saved": saved, "failed": len(res) - saved,
                                "partial": sum(bool(reg_now.get(i, {}).get("partial")) for i in res),
                                "second_regexes": sum(e.get("parent") in res for e in reg_now.values()), "coverage_pct": round(100 * got, 1)})
                for i, v in res.items():
                    if v not in ("SAVED", "already saved"):
                        failed.append({"category": cat, "family_id": i, "messages": int(ws.registry.at[i, "message_count"]), "result": str(v)[:60],
                                       "blocked_by": ((ws.last_problems.get(i) or ["the checks were not reached"])[0])[:200],
                                       "template": str(ws.registry.at[i, "template"])[:110]})
        finally:
            if effort_before is None:
                os.environ.pop("LLM_REASONING_EFFORT", None)
            else:
                os.environ["LLM_REASONING_EFFORT"] = effort_before

    # ---- 3. read every message of the chosen categories
    extractor = RegexExtractor(tmodel, rw.load_registry(), keep_identifiers)
    res_df = extract_information(sub, extractor, TEXT_COL, CAT_COL, categories=cats, fallback=fallback)
    infos, patterns = [None] * len(df), [None] * len(df)
    for pos, info, pat in zip(positions, res_df[INFO_COL], res_df[REGEX_COL]):
        infos[pos], patterns[pos] = info, pat
    out = df.copy()
    out[INFO_COL], out[REGEX_COL] = infos, patterns
    out.attrs["extraction_stats"] = res_df.attrs.get("extraction_stats", {})
    out.attrs["regex_summary"] = pd.DataFrame(summary)
    out.attrs["failed_families"] = pd.DataFrame(failed, columns=["category", "family_id", "messages", "result", "blocked_by", "template"]).sort_values(
        "messages", ascending=False, ignore_index=True)
    out.attrs["server_restarts"] = restarts
    if verbose:
        got = out[INFO_COL].notna() & out[category_col].isin(cats)
        print("\nrows with information:")
        for c in cats:
            n = int((out[category_col] == c).sum())
            print(f"   {c:36} {int((got & (out[category_col] == c)).sum()):7,} of {n:7,}")
        print(f"regexes are kept in {rw.REGISTRY_PATH}")
        for s in summary:
            if s["failed"]:
                print(f"WARNING: {s['category']}: {s['failed']} of {s['regexes_wanted']} families have no regex (checks not passed; see {rw.LOG_DIR}). "
                      "Their messages got the generic reading only.")
        for s in summary:
            if s.get("partial") or s.get("second_regexes"):
                print(f"{s['category']}: {s['partial']} family regex(es) accepted at 85% (the soft landing), {s['second_regexes']} second regex(es) for what they miss.")
        if failed:
            print("\nfamilies still without a regex, the biggest first, and the check that blocked them:")
            for r in out.attrs["failed_families"].head(10).itertuples():
                print(f"   {r.messages:6,} msgs | {r.category[:14]:14} | {r.family_id} | {r.blocked_by[:100]}")
        if restarts:
            print(f"the vLLM server was restarted {restarts} time(s) during this call.")
    return out.iloc[positions] if only_selected else out


def field_summary(result: pd.DataFrame, category: Optional[str] = None, category_col: str = "Predicted_Category") -> pd.DataFrame:
    """Which fields the information holds: for every field the number of rows, the share of the rows that have information, and how many of
    them come from a regex of the agent and how many from the generic fallback."""
    from sms_profiling.generic_extractors import GENERIC_PATTERN
    x = result if category is None else result[result[category_col] == category]
    x = x[x["information"].notna()]
    rows: dict[str, list[int]] = {}
    for info, pat in zip(x["information"], x["regex_pattern"]):
        for k in json.loads(info):
            r = rows.setdefault(k, [0, 0, 0])
            r[0] += 1
            r[2 if pat == GENERIC_PATTERN else 1] += 1
    out = pd.DataFrame([{"field": k, "rows": r[0], "share_%": round(100 * r[0] / max(len(x), 1), 1), "from_regex": r[1], "from_generic": r[2]}
                        for k, r in rows.items()], columns=["field", "rows", "share_%", "from_regex", "from_generic"])
    return out.sort_values("rows", ascending=False, ignore_index=True)
