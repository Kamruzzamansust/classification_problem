"""pipeline.py - SMS profiling: classify -> templates -> regex, for one category or all of them.

Stages (each can be run on its own; `run` does all four):
    classify   decode hex messages, flag the language, classify every message into a category
               (Gemma 12B via vLLM, offline; needs the GPU)                      -> output/classified.csv
    templates  per category, reduce messages to templates                        -> output/templates_registry.csv ...
    regex      per category, take the biggest templates that together cover 80% of its messages and let the
               Qwen agent write + validate one regex per template (needs the Qwen vLLM server running)
                                                                                 -> output/regex_registry.json
    extract    apply those regexes to every message: adds the columns `information` (JSON) and `regex_pattern`
               (script 2, sms_profiling/extract_information.py; no GPU, no LLM)                    -> output/extracted.csv

Examples
    python pipeline.py classify data/original_message_part_1.csv --limit 1000
    python pipeline.py classify data/original_message_part_1.csv --labels-from /path/earlier_results.csv   # reuse labels, no GPU
    python pipeline.py templates
    python pipeline.py regex --dry-run --all                          # what would be done, no LLM
    python pipeline.py regex --all --min-messages 1                   # also write regexes for tiny templates (small test samples)
    python pipeline.py regex --category "Telecom / Operator Service"  # one category
    python pipeline.py regex --category "Utility Bills" --category "Education" --think
    python pipeline.py regex --all --workers 5                        # every category
    python pipeline.py extract                                        # classified.csv + regexes -> extracted.csv
    python pipeline.py inspect --category "Utility Bills" -n 5        # look at messages: template, regex, information
    python pipeline.py inspect --regexes                              # which fields each category's regexes extract
    python pipeline.py run data/original_message_part_1.csv --all --manage-server

GPU: the classifier (about 14 GB) and the Qwen server (about 20 GB) do not fit on the 24 GB card together.
Run the stages one after the other (the files in output/ carry the data between them). `run --manage-server`
stops the Qwen server before classifying and starts it again afterwards; without that flag nothing is ever
stopped, and a stage that cannot get the GPU or the server tells you what to do.

Python environments: `classify` needs the vLLM virtualenv, `regex` the regex_writer one. `run` starts each
stage with the right interpreter (PY_CLASSIFY / PY_REGEX environment variables override the defaults).
"""
import argparse
import os
import unicodedata
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
PY_CLASSIFY = os.environ.get("PY_CLASSIFY", os.path.expanduser("~/venvs/vllm/bin/python"))
PY_REGEX = os.environ.get("PY_REGEX", os.path.expanduser("~/venvs/regex_writer/bin/python"))
START_SERVER = HERE / "start_vllm.sh"
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
DEFAULT_MIN_MESSAGES = 5       # templates with fewer messages get no regex (override with --min-messages)
MIN_FREE_GPU_GIB = 15          # the classifier reserves ~14.6 GiB

TEXT_COL_DEFAULT, ID_COL_DEFAULT = "msg_body", "rid"
CLASSIFIED = "classified.csv"


# ==========================================
# helpers
# ==========================================
def _gpu_free_gib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split(",")
        return (float(out[0]) - float(out[1])) / 1024
    except Exception:
        return None            # no nvidia-smi: do not block


def server_up():
    try:
        urllib.request.urlopen(f"{VLLM_URL}/models", timeout=5).read()
        return True
    except Exception:
        return False


def stop_server():
    print("stopping the vLLM server ...")
    subprocess.run(["pkill", "-f", "[v]llm serve"], check=False)
    for _ in range(30):
        time.sleep(2)
        if subprocess.run(["pgrep", "-f", "[v]llm"], capture_output=True).returncode != 0:
            return
    print("warning: the vLLM server did not stop within 60 s")


def start_server(wait_s=420):
    print("starting the vLLM server (about 1-2 minutes) ...")
    subprocess.run(["bash", str(START_SERVER)], check=True, stdout=subprocess.DEVNULL)
    for _ in range(wait_s // 5):
        time.sleep(5)
        if server_up():
            print("vLLM server is up")
            return
    raise SystemExit("the vLLM server did not come up; see vllm_server.log")


def _need_gpu_for_classify():
    free = _gpu_free_gib()
    if free is not None and free < MIN_FREE_GPU_GIB:
        raise SystemExit(f"Only {free:.1f} GiB of GPU memory is free, the classifier needs about {MIN_FREE_GPU_GIB}.\n"
                         "The Qwen server is probably running. Stop it (pkill -f '[v]llm serve') or use "
                         "`run ... --manage-server`, which stops and restarts it for you.")


def _read_registry(out_dir):
    path = out_dir / "templates_registry.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run the `templates` stage first (python pipeline.py templates)")
    return pd.read_csv(path)


# ==========================================
# stages
# ==========================================
def _load_and_decode(a):
    from sms_profiling.preprocess import prepare_messages
    df = pd.read_csv(a.csv, dtype={a.id_col: str})
    if a.limit:
        df = df.head(a.limit)
    if a.id_col not in df.columns:
        print(f"note: no column {a.id_col!r}; using the row number as the message id")
        df[a.id_col] = [str(i) for i in range(len(df))]
    print(f"loaded {len(df):,} rows from {a.csv}")
    df = prepare_messages(df, a.text_col)                      # 1. decode hex + language flag  2. masked text (no network)
    print("language:", df["original_language_flag"].value_counts(dropna=False).to_dict())
    n_raw, n_masked = df["message_decoded"].nunique(), df["message_masked"].nunique()
    print(f"masked text: {n_masked:,} distinct templates for {len(df):,} messages ({n_raw:,} distinct raw texts): "
          f"classifying the templates once is {n_raw / max(n_masked, 1):.1f}x less work")
    return df


def _classify_from_labels(a):
    """No model: decode the messages and attach the categories of an EARLIER classification (matched on the message id)."""
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_and_decode(a)
    keep = {"Predicted_Category", "Predicted_Probability", "System1_Confidence", "Model_Choice"}
    lab = pd.read_csv(a.labels_from, dtype={a.labels_id_col: str}, usecols=lambda c: c == a.labels_id_col or c in keep)
    if "Predicted_Category" not in lab.columns:
        raise SystemExit(f"{a.labels_from} has no Predicted_Category column")
    dup = int(lab[a.labels_id_col].duplicated().sum())
    lab = lab.drop_duplicates(a.labels_id_col).set_index(a.labels_id_col)
    key = df[a.id_col].astype(str)
    for col in [c for c in ("Model_Choice", "Predicted_Category", "Predicted_Probability", "System1_Confidence") if c in lab.columns]:
        df[col] = key.map(lab[col])
    missing = int(df["Predicted_Category"].isna().sum())
    print(f"reused the labels of {a.labels_from} (matched {a.id_col} = {a.labels_id_col}): {len(df) - missing:,} labelled, "
          f"{missing:,} without a label{f', {dup:,} duplicate ids ignored' if dup else ''}")
    print("note: these labels were NOT computed on this text; re-run without --labels-from to classify the decoded messages")
    path = out_dir / CLASSIFIED
    df.to_csv(path, index=False)
    print("\n" + df["Predicted_Category"].value_counts(dropna=False).to_string())
    print(f"\nsaved {path}")


def stage_classify(a):
    if a.labels_from:
        return _classify_from_labels(a)
    _need_gpu_for_classify()
    try:
        from sms_profiling.classifier import SMSClassifier, classify
    except ImportError as e:
        raise SystemExit(f"{e}\nRun this stage with the vLLM environment: {PY_CLASSIFY} pipeline.py classify ...")
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_and_decode(a)
    with SMSClassifier(gpu_memory_utilization=a.gpu_mem, other_threshold=a.threshold) as clf:
        res = classify(df, "message_masked" if a.classify_text == "masked" else "message_decoded", classifier=clf, probabilities=False,
                       checkpoint_path=a.checkpoint or out_dir / "classify_checkpoint.csv")
    path = out_dir / CLASSIFIED
    res.to_csv(path, index=False)
    print("\n" + res["Predicted_Category"].value_counts(dropna=False).to_string())
    print(f"\nsaved {path}")


def stage_templates(a):
    from sms_profiling.templates import TEXT_COL, CAT_COL, add_families, generate_all, template_report
    out_dir = Path(a.out_dir)
    src = Path(a.classified) if a.classified else out_dir / CLASSIFIED
    if not src.exists():
        raise SystemExit(f"{src} not found: run the `classify` stage first")
    df = pd.read_csv(src)
    if a.id_col not in df.columns:
        raise SystemExit(f"column {a.id_col!r} missing in {src}")
    before = len(df)
    df = df.dropna(subset=[TEXT_COL, CAT_COL]).copy()
    # one Unicode encoding per letter: Bengali য় exists as U+09DF and as U+09AF+U+09BC, and the two spellings must give ONE template
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(lambda s: unicodedata.normalize("NFC", s))
    if len(df) < before:
        print(f"skipped {before - len(df):,} rows without text or category")
    registry, assignments, model = generate_all(df, TEXT_COL, CAT_COL, a.id_col)
    registry = add_families(registry)                      # + family_id, family_key: templates that differ only in one varying part
    out_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(out_dir / "templates_registry.csv", index=False)
    assignments.to_csv(out_dir / "message_templates.csv", index=False)       # id, category, template_id, text
    model.save(out_dir / "template_model.json")
    report = template_report(registry)
    report.to_csv(out_dir / "template_report.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.to_string(index=False))
    print(f"\n{len(registry):,} templates in {registry['family_id'].nunique():,} families for {len(df):,} messages -> {out_dir}")


def _chosen_categories(a, registry):
    known = sorted(registry["category"].unique())
    if a.all:
        return known
    if not a.category:
        raise SystemExit("give --category \"<name>\" (repeatable) or --all.\nCategories in this data:\n  " + "\n  ".join(known))
    bad = [c for c in a.category if c not in known]
    if bad:
        raise SystemExit(f"unknown category {bad}. Categories in this data:\n  " + "\n  ".join(known))
    return a.category


def stage_regex(a):
    from sms_profiling.templates import coverage_plan, family_registry, select_templates
    out_dir = Path(a.out_dir)
    registry = _read_registry(out_dir)
    if not a.no_families:
        registry = family_registry(registry)               # one unit per family of templates; a template in no family is a family of one
    units, one = ("templates", "template") if a.no_families else ("template families", "family")
    cats = _chosen_categories(a, registry)
    plan = coverage_plan(registry, cats, a.coverage, a.min_messages, a.max_templates)
    print(f"Plan: {units} covering {a.coverage:.0%} of each category's messages "
          f"(min {a.min_messages} msg/{one}, cap {a.max_templates or 'none'}):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(plan.to_string(index=False))
    print(f"total: {int(plan['selected'].sum()):,} regexes to write\n")
    if a.dry_run:
        return

    try:
        from sms_profiling import regex_writer as rw
    except ImportError as e:
        raise SystemExit(f"{e}\nRun this stage with the regex_writer environment: {PY_REGEX} pipeline.py regex ...")
    rw.set_output_dir(out_dir)
    model = a.model or rw.default_model(a.backend)
    rw.check_backend(a.backend, model)
    ws = rw.Workspace.from_output(out_dir, n_samples=a.samples, families=not a.no_families)
    ws.budget_s = a.family_seconds

    tally = []
    for cat in cats:
        sel, got = select_templates(registry, cat, a.coverage, a.min_messages, a.max_templates)
        if sel.empty:
            print(f"[{cat}] no templates selected, skipping")
            continue
        print(f"\n=== {cat}: {len(sel)} {units}, {got:.0%} of its messages ===")
        res = rw.run_templates(ws, sel["template_id"].tolist(), model, verbose=not a.quiet, skip_done=not a.redo,
                               think=a.think, backend=a.backend, workers=a.workers, retries=a.retries, retry_failed=a.retry_failed)
        saved = sum(v in rw.OK_RESULTS for v in res.values())
        tally.append({"category": cat, "templates": len(res), "saved": saved,
                      "failed": len(res) - saved, "coverage_pct": round(100 * got, 1)})
    if tally:
        print("\n=== overall ===")
        print(pd.DataFrame(tally).to_string(index=False))
        print(f"\nregexes are in {rw.REGISTRY_PATH}  |  agent transcripts in {rw.LOG_DIR}")


def stage_extract(a):
    from sms_profiling.extract_information import RegexExtractor, extract_information, summary
    out_dir = Path(a.out_dir)
    src = Path(a.classified) if a.classified else out_dir / CLASSIFIED
    if not src.exists():
        raise SystemExit(f"{src} not found: run the `classify` stage first")
    extractor = RegexExtractor.load(out_dir, a.keep_identifiers)
    result = extract_information(pd.read_csv(src), extractor, categories=a.category, fallback=not a.no_fallback)
    dest = Path(a.out) if a.out else out_dir / "extracted.csv"
    result.to_csv(dest, index=False)
    print(summary(result))
    print(f"\nsaved {dest}")


def stage_inspect(a):
    from sms_profiling import inspect_messages as im
    from sms_profiling.extract_information import RegexExtractor
    out_dir = Path(a.out_dir)
    src = Path(a.classified) if a.classified else out_dir / CLASSIFIED
    if not src.exists():
        raise SystemExit(f"{src} not found: run the `classify` stage first")
    if a.regexes:
        print(im.show_regexes(out_dir, src, a.category))
        return
    ex = RegexExtractor.load(out_dir)
    df = pd.read_csv(src, usecols=lambda c: c in (a.id_col, im.TEXT_COL, im.CAT_COL))
    if not a.category:
        print(im.list_categories(df, ex).to_string(index=False))
        print('\nInspect one:  python pipeline.py inspect --category "<category>" [-n 5] [--only matched|unmatched]')
        return
    items = im.sample_category(df, ex, a.category, a.n, a.only, a.seed)
    if not items:
        print(f"no {a.only} messages found in {a.category!r}")
        return
    print(im.render(items, a.category, a.mask, a.id_col))
    print(f"\n{sum(i['information'] is not None for i in items)} of {len(items)} shown messages have information")


def _run_sub(py, args):
    cmd = [py, str(HERE / "pipeline.py")] + [str(x) for x in args]
    print("\n$ " + " ".join(cmd) + "\n", flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        raise SystemExit(f"stage failed (exit {r.returncode}): {' '.join(args[:1])}")


def stage_run(a):
    if not (a.all or a.category):
        raise SystemExit("give --category \"<name>\" (repeatable) or --all")
    common = ["--out-dir", a.out_dir]
    cls_args = ["classify", a.csv, "--text-col", a.text_col, "--id-col", a.id_col, "--gpu-mem", a.gpu_mem] + common
    if a.limit:
        cls_args += ["--limit", a.limit]
    reg_args = ["regex"] + common + ["--coverage", a.coverage, "--min-messages", a.min_messages,
                                     "--workers", a.workers, "--samples", a.samples, "--retries", a.retries,
                                     "--backend", a.backend]
    if a.max_templates:
        reg_args += ["--max-templates", a.max_templates]
    reg_args += ["--all"] if a.all else sum((["--category", c] for c in a.category), [])
    reg_args += ["--no-families"] if a.no_families else []
    reg_args += ["--retry-failed"] if a.retry_failed else []
    reg_args += ["--family-seconds", a.family_seconds]
    reg_args += ["--think"] if a.think else []
    reg_args += ["--quiet"] if a.quiet else []

    was_up = server_up()
    if a.manage_server and was_up:
        stop_server()
    elif was_up:
        free = _gpu_free_gib()
        if free is not None and free < MIN_FREE_GPU_GIB:
            raise SystemExit("The Qwen server is using the GPU, so the classifier cannot load. Re-run with "
                             "--manage-server (stops and restarts it for you) or stop it yourself.")
    _run_sub(PY_CLASSIFY, cls_args)
    _run_sub(sys.executable, ["templates", "--out-dir", a.out_dir, "--id-col", a.id_col])
    if a.backend == "vllm" and not server_up():
        if not a.manage_server:
            raise SystemExit("The classification is done (output/classified.csv, templates written) but the Qwen "
                             "server is not running. Start it with ./start_vllm.sh and run the `regex` stage, "
                             "or re-run with --manage-server.")
        start_server()
    _run_sub(PY_REGEX, reg_args)
    _run_sub(PY_REGEX, ["extract", "--out-dir", a.out_dir] + (["--keep-identifiers"] if a.keep_identifiers else []) + (["--no-fallback"] if a.no_fallback else []))


# ==========================================
# command line
# ==========================================
def _add_common(p, with_regex=True, with_classify=False):
    p.add_argument("--out-dir", default=str(OUTPUT_DIR), help="folder for all intermediate and final files (default output/)")
    if with_regex:
        p.add_argument("--category", action="append", help='category name (repeatable), e.g. --category "Utility Bills"')
        p.add_argument("--all", action="store_true", help="every category")
        p.add_argument("--coverage", type=float, default=0.8, help="share of a category's messages the chosen templates must cover (default 0.8)")
        p.add_argument("--min-messages", type=int, default=DEFAULT_MIN_MESSAGES,
                       help="skip templates with fewer messages than this: a regex written from 1-4 examples is usually overfitted and "
                            "cannot be checked for generalization (default %(default)s; use 1 for small test samples)")
        p.add_argument("--max-templates", type=int, help="cap on templates (families) per category (coverage may then stay below target)")
        p.add_argument("--redo", action="store_true", help="write the regexes again even for templates / families that already have one (the old one is replaced when the new one passes)")
        p.add_argument("--retry-failed", action="store_true", help="try the families again that failed recently (otherwise they are left alone for 7 days and until the checks change)")
        p.add_argument("--family-seconds", type=int, default=1200, help="the most time one family may use over all its attempts (default 1200)")
        p.add_argument("--no-families", action="store_true",
                       help="one regex per template instead of one per family (templates that differ only in one varying part, e.g. the branch)")
        p.add_argument("--workers", type=int, default=5, help="templates written in parallel (default 5)")
        p.add_argument("--samples", type=int, default=5, help="sample messages the agent reads per template (default 5)")
        p.add_argument("--retries", type=int, default=2, help="extra agent runs when a family is not solved: 1 = a deeper run, 2 = also a soft-landing run that "
                       "accepts 85%% and writes a second regex for the rest (default 2)")
        p.add_argument("--backend", choices=["vllm", "ollama"], default="vllm")
        p.add_argument("--think", action="store_true", help="Qwen reasoning mode (slower, often better)")
        p.add_argument("--quiet", action="store_true")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    c = sub.add_parser("classify", help="decode + language flag + category for every message")
    c.add_argument("csv")
    c.add_argument("--text-col", default=TEXT_COL_DEFAULT)
    c.add_argument("--id-col", default=ID_COL_DEFAULT)
    c.add_argument("--limit", type=int, help="only the first N rows (testing)")
    c.add_argument("--gpu-mem", type=float, default=0.60)
    c.add_argument("--threshold", type=float, default=0.40)
    c.add_argument("--checkpoint", help="classification checkpoint (default <out-dir>/classify_checkpoint.csv)")
    c.add_argument("--classify-text", choices=["masked", "decoded"], default="masked",
                   help="what the model reads: the masked text, i.e. the template, classified once per distinct template (default, ~35,000 "
                        "instead of ~277,000 texts); or the decoded raw text (slow)")
    c.add_argument("--labels-from", help="do NOT run the model: reuse the Predicted_Category of an earlier classification CSV (matched on the id)")
    c.add_argument("--labels-id-col", default="source_id", help="id column of the --labels-from file (default source_id)")
    _add_common(c, with_regex=False)

    t = sub.add_parser("templates", help="templates per category")
    t.add_argument("--classified", help="classified CSV (default <out-dir>/classified.csv)")
    t.add_argument("--id-col", default=ID_COL_DEFAULT)
    _add_common(t, with_regex=False)

    r = sub.add_parser("regex", help="write + validate regexes for the templates that cover 80%% of a category")
    r.add_argument("--model", help="served model name (default qwen3.5-9b)")
    r.add_argument("--dry-run", action="store_true", help="show the plan only, do not call the LLM")
    _add_common(r)

    e = sub.add_parser("extract", help="apply the saved regexes: adds the columns `information` (JSON) and `regex_pattern`")
    e.add_argument("--out-dir", default=str(OUTPUT_DIR))
    e.add_argument("--classified", help="classified CSV (default <out-dir>/classified.csv)")
    e.add_argument("--out", help="result CSV (default <out-dir>/extracted.csv)")
    e.add_argument("--category", action="append", help="only this category (repeatable); default every category")
    e.add_argument("--keep-identifiers", action="store_true", help="keep full account / user / phone identifiers instead of '****1234'")
    e.add_argument("--no-fallback", action="store_true", help="leave rows without a template regex empty instead of reading amounts, dates, times, links, phones")

    i = sub.add_parser("inspect", help="look at sample messages of a category: template, regex, extracted information")
    i.add_argument("--category", help="category to inspect; without it the categories are listed")
    i.add_argument("-n", type=int, default=5)
    i.add_argument("--only", choices=["all", "matched", "unmatched"], default="all")
    i.add_argument("--seed", type=int, default=1)
    i.add_argument("--mask", action="store_true", help="mask long numbers in the printed message")
    i.add_argument("--regexes", action="store_true", help="list the saved regexes and the fields each one extracts (use with --category to narrow)")
    i.add_argument("--out-dir", default=str(OUTPUT_DIR))
    i.add_argument("--classified", help="classified CSV (default <out-dir>/classified.csv)")
    i.add_argument("--id-col", default=ID_COL_DEFAULT)

    u = sub.add_parser("run", help="classify + templates + regex + extract")
    u.add_argument("csv")
    u.add_argument("--text-col", default=TEXT_COL_DEFAULT)
    u.add_argument("--id-col", default=ID_COL_DEFAULT)
    u.add_argument("--limit", type=int)
    u.add_argument("--gpu-mem", type=float, default=0.60)
    u.add_argument("--keep-identifiers", action="store_true", help="extract: keep full identifiers instead of '****1234'")
    u.add_argument("--no-fallback", action="store_true", help="extract: leave rows without a template regex empty")
    u.add_argument("--manage-server", action="store_true",
                   help="stop the Qwen vLLM server before classifying and start it again afterwards")
    _add_common(u)
    return ap


def main():
    a = build_parser().parse_args()
    {"classify": stage_classify, "templates": stage_templates, "regex": stage_regex, "extract": stage_extract, "inspect": stage_inspect, "run": stage_run}[a.stage](a)


if __name__ == "__main__":
    main()
