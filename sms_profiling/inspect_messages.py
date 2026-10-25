"""inspect_messages.py - look at a few messages of a category: the template, the regex, the extracted information.

    python pipeline.py inspect                                   # list the categories (rows, templates, regexes written)
    python pipeline.py inspect --category "Telecom / Operator Service"            # 5 random messages
    python pipeline.py inspect --category "Utility Bills" -n 10 --only matched    # only rows the regex extracted
    python pipeline.py inspect --category "Utility Bills" --only unmatched        # rows with no information, and why
    python pipeline.py inspect --category "Utility Bills" --seed 7 --mask         # another sample; long numbers masked
    python pipeline.py inspect --regexes                                          # WHICH FIELDS each category's regexes extract
    python pipeline.py inspect --regexes --category "Utility Bills"

For every message it prints
    message      the decoded text
    template     the message reduced to its template (fixed wording kept, variable parts as [NUM] / [VAR] ...) and its id
    regex        the regular expression that is applied to it (what ends up in the `regex_pattern` column)
    information  what was extracted (what ends up in the `information` column), or why the row stays empty
Read-only: it never changes any file.
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
import unicodedata
from pathlib import Path

import pandas as pd
import regex

from . import OUTPUT_DIR
from .completeness import money_problem
from .extract_information import RegexExtractor, _IDENTIFIER, _SECRET
from .templates import CAT_COL, TEXT_COL, _safe_example

WHY = {
    "template has no regex yet": "no regex has been written for this template yet "
                                 "(python pipeline.py regex --category \"<category>\" writes regexes for the biggest templates)",
    "regex did not match": "a regex exists for this template but it did not match this message (nothing is guessed)",
    "category not modelled": "this category is not in the template model (run the `templates` stage)",
    "empty text": "the message text is empty",
}


def list_categories(df: pd.DataFrame, ex: RegexExtractor, category_col: str = CAT_COL) -> pd.DataFrame:
    """One row per category: messages, regexes written for it, and the share of messages those regexes can read
    (estimated on up to 2,000 messages per category)."""
    rows = []
    for cat, g in df.groupby(category_col):
        sample = g.sample(min(len(g), 2000), random_state=0)
        hits = sum(ex.extract_one(t, cat)[0] is not None for t in sample[TEXT_COL])
        with_fallback = sum(ex.extract_one(t, cat)[0] is not None or ex.fallback(t) is not None for t in sample[TEXT_COL])
        rows.append({"category": cat, "messages": len(g),
                     "regexes": sum(e["category"] == cat for e in ex.registry.values()),
                     "messages_with_information_%": round(100 * hits / len(sample), 1),
                     "plus_generic_fallback_%": round(100 * with_fallback / len(sample), 1)})
    return pd.DataFrame(rows).sort_values("messages", ascending=False, ignore_index=True)


def list_regexes(ex: RegexExtractor, templates: pd.DataFrame, category: str | None = None, messages: pd.DataFrame | None = None,
                 sample: int = 300) -> list[dict]:
    """One entry per saved regex: the template, how many messages it covers, and the fields it extracts (with the privacy
    treatment each field gets in the `information` column). templates = templates_registry.csv as a DataFrame."""
    tinfo = templates.set_index("template_id") if "template_id" in templates.columns else pd.DataFrame()
    by_tid = {t: g for t, g in messages.groupby("template_id")} if messages is not None else {}
    out = []
    for tid, e in ex.registry.items():
        if category and e["category"] != category:
            continue
        members = e.get("members") or [tid]                  # a family's regex covers all of its member templates
        names = list(regex.compile(e["expanded_pattern"]).groupindex)
        fields = []
        for n in names:
            treat = "never output (secret)" if _SECRET.search(n) else "masked to last 4 digits" if _IDENTIFIER.search(n) and not ex.keep_identifiers else ""
            fields.append({"name": n, "description": e.get("fields", {}).get(n, ""), "treatment": treat})
        hit_rate, warnings = None, []                     # SELF-CHECK: does the regex really extract from this template's messages, and do the values make sense?
        have = [by_tid[m] for m in members if m in by_tid]
        if have:
            g = pd.concat(have).message_decoded.drop_duplicates()
            g = g.sample(min(len(g), sample), random_state=0)
            infos = [ex.extract_one(t, e["category"])[0] for t in g]
            hit_rate = sum(i is not None for i in infos) / max(len(g), 1)
            warnings = [w for n in names if (w := money_problem(n, [i.get(n) for i in infos if i]))]
        counted = [int(tinfo.at[m, "message_count"]) for m in members if m in tinfo.index]
        out.append({"template_id": tid, "category": e["category"], "works_on": hit_rate, "event": e.get("event", ""),
                    "facts": e.get("facts", {}), "unextracted": e.get("unextracted", {}), "family_size": len(members) if e.get("members") else 1, "warnings": warnings, "variant_of": e.get("parent"),
                    "messages": sum(counted) if counted else None,
                    "template": e.get("template") if e.get("members") else (tinfo.at[tid, "template"] if tid in tinfo.index else e.get("template", "")),
                    "fields": fields})
    return sorted(out, key=lambda x: (x["category"], -(x["messages"] or 0)))


def render_regexes(items: list[dict], total_by_category: dict | None = None, width: int = 108) -> str:
    lines, cat = [], None
    for it in items:
        if it["category"] != cat:
            cat = it["category"]
            mine = [i for i in items if i["category"] == cat]
            covered = sum(i["messages"] or 0 for i in mine)
            tot = (total_by_category or {}).get(cat)
            lines += ["", "=" * width, f"{cat}: {len(mine)} regex(es) cover {covered:,} messages" + (f" of {tot:,} ({100 * covered / tot:.0f}%)" if tot else ""), "=" * width]
        works = "" if it.get("works_on") is None else (f" | extracts from {100 * it['works_on']:.0f}% of a sample" if it["works_on"] >= 0.9
                                                       else f" | WARNING extracts from only {100 * it['works_on']:.0f}% of a sample")
        kind = (f"second regex {it['template_id']} for what the regex of {it['variant_of']} misses" if it.get("variant_of")
                else f"family of {it['family_size']} templates {it['template_id']}" if it.get("family_size", 1) > 1 else f"template {it['template_id']}")
        lines += ["", textwrap.fill(f"{kind} | {it['messages'] or 0:,} messages{works} | {it['template']}", width=width, subsequent_indent="    ")]
        if it.get("event"):
            lines.append(f"    {'event':<20} {it['event']}   [label, same for every message of this template]")
        for k, v in (it.get("facts") or {}).items():
            lines.append(textwrap.fill(f"{k:<20} {v}   [fixed text, same for every message of this template]", width=width, initial_indent="    ", subsequent_indent=" " * 25))
        for f in it["fields"]:
            note = f"   [{f['treatment']}]" if f["treatment"] else ""
            lines.append(textwrap.fill(f"{f['name']:<20} {f['description']}{note}", width=width, initial_indent="    ", subsequent_indent=" " * 25))
        for w in it.get("warnings", []):
            lines.append(textwrap.fill(f"WARNING: {w}", width=width, initial_indent="    ", subsequent_indent=" " * 13))
        if it.get("unextracted"):
            lines.append(textwrap.fill(f"NOT EXTRACTED: {json.dumps(it['unextracted'], ensure_ascii=False)}", width=width, initial_indent="    ", subsequent_indent=" " * 8))
    return "\n".join(lines)


def sample_category(df: pd.DataFrame, ex: RegexExtractor, category: str, n: int = 5, only: str = "all", seed: int = 1,
                    text_col: str = TEXT_COL, category_col: str = CAT_COL, scan_limit: int = 100_000) -> list[dict]:
    """n random messages of `category` with their template, regex and information. `only` = all | matched | unmatched.
    Scans a shuffled copy of the category until n suitable rows are found (at most scan_limit rows)."""
    sub = df[df[category_col] == category]
    if sub.empty:
        raise ValueError(f"no rows for category {category!r}. Categories: {sorted(df[category_col].dropna().unique())}")
    order = list(range(len(sub)))
    random.Random(seed).shuffle(order)
    out, scanned = [], 0
    for pos in order[:scan_limit]:
        scanned += 1
        row = sub.iloc[pos]
        text = row[text_col]
        info, pattern, reason = ex.extract_one(text, category)
        if (only == "matched" and info is None) or (only == "unmatched" and info is not None):
            continue
        generic = ex.fallback(text) if info is None else None            # what the generic fallback reads when no template regex does
        template = None
        if isinstance(text, str) and text.strip() and category in ex.model.vocab:
            text = unicodedata.normalize("NFC", text)
            template = ex.model.transform(text, category)
        tid = ex.model.template_id(text, category) if template is not None else None
        family = ex.family_of(tid, text, category) if template is not None else None
        out.append({"row": row, "template_id": tid, "template": template, "regex": pattern, "information": info, "generic": generic,
                    "reason": reason, "scanned": scanned, "family_id": family if family != tid else None})
        if len(out) >= n:
            break
    return out


def render(items: list[dict], category: str, mask: bool = False, id_col: str = "rid", text_col: str = TEXT_COL, width: int = 110) -> str:
    wrap = lambda label, s: textwrap.fill(str(s), width=width, initial_indent=f"  {label:<12}: ", subsequent_indent=" " * 16)
    shown = (lambda t: _safe_example(t)) if mask else (lambda t: t)
    lines = []
    for i, it in enumerate(items, 1):
        r = it["row"]
        status = "EXTRACTED" if it["information"] is not None else "generic fallback" if it.get("generic") else "empty"
        lines += ["", f"[{i}/{len(items)}] {category} | {id_col} {r.get(id_col, '?')} | {status}",
                  wrap("message", shown(str(r[text_col]))),
                  wrap("template", f"{it['template']}   (id {it['template_id']}" + (f", regex of family {it['family_id']}" if it.get("family_id") else "") + ")")
                  if it["template"] else "  template    : (none)"]
        if it["regex"]:
            lines.append(wrap("regex", it["regex"]))
            lines.append(wrap("information", json.dumps(it["information"], ensure_ascii=False)))
        elif it.get("generic"):
            lines.append(wrap("regex", f"(none: {WHY.get(it['reason'], it['reason'])})"))
            lines.append(wrap("information", json.dumps(it["generic"], ensure_ascii=False) + "   [generic fallback]"))
        else:
            lines.append(wrap("regex", "(none)"))
            lines.append(wrap("information", f"(empty) - {WHY.get(it['reason'], it['reason'])}"))
    return "\n".join(lines)


def show_regexes(out_dir: Path, classified: Path, category: str | None = None) -> str:
    ex = RegexExtractor.load(out_dir)
    tpl_path = Path(out_dir) / "templates_registry.csv"
    templates = pd.read_csv(tpl_path) if tpl_path.exists() else pd.DataFrame()
    totals = pd.read_csv(classified, usecols=[CAT_COL])[CAT_COL].value_counts().to_dict() if Path(classified).exists() else {}
    mpath = Path(out_dir) / "message_templates.csv"
    messages = pd.read_csv(mpath, usecols=["template_id", "message_decoded"]) if mpath.exists() else None
    items = list_regexes(ex, templates, category, messages)
    if not items:
        return "no regexes saved yet" + (f" for {category!r}" if category else "") + ": run the regex stage (python pipeline.py regex --category ...)"
    return render_regexes(items, totals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", help="category to inspect; without it the categories are listed")
    ap.add_argument("-n", type=int, default=5, help="how many messages (default 5)")
    ap.add_argument("--only", choices=["all", "matched", "unmatched"], default="all", help="all rows, only rows with information, or only rows without")
    ap.add_argument("--seed", type=int, default=1, help="change it to see another sample")
    ap.add_argument("--mask", action="store_true", help="mask long numbers in the printed message")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR), help="folder with template_model.json and regex_registry.json (default output/)")
    ap.add_argument("--classified", help="classified CSV (default <out-dir>/classified.csv)")
    ap.add_argument("--id-col", default="rid")
    ap.add_argument("--regexes", action="store_true", help="list the saved regexes and the fields each one extracts (use with --category to narrow)")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    src = Path(a.classified) if a.classified else out_dir / "classified.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found: run the classify stage first")
    ex = RegexExtractor.load(out_dir)
    df = pd.read_csv(src, usecols=lambda c: c in (a.id_col, TEXT_COL, CAT_COL))
    if a.regexes:
        print(show_regexes(out_dir, src, a.category))
        return
    if not a.category:
        with pd.option_context("display.width", 200):
            print(list_categories(df, ex).to_string(index=False))
        print('\nInspect one:  python pipeline.py inspect --category "<category>" [-n 5] [--only matched|unmatched]')
        return
    items = sample_category(df, ex, a.category, a.n, a.only, a.seed)
    if not items:
        print(f"no {a.only} messages found in {a.category!r}")
        return
    print(render(items, a.category, a.mask, a.id_col))
    ok = sum(i["information"] is not None for i in items)
    print(f"\n{ok} of {len(items)} shown messages have information (scanned {items[-1]['scanned']} messages of the category)")


if __name__ == "__main__":
    main()
