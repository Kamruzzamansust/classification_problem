"""templates.py - turn classified SMS into templates, and pick the templates worth writing a regex for.

Idea: inside one category every message is reduced to a *template* (fixed wording kept, variable parts
masked). Messages that share a template can share one regex. A few templates usually cover most messages,
so for each category we take the biggest templates until `coverage` (default 80%) of its messages is covered.

Per category
    1. normalize : letter+digit codes -> [CODE], stray digits -> [NUM], punctuation/spacing ignored
    2. prune     : words that are rare in the category (merchant names, typos) -> [VAR]
    3. diversity : words that appear next to many different words (merchant / place words) -> [VAR]
    4. collapse  : consecutive [VAR] -> one [VAR]; consecutive [NUM] -> one [NUM]

Ported from the regex_message project's main.py; the ID column is generic here.
"""
import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict

import pandas as pd

TEXT_COL = "message_decoded"          # decoded SMS text (translator.prepare_messages)
CAT_COL = "Predicted_Category"        # category written by classifier.classify
ID_COL = "rid"
VAR = "[VAR]"

_PUNCT = re.escape(string.punctuation + "।॥“”‘’–—")
_TOKEN = re.compile(r"\[[A-Za-z_]+\]|[^\s%s]+|[%s]" % (_PUNCT, _PUNCT))
_PUNCT_ONLY = re.compile("[%s]" % _PUNCT)
_DIGITS = re.compile(r"[0-9০-৯]+")  # latin + bangla digits
# PNRs, coach/seat/ref codes: letters mixed with digits, e.g. 6A2432E3C2629, VQ968
_CODE = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{3,}\b")
_LONG_DIGITS = re.compile(r"[0-9০-৯]{6,}")


def tokenize(text):
    """Placeholders ([AMOUNT]...) and words survive; punctuation and spacing are dropped.
    Mixed letter+digit codes -> [CODE], remaining digit runs -> [NUM]."""
    tokens = _TOKEN.findall(_DIGITS.sub("[NUM]", _CODE.sub("[CODE]", text)))
    return [t for t in tokens if not _PUNCT_ONLY.fullmatch(t)]


def _is_placeholder(token):
    return len(token) > 1 and token[0] == "["


def _template_id(category, template):
    return hashlib.md5(f"{category}|{template}".encode()).hexdigest()[:12]


class TemplateModel:
    """Per-category vocabulary of 'fixed' words. fit() learns it, transform() applies it."""

    def __init__(self, min_df_frac=0.002, min_df_abs=3, diversity=0.2,
                 diversity_min_support=30, min_rows_to_prune=50):
        self.min_df_frac = min_df_frac          # word must be in >= this share of the category's messages
        self.min_df_abs = min_df_abs            # ...and at least this many messages
        self.diversity = diversity              # mask words whose neighbours vary more than this (None = off)
        self.diversity_min_support = diversity_min_support
        self.min_rows_to_prune = min_rows_to_prune  # tiny categories: keep every word
        self.vocab = {}                         # category -> set of fixed words

    # ---- learning -------------------------------------------------------------------
    def fit_category(self, category, texts):
        counts = Counter(texts)                       # work on unique texts only
        toks = {t: tokenize(t) for t in counts}
        n_rows = sum(counts.values())

        if n_rows < self.min_rows_to_prune:
            self.vocab[category] = {w for t in toks.values() for w in t}
            return self

        df = Counter()                                # message frequency of each word
        for text, tl in toks.items():
            for w in set(tl):
                df[w] += counts[text]
        thr = max(self.min_df_abs, math.ceil(self.min_df_frac * n_rows))
        keep = {w for w, c in df.items() if c >= thr and not _is_placeholder(w)}

        if self.diversity is not None:
            udf = Counter()                           # unique-text frequency
            left, right = defaultdict(set), defaultdict(set)
            for tl in toks.values():
                padded = ["<s>"] + tl + ["</s>"]
                for i in range(1, len(padded) - 1):
                    left[padded[i]].add(padded[i - 1])
                    right[padded[i]].add(padded[i + 1])
                udf.update(set(tl))
            for w in list(keep):
                if udf[w] >= self.diversity_min_support:
                    if max(len(left[w]), len(right[w])) / udf[w] > self.diversity:
                        keep.discard(w)
        self.vocab[category] = keep
        return self

    def fit(self, df, text_col=TEXT_COL, cat_col=CAT_COL):
        for cat, g in df.groupby(cat_col):
            self.fit_category(cat, g[text_col].tolist())
        return self

    # ---- applying -------------------------------------------------------------------
    def transform(self, text, category):
        keep = self.vocab[category]
        out = []
        for w in tokenize(text):
            w = w if (_is_placeholder(w) or w in keep) else VAR
            if w in (VAR, "[NUM]") and out and out[-1] == w:
                continue
            out.append(w)
        return " ".join(out)

    def template_id(self, text, category):
        return _template_id(category, self.transform(text, category))

    # ---- persistence ----------------------------------------------------------------
    def save(self, path):
        cfg = {k: v for k, v in self.__dict__.items() if k != "vocab"}
        data = {"config": cfg, "vocab": {c: sorted(v) for c, v in self.vocab.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = cls(**data["config"])
        model.vocab = {c: set(v) for c, v in data["vocab"].items()}
        return model


def _safe_example(text):
    """Example messages are shown to the LLM: never keep long raw digit runs."""
    return _LONG_DIGITS.sub("[NUM]", text)


def generate_templates(df, category, model=None, text_col=TEXT_COL, cat_col=CAT_COL, id_col=ID_COL):
    """Unique templates for one category.
    Returns (templates, assignments)
      templates   : category, template_id, template, message_count, share_pct, example (largest first)
      assignments : the category's rows with `template_id` and `template` added"""
    sub = df[df[cat_col] == category]
    if sub.empty:
        raise ValueError(f"No rows for category {category!r}. Known: {sorted(df[cat_col].dropna().unique())}")
    if model is None:
        model = TemplateModel()
    if category not in model.vocab:
        model.fit_category(category, sub[text_col].tolist())

    unique_texts = sub[text_col].drop_duplicates()
    mapping = {t: model.transform(t, category) for t in unique_texts}
    assigned = sub.assign(template=sub[text_col].map(mapping))
    assigned["template_id"] = [_template_id(category, t) for t in assigned["template"]]

    templates = (
        assigned.groupby(["template_id", "template"], sort=False)
        .agg(message_count=(text_col, "size"), example=(text_col, "first"))
        .reset_index()
        .sort_values("message_count", ascending=False, ignore_index=True)
    )
    templates["share_pct"] = (100 * templates["message_count"] / len(sub)).round(2)
    templates["example"] = templates["example"].map(_safe_example)
    templates.insert(0, "category", category)
    return templates[["category", "template_id", "template", "message_count", "share_pct", "example"]], assigned


def generate_all(df, text_col=TEXT_COL, cat_col=CAT_COL, id_col=ID_COL, model=None, **kw):
    """Fit one model over every category. Returns (registry, assignments, model).
    assignments: one row per message - id, category, template_id, text (kept for the regex stage)."""
    df = df.dropna(subset=[text_col, cat_col])
    model = model or TemplateModel(**kw).fit(df, text_col, cat_col)
    regs, assigns = [], []
    for cat in sorted(df[cat_col].unique()):
        t, a = generate_templates(df, cat, model, text_col, cat_col, id_col)
        regs.append(t)
        assigns.append(a[[id_col, cat_col, "template_id", text_col]])
    return pd.concat(regs, ignore_index=True), pd.concat(assigns, ignore_index=True), model


def template_report(registry, coverage=0.8):
    """One row per category: how many unique templates, and how concentrated they are."""
    rows = []
    for cat, g in registry.groupby("category"):
        counts = g["message_count"].sort_values(ascending=False).reset_index(drop=True)
        total = int(counts.sum())
        rows.append({
            "category": cat,
            "messages": total,
            "unique_templates": len(counts),
            "msgs_per_template": round(total / len(counts), 1),
            f"templates_for_{int(coverage * 100)}pct_msgs": int((counts.cumsum() < coverage * total).sum() + 1),
            "singleton_templates": int((counts == 1).sum()),
            "top_template_share_pct": round(100 * counts.iloc[0] / total, 1),
        })
        if "family_id" in g.columns:                                   # families: templates that differ only in one varying part
            fam = g.groupby("family_id")["message_count"].sum().sort_values(ascending=False).reset_index(drop=True)
            rows[-1]["families"] = len(fam)
            rows[-1][f"families_for_{int(coverage * 100)}pct_msgs"] = int((fam.cumsum() < coverage * total).sum() + 1)
    return pd.DataFrame(rows).sort_values("messages", ascending=False, ignore_index=True)


# ==========================================
# WHICH TEMPLATES GET A REGEX
# ==========================================
def select_templates(registry, category, coverage=0.8, min_messages=1, max_templates=None):
    """The biggest templates of a category until `coverage` of ALL its messages is covered.

    min_messages   ignore templates with fewer messages than this (coverage may then stay below target).
    max_templates  hard cap on how many templates to take (coverage may then stay below target).
    Returns (selected, achieved) - selected is largest-first, achieved is the share of the category's
    messages those templates cover (0..1)."""
    cat = registry[registry["category"] == category]
    total = int(cat["message_count"].sum())
    if total == 0:
        return cat.iloc[0:0], 0.0
    ranked = cat[cat["message_count"] >= min_messages].sort_values("message_count", ascending=False, kind="stable")
    k = int((ranked["message_count"].cumsum() < coverage * total).sum() + 1)   # first k reach the target
    selected = ranked.head(k)
    if max_templates:
        selected = selected.head(max_templates)
    return selected, float(selected["message_count"].sum()) / total


def coverage_plan(registry, categories=None, coverage=0.8, min_messages=1, max_templates=None):
    """One row per category: what the regex stage would do (messages, templates available/selected,
    coverage reached). Used by the CLI's --dry-run."""
    cats = categories or sorted(registry["category"].unique())
    rows = []
    for c in cats:
        sel, got = select_templates(registry, c, coverage, min_messages, max_templates)
        total = int(registry.loc[registry["category"] == c, "message_count"].sum())
        rows.append({"category": c, "messages": total,
                     "templates": int((registry["category"] == c).sum()),
                     "selected": len(sel), "coverage_pct": round(100 * got, 1)})
    return pd.DataFrame(rows).sort_values("messages", ascending=False, ignore_index=True)


# ==========================================
# FAMILIES: templates that differ only in one varying part
# ==========================================
# A branch, an ATM, a merchant or a department is a FIXED word as far as the template builder can tell, so every branch becomes its own
# template ("... Popular Diagnostic Centre Narayanganj as your ...", "... Centre Gazipur as your ..."). Writing one regex per branch is slow
# and never covers the next branch. Templates that are the same sentence except for one short stretch of words (the "wildcard") form a
# FAMILY: one regex per family, with that stretch captured as a value.
#
# How: every template is turned into its "family keys": the template with each run of 1..MAX_WILDCARD tokens replaced by `*`
# (masked slots [NUM] / [VAR] / [CODE] all count as the same `[X]`). Templates that share a key are the same sentence around a varying
# stretch. Each template joins the largest such group it belongs to, so families never overlap and never chain.
MAX_WILDCARD = 4          # tokens the varying part may span ("Ltd Barishal", "English Road")
MIN_FIXED_WORDS = 4       # fixed words the members must share around it: keeps short generic templates apart
WILD = "*"
_SLOT = re.compile(r"\[[A-Z_]+\]")


def _norm_tokens(template):
    """Template tokens with every masked slot as `[X]` (consecutive slots collapsed); a bracketed brand like [Pathao] stays a word."""
    out = []
    for t in str(template).split():
        if _SLOT.fullmatch(t):
            t = "[X]"
            if out and out[-1] == "[X]":
                continue
        out.append(t)
    return out


def family_keys(template):
    """Every key under which `template` can join a family: its exact normalised form, then the form with each run of up to
    MAX_WILDCARD tokens replaced by `*` (only if at least MIN_FIXED_WORDS fixed words remain)."""
    toks = _norm_tokens(template)
    keys = [" ".join(toks)]
    for i in range(len(toks)):
        for n in range(1, min(MAX_WILDCARD, len(toks) - i) + 1):
            rest = toks[:i] + [WILD] + toks[i + n:]
            if sum(t != WILD and t != "[X]" for t in rest) >= MIN_FIXED_WORDS:
                keys.append(" ".join(rest))
    return keys


def add_families(registry):
    """Add `family_id` and `family_key` to a templates registry (one row per template). A template that joins no family is its own family
    (family_id == template_id, empty key), so everything downstream also works on one-template families."""
    reg = registry.copy()
    reg["family_id"] = reg["template_id"]
    reg["family_key"] = ""
    for cat, g in reg.groupby("category"):
        count = dict(zip(g.index, g["message_count"]))
        groups = defaultdict(list)
        for idx, tpl in zip(g.index, g["template"]):
            for k in set(family_keys(tpl)):
                groups[k].append(idx)
        fixed = lambda k: sum(t != WILD and t != "[X]" for t in k.split())          # the more fixed wording a key keeps, the narrower its wildcard
        cands = sorted(((k, ids) for k, ids in groups.items() if len(ids) >= 2),
                       key=lambda kv: (-len(kv[1]), -sum(count[i] for i in kv[1]), -fixed(kv[0]), kv[0]))
        taken = set()
        for k, ids in cands:
            free = [i for i in ids if i not in taken]
            if len(free) < 2:
                continue
            reg.loc[free, "family_id"] = _template_id(cat, "FAMILY " + k)
            reg.loc[free, "family_key"] = k
            taken.update(free)
    return reg


def family_display(key, fallback=""):
    """A family's template for people and for the agent: the key with `[*]` where the members differ and `[VAR]` for masked slots."""
    if not key:
        return fallback
    return " ".join("[*]" if t == WILD else "[VAR]" if t == "[X]" else t for t in key.split())


def family_registry(registry):
    """One row per family (templates that are in no family are families of one): category, template_id (= family_id), template,
    message_count, share_pct, example, members (template ids, biggest first), family_key. Same columns the selection code uses."""
    reg = registry if "family_id" in registry.columns else registry.assign(family_id=registry["template_id"], family_key="")
    rows = []
    for fid, g in reg.groupby("family_id", sort=False):
        g = g.sort_values("message_count", ascending=False, kind="stable")
        key = str(g["family_key"].iloc[0]) if isinstance(g["family_key"].iloc[0], str) else ""
        top = g.iloc[0]
        rows.append({"category": top["category"], "template_id": fid,
                     "template": family_display(key, top["template"]) if WILD in key.split() else top["template"],
                     "message_count": int(g["message_count"].sum()),
                     "share_pct": round(float(g["share_pct"].sum()), 2) if "share_pct" in g else 0.0,
                     "example": top["example"] if "example" in g else "",
                     "members": g["template_id"].tolist(), "family_key": key})
    return pd.DataFrame(rows).sort_values(["category", "message_count"], ascending=[True, False], kind="stable", ignore_index=True)
