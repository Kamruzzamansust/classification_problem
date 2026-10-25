"""regex_writer.py - the Qwen agent that writes AND verifies an extraction regex for one template at a time.

For each template the agent
    1. reads a few sample messages           (get_template_samples; default 4)
    2. writes a regex with named groups      (macros like {AMOUNT} stand for values)
    3. tests it on EVERY message of the template, and on messages of other templates (test_regex) - and iterates
    4. saves it only if every check passes   (save_regex -> output/regex_registry.json)
A regex only reads what VARIES inside a template. Fixed wording that says who / where / which product (organisation, branch, plan)
is declared as `fixed_facts`, plus an `event` label for what the message announces; completeness.py checks that nothing useful
in the messages is left out (neither captured by a group nor declared as a fact).

Runs fully on the local GPU server (Qwen 3.5 9B served by vLLM, see start_vllm.sh): no SMS text leaves the
machine. It is driven by pipeline.py (stage "regex"); see there for the command line.
"""
import json
import os
import random
import re
import threading
from difflib import SequenceMatcher
import time
import traceback
import unicodedata
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional

import pandas as pd
import regex  # drop-in for re, but supports match timeouts (guards against catastrophic backtracking)
import requests
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from deepagents import create_deep_agent
from . import completeness as comp
from .templates import CAT_COL, ID_COL, TEXT_COL, family_keys, family_registry

from . import OUTPUT_DIR
REGISTRY_PATH = OUTPUT_DIR / "regex_registry.json"
FAILED_PATH = OUTPUT_DIR / "failed_families.json"   # families that did not get a regex, so that the next run does not spend its time on them again
LOG_DIR = OUTPUT_DIR / "logs"    # one transcript per template: what the agent tried and why a check failed
DEFAULT_SAMPLES = 5              # sample messages the agent gets to read per template


def set_output_dir(path):
    """Where templates_registry.csv / message_templates.csv are read from and regex_registry.json + logs written."""
    global OUTPUT_DIR, REGISTRY_PATH, LOG_DIR, FAILED_PATH
    OUTPUT_DIR = Path(path)
    REGISTRY_PATH = OUTPUT_DIR / "regex_registry.json"
    FAILED_PATH = OUTPUT_DIR / "failed_families.json"
    LOG_DIR = OUTPUT_DIR / "logs"
DEFAULT_BACKEND = "vllm"
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
VLLM_MODEL = "qwen3.5-9b"          # --served-model-name in start_vllm.sh
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "qwen3.5:9b"
_REGISTRY_LOCK = threading.Lock()  # parallel workers all append to regex_registry.json
# Ollama's default context is 4096 tokens and silently truncates; the agent keeps samples, test
# metrics and the pattern history in context, so give it room.
NUM_CTX = 16384

MIN_MATCH_RATE = 0.95      # share of the template's messages the regex must match
MIN_HYDRATED_RATE = 0.95   # ...and of the same messages with realistic raw values filled in
MAX_FALSE_POSITIVE = 0.05  # share of OTHER templates' messages it may match before we refuse it
MIN_ANCHOR_WORDS = 2      # literal (non-placeholder) words the pattern must contain, so it cannot be all {MACRO}s and .*?
DEFAULT_RETRIES = 2       # extra agent runs when a family is not solved: 1 = deeper run, 2 = soft-landing run (see LADDER)
SOFT_MATCH_RATE = 0.85    # the last rung accepts a regex that reads at least this share of the family; the rest gets a second regex
LEFTOVER_MIN_TEXTS = 3    # distinct messages a regex missed before it is worth a second regex for them
MAX_LEFTOVER_DEPTH = 2    # a family may have a main regex plus two more for what the earlier ones missed
OK_RESULTS = ("SAVED", "already saved", "constant notice")   # results that are not failures
MIN_OTHER_TEMPLATE_MESSAGES = 5   # a unit with fewer messages never gets a regex of its own (min_messages): it is not an 'other template' for the too-generic check
SIMILAR_TEMPLATE_RATIO = 0.8   # templates of one category that are at least this similar are the same kind of message (see Workspace._similar_units;
                               # measured on card-purchase messages: the same message in a close shape is 0.84, different messages of one bank are <= 0.5)
# What happens, on its own, when a family does not pass the checks: (name, sample messages, step limit, reasoning effort for Qwen3.8).
# More, MORE DIFFERENT samples and more steps for a messy family; a lower match floor on the last rung.
LADDER = [("normal", None, 60, None), ("deeper", 10, 100, "xhigh"), ("soft landing", 10, 100, "xhigh")]
SOFT_FP_CEILING = 0.5     # from the second rung on, a regex that passes every other check may match up to this share of other templates' messages
                          # (it cannot make the family's own extraction wrong: a message is read by the regex of its own template / family); recorded as "waived"
STALL_ATTEMPTS = 4        # this many tests in a row that hit the same wall without progress end the run: the next rung tries another way
STALL_PROGRESS = 0.02     # progress = the score (match rate - false positives - 2% per item left out) rose by at least this much
FAMILY_BUDGET_S = 1200    # wall-clock seconds one family may use over all its rungs (no new rung starts after it; a running one stops at its next step)
# CHECKS_VERSION (end of this file) fingerprints the checks' code and limits: families that failed under other checks are tried again by themselves
RETRY_FAILED_DAYS = 7     # a family that failed is left alone for this long (retry_failed=True ignores it)
MAX_OUTPUT_TOKENS = None  # no cap on a reply: vLLM rejects a request when conversation + cap exceed its 16K window (a cap of 8192 left only 8K for the
                          # conversation and the run died with ContextOverflowError). A run that never tests anything is retried without reasoning instead.
MAX_TEST_TEXTS = 2000
HEARTBEAT_S = 60          # with several workers a line is printed only when a family finishes; this says "still working" when none did for a while
# Never extract login credentials / card secrets from SMS (PCI-DSS, data privacy). OTP is allowed:
# extracting it is the point of the OTP category.
SENSITIVE_FIELD_WORDS = ("password", "passwd", "passcode", "pin", "cvv", "cvc", "secret")
_SENSITIVE = re.compile(r"(^|_)(" + "|".join(SENSITIVE_FIELD_WORDS) + r")($|_)")


def _kind(problem):
    """What kind of wall a problem is (its text without numbers, names and values): lets the ladder see that the agent keeps hitting the same one."""
    return re.sub(r"\[[^\]]*\]|'[^']*'|\{[^}]*\}|[0-9][0-9.,%]*", "#", str(problem))[:45]


def is_sensitive_name(name):
    """A group name that means a credential / card secret. Whole words only: 'shipping_fee' is not 'pin'."""
    return bool(_SENSITIVE.search(name.lower()))
MATCH_TIMEOUT_S = 0.2

# The anonymized text has [AMOUNT], [DATE]... where the raw SMS has real values. A macro
# matches the placeholder itself AND a realistic raw value, so one regex serves both.
# Tune these against real raw SMS - they are the only place raw formats are encoded.
MACROS = {
    "AMOUNT": r"(?:\[AMOUNT\]|(?:BDT|Tk\.?|TK\.?|Rs\.?|৳)?\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:BDT|Tk|TK))?)",
    "NUM": r"(?:\[NUM\]|\d+)",
    "BN_NUM": r"(?:\[BN_NUM\]|[০-৯]+)",
    "DATE": r"(?:\[DATE\]|\d{1,4}[-/.]\w{1,9}[-/.]\d{2,4}|\d{1,2}\s?[A-Za-z]{3,9},?\s?\d{2,4})",
    "TIME": r"(?:\[TIME\]|\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?)",
    "TXN": r"(?:\[TXN\]|[A-Za-z0-9]{6,})",
    "ACCOUNT": r"(?:\[ACCOUNT\]|[A-Za-z0-9*Xx#\-]{4,})",
    "PHONE": r"(?:\[PHONE\]|\+?\d[\d\- ]{6,14}\d)",
    "URL": r"(?:\[URL\]|(?:https?://|www\.)\S+)",
    "NAME": r"(?:\[NAME\]|[^\s,.:;]+(?:\s[^\s,.:;]+){0,3})",
    "REF": r"(?:\[REF\]|[A-Za-z0-9\-]{4,})",
    "OTP": r"(?:\[OTP\]|\d{4,8})",
    "PASSWORD": r"(?:\[PASSWORD\]|\S+)",
    "USER": r"(?:\[USER\]|\S+)",
    "EMAIL": r"(?:\[EMAIL\]|[\w.+-]+@[\w-]+\.[\w.]+)",
}
# What a placeholder might look like in a raw SMS - used for the "hydrated" round-trip test.
HYDRATE = {
    "[AMOUNT]": "Tk 1,250.50", "[NUM]": "4821", "[BN_NUM]": "১২৩৪", "[DATE]": "08-Jun-26",
    "[TIME]": "03:42:07", "[TXN]": "TX93KD72", "[ACCOUNT]": "1234567890", "[PHONE]": "01712345678",
    "[URL]": "https://example.com/x", "[NAME]": "Rahim Uddin", "[REF]": "REF48392", "[OTP]": "482915",
    "[PASSWORD]": "Pw8392", "[USER]": "user01", "[EMAIL]": "user@example.com",
}
_MACRO_REF = re.compile(r"\{([A-Z_]+)\}")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


def normalize(text):
    """NFC. Bengali has two encodings of the same letter (e.g. য় = U+09DF or U+09AF+U+09BC); a regex
    literal in one form would silently miss the other. Apply this to every incoming SMS before matching."""
    return unicodedata.normalize("NFC", text)


_PHRASE_REF = re.compile(r"\{L(\d+)\}")
MIN_PHRASE_LEN = 4        # a stretch of fixed text shorter than this is not offered as a phrase
MAX_PHRASES = 20
PHRASE_SAMPLE = 30        # messages compared to find the fixed text


def fixed_phrases(texts, min_len=MIN_PHRASE_LEN, max_texts=PHRASE_SAMPLE, max_phrases=MAX_PHRASES):
    """The FIXED text of a group of messages, in order: every stretch of at least `min_len` characters that is present, in the same place,
    in all of the first `max_texts` distinct messages (spacing ignored). The agent writes {L1}, {L2}... for them in its regex and the system
    inserts the exact original characters, so it never has to type (and mistype) Bengali words. [] when there are fewer than 3 distinct
    messages: nothing can be told apart from the values then."""
    seen = list(dict.fromkeys(re.sub(r"\s+", " ", t).strip() for t in texts if isinstance(t, str) and t.strip()))[:max_texts]
    if len(seen) < 3:
        return []
    ref = seen[0]
    common = [True] * len(ref)
    for other in seen[1:]:
        hit = [False] * len(ref)
        for b in SequenceMatcher(None, ref, other, autojunk=False).get_matching_blocks():
            if b.size >= min_len:
                for k in range(b.a, b.a + b.size):
                    hit[k] = True
        common = [c and h for c, h in zip(common, hit)]
    for m in re.finditer(r"\S+", ref):                    # a word with a digit in it is a value (a date, an id, a number), even when it is the same in every
        if any(ch.isdigit() for ch in m.group(0)):          # sampled message: it is never part of a "fixed" phrase, so a regex can still capture it
            for k in range(m.start(), m.end()):
                common[k] = False
    phrases, start = [], None
    for k, c in enumerate(common + [False]):
        if c and start is None:
            start = k
        elif not c and start is not None:
            p = ref[start:k]
            # A piece that starts or ends in the MIDDLE of a word is the shared beginning / end of a value that varies (a number prefix, a name
            # glued to a comma): that partial word is dropped, so the phrase never contains part of a value.
            if start > 0 and ref[start - 1] != " " and not p.startswith(" "):
                p = p[p.find(" ") + 1:] if " " in p else ""
            if k < len(ref) and ref[k] != " " and not p.endswith(" "):
                p = p[:p.rfind(" ")] if " " in p else ""
            p = p.strip()
            if len(p) >= min_len and any(ch.isalnum() for ch in p):
                phrases.append(p)
            start = None
    if len(phrases) > max_phrases:                       # keep the longest ones, still in order
        keep = set(sorted(range(len(phrases)), key=lambda i: -len(phrases[i]))[:max_phrases])
        phrases = [p for i, p in enumerate(phrases) if i in keep]
    return phrases


def expand_macros(pattern, phrases=None):
    """{AMOUNT}, {NUM}... -> their patterns; {L1}, {L2}... -> the exact fixed text of the family (spacing inside it matches any whitespace)."""
    def sub(m):
        if m.group(1) not in MACROS:
            raise ValueError(f"unknown macro {{{m.group(1)}}}; available: {sorted(MACROS)}")
        return MACROS[m.group(1)]
    def phrase(m):
        i = int(m.group(1))
        if not phrases or not 1 <= i <= len(phrases):
            raise ValueError(f"unknown phrase {{L{i}}}; the fixed phrases are " + (f"L1..L{len(phrases)}" if phrases else "none for this template (type the words)"))
        return r"\s+".join(re.escape(w) for w in phrases[i - 1].split())
    return _MACRO_REF.sub(sub, _PHRASE_REF.sub(phrase, pattern))


_ESCAPE = re.compile(r"\\.")                      # \s \d \[ ...
_CHAR_CLASS = re.compile(r"\[[^\]]*\]")
_MACRO_TOKEN = re.compile(r"\{[A-Z_]+\}")
_QUANTIFIER = re.compile(r"\{\d+(?:,\d*)?\}")
_GROUP_HEAD = re.compile(r"\(\?(?:P<[^>]*>|<[^>]*>|#[^)]*\)|[a-zA-Z:!=<]+\)?)")   # (?P<name>  (?:  (?i)  (?=


def anchor_words(pattern, phrases=None):
    """Distinct literal words (>= 3 letters, any script) left in the pattern after removing macros,
    character classes, escapes, group names and quantifiers. A pattern with fewer than
    MIN_ANCHOR_WORDS of these is just {MACRO}s and wildcards and would match almost any SMS.
    The words of a {L1} phrase count: it is fixed text of the template."""
    if phrases:
        pattern = _PHRASE_REF.sub(lambda m: " " + phrases[int(m.group(1)) - 1] + " " if 1 <= int(m.group(1)) <= len(phrases) else " ", pattern)
    p = _ESCAPE.sub(" ", pattern)
    for rx in (_MACRO_TOKEN, _CHAR_CLASS, _GROUP_HEAD, _QUANTIFIER):
        p = rx.sub(" ", p)
    words, cur = [], ""
    for ch in p:
        if unicodedata.category(ch)[0] in "LM":     # letters + combining marks (Bengali vowel signs)
            cur += ch
        else:
            if cur:
                words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    return list(dict.fromkeys(w for w in words if len(w) >= 3))


def named_group_bodies(pattern):
    """{group name: its sub-pattern text} for every (?P<name>...) (handles nesting, escapes, char classes)."""
    out = {}
    for m in re.finditer(r"\(\?P<(\w+)>", pattern):
        i, depth, in_class = m.end(), 1, False
        while i < len(pattern) and depth:
            c = pattern[i]
            if c == "\\":
                i += 2
                continue
            if in_class:
                in_class = c != "]"
            elif c == "[":
                in_class = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        out[m.group(1)] = pattern[m.end():i - 1]
    return out


_LITERAL_BODY = re.compile(r"[^\\.\[\](){}*+?|^$]+")


def literal_only_groups(pattern):
    """Named groups whose sub-pattern is plain text (e.g. (?P<greeting>আপনার)): they can only ever capture
    the same fixed wording, never a value."""
    return sorted(n for n, body in named_group_bodies(pattern).items()
                  if body.strip() and _LITERAL_BODY.fullmatch(body.strip()) and any(c.isalpha() for c in body))


_TEMPORAL = re.compile(r"\W*(?:\d{1,4}[-/.\s]\w{1,9}[-/.\s]\d{2,4}|\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?)\W*")
# a group named like a date part (year, month, day, hour...) varies in real life even when a one-day sample holds one value
_TEMPORAL_NAME = re.compile(r"(^|_)(date|dt|time|year|month|day|hour|minute|period|expiry|expire|expires|due|dob|yyyy|mm|dd)($|_)")
MIN_UNIQUE_FOR_CONSTANT_CHECK = 5   # a group that returns ONE value across this many different messages is fixed wording


def hydrate(text):
    return _PLACEHOLDER.sub(lambda m: HYDRATE.get(m.group(0), m.group(0)), text)


_LONG_DIGITS = re.compile(r"[0-9\u09E6-\u09EF]{6,}")


def _mask(text, limit=300):
    """Everything the LLM sees goes through here. Long digit runs (patient / account / phone numbers) are replaced by RANDOM
    digits of the same length, so the agent sees what a real number looks like (and writes \\d+) while the real value never
    reaches it. Deterministic per text. (The old '[NUM]' placeholder made the model copy '[NUM]' into its regexes.)"""
    rng = random.Random(text)
    def scramble(m):
        return "".join(rng.choice("123456789") if i == 0 else rng.choice("0123456789") for i in range(len(m.group(0))))
    return _LONG_DIGITS.sub(scramble, text)[:limit]


def diverse_samples(texts, n):
    """n messages that look as DIFFERENT from each other as possible: texts are grouped by their layout (letters and digits replaced, spacing
    collapsed) and taken one per layout in turn, the biggest layouts first. A messy template shows all its layouts instead of five look-alikes."""
    groups: dict[str, list[str]] = defaultdict(list)
    for t in sorted(texts):
        groups[re.sub(r"\s+", " ", re.sub(r"\d+", "0", re.sub(r"[^\W\d_]+", "a", t))).strip()].append(t)
    order = sorted(groups.values(), key=lambda g: -len(g))
    out = []
    for r in range(max((len(g) for g in order), default=0)):
        for g in order:
            if r < len(g):
                out.append(g[r])
                if len(out) == n:
                    return out
    return out


def leftover_texts(expanded_pattern, texts):
    """The distinct messages a saved regex does NOT match."""
    rx = regex.compile(normalize(expanded_pattern))
    miss = []
    for t in texts:
        try:
            if not rx.search(t, timeout=MATCH_TIMEOUT_S):
                miss.append(t)
        except TimeoutError:
            miss.append(t)
    return miss


def _interleave(lists):
    """One list from several, taking one text from each in turn (biggest first), without duplicates."""
    out, seen = [], set()
    for r in range(max((len(l) for l in lists), default=0)):
        for l in lists:
            if r < len(l) and l[r] not in seen:
                seen.add(l[r])
                out.append(l[r])
    return out


class Workspace:
    """Template data + regex registry, shared by the agent's tools. A "template id" here is the id of a unit of work: one template, or
    a family of templates (Workspace.from_output)."""

    def __init__(self, registry, texts_by_template, n_samples=DEFAULT_SAMPLES):
        self.registry = registry.set_index("template_id")
        self.texts = texts_by_template
        self.n_samples = n_samples
        self.members = {}                                 # family id -> member template ids (empty: every template is its own unit)
        self.member_templates = {}                        # family id -> the member templates' text
        self._key_index = {}                              # category -> (key -> units with a member that produces it, key -> unit)
        self._phrases = {}                                # unit id -> its fixed phrases (see fixed_phrases)
        self._similar = {}                                # unit id -> units of its category with a near-identical template
        self.family_keys = {}                             # family id -> the wildcard key shared by the members
        self.check_completeness = True                    # False = do not send "you left X out" back to the agent
        self.incomplete = defaultdict(set)                # template_id -> distinct incomplete regexes already sent back
        self.match_floor = {}                             # unit id -> lowest match rate accepted (default MIN_MATCH_RATE; lowered on the last rung)
        self.last_problems = {}                           # unit id -> the problems the checks reported last (why it failed)
        self.samples_for = {}                             # unit id -> number of sample messages to show (default n_samples)
        self.parent_of = {}                               # leftover unit id -> the unit whose regex missed these messages
        self.notes = defaultdict(list)                    # unit id -> what the ladder did (for the report)
        self.fp_ceiling = {}                              # unit id -> share of other templates' messages a regex may match (default MAX_FALSE_POSITIVE; raised from rung 2)
        self.history = defaultdict(list)                  # unit id -> (kind of the first problem, score) of every test of the current run
        self.deadline = {}                                # unit id -> time.time() after which no new rung starts
        self.budget_s = FAMILY_BUDGET_S                   # seconds a family may use in all (extract_categories(family_seconds=...))
        self._lock = threading.Lock()

    @classmethod
    def from_output(cls, output_dir=None, n_samples=DEFAULT_SAMPLES, families=True):
        """Read the files written by `pipeline.py templates`. families=True: the unit the agent works on is a FAMILY of templates that
        differ only in one varying part (a branch, an ATM, a merchant): one regex per family, its messages taken from every member in turn
        so that the biggest members do not crowd the others out. families=False: one unit per template."""
        out = Path(output_dir or OUTPUT_DIR)
        return cls.from_frames(pd.read_csv(out / "templates_registry.csv"), pd.read_csv(out / "message_templates.csv", usecols=["template_id", TEXT_COL]),
                               n_samples, families)

    @classmethod
    def from_frames(cls, reg, msgs, n_samples=DEFAULT_SAMPLES, families=True):
        """The same from DataFrames instead of files: reg = the templates registry (with family_id / family_key for families), msgs = one row per
        message with `template_id` and the message text (TEXT_COL)."""
        reg = reg.sort_values(["category", "message_count"], ascending=[True, False], kind="stable")
        msgs = msgs[["template_id", TEXT_COL]].copy()
        msgs[TEXT_COL] = msgs[TEXT_COL].astype(str).map(normalize)
        texts = {tid: g[TEXT_COL].drop_duplicates().tolist() for tid, g in msgs.groupby("template_id")}
        if not (families and "family_id" in reg.columns):
            return cls(reg, texts, n_samples)
        freg = family_registry(reg)
        by_family = {fid: _interleave([texts.get(m, []) for m in members]) for fid, members in zip(freg["template_id"], freg["members"])}
        ws = cls(freg.drop(columns=["members", "family_key"]), by_family, n_samples)
        ws.members = dict(zip(freg["template_id"], freg["members"]))
        text_of = dict(zip(reg["template_id"], reg["template"]))
        ws.member_templates = {f: [text_of[m] for m in ms] for f, ms in ws.members.items() if len(ms) > 1}
        ws.family_keys = {f: k for f, k in zip(freg["template_id"], freg["family_key"]) if k}
        return ws

    def category_of(self, tid):
        return self.registry.at[tid, "category"]

    def add_unit(self, uid, category, template, texts, parent):
        """A new unit of work made of the messages a saved regex did not match: it gets its own (second) regex."""
        row = pd.DataFrame({"category": [category], "template": [template], "message_count": [len(texts)], "share_pct": [0.0], "example": [""]},
                           index=pd.Index([uid], name="template_id"))
        with self._lock:
            self.texts[uid] = list(texts)
            self.parent_of[uid] = parent
            self.registry = pd.concat([self.registry, row])

    def test_texts(self, tid):
        t = self.texts[tid]
        return t if len(t) <= MAX_TEST_TEXTS else pd.Series(t).sample(MAX_TEST_TEXTS, random_state=0).tolist()

    @staticmethod
    def _skeleton(template):
        """The fixed wording of a template, ignoring its variable slots ([VAR], [NUM], [CODE], [Pathao]-style tokens)."""
        return tuple(w for w in template.split() if not (len(w) > 1 and w[0] == "[" and w[-1] == "]"))

    def family_texts(self, tid, cap=500):
        """Messages of this template plus every same-wording sibling template (differing only in variable slots).
        A group that varies across the family is a real field even if this one template happens to hold one value."""
        cat = self.category_of(tid)
        skel = self._skeleton(self.registry.at[tid, "template"])
        sib = self.registry[(self.registry["category"] == cat) & (self.registry.index != tid)]
        sib = sib[pd.Series([self._skeleton(t) == skel for t in sib["template"]], index=sib.index, dtype=bool)]
        pool = list(self.texts[tid]) + [t for o in sib.index[:60] for t in self.texts.get(o, [])[:50]]
        pool = list(dict.fromkeys(pool))
        return pool if len(pool) <= cap else pool[:cap]

    def _similar_units(self, tid):
        """Units of the same category whose template is almost the same sentence (>= SIMILAR_TEMPLATE_RATIO of the words in the same order, slots
        ignored). One message type often exists as several such families (the e-GP tender submission as 49 + 9 + a few small ones): a regex
        for one of them rightly matches the others, so they are not 'other templates'."""
        if tid not in self._similar:
            cat = self.category_of(tid)
            toks = lambda t: [("[X]" if re.fullmatch(r"\[[A-Z_*]+\]", w) else w) for w in str(t).split()]
            a = toks(self.registry.at[tid, "template"])
            close = set()
            for u in self.registry.index[(self.registry["category"] == cat) & (self.registry.index != tid)]:
                b = toks(self.registry.at[u, "template"])
                if len(set(a) & set(b)) / max(1, len(set(a) | set(b))) >= 0.6 and SequenceMatcher(None, a, b, autojunk=False).ratio() >= SIMILAR_TEMPLATE_RATIO:
                    close.add(u)
            self._similar[tid] = close
        return self._similar[tid]

    def record_attempt(self, tid, problems, score):
        self.history[tid].append((_kind(problems[0]) if problems else "", score))

    def stalled(self, tid, n=STALL_ATTEMPTS):
        """The last n tests all failed on the same kind of problem and the score did not improve: the agent is going round in circles."""
        h = self.history[tid][-n:]
        if len(h) < n or not h[0][0] or len({k for k, _ in h}) != 1:
            return False
        return max(s for _, s in h[1:]) < h[0][1] + STALL_PROGRESS

    def is_constant_notice(self, tid):
        """Every message of the unit is the very same text and the template has no variable part (a fixed notice such as 'your cheque book is ready').
        There is nothing to extract from it: a regex would have to capture fixed text, which the checks rightly refuse, so the agent is not asked."""
        texts = {re.sub(r"\s+", " ", t).strip() for t in self.texts.get(tid, []) if isinstance(t, str) and t.strip()}
        return len(texts) == 1 and all(not re.search(r"\[[A-Z_*]+\]", str(t)) for t in self._templates_of(tid))

    def phrases_of(self, tid):
        """The fixed text of a unit's messages as numbered pieces {L1}, {L2}... (computed once)."""
        if tid not in self._phrases:
            self._phrases[tid] = fixed_phrases(self.texts[tid])
        return self._phrases[tid]

    def _templates_of(self, tid):
        return self.member_templates.get(tid) or [self.registry.at[tid, "template"]]

    def related_units(self, tid):
        """Units of the same category that are the SAME sentence as `tid` split differently: one of their templates would join this unit's
        family (or the other way round). A regex that also matches them is right, not too generic. (The Popular Diagnostic registration message
        exists as one family of branches and as further families where only the name part varies.)"""
        cat = self.category_of(tid)
        if cat not in self._key_index:
            joins, owner = {}, {}
            for u in self.registry.index[self.registry["category"] == cat]:
                for t in self._templates_of(u):
                    for k in family_keys(t):
                        joins.setdefault(k, set()).add(u)
                if self.family_keys.get(u):
                    owner[self.family_keys[u]] = u
            self._key_index[cat] = (joins, owner)
        joins, owner = self._key_index[cat]
        mine = {k for t in self._templates_of(tid) for k in family_keys(t)}
        rel = {owner[k] for k in mine if k in owner}                       # their key is one of my templates' keys
        own_key = self.family_keys.get(tid)
        if own_key:
            rel |= joins.get(own_key, set())                              # my key is one of their templates' keys
        rel |= {u for k in family_keys(self.registry.at[tid, "template"])[:1] for u in joins.get(k, set())}
        rel |= self._similar_units(tid)
        if tid in self.parent_of:                                     # a leftover unit is the same sentence as its parent: a regex that also fits the parent is fine
            parent = self.parent_of[tid]
            rel |= {parent} | self.related_units(parent)
        rel.discard(tid)
        return rel

    def other_template_texts(self, tid, n=300):
        """Messages of the category's OTHER templates, used to detect a regex that is too generic. Templates that
        differ from this one only in their variable slots (same skeleton) are the same kind of message, so a regex
        that also matches them is correct, not generic: they are left out."""
        cat = self.category_of(tid)
        skel = self._skeleton(self.registry.at[tid, "template"])
        related = self.related_units(tid)
        others = self.registry[(self.registry["category"] == cat) & (self.registry.index != tid) & ~self.registry.index.isin(related)]
        others = others[pd.Series([self._skeleton(t) != skel for t in others["template"]], index=others.index, dtype=bool)]
        if "message_count" in others.columns:                # 3 messages of a sibling shape would weigh 3 / 28 = 11% in a small category's pool
            others = others[others["message_count"].astype(float) >= MIN_OTHER_TEMPLATE_MESSAGES]
        pool = [t for o in others.index[:60] for t in self.texts.get(o, [])[:20]]  # biggest other templates
        return pool if len(pool) <= n else pd.Series(pool).sample(n, random_state=0).tolist()


def load_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else {}


def load_failed():
    try:
        return json.loads(FAILED_PATH.read_text(encoding="utf-8")) if FAILED_PATH.exists() else {}
    except ValueError:
        return {}


def known_failure(tid):
    """The stored record of a family that failed recently under the CURRENT checks (None: it is worth a try: never failed, failed under older
    rules, or failed more than RETRY_FAILED_DAYS ago). Without this every daily run would spend its time on the same hopeless families."""
    e = load_failed().get(tid)
    if not e or e.get("checks_version") != CHECKS_VERSION:
        return None
    try:
        age_days = (time.time() - time.mktime(time.strptime(e["failed_at"], "%Y-%m-%d %H:%M:%S"))) / 86400
    except (KeyError, ValueError):
        return None
    return e if age_days < RETRY_FAILED_DAYS else None


def _store_failures(failed_now, solved):
    with _REGISTRY_LOCK:
        failed = load_failed()
        before = dict(failed)
        failed.update(failed_now)
        for tid in solved:
            failed.pop(tid, None)
        if failed != before:
            FAILED_PATH.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")


def explain_mismatch(pattern, text, phrases=None):
    """Where a regex stops matching one message: the longest prefix of the pattern that still matches somewhere in the text, the part of the
    pattern that fails next, and the text at that point (as repr, so a '.', a space or a line break shows). The usual cause is one small thing."""
    best, end = 0, 0
    for i in range(len(pattern), 0, -1):
        try:
            m = regex.compile(normalize(expand_macros(pattern[:i], phrases))).search(text, timeout=MATCH_TIMEOUT_S)
        except Exception:
            continue                                   # a cut inside a group / macro is not a valid pattern
        if m:
            best, end = i, m.end()
            break
    shown = _mask(text[end:end + 40], 40)
    if not best:
        return f"where it breaks: not even the start of the pattern ({pattern[:40]!r}) matches this message ({_mask(text, 60)!r})"
    msg = (f"where it breaks: the pattern matches up to {pattern[max(0, best - 30):best]!r} but the next part {pattern[best:best + 40]!r} "
           f"does not match the text there, which is {shown!r} (check punctuation such as a full stop, spaces and line breaks)")
    for k in range(1, 81):                              # would the next part match if letter case were ignored ('Jun' vs [A-Z]{3})?
        try:
            if regex.compile("(?i)" + normalize(expand_macros(pattern[:best + k], phrases))).search(text, timeout=MATCH_TIMEOUT_S):
                msg += "; it WOULD match if letter case were ignored: use [A-Za-z] or (?i) instead of an upper-case-only class"
                break
        except Exception:
            continue
    if re.search(r"[^\x00-\u02ff]{3,}", pattern[best:best + 80]) and phrases:      # Bengali (or other non-Latin) letters typed by hand
        msg += ("; the failing part contains non-English letters, which are easy to mistype (look-alike letters, a missing joining mark): "
                "do not type them, use the fixed phrases " + ", ".join(f"{{L{i + 1}}}" for i in range(min(len(phrases), 8))) +
                " (exact text from the messages, see get_template_samples)")
    macros = list(dict.fromkeys(_MACRO_REF.findall(pattern[best:best + 60])))
    if macros:
        msg += "; " + ", ".join(f"{{{m}}} means {MACROS[m][:60]}" for m in macros if m in MACROS)
    return msg


def evaluate(ws, tid, pattern, facts=None):
    """Run every check on a pattern (and on the fixed facts declared with it). Returns a metrics dict (with 'error' if it can't run)."""
    phrases = ws.phrases_of(tid)
    try:
        expanded = expand_macros(pattern, phrases)
        rx = regex.compile(normalize(expanded))
    except Exception as e:  # bad macro, bad syntax...
        ws.record_attempt(tid, ["error"], -1.0)
        return {"error": f"{type(e).__name__}: {e}"}
    names = list(rx.groupindex)
    if not names:
        ws.record_attempt(tid, ["error"], -1.0)
        return {"error": "pattern has no named groups; use (?P<field_name>...) for every value to extract"}

    anchors = anchor_words(pattern, phrases)
    texts = ws.test_texts(tid)
    timeouts = 0

    def run(text):
        nonlocal timeouts
        try:
            return rx.search(text, timeout=MATCH_TIMEOUT_S)
        except TimeoutError:
            timeouts += 1
            return None

    matches = [(t, run(t)) for t in texts]
    matched = [(t, m) for t, m in matches if m]
    fields = {}
    for n in names:
        vals = [m.group(n) for _, m in matched if m.group(n) and m.group(n).strip()]
        fields[n] = {
            "filled_pct": round(100 * len(vals) / max(1, len(matched)), 1),
            "max_len": max((len(v) for v in vals), default=0),
            "distinct_values": len(set(vals)),
            "examples": [_mask(v, 60) for v in list(dict.fromkeys(vals))[:4]],
        }
    has_placeholders = any(_PLACEHOLDER.search(t) for t in texts)
    hyd = sum(1 for t in texts if run(hydrate(t))) if has_placeholders else len(matched)   # raw text: same as match rate
    others = ws.other_template_texts(tid)
    fp = sum(1 for t in others if run(t))

    match_rate = len(matched) / len(texts)
    hyd_rate = hyd / len(texts)
    fp_rate = fp / len(others) if others else 0.0
    problems = []
    floor = ws.match_floor.get(tid, MIN_MATCH_RATE)
    if match_rate < floor:
        problems.append(f"match_rate {match_rate:.2f} < {floor}: see not_matched")
        miss = next((t for t, m in matches if not m), None)
        if miss:
            problems.append(explain_mismatch(pattern, miss, phrases))
    if has_placeholders and hyd_rate < min(MIN_HYDRATED_RATE, floor):
        problems.append(f"hydrated_rate {hyd_rate:.2f} < {MIN_HYDRATED_RATE}: pattern relies on literal "
                        "placeholders; use {MACRO}s for [AMOUNT]/[DATE]/... so raw values match too")
    if timeouts:
        problems.append(f"{timeouts} match(es) timed out: remove nested/ambiguous quantifiers")
    ceiling = ws.fp_ceiling.get(tid, MAX_FALSE_POSITIVE)
    if fp_rate > ceiling:
        problems.append(f"too generic: matches {fp_rate:.0%} of other templates' messages (max {ceiling:.0%}); "
                        "add fixed anchor words from this template's own wording")
    if len(anchors) < MIN_ANCHOR_WORDS:
        problems.append(f"too generic: only {len(anchors)} fixed word(s) in the pattern (need >= {MIN_ANCHOR_WORDS}); "
                        "copy literal wording from the template (service/bank name, fixed phrases) as anchors")
    literal = literal_only_groups(pattern)
    if literal:
        problems.append(f"group(s) {literal} contain only fixed text, so they can never extract a value; "
                        "make them capture a variable value or drop the group (plain text needs no group)")
    fam_matches = [m for m in (run(t) for t in ws.family_texts(tid)) if m]          # this template + same-wording siblings
    def _family_values(n):
        return {m.group(n) for m in fam_matches if m.group(n) and m.group(n).strip()}
    # A date / time that is the same in every message is not fixed wording: the sample covers a short period (often a single day), so
    # "expired on 06-Jun-26" looks constant but varies in real life. The same holds for any value with a digit in it (an id, a code, a number,
    # a challan date): the template builder masks those as variable slots, a small family merely happens to hold one value. Only constant
    # WORDS ("Standard", "Akash") are wording that belongs in fixed_facts.
    constant = sorted(n for n in fields if n not in literal and len(fam_matches) >= MIN_UNIQUE_FOR_CONSTANT_CHECK
                      and len(_family_values(n)) == 1 and not _TEMPORAL_NAME.search(n.lower())
                      and not all(_TEMPORAL.fullmatch(v.strip()) for v in _family_values(n))
                      and not all(any(ch.isdigit() for ch in v) for v in _family_values(n)))
    if constant:
        problems.append(f"group(s) {constant} return the same value in all {len(fam_matches)} different messages "
                        "(this template and its same-wording siblings), i.e. they capture fixed wording: "
                        "remove the group or capture only what really varies (if the fixed text is useful, such as an organisation "
                        "or branch name, declare it in fixed_facts instead)")
    clean_facts, fact_problems = comp.check_facts(facts, texts, set(names), ws.registry.at[tid, "template"])
    problems += fact_problems
    meas = comp.measure(matched[:300], clean_facts, ws.registry.at[tid, "template"])
    if meas["free_text_groups"]:
        problems.append(f"group(s) {meas['free_text_groups']} swallow most of the message (free text), which can hold anything and hides "
                        "what is not captured: capture the specific values instead")
    split = comp.split_amount_problem(meas)
    if split:
        problems.append(split)
    for n in names:
        issue = comp.money_problem(n, [m.group(n) for _, m in matched if m.group(n)])
        if issue:
            problems.append(issue)
    left_out = comp.leftover(meas) if comp.gaps(meas) else {}
    blocking = False
    if left_out and ws.check_completeness:
        if problems:                                       # other checks still fail: say what is left out now, but do not use up an attempt
            blocking = True
        else:
            seen = ws.incomplete[tid]
            seen.add(json.dumps([pattern, facts], sort_keys=True, default=str))
            blocking = len(seen) <= comp.MAX_BLOCKS        # after MAX_BLOCKS otherwise-good but incomplete regexes the gaps are recorded, not enforced
        if blocking:
            problems.append(comp.describe_gaps(meas, lambda x: _mask(x, 40)))
    ws.last_problems[tid] = list(problems)                  # what blocked this unit last: shown in the report of the families that stay without a regex
    ws.record_attempt(tid, problems, match_rate - fp_rate - 0.02 * sum(len(v) for v in left_out.values() if isinstance(v, (list, tuple, set))))
    return {
        "passed": not problems,
        "problems": problems,
        "floor": floor,
        "tested_on": len(texts),
        "match_rate": round(match_rate, 3),
        "hydrated_rate": round(hyd_rate, 3),
        "other_templates_match_rate": round(fp_rate, 3),
        "other_templates_ceiling": ceiling,
        "timeouts": timeouts,
        "anchor_words": anchors,
        "fields": fields,
        "completeness": {"left_out": left_out, "enforced": blocking, "facts": clean_facts},
        "not_matched": [_mask(t) for t, m in matches if not m][:3],
        "_expanded": expanded,
    }


def make_tools(ws, model_name):
    def get_template_samples(template_id: str) -> str:
        """Return the template and a few example messages (diverse; long digit runs masked) for a template_id."""
        if template_id not in ws.texts:
            return f"unknown template_id {template_id!r}"
        n = max(1, ws.samples_for.get(template_id, ws.n_samples))
        members = ws.members.get(template_id, [])
        if len(members) > 1:                                 # a family: the samples come from different members, so the varying part shows
            texts, samples = ws.texts[template_id], ws.texts[template_id][:n]
        else:
            texts = sorted(ws.texts[template_id])
            samples = diverse_samples(texts, n)
        info = {
            "category": ws.category_of(template_id),
            "template": ws.registry.at[template_id, "template"],
            "message_count": int(ws.registry.at[template_id, "message_count"]),
            "unique_messages": len(texts),
            "samples": [_mask(t, 600) for t in samples],
            "macros": sorted(MACROS),
        }
        phrases = ws.phrases_of(template_id)
        if phrases:
            info["phrases"] = {f"L{i + 1}": _mask(p, 240) for i, p in enumerate(phrases)}
            info["phrases_note"] = ("The FIXED text of these messages, in order (the same in every message). In your regex write {L1}, {L2}... where "
                                    "that text appears: the system inserts the exact original characters, spaces match any whitespace. Use them "
                                    "for every Bengali / non-English word instead of typing it. The text between two phrases is what varies.")
        if len(members) > 1:
            info["family"] = (f"This is a FAMILY of {len(members)} similar templates. The part marked [*] differs between them (a branch, a place, "
                              "a product, an ATM, a status...): capture it with a group, with lazy matching between the fixed words around it "
                              "(it may be one word or several, and it may be missing a space). The samples are from different members. "
                              "Fixed words around [*] are shared by every member and may be declared in fixed_facts.")
        return json.dumps(info, ensure_ascii=False)

    def test_regex(template_id: str, pattern: str, fixed_facts: Optional[dict[str, str]] = None) -> str:
        """Test a regex (with {MACRO}s and named groups) against all messages of the template. fixed_facts = {name: exact text} for
        fixed wording worth keeping (organisation, branch, product, support number or link). Returns metrics, including what is still left out."""
        if template_id not in ws.texts:
            return f"unknown template_id {template_id!r}"
        res = evaluate(ws, template_id, pattern, fixed_facts)
        res.pop("_expanded", None)
        return json.dumps(res, ensure_ascii=False)

    def save_regex(template_id: str, pattern: str, field_descriptions: dict[str, str], event: str,
                   fixed_facts: Optional[dict[str, str]] = None) -> str:
        """Save a regex for the template. Refused unless every automatic check passes.
        field_descriptions maps each named group to a one-line meaning. event = short snake_case label of what the message
        announces (e.g. registration_confirmation). fixed_facts = the same {name: exact text} that passed test_regex."""
        if template_id not in ws.texts:
            return f"unknown template_id {template_id!r}"
        res = evaluate(ws, template_id, pattern, fixed_facts)
        if "error" in res or not res["passed"]:
            return "NOT SAVED: " + json.dumps(res.get("problems") or res["error"])
        banned = sorted(n for n in res["fields"] if is_sensitive_name(n))
        if banned:
            return (f"NOT SAVED: group(s) {banned} would extract credentials/secrets. Remove them "
                    "(match that part with a non-capturing group such as (?:{PASSWORD}))")
        missing = set(res["fields"]) - set(field_descriptions)
        if missing:
            return f"NOT SAVED: add a description for groups {sorted(missing)}"
        extra = set(field_descriptions) - set(res["fields"])
        if extra:
            return (f"NOT SAVED: descriptions given for {sorted(extra)} but the pattern has no such named group; "
                    "describe only groups that exist, and capture every field you describe")
        bad_event = comp.check_event(event)
        if bad_event:
            return "NOT SAVED: " + bad_event
        entry = {
            "category": ws.category_of(template_id),
            "template": ws.registry.at[template_id, "template"],
            "pattern": pattern,
            "expanded_pattern": res["_expanded"],
            "fields": field_descriptions,
            **({"members": ws.members[template_id], "family_key": ws.family_keys.get(template_id, "")} if len(ws.members.get(template_id, [])) > 1 else {}),
            "facts": res["completeness"]["facts"],      # fixed wording kept as information (verified to occur in the messages)
            "event": event.strip(),
            **({"phrases": ws.phrases_of(template_id)} if ws.phrases_of(template_id) else {}),
            "unextracted": res["completeness"]["left_out"],   # what the agent could not close (empty = nothing useful left out)
            "text_normalization": "NFC",  # normalize incoming messages the same way before matching
            "metrics": {k: res[k] for k in ("tested_on", "match_rate", "hydrated_rate", "other_templates_match_rate")},
            **({"partial": True} if res["match_rate"] < MIN_MATCH_RATE else {}),          # accepted on the last rung: it reads most of the family, not 95%
            **({"waived": {"other_templates_match_rate": res["other_templates_match_rate"]}} if res["other_templates_match_rate"] > MAX_FALSE_POSITIVE else {}),   # accepted under the relaxed limit of rung 2+
            **({"parent": ws.parent_of[template_id]} if template_id in ws.parent_of else {}),   # a second regex for what the parent's regex misses
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with _REGISTRY_LOCK:
            reg = load_registry()
            reg[template_id] = entry
            REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        return "SAVED"

    return [get_template_samples, test_regex, save_regex]


SYSTEM_PROMPT = """You write regular expressions that extract meaningful fields from SMS messages that share one template.

Workflow for the template_id you are given (do not use the file-system or sub-agent tools; they are not needed):
1. Call get_template_samples and read the template and the sample messages.
2. Account for EVERYTHING in the messages that carries information. Extract as much MEANINGFUL information as the messages carry. The fields will be used to PROFILE the customer's activity, so capture every variable value that says what happened (transaction type, status), how much (amount, fee, balance, quantity), when (date, time, due date, period), where and with whom (merchant, counterparty, branch, location, route), and through what (channel, service, package, product, plan), plus the references that identify the event (transaction / order / invoice / booking / account-last-digits id). Name each field in snake_case. Do NOT make groups for fixed wording (greetings like আপনার / Dear customer, labels): every group must capture something that differs between messages.
   Fixed text that says WHO sent it or WHERE or WHICH product (organisation / brand / bank / hospital / shop, branch / city / ATM, plan / package / service, a support number or link) is information too, even though it never changes inside this template: do not drop it, declare it in fixed_facts = {"organisation": "Popular Diagnostic Centre", "branch": "Narayanganj"} using the EXACT text as written in the messages (it is checked against all of them). Never put a value that varies (a name, an amount, an id) in fixed_facts: capture those with a group. Also choose an event label: what the message announces, in snake_case (registration_confirmation, atm_withdrawal, appointment_booked, report_ready, otp_sent...).
3. Write ONE Python regex using named groups (?P<field>...) for every value that varies.
4. Call test_regex(template_id, pattern, fixed_facts). Read the metrics (they list words, numbers and links that are still neither captured nor declared), fix the pattern or the facts, and test again. Stop iterating after about 6 attempts.
5. When test_regex reports passed=true, call save_regex with the pattern, a one-line description for every named group, the event label and the same fixed_facts.
If you cannot reach passed=true, do NOT save; explain briefly what blocked you.

Rules:
- The messages hold real values (numbers, amounts, dates, ids), or placeholders such as [AMOUNT], [DATE], [TIME], [TXN], [ACCOUNT], [NUM], [PHONE], [URL], [NAME]. Long numbers in the samples (ids, phone numbers) are replaced by random digits of the same length for privacy: match every number by its shape (\\d+ or {NUM}), never by the digits you see, and NEVER write a placeholder such as [NUM] literally. The 'template' line uses [NUM] / [VAR] only to mark the variable parts; the messages hold real values there. Use a macro instead, e.g. (?P<amount>{AMOUNT}), (?P<date>{DATE}), {NUM}, {TXN}. Macros match both the placeholder and a realistic raw value. Available macros: """ + ", ".join(sorted(MACROS)) + """.
- If get_template_samples says the unit is a FAMILY, the part marked [*] in its template is different in every member (branch, place, product, status): capture it with a named group between the fixed words around it, never hard-code one member's value.
- If get_template_samples returns `phrases` (L1, L2, ...), they are the exact fixed text of the messages: write {L1}, {L2}... in your regex instead of typing that text, ALWAYS for Bengali or other non-English words (look-alike letters and joining marks get mistyped and then nothing matches). Put \\s* between a phrase and what varies, e.g. {L1}\\s*(?P<course>.+?)\\s*{L2}. A phrase may stop before a value that is glued to it (a comma, a number): type that punctuation yourself.
- Text that is not a placeholder (merchant, branch, wording) is literal in the messages. Capture variable free text with lazy groups between fixed anchor words, e.g. from (?P<merchant>.+?) on, and keep the fixed wording as literal anchors.
- Generalise: tolerate spacing and punctuation differences (\\s*, optional parts), do not hard-code values that vary between samples, and do not rely on only the few samples you saw: test_regex checks every message of the template.
- A number written right after a currency word (TK2613.0, BDT500, ৳1,250.50) is ONE amount: capture it whole, with its decimals, in a single group such as TK(?P<amount>\\d+(?:\\.\\d+)?) - never split it into two groups, and never call a part of it a "code". The template may show 'TK2613.0' as [CODE] [NUM] because of how it splits words; when the template and the sample messages disagree, trust the sample messages.
- Never capture passwords, PINs, CVV or other secrets in a named group; match them with a non-capturing group.
- Never use nested quantifiers like (a+)+ or (.*)* : the regex runs on every incoming SMS.
- The regex must keep at least 2 literal words from the template (service / bank name, fixed phrases) and may match at most 5% of OTHER templates' messages. A pattern made only of {MACRO}s and .*? is rejected as too generic.
- Messages are Unicode NFC-normalized; copy Bengali literals from the samples exactly. The Bengali full stop is `।` (danda): allow it wherever the samples show it, e.g. [\\s।\\.,]* , because it is not matched by \\. or \\s.
- Python `regex`/`re` syntax. Case-sensitive unless you add (?i).
"""


def default_model(backend):
    return os.environ.get("LLM_MODEL") or (VLLM_MODEL if backend == "vllm" else OLLAMA_MODEL)


def build_agent(ws, model_name=None, think=False, backend=DEFAULT_BACKEND, effort=None):
    model_name = model_name or default_model(backend)
    effort = effort or os.environ.get("LLM_REASONING_EFFORT")
    temperature = 0.6 if think else 0      # Qwen recommends ~0.6 when thinking
    if backend == "vllm":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name, base_url=VLLM_URL, api_key="EMPTY", temperature=temperature,
            max_retries=3, timeout=600, **({"max_tokens": MAX_OUTPUT_TOKENS} if MAX_OUTPUT_TOKENS else {}),
            # Qwen reasons by default; off = faster, fewer tokens. preserve_thinking=False: Qwen3.8 keeps the reasoning of EVERY earlier turn in the
            # prompt by default, which would fill the context within a few test_regex rounds (older templates simply ignore the setting).
            extra_body={"chat_template_kwargs": {"enable_thinking": think, "preserve_thinking": False}},
            # Qwen3.8: how long it reasons (xhigh by default; "medium" / "low" are quicker). Unset = the server's default.
            **({"reasoning_effort": effort} if effort else {}),
        )
    else:
        from langchain_ollama import ChatOllama
        # Ollama's default context is 4096 tokens and silently truncates; give the agent room.
        llm = ChatOllama(model=model_name, base_url=OLLAMA_URL, num_ctx=16384,
                         reasoning=think, temperature=temperature, keep_alive="30m")
    agent = create_deep_agent(model=llm, tools=make_tools(ws, model_name), system_prompt=SYSTEM_PROMPT)
    return agent, model_name


def check_backend(backend, model_name):
    """Fail early with a clear message if the server is down or the model is not available."""
    if backend == "vllm":
        try:
            names = {m["id"] for m in requests.get(f"{VLLM_URL}/models", timeout=10).json()["data"]}
        except Exception as e:
            raise SystemExit(f"Cannot reach vLLM at {VLLM_URL}: {e}\nStart it with: ./start_vllm.sh")
        hint = "check --model / start_vllm.sh (--served-model-name)"
    else:
        try:
            tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()["models"]
        except Exception as e:
            raise SystemExit(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")
        names = {m["name"] for m in tags} | {m["name"].removesuffix(":latest") for m in tags}
        hint = f"run: ollama pull {model_name}"
    if model_name not in names:
        raise SystemExit(f"Model {model_name!r} not served by {backend} (have: {sorted(names)}); {hint}")


def _text(content):
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def _update_messages(update):
    """Messages inside one LangGraph 'updates' chunk (ignores interrupts / non-message updates)."""
    if not isinstance(update, dict):
        return []
    msgs = update.get("messages") or []
    msgs = getattr(msgs, "value", msgs)  # state-overwrite wrapper
    return list(msgs) if isinstance(msgs, (list, tuple)) else [msgs]


class Progress:
    """Pretty-prints what the agent is doing as it happens.
    Dynamic text always goes through Text(): messages contain [AMOUNT]-style tokens that rich
    would otherwise parse as markup and silently drop."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.console = Console(highlight=False)
        self.attempts = 0
        self.seen = set()
        self._log_fh = None
        self._log = None

    def set_log(self, path):
        """Mirror everything shown (plus the agent's full reasoning text) into a plain-text log file."""
        self.close_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(path, "a", encoding="utf-8")
        self._log = Console(file=self._log_fh, highlight=False, force_terminal=False, color_system=None, width=140)
        self._log.print(Text(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} ====="))

    def close_log(self):
        if self._log_fh:
            self._log_fh.close()
        self._log_fh = self._log = None

    def log_only(self, *objs):
        if self._log:
            self._log.print(*objs)

    def say(self, *objs):
        if self.enabled:
            self.console.print(*objs)
        self.log_only(*objs)

    def start_template(self, ws, tid, i, n):
        self.attempts, self.seen = 0, set()
        row = ws.registry.loc[tid]
        body = Text()
        body.append(f"{row['category']}  ·  {int(row['message_count']):,} messages  ·  "
                    f"{len(ws.texts[tid])} unique\n", style="dim")
        body.append(row["template"][:300])
        self.say(Panel(body, title=f"Template {i}/{n}  ·  {tid}", border_style="cyan", title_align="left"))

    def on_message(self, msg):
        key = getattr(msg, "id", None) or id(msg)
        if key in self.seen:
            return
        self.seen.add(key)
        if msg.type == "ai":
            text = _text(msg.content).strip()
            if text:
                self.say(Text("🧠 " + text[:400], style="italic"))
                if len(text) > 400:
                    self.log_only(Text("   … " + text[400:6000], style="italic"))
            for call in msg.tool_calls or []:
                self._on_call(call["name"], call.get("args") or {})
        elif msg.type == "tool":
            self._on_result(msg.name, _text(msg.content))

    def _on_call(self, name, args):
        if name == "get_template_samples":
            self.say(Text("📖 reading example messages", style="bold"))
        elif name == "test_regex":
            self.attempts += 1
            self.say(Text(f"🧪 attempt #{self.attempts}: testing this regex", style="bold yellow"))
            self.say(Text("   " + str(args.get("pattern", "")), style="dim"))
        elif name == "save_regex":
            self.say(Text("💾 saving regex", style="bold"))
        else:  # deepagents built-ins (planning, files...)
            self.say(Text(f"⚙  {name}", style="dim"))

    def _on_result(self, name, content):
        if name == "get_template_samples":
            try:
                samples = json.loads(content).get("samples", [])
            except ValueError:
                return self.say(Text("   " + content[:200], style="red"))
            for k, s in enumerate(samples, 1):
                self.say(Text(f"   {k}. {s[:170]}", style="dim"))
        elif name == "test_regex":
            self._test_result(content)
        elif name == "save_regex":
            ok = content.startswith("SAVED")
            self.say(Text(("✅ " if ok else "⛔ ") + content[:300], style="bold green" if ok else "bold red"))

    def _test_result(self, content):
        try:
            r = json.loads(content)
        except ValueError:
            return self.say(Text("   " + content[:200], style="red"))
        if "error" in r:
            return self.say(Text("   ✗ " + r["error"], style="red"))
        checks = Table.grid(padding=(0, 2))
        for label, val, ok in [
            ("matches its own template", f"{r['match_rate']:.0%}  of {r['tested_on']} messages", r["match_rate"] >= MIN_MATCH_RATE),
            ("works on raw-style values", f"{r['hydrated_rate']:.0%}", r["hydrated_rate"] >= MIN_HYDRATED_RATE),
            ("matches OTHER templates", f"{r['other_templates_match_rate']:.0%}  (lower is better)", r["other_templates_match_rate"] <= r.get("other_templates_ceiling", MAX_FALSE_POSITIVE)),
            ("fixed anchor words", f"{len(r['anchor_words'])}  ({', '.join(r['anchor_words'][:4])[:50]})",
             len(r["anchor_words"]) >= MIN_ANCHOR_WORDS),
            ("timeouts", str(r["timeouts"]), r["timeouts"] == 0),
            ("nothing useful left out", "yes" if not r["completeness"]["left_out"] else
             "no: " + ", ".join(f"{k} {v}" for k, v in r["completeness"]["left_out"].items())[:70],
             not (r["completeness"]["left_out"] and r["completeness"]["enforced"])),
        ]:
            checks.add_row(Text("   ✓" if ok else "   ✗", style="green" if ok else "red"), label, val)
        self.say(checks)
        fields = Table("field", "filled", "examples", box=None, padding=(0, 2), header_style="dim")
        for n, f in r["fields"].items():
            fields.add_row(Text("   " + n), f"{f['filled_pct']:.0f}%", Text(" | ".join(f["examples"])[:70], style="dim"))
        self.say(fields)
        for p in r["problems"]:
            self.say(Text("   ⚠ " + p, style="yellow"))
        if not r["passed"]:
            for t in r["not_matched"]:
                self.say(Text("   not matched: " + t[:150], style="dim"))
            self.say(Text("   → FAILED, agent will revise", style="red"))
        else:
            self.say(Text("   → PASSED", style="green"))

    def summary(self, rows):
        t = Table(title="Summary", header_style="bold")
        for col in ("template", "messages", "result", "attempts", "time"):
            t.add_column(col)
        for tid, msgs, result, attempts, secs in rows:
            style = "green" if result in OK_RESULTS else "red"
            t.add_row(tid, f"{msgs:,}", Text(result, style=style), str(attempts), f"{secs:.0f}s")
        self.say(t)


def is_connection_error(result):
    """A run that failed because the vLLM server did not answer (not because the regex was bad)."""
    return str(result).startswith("error:") and "onnection" in str(result)


def _effort_for(model_name, effort):
    """Reasoning effort is a Qwen3.8 feature: other models are left at their default."""
    return effort if effort and "3.8" in (model_name or "") else None


def _agent_run(ws, tid, prompt, model_name, think, backend, ui, effort, recursion):
    """One fresh agent run (its own model client, thread-safe). Returns 'SAVED', 'no regex saved (...)' or 'error: ...'."""
    before = load_registry().get(tid)
    agent, _ = build_agent(ws, model_name, think, backend, effort)
    try:
        for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]}, config={"recursion_limit": recursion}, stream_mode="updates"):
            for update in chunk.values():
                for msg in _update_messages(update):
                    ui.on_message(msg)
            if ws.stalled(tid):                                # same wall STALL_ATTEMPTS times without progress: stop this run, the next rung goes another way
                ui.log_only(Text(f"--- stopped: the last {STALL_ATTEMPTS} tests failed the same way ({ws.history[tid][-1][0]!r}) without progress"))
                return f"no regex saved (stalled on: {ws.last_problems.get(tid, ['?'])[0][:90]})"
            if ws.deadline.get(tid) and time.time() > ws.deadline[tid]:
                ui.log_only(Text(f"--- stopped: the time budget of {ws.budget_s:.0f}s for this family is used"))
                return "no regex saved (time budget used)"
        after = load_registry().get(tid)
        return "SAVED" if after is not None and after != before else "no regex saved (checks not passed)"
    except Exception as e:
        ui.log_only(Text(traceback.format_exc()))
        return f"error: {type(e).__name__}: {str(e)[:150]}"


def _run_one(ws, tid, model_name, think, backend, ui, retries=DEFAULT_RETRIES, depth=0):
    """One unit of work (a template or a family). It climbs LADDER by itself until a regex passes the checks:
        normal       the first run;
        deeper       more and more DIFFERENT sample messages, more steps, more reasoning, and what the checks said last;
        soft landing a regex that reads at least SOFT_MATCH_RATE of the family is accepted (a messy family has several layouts).
    The messages a soft-landing regex misses become a unit of their own and get a second regex (up to MAX_LEFTOVER_DEPTH), so nothing is lost.
    A server that does not answer ends the climb at once (the caller restarts it). Returns (result, attempts, seconds); the transcript goes
    to logs/<template_id>.log."""
    ui.set_log(LOG_DIR / f"{tid}.log")
    t0 = time.time()
    result = "not run"
    ws.match_floor.pop(tid, None)                         # a ladder always starts at the normal rung, also on a second call
    ws.samples_for.pop(tid, None)
    ws.fp_ceiling.pop(tid, None)
    ws.deadline[tid] = t0 + ws.budget_s
    rungs = LADDER[:1 + max(0, retries)]
    for run, (rung, n_samples, recursion, effort) in enumerate(rungs, 1):
        if run > 1 and time.time() > ws.deadline[tid]:
            result = f"no regex saved (time budget of {ws.budget_s:.0f}s used)"
            ui.log_only(Text(f"--- no rung {run}: {result}"))
            break
        ws.history[tid].clear()                           # every rung gets its own window for the stall check
        prompt = f"Write, verify and save the extraction regex for template_id={tid}"
        if run > 1:
            problems = "; ".join(p[:140] for p in ws.last_problems.get(tid, [])[:3])
            ui.say(Text(f"↻ rung {run}/{len(rungs)} ({rung}): the previous run ended with: {result}", style="bold yellow"))
            prompt += (f". Earlier runs failed ({result}). The checks last said: {problems or 'nothing recorded'}. Take a different approach: keep "
                       "this template's fixed wording as literal anchors and fix whatever check failed.")
            if ui.attempts == 0:                      # the model answered in text without ever calling a tool
                prompt += (" The previous run ended without a single test_regex call: call get_template_samples, then write a pattern and "
                           "call test_regex, then save_regex. Do not reply with text only.")
            ws.samples_for[tid] = n_samples
            ws.notes[tid].append(f"rung {run}: {rung}")
            ws.fp_ceiling[tid] = SOFT_FP_CEILING
            prompt += (f" The limit for matching OTHER templates' messages is now {SOFT_FP_CEILING:.0%} (test_regex reports it): close variants of this "
                       "message type exist, so a regex that also reads them is fine; keep it specific to this template's own fixed wording otherwise.")
        if rung == "soft landing":
            ws.match_floor[tid] = SOFT_MATCH_RATE
            prompt += (f" This template has several different layouts, which is why the normal checks keep failing. Write the regex for the MAIN "
                       f"layout(s) so that it matches at least {SOFT_MATCH_RATE:.0%} of the messages (test_regex reports the floor); the messages "
                       "it does not match get their own regex afterwards. Do not try to cover every layout in one pattern.")
        # a run that never tested anything usually thought for too long: retry without the reasoning mode
        result = _agent_run(ws, tid, prompt, model_name, think and not (run > 1 and ui.attempts == 0), backend, ui, _effort_for(model_name, effort), recursion)
        ui.log_only(Text(f"--- run {run}/{len(rungs)} ({rung}) result: {result}"))
        if result == "SAVED" or is_connection_error(result):
            break
    if result == "SAVED" and depth < MAX_LEFTOVER_DEPTH:
        entry = load_registry().get(tid, {})
        if entry.get("partial"):
            left = leftover_texts(entry["expanded_pattern"], ws.texts[tid])
            if len(left) >= LEFTOVER_MIN_TEXTS:
                child = f"{tid}~{depth + 1}"
                ws.add_unit(child, ws.category_of(tid), ws.registry.at[tid, "template"], left, parent=tid)
                ws.notes[tid].append(f"{len(left)} messages left for a second regex ({child})")
                ui.say(Text(f"↳ the regex misses {len(left)} distinct messages: writing a second regex for them ({child})", style="bold yellow"))
                sub_result, sub_attempts, _ = _run_one(ws, child, model_name, think, backend, Progress(False), retries, depth + 1)
                ui.attempts += sub_attempts
                ws.notes[tid].append(f"second regex {child}: {sub_result}")
                ui.log_only(Text(f"--- second regex for the leftover messages ({child}): {sub_result}"))
    ui.close_log()
    return result, ui.attempts, time.time() - t0


def run_templates(ws, template_ids, model_name=None, skip_done=True, verbose=True, think=False,
                  backend=DEFAULT_BACKEND, workers=1, retries=DEFAULT_RETRIES, retry_failed=False):
    model_name = model_name or default_model(backend)
    ui = Progress(verbose)
    ui.say(Rule(f"{model_name}  ·  {backend}  ·  {len(template_ids)} template(s)  ·  {workers} worker(s)"))
    results, rows = {}, {}
    todo = []
    for i, tid in enumerate(template_ids, 1):
        msgs = int(ws.registry.at[tid, "message_count"])
        if ws.is_constant_notice(tid) and not (skip_done and tid in load_registry()):
            results[tid] = "constant notice"
            ui.say(Text(f"Template {i}/{len(template_ids)} {tid}: constant notice (all {msgs} messages are the same text, nothing to extract), no regex needed", style="dim"))
            rows[tid] = (tid, msgs, results[tid], 0, 0.0)
        elif skip_done and not retry_failed and (known := known_failure(tid)):
            results[tid] = "failed before (skipped)"
            ws.last_problems[tid] = [known["blocked_by"]] if known.get("blocked_by") else []
            ui.say(Text(f"Template {i}/{len(template_ids)} {tid}: failed on {known['failed_at'][:10]} ({str(known.get('result', ''))[:60]}), skipping "
                        "(retry_failed=True tries it again; it is tried again by itself after the checks change or after "
                        f"{RETRY_FAILED_DAYS} days)", style="dim"))
            rows[tid] = (tid, msgs, results[tid], 0, 0.0)
        elif skip_done and tid in load_registry():
            results[tid] = "already saved"
            ui.say(Text(f"Template {i}/{len(template_ids)} {tid}: already saved, skipping", style="dim"))
            rows[tid] = (tid, msgs, results[tid], 0, 0.0)
        else:
            todo.append((i, tid, msgs))

    wall0 = time.time()
    if workers <= 1:   # one agent at a time, with the live step-by-step display
        for i, tid, msgs in todo:
            ui.start_template(ws, tid, i, len(template_ids))
            results[tid], attempts, secs = _run_one(ws, tid, model_name, think, backend, ui, retries)
            ui.say(Text(f"[{tid}] {results[tid]}  ({secs:.0f}s)\n", style="bold"))
            rows[tid] = (tid, msgs, results[tid], attempts, secs)
    else:              # several agents at once: vLLM batches their requests; one line per finished template
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_run_one, ws, tid, model_name, think, backend, Progress(False), retries): (i, tid, msgs)
                    for i, tid, msgs in todo}
            pending = set(futs)
            while pending:
                done, pending = wait(pending, timeout=HEARTBEAT_S, return_when=FIRST_COMPLETED)
                if not done:
                    ui.say(Text(f"… still working ({time.time() - wall0:.0f}s): {len(rows)}/{len(template_ids)} done, {len(pending)} left "
                                f"(a hard family can take several minutes)", style="dim"))
                    continue
                for fut in done:
                    i, tid, msgs = futs[fut]
                    results[tid], attempts, secs = fut.result()
                    ui.say(Text(f"[{tid}] {results[tid]}  ({attempts} attempts, {secs:.0f}s)  "
                                f"{len(rows) + 1}/{len(template_ids)} done", style="bold"))
                    rows[tid] = (tid, msgs, results[tid], attempts, secs)
    ui.say(Text(f"wall time: {time.time() - wall0:.0f}s", style="dim"))
    failed_now = {tid: {"category": ws.category_of(tid), "messages": msgs, "result": str(results[tid])[:120],
                        "blocked_by": ((ws.last_problems.get(tid) or [""])[0])[:200], "checks_version": CHECKS_VERSION,
                        "failed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                  for _, tid, msgs in todo if results[tid] not in OK_RESULTS and not is_connection_error(results[tid])}
    _store_failures(failed_now, [tid for _, tid, _ in todo if results[tid] in OK_RESULTS])      # a family that now has a regex is no longer "failed"
    ui.summary([rows[t] for t in template_ids])
    failed = [t for t in template_ids if results[t] not in OK_RESULTS]
    for t in failed:
        ui.say(Text(f"  ✗ {t}: see {LOG_DIR.name}/{t}.log", style="red"))
    return results


def _checks_fingerprint():
    """A short hash of everything that decides whether a regex passes: the code of the checks and their limits. A family that failed under other
    checks is worth another try, so this changes by itself whenever a check or a limit is changed (nobody has to remember to bump a number)."""
    try:
        import hashlib
        import inspect
        parts = [inspect.getsource(evaluate), inspect.getsource(Workspace.other_template_texts), inspect.getsource(Workspace._similar_units),
                 inspect.getsource(Workspace.related_units), inspect.getsource(comp),
                 repr((MIN_MATCH_RATE, MIN_HYDRATED_RATE, MAX_FALSE_POSITIVE, MIN_ANCHOR_WORDS, SOFT_MATCH_RATE, SOFT_FP_CEILING, SIMILAR_TEMPLATE_RATIO,
                       MIN_OTHER_TEMPLATE_MESSAGES, STALL_ATTEMPTS, STALL_PROGRESS, LADDER))]
        return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:10]
    except (OSError, TypeError):                       # no source available (a compiled-only install): fall back to a fixed label
        return "unknown"


CHECKS_VERSION = _checks_fingerprint()
