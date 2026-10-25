"""completeness.py - has the regex (plus the fixed facts the agent declared) accounted for everything in the messages that carries information?

The regex agent used to decide for itself what was worth extracting, so it silently dropped fixed wording such as the organisation or
the branch ("... for choosing Popular Diagnostic Centre (Narayanganj) as your healthcare partner"). A regex only reads what VARIES inside
one template, and the branch is the same in every message of that template. This module is the objective check that closes the gap -
no LLM, no per-category schema:

    in every message the regex matched, find the parts that usually carry information
        - numbers   (amounts, dates, times, ids, phone numbers, "5")
        - links     (urls, e-mail addresses)
        - names     (proper-noun-like words: capitalised or ALL-CAPS Latin words that are not common SMS wording)
    and report the ones that are neither inside a captured group nor inside a declared fixed fact.

The agent then either captures them (value varies) or declares them (value is fixed wording): fixed_facts = {"branch": "Narayanganj"}.
Facts are checked too: the text must really occur in the template's messages, so nothing can be invented.

Limits (by design): Bengali words have no capitals, so only digits and links are checked in Bengali text; the check looks at what is
left out, not at whether a captured value is right (the other checks in regex_writer.evaluate do that).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import NamedTuple, Optional

from . import normalizers

from .generic_extractors import amount_spans
from .templates import tokenize

MIN_LEFT_OUT_SHARE = 0.30       # a part counts as dropped when it is left out in at least this share of the matched messages
MAX_FACTS = 8
MAX_FACT_LEN = 80
FACT_MIN_SHARE = 0.95           # a declared fact must occur verbatim in this share of the template's messages
MAX_BLOCKS = 4                  # distinct incomplete regexes per template that are sent back to the agent; after that the gaps are only recorded
MAX_GROUP_SHARE = 0.5           # a group that swallows more than this share of the message (and >= MIN_FREE_TEXT chars) is free text, not a field
MIN_FREE_TEXT = 40
MAX_FIELD_LEN = 120

_NAME_OK = re.compile(r"[a-z][a-z0-9_]{0,39}")
_EVENT_OK = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+){0,4}")
# field names that mean a login credential or card secret (whole words: "shipping" is not "pin"); shared with extract_information
SECRET_NAME = re.compile(r"(^|_)(otp|pin|password|passwd|passcode|cvv|cvc|secret|verification_code|security_code|username|user_name|login|login_id|userid)($|_)")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+|(?<![\w@.-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|bd|co|io|info|app|gov|edu|ly|me|xyz)\b(?:/\S*)?", re.I)
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
_WORD = re.compile(r"(?<![A-Za-z])[A-Z][A-Za-z]{2,}(?![A-Za-z])")
_SENTENCE_END = re.compile(r"(?:^|[.!?।\n])\s*$")
# A word right before ':' / '-' / '=' ("Qty:1.00", "Driver- ZAMAN"), or before "No" / "ID" / "Dt" and a separator or digit ("Challan No:123", "Trip No 6000073",
# "Vehicle No.Dhaka") is the LABEL of a field, i.e. fixed wording: it is not asked for.
_LABEL_AFTER = re.compile(r"\s*[:=\-–#]|\s*(?:no|id|dt|date|nr|number|name|code|type)\b\.?\s*(?:[:=\-–#.]|\d)", re.I)
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATETIME = re.compile(rf"\d{{1,4}}[-/.]\d{{1,2}}[-/.]\d{{1,4}}|\d{{1,2}}(?:st|nd|rd|th)?[-\s/.]*{_MONTH}\.?,?[-\s/.]*\d{{2,4}}"
                       rf"|{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}|\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:\s*[ap]\.?m\.?)?", re.I)
CONSTANT_SHARE = 0.95            # left out in this share of the messages with the very same text = wording that never changes
MIN_VARIATION_EVIDENCE = 5       # fewer distinct messages than this cannot show whether a number varies

# Ordinary wording that is often capitalised in an SMS: not information about the customer's activity on its own.
STOPWORDS = frozenset("""
the and for you your yours our ours not are was were has had have been being will would shall should can could may might must this that these those
with from into onto over under than then there here where when what which who whom whose why how all any each every some such only also just very
please pls kindly thank thanks thankyou welcome hello hi dear sir madam madame regards sincerely greetings congratulations congratulation sorry
good morning evening night happy mr mrs ms miss dr prof md mst mohammad mohammed
customer customers client user users member subscriber valued team staff agent
sms message messages text mail email call calls dial press reply send sent receive received get got use used using visit click tap open login logout
download install update upgrade app apps application website web site link links page portal online offline
account accounts balance balances amount amounts total due paid pay payment payments charge charges fee fees vat tax taxes cost price
bill bills invoice invoices receipt transaction transactions trxid txnid txn trx ref reference number numbers code codes id ids no
date dates time times day days week weeks month months year years hour hours minute minutes second seconds today tomorrow yesterday now
jan feb mar apr may jun jul aug sep sept oct nov dec january february march april june july august september october november december
mon tue wed thu fri sat sun monday tuesday wednesday thursday friday saturday sunday
tk bdt taka usd bank card cards cash credit debit deposit withdraw withdrawal withdrawn transfer transferred cashout cashin balance
mobile phone sim number helpline hotline line care service services support help office branch center centre agent
successful successfully success failed failure fail pending approved confirmed confirm confirmation cancelled canceled request requested
offer offers pack package packages plan plans data internet minutes validity valid expire expires expired expiry expiration recharge bonus
current available last next new old first second third
report reports ready collect delivery counter appointment appointments registration register registered serial doctor doctors patient patients
password pin otp secret cvv username login
limited ltd plc inc company corp
bangladesh bangla english
details detail info information note notice alert reminder update status
fund funds loan loans installment installments emi premium policy
order orders item items product products parcel parcels shipment tracking
uhid mrn name url choosing refund consultant lab unit opd health
in on at of to is it as by be or an if so we us my me no yes ok
""".split())


class Part(NamedTuple):
    kind: str       # "number" | "url" | "word"
    text: str
    start: int
    end: int


def _overlaps(a0, a1, spans):
    return any(a0 < e and s < a1 for s, e in spans)


def salient_parts(text: str) -> list[Part]:
    """The parts of one message that usually carry information (see the module docstring)."""
    text = unicodedata.normalize("NFC", text)
    parts, taken = [], []
    for m in list(_EMAIL.finditer(text)) + list(_URL.finditer(text)):
        if _overlaps(m.start(), m.end(), taken):
            continue
        taken.append((m.start(), m.end()))
        parts.append(Part("url", m.group(0), m.start(), m.end()))
    for m in _NUMBER.finditer(text):
        if not _overlaps(m.start(), m.end(), taken):
            parts.append(Part("number", m.group(0), m.start(), m.end()))
    for m in _WORD.finditer(text):
        if _overlaps(m.start(), m.end(), taken):
            continue
        w = m.group(0)
        if w.lower() in STOPWORDS:
            continue
        if not w.isupper() and _SENTENCE_END.search(text[:m.start()]):
            continue                                     # "Congratulations ..." at the start of a sentence is ordinary capitalisation
        if _LABEL_AFTER.match(text, m.end()):
            continue                                     # a field label ("Qty:", "Challan No:")
        parts.append(Part("word", w, m.start(), m.end()))
    return sorted(parts, key=lambda p: p.start)


def group_spans(match) -> dict[str, tuple[int, int]]:
    return {n: match.span(n) for n in match.re.groupindex if match.group(n) and match.group(n).strip()}


def _fact_rx(value: str):
    """The fact as a regex that ignores differences in spacing / line breaks ('Epic Health\nCare', double spaces)."""
    return re.compile(r"\s+".join(re.escape(w) for w in value.split()))


def fact_spans(text: str, facts: Optional[dict]) -> list[tuple[int, int]]:
    return [m.span() for v in (facts or {}).values() for m in _fact_rx(v).finditer(text)]


# A value right after one of these words is a credential: it is never output, so the agent must not be asked to capture or declare it.
_SECRET_BEFORE = re.compile(r"\b(?:password|passwd|passcode|pin|otp|cvv|cvc|secret|user\s*name|user\s*id|user|login(?:\s*id)?|verification\s*code)"
                            r"\s*[:=\-]?\s*(?:is|are)?\s*[:=\-]?\s*$", re.I)


def is_secret_value(text: str, part: Part) -> bool:
    before = re.sub(r"[A-Za-z]{1,3}$", "", text[max(0, part.start - 40):part.start])        # an id prefix glued to the value: 'User: H79609008757'
    return bool(_SECRET_BEFORE.search(before))


def uncovered_parts(text: str, match, facts: Optional[dict] = None) -> list[Part]:
    text = unicodedata.normalize("NFC", text)
    covered = list(group_spans(match).values()) + fact_spans(text, facts)
    return [p for p in salient_parts(text) if not _overlaps(p.start, p.end, covered) and not is_secret_value(text, p)]


_MONEY_WORDS = {"amount", "balance", "fee", "fees", "charge", "charges", "price", "fare", "cost", "paid", "due", "emi", "installment", "instalment",
                "total", "payment", "principal", "interest", "deposit", "withdrawal", "cashback", "bonus", "premium", "limit"}
_NOT_MONEY_WORDS = {"no", "number", "count", "code", "id", "ref", "reference", "date", "time", "month", "months", "days", "day", "year", "tenure",
                    "period", "rate", "percent", "pct", "type", "mode", "status", "name", "method", "channel", "frequency", "txn", "trx"}


def money_problem(name: str, values) -> Optional[str]:
    """A field named like money (amount, balance, fee, emi...) must hold amounts that are not all zero. Catches 'TK2613.0' cut into a
    'repayment_code' 2613 and an 'amount' 0."""
    parts = set(name.lower().split("_"))
    vals = [v for v in values if v is not None and str(v).strip()]
    if not (parts & _MONEY_WORDS) or parts & _NOT_MONEY_WORDS or len(vals) < 5:
        return None
    nums = []
    for v in vals:
        try:
            nums.append(float(v) if isinstance(v, (int, float)) else normalizers.amount(str(v)))
        except normalizers.NormalizationError:
            pass
    if len(nums) / len(vals) < 0.7:
        bad = next(str(v) for v in vals if str(v) not in {str(n) for n in nums})
        return f"field {name!r} is named like money but most of its values are not amounts (e.g. {bad[:30]!r}): capture the number itself"
    if sum(n == 0 for n in nums) / len(nums) >= 0.9:
        return (f"field {name!r} is 0 in almost every message: it captures the wrong part of the number (for 'TK2613.0' the amount is 2613.0, "
                "not the 0 after the dot): capture the whole number, including its decimals, in one group")
    return None


def measure(pairs, facts: Optional[dict] = None, template: Optional[str] = None) -> dict:
    """pairs = [(text, regex match)] for the messages the regex matched. Returns, for each kind of part, the share of messages in
    which it is left out, plus the groups that swallow free text.
    Not counted as left out (nothing sensible the agent could do about them):
      - a number that is the very same in (almost) every message, or when there are too few distinct messages to see it vary
        (a hotline, "UNIT-2"), unless it is part of a date or time: the sample covers a short period, a date that is constant in it varies in
        real life and must be captured;
      - a word that is the same in (almost) every message but is not fixed wording of the template (`template` given): it sits in a
        masked slot that happens to hold one value in this sample."""
    n = max(1, len(pairs))
    fixed = _literal_words(template) if template is not None else None
    split_msgs, split_examples, split_groups = 0, [], set()
    per_msg: list[list[Part]] = []
    dated: set[str] = set()
    num_seen: Counter = Counter()
    word_seen: Counter = Counter()
    big: Counter = Counter()
    for text, m in pairs:
        text = unicodedata.normalize("NFC", text)
        left = uncovered_parts(text, m, facts)
        per_msg.append(left)
        dt = [x.span() for x in _DATETIME.finditer(text)]
        for p in left:
            if p.kind == "number" and _overlaps(p.start, p.end, dt):
                dated.add(p.text)
        for t in {p.text for p in left if p.kind == "number"}:
            num_seen[t] += 1
        for w in {p.text for p in left if p.kind == "word"}:
            word_seen[w] += 1
        spans = group_spans(m)
        for name, (s, e) in spans.items():
            if e - s > MAX_FIELD_LEN or (e - s >= MIN_FREE_TEXT and (e - s) > MAX_GROUP_SHARE * len(text)):
                big[name] += 1
        cut = [(ns, ne, v, [g for g, (s, e) in spans.items() if s < ne and ns < e]) for ns, ne, v in amount_spans(text)
               if not any(s <= ns and e >= ne for s, e in spans.values())]
        cut = [c for c in cut if c[3]]                    # an amount some group touches but none captures whole; an untouched one is just left out
        if cut:
            split_msgs += 1
            split_groups.update(g for c in cut for g in c[3])
            if len(split_examples) < 3:
                split_examples.append(f"{cut[0][2]:g} (cut by {', '.join(cut[0][3])})")
    few = len(pairs) < MIN_VARIATION_EVIDENCE
    const_nums = {t for t, c in num_seen.items() if t not in dated and (few or c / n >= CONSTANT_SHARE)}
    const_words = {w for w, c in word_seen.items() if fixed is not None and w not in fixed and c / n >= CONSTANT_SHARE and not few}
    words: Counter = Counter()
    kinds: Counter = Counter()
    examples: dict[str, list[str]] = {"number": [], "url": []}
    with_word = 0
    for left in per_msg:
        left = [p for p in left if not (p.kind == "number" and p.text in const_nums) and not (p.kind == "word" and p.text in const_words)]
        for w in dict.fromkeys(p.text for p in left if p.kind == "word"):          # in order of appearance
            words[w] += 1
        with_word += any(p.kind == "word" for p in left)
        for k in dict.fromkeys(p.kind for p in left if p.kind != "word"):
            kinds[k] += 1
            for p in left:
                if p.kind == k and p.text not in examples[k] and len(examples[k]) < 4:
                    examples[k].append(p.text)
    return {
        # judged per MESSAGE, not per word: a varying part (a branch, a merchant) is a different word in every message, so each word is rare
        "words": {w: round(c / n, 2) for w, c in words.most_common(10)} if with_word / n >= MIN_LEFT_OUT_SHARE else {},
        "numbers": round(kinds["number"] / n, 2), "links": round(kinds["url"] / n, 2),
        "number_examples": examples["number"], "link_examples": examples["url"],
        "free_text_groups": sorted(g for g, c in big.items() if c / n >= MIN_LEFT_OUT_SHARE),
        "split_amounts": {"share": round(split_msgs / n, 2), "examples": split_examples, "groups": sorted(split_groups)},
        "messages": len(pairs),
    }


def gaps(meas: dict) -> bool:
    return bool(meas["words"] or meas["numbers"] >= MIN_LEFT_OUT_SHARE or meas["links"] >= MIN_LEFT_OUT_SHARE)


def describe_gaps(meas: dict, mask=lambda s: s) -> str:
    """The message sent back to the agent."""
    bits = []
    if meas["words"]:
        bits.append("names/words " + str(list(meas["words"])[:10]))
    if meas["numbers"] >= MIN_LEFT_OUT_SHARE:
        bits.append(f"numbers (e.g. {', '.join(repr(mask(x)) for x in meas['number_examples'][:3])}) in {meas['numbers']:.0%} of the messages")
    if meas["links"] >= MIN_LEFT_OUT_SHARE:
        bits.append(f"links (e.g. {', '.join(repr(mask(x)) for x in meas['link_examples'][:2])}) in {meas['links']:.0%} of the messages")
    return ("incomplete: these parts of the messages are neither captured by a group nor declared in fixed_facts: " + "; ".join(bits) +
            ". If the value changes between messages, capture it with a named group. If it is the same in every message but says "
            "who / where / which product or plan / which number or link (organisation, brand, branch, city, plan, support number...), "
            "declare it: fixed_facts={\"branch\": \"<exact text from the messages>\"}. Greetings and labels need nothing.")


def leftover(meas: dict) -> dict:
    """What stays unextracted (kept in the registry when the agent could not close the gap)."""
    out = {}
    if meas["words"]:
        out["words"] = list(meas["words"])[:10]
    if meas["numbers"] >= MIN_LEFT_OUT_SHARE:
        out["numbers_in_share_of_messages"] = meas["numbers"]
    if meas["links"] >= MIN_LEFT_OUT_SHARE:
        out["links_in_share_of_messages"] = meas["links"]
    return out


def _literal_words(template: str) -> set[str]:
    """The fixed words of a template: its tokens minus the masked slots ([NUM] [VAR] [CODE]...). A bracketed brand such as
    [Pathao] is a word, not a slot."""
    out = set()
    for t in tokenize(template):
        if len(t) > 1 and t[0] == "[" and t[-1] == "]":
            if t[1:-1].isupper():
                continue                                      # masked slot
            t = t[1:-1]
        out.add(t)
    return out


def split_amount_problem(meas: dict) -> Optional[str]:
    """Always enforced (a wrong value is worse than a missing one): an amount must be captured whole by ONE group."""
    sp = meas["split_amounts"]
    if sp["share"] < MIN_LEFT_OUT_SHARE:
        return None
    return (f"an amount is cut into pieces by different groups in {sp['share']:.0%} of the messages (e.g. {'; '.join(sp['examples'][:2])}). One group must "
            "capture the whole number with its decimals, e.g. TK(?P<amount>\\d+(?:\\.\\d+)?) or (?P<amount>{AMOUNT}). The template may show 'TK2613.0' as "
            "[CODE] [NUM], but the sample messages show it is one amount")


def check_facts(facts, texts: list[str], group_names, template: Optional[str] = None) -> tuple[dict, list[str]]:
    """Validate the facts the agent declared. Returns (clean facts, problems). `texts` = the template's messages the regex matched.
    `template` = the template text: every word of a fact must be fixed wording of it. The messages come from a short period (often one day), so a
    date, an amount, a name or a number that merely happens to be constant in them is NOT a fact: it varies in real life, and a stored fact
    would be wrong for every other message."""
    if facts in (None, {}):
        return {}, []
    problems: list[str] = []
    if not isinstance(facts, dict):
        return {}, ["fixed_facts must be a dict {name: exact text from the messages}"]
    if len(facts) > MAX_FACTS:
        problems.append(f"fixed_facts has {len(facts)} entries (max {MAX_FACTS}): keep only the ones that say who / where / which product")
    clean: dict[str, str] = {}
    texts = [unicodedata.normalize("NFC", t) for t in texts]
    for name, value in facts.items():
        if not isinstance(name, str) or not _NAME_OK.fullmatch(name):
            problems.append(f"fixed fact name {name!r} must be snake_case (e.g. branch, organisation)")
            continue
        if SECRET_NAME.search(name):
            problems.append(f"fixed fact {name!r}: never extract credentials or secrets")
            continue
        if name in group_names:
            problems.append(f"fixed fact {name!r} has the same name as a regex group; rename one of them")
            continue
        if not isinstance(value, str):
            problems.append(f"fixed fact {name!r} must be text copied from the messages")
            continue
        value = unicodedata.normalize("NFC", value).strip()
        if not (2 <= len(value) <= MAX_FACT_LEN):
            problems.append(f"fixed fact {name!r}: text must be 2-{MAX_FACT_LEN} characters")
            continue
        if template is not None:
            fixed = _literal_words(template)
            odd = [w for w in tokenize(value) if w not in fixed]
            if odd:
                problems.append(f"fixed fact {name}={value!r}: {odd[:3]} is not fixed wording of this template (a number, date, name or other value that "
                                "varies from message to message); capture it with a group instead")
                continue
        rx = _fact_rx(value)
        found = sum(bool(rx.search(t)) for t in texts) / max(1, len(texts))
        if found < FACT_MIN_SHARE:
            problems.append(f"fixed fact {name}={value!r} appears in only {found:.0%} of the template's messages (need >= {FACT_MIN_SHARE:.0%}); "
                            "copy the text exactly as it is written, or capture it with a group if it varies")
            continue
        clean[name] = value
    return clean, problems


def check_event(event) -> Optional[str]:
    """None if the event label is fine, else why not."""
    if not isinstance(event, str) or not _EVENT_OK.fullmatch(event.strip()) or len(event.strip()) > 50:
        return ("event must be a short snake_case label of what the message announces (max 5 words), "
                "e.g. registration_confirmation, atm_withdrawal, appointment_booked, report_ready")
    return None
