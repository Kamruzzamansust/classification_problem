"""generic_extractors.py - the fallback for messages whose template has no regex (yet): amounts, dates, times, links, phone numbers.

A new template appears every day, and the regex agent only covers the templates that make up 80% of a category. Instead of leaving the rest empty,
these fixed rules read what every SMS has in common. They are deterministic (no LLM, no network), fast, and deliberately conservative: a value is only
returned when it is clearly of its kind, and what is returned is minimised for privacy.

    extract_generic("TK 18,500.00 withdrawn from A/C#10000**6001 on 06-06-26 @08:34 PM from SERPUR ATM. Balance TK 2,949.31. Helpline 16216")
    -> {"debited_amount": 18500.0, "balance": 2949.31, "currency": "BDT", "date": "2026-06-06", "time": "20:34"}

What is read
    amounts   with a currency (Tk / BDT / ৳ / taka / Rs / $ ...) or written as 1,250.00 / 99.50. A plain integer is NOT an amount (it may be an OTP or an id).
              The key says what the amount is when the words around it tell: balance, fee, due_amount, paid_amount, recharge_amount, credited_amount,
              debited_amount, reward_amount, total_amount, price, limit_amount; otherwise `amount`. A second one gets `_2`.
    dates     06-06-26, 06/06/2026, 2026-06-06, 6 Jun 2026, 06-Jun-26, Jun 6, 2026 -> 2026-06-06 (day first); `expiry_date` / `due_date` when the words say so
    times     08:34 PM, 20:15, 9:05:30 -> HH:MM
    links     only the DOMAIN of a link (labaiddiagnostics.com): the path and query can carry tokens or ids, so they are dropped
    phones    Bangladeshi mobile numbers, reduced to the last four digits ("****5678") like every other identifier
Never read: OTPs, PINs, passwords, e-mail addresses, account / card numbers, plain integers.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Optional

import regex

from . import normalizers

GENERIC_PATTERN = "generic-fallback"          # what the `regex_pattern` column holds for a row that was filled by the fallback

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_NUM = r"(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?"                  # 1,250.50  1,00,000  99.5  500
_WORDS = regex.compile(r"[\p{L}\p{M}]+")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+"
                  r"|(?<![\w@./-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|app|bd|co|io|ly|me|info|xyz|gov|edu)\b(?:/[^\s<>\"')\]]*)?", re.I)
_NOT_A_HOST = {"md", "mr", "mrs", "ms", "dr", "mst", "prof", "no", "tk", "rs", "e", "i", "a", "b", "etc", "vs"}      # "Md.Co", "Dr.Me": a title and a name, not a link
_DATE = re.compile(r"(?<!\d)\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?!\d)"
                   r"|(?<![\d.])\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}(?!\d)"
                   rf"|(?<!\d)\d{{1,2}}(?:st|nd|rd|th)?[-\s/.]*{_MONTHS}\.?,?[-\s/.]*\d{{2,4}}(?!\d)"
                   rf"|\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}(?!\d)", re.I)
_TIME = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?(?![\d:])", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?88[-\s]?)?0?1[3-9](?:[-\s]?\d){8}(?!\d)")                  # 01712345678, 01712-345678, 017-1234-5678, +88 017...
_CUR = r"(?:tk|bdt|rs|usd|inr|taka|৳|\$|টাকা)"
_AMOUNT_PREFIX = re.compile(rf"(?<![A-Za-z]){_CUR}\.?\s*[-:]?\s*({_NUM})(?:\s*/-)?", re.I)
_AMOUNT_WORD = re.compile(rf"(?<![A-Za-z])(?:amt|amount)\b\.?\s*(?:of\s+|is\s+)?[:=-]?\s*({_NUM})", re.I)       # "Amt.1234.50", "Amount: 500": the word says it is money
_AMOUNT_SUFFIX = re.compile(rf"(?<![\d,.])({_NUM})\s*(?:/-|{_CUR}(?![A-Za-z]))", re.I)
_AMOUNT_FORMATTED = re.compile(r"(?<![\d,.])(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,4})?(?!\d|,\d)|\d+\.\d{1,2}(?!\d|\.\d))"
                               r"(?!\s*(?:gb|mb|kb|tb|%|percent|km|kg|gm|ml|mbps|kbps|minutes?|mins?|hours?|hrs?|days?|sms|x\b))", re.I)
_CURRENCY_OF = {"tk": "BDT", "bdt": "BDT", "taka": "BDT", "৳": "BDT", "টাকা": "BDT", "rs": "INR", "inr": "INR", "usd": "USD", "$": "USD"}

# What an amount is, from the words around it (nearest first). Anything not listed stays a plain `amount`.
_BEFORE = {
    "balance": "balance", "bal": "balance", "avl": "balance", "available": "balance", "avail": "balance", "remaining": "balance",
    "ব্যালেন্স": "balance", "ব্যালান্স": "balance", "বর্তমান": "balance",
    "total": "total_amount", "due": "due_amount", "payable": "due_amount", "outstanding": "due_amount", "বকেয়া": "due_amount",
    "bill": "bill_amount", "বিল": "bill_amount", "fee": "fee", "fees": "fee", "charge": "fee", "charges": "fee", "vat": "fee", "tax": "fee",
    "ফি": "fee", "চার্জ": "fee", "paid": "paid_amount", "payment": "paid_amount", "পরিশোধ": "paid_amount",
    "recharge": "recharge_amount", "recharged": "recharge_amount", "topup": "recharge_amount", "রিচার্জ": "recharge_amount",
    "cashback": "reward_amount", "bonus": "reward_amount", "reward": "reward_amount", "limit": "limit_amount",
    "price": "price", "fare": "price", "cost": "price", "মূল্য": "price",
    "deposit": "credited_amount", "deposited": "credited_amount", "received": "credited_amount", "refund": "credited_amount", "জমা": "credited_amount",
    "withdraw": "debited_amount", "withdrawal": "debited_amount", "cashout": "debited_amount", "উত্তোলন": "debited_amount",
    "sent": "debited_amount", "send": "debited_amount", "transfer": "debited_amount", "purchase": "debited_amount",
    "credited": "credited_amount", "debited": "debited_amount", "debit": "debited_amount", "cod": "cod_amount",
}
_AFTER = {
    "credited": "credited_amount", "deposited": "credited_amount", "deposit": "credited_amount", "received": "credited_amount", "added": "credited_amount",
    "refunded": "credited_amount", "জমা": "credited_amount", "withdrawal": "debited_amount", "withdraw": "debited_amount", "vat": "fee", "charge": "fee",
    "debited": "debited_amount", "withdrawn": "debited_amount", "spent": "debited_amount", "purchased": "debited_amount", "purchase": "debited_amount",
    "paid": "paid_amount", "transferred": "debited_amount", "sent": "debited_amount", "charged": "debited_amount", "deducted": "debited_amount",
    "cashout": "debited_amount", "উত্তোলন": "debited_amount", "কর্তন": "debited_amount", "recharged": "recharge_amount",
}                                                                           # (no "due": "declined due to ..." is not a due amount)
_FILLER = {"has", "have", "had", "been", "is", "was", "are", "were", "will", "be", "successfully", "a", "an", "the", "your", "of", "to", "from", "in", "on", "by"}
_CUR_WORDS = {"tk", "bdt", "rs", "usd", "inr", "taka", "টাকা"}
_DATE_LABEL = {"expiry": "expiry_date", "expires": "expiry_date", "expire": "expiry_date", "expired": "expiry_date", "valid": "expiry_date",
               "validity": "expiry_date", "till": "expiry_date", "until": "expiry_date", "মেয়াদ": "expiry_date", "due": "due_date"}
_EXTRA_DATE_FORMATS = ("%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%B %d %Y", "%b %d, %y", "%B %d, %y", "%d %B, %Y", "%d %B, %y", "%d-%B-%Y", "%d-%B-%y", "%d%b%Y")
MIN_YEAR, MAX_YEAR = 2015, 2040                   # a "date" outside this window is more likely a version number or an id
MAX_AMOUNT = 1e9


def _overlaps(a0: int, a1: int, spans) -> bool:
    return any(a0 < e and s < a1 for s, e in spans)


def _words_before(s: str, start: int, n: int, skip=()) -> list[str]:
    """The n words just before `start`, nearest first - but only back to the previous number: 'Vat:75.24,Rebate:' labels the next amount 'rebate',
    not 'vat'."""
    window = re.split(r"\d", s[max(0, start - 45):start])[-1]
    ws = [w for w in _WORDS.findall(window.lower()) if w not in skip]
    return ws[-n:][::-1]


def _words_after(s: str, end: int, n: int) -> list[str]:
    """The first n meaningful words after `end`, up to the next number."""
    window = re.split(r"\d", s[end:end + 45])[0]
    return [w for w in _WORDS.findall(window.lower()) if w not in _FILLER][:n]


def _amount_label(s: str, start: int, end: int) -> str:
    for w in _words_before(s, start, 3, _CUR_WORDS):
        if w in _BEFORE:
            return _BEFORE[w]
    for w in _words_after(s, end, 3):
        if w in _AFTER:
            return _AFTER[w]
    return "amount"


def _date(raw: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", raw, flags=re.I)).strip(" .,")
    iso = None
    try:
        iso = normalizers.date_dmy(t)
    except normalizers.NormalizationError:
        for f in _EXTRA_DATE_FORMATS:
            try:
                iso = datetime.strptime(t, f).date().isoformat()
                break
            except ValueError:
                continue
    return iso if iso and MIN_YEAR <= int(iso[:4]) <= MAX_YEAR else None


def _time(raw: str) -> Optional[str]:
    try:
        return normalizers.time_24h(raw)
    except normalizers.NormalizationError:
        return None


def _domain(url: str) -> Optional[str]:
    host = re.sub(r"^(?:https?://)?(?:[^@/]*@)?", "", url, flags=re.I)
    host = re.split(r"[/:?#]", host)[0].lower().strip(".,;")
    host = host[4:] if host.startswith("www.") else host
    if not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", host) or host.split(".")[0] in _NOT_A_HOST:
        return None
    return host


def _put(out: dict, key: str, value) -> None:
    """Add `value` under `key`; a second different value goes to key_2, key_3...; the same value twice is kept once."""
    if key not in out:
        out[key] = value
        return
    existing = [out[key]] + [out[k] for k in (f"{key}_{i}" for i in range(2, 9)) if k in out]
    if value in existing or len(existing) >= 8:
        return
    out[f"{key}_{len(existing) + 1}"] = value


def amount_spans(text) -> list[tuple[int, int, float]]:
    """Where the amounts of a message are: (start, end, value) of each NUMBER (without its currency word), in the text as NFC-normalised.
    Used to check that a regex captures an amount in ONE group instead of cutting it into pieces."""
    if not isinstance(text, str) or not text.strip():
        return []
    return _scan(text, False)[1]


def extract_generic(text, keep_identifiers: bool = False) -> Optional[dict]:
    """The generic information of one message, or None when it has none (see the module docstring)."""
    if not isinstance(text, str) or not text.strip():
        return None
    return _scan(text, keep_identifiers)[0]


def _scan(text: str, keep_identifiers: bool):
    """(information dict or None, [(num_start, num_end, value)] of the amounts)."""
    s = unicodedata.normalize("NFC", text).translate(_BN_DIGITS)
    taken: list[tuple[int, int]] = []                                  # spans already read: nothing is read twice
    domains: list[str] = []
    for m in _URL.finditer(s):                                         # links first: a URL may hold 'user:pw@host', which is not an e-mail address
        taken.append(m.span())
        d = _domain(m.group(0))
        if d and d not in domains:
            domains.append(d)
    for m in _EMAIL.finditer(s):
        if not _overlaps(*m.span(), taken):
            taken.append(m.span())                                     # an e-mail address is personal data: skipped, and not mistaken for a link

    dates, times, phones, amounts, spans = [], [], [], [], []
    if any(c.isdigit() for c in s):
        for m in _DATE.finditer(s):
            if _overlaps(*m.span(), taken):
                continue
            iso = _date(m.group(0))
            if iso:
                taken.append(m.span())
                words = _words_before(s, m.start(), 2)
                dates.append((m.start(), next((_DATE_LABEL[w] for w in words if w in _DATE_LABEL), "date"), iso))
        for m in _TIME.finditer(s):
            if _overlaps(*m.span(), taken):
                continue
            t = _time(m.group(0))
            if t:
                taken.append(m.span())
                times.append((m.start(), "time", t))
        for m in _PHONE.finditer(s):
            if _overlaps(*m.span(), taken):
                continue
            digits = re.sub(r"\D", "", m.group(0))
            digits = digits[2:] if digits.startswith("88") and len(digits) == 13 else digits
            digits = "0" + digits if len(digits) == 10 else digits
            taken.append(m.span())
            phones.append((m.start(), "phone", digits if keep_identifiers else "****" + digits[-4:]))
        for rx in (_AMOUNT_PREFIX, _AMOUNT_SUFFIX, _AMOUNT_WORD, _AMOUNT_FORMATTED):   # explicit currency first, then "Amt", then the formatted numbers
            for m in rx.finditer(s):
                if _overlaps(*m.span(), taken):
                    continue
                try:
                    value = normalizers.amount(m.group(1))
                except normalizers.NormalizationError:
                    continue
                if value > MAX_AMOUNT:
                    continue
                num_start = m.start(1)
                if num_start > 0 and s[num_start - 1] == "-" and (num_start == 1 or s[num_start - 2] in " :(=") and rx is not _AMOUNT_PREFIX:
                    value = -value                                          # "Rebate:-7.29" is a negative amount, not 7.29
                cur_word = re.match(rf"\s*({_CUR})", m.group(0), re.I) or re.search(rf"({_CUR})(?![A-Za-z])\s*(?:/-)?$", m.group(0), re.I)
                cur = _CURRENCY_OF.get(cur_word.group(1).lower()) if cur_word else None
                taken.append(m.span())
                amounts.append((m.start(), _amount_label(s, m.start(), m.end()), value, cur))
                spans.append((m.start(1), m.end(1), value))

    out: dict = {}
    currency = None
    for _, label, value, cur in sorted(amounts):
        _put(out, label, value)
        currency = currency or cur
    if currency:
        out["currency"] = currency
    for _, label, value in sorted(dates) + sorted(times) + sorted(phones):
        _put(out, label, value)
    for d in domains:
        _put(out, "link_domain", d)
    return (out or None), sorted(spans)
