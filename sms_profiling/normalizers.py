"""normalizers.py - turn raw text captured from a message into typed, normalized values.

Every normalizer takes (raw_text, NormContext) and returns the value, or raises NormalizationError - it never
guesses. Assumptions are explicit and documented below:
  * amounts: thousands separators are removed; Bengali digits are converted; up to 4 decimals.
  * dates: DAY-FIRST (dd-mm-yy / dd/mm/yyyy), which is the convention of Bangladeshi senders; a two-digit year
    means 20yy. The `date_near_message` validation rule catches day/month mix-ups.
  * times: 12-hour AM/PM or 24-hour; returned as HH:MM (or HH:MM:SS if seconds were present).
  * account references are sensitive: by default only a masked form with the LAST FOUR digits is kept
    (data minimization); NormContext.keep_full_account=True keeps the masked string exactly as written.
"""
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_CURRENCY_TOKENS = {"tk": "BDT", "tk.": "BDT", "bdt": "BDT", "৳": "BDT", "taka": "BDT", "টাকা": "BDT"}


class NormalizationError(ValueError):
    pass


@dataclass
class NormContext:
    message_timestamp: Optional[datetime] = None
    keep_full_account: bool = False


def _digits(s):
    return s.translate(_BN_DIGITS)


def amount(raw, ctx=None):
    """'TK 18,500.00' -> 18500.0 ; '৳ ১,২৫০' -> 1250.0. Currency symbols/words are ignored here (see `currency`)."""
    s = _digits(raw).strip()
    m = re.fullmatch(r"(?:BDT|TK\.?|Rs\.?|৳|taka|টাকা)?\s*([0-9][0-9,]*(?:\.[0-9]{1,4})?)\s*(?:BDT|TK|taka|টাকা|/-)?", s, re.I)
    if not m:
        raise NormalizationError(f"not an amount: {raw!r}")
    num = m.group(1)
    if not re.fullmatch(r"[0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?", num):
        raise NormalizationError(f"badly grouped number: {raw!r}")
    return round(float(num.replace(",", "")), 4)


def currency(raw, ctx=None):
    """Only a currency that is written in the message: 'TK' / 'Tk' / '৳' / 'BDT' -> 'BDT'. Anything else is an error,
    so a missing or unknown currency stays null instead of being assumed."""
    s = raw.strip().casefold()
    if s in _CURRENCY_TOKENS:
        return _CURRENCY_TOKENS[s]
    raise NormalizationError(f"unknown currency: {raw!r}")


_DATE_FORMATS = ("%d-%m-%y", "%d-%m-%Y", "%d/%m/%y", "%d/%m/%Y", "%d.%m.%y", "%d.%m.%Y", "%d-%b-%y", "%d-%b-%Y",
                 "%d %b %Y", "%d %b, %Y", "%d %b %y", "%d %B %Y", "%Y-%m-%d", "%Y/%m/%d")


def date_dmy(raw, ctx=None):
    s = re.sub(r"\s+", " ", _digits(raw).strip().rstrip("."))
    for f in _DATE_FORMATS:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            continue
    raise NormalizationError(f"not a date: {raw!r}")


def time_24h(raw, ctx=None):
    s = _digits(raw).strip().upper().replace(".", "")
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AP]M)?", s)
    if not m:
        raise NormalizationError(f"not a time: {raw!r}")
    h, mi, sec, ap = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    if ap:
        if not 1 <= h <= 12:
            raise NormalizationError(f"bad 12-hour time: {raw!r}")
        h = h % 12 + (12 if ap == "PM" else 0)
    if h > 23 or mi > 59 or (sec is not None and int(sec) > 59):
        raise NormalizationError(f"time out of range: {raw!r}")
    return f"{h:02d}:{mi:02d}" + (f":{sec}" if sec else "")


def account_reference(raw, ctx=None):
    """Sensitive. '10000**6001' -> '****6001' (last four digits only) unless keep_full_account is set."""
    s = raw.strip()
    if not re.fullmatch(r"[0-9A-Za-z*xX#\-/ ]{4,40}", s):
        raise NormalizationError(f"not an account reference: {raw!r}")
    if ctx is not None and ctx.keep_full_account:
        return s
    digits = re.findall(r"[0-9]", s)
    if len(digits) < 4:
        raise NormalizationError("account reference has fewer than 4 digits")
    return "****" + "".join(digits[-4:])


def integer(raw, ctx=None):
    s = _digits(raw).strip().replace(",", "")
    if not re.fullmatch(r"-?[0-9]+", s):
        raise NormalizationError(f"not an integer: {raw!r}")
    return int(s)


def text(raw, ctx=None):
    """Collapse whitespace and drop trailing sentence punctuation."""
    s = re.sub(r"\s+", " ", raw).strip().rstrip(".,;:|")
    if not s:
        raise NormalizationError("empty text")
    return s


NORMALIZERS = {
    "amount": amount, "currency": currency, "date_dmy": date_dmy, "time_24h": time_24h,
    "account_reference": account_reference, "integer": integer, "text": text,
}

# field-name hints used for values that an LLM extracted (names are free-form there); matched on whole name parts
_AMOUNT_WORDS = {"amount", "balance", "fee", "charge", "cost", "price", "fare"}


def normalizer_for_name(field_name):
    parts = field_name.split("_")
    if "currency" in parts:
        return "currency"
    if parts[-1] == "date" or field_name == "date":
        return "date_dmy"
    if parts[-1] == "time" or field_name == "time":
        return "time_24h"
    if _AMOUNT_WORDS & set(parts):
        return "amount"
    return None


def apply(name, raw, ctx):
    if name not in NORMALIZERS:
        raise NormalizationError(f"unknown normalizer {name!r}")
    return NORMALIZERS[name](raw, ctx)


def check_type(value: Any, type_name: str):
    """Does the normalized value have the declared type? (raises NormalizationError)"""
    ok = {
        "string": isinstance(value, str), "enum": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "date": isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
        "time": isinstance(value, str) and re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", value) is not None,
    }.get(type_name, True)
    if not ok:
        raise NormalizationError(f"value {value!r} is not a valid {type_name}")
