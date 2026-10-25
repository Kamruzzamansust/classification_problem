"""preprocess.py - prepare SMS text in a DataFrame: decode hex-encoded messages, flag the language, and
(optionally) translate Bangla to English.

Default workflow - decode only, nothing leaves the server:

    import pandas as pd
    from sms_profiling.preprocess import prepare_messages
    from sms_profiling.classifier import classify

    df = pd.read_csv("data/original_message_part_1.csv")        # or a DataFrame read from a database
    df = prepare_messages(df, text_col="msg_body")               # 1) decode hex, flag the language
    result = classify(df, text_col="message_masked")             # 2) classify the masked text once per distinct template

Columns added (your original text column is left untouched):
    message_decoded          the readable text: hex-encoded messages decoded, Unicode-normalized (NFC), everything else unchanged
    original_language_flag   "bangla" if the message contains any Bangla letter, else "english"
    message_masked           the message with amounts, dates, times, numbers, ids, links and names replaced by placeholders
                             (masking.py). Messages with the same masked text are the same kind of message, so classification
                             runs once per distinct masked text (about 35,000 instead of 300,000) and every message gets its category.

Why no translation by default: the models used here read Bengali, and a check on 40,376 Bangla messages
classified directly (no translation) showed confidence close to English. Translating would also send
customer SMS to Google and break any regex written against the original Bangla wording.

Hex decoding: telecom systems send non-Latin SMS as hex-encoded UTF-16BE text (e.g. "005B0050..."). A value
that is entirely hex and decodes to plausible SMS characters is decoded; anything else is left alone.
Language flag: any character in the Bangla Unicode block -> "bangla" (mixed Bangla + English counts as
Bangla); everything else -> "english".

OPTIONAL translation: prepare_messages(df, translate=True) (or add_english_translation) also adds
    message_english          English text: the translation for Bangla rows, the decoded text otherwise
    translation_status       "translated" | "not_needed" | "failed"  (empty for blank messages)
using Google Translate through deep-translator (source "bn"); each distinct text is sent once and results
can be checkpointed. PRIVACY: that sends text to Google's servers, so by default (mask=True) long digit runs,
links, e-mail addresses and reference codes are replaced by [NUM] / [URL] / [EMAIL] / [CODE] BEFORE sending,
and only Bangla rows are sent. Google rate-limits servers (HTTP 429); requests are throttled, retried with
long waits, and stopped early with a warning if the service keeps refusing.
"""
import hashlib
import re
import threading
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from ._compat import tqdm

BANGLA_RE = re.compile(r"[ঀ-৿]")
_HEX_RE = re.compile(r"(?:[0-9A-Fa-f]{4})+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CODE_RE = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{5,}\b")   # PNR / txn / ref codes
_DIGITS_RE = re.compile(r"[0-9০-৯]{4,}")                                       # Latin + Bangla digits

LANG_BANGLA, LANG_ENGLISH = "bangla", "english"
MAX_CHARS = 4500              # Google Translate accepts at most 5000 characters per request
REQUESTS_PER_SECOND = 2.0     # Google allows ~5/s per IP; stay well under it, shared by all workers
RATE_LIMIT_WAIT = 15.0        # seconds to wait after a "too many requests" answer (grows: 15, 30, 45 ...)
GIVE_UP_AFTER = 3             # stop sending after this many texts in a row were refused by rate limiting


# ==========================================
# 1. TEXT HELPERS
# ==========================================
# Characters a real SMS can contain. A hex string that is NOT UTF-16 text (an account number such as
# 12345678, say) "decodes" into random code points from unrelated scripts, so it fails this check.
_PLAUSIBLE_RE = re.compile(
    "^[\\t\\n\\r\\x20-\\x7e\\xa0-\\u024f\\u0600-\\u06ff\\u0900-\\u097f\\u0980-\\u09ff"
    "\\u2000-\\u206f\\u20a0-\\u20cf\\u2190-\\u21ff\\u2600-\\u27bf\\ufe0f\\U0001f000-\\U0001faff]+$")


def decode_hex_sms(text):
    """Hex-encoded UTF-16BE -> text. Anything that is not such a payload is returned unchanged."""
    s = text.strip()
    if not _HEX_RE.fullmatch(s):
        return text
    try:
        decoded = bytes.fromhex(s).decode("utf-16-be")
    except (ValueError, UnicodeDecodeError):
        return text
    return decoded if _PLAUSIBLE_RE.match(decoded) and any(c.isalpha() for c in decoded) else text


def language_flag(text):
    return LANG_BANGLA if BANGLA_RE.search(text) else LANG_ENGLISH


def mask_sensitive(text):
    """Replace what could identify a customer or an account before the text goes to an outside service."""
    text = _URL_RE.sub("[URL]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _CODE_RE.sub("[CODE]", text)
    return _DIGITS_RE.sub("[NUM]", text)


def _key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ==========================================
# 2. THE TRANSLATOR (Google, through deep-translator)
# ==========================================
class RateLimited(Exception):
    """The translation service keeps answering 'too many requests'."""


def _is_rate_limit(e):
    return type(e).__name__ == "TooManyRequests" or "too many requests" in str(e).lower() or "429" in str(e)


class _Throttle:
    """At most `rate` request starts per second across all threads."""
    def __init__(self, rate):
        self.interval, self.next, self.lock = 1.0 / rate, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next - now)
            self.next = max(now, self.next) + self.interval
        if delay:
            time.sleep(delay)


_local = threading.local()
_throttle = _Throttle(REQUESTS_PER_SECOND)


def google_bn_to_en(text):
    """Translate one Bangla text to English with Google Translate (one HTTP request)."""
    from deep_translator import GoogleTranslator   # imported late: only needed for real translation
    if not hasattr(_local, "t"):
        _local.t = GoogleTranslator(source="bn", target="en")   # source is known: the text has Bangla in it
    _throttle.wait()
    return _local.t.translate(text[:MAX_CHARS])


def _translate_with_retry(fn, text, retries=3, wait=2.0):
    """Retry a failing request. Rate-limit answers wait much longer (and raise RateLimited at the end)."""
    last = None
    for attempt in range(retries):
        try:
            out = fn(text)
            if isinstance(out, str) and out.strip():
                return out
            last = ValueError("empty translation")
        except Exception as e:                      # rate limit, network error, bad response ...
            last = e
        time.sleep(RATE_LIMIT_WAIT * (attempt + 1) if _is_rate_limit(last) else wait * (2 ** attempt))
    if _is_rate_limit(last):
        raise RateLimited(str(last)[:200])
    raise last


# ==========================================
# 3. THE FUNCTIONS YOU CALL
# ==========================================
def _check_df(df, text_col):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    if text_col not in df.columns:
        raise KeyError(f"column {text_col!r} not in the DataFrame; columns are {list(df.columns)}")


def decode_messages(df, text_col="msg_body", *, decoded_col="message_decoded", flag_col="original_language_flag"):
    """Return a COPY of df with message_decoded and original_language_flag added. No network, no model."""
    _check_df(df, text_col)
    n = len(df)
    decoded_out = np.full(n, None, dtype=object)
    flags = np.full(n, None, dtype=object)
    for i, t in enumerate(df[text_col].tolist()):
        if t is None or (not isinstance(t, str) and pd.isna(t)) or not str(t).strip():
            continue                                      # blank / missing: stays empty
        decoded = unicodedata.normalize("NFC", decode_hex_sms(str(t))).strip()      # one encoding per letter (Bengali য় has two)
        decoded_out[i], flags[i] = decoded, language_flag(decoded)
    out = df.copy()
    out[decoded_col], out[flag_col] = decoded_out, flags
    return out


def add_masked_text(df, decoded_col="message_decoded", masked_col="message_masked"):
    """Return a COPY of df with the MASKED TEXT of every message (see masking.py): amounts, dates, numbers, ids, links and
    names replaced by placeholders. Each distinct text is masked once. Blank messages get an empty masked text."""
    from .masking import mask_text
    out = df.copy()
    distinct = out[decoded_col].dropna().drop_duplicates()
    lookup = dict(zip(distinct, distinct.map(mask_text)))
    out[masked_col] = out[decoded_col].map(lookup).where(out[decoded_col].notna())
    return out


def prepare_messages(df, text_col="msg_body", *, translate=False, mask=True, **translate_kwargs):
    """The preparation for classification, in this order:
         1. decode hex-encoded messages -> message_decoded, and flag the language -> original_language_flag
         2. create the masked text (the template of the message)               -> message_masked   (mask=False skips it)
    translate=True additionally translates the Bangla rows to English (see add_english_translation for translate_kwargs)."""
    out = decode_messages(df, text_col)
    if mask:
        out = add_masked_text(out)
    if translate:
        out = add_english_translation(out, text_col, **translate_kwargs)
    return out


# ==========================================
# 3b. OPTIONAL: TRANSLATION
# ==========================================
def add_english_translation(df, text_col="msg_body", *, mask=True, translator=None, checkpoint_path=None,
                            max_workers=2, show_progress=True, english_col="message_english",
                            flag_col="original_language_flag", status_col="translation_status"):
    """Return a COPY of df with original_language_flag, message_english and translation_status added.

    mask             True (default): mask numbers / links / e-mails / codes before sending text to Google.
    translator       function text -> English text; default is Google through deep-translator.
    checkpoint_path  optional CSV of finished translations so an interrupted run resumes (it holds the
                     MASKED text and its translation - not the original messages - when mask=True).
    max_workers      parallel requests to the translation service (default 2; all workers together are
                     limited to REQUESTS_PER_SECOND, because Google blocks an IP that sends too many).
    A message that cannot be translated keeps its Bangla text in message_english, status "failed"."""
    _check_df(df, text_col)
    fn = translator or google_bn_to_en

    n = len(df)
    flags = np.full(n, None, dtype=object)
    english = np.full(n, None, dtype=object)
    status = np.full(n, None, dtype=object)

    # decode + flag every row; collect the Bangla ones
    bangla_rows = []                       # (row position, decoded text)
    for i, t in enumerate(df[text_col].tolist()):
        if t is None or (not isinstance(t, str) and pd.isna(t)) or not str(t).strip():
            continue
        decoded = decode_hex_sms(str(t)).strip()
        flags[i] = language_flag(decoded)
        if flags[i] == LANG_BANGLA:
            bangla_rows.append((i, decoded))
        else:
            english[i], status[i] = decoded, "not_needed"

    # one translation request per distinct (masked) text
    to_send = {}                           # key -> text that will be sent
    row_key = {}                           # row position -> key
    for i, decoded in bangla_rows:
        sent = mask_sensitive(decoded) if mask else decoded
        k = _key(sent)
        to_send[k] = sent
        row_key[i] = k

    done = {}                              # key -> translation
    if checkpoint_path and Path(checkpoint_path).exists():
        cache = pd.read_csv(checkpoint_path, dtype=str).dropna()
        done = dict(zip(cache["_key"], cache["english"]))
    todo = [k for k in to_send if k not in done]
    failed = set()
    breaker = {"refused": 0}                # consecutive texts refused by rate limiting
    breaker_lock = threading.Lock()

    def attempt(text):
        if breaker["refused"] >= GIVE_UP_AFTER:           # service keeps refusing: stop sending
            raise RateLimited("gave up after repeated rate limiting")
        try:
            out = _translate_with_retry(fn, text)
        except RateLimited:
            with breaker_lock:
                breaker["refused"] += 1
            raise
        with breaker_lock:
            breaker["refused"] = 0
        return out

    def save():
        if checkpoint_path:
            pd.DataFrame({"_key": list(done), "english": list(done.values())}).to_csv(checkpoint_path, index=False)

    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as pool, \
                tqdm(total=len(todo), desc="Translating", unit="msg", disable=not show_progress) as bar:
            futs = {pool.submit(attempt, to_send[k]): k for k in todo}
            since_save = 0
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    done[k] = fut.result()
                except Exception:
                    failed.add(k)
                bar.update(1)
                since_save += 1
                if since_save >= 200:
                    save()
                    since_save = 0
        save()
        if breaker["refused"] >= GIVE_UP_AFTER:
            warnings.warn("The translation service is rate limiting this machine (HTTP 429), so translation was "
                          "stopped early. Untranslated rows have translation_status 'failed'; run again later "
                          "(finished translations are kept in the checkpoint) or lower max_workers.", RuntimeWarning)

    for i, decoded in bangla_rows:
        k = row_key[i]
        if k in done:
            english[i], status[i] = done[k], "translated"
        else:
            english[i], status[i] = decoded, "failed"      # keep the Bangla so nothing is lost

    out = df.copy()
    out[flag_col], out[english_col], out[status_col] = flags, english, status
    return out
