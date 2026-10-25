"""extract_information.py - script 2: turn classified SMS into structured information. Extraction only.

Input : a DataFrame that already has the message text and its category (the output of script 1, classifier.py):
            message_decoded      the readable text
            Predicted_Category   the category
Output: the SAME DataFrame plus exactly two columns
            information     a JSON object  {"field": value, ...}  read from the message, or empty if nothing could be extracted
            regex_pattern   the regular expression that produced it, or empty if no regex applied

    import pandas as pd
    from sms_profiling.extract_information import RegexExtractor, extract_information

    df = pd.read_csv("output/classified.csv")                 # or a DataFrame fetched from a database
    result = extract_information(df, RegexExtractor.load("output"))
    result[["message_decoded", "information", "regex_pattern"]].head()

    python pipeline.py extract                                # output/classified.csv -> output/extracted.csv
    python pipeline.py extract --category "Utility Bills" --category "Education"

How a row is handled (deterministic, no LLM, no network):
  1. the category's template model maps the message to its template (the same model `pipeline.py templates` fitted);
  2. if the regex agent saved a regex for that template (output/regex_registry.json), it is applied to the message;
  3. the named groups that matched become the fields; values are cleaned and typed (amounts -> numbers,
     dates -> YYYY-MM-DD, times -> HH:MM); anything that cannot be typed is kept as the text that was read.
A row whose template has no regex yet (or whose regex does not match) is read by the GENERIC FALLBACK instead (generic_extractors.py): amounts,
dates, times, link domains and phone numbers that are clearly of their kind, with `regex_pattern` = "generic-fallback". Pass fallback=False /
--no-fallback to leave such rows empty. A row stays empty when even the fallback finds nothing - nothing is ever guessed. Profiling is NOT done here.

Privacy: secrets and login credentials (password / PIN / OTP / CVV / username ...) are never put in `information`; customer identifiers (account / card /
wallet / user / customer / subscriber / patient numbers and ids, UHID / MRN, lab numbers, phone numbers) are reduced to their last four
digits ("****6001") unless keep_identifiers=True. Order ids, merchants, amounts, dates and locations are kept as read.
Names (patient, doctor, merchant ...) are NOT masked: if the regex captures them they appear in `information`.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import regex

from . import OUTPUT_DIR, normalizers
from .completeness import SECRET_NAME
from .generic_extractors import GENERIC_PATTERN, extract_generic
from .templates import CAT_COL, TEXT_COL, TemplateModel, family_keys

INFO_COL, REGEX_COL = "information", "regex_pattern"
MATCH_TIMEOUT_S = 0.2

_SECRET = SECRET_NAME
# customer identifiers: account / card / wallet / user / customer / subscriber / patient numbers and ids, phone numbers,
# hospital patient ids (UHID / MRN) and national ids
_IDENTIFIER = re.compile(r"(account|card|wallet|user|customer|subscriber|member|client|patient).*(ref|reference|number|no|id)$"
                         r"|(^|_)(lab|registration|reg|admission)(_[a-z]+)*_(ref|reference|number|no|id)$"
                         r"|(^|_)(phone|mobile|msisdn|contact)(_number|_no)?$"
                         r"|^(id|uid|account|card|wallet|user|customer|subscriber|member|client|patient|lab|registration|reg|admission)$"
                         r"|(^|_)(uhid|mrn|nid|passport|national_id)(_number|_no)?$")


class RegexExtractor:
    """Finds a message's template and applies that template's regex."""

    def __init__(self, model: TemplateModel, registry: dict, keep_identifiers: bool = False):
        self.model, self.registry, self.keep_identifiers = model, registry, keep_identifiers
        self._compiled: dict[str, Optional[regex.Pattern]] = {}
        # A regex may belong to a FAMILY of templates (same sentence, one varying part such as the branch): find it by the member's
        # template id, or - for a template never seen before, e.g. tomorrow's new branch - by the family's wildcard key.
        self._member_of = {m: fid for fid, e in registry.items() for m in e.get("members", ())}
        self._by_key = {(e["category"], e["family_key"]): fid for fid, e in registry.items() if e.get("family_key")}
        self._family_cache: dict[tuple, Optional[str]] = {}
        # A family whose main regex missed some messages has second regexes ("parent" = the family): tried in order when the main one does not match.
        self._variants: dict[str, list[str]] = {}
        for rid, e in sorted(registry.items()):
            if e.get("parent"):
                self._variants.setdefault(e["parent"], []).append(rid)

    @classmethod
    def load(cls, output_dir=OUTPUT_DIR, keep_identifiers: bool = False) -> "RegexExtractor":
        out = Path(output_dir)
        for name in ("template_model.json", "regex_registry.json"):
            if not (out / name).exists():
                raise FileNotFoundError(f"{out / name} not found: run the `templates` stage (and the `regex` stage) of pipeline.py first")
        return cls(TemplateModel.load(out / "template_model.json"),
                   json.loads((out / "regex_registry.json").read_text(encoding="utf-8")), keep_identifiers)

    # ------------------------------------------------------------------ one message
    def _regex(self, tid: str) -> Optional[regex.Pattern]:
        if tid not in self._compiled:
            try:
                self._compiled[tid] = regex.compile(unicodedata.normalize("NFC", self.registry[tid]["expanded_pattern"]))
            except Exception:                                # a broken stored pattern disables only its own template
                self._compiled[tid] = None
        return self._compiled[tid]

    def clean(self, groups: dict) -> dict:
        """Matched groups -> typed, minimized values."""
        info: dict = {}
        for name, raw in groups.items():
            if raw is None or not str(raw).strip():
                continue
            if _SECRET.search(name):
                continue                                      # never output secrets, whatever the regex captured
            raw = str(raw).strip()
            try:
                if _IDENTIFIER.search(name) and not self.keep_identifiers:
                    value = self._mask(raw)                                                    # '****6001'
                elif (norm := normalizers.normalizer_for_name(name)) is not None:
                    value = normalizers.apply(norm, raw, normalizers.NormContext())
                else:
                    value = normalizers.text(raw)
            except normalizers.NormalizationError:
                if _IDENTIFIER.search(name) and not self.keep_identifiers:
                    continue                                  # cannot be minimized -> not output
                try:
                    value = normalizers.text(raw)             # could not be typed: keep what was read
                except normalizers.NormalizationError:
                    continue
            info[name] = value
        return info

    @staticmethod
    def _mask(raw: str) -> str:
        digits = re.findall(r"[0-9]", unicodedata.normalize("NFKC", raw))
        if len(digits) < 4:
            raise normalizers.NormalizationError("identifier with fewer than 4 digits")
        return "****" + "".join(digits[-4:])

    def family_of(self, tid: str, text: str, category: str) -> Optional[str]:
        """The registry key whose regex applies to a message with template id `tid`: the template itself, its family, or None."""
        if tid in self.registry:
            return tid
        if tid in self._member_of:
            return self._member_of[tid]
        if not self._by_key:
            return None
        ck = (category, tid)
        if ck not in self._family_cache:
            self._family_cache[ck] = next((self._by_key[(category, k)] for k in family_keys(self.model.transform(text, category))
                                           if (category, k) in self._by_key), None)
        return self._family_cache[ck]

    def fallback(self, text) -> Optional[dict]:
        """Generic information of a message (amounts, dates, times, link domains, phones), for a message no template regex could read."""
        return extract_generic(text, self.keep_identifiers)

    def extract_one(self, text, category) -> tuple[Optional[dict], Optional[str], str]:
        """(information, regex pattern, reason). information / pattern are None when nothing was extracted."""
        if not isinstance(text, str) or not text.strip():
            return None, None, "empty text"
        if not isinstance(category, str) or category not in self.model.vocab:
            return None, None, "category not modelled"
        raw_text = text
        text = unicodedata.normalize("NFC", text)
        tid = self.family_of(self.model.template_id(text, category), text, category)
        if tid is None and raw_text != text:                        # registries built from un-normalized text
            tid = self.family_of(self.model.template_id(raw_text, category), raw_text, category)
        entry = self.registry.get(tid) if tid else None
        if entry is None:
            return None, None, "template has no regex yet"
        reason = "regex did not match"
        for rid in [tid, *self._variants.get(tid, [])]:           # the main regex, then the second regexes for what it misses
            rx = self._regex(rid)
            if rx is None:
                if rid == tid:
                    reason = "stored regex is invalid"
                continue
            try:
                m = rx.search(text, timeout=MATCH_TIMEOUT_S)
            except TimeoutError:
                continue
            if not m:
                continue
            e = self.registry[rid]
            values = self.clean(m.groupdict())
            info = {"event": e["event"]} if e.get("event") and "event" not in values else {}
            info.update(values)
            for name, value in self.clean(e.get("facts") or {}).items():     # fixed wording worth keeping (organisation, branch, plan...)
                info.setdefault(name, value)
            if info:
                return info, e["expanded_pattern"], "extracted"
        return None, None, reason


def extract_information(df: pd.DataFrame, extractor: RegexExtractor, text_col: str = TEXT_COL, category_col: str = CAT_COL,
                        categories: Optional[list[str]] = None, info_col: str = INFO_COL, regex_col: str = REGEX_COL,
                        fallback: bool = True) -> pd.DataFrame:
    """Return a COPY of df with two new columns (information, regex_pattern). df itself is not modified.
    categories: only rows of these categories are processed; other rows keep empty values in the new columns.
    fallback: a row no template regex could read gets the generic information (amounts, dates, times, link domains, phones) and
    regex_pattern "generic-fallback"; False leaves it empty.
    Counts of what happened to each row are in result.attrs["extraction_stats"]."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    for c in (text_col, category_col):
        if c not in df.columns:
            raise KeyError(f"column {c!r} not in the DataFrame; columns are {list(df.columns)}")
    wanted = set(categories) if categories else None
    n = len(df)
    infos: list = [None] * n
    patterns: list = [None] * n
    stats: Counter = Counter()
    for i, (text, cat) in enumerate(zip(df[text_col].tolist(), df[category_col].tolist())):
        if wanted is not None and cat not in wanted:
            stats["skipped (other category)"] += 1
            continue
        info, pattern, reason = extractor.extract_one(text, cat)
        stats[reason] += 1
        if info is None and fallback:
            info = extractor.fallback(text)
            if info is not None:
                pattern = GENERIC_PATTERN
                stats["generic fallback (a subset of the rows above)"] += 1
        if info is not None:
            infos[i] = json.dumps(info, ensure_ascii=False)
            patterns[i] = pattern
    out = df.copy()
    out[info_col] = infos
    out[regex_col] = patterns
    out.attrs["extraction_stats"] = dict(stats)
    return out


# ======================================================================================
# command line
# ======================================================================================
def summary(result: pd.DataFrame, category_col: str = CAT_COL) -> str:
    got = result[INFO_COL].notna()
    lines = [f"rows: {len(result):,} | with information: {int(got.sum()):,} ({100 * got.mean():.1f}%)"]
    for k, v in sorted(result.attrs.get("extraction_stats", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"   {k:28} {v:,}")
    if got.any() and category_col in result.columns:
        lines.append("\nby category (rows / with information):")
        g = result.assign(_got=got).groupby(category_col)["_got"].agg(["size", "sum"]).sort_values("size", ascending=False)
        lines += [f"   {c:34} {int(r['size']):7,} {int(r['sum']):7,}" for c, r in g.iterrows()]
        fields = Counter(k for s in result.loc[got, INFO_COL] for k in json.loads(s))
        lines.append("\nmost common fields: " + ", ".join(f"{k} ({v})" for k, v in fields.most_common(12)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR), help="folder with template_model.json and regex_registry.json (default output/)")
    ap.add_argument("--classified", help="classified CSV (default <out-dir>/classified.csv)")
    ap.add_argument("--out", help="result CSV (default <out-dir>/extracted.csv)")
    ap.add_argument("--category", action="append", help="only this category (repeatable); default every category")
    ap.add_argument("--text-col", default=TEXT_COL)
    ap.add_argument("--category-col", default=CAT_COL)
    ap.add_argument("--keep-identifiers", action="store_true", help="keep full account / user / phone identifiers instead of '****1234'")
    ap.add_argument("--no-fallback", action="store_true", help="leave rows without a template regex empty instead of reading amounts, dates, times, links, phones")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    src = Path(a.classified) if a.classified else out_dir / "classified.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found: run the classify stage first")
    extractor = RegexExtractor.load(out_dir, a.keep_identifiers)
    df = pd.read_csv(src)
    result = extract_information(df, extractor, a.text_col, a.category_col, a.category, fallback=not a.no_fallback)
    dest = Path(a.out) if a.out else out_dir / "extracted.csv"
    result.to_csv(dest, index=False)
    print(summary(result, a.category_col))
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
