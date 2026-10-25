"""masking.py - turn a message into its MASKED TEXT (a template): the parts that change from message to message
(amounts, dates, times, numbers, phone numbers, ids, links, e-mails, people's names) are replaced by placeholders, the
fixed wording stays. Messages that differ only in those parts get the SAME masked text.

    "TK 18,500.00 withdrawn from A/C#10000**6001 on 06-06-26 @08:34 PM from SERPUR ATM. Balance TK 2,949.31."
 -> "[AMOUNT] withdrawn from A/C#[NUM]**[NUM] on [DATE] @[TIME] from SERPUR ATM. Balance [AMOUNT]."

Why: classification only needs the wording. Classifying each DISTINCT masked text once and giving that category to every
message that has it is far cheaper (the 300,000-message file has about 277,000 distinct raw texts but only about 45,000
distinct masked texts) and it is also privacy-friendly: the model never sees the real numbers or names.

The placeholders are the same ones the earlier masked dataset used ([NUM] [AMOUNT] [DATE] [TIME] [URL] [PHONE] [EMAIL]
[REF] [NAME]); the classifier prompt already tells the model what they mean.
"""
import re
import unicodedata

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_CUR = r"(?:tk|bdt|rs|usd|inr|taka|৳|\$|টাকা)"
_NUMBER = r"\d[\d,]*(?:\.\d+)?"

# order matters: most specific first, plain numbers last
_RULES = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[URL]", re.compile(r"(?:https?://|www\.)[^\s<>\"']+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|app|bd|co|io|ly|me|info|xyz)(?:/[^\s<>\"']*)?", re.I)),
    ("[DATE]", re.compile(rf"\b\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}\b|\b\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}\b"
                          rf"|\b\d{{1,2}}(?:st|nd|rd|th)?[-\s/]*{_MONTHS}\b\.?,?[-\s/]*\d{{2,4}}\b"
                          rf"|\b{_MONTHS}\b\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}\b", re.I)),
    ("[TIME]", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?(?!\d)", re.I)),
    ("[AMOUNT]", re.compile(rf"{_CUR}\.?\s*[-:]?\s*{_NUMBER}(?:\s*/-)?|{_NUMBER}\s*(?:/-|{_CUR}\b(?![\s.:-]*\d))|(?<![\d,.])\d{{1,3}}(?:,\d{{2,3}})+(?:\.\d+)?(?![\d,])|(?<![\d,.])\d+\.\d{{1,2}}(?![\d.])", re.I)),
    ("[PHONE]", re.compile(r"(?<!\d)(?:\+?88[-\s]?)?0?1[3-9][-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{3}(?!\d)")),
    ("[REF]", re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{5,}\b")),
    ("[NUM]", re.compile(r"\d+")),
]
_PLACEHOLDER_SPLIT = re.compile(r"(\[[A-Z]+\])")           # never re-mask inside an existing placeholder

# people's names: after a salutation / title, up to 4 capitalised words
_NOT_NAMES = {"sir", "madam", "madame", "customer", "client", "valued", "member", "user", "subscriber", "doctor", "student", "parent",
              "guardian", "all", "team", "sir/madam", "sir/ma'am", "agent", "merchant", "partner", "friend", "applicant", "candidate", "patient",
              "passenger", "sirs", "ma'am", "colleague", "teacher", "gentleman"}
_STOP = {"welcome", "your", "thank", "thanks", "please", "greetings", "congratulations", "reports", "report", "hope", "we", "you", "i", "kindly",
         "regards", "sorry", "good", "is", "are", "has", "have", "this", "the", "a", "an", "to", "for", "from", "on", "at", "in", "and", "with", "as"}
_NAME = re.compile(r"\b((?i:dear|mr|mrs|ms|miss|dr|prof|hello|hi)\b\.?\s*(?:(?i:md|mohammad|mohammed|mst|mrs|mr|ms|dr|prof)\b\.?\s*)*)"
                   r"((?:[A-Z][A-Za-z.'-]*\s+){0,3}[A-Z][A-Za-z.'-]*)")


_TITLES = {"dear", "mr", "mrs", "ms", "miss", "dr", "prof", "md", "mohammad", "mohammed", "mst", "hello", "hi"}


def _name_sub(m):
    words = re.split(r"\s+", m.group(2).strip())
    prefix = m.group(1).rstrip()
    while words and words[0].lower().strip(".,;:") in _TITLES:       # a title is not a name ("Dear Mr. [NAME]" stays as it is)
        prefix += " " + words.pop(0)
    keep = []
    for w in words:                                                  # the name ends at the first ordinary sentence word
        lw = w.lower().strip(".,;:")
        if lw in _STOP or (lw in _NOT_NAMES and not keep):
            break
        keep.append(w)
    if not keep:
        return m.group(0)
    tail = " ".join(words[len(keep):])
    return prefix + " [NAME]" + (" " + tail if tail else "")


def mask_text(text) -> str:
    """The masked text of one message ('' for missing / blank input)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    s = unicodedata.normalize("NFC", text).translate(_BN_DIGITS)
    s = re.sub(r"[ \t\r\n ​]+", " ", s).strip()
    for token, rx in _RULES:
        parts = _PLACEHOLDER_SPLIT.split(s)
        s = "".join(p if _PLACEHOLDER_SPLIT.fullmatch(p) else rx.sub(token, p) for p in parts)
    s = _NAME.sub(_name_sub, s)
    return re.sub(r"\s+", " ", s).strip()
