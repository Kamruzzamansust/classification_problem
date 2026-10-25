"""classifier.py - classify SMS messages held in a pandas DataFrame, using a local LLM served by vLLM.

Typical use (the DataFrame can come from a CSV today and from a database later):

    import pandas as pd
    from sms_profiling.classifier import classify

    df = pd.read_csv("data/original_message_part_1.csv")      # or: df = pd.read_sql(query, connection)
    result = classify(df, text_col="msg_body")                  # same rows + prediction columns
    result[["msg_body", "Predicted_Category", "Predicted_Probability"]].head()

Added columns: Predicted_Category, Model_Choice, Predicted_Probability, System1_Confidence and one
Prob_<category> column per category (probabilities=False drops the Prob_ columns).

How it works (same method as python_v5.py): each message is put in a prompt that lists the categories
as single letters (A, B, C ...), the model answers with ONE token, and the probabilities come from that
token's log-probabilities. A message the model says is "Other", or whose best category has probability
below OTHER_THRESHOLD, is labelled "Other". Identical messages are classified once and the result is
copied to every duplicate. Everything runs on this server's GPU; no message text leaves the machine.

Run with the vLLM virtualenv:   ~/venvs/vllm/bin/python -m sms_profiling.classifier <csv> --text-col msg_body --limit 1000
The model is loaded on the first call (about a minute) and then reused, so call classify() repeatedly
from one process rather than starting a new process per batch.
"""
import os

# Server has no system CUDA toolkit (no nvcc): skip the FlashInfer sampler, whose warmup needs to compile a kernel.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import gc
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from ._compat import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "google/gemma-4-12B-it-qat-w4a16-ct"   # 4-bit build, ~8 GB of weights
OTHER_CATEGORY = "Other"
OTHER_THRESHOLD = 0.40        # best-category probability below this -> "Other"
CHUNK_SIZE = 2000             # unique messages per vLLM batch (a checkpoint is written after each chunk)
MAX_MODEL_LEN = 1024          # prompt (~350 tokens of instructions + message) must fit in this
GPU_MEM_UTIL = 0.60           # fraction of the 24 GB GPU vLLM reserves; lower it if another job uses the GPU
DEFAULT_TEXT_COL = "msg_body"

# Same 20 categories (and the same letters) as the SMS_Category_Dataset used by python_v5.py.
DEFAULT_CATEGORIES = [
    "E-commerce / Delivery", "Education", "Emergency / Disaster Alert", "Government Information",
    "Insurance", "Job / Recruitment", "Loan / EMI / Installment", "Medical / Healthcare",
    "Mobile Financial Services (MFS)", "OTP / Verification Code", "Offer / Discount", "Personal / P2P",
    "Promotional (Marketing)", "Religious / Seasonal", "Social Safety Net / Allowance",
    "Spam / Fraud / Scam", "Telecom / Operator Service", "Transactional (Banking)",
    "Travel & Ticketing", "Utility Bills",
]


# ==========================================
# 2. PROMPT / SCORING
# ==========================================
class CategoryPrompt:
    """Maps categories to single-letter keys and builds the classification prompt."""

    def __init__(self, categories):
        cats = [str(c).strip() for c in categories if str(c).strip().lower() != OTHER_CATEGORY.lower()]
        self.categories = cats + [OTHER_CATEGORY]
        if len(self.categories) > 52:
            raise ValueError("at most 51 categories (+ Other) are supported (single-letter keys)")
        self.key_to_cat = {}
        for idx, cat in enumerate(self.categories):
            key = chr(65 + idx) if idx < 26 else chr(97 + (idx - 26))
            self.key_to_cat[key] = cat
        self.other_key = list(self.key_to_cat)[-1]
        options_block = "\n".join(f"{k}: {cat}" for k, cat in self.key_to_cat.items())
        # Everything before the message is identical for every SMS -> vLLM prefix-caches it.
        self.prefix = (
            "You are a System 1 classification engine. Read the SMS message state "
            "and select the single best matching category key from the allowed options.\n"
            "Placeholders like [NUM], [AMOUNT], [DATE], [URL] are masked values.\n"
            f"If the message clearly does not fit any category, choose {self.other_key} ({OTHER_CATEGORY}).\n\n"
            f"Allowed Options:\n{options_block}\n\n"
        )

    def build(self, message):
        return (
            f"{self.prefix}"
            f"State (SMS Message): \"{message}\"\n\n"
            "Respond with ONLY the single option letter (e.g. A, B, C) and nothing else.\n"
            "Selected Option:"
        )


def score(prompt, raw_logprobs, other_threshold):
    """raw_logprobs: {option letter: log-probability of that letter as the first generated token}.
    Softmax over the options, then apply the 'Other' rules. Returns one result row (dict)."""
    raw = {k: v for k, v in raw_logprobs.items() if k in prompt.key_to_cat}
    if not raw:                                   # model produced no valid option at all -> Other
        raw = {prompt.other_key: 0.0}
    keys = list(prompt.key_to_cat)
    logits = np.array([raw.get(k, -20.0) for k in keys])   # floor for options outside the top-20 tokens
    p = np.exp(logits - logits.max())
    p /= p.sum()
    dist = {prompt.key_to_cat[k]: float(round(v, 4)) for k, v in zip(keys, p)}

    model_choice = max(dist, key=dist.get)
    max_prob = dist[model_choice]
    choice = OTHER_CATEGORY if (model_choice != OTHER_CATEGORY and max_prob < other_threshold) else model_choice
    entropy = -sum(v * math.log(v + 1e-12) for v in p)
    confidence = max(0.0, min(1.0, 1.0 - entropy / math.log(len(keys))))
    return {
        "Predicted_Category": choice,
        "Model_Choice": model_choice,
        "Predicted_Probability": max_prob,
        "System1_Confidence": round(confidence, 4),
        **{f"Prob_{c}": v for c, v in dist.items()},
    }


# ==========================================
# 3. THE CLASSIFIER (owns the vLLM engine)
# ==========================================
class SMSClassifier:
    def __init__(self, categories=None, model=MODEL_NAME, gpu_memory_utilization=GPU_MEM_UTIL,
                 max_model_len=MAX_MODEL_LEN, other_threshold=OTHER_THRESHOLD):
        self.prompt = CategoryPrompt(categories or DEFAULT_CATEGORIES)
        self.model = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.other_threshold = other_threshold
        self._llm = None
        self._tokenizer = None
        self._sampling = None

    # ---- engine ---------------------------------------------------------------------------
    def _load(self):
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams   # imported late so this module imports without vLLM/GPU
        self._llm = LLM(
            model=self.model,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 0, "audio": 0},   # text-only: skip multimodal memory
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)   # 1 token, top-20 logprobs

    def close(self):
        """Free the GPU memory (the engine otherwise stays loaded for the life of the process)."""
        if self._llm is not None:
            try:
                self._llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
        self._llm = self._tokenizer = self._sampling = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- prompting ------------------------------------------------------------------------
    def _build_ids(self, text, tokenizer=None):
        """Chat-template token ids for one message. Very long messages are shortened until the prompt
        fits max_model_len (one over-long message would otherwise make vLLM reject the whole batch)."""
        tok = tokenizer or self._tokenizer
        limit = self.max_model_len - 8            # room for the 1 generated token
        while True:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": self.prompt.build(text)}],
                add_generation_prompt=True, tokenize=True, enable_thinking=False)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
            if len(ids) <= limit or len(text) <= 20:
                return ids
            text = text[: int(len(text) * 0.8)]

    def _first_token_logprobs(self, texts):
        """For each text: ({option letter: logprob of first generated token}, generated text)."""
        self._load()
        outs = self._llm.generate([{"prompt_token_ids": self._build_ids(t)} for t in texts],
                                  self._sampling, use_tqdm=False)
        results = []
        for out in outs:
            c = out.outputs[0]
            first = c.logprobs[0] if c.logprobs else {}
            raw = {}
            for lp in first.values():
                letter = (lp.decoded_token or "").strip()
                if letter in self.prompt.key_to_cat and (letter not in raw or lp.logprob > raw[letter]):
                    raw[letter] = lp.logprob
            if not raw and c.text.strip() in self.prompt.key_to_cat:    # no logprobs returned: use the text
                raw[c.text.strip()] = 0.0
            results.append((raw, c.text))
        return results

    def predict(self, texts):
        """Classify a list of (already clean, non-empty) message strings. One result row per text."""
        rows = [score(self.prompt, raw, self.other_threshold) for raw, _ in self._first_token_logprobs(texts)]
        return pd.DataFrame(rows)


_default_classifier = None


def get_classifier():
    """The shared classifier: created on first use and reused by later classify() calls."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SMSClassifier()
    return _default_classifier


# ==========================================
# 4. THE FUNCTION YOU CALL
# ==========================================
def _key(text):
    """Checkpoint key: a hash, so no message text is ever written to the checkpoint file."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def classify(df, text_col=DEFAULT_TEXT_COL, *, classifier=None, probabilities=True,
             checkpoint_path=None, chunk_size=CHUNK_SIZE, show_progress=True):
    """Classify every message in df[text_col]. Returns a COPY of df with the prediction columns added
    (same rows, same order, same index); df itself is not modified.

    text_col         column holding the SMS text (raw or masked).
    classifier       an SMSClassifier to use; default is a shared one that is loaded once and reused.
    probabilities    False = omit the per-category Prob_* columns.
    checkpoint_path  optional CSV: unique-message results are saved after every chunk and reused, so an
                     interrupted run resumes where it stopped (only message hashes are stored there).
    Rows whose text is missing or blank get empty (None/NaN) predictions - they cannot be classified."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"classify() expects a pandas DataFrame, got {type(df).__name__}")
    if text_col not in df.columns:
        raise KeyError(f"column {text_col!r} not in the DataFrame; columns are {list(df.columns)}")
    clf = classifier or get_classifier()

    n = len(df)
    texts = df[text_col].tolist()
    pos, clean = [], []                      # positions of classifiable rows and their text
    for i, t in enumerate(texts):
        if isinstance(t, str) and t.strip():
            pos.append(i)
            clean.append(t.strip())
        elif pd.notna(t) and not isinstance(t, str) and str(t).strip():
            pos.append(i)
            clean.append(str(t).strip())
    keys = [_key(t) for t in clean]

    # one result per unique message
    unique_text = dict(zip(keys, clean))                 # key -> text (first occurrence)
    results = {}                                         # key -> result dict
    if checkpoint_path and Path(checkpoint_path).exists():
        cache = pd.read_csv(checkpoint_path)
        results = {r["_key"]: {k: v for k, v in r.items() if k != "_key"} for r in cache.to_dict("records")}
    todo = [k for k in unique_text if k not in results]

    if todo:
        with tqdm(total=len(todo), desc="Classifying SMS", unit="msg", disable=not show_progress) as bar:
            for start in range(0, len(todo), chunk_size):
                chunk = todo[start:start + chunk_size]
                preds = clf.predict([unique_text[k] for k in chunk])
                for k, row in zip(chunk, preds.to_dict("records")):
                    results[k] = row
                if checkpoint_path:
                    pd.DataFrame([{"_key": k, **v} for k, v in results.items()]).to_csv(checkpoint_path, index=False)
                bar.update(len(chunk))

    # map the unique results back onto every row (by position, so a non-unique df index is fine)
    prob_cols = [f"Prob_{c}" for c in clf.prompt.categories]
    out = df.copy()
    columns = ["Predicted_Category", "Model_Choice", "Predicted_Probability", "System1_Confidence"]
    columns += prob_cols if probabilities else []
    for col in columns:
        is_text = col in ("Predicted_Category", "Model_Choice")
        full = np.full(n, None, dtype=object) if is_text else np.full(n, np.nan)
        full[pos] = [results[k].get(col) for k in keys] if keys else []
        out[col] = full
    return out


# ==========================================
# 5. COMMAND LINE (CSV in -> CSV out)
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="Classify the SMS messages of a CSV file.")
    ap.add_argument("csv", help="input CSV")
    ap.add_argument("--text-col", default=DEFAULT_TEXT_COL, help="column with the SMS text (default %(default)s)")
    ap.add_argument("--out", help="output CSV (default: <input name>_classified.csv next to the input)")
    ap.add_argument("--limit", type=int, help="only the first N rows (testing)")
    ap.add_argument("--checkpoint", help="checkpoint CSV so an interrupted run can resume")
    ap.add_argument("--gpu-mem", type=float, default=GPU_MEM_UTIL)
    ap.add_argument("--threshold", type=float, default=OTHER_THRESHOLD)
    ap.add_argument("--no-probabilities", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df):,} rows from {args.csv}")
    with SMSClassifier(gpu_memory_utilization=args.gpu_mem, other_threshold=args.threshold) as clf:
        result = classify(df, args.text_col, classifier=clf, probabilities=not args.no_probabilities,
                          checkpoint_path=args.checkpoint)
    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_classified.csv")
    result.to_csv(out, index=False)
    print("\nCategory distribution:")
    print(result["Predicted_Category"].value_counts(dropna=False).to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
