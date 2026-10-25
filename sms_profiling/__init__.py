"""sms_profiling - SMS classification, template building, regex writing and information extraction.

    from sms_profiling.classifier import classify                       # script 1: add the category
    from sms_profiling.preprocess import prepare_messages               #   (decode hex messages first)
    from sms_profiling.extract_information import RegexExtractor, extract_information   # script 2: add information + regex_pattern

Everything is run through `python pipeline.py <stage>` in the project root (see README.md); each module can also be run
with `python -m sms_profiling.<module>`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # the project folder
OUTPUT_DIR = ROOT / "output"                     # every generated file goes here
