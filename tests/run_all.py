"""Run every test file (each is a plain script that exits non-zero on failure).

    ~/venvs/regex_writer/bin/python tests/run_all.py            # all
    ~/venvs/regex_writer/bin/python tests/run_all.py extract    # only files whose name contains "extract"

No GPU, no LLM and no network are used: all tests run offline in a few seconds.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
want = sys.argv[1:]
files = sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py") and (not want or any(w in f for w in want)))
failed, skipped = [], []
for f in files:
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, f)], capture_output=True, text=True, env={**os.environ, "VLLM_NO_RESTART": "1"})
    if r.returncode == 77:                                  # a test that cannot run in this environment says so
        skipped.append(f)
        print(f"  skip  {f:34} {r.stdout.strip().splitlines()[-1][:90]}")
        continue
    ok = r.returncode == 0
    print(f"  {'ok  ' if ok else 'FAIL'}  {f:34} {time.time() - t0:5.1f}s")
    if not ok:
        failed.append(f)
        print("".join("        " + l for l in (r.stdout + r.stderr).splitlines(True)[-12:]))
print(f"\n{len(files) - len(failed) - len(skipped)} of {len(files)} test files passed" + (f", {len(skipped)} skipped" if skipped else ""))
sys.exit(1 if failed else 0)
