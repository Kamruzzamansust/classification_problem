"""model_server.py - bring the vLLM server up (again) when it does not answer, so a long run does not stop when the server dies or the machine
restarts. Used by extract_categories(auto_restart=True); it only works when the code runs on the machine that hosts the server.

    ensure_server(url)                  # the name of the model that is served; starts the server first if nothing answers

What is started is the recipe below (Qwen3.8-27B in 4 bit, the settings that fit the 24 GB card, see the header of start_vllm.sh), through
start_vllm.sh. Set the environment variable VLLM_START_CMD to a shell command to start something else. A server that is up but serves a
DIFFERENT model than the one asked for is never touched.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

PROJECT = Path(__file__).resolve().parent.parent
NO_RESTART_ENV = "VLLM_NO_RESTART"          # set to anything: never start or stop a server (the tests set it)
START_SCRIPT = PROJECT / "start_vllm.sh"
QWEN38_27B = {"MODEL": "RedHatAI/Qwen3.8-27B-INT4", "SERVED": "qwen3.8-27b", "TOOL_PARSER": "qwen3_xml", "GPU_MEM": "0.95",
              "EXTRA_ARGS": "--language-model-only --max-num-seqs 4 --max-model-len 12288 --kv-cache-dtype bfloat16 --attention-backend FLASH_ATTN"}


def served_models(url: str, timeout: float = 5) -> Optional[list[str]]:
    """The models the server serves, or None when it does not answer."""
    try:
        return [m["id"] for m in json.load(urllib.request.urlopen(f"{url}/models", timeout=timeout))["data"]]
    except Exception:
        return None


def _pids(pattern: str) -> list[int]:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [p for p in map(int, r.stdout.split()) if p != os.getpid()] if r.returncode == 0 else []


QWEN35_9B = {"MODEL": "RedHatAI/Qwen3.5-9B-FP8-dynamic", "SERVED": "qwen3.5-9b", "TOOL_PARSER": "qwen3_coder", "GPU_MEM": "0.85",
             "EXTRA_ARGS": "--language-model-only"}
RECIPES = {QWEN38_27B["SERVED"]: QWEN38_27B, QWEN35_9B["SERVED"]: QWEN35_9B}


def stop_server(wait_s: int = 60, log=print) -> None:
    """Stop what is left of a dead or stuck server: the 'vllm serve' process and the orphaned EngineCore workers that keep the GPU memory.
    (The patterns have one letter in brackets so that this very command line does not match them.)"""
    patterns = ("[v]llm serve", "[E]ngineCore")
    for pattern in patterns:
        for pid in _pids(pattern):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for _ in range(max(1, wait_s // 2)):
        if not any(_pids(p) for p in patterns):
            return
        time.sleep(2)
    for pattern in patterns:
        for pid in _pids(pattern):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    log("warning: the old vLLM processes had to be killed")


def start_server(recipe: Optional[dict] = None, url: str = "http://localhost:8000/v1", wait_s: int = 600, log=print) -> str:
    """Start the server and wait until it answers. Returns the name of the model it serves."""
    recipe = recipe or QWEN38_27B
    env = {**os.environ, **recipe}
    cmd = os.environ.get("VLLM_START_CMD")
    argv = ["bash", "-c", cmd] if cmd else ["bash", str(START_SCRIPT)]
    subprocess.run(argv, env=env, cwd=str(PROJECT), check=True, stdout=subprocess.DEVNULL)      # start_vllm.sh puts vLLM in the background and returns
    served = recipe.get("SERVED")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(5)
        names = served_models(url)
        if names and (served is None or served in names):
            log(f"the vLLM server is up again ({time.time() - t0:.0f}s)")
            return served or names[0]
        if time.time() - t0 > 20 and not _pids("[v]llm serve"):
            tail = ""
            try:
                tail = " | ".join((PROJECT / "vllm_server.log").read_text(errors="replace").strip().splitlines()[-3:])[:400]
            except OSError:
                pass
            raise RuntimeError(f"the vLLM server stopped while starting; the end of vllm_server.log: {tail}")
    raise RuntimeError(f"the vLLM server did not answer within {wait_s} s after it was started; see {PROJECT / 'vllm_server.log'}")


def ensure_server(url: str, model: Optional[str] = None, recipe: Optional[dict] = None, wait_s: int = 600, log=print) -> str:
    """The name of a model the server serves. If nothing answers, the server is (re)started with the recipe first. With model=... exactly
    that model is required; a server that serves other models is left alone and an error says so."""
    names = served_models(url)
    if names:
        if model is None or model in names:
            return model or names[0]
        raise RuntimeError(f"the vLLM server serves {names}, not {model!r}: it is not touched (stop it yourself to start the other model)")
    if os.environ.get(NO_RESTART_ENV):
        raise RuntimeError(f"the vLLM server does not answer and starting it is switched off ({NO_RESTART_ENV} is set)")
    log("the vLLM server does not answer: clearing what is left of it and starting it again (about 2 minutes) ...")
    stop_server(log=log)
    return start_server(recipe or RECIPES.get(model or "") or QWEN38_27B, url, wait_s, log)
