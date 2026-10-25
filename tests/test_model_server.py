"""Offline tests of model_server.py: the vLLM server is started (again) when it does not answer. No real server, process or GPU is touched."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import signal
from sms_profiling import model_server as ms

URL = "http://localhost:8000/v1"
saved_env = os.environ.pop(ms.NO_RESTART_ENV, None)


def patched(**kw):
    """Replace module functions for one test and restore them afterwards."""
    class Ctx:
        def __enter__(self_):
            self_.old = {k: getattr(ms, k) for k in kw}
            for k, v in kw.items():
                setattr(ms, k, v)
        def __exit__(self_, *a):
            for k, v in self_.old.items():
                setattr(ms, k, v)
    return Ctx()


def test_a_server_that_answers_is_left_alone():
    with patched(served_models=lambda url, timeout=5: ["qwen3.8-27b"], stop_server=lambda **k: 1 / 0, start_server=lambda *a, **k: 1 / 0):
        assert ms.ensure_server(URL) == "qwen3.8-27b" and ms.ensure_server(URL, "qwen3.8-27b") == "qwen3.8-27b"
    with patched(served_models=lambda url, timeout=5: ["qwen3.5-9b"], stop_server=lambda **k: 1 / 0, start_server=lambda *a, **k: 1 / 0):
        try:
            ms.ensure_server(URL, "qwen3.8-27b"); raise AssertionError("no error")
        except RuntimeError as e:
            assert "serves ['qwen3.5-9b']" in str(e) and "not touched" in str(e), "a server with ANOTHER model is never stopped"
    print("OK  a server that answers is left alone, also when it serves another model (clear error, never stopped)")


def test_a_dead_server_is_cleared_and_started_with_the_recipe():
    events = []
    with patched(served_models=lambda url, timeout=5: None, stop_server=lambda **k: events.append("stop"),
                 start_server=lambda recipe, url, wait_s, log: (events.append(("start", recipe["SERVED"])), recipe["SERVED"])[1]):
        assert ms.ensure_server(URL, "qwen3.8-27b") == "qwen3.8-27b" and events == ["stop", ("start", "qwen3.8-27b")]
        events.clear()
        assert ms.ensure_server(URL, "qwen3.5-9b") == "qwen3.5-9b" and events[1] == ("start", "qwen3.5-9b"), "the recipe of the model that was asked for"
        events.clear()
        assert ms.ensure_server(URL) == "qwen3.8-27b", "no model named: the 27B recipe"
        custom = {"SERVED": "mine", "MODEL": "x"}
        events.clear()
        assert ms.ensure_server(URL, "mine", recipe=custom) == "mine"
    os.environ[ms.NO_RESTART_ENV] = "1"
    try:
        with patched(served_models=lambda url, timeout=5: None, stop_server=lambda **k: 1 / 0, start_server=lambda *a, **k: 1 / 0):
            try:
                ms.ensure_server(URL); raise AssertionError("no error")
            except RuntimeError as e:
                assert ms.NO_RESTART_ENV in str(e)
    finally:
        os.environ.pop(ms.NO_RESTART_ENV)
    print("OK  a dead server is cleared and started with the recipe of the requested model; VLLM_NO_RESTART switches it off")


def test_start_waits_until_the_model_answers_and_reports_failures():
    ran, clock = [], {"t": 0.0}
    answers = iter([None, None, ["qwen3.8-27b"]])
    fake_run = lambda argv, **kw: ran.append((argv, kw.get("env", {}).get("MODEL"), kw.get("cwd"))) or None
    import subprocess, time
    real_run, real_sleep, real_time = subprocess.run, time.sleep, time.time
    subprocess.run = fake_run
    time.sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
    time.time = lambda: clock["t"]
    try:
        with patched(served_models=lambda url, timeout=5: next(answers), _pids=lambda p: [123]):
            assert ms.start_server(None, URL, wait_s=100, log=lambda m: None) == "qwen3.8-27b"
        assert ran[0][0][0] == "bash" and ran[0][0][1].endswith("start_vllm.sh") and ran[0][1] == "RedHatAI/Qwen3.8-27B-INT4" and ran[0][2] == str(ms.PROJECT)
        clock["t"] = 0.0
        with patched(served_models=lambda url, timeout=5: None, _pids=lambda p: []):          # the process dies while starting
            try:
                ms.start_server(None, URL, wait_s=100, log=lambda m: None); raise AssertionError("no error")
            except RuntimeError as e:
                assert "stopped while starting" in str(e)
        clock["t"] = 0.0
        with patched(served_models=lambda url, timeout=5: None, _pids=lambda p: [1]):          # alive but never answers
            try:
                ms.start_server(None, URL, wait_s=30, log=lambda m: None); raise AssertionError("no error")
            except RuntimeError as e:
                assert "did not answer within 30 s" in str(e)
        os.environ["VLLM_START_CMD"] = "echo custom"
        with patched(served_models=lambda url, timeout=5: ["qwen3.8-27b"], _pids=lambda p: [1]):
            ms.start_server(None, URL, wait_s=10, log=lambda m: None)
        assert ran[-1][0] == ["bash", "-c", "echo custom"], "VLLM_START_CMD replaces start_vllm.sh"
    finally:
        subprocess.run, time.sleep, time.time = real_run, real_sleep, real_time
        os.environ.pop("VLLM_START_CMD", None)
    print("OK  start: runs start_vllm.sh with the recipe, waits until the model answers; a dying or silent server raises a clear error; VLLM_START_CMD overrides")


def test_stop_clears_what_is_left_and_never_kills_itself():
    killed, alive = [], {"vllm": [10, os.getpid()], "engine": [11, 12]}
    def fake_pids(pattern):
        key = "vllm" if "llm serve" in pattern else "engine"
        return [p for p in alive[key] if p != os.getpid()]            # the real _pids already excludes the current process
    def fake_kill(pid, sig):
        killed.append((pid, sig))
        for k in alive:
            if pid in alive[k]:
                alive[k].remove(pid)
    import time
    real_kill, real_sleep = os.kill, time.sleep
    os.kill, time.sleep = fake_kill, lambda s: None
    try:
        with patched(_pids=fake_pids):
            ms.stop_server(wait_s=10, log=lambda m: None)
    finally:
        os.kill, time.sleep = real_kill, real_sleep
    assert sorted(p for p, sig in killed) == [10, 11, 12] and all(sig == signal.SIGTERM for _, sig in killed) and os.getpid() not in [p for p, _ in killed]
    assert ms._pids("[n]o-such-process-xyz-123") == [] and os.getpid() not in ms._pids("python")
    print("OK  stop: ends the vllm process and the orphaned EngineCore workers (TERM first), never the current process")


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        fn()
    if saved_env is not None:
        os.environ[ms.NO_RESTART_ENV] = saved_env
    print("\nALL MODEL SERVER TESTS PASSED")
