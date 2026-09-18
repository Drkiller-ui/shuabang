"""Sandbox-side LiveCodeBench worker. Uses only the Python standard library."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def parse_value(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return value


def equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
        except Exception:
            return False
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(equivalent(a, e) for a, e in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], expected[k]) for k in actual)
    return actual == expected


def normalized_stdout(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def stdout_equivalent(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    actual_tokens, expected_tokens = actual.split(), expected.split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    for left, right in zip(actual_tokens, expected_tokens):
        if left == right:
            continue
        try:
            if not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6):
                return False
        except ValueError:
            return False
    return True


def run_stdin(code_file: Path, case: dict[str, Any], timeout: float) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-I", str(code_file)], input=str(case["input"]), text=True,
        capture_output=True, timeout=timeout, cwd=code_file.parent,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
    )
    if completed.returncode != 0:
        return False, f"returncode={completed.returncode}; stderr={completed.stderr[-1200:]}"
    actual, expected = normalized_stdout(completed.stdout), normalized_stdout(str(case["output"]))
    return stdout_equivalent(actual, expected), f"expected={expected!r}; actual={actual!r}"


def load_namespace(code: str) -> dict[str, Any]:
    prelude = (
        "from typing import *\nfrom collections import *\nfrom functools import *\n"
        "from itertools import *\nfrom math import *\nfrom heapq import *\nfrom bisect import *\n"
    )
    # Future imports must remain at the beginning of the compiled module.
    future, body = [], []
    for line in code.splitlines():
        (future if line.lstrip().startswith("from __future__ import ") else body).append(line)
    combined = "\n".join(future) + "\n" + prelude + "\n".join(body)
    namespace: dict[str, Any] = {"__name__": "candidate"}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(combined, "candidate.py", "exec"), namespace, namespace)
    return namespace


@contextmanager
def time_limit(seconds: float):
    """Apply a per-case alarm inside the Linux sandbox; outer timeout remains the fallback."""
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        yield
        return

    def expired(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"timeout after {seconds}s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_functional(namespace: dict[str, Any], func_name: str, case: dict[str, Any],
                   timeout: float) -> tuple[bool, str]:
    args = [parse_value(line) for line in str(case["input"]).splitlines()]
    expected = parse_value(str(case["output"]))
    target = namespace.get("Solution")
    target = target() if isinstance(target, type) else namespace
    function = getattr(target, func_name) if not isinstance(target, dict) else target[func_name]
    with time_limit(timeout), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        actual = function(*args)
    return equivalent(actual, expected), f"expected={expected!r}; actual={actual!r}"


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    code = job["code"]
    tests = job["tests"]
    timeout = float(job.get("per_test_timeout", 3.0))
    code_file = Path(sys.argv[1]).with_name("candidate.py")
    code_file.write_text(code, encoding="utf-8")
    failures, passed = [], 0
    started = time.perf_counter()
    namespace = None
    for index, case in enumerate(tests):
        try:
            if case["testtype"] == "stdin":
                ok, detail = run_stdin(code_file, case, timeout)
            elif case["testtype"] == "functional":
                if namespace is None:
                    namespace = load_namespace(code)
                ok, detail = run_functional(namespace, job["func_name"], case, timeout)
            else:
                ok, detail = False, f"unsupported testtype={case.get('testtype')!r}"
        except subprocess.TimeoutExpired:
            ok, detail = False, f"timeout after {timeout}s"
        except BaseException as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if ok:
            passed += 1
        elif len(failures) < 10:
            failures.append({"index": index, "testtype": case.get("testtype"), "detail": detail})
    print(json.dumps({
        "ok": passed == len(tests), "passed_tests": passed, "total_tests": len(tests),
        "failures": failures, "elapsed_seconds": time.perf_counter() - started,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(json.dumps({"ok": False, "worker_error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(limit=8)}, ensure_ascii=False))
        raise SystemExit(1)
