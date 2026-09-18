"""Worker copied into the isolated Python-tool directory/container."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path("candidate.py").write_text(str(job["code"]), encoding="utf-8")
    timeout = max(0.1, min(60.0, float(job.get("timeout", 10.0))))
    timed_out = False
    with Path("stdout.txt").open("wb") as stdout_file, Path("stderr.txt").open("wb") as stderr_file:
        process = subprocess.Popen(
            [sys.executable, "-I", "candidate.py"], stdout=stdout_file, stderr=stderr_file,
            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows local debug mode
                process.kill()
            process.wait()
    stdout = Path("stdout.txt").read_bytes()[:12001].decode("utf-8", errors="replace")
    stderr = Path("stderr.txt").read_bytes()[:12001].decode("utf-8", errors="replace")
    generated = []
    for path in sorted(Path(".").glob("*.png"))[:3]:
        if path.is_file() and path.stat().st_size <= 20_000_000:
            generated.append(path.name)
    result = {
        "ok": process.returncode == 0 and not timed_out,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "generated_images": generated,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
