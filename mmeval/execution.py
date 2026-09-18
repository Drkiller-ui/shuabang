from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.I | re.S)
    if blocks:
        return blocks[-1].strip()
    cleaned = re.sub(r"^\s*(?:Here(?:'s| is).*?:|Solution:)\s*", "", text.strip(), flags=re.I)
    return cleaned


class SandboxExecutor:
    def __init__(self, mode: str, image: str, timeout: float, work_root: Path):
        self.mode, self.image, self.timeout = mode, image, timeout
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True)
        if mode in {"docker", "podman"}:
            probe = subprocess.run([mode, "image", "inspect", image], capture_output=True, text=True)
            if probe.returncode != 0:
                raise RuntimeError(
                    f"Container image {image!r} is unavailable. Run: "
                    f"{mode} build -t {image} -f docker/Dockerfile ."
                )
        elif mode != "local":
            raise ValueError("executor must be docker, podman, or local")

    def _run(self, worker: Path, job: dict[str, Any], name: str) -> tuple[dict[str, Any], Path]:
        directory = Path(tempfile.mkdtemp(prefix=re.sub(r"[^A-Za-z0-9_-]", "_", name) + "_", dir=self.work_root))
        shutil.copyfile(worker, directory / "worker.py")
        (directory / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        if self.mode in {"docker", "podman"}:
            command = [
                self.mode, "run", "--rm", "--network", "none", "--read-only",
                "--memory", "768m", "--cpus", "1", "--pids-limit", "64",
                "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
            ]
            if hasattr(os, "getuid"):
                command += ["--user", f"{os.getuid()}:{os.getgid()}"]
            command += [
                "-v", f"{directory.resolve()}:/work:rw", "-w", "/work",
                self.image, "python", "worker.py", "job.json",
            ]
        else:
            command = [sys.executable, str(directory / "worker.py"), str(directory / "job.json")]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            result = json.loads(lines[-1]) if lines else {"ok": False, "worker_error": "empty worker output"}
            result["returncode"] = completed.returncode
            if completed.stderr:
                result["stderr_tail"] = completed.stderr[-3000:]
            return result, directory
        except subprocess.TimeoutExpired:
            return {"ok": False, "worker_error": f"problem timeout after {self.timeout}s"}, directory
        except Exception as exc:
            return {"ok": False, "worker_error": f"{type(exc).__name__}: {exc}"}, directory

    def livecodebench(self, job: dict[str, Any], name: str) -> dict[str, Any]:
        worker = Path(__file__).with_name("lcb_worker.py")
        result, directory = self._run(worker, job, name)
        shutil.rmtree(directory, ignore_errors=True)
        return result
