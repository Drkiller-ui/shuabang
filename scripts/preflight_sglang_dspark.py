#!/usr/bin/env python3
"""Validate a single-GPU SGLang + Qwen3.5-4B DSpark environment.

Only small metadata files are downloaded.  Model weights are not downloaded.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def nested_dicts(value: dict[str, Any]):
    yield value
    for key in ("dspark_config", "text_config", "draft_config"):
        child = value.get(key)
        if isinstance(child, dict):
            yield child


def first_value(config: dict[str, Any], keys: tuple[str, ...]):
    for scope in nested_dicts(config):
        for key in keys:
            if key in scope:
                return scope[key]
    return None


def target_num_layers(config: dict[str, Any]) -> int | None:
    value = first_value(config, ("num_hidden_layers",))
    return int(value) if value is not None else None


def draft_layer_ids(config: dict[str, Any]) -> list[int]:
    value = first_value(
        config,
        ("target_layer_ids", "dspark_target_layer_ids", "dflash_target_layer_ids"),
    )
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def draft_gamma(config: dict[str, Any]) -> int | None:
    value = first_value(config, ("dspark_block_size", "block_size", "gamma"))
    return int(value) if value is not None else None


def get_config(
    repo_id: str, revision: str, cache_dir: str | None
) -> tuple[dict[str, Any], str, str]:
    from huggingface_hub import HfApi, hf_hub_download

    resolved_revision = HfApi().model_info(repo_id, revision=revision).sha
    if not resolved_revision:
        raise RuntimeError(f"Could not resolve a commit SHA for {repo_id}@{revision}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=resolved_revision,
        cache_dir=cache_dir,
    )
    return json.loads(Path(path).read_text(encoding="utf-8")), path, resolved_revision


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def has_sglang_patch() -> tuple[bool, str | None]:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        return False, None
    path = Path(spec.origin).resolve().parent / "srt" / "models" / "qwen3_5.py"
    if not path.is_file():
        return False, str(path)
    source = path.read_text(encoding="utf-8")
    return (
        "_capture_after_last_layer" in source
        and "HF layer num_layers - 1 is captured after the decoder loop" in source,
        str(path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument(
        "--draft", default="shanjiaz/qwen3_5_4b_perfectblend_regen_dspark"
    )
    parser.add_argument("--draft-revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--report", default="environment-dspark-single.json")
    parser.add_argument("--revision-lock", default=None)
    parser.add_argument("--recommended-vram-gib", type=float, default=40.0)
    args = parser.parse_args()

    failures: list[str] = []
    notes: list[str] = []
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "draft_model": args.draft,
        "draft_revision_requested": args.draft_revision,
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ("sglang", "torch", "transformers", "huggingface-hub")
        },
    }

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_rows = []
        for index in range(torch.cuda.device_count() if cuda_available else 0):
            props = torch.cuda.get_device_properties(index)
            gpu_rows.append(
                {
                    "index": index,
                    "name": props.name,
                    "vram_gib": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        report["cuda"] = {
            "available": cuda_available,
            "torch_runtime": torch.version.cuda,
            "gpu_count": len(gpu_rows),
            "gpus": gpu_rows,
        }
        if not cuda_available or not gpu_rows:
            failures.append("PyTorch cannot access an NVIDIA GPU")
        elif gpu_rows[0]["vram_gib"] < args.recommended_vram_gib:
            notes.append(
                f"GPU 0 has {gpu_rows[0]['vram_gib']} GiB VRAM; use this host for "
                f"environment validation only. Full target+draft serving is sized for "
                f"at least {args.recommended_vram_gib:.0f} GiB."
            )
    except Exception as exc:  # pragma: no cover - executed on the rented GPU host
        report["cuda_error"] = repr(exc)
        failures.append(f"Torch/CUDA inspection failed: {exc}")

    help_text = ""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "sglang.launch_server", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
        help_text = proc.stdout
        required_flags = (
            "--revision",
            "--speculative-algorithm",
            "--speculative-draft-model-path",
            "--speculative-draft-model-revision",
            "--speculative-dspark-block-size",
        )
        report["sglang_cli"] = {
            "returncode": proc.returncode,
            "required_flags": {flag: flag in help_text for flag in required_flags},
        }
        if proc.returncode != 0 or not all(flag in help_text for flag in required_flags):
            failures.append("Installed SGLang does not expose the complete DSpark CLI")
    except Exception as exc:
        report["sglang_cli_error"] = repr(exc)
        failures.append(f"SGLang CLI inspection failed: {exc}")

    patched, patch_path = has_sglang_patch()
    report["qwen35_last_layer_patch"] = {"present": patched, "path": patch_path}
    if not patched:
        failures.append("Qwen3.5 final-layer capture patch is missing")

    try:
        target_config, target_path, target_revision = get_config(
            args.model, args.model_revision, args.cache_dir
        )
        draft_config, draft_path, draft_revision = get_config(
            args.draft, args.draft_revision, args.cache_dir
        )
        layers = target_num_layers(target_config)
        taps = draft_layer_ids(draft_config)
        gamma = draft_gamma(draft_config)
        architectures = draft_config.get("architectures", [])
        report["checkpoint"] = {
            "target_config_path": target_path,
            "draft_config_path": draft_path,
            "target_revision": target_revision,
            "draft_revision": draft_revision,
            "target_num_hidden_layers": layers,
            "draft_architectures": architectures,
            "target_layer_ids": taps,
            "gamma": gamma,
        }
        if "Qwen3DSparkModel" not in architectures:
            failures.append(f"Unexpected DSpark architecture: {architectures!r}")
        if layers is None or not taps:
            failures.append("Could not resolve target layer count or DSpark tap layers")
        elif max(taps) >= layers:
            failures.append(f"DSpark tap layer is outside target range: {taps} vs {layers}")
        elif max(taps) == layers - 1:
            notes.append(
                "The draft consumes the target's final decoder layer; the installed "
                "SGLang source patch is required until upstream PR #32206 is merged."
            )
        if gamma is not None and gamma != 8:
            notes.append(f"Checkpoint gamma is {gamma}; do not force the launch default of 8")
    except Exception as exc:
        report["checkpoint_error"] = repr(exc)
        failures.append(f"Hugging Face config validation failed: {exc}")

    report["failures"] = failures
    report["notes"] = notes
    report["environment_ready"] = not failures
    gpu_vram = (report.get("cuda", {}).get("gpus") or [{}])[0].get("vram_gib", 0)
    report["mode"] = (
        "full_server_candidate"
        if not failures and gpu_vram >= args.recommended_vram_gib
        else "environment_only"
    )

    output = Path(args.report)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.revision_lock and not failures:
        revision_lock = Path(args.revision_lock)
        revision_lock.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "model_revision": report["checkpoint"]["target_revision"],
                    "draft_model": args.draft,
                    "draft_revision": report["checkpoint"]["draft_revision"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote model revision lock: {revision_lock.resolve()}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote preflight report: {output.resolve()}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
