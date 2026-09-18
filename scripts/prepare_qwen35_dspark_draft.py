#!/usr/bin/env python3
"""Mirror a locked public DSpark draft with an SGLang-readable HF config.

The published checkpoint stores its Qwen3 decoder config under
transformer_layer_config and omits model_type. SGLang 0.5.19 loads the draft
through transformers AutoConfig, which needs those fields at the top level.
Weights are symlinked unchanged from the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = Path(snapshot_download(repo_id=args.repo, revision=args.revision))
    original = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if original.get("architectures") != ["DSparkDraftModel"]:
        raise SystemExit("Unexpected draft architecture; refusing to rewrite config")
    text = original.get("transformer_layer_config")
    taps = original.get("aux_hidden_state_layer_ids")
    if not isinstance(text, dict) or text.get("model_type") != "qwen3":
        raise SystemExit("Unexpected draft transformer config")
    if taps != [19, 23, 27, 29, 31] or text.get("num_hidden_layers") != 5:
        raise SystemExit("Unexpected DSpark target taps or draft layer count")

    normalized = dict(original)
    normalized.update(text)
    normalized.pop("auto_map", None)
    normalized["architectures"] = ["DSparkDraftModel"]
    normalized["model_type"] = "qwen3"
    normalized["target_layer_ids"] = taps
    normalized["num_target_layers"] = 32

    args.output.mkdir(parents=True, exist_ok=True)
    for source in snapshot.rglob("*"):
        if not source.is_file() or source.name == "config.json":
            continue
        destination = args.output / source.relative_to(snapshot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() == source.resolve():
                continue
            destination.unlink()
        elif destination.exists():
            raise SystemExit(f"Refusing to overwrite {destination}")
        destination.symlink_to(source.resolve())

    (args.output / "config.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "source-revision.json").write_text(
        json.dumps({"repo": args.repo, "revision": args.revision}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared unchanged draft weights and normalized config: {args.output.resolve()}")


if __name__ == "__main__":
    main()
