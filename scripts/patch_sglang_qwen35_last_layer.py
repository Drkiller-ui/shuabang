#!/usr/bin/env python3
"""Apply the narrow Qwen3.5 final-layer capture fix proposed in SGLang PR #32206.

The public Qwen3.5-4B DSpark checkpoint consumes target layer 31.  SGLang
converts HF layer ids to "capture before layer k + 1", so an unpatched build
tries to index layer 32 in a 32-layer target.  This patch is deliberately
strict and refuses to edit an unknown source layout.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
from pathlib import Path


UNPATCHED_SETTER = """    def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)
"""

PATCHED_SETTER = """    def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
        self.layers_to_capture = layers_to_capture
        self._capture_after_last_layer = False
        for layer_id in self.layers_to_capture:
            if layer_id >= len(self.layers):
                self._capture_after_last_layer = True
                continue
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)
"""

INSERT_BEFORE = """        # Return intermediate tensors for pipeline parallelism
"""

CAPTURE_BLOCK = """        # HF layer num_layers - 1 is captured after the decoder loop.
        if (
            getattr(self, "_capture_after_last_layer", False)
            and self.pp_group.is_last_rank
        ):
            aux_hidden_states.append(
                hidden_states + residual if residual is not None else hidden_states
            )

"""


def locate_qwen35() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise SystemExit("sglang is not installed in this Python environment")
    return Path(spec.origin).resolve().parent / "srt" / "models" / "qwen3_5.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify only; do not modify files")
    args = parser.parse_args()

    path = locate_qwen35()
    if not path.is_file():
        raise SystemExit(f"Cannot find SGLang Qwen3.5 source: {path}")
    source = path.read_text(encoding="utf-8")

    setter_patched = PATCHED_SETTER in source
    capture_patched = CAPTURE_BLOCK in source
    if setter_patched and capture_patched:
        print(f"SGLang Qwen3.5 final-layer capture patch is present: {path}")
        return
    if args.check:
        raise SystemExit(
            "SGLang Qwen3.5 final-layer capture patch is missing; "
            "Qwen3.5-4B + this DSpark checkpoint will fail at startup"
        )
    if setter_patched != capture_patched:
        raise SystemExit(f"Refusing to modify a partially patched file: {path}")
    if source.count(UNPATCHED_SETTER) != 1:
        raise SystemExit(
            "Installed SGLang has an unknown set_dflash_layers_to_capture layout. "
            "Check whether upstream PR #32206 was merged before updating this patch."
        )
    if source.count(INSERT_BEFORE) != 1:
        raise SystemExit(
            "Installed SGLang has an unknown Qwen3.5 forward layout. "
            "No source file was modified."
        )

    updated = source.replace(UNPATCHED_SETTER, PATCHED_SETTER, 1)
    updated = updated.replace(INSERT_BEFORE, CAPTURE_BLOCK + INSERT_BEFORE, 1)
    ast.parse(updated, filename=str(path))

    backup = path.with_suffix(".py.dspark-original")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    print(f"Applied SGLang PR #32206 equivalent patch: {path}")
    print(f"Original source saved as: {backup}")


if __name__ == "__main__":
    main()
