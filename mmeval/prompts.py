from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _options(options: list[str]) -> str:
    if not options:
        return ""
    return "\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {value}" for i, value in enumerate(options))


def prompt_text(row: dict[str, Any]) -> str:
    benchmark = row["benchmark"]
    question = row["question"]
    options = _options(row["options"])
    if benchmark == "mathvision":
        return (
            f"{question}{options}\n\nPlease reason step by step, and put only your final answer "
            "within \\boxed{...}. For multiple-choice questions, put the option letter in the box."
        )
    if benchmark == "mmmu":
        return (
            f"{question}{options}\n\nAnalyze the visual information and reason step by step. "
            "End with exactly 'Final answer: X', where X is the option letter for a multiple-choice "
            "question or a concise answer for an open question."
        )
    if benchmark == "mmlu_pro":
        return (
            f"Question:\n{question}{options}\n\nAnswer: Let's think step by step. "
            "At the end write exactly 'The answer is (X).', replacing X with one option letter."
        )
    if benchmark == "gpqa":
        return (
            f"What is the correct answer to this graduate-level science question?\n\n"
            f"{question}{options}\n\nReason step by step. When ready, end with exactly "
            "'The correct answer is (X).', replacing X with one option letter."
        )
    if benchmark == "livecodebench":
        starter = row["metadata"].get("starter_code", "").strip()
        suffix = f"\n\nStarter code:\n```python\n{starter}\n```" if starter else ""
        return (
            f"Solve the following programming problem in Python 3. Think through correctness and complexity first.\n\n"
            f"{question}{suffix}\n\nReturn the complete executable solution in one ```python``` code block. "
            "Do not put tests or explanations inside that final code block."
        )
    if benchmark == "multimodalqa":
        return (
            f"Answer this MultiModalQA question by jointly reasoning over the candidate text passages, "
            f"table, and numbered images. The official full context contains distractors.\n\n"
            f"Question:\n{question}\n\n{row['metadata']['context_text']}\n\n"
            "Reason step by step. End with exactly one <answer>...</answer> block containing a JSON "
            "array of answer strings, for example <answer>[\"item 1\", \"item 2\"]</answer>. "
            "Use a one-item array for a single answer."
        )
    raise ValueError(f"Unknown benchmark: {benchmark}")


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _image_item(path: Path) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_data_url(path)}}


def build_user_content(row: dict[str, Any], suite: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = prompt_text(row)
    paths = [suite / relative for relative in row["images"]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    if row["benchmark"] == "mmmu" and row["metadata"].get("image_map"):
        mapping = row["metadata"]["image_map"]
        pattern = re.compile("(" + "|".join(re.escape(token) for token in sorted(mapping, key=len, reverse=True)) + ")")
        content: list[dict[str, Any]] = []
        for part in pattern.split(text):
            if not part:
                continue
            if part in mapping:
                content.append(_image_item(suite / mapping[part]))
            else:
                content.append({"type": "text", "text": part})
    elif row["benchmark"] == "multimodalqa":
        content = [{"type": "text", "text": text}]
        candidates = row["metadata"]["image_candidates"]
        if len(candidates) != len(paths):
            raise ValueError(f"{row['id']}: image candidate/path count mismatch")
        for candidate, path in zip(candidates, paths):
            content.extend([
                {"type": "text", "text": f"\n[{candidate['label']}] {candidate['title']}"},
                _image_item(path),
            ])
    elif paths:
        # MathVision images may already contain several labeled panels. Send each
        # official image once and keep the panel labels in the text.
        content = [_image_item(path) for path in paths] + [{"type": "text", "text": text}]
    else:
        content = [{"type": "text", "text": text}]
    log = {"prompt_text": text, "image_paths": [str(path.relative_to(suite)).replace("\\", "/") for path in paths]}
    return content, log
