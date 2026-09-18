from __future__ import annotations

import math
import re
from typing import Any

from sympy import simplify, sympify


def last_boxed(text: str) -> str | None:
    positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not positions:
        return None
    start = positions[-1]
    brace = text.find("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:index].strip()
    return None


def extract_final(text: str) -> str:
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if matches:
        return matches[-1].strip()
    patterns = [
        r"(?:final answer|the answer is|the correct answer is|answer)\s*[:=]?\s*(.+)",
        r"答案\s*(?:是|为|：|:)\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].splitlines()[0].strip().rstrip(".")
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def extract_choice(text: str, choices: str) -> str | None:
    candidates = [extract_final(text), text]
    escaped = re.escape(choices)
    patterns = [
        rf"(?<![A-Z])\(?([{escaped}])\)?(?![A-Z])",
        rf"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([{escaped}])\)?",
    ]
    for candidate in candidates:
        for pattern in patterns[::-1]:
            matches = re.findall(pattern, candidate.upper(), flags=re.I)
            if matches:
                return matches[-1].upper()
    return None


def normalize_math(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("∞", r"\infty").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("$", "").replace(" ", "")
    text = text.rstrip(".,:;")
    text = re.sub(r"^\((.*)\)$", r"\1", text)
    return text


def _number(value: str) -> float | None:
    cleaned = value.replace(",", "").replace("%", "")
    try:
        result = float(cleaned)
        if "%" in value:
            result /= 100.0
        return result
    except ValueError:
        return None


def math_equal(expected: Any, predicted: Any) -> bool:
    left, right = normalize_math(expected), normalize_math(predicted)
    if not left or not right:
        return False
    if left == right:
        return True
    left_number, right_number = _number(left), _number(right)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=1e-6, abs_tol=1e-6)
    try:
        from latex2sympy2_extended import latex2sympy
        return simplify(latex2sympy(left) - latex2sympy(right)) == 0
    except Exception:
        pass
    try:
        return simplify(sympify(left) - sympify(right)) == 0
    except Exception:
        return False


def normalized_open_values(text: str) -> list[str]:
    final = extract_final(text)
    values = [normalize_math(final)]
    values.extend(normalize_math(item) for item in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", final))
    return list(dict.fromkeys(value for value in values if value))
