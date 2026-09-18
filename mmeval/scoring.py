from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .answer import extract_choice, extract_final, math_equal, normalize_math, normalized_open_values
from .common import (append_jsonl, latest_records, load_benchmark,
                     read_json, utc_now, write_jsonl)
from .execution import SandboxExecutor, extract_code
from .infer import parse_benchmarks

try:
    from word2number.w2n import word_to_num
except ImportError:  # pragma: no cover - requirements-eval installs it on the server
    word_to_num = None


def response_hash(response: dict[str, Any]) -> str:
    value = json.dumps({"status": response.get("status"), "answer": response.get("answer"),
                        "reasoning": response.get("reasoning"),
                        "thinking_diagnostics": response.get("thinking_diagnostics"),
                        "tool_profile": response.get("tool_profile"),
                        "evaluation_track": response.get("evaluation_track"),
                        "tool_mode": response.get("tool_mode"),
                        "tool_trajectory": response.get("tool_trajectory"),
                        "tool_diagnostics": response.get("tool_diagnostics")},
                       sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def trace(response: dict[str, Any]) -> dict[str, Any]:
    return {key: response.get(key) for key in (
        "status", "reasoning", "answer", "finish_reason", "usage", "latency_seconds", "prompt",
        "thinking_diagnostics", "error", "tool_profile", "evaluation_track", "tool_mode",
        "tool_trajectory", "tool_diagnostics"
    ) if key in response}


def base_result(question: dict[str, Any], reference: dict[str, Any], response: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "schema_version": 1, "id": question["id"], "benchmark": question["benchmark"],
        "scored_at": utc_now(), "method": method, "response_sha256": response_hash(response),
        "metadata": question["metadata"], "model_trace": trace(response),
    }


def score_mathvision(question: dict[str, Any], reference: dict[str, Any], response: dict[str, Any], method: str) -> dict[str, Any]:
    result = base_result(question, reference, response, method)
    answer = response.get("answer", "") if response.get("status") == "ok" else ""
    expected = str(reference["answer"])
    extracted = extract_final(answer)
    choice = extract_choice(answer, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(question["options"])]) if question["options"] else None
    expected_value = question["options"][ord(expected) - ord("A")] if question["options"] else ""
    correct = ((choice == expected) if question["options"] else math_equal(expected, extracted))
    if question["options"] and not correct:
        correct = math_equal(expected_value, extracted)
    result.update({"correct": bool(correct), "prediction": choice or extracted,
                   "expected": expected, "expected_value": expected_value,
                   "scoring_detail": "official-style boxed extraction plus symbolic/numeric equivalence"})
    return result


def score_mmmu(question: dict[str, Any], reference: dict[str, Any], response: dict[str, Any], method: str) -> dict[str, Any]:
    result = base_result(question, reference, response, method)
    answer = response.get("answer", "") if response.get("status") == "ok" else ""
    expected = str(reference["answer"])
    if question["metadata"].get("question_type") == "multiple-choice":
        prediction = extract_choice(answer, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(question["options"])])
        correct = prediction == expected
        detail = "MMMU multiple-choice parser"
    else:
        values = normalized_open_values(answer)
        normalized_expected = normalize_math(expected)
        phrase_match = bool(normalized_expected and len(normalized_expected) >= 4
                            and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized_expected))
        correct = any(math_equal(expected, value) or (phrase_match and normalized_expected in value) for value in values)
        prediction = values[0] if values else ""
        detail = "MMMU normalized open-answer matching"
    result.update({"correct": bool(correct), "prediction": prediction, "expected": expected,
                   "scoring_detail": detail})
    return result


def score_mmlu_pro(question: dict[str, Any], reference: dict[str, Any], response: dict[str, Any], method: str) -> dict[str, Any]:
    result = base_result(question, reference, response, method)
    answer = response.get("answer", "") if response.get("status") == "ok" else ""
    expected = str(reference["answer"])
    prediction = extract_choice(answer, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(question["options"])])
    result.update({"correct": prediction == expected, "prediction": prediction, "expected": expected,
                   "scoring_detail": "official-style final option extraction"})
    return result


def score_gpqa(question: dict[str, Any], reference: dict[str, Any], response: dict[str, Any], method: str) -> dict[str, Any]:
    result = base_result(question, reference, response, method)
    answer = response.get("answer", "") if response.get("status") == "ok" else ""
    expected = str(reference["answer"])
    prediction = extract_choice(answer, "ABCD")
    result.update({"correct": prediction == expected, "prediction": prediction, "expected": expected,
                   "expected_value": reference.get("answer_text"),
                   "scoring_detail": "GPQA official-style shuffled A-D exact match"})
    return result


def extract_multimodalqa_answer(text: str) -> list[str]:
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    candidate = tagged[-1] if tagged else text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", candidate, flags=re.I | re.S)
    candidate = fenced[-1].strip() if fenced else candidate.strip()
    try:
        value = json.loads(candidate)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, (str, int, float, bool)):
            return [str(value).strip()]
    except json.JSONDecodeError:
        pass
    match = re.search(r"(?:final\s+answer|answer)\s*:\s*(.+)$", candidate, flags=re.I | re.S)
    value = (match.group(1) if match else candidate).strip()
    return [value] if value else []


def _mmqa_is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _mmqa_normalize_number(text: str) -> str:
    if _mmqa_is_number(text):
        return str(float(text))
    if word_to_num is not None:
        try:
            return str(float(word_to_num(text)))
        except ValueError:
            pass
    return text


def _mmqa_normalize(text: str) -> str:
    def normalize_token(token: str) -> str:
        lowered = token.lower()
        if not _mmqa_is_number(lowered):
            lowered = "".join(char for char in lowered if char not in set(string.punctuation))
        lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
        return " ".join(_mmqa_normalize_number(lowered).split())

    return " ".join(value for value in (normalize_token(token) for token in re.split(r" |‑|-", text))
                    if value.strip()).strip()


def _mmqa_answer_bags(answer: list[str]) -> tuple[list[str], list[set[str]]]:
    normalized = [_mmqa_normalize(str(value)) for value in answer]
    return normalized, [set(value.split()) for value in normalized]


def multimodalqa_metrics(prediction: list[str], gold: list[str]) -> tuple[float, float]:
    """Official MultiModalQA list EM/F1, adapted without the upstream CLI wrapper."""
    pred_spans, pred_bags = _mmqa_answer_bags(prediction)
    gold_spans, gold_bags = _mmqa_answer_bags(gold)
    exact_match = float(set(pred_spans) == set(gold_spans) and len(pred_spans) == len(gold_spans))
    if not pred_bags and not gold_bags:
        return exact_match, 1.0
    if not pred_bags or not gold_bags:
        return exact_match, 0.0
    scores = np.zeros((len(gold_bags), len(pred_bags)))
    for gold_index, gold_bag in enumerate(gold_bags):
        for pred_index, pred_bag in enumerate(pred_bags):
            gold_numbers = {value for value in gold_bag if _mmqa_is_number(value)}
            pred_numbers = {value for value in pred_bag if _mmqa_is_number(value)}
            if gold_numbers and not gold_numbers.intersection(pred_numbers):
                continue
            overlap = len(gold_bag.intersection(pred_bag))
            precision = overlap / len(pred_bag) if pred_bag else 1.0
            recall = overlap / len(gold_bag) if gold_bag else 1.0
            scores[gold_index, pred_index] = (2 * precision * recall / (precision + recall)
                                              if precision or recall else 0.0)
    row_indices, column_indices = linear_sum_assignment(-scores)
    aligned = np.zeros(max(len(gold_bags), len(pred_bags)))
    for row_index, column_index in zip(row_indices, column_indices):
        aligned[row_index] = max(aligned[row_index], scores[row_index, column_index])
    return exact_match, round(float(np.mean(aligned)), 2)


def score_multimodalqa(question: dict[str, Any], reference: dict[str, Any],
                       response: dict[str, Any], method: str) -> dict[str, Any]:
    result = base_result(question, reference, response, method)
    answer = response.get("answer", "") if response.get("status") == "ok" else ""
    prediction = extract_multimodalqa_answer(answer)
    expected = [str(value) for value in reference["answers"]]
    exact_match, f1 = multimodalqa_metrics(prediction, expected)
    result.update({
        "correct": bool(exact_match), "prediction": prediction, "expected": expected,
        "exact_match": exact_match, "f1": f1, "primary_score": 100.0 * f1,
        "answer_modality": sorted({record.get("modality", "unknown")
                                   for record in reference.get("answer_records", [])}),
        "scoring_detail": "official MultiModalQA unordered-list EM and aligned token F1",
    })
    return result


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def score_lcb(question: dict[str, Any], reference: dict[str, Any], response: dict[str, Any], method: str,
              suite: Path, run: Path, executor: SandboxExecutor, per_test_timeout: float) -> dict[str, Any]:
    result = base_result(question, reference, response, method)
    if response.get("status") != "ok":
        result.update({"correct": False, "prediction": None, "execution": {"ok": False, "worker_error": "no successful response"}})
        return result
    code = extract_code(response.get("answer", ""))
    generated = run / "artifacts" / "livecodebench" / f"{safe_name(question['id'])}.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(code, encoding="utf-8")
    tests = read_json(suite / reference["tests_file"])
    job = {"code": code, "tests": tests["public"] + tests["private"],
           "func_name": question["metadata"].get("execution_metadata", {}).get("func_name"),
           "per_test_timeout": per_test_timeout}
    execution = executor.livecodebench(job, question["id"])
    result.update({"correct": bool(execution.get("ok")), "prediction": str(generated.relative_to(run)).replace("\\", "/"),
                   "execution": execution, "expected": "all public and private tests pass"})
    return result


def score_run(args: argparse.Namespace) -> None:
    suite, run = Path(args.suite).resolve(), Path(args.run).resolve()
    benchmarks = parse_benchmarks(args.benchmarks)
    executor = None
    if "livecodebench" in benchmarks:
        executor = SandboxExecutor(args.executor, args.container_image, args.problem_timeout, run / "work")
    method = (f"mmeval-v1;executor={args.executor};image={args.container_image};"
              f"problem_timeout={args.problem_timeout};per_test_timeout={args.per_test_timeout}")

    for benchmark in benchmarks:
        questions, references = load_benchmark(suite, benchmark)
        responses = latest_records(run / "responses" / f"{benchmark}.jsonl")
        score_path = run / "scores" / f"{benchmark}.jsonl"
        existing = latest_records(score_path)
        pending = []
        final: dict[str, dict[str, Any]] = {}
        for question in questions:
            response = responses.get(question["id"], {"id": question["id"], "status": "missing", "answer": "", "reasoning": ""})
            previous = existing.get(question["id"])
            if (not args.force and previous and previous.get("response_sha256") == response_hash(response)
                    and previous.get("method") == method):
                final[question["id"]] = previous
            else:
                pending.append((question, references[question["id"]], response))
        print(f"Scoring {benchmark}: {len(final)} cached, {len(pending)} pending", flush=True)

        def score_one(item: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
            question, reference, response = item
            if benchmark == "mathvision":
                return score_mathvision(question, reference, response, method)
            if benchmark == "mmmu":
                return score_mmmu(question, reference, response, method)
            if benchmark == "mmlu_pro":
                return score_mmlu_pro(question, reference, response, method)
            if benchmark == "gpqa":
                return score_gpqa(question, reference, response, method)
            if benchmark == "multimodalqa":
                return score_multimodalqa(question, reference, response, method)
            if benchmark == "livecodebench":
                assert executor is not None
                return score_lcb(question, reference, response, method, suite, run, executor, args.per_test_timeout)
            raise AssertionError(f"unhandled benchmark {benchmark}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(score_one, item): item[0]["id"] for item in pending}
            for future in concurrent.futures.as_completed(futures):
                item_id = futures[future]
                try:
                    value = future.result()
                except Exception as exc:
                    question, reference, response = next(item for item in pending if item[0]["id"] == item_id)
                    value = base_result(question, reference, response, method)
                    value.update({"correct": False, "score_error": f"{type(exc).__name__}: {exc}"})
                final[item_id] = value
                append_jsonl(score_path, value)
                print(f"[{benchmark}] {item_id}: {'correct' if value.get('correct') else 'badcase'}", flush=True)
        ordered = [final[question["id"]] for question in questions]
        write_jsonl(score_path, ordered)
