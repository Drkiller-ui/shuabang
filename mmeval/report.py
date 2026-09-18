from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import latest_records, load_benchmark, read_json, utc_now, write_json, write_jsonl
from .infer import parse_benchmarks


GROUP_FIELDS = {
    "mmmu": ("subject", "difficulty"),
    "multimodalqa": ("modality_composition", "question_type"),
    "gpqa": ("domain", "subdomain"),
}


def aggregate(rows: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    total = len(rows)
    scored = sum("score_error" not in row and row.get("model_trace", {}).get("status") == "ok" for row in rows)
    result: dict[str, Any] = {"total": total, "successful_responses": scored, "coverage": scored / total if total else 0.0}
    result["response_status"] = dict(Counter(row.get("model_trace", {}).get("status", "missing") for row in rows))
    result["evaluation_tracks"] = dict(Counter(
        row.get("model_trace", {}).get("evaluation_track", "unknown") for row in rows))
    result["tool_profiles"] = dict(Counter(
        row.get("model_trace", {}).get("tool_profile", "unknown") for row in rows))
    result["score_errors"] = sum("score_error" in row for row in rows)
    if benchmark == "multimodalqa":
        f1_values = [float(row.get("f1", 0.0)) for row in rows]
        em_values = [float(row.get("exact_match", 0.0)) for row in rows]
        result.update({"score": 100.0 * sum(f1_values) / total if total else 0.0,
                       "f1": sum(f1_values) / total if total else 0.0,
                       "exact_match": sum(em_values) / total if total else 0.0,
                       "correct": sum(bool(row.get("correct")) for row in rows)})
    else:
        correct = sum(bool(row.get("correct")) for row in rows)
        result.update({"correct": correct, "accuracy": correct / total if total else 0.0})
    groups = {}
    for field in GROUP_FIELDS[benchmark]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get("metadata", {}).get(field, "unknown"))].append(row)
        groups[field] = {}
        for key, items in sorted(buckets.items()):
            if benchmark == "multimodalqa":
                groups[field][key] = {
                    "count": len(items),
                    "f1": sum(float(item.get("f1", 0.0)) for item in items) / len(items),
                    "exact_match": sum(float(item.get("exact_match", 0.0)) for item in items) / len(items),
                }
            else:
                count = sum(bool(item.get("correct")) for item in items)
                groups[field][key] = {"count": len(items), "correct": count, "accuracy": count / len(items)}
    result["groups"] = groups
    usage = defaultdict(float)
    latencies = []
    for row in rows:
        trace = row.get("model_trace", {})
        for key, value in (trace.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
        if isinstance(trace.get("latency_seconds"), (int, float)):
            latencies.append(trace["latency_seconds"])
    result["usage"] = dict(usage)
    if latencies:
        ordered = sorted(latencies)

        def percentile(fraction: float) -> float:
            return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

        result["latency_seconds"] = {
            "mean": sum(latencies) / len(latencies), "p50": percentile(0.50),
            "p95": percentile(0.95), "max": ordered[-1],
        }
    else:
        result["latency_seconds"] = None
    result["finish_reasons"] = dict(Counter(
        row.get("model_trace", {}).get("finish_reason", "missing") for row in rows
    ))
    diagnostics = [row.get("model_trace", {}).get("thinking_diagnostics") for row in rows]
    diagnostics = [value for value in diagnostics if isinstance(value, dict)]
    diagnostic_counts = {
        "available": len(diagnostics),
        "flagged": sum(bool(value.get("flagged")) for value in diagnostics),
        "long_reasoning_over_16k": sum(bool(value.get("long_reasoning_over_16k")) for value in diagnostics),
        "truncated_at_limit": sum(bool(value.get("truncated_at_token_limit")) for value in diagnostics),
        "loop_detected": sum(bool(value.get("loop_detected")) for value in diagnostics),
        "answer_then_reopen": sum(bool(value.get("answer_then_reopen")) for value in diagnostics),
        "continued_long_after_final": sum(bool(value.get("continued_long_after_final_answer")) for value in diagnostics),
    }
    denominator = len(diagnostics)
    diagnostic_counts["flagged_rate"] = diagnostic_counts["flagged"] / denominator if denominator else None
    diagnostic_counts["long_reasoning_rate"] = (
        diagnostic_counts["long_reasoning_over_16k"] / denominator if denominator else None)
    diagnostic_counts["truncation_rate"] = diagnostic_counts["truncated_at_limit"] / denominator if denominator else None
    result["thinking_diagnostics"] = diagnostic_counts
    tool_diagnostics = [row.get("model_trace", {}).get("tool_diagnostics") for row in rows]
    tool_diagnostics = [value for value in tool_diagnostics if isinstance(value, dict)]
    tool_names: Counter[str] = Counter()
    for value in tool_diagnostics:
        tool_names.update(value.get("tool_counts") or {})
    result["tools"] = {
        "enabled_responses": len(tool_diagnostics),
        "calls": sum(int(value.get("tool_calls", 0)) for value in tool_diagnostics),
        "successful_calls": sum(int(value.get("successful_tool_calls", 0)) for value in tool_diagnostics),
        "failed_calls": sum(int(value.get("failed_tool_calls", 0)) for value in tool_diagnostics),
        "loop_stops": sum(bool(value.get("tool_loop_detected")) for value in tool_diagnostics),
        "turn_limit_stops": sum(bool(value.get("tool_turn_limit_reached")) for value in tool_diagnostics),
        "token_budget_stops": sum(bool(value.get("output_token_budget_exhausted")) for value in tool_diagnostics),
        "by_name": dict(tool_names),
    }
    return result


def make_report(args: argparse.Namespace) -> None:
    suite, run = Path(args.suite).resolve(), Path(args.run).resolve()
    benchmarks = parse_benchmarks(args.benchmarks)
    manifest_path = run / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    summary = {"generated_at": utc_now(), "suite": str(suite), "run": str(run),
               "evaluation_track": manifest.get("evaluation_track", "unknown"),
               "tools": manifest.get("tools", {}), "benchmarks": {}}
    for benchmark in benchmarks:
        questions, references = load_benchmark(suite, benchmark)
        responses = latest_records(run / "responses" / f"{benchmark}.jsonl")
        scores = latest_records(run / "scores" / f"{benchmark}.jsonl")
        rows = []
        badcases = []
        complete_results = []
        thinking_cases = []
        for question in questions:
            score = scores.get(question["id"], {
                "id": question["id"], "benchmark": benchmark, "correct": False,
                "metadata": question["metadata"], "score_error": "missing score",
                "model_trace": responses.get(question["id"], {"status": "missing"}),
            })
            rows.append(score)
            joined = {"id": question["id"], "benchmark": benchmark,
                      "question": question, "reference": references[question["id"]],
                      "response": responses.get(question["id"], {"status": "missing"}),
                      "score": score}
            complete_results.append(joined)
            if joined["response"].get("thinking_diagnostics", {}).get("flagged"):
                thinking_cases.append(joined)
            if not score.get("correct"):
                badcases.append(joined)
        summary["benchmarks"][benchmark] = aggregate(rows, benchmark)
        write_jsonl(run / "results" / f"{benchmark}.jsonl", complete_results)
        write_jsonl(run / "badcases" / f"{benchmark}.jsonl", badcases)
        write_jsonl(run / "thinking_cases" / f"{benchmark}.jsonl", thinking_cases)
    write_json(run / "summary.json", summary)

    lines = ["# Evaluation summary", "", f"Generated: {summary['generated_at']}",
             f"Evaluation track: `{summary['evaluation_track']}`", "",
             "| Benchmark | Coverage | Main metric | Value | Tool calls | Over 16K | Truncated | Loop | Reopen |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in summary["benchmarks"].items():
        metric = "F1 / 100" if name == "multimodalqa" else "accuracy"
        number = value["score"] if name == "multimodalqa" else 100 * value["accuracy"]
        thinking = value["thinking_diagnostics"]
        lines.append(
            f"| {name} | {100 * value['coverage']:.2f}% | {metric} | {number:.2f} | "
            f"{value['tools']['calls']} | "
            f"{thinking['long_reasoning_over_16k']} | "
            f"{thinking['truncated_at_limit']} | {thinking['loop_detected']} | "
            f"{thinking['answer_then_reopen']} |"
        )
    lines += ["", "Missing or failed responses count as incorrect in the main metric.",
              "MultiModalQA reports the official unordered-list token F1 as its main metric and exact match separately.", ""]
    for name, value in summary["benchmarks"].items():
        lines += [f"## {name}", "", "```json", __import__("json").dumps(value, ensure_ascii=False, indent=2), "```", ""]
    (run / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {run / 'summary.md'}")
