from __future__ import annotations

import re
from collections import Counter
from typing import Any


DIAGNOSTICS_VERSION = "thinking-v2"
LONG_REASONING_TOKENS = 16_384
FINAL_PATTERN = re.compile(
    r"(?:final\s+answer|correct\s+answer|the\s+answer\s+is|answer\s*[:=]|"
    r"最终答案|正确答案|答案(?:是|为|[:：])|\\boxed\s*\{)", re.I
)
REOPEN_PATTERN = re.compile(
    r"(?:\bwait\b|\bhowever\b|\breconsider\b|re[- ]?evaluate|re[- ]?check|"
    r"on second thought|let me (?:double[- ]?check|check again|rethink)|"
    r"等等|不过|但是|重新(?:考虑|检查|推导)|再(?:检查|想想|推导)|回过头)", re.I
)


def _completion_tokens(usage: dict[str, Any]) -> int | None:
    for key in ("completion_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _repetition(reasoning: str) -> dict[str, Any]:
    sentences = [
        re.sub(r"\s+", " ", value).strip().lower()
        for value in re.split(r"[.!?。！？]+|[\r\n]+", reasoning)
    ]
    substantial = [value for value in sentences if len(value) >= 30]
    sentence_counts = Counter(substantial)
    repeated_sentences = {value: count for value, count in sentence_counts.items() if count >= 3}

    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", reasoning.lower())
    window = 12
    shingles = [tuple(tokens[index:index + window]) for index in range(max(0, len(tokens) - window + 1))]
    counts = Counter(shingles)
    duplicate_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    ratio = duplicate_occurrences / len(shingles) if shingles else 0.0
    max_shingle_repeats = max(counts.values(), default=0)

    reasons = []
    if repeated_sentences:
        reasons.append("sentence_repeated_at_least_3_times")
    if len(tokens) >= 80 and ratio >= 0.20:
        reasons.append("high_12gram_repetition")
    if len(tokens) >= 40 and max_shingle_repeats >= 4:
        reasons.append("same_12gram_repeated_at_least_4_times")
    return {
        "loop_detected": bool(reasons),
        "loop_reasons": reasons,
        "estimated_reasoning_tokens": len(tokens),
        "repeated_sentence_types": len(repeated_sentences),
        "max_sentence_repeats": max(sentence_counts.values(), default=0),
        "ngram_window": window,
        "ngram_repetition_ratio": round(ratio, 6),
        "max_ngram_repeats": max_shingle_repeats,
    }


def _reopen(reasoning: str, answer: str, raw_content: str) -> dict[str, Any]:
    markers = list(FINAL_PATTERN.finditer(reasoning))
    reopen_after_answer = False
    reopen_marker = None
    if markers:
        after = reasoning[markers[0].end():]
        reopened = REOPEN_PATTERN.search(after)
        if reopened:
            reopen_after_answer = True
            reopen_marker = reopened.group(0)

    # Some servers return embedded think blocks. Detect a final answer between
    # two thinking blocks before split_reasoning removes the tags.
    embedded_reopen = bool(re.search(
        r"</think>.*?" + FINAL_PATTERN.pattern + r".*?<think>", raw_content,
        flags=re.I | re.S,
    )) if "<think>" in raw_content.lower() else False

    answer_markers = list(FINAL_PATTERN.finditer(answer))
    continued_after_final = False
    if answer_markers:
        tail = answer[answer_markers[-1].end():]
        continued_after_final = len(re.findall(r"\w+|[\u4e00-\u9fff]", tail)) > 40

    detected = reopen_after_answer or embedded_reopen
    return {
        "answer_then_reopen": detected,
        "reopen_marker": reopen_marker,
        "embedded_think_reopen": embedded_reopen,
        "continued_long_after_final_answer": continued_after_final,
        "reasoning_final_markers": len(markers),
        "answer_final_markers": len(answer_markers),
    }


def analyze_thinking(*, reasoning: str, answer: str, raw_content: str,
                     finish_reason: str | None, usage: dict[str, Any],
                     max_tokens: int) -> dict[str, Any]:
    completion_tokens = _completion_tokens(usage)
    finish_limit = str(finish_reason or "").lower() in {"length", "max_tokens", "max_token", "limit"}
    usage_limit = completion_tokens is not None and completion_tokens >= max_tokens
    truncated = finish_limit or usage_limit
    long_reasoning = completion_tokens is not None and completion_tokens >= LONG_REASONING_TOKENS
    repetition = _repetition(reasoning)
    reopen = _reopen(reasoning, answer, raw_content)
    flagged = (long_reasoning or truncated or repetition["loop_detected"] or reopen["answer_then_reopen"]
               or reopen["continued_long_after_final_answer"])
    return {
        "version": DIAGNOSTICS_VERSION,
        "max_tokens": max_tokens,
        "long_reasoning_threshold": LONG_REASONING_TOKENS,
        "completion_tokens": completion_tokens,
        "long_reasoning_over_16k": long_reasoning,
        "truncated_at_token_limit": truncated,
        "overthinking": long_reasoning or truncated,
        "flagged": flagged,
        **repetition,
        **reopen,
    }
