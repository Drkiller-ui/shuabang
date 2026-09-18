from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from PIL import Image

from mmeval.agent_tools import PythonToolExecutor, ResearchToolSession, _validate_public_url
from mmeval.answer import extract_choice, extract_final, math_equal
from mmeval.client import Generation, split_reasoning
from mmeval.common import read_json, read_jsonl, write_jsonl
from mmeval.diagnostics import analyze_thinking
from mmeval.execution import SandboxExecutor, extract_code
from mmeval.infer import effective_tool_profile, run_inference
from mmeval.prompts import build_user_content
from mmeval.report import aggregate, make_report
from mmeval.scoring import (multimodalqa_metrics, score_gpqa,
                            score_multimodalqa, score_run)
from mmeval.tool_agent import run_tool_agent
from mmeval.vision_tools import VisionToolSession, extract_tool_call


class AnswerTests(unittest.TestCase):
    def test_reasoning_fields_and_embedded_think(self):
        answer, reasoning = split_reasoning({
            "reasoning_content": "server trace",
            "content": "<think>embedded trace</think>Final answer: C",
        })
        self.assertEqual(answer, "Final answer: C")
        self.assertIn("server trace", reasoning)
        self.assertIn("embedded trace", reasoning)

    def test_answer_extractors(self):
        self.assertEqual(extract_final(r"work... \\boxed{\\frac{1}{2}}"), r"\\frac{1}{2}")
        self.assertEqual(extract_choice("The answer is (J).", "ABCDEFGHIJ"), "J")
        self.assertIsNone(extract_choice("This looks good", "ABCD"))
        self.assertTrue(math_equal("0.5", "1/2"))
        self.assertEqual(extract_code("text\n```python\nprint(1)\n```"), "print(1)")

    def test_thinking_diagnostics_detect_limit_loop_and_reopen(self):
        sentence = "This calculation repeats the same intermediate conclusion for no useful reason."
        value = analyze_thinking(
            reasoning=" ".join([sentence] * 4) + " Therefore the final answer is A. Wait, reconsider the derivation.",
            answer="The correct answer is (A).", raw_content="", finish_reason="length",
            usage={"completion_tokens": 16384}, max_tokens=16384,
        )
        self.assertTrue(value["truncated_at_token_limit"])
        self.assertTrue(value["overthinking"])
        self.assertTrue(value["loop_detected"])
        self.assertTrue(value["answer_then_reopen"])
        self.assertTrue(value["flagged"])

    def test_thinking_diagnostics_clean_short_reasoning(self):
        value = analyze_thinking(
            reasoning="Use conservation of energy, substitute the values, and simplify.",
            answer="The correct answer is (B).", raw_content="", finish_reason="stop",
            usage={"completion_tokens": 80}, max_tokens=16384,
        )
        self.assertFalse(value["flagged"])


class PromptTests(unittest.TestCase):
    def test_mmmu_images_are_interleaved_without_base64_in_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("one.png", "two.png"):
                Image.new("RGB", (2, 2), "white").save(root / name)
            row = {
                "benchmark": "mmmu", "question": "First <image 1>, then <image 2>",
                "options": ["x", "y"], "images": ["one.png", "two.png"],
                "metadata": {"image_map": {"<image 1>": "one.png", "<image 2>": "two.png"}},
            }
            content, log = build_user_content(row, root)
            self.assertEqual(sum(item["type"] == "image_url" for item in content), 2)
            self.assertEqual(log["image_paths"], ["one.png", "two.png"])
            self.assertNotIn("base64", json.dumps(log))

    def test_gpqa_prompt_requests_reasoning_and_final_letter(self):
        row = {"benchmark": "gpqa", "question": "Synthetic science question?",
               "options": ["one", "two", "three", "four"], "images": [], "metadata": {}}
        content, log = build_user_content(row, Path("."))
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("The correct answer is (X)", log["prompt_text"])

    def test_multimodalqa_prompt_interleaves_labeled_candidate_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("one.png", "two.png"):
                Image.new("RGB", (3, 3), "white").save(root / name)
            row = {
                "id": "multimodalqa:q", "benchmark": "multimodalqa", "question": "Which item?",
                "options": [], "images": ["one.png", "two.png"],
                "metadata": {"context_text": "[Text 1] evidence\n[Table] data",
                             "image_candidates": [{"label": "Image 1", "title": "First"},
                                                  {"label": "Image 2", "title": "Second"}]}}
            content, log = build_user_content(row, root)
            self.assertEqual(sum(item["type"] == "image_url" for item in content), 2)
            self.assertIn("official full context contains distractors", log["prompt_text"])
            self.assertIn("<answer>", log["prompt_text"])
            self.assertEqual(content[1]["text"].strip(), "[Image 1] First")


class DatasetTests(unittest.TestCase):
    def test_gpqa_diamond_is_complete_and_answer_mapping_is_consistent(self):
        suite = Path(__file__).resolve().parents[1] / "data/mini_eval_v1"
        questions = read_jsonl(suite / "questions/gpqa.jsonl")
        references = read_jsonl(suite / "references/gpqa.jsonl")
        self.assertEqual(len(questions), 198)
        self.assertEqual([q["id"] for q in questions], [r["id"] for r in references])
        for question, reference in zip(questions, references):
            self.assertEqual(question["options"][reference["answer_index"]], reference["answer_text"])
            self.assertEqual(reference["answer"], "ABCD"[reference["answer_index"]])
        self.assertEqual(sum(q["metadata"]["duplicate_distractor"] for q in questions), 2)

    def test_gpqa_scorer(self):
        question = {"id": "gpqa:synthetic", "benchmark": "gpqa",
                    "options": ["a", "b", "c", "d"], "metadata": {}}
        reference = {"id": question["id"], "answer": "C", "answer_text": "c"}
        response = {"id": question["id"], "status": "ok", "reasoning": "trace",
                    "answer": "The correct answer is (C)."}
        score = score_gpqa(question, reference, response, "test")
        self.assertTrue(score["correct"])
        self.assertEqual(score["prediction"], "C")

    def test_multimodalqa_cross_modal_sample_and_scorer(self):
        suite = Path(__file__).resolve().parents[1] / "data/mini_eval_v1"
        questions = read_jsonl(suite / "questions/multimodalqa.jsonl")
        references = read_jsonl(suite / "references/multimodalqa.jsonl")
        self.assertEqual(len(questions), 100)
        self.assertEqual([row["id"] for row in questions], [row["id"] for row in references])
        self.assertTrue(all(len(set(row["metadata"]["modalities"])) >= 2 for row in questions))
        self.assertEqual(len({path for row in questions for path in row["images"]}), 958)
        serialized = json.dumps(questions)
        self.assertNotIn("supporting_context", serialized)
        self.assertNotIn("intermediate_answers", serialized)

        question = {"id": "multimodalqa:synthetic", "benchmark": "multimodalqa",
                    "metadata": {"modality_composition": "image+table", "question_type": "Compose"}}
        reference = {"id": question["id"], "answers": ["New York", "42"],
                     "answer_records": [{"answer": "New York", "modality": "image"},
                                        {"answer": "42", "modality": "table"}]}
        response = {"id": question["id"], "status": "ok", "reasoning": "trace",
                    "answer": '<answer>["42", "New York"]</answer>'}
        score = score_multimodalqa(question, reference, response, "test")
        self.assertTrue(score["correct"])
        self.assertEqual(score["f1"], 1.0)
        self.assertEqual(multimodalqa_metrics(["New York"], ["New York", "42"]), (0.0, 0.5))


class InferenceTests(unittest.TestCase):
    def test_openai_endpoint_writes_reasoning_and_answer(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002
                return

            def do_GET(self):  # noqa: N802
                body = {"data": [{"id": "mock-model"}]}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_POST(self):  # noqa: N802
                length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(length))
                self.server.last_request = request
                self.server.request_count += 1
                body = {
                    "choices": [{"message": {"reasoning_content": "visible trace", "content": "Final answer: A"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 5},
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.request_count = 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                suite, run = root / "suite", root / "run"
                question = {
                    "id": "mmlu_pro:q1", "benchmark": "mmlu_pro", "question": "Q?",
                    "options": ["yes", "no"], "images": [], "metadata": {"category": "logic"},
                }
                write_jsonl(suite / "questions/mmlu_pro.jsonl", [question])
                write_jsonl(suite / "references/mmlu_pro.jsonl", [{"id": "mmlu_pro:q1", "answer": "A"}])
                args = argparse.Namespace(
                    suite=str(suite), run=str(run), benchmarks="mmlu_pro",
                    api_base=f"http://127.0.0.1:{server.server_port}/v1", api_key_env="MISSING_KEY",
                    model="mock-model", concurrency=1, temperature=0.0, seed=1,
                    max_tokens=32, timeout=10.0, retries=1, limit=None,
                    force=False,
                    extra_body_json='{"chat_template_kwargs":{"enable_thinking":true}}',
                )
                asyncio.run(run_inference(args))
                record = read_jsonl(run / "responses/mmlu_pro.jsonl")[0]
                self.assertEqual(record["reasoning"], "visible trace")
                self.assertEqual(record["answer"], "Final answer: A")
                self.assertTrue(server.last_request["chat_template_kwargs"]["enable_thinking"])
                asyncio.run(run_inference(args))
                self.assertEqual(server.request_count, 1, "matching successful request should resume without regeneration")
                score_run(argparse.Namespace(
                    suite=str(suite), run=str(run), benchmarks="mmlu_pro", executor="local",
                    container_image="unused", workers=1, problem_timeout=10.0,
                    per_test_timeout=1.0, force=False, judge_api_base=args.api_base,
                ))
                make_report(argparse.Namespace(suite=str(suite), run=str(run), benchmarks="mmlu_pro"))
                self.assertEqual(read_json(run / "summary.json")["benchmarks"]["mmlu_pro"]["accuracy"], 1.0)
                self.assertEqual(read_jsonl(run / "results/mmlu_pro.jsonl")[0]["response"]["reasoning"], "visible trace")
        finally:
            server.shutdown()
            server.server_close()

    def test_agentic_inference_persists_full_tool_trace(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002
                return

            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "mock-model"}]}).encode())

            def do_POST(self):  # noqa: N802
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assistant_turns = sum(item["role"] == "assistant" for item in request["messages"])
                content = ('<tool_call>{"name":"python_interpreter","arguments":{"code":"print(21*2)"}}</tool_call>'
                           if assistant_turns == 0 else "The correct answer is (A).")
                body = {"choices": [{"message": {"reasoning_content": "trace", "content": content},
                                     "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 4}}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        prior_key = os.environ.get("SERPER_API_KEY")
        os.environ["SERPER_API_KEY"] = "test-only"
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                suite, run = root / "suite", root / "run"
                write_jsonl(suite / "questions/gpqa.jsonl", [{
                    "id": "gpqa:agent", "benchmark": "gpqa", "question": "Q?",
                    "options": ["yes", "no", "maybe", "unknown"], "images": [],
                    "metadata": {"domain": "science", "subdomain": "test"}}])
                write_jsonl(suite / "references/gpqa.jsonl", [
                    {"id": "gpqa:agent", "answer": "A", "answer_text": "yes"}])
                asyncio.run(run_inference(argparse.Namespace(
                    suite=str(suite), run=str(run), benchmarks="gpqa",
                    api_base=f"http://127.0.0.1:{server.server_port}/v1", api_key_env="MISSING_KEY",
                    model="mock-model", concurrency=1, temperature=0.0, seed=1,
                    max_tokens=32, timeout=10.0, retries=1, limit=None, force=False,
                    tools="agentic", max_tool_turns=2, tool_executor="local",
                    tool_container_image="unused", tool_python_timeout=5.0, extra_body_json=None)))
                response = read_jsonl(run / "responses/gpqa.jsonl")[0]
                manifest = read_json(run / "run_manifest.json")
                self.assertEqual(response["answer"], "The correct answer is (A).")
                self.assertEqual(response["tool_trajectory"][0]["tool_result"]["stdout"].strip(), "42")
                self.assertEqual(response["tool_diagnostics"]["tool_calls"], 1)
                self.assertTrue(response["tool_diagnostics"]["network_access"])
                self.assertEqual(manifest["evaluation_track"], "open_book_agentic")
                self.assertEqual(manifest["tools"]["network_search_available_on"], ["gpqa"])
        finally:
            if prior_key is None:
                os.environ.pop("SERPER_API_KEY", None)
            else:
                os.environ["SERPER_API_KEY"] = prior_key
            server.shutdown()
            server.server_close()


class SandboxTests(unittest.TestCase):
    def test_stdin_and_future_functional_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            executor = SandboxExecutor("local", "unused", 10, Path(temporary))
            stdin = executor.livecodebench({
                "code": "a,b=map(int,input().split());print((a+b)**2)",
                "tests": [{"input": "2 3", "output": "25", "testtype": "stdin"}],
                "func_name": None, "per_test_timeout": 1,
            }, "stdin")
            self.assertTrue(stdin["ok"], stdin)
            functional = executor.livecodebench({
                "code": "from __future__ import annotations\nclass Solution:\n def add(self,a:int,b:int)->int:return a+b",
                "tests": [{"input": "2\n3", "output": "5", "testtype": "functional"}],
                "func_name": "add", "per_test_timeout": 1,
            }, "functional")
            self.assertTrue(functional["ok"], functional)


class AgentToolTests(unittest.TestCase):
    def test_benchmark_policy_registers_network_search_only_for_mmqa_and_gpqa(self):
        profiles = {
            name: effective_tool_profile(name, "agentic")
            for name in ("mathvision", "mmmu", "mmlu_pro", "livecodebench", "multimodalqa", "gpqa")
        }
        self.assertEqual(profiles, {
            "mathvision": "local-vision-python", "mmmu": "local-vision",
            "mmlu_pro": "none", "livecodebench": "python",
            "multimodalqa": "agentic", "gpqa": "agentic",
        })

    def test_offline_python_registry_excludes_every_network_tool(self):
        class FakeClient:
            def __init__(self):
                self.system_prompt = ""

            async def generate_messages(self, **kwargs):
                self.system_prompt = kwargs["messages"][0]["content"]
                return Generation(content="done", reasoning="", raw={}, latency_seconds=0,
                                  usage={"completion_tokens": 1}, finish_reason="stop")

        async def exercise(root: Path):
            client = FakeClient()
            result = await run_tool_agent(
                client=client, model="mock", initial_content=[{"type": "text", "text": "Q?"}],
                image_paths=[], artifact_dir=root / "artifacts", max_tokens=16, max_turns=1,
                temperature=0, seed=1, extra_body={}, tool_mode="python",
                python_executor=PythonToolExecutor("local", "unused", 5, root / "work"))
            return client.system_prompt, result

        with tempfile.TemporaryDirectory() as temporary:
            system_prompt, result = asyncio.run(exercise(Path(temporary)))
        self.assertIn('"python_interpreter"', system_prompt)
        for forbidden in ('"web_search"', '"text_search"', '"image_search"', '"visit"'):
            self.assertNotIn(forbidden, system_prompt)
        self.assertFalse(result.diagnostics["network_access"])

    def test_offline_executor_rejects_forged_network_call(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            async def generate_messages(self, **kwargs):
                self.calls += 1
                content = ('<tool_call>{"name":"web_search","arguments":{"query":"leaked answer"}}</tool_call>'
                           if self.calls == 1 else "done")
                return Generation(content=content, reasoning="", raw={}, latency_seconds=0,
                                  usage={"completion_tokens": 1}, finish_reason="stop")

        async def exercise(root: Path):
            return await run_tool_agent(
                client=FakeClient(), model="mock", initial_content=[{"type": "text", "text": "Q?"}],
                image_paths=[], artifact_dir=root / "artifacts", max_tokens=16, max_turns=2,
                temperature=0, seed=1, extra_body={}, tool_mode="python",
                python_executor=PythonToolExecutor("local", "unused", 5, root / "work"))

        with tempfile.TemporaryDirectory() as temporary:
            result = asyncio.run(exercise(Path(temporary)))
        rejected = result.trajectory[0]["tool_result"]
        self.assertFalse(rejected["ok"])
        self.assertIn("disabled tool", rejected["error"])

    def test_visual_tool_registry_crop_and_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (20, 10), "navy").save(source)
            session = VisionToolSession([source], root / "outputs")
            call = extract_tool_call(
                '<tool_call>{"name":"crop","arguments":{"image":"img_1","x":2,"y":1,"width":5,"height":4}}</tool_call>')
            self.assertIsNotNone(call)
            result, output = session.execute(call or {})
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["output_size"], [5, 4])
            self.assertTrue(output and output.is_file())
            self.assertEqual(session.register_image(output), "img_3")

    def test_python_interpreter_captures_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor = PythonToolExecutor("local", "unused", 5, root / "work")
            result, output = executor.execute(
                {"name": "python_interpreter", "arguments": {"code": "print(sum(i*i for i in range(5)))"}},
                root / "artifacts")
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["stdout"].strip(), "30")
            self.assertIsNone(output)

    def test_visit_rejects_private_targets(self):
        for value in ("http://127.0.0.1/x", "http://169.254.169.254/latest", "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                _validate_public_url(value)

    def test_search_and_image_search_parse_and_cache_results(self):
        async def exercise(root: Path):
            calls = []

            def handler(request: httpx.Request) -> httpx.Response:
                calls.append(request.url.path)
                if request.url.path.endswith("/images"):
                    return httpx.Response(200, json={"images": [{
                        "title": "diagram", "source": "example", "link": "https://example.org/page",
                        "imageUrl": "https://example.org/image.png"}]})
                return httpx.Response(200, json={"organic": [{
                    "title": "result", "link": "https://example.org/page", "snippet": "evidence"}]})

            session = ResearchToolSession(root / "cache")
            await session.client.aclose()
            session.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            session.serper_key = "test-only"
            try:
                web, _ = await session.execute(
                    {"name": "web_search", "arguments": {"query": "q", "num_results": 3}})
                cached, _ = await session.execute(
                    {"name": "web_search", "arguments": {"query": "q", "num_results": 3}})
                images, _ = await session.execute(
                    {"name": "image_search", "arguments": {"query": "q", "num_results": 3}})
                return web, cached, images, calls
            finally:
                await session.close()

        with tempfile.TemporaryDirectory() as temporary:
            web, cached, images, calls = asyncio.run(exercise(Path(temporary)))
            self.assertTrue(web["ok"], web)
            self.assertFalse(web["cache_hit"])
            self.assertTrue(cached["cache_hit"])
            self.assertIn("imageUrl", images["observation"])
            self.assertEqual(calls, ["/search", "/images"])

    def test_agentic_tool_turn_is_recorded(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            async def generate_messages(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    content = '<tool_call>{"name":"python_interpreter","arguments":{"code":"print(6*7)"}}</tool_call>'
                else:
                    content = "The correct answer is (B)."
                return Generation(content=content, reasoning=f"reasoning {self.calls}",
                                  raw={"turn": self.calls}, latency_seconds=0.01,
                                  usage={"completion_tokens": 4}, finish_reason="stop")

        async def exercise(root: Path):
            research = ResearchToolSession(root / "cache")
            try:
                result = await run_tool_agent(
                    client=FakeClient(), model="mock", initial_content=[{"type": "text", "text": "Q?"}],
                    image_paths=[], artifact_dir=root / "artifacts", max_tokens=16, max_turns=3,
                    temperature=0, seed=1, extra_body={}, tool_mode="agentic",
                    research_session=research,
                    python_executor=PythonToolExecutor("local", "unused", 5, root / "work"))
                return result
            finally:
                await research.close()

        with tempfile.TemporaryDirectory() as temporary:
            result = asyncio.run(exercise(Path(temporary)))
            self.assertEqual(result.content, "The correct answer is (B).")
            self.assertEqual(result.diagnostics["tool_counts"], {"python_interpreter": 1})
            self.assertEqual(result.diagnostics["remaining_output_tokens"], 8)
            self.assertEqual(result.trajectory[0]["tool_result"]["stdout"].strip(), "42")


class ReportTests(unittest.TestCase):
    def test_multimodalqa_report_uses_f1_as_primary_score(self):
        rows = [
            {"correct": True, "f1": 1.0, "exact_match": 1.0,
             "metadata": {"modality_composition": "image+table", "question_type": "Compose"},
             "model_trace": {"status": "ok"}},
            {"correct": False, "f1": 0.5, "exact_match": 0.0,
             "metadata": {"modality_composition": "image+table", "question_type": "Compose"},
             "model_trace": {"status": "ok"}},
        ]
        result = aggregate(rows, "multimodalqa")
        self.assertEqual(result["score"], 75.0)
        self.assertEqual(result["f1"], 0.75)
        self.assertEqual(result["exact_match"], 0.5)

    def test_badcase_contains_question_reference_and_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite, run = root / "suite", root / "run"
            question = {
                "id": "mmlu_pro:q1", "benchmark": "mmlu_pro", "question": "Q?",
                "options": ["yes", "no"], "images": [], "metadata": {"category": "logic"},
            }
            reference = {"id": "mmlu_pro:q1", "answer": "A"}
            response = {"id": "mmlu_pro:q1", "status": "ok", "reasoning": "trace", "answer": "B",
                        "thinking_diagnostics": {"flagged": True, "loop_detected": True,
                                                 "truncated_at_token_limit": False,
                                                 "answer_then_reopen": False,
                                                 "continued_long_after_final_answer": False}}
            score = {
                "id": "mmlu_pro:q1", "benchmark": "mmlu_pro", "correct": False,
                "metadata": question["metadata"], "model_trace": response,
            }
            write_jsonl(suite / "questions/mmlu_pro.jsonl", [question])
            write_jsonl(suite / "references/mmlu_pro.jsonl", [reference])
            write_jsonl(run / "responses/mmlu_pro.jsonl", [response])
            write_jsonl(run / "scores/mmlu_pro.jsonl", [score])
            make_report(argparse.Namespace(suite=str(suite), run=str(run), benchmarks="mmlu_pro"))
            badcase = read_jsonl(run / "badcases/mmlu_pro.jsonl")[0]
            self.assertEqual(badcase["response"]["reasoning"], "trace")
            self.assertEqual(badcase["reference"]["answer"], "A")
            self.assertEqual(read_json(run / "summary.json")["benchmarks"]["mmlu_pro"]["accuracy"], 0.0)
            self.assertEqual(len(read_jsonl(run / "results/mmlu_pro.jsonl")), 1)
            self.assertEqual(len(read_jsonl(run / "thinking_cases/mmlu_pro.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
