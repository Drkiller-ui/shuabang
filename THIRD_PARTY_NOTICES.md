# Third-party notices

## OpenSearch-VL

Parts of `mmeval/vision_tools.py`, `mmeval/agent_tools.py`, and the multi-turn
tool protocol in `mmeval/tool_agent.py` were adapted from OpenSearch-VL:

- Source: https://github.com/shawn0728/OpenSearch-VL
- Revision: `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`
- Upstream files: `opensearch_vl/opensearch_infer/image_engines.py`,
  `opensearch_vl/opensearch_infer/tools.py`,
  `opensearch_vl/opensearch_infer/search.py`, and the search, visit, Python,
  and crop tool modules under `RL/rllm/vision_deepresearch_async_workflow/tools/`.
- Copyright: 2026 OpenSearch-VL Authors
- License: Apache License 2.0; see `licenses/OpenSearch-VL-Apache-2.0.txt`.

The adapted implementation was changed to fit this evaluator: it uses a
path-confined image registry, deterministic image transforms, structured
per-turn traces, output and resource limits, disk-cached search responses,
public-URL validation, and a network-disabled Docker Python sandbox. It does
not copy OpenSearch-VL model inference or training code.
