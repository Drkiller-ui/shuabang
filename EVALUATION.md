# Qwen3.5-4B mini 评测程序

评测程序直接读取 `data/mini_eval_v1`，通过 OpenAI 兼容接口请求模型；当前固定部署为 SGLang 0.5.19 + DSpark。每一道题都会保存输入摘要、模型显式输出的思考内容、最终答案、原始 API 响应、token 用量、耗时、评分细节和错误信息。模型输出超过 16,384 tokens 会标为长思考，32,768 tokens 为硬上限并截断。程序支持断点续跑，已成功生成且评分配置未改变的题目不会重复执行。

## 推荐机器

当前环境验收机器为 Linux + 单张 RTX 4090 48GB。Qwen3.5-4B target 与 2B BF16 DSpark draft 放在同一张卡上，以 32K context、并发 1 验证完整链路。正式评测换成两张 RTX 4090 48GB，以 DP=2、TP=1 运行两个完整副本，再测试 64K context；Harness 并发设为 4。评测速度主要受输出长度、图片数量以及代码沙箱吞吐影响。

服务器需要 Linux、Python 3.11、CUDA 13.0 兼容环境和 R580 或更新的 NVIDIA 驱动。推荐使用 Docker 或 rootless Podman 隔离模型生成的代码；SGLang 推理本身不需要容器。使用 `uv` 安装并完成单卡验收：

```bash
bash scripts/validate_qwen35_4b_sglang_dspark_4090.sh
```

该入口用 `uv` 创建 `.venv-dspark`，精确安装 SGLang 0.5.19 和 PyTorch 2.13.0，验证 CUDA 13.0、单张 48GB Ada GPU、DSpark CLI 和 checkpoint 配置，并锁定 target/draft commit。随后完整加载模型并检查文本、图片及 `spec_verify_ct`/`spec_accept_length`。只有全部通过才生成 `runs/dspark-4090-install-validation/PASS.txt`。完整步骤、上游 Qwen3.5 最后一层捕获问题及临时补丁说明见 [DSPARK_SGLANG_DEPLOYMENT.md](DSPARK_SGLANG_DEPLOYMENT.md)。容器运行时由操作系统提供，不归 `uv` 管理。

## 启动模型

验收完成并换到双卡机器后，以 DP=2、TP=1 启动正式服务：

```bash
bash scripts/launch_qwen35_4b_sglang_dspark_dual.sh
```

正式配置使用 64K 上下文、总请求并发 4、DSpark block size 8，并读取安装时生成的 target/draft revision 锁。每张卡各放一个完整副本；单请求不跨卡切分。评测请求会传入 `chat_template_kwargs.enable_thinking=true`，程序兼容服务返回的 `reasoning_content`、`reasoning` 字段和答案内的 `<think>...</think>`。

可通过环境变量覆盖主要设置：

```bash
CUDA_VISIBLE_DEVICES=0,1 DP_SIZE=2 TP_SIZE=1 CONTEXT_LENGTH=65536 \
MAX_RUNNING_REQUESTS=4 DSPARK_BLOCK_SIZE=8 DISABLE_CUDA_GRAPH=0 \
  bash scripts/launch_qwen35_4b_sglang_dspark_dual.sh
```

启动脚本允许每题最多输入 16 张图片。当前 MultiModalQA 子集中单题最多有 15 张官方候选图片；不要把 `--limit-mm-per-prompt` 调到 15 以下。

当前仓库只维护 SGLang + DSpark 启动路径。若需比较训练前后效果，两次评测应使用相同的 SGLang、DSpark draft、服务参数和 Harness 配置，仅替换 target checkpoint。已有兼容服务时可跳过启动脚本，在运行评测时设置 `API_BASE` 和 `MODEL`。

## 完整评测

在第二个终端运行：

```bash
bash scripts/run_full_eval.sh runs/qwen35-4b-mini-v1
```

脚本顺序执行生成、评分和报告。公开仓库不含 GPQA 及含 GPQA 的汇总文件；运行 `BENCHMARKS=all` 前必须先完成私有重建。缺文件时脚本会列出具体路径并停止，避免误报不完整结果。任何阶段中断后，重复同一命令即可续跑。先做冒烟测试时可以限制每个数据集的题数：

```bash
python -m mmeval infer \
  --run runs/smoke-five \
  --model Qwen/Qwen3.5-4B \
  --limit 5 \
  --concurrency 2 \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":true}}'

python -m mmeval score --run runs/smoke-five --executor docker
python -m mmeval report --run runs/smoke-five
```

`--limit` 只限制生成。报告仍按完整套件计分，因此未生成题目会作为未覆盖且错误，这可以防止把局部结果误报为完整成绩。

## 工具评测模式

`infer` 提供三种请求模式。`agentic` 会应用固定的逐数据集权限，而不是给所有题注册同一组工具：

| 数据集 | `TOOLS=agentic` 时注册的工具 | 联网 | 逐题轨道 |
|---|---|---:|---|
| MultiModalQA | 本地视觉、Docker Python、`web_search`/`text_search`、`image_search`、`visit` | 是 | `open_book_agentic` |
| GPQA | Docker Python、`web_search`/`text_search`、`image_search`、`visit` | 是 | `open_book_agentic` |
| MathVision | 本地视觉、Docker Python（含 SymPy） | 否 | `closed_book_tool_augmented` |
| MMMU | 本地视觉 | 否 | `closed_book_tool_augmented` |
| LiveCodeBench | Docker Python REPL | 否 | `closed_book_tool_augmented` |
| MMLU-Pro | 无，纯 CoT | 否 | `closed_book` |

`TOOLS=none` 会关闭全部工具。`TOOLS=local-vision` 只会给 MathVision、MMMU 和 MultiModalQA 注册本地视觉工具。联网搜索白名单固定为 MultiModalQA 与 GPQA；其他数据集的工具定义中没有搜索、图片搜索或页面访问，执行器也会拒绝伪造的联网工具调用。

搜索使用 Serper，页面正文通过 Jina Reader 获取。联网模式至少设置 `SERPER_API_KEY`；`JINA_API_KEY` 可选。密钥只由 Harness 读取，不写入提示、响应或 manifest：

```bash
export SERPER_API_KEY=...
export JINA_API_KEY=...       # 可选
TOOLS=agentic MAX_TOOL_TURNS=8 \
  bash scripts/run_full_eval.sh runs/qwen35-4b-mini-v1-agentic
```

模型每轮只能调用一个工具。`--max-tool-turns` 限制工具调用轮数；32,768 tokens 是一整道题所有模型轮次共享的总输出预算，不会因为调用工具而重新获得预算。超过 16,384 tokens 会单独标记为长思考。相同调用连续出现三次会被判为工具循环并停止。MultiModalQA 与 GPQA 的搜索和访问结果缓存到 `tool_cache/`，断点续跑时复用；容器模式下的 `python_interpreter` 在无网络容器中运行，只挂载该次调用的临时空目录，不能访问题库、reference 或其他样本。

联网评测应使用独立的 `--run` 目录。一次全套 `TOOLS=agentic` 运行会在 manifest 中标为 `mixed_benchmark_policy`，并在 `benchmark_tracks`、每条 response 的 `tool_profile`/`evaluation_track` 和完整 `tool_trajectory` 中记录实际权限。搜索可能找到题目原文或现成答案，因此 MultiModalQA 与 GPQA 的 `open_book_agentic` 成绩不能与标准闭卷榜单分数直接比较。

只跑部分任务时使用统一的 `--benchmarks`：

```bash
python -m mmeval infer --run runs/text --benchmarks mmlu_pro,livecodebench
python -m mmeval score --run runs/text --benchmarks mmlu_pro,livecodebench
python -m mmeval report --run runs/text --benchmarks mmlu_pro,livecodebench
```

## 输出文件

一次运行的目录结构如下：

```text
runs/qwen35-4b-mini-v1/
├── run_manifest.json
├── responses/<benchmark>.jsonl
├── scores/<benchmark>.jsonl
├── results/<benchmark>.jsonl
├── badcases/<benchmark>.jsonl
├── thinking_cases/<benchmark>.jsonl
├── tool_cache/*.json
├── artifacts/tools/<question-id>/*
├── artifacts/livecodebench/*.py
├── summary.json
└── summary.md
```

`responses/*.jsonl` 是原始生成日志。关键字段如下：

| 字段 | 含义 |
|---|---|
| `reasoning` | 服务显式返回的思考轨迹；不会从最终答案猜测或重建 |
| `answer` | 去除 `<think>` 块后的最终回答或代码 |
| `raw_response` | 完整 OpenAI 兼容响应，便于排查服务端格式 |
| `prompt` | 文本提示和图片相对路径；不重复写入 base64 图片 |
| `usage` | 服务返回的 token 统计 |
| `latency_seconds` | 单题 API 耗时 |
| `thinking_diagnostics` | 超过 16K、32K 截断、重复循环、答后回退和长时间继续生成的诊断 |
| `tool_profile` / `evaluation_track` | 该题实际注册的工具权限与闭卷/联网轨道 |
| `tool_mode` | 固定工具协议版本 |
| `tool_trajectory` | 每轮思考、模型工具调用、参数、observation、URL、代码、stdout/stderr、耗时和错误 |
| `tool_diagnostics` | 工具调用数、按名称统计、循环/轮数/总 token 预算停止原因 |
| `status` / `error` | 成功状态或可恢复的错误信息 |

`scores/*.jsonl` 保存解析出的预测、正确答案、逐题正确性、代码测试通过数、MultiModalQA 的 F1/EM 和一份 `model_trace`。`results/*.jsonl` 将每道题的题目、参考答案、完整响应和评分合并到同一条记录；`badcases/*.jsonl` 是其中未通过项的子集。`thinking_cases/*.jsonl` 收集所有命中思考诊断的题目，即使最终答案正确也会保留。

思考诊断是确定性启发式规则：超过 16K、命中 32K 硬限制、同一完整句出现至少三次、12-gram 重复率过高、给出答案后出现 `wait`、`reconsider`、`重新检查` 等回退词，或者在最终答案后又继续生成大量内容。逐题记录重复率、最大重复次数、触发原因和回退标记，汇总报告给出长思考率、截断率、循环率和答后回退率。它用于筛选分析样本，不直接修改任务正确率。

`summary.json` 是机器可读的汇总；`summary.md` 包含每个数据集总分、覆盖率、分组结果、token 用量和平均耗时。缺失响应、接口错误和评分错误都会进入分母并计为错误。

## 各数据集评分

| 数据集 | 主指标 | 细节 |
|---|---|---|
| MathVision | accuracy | 提取最后一个 `boxed` 答案，选择题匹配选项，开放题做数值或符号等价判断 |
| MMMU | accuracy | 选择题解析字母；开放题做规范化、数值和符号匹配 |
| MMLU-Pro | accuracy | 从最终回答解析 A-J 选项 |
| LiveCodeBench | pass@1 | 生成代码必须通过本项目保存的全部公开与隐藏测试 |
| MultiModalQA | F1（0-100） | 复现官方无序答案列表对齐与规范化 token F1；同时报告 exact match，并按模态组合和题型分组 |
| GPQA Diamond | accuracy | 固定随机排列后的 A-D 精确匹配；按 domain 和 subdomain 输出分组结果 |

MultiModalQA 的模型输出格式为 `<answer>["item 1", "item 2"]</answer>`；单答案也使用一项 JSON 数组。评分端允许列表顺序不同，并用匈牙利匹配对齐预测与金标。该子集使用官方 dev 的完整候选上下文，其中包含干扰文本和图片；联网工具模式仍应单独作为 `open_book_agentic` 轨道报告。

## 代码执行隔离

LiveCodeBench 和 `python_interpreter` 的模型输出属于不可信代码。默认使用 Docker，也支持 rootless Podman；两者都设置无网络、只读根文件系统、内存/CPU/进程数限制、移除 Linux capabilities 和总超时。Podman 环境先执行 `podman build -t qwen35-eval-sandbox:latest -f docker/Dockerfile .`，再以 `TOOL_EXECUTOR=podman SCORE_EXECUTOR=podman` 运行评测。`uv` 只管理 Python 包，不能替代代码沙箱。`local` 模式会让模型生成的代码拥有当前 Linux 用户权限，只能在没有密钥和重要文件的一次性机器上作为临时降级方案，不能作为正式安全配置。

GPQA 官方数据包含防训练污染 canary，并要求避免在线公开题目样例。不要发布 `questions/gpqa.jsonl`、`references/gpqa.jsonl`、GPQA badcase 内容或包含这些文件的完整评测压缩包。

评分缓存由模型回答哈希和执行器配置共同决定。修改模型输出后会自动重评；需要无条件重评时传 `--force`。

## 手动命令

```bash
python -m mmeval --help
python -m mmeval infer --help
python -m mmeval score --help
python -m mmeval report --help
```

本地验证：

```bash
python -m compileall -q mmeval
python -m unittest discover -s tests -v
python scripts/validate_mini_eval.py
```

工具实现参考并修改自 Apache-2.0 的 OpenSearch-VL revision `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`。来源文件、修改说明和许可副本见 `THIRD_PARTY_NOTICES.md` 与 `licenses/OpenSearch-VL-Apache-2.0.txt`。
