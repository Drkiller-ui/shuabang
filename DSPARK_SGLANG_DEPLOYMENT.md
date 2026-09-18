# Qwen3.5-4B + DSpark + SGLang：单卡验收，双卡正式评测

当前单张 RTX 4090 48GB 只用于环境验收：CUDA 13.0、Python 3.11、PyTorch 2.13.0、SGLang 0.5.19、Qwen3.5-4B target、公开 DSpark draft 和多模态请求都能工作。正式评测换成两张 RTX 4090 48GB，以 DP=2、TP=1 启动两个完整副本，使用 64K context；Harness 并发设为 4。DFlash2 不参与这套方案。

## 一键安装与验收

系统需要 Linux、可运行 CUDA 13.0 wheel 的 NVIDIA 驱动，以及足够的系统内存和磁盘。建议把 Hugging Face 缓存放到容量较大的数据盘：

```bash
cd /path/to/project
export HF_HOME=/path/to/large-disk/huggingface
bash scripts/validate_qwen35_4b_sglang_dspark_4090.sh
```

脚本将依次完成：

1. 创建 `.venv-dspark`，全部 Python 包由 `uv` 安装和管理。
2. 验证当前只暴露一张 GPU、显存至少 40GiB、compute capability 为 8.9、PyTorch 版本为 2.13.0 且 runtime 为 CUDA 13.0。
3. 精确安装 SGLang 0.5.19 并运行项目的全部本地测试。
4. 下载并检查 target/draft 的配置，将两者当前 revision 解析成完整 commit SHA 并写入锁文件，应用并验证 Qwen3.5 最后一层捕获补丁。
5. 下载完整模型并以 32K context、并发 1、关闭 CUDA graph 的保守配置启动。
6. 验证 OpenAI 文本接口、图片输入接口和 SGLang 原生生成接口。
7. 要求原生响应包含 `spec_verify_ct > 0` 且 `spec_accept_length >= 1.01`，证明 DSpark 确实执行。
8. 停止验证服务并保留所有日志。

验收结果保存在：

```text
runs/dspark-4090-install-validation/
├── setup.log
├── nvidia-smi.txt
├── preflight.json
├── model-revisions.json
├── server.log
├── validation.json
└── PASS.txt                 # 只有完整验收通过才生成
```

只要 `PASS.txt` 存在，且 `validation.json` 中 `passed` 为 `true`、`spec_verify_ct` 大于 0，就可以确认安装、模型加载、多模态请求和 DSpark 路径都已打通。首次运行需要下载完整 target 和 draft，启动等待上限默认一小时，可用 `STARTUP_TIMEOUT_SECONDS` 调整。

已经执行过安装后，可跳过依赖重装：

```bash
SKIP_SETUP=1 bash scripts/validate_qwen35_4b_sglang_dspark_4090.sh
```

## SGLang 临时补丁

公开 DSpark checkpoint 使用 target layers `19, 23, 27, 29, 31`。Qwen3.5-4B 有 32 个 decoder layers；当前 SGLang 将 HF layer 31 转换成“在第 32 层之前捕获”，现有实现会直接索引不存在的 `layers[32]`。

`scripts/patch_sglang_qwen35_last_layer.py` 等价实现上游 PR #32206 的窄修复：最后一层输出在 decoder loop 后、final norm 前捕获。补丁器只接受已知源码布局，修改前保存 `.py.dspark-original`；一旦 SGLang 上游源码变化，它会拒绝继续修改，避免静默打错补丁。

参考：

- SGLang issue #32204: https://github.com/sgl-project/sglang/issues/32204
- SGLang PR #32206: https://github.com/sgl-project/sglang/pull/32206
- DSpark checkpoint: https://huggingface.co/shanjiaz/qwen3_5_4b_perfectblend_regen_dspark

## 验收通过后的双卡正式启动

单卡上先手动复现 32K 配置：

```bash
CUDA_VISIBLE_DEVICES=0 \
CONTEXT_LENGTH=32768 \
MAX_RUNNING_REQUESTS=1 \
DISABLE_CUDA_GRAPH=1 \
MEM_FRACTION_STATIC=0.85 \
  bash scripts/launch_qwen35_4b_sglang_dspark_single.sh
```

换到双卡机器后启动正式服务：

```bash
bash scripts/launch_qwen35_4b_sglang_dspark_dual.sh
```

双卡脚本默认 `CUDA_VISIBLE_DEVICES=0,1`、`DP_SIZE=2`、`TP_SIZE=1`、64K context、并发 4。每张卡各加载一套 target+draft；这提高的是吞吐，不会把单个请求跨卡切分。checkpoint 的公开验证配置是 32K context、最大输出 4K，因此 64K、32K 输出上限、多图片输入和并发 4 仍需在正式机器上压测。不要修改 `DSPARK_BLOCK_SIZE=8`，除非新的 checkpoint 配置或实测证明其他值可靠。启动脚本固定 `SGLANG_RAGGED_VERIFY_MODE=compact`。

## 接入评测 Harness

服务启动后，Harness 无需修改：

```bash
API_BASE=http://127.0.0.1:8000/v1 \
MODEL=Qwen/Qwen3.5-4B \
CONCURRENCY=4 \
TOOLS=agentic \
  bash scripts/run_full_eval.sh runs/qwen35-4b-dspark-baseline
```

`TOOLS=agentic` 仍按数据集注册权限：只有 MultiModalQA 和 GPQA 获得联网搜索；LiveCodeBench、MathVision 等只获得各自的离线工具。联网工具需要 `SERPER_API_KEY`。若租用机确实没有 Docker/Podman，可在一次性、无密钥的机器上显式设置 `TOOL_EXECUTOR=local SCORE_EXECUTOR=local`；`uv` 只管理依赖，不能提供代码隔离。

正式训练前后保持 SGLang lock 文件、`model-revisions.json`、gamma、temperature、context、输出上限和并发一致。启动脚本会把锁定的两个 commit SHA 分别传给 `--revision` 和 `--speculative-draft-model-revision`。这样分数、思考轨迹、工具轨迹和 bad case 可以直接比较。
