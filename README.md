# Qwen3.5-4B 三项评测集

本项目评测 **498 道题**：GPQA Diamond 198 题、MMMU 200 题、MultiModalQA 100 题。GPQA 测研究生级科学推理；MMMU 测多学科图文理解；MultiModalQA 测跨模态多跳问答。后两组均有图片输入。MMMU 和 MultiModalQA 是从固定上游版本按固定种子抽取的本项目子集，不应称为完整官方榜单成绩。

| 数据集 | 来源与划分 | 题数 | 选择方法 |
|---|---|---:|---|
| GPQA Diamond | idavidrein/gpqa，gpqa_diamond | 198 | 完整 Diamond，选项按题号和固定种子稳定打乱 |
| MMMU | MMMU/MMMU，validation | 200 | 按 30 个学科比例分层 |
| MultiModalQA | allenai/multimodalqa，dev | 100 | 真实跨模态题，按模态组合和题型比例分层 |

数据位于 `data/mini_eval_v1`。目录名保留以兼容现有运行路径，目录内容和 `suite.json` 已缩为上述三组。`questions/all.jsonl` 汇总 498 题；`questions/smoke_200.jsonl` 是其中固定的 200 题快速子集。每组分别有 `questions/<benchmark>.jsonl` 和 `references/<benchmark>.jsonl`，输入图片放在 `assets/`。`manifests/` 记录来源版本与抽样 ID，`checksums.json` 记录可公开文件的 SHA-256。GPQA 题目和答案、本地汇总题单、相关抽样 ID、校验值及生成归档不进入公开 Git 历史。

## 当前部署方案

模型使用 Qwen3.5-4B，推理服务使用 SGLang 0.5.19 + DSpark。单张 RTX 4080 SUPER 32GB 已完成 32K context、并发 1 的链路验收；双张 RTX 4090 48GB 计划以 DP=2、TP=1 运行正式评测。安装、验收和模型 revision 锁见 [DSpark 部署说明](DSPARK_SGLANG_DEPLOYMENT.md)。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/validate_qwen35_4b_sglang_dspark_4080super.sh
```

## 运行

先启动 Qwen3.5-4B 的 OpenAI 兼容服务，再运行：

```bash
bash scripts/run_full_eval.sh runs/qwen35-4b-gpqa-mmmu-mmqa-v1
```

脚本依次执行生成、评分和报告，支持断点续跑。默认 `TOOLS=local-vision`：MMMU 和 MultiModalQA 可调用离线视觉工具，GPQA 闭卷作答。**默认评测不需要 Docker 或搜索密钥。** 模型如果没有主动调用视觉工具，日志会显示零次调用。`MAX_TOKENS` 默认 32768，避免先前 2048 token 测试中多数回答被截断。输出和工具调用细节见 [评测说明](EVALUATION.md)。

GPQA 与 MultiModalQA 也支持单独运行 `TOOLS=agentic` 的联网轨道；此模式需要 `SERPER_API_KEY`，并且 Python 工具需要外部可用的 Docker/Podman 沙箱。联网轨道成绩应单独报告，不与默认闭卷成绩混合。

## 数据约定

每条 `questions` 记录包含全局 `id`、上游 `source_id`、固定版本 `source`、题目文本、选项、图片相对路径和必要的 `metadata`。模型只接收 `questions` 和对应图片；`references` 中的答案、支持证据及评分信息不发送给模型。

- MMMU 的 `<image 1>` 等占位符由 `metadata.image_map` 映射到原始图片，保持图文顺序。
- MultiModalQA 保留官方完整候选上下文：文本、表格及候选图片，包含干扰项。正确答案和支持证据只存于 `references`。
- GPQA 的四个选项按固定种子稳定打乱；reference 端记录对应的 A–D 标签。不得恢复为“正确答案恒在第一项”。

固定种子为 `20260916`。各层数量按最大余数法分配，层内按 `SHA256(seed|namespace|source_id)` 排序取题。抽样不依据模型输出。此套件适合开发期比较 checkpoint；`smoke_200` 不是独立验收集。

## 重建和校验

```bash
python -m pip install --target .deps -r requirements-data.txt
python scripts/fetch_sources.py
GPQA_ZIP_PASSWORD=<your-authorized-password> python scripts/build_mini_eval.py
python scripts/finalize_mini_eval.py
python scripts/validate_mini_eval.py
python scripts/package_mini_eval.py
```

下载脚本使用 `config/sources.lock.json` 中的固定版本。已有数据可离线执行 `python scripts/validate_mini_eval.py`；它检查 498 题的 ID、图片、答案映射、跨模态条件、合并题单及校验和，不执行模型推理。完整运行方法见 [EVALUATION.md](EVALUATION.md)。

## 来源与发布

- [GPQA](https://github.com/idavidrein/gpqa)
- [MMMU](https://huggingface.co/datasets/MMMU/MMMU)
- [MultiModalQA](https://github.com/allenai/multimodalqa)

GPQA 官方要求避免在线公开数据样例，因此含 GPQA 的题单、答案、badcase 和归档应保持私有。MultiModalQA 官方仓库未单独提供 LICENSE 文件；公开再分发前需确认授权。各来源的原始说明保存在 `manifests/*.source_README.md`。工具接口的第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
