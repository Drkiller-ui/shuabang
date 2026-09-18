# Qwen3.5-4B 小规模多模态评测集

> **Public repository notice:** GPQA question/answer files are intentionally excluded from the public Git history, as are `all.jsonl` and `smoke_200.jsonl` because they contain embedded GPQA rows. Authorized users should reconstruct those local files from the locked source using the build scripts. Generated ZIP bundles are also excluded because they contain the complete local evaluation data and exceed GitHub's normal single-file limit.

本项目已准备评测数据和可执行评测程序；仓库中的现成归档不包含任何模型跑分。主套件为 **1,202 题**，另有其内部固定的 **200 题快速检查子集**。

评测程序现已包含在项目中。它支持 Qwen3.5-4B 的 vLLM/SGLang OpenAI 兼容接口、显式思考轨迹与最终答案记录、16K 长思考告警和 32K 输出硬限制、循环与答后回退诊断、断点续跑、Docker/Podman 代码沙箱、逐数据集得分和 badcase 导出。Harness 在工具注册层执行固定的数据集权限：只有 MultiModalQA 和 GPQA 注册网页搜索、图片搜索与页面访问；其他任务最多获得离线视觉或沙箱 Python 工具。联网题逐题标为 `open_book_agentic`，不能与闭卷成绩混报。完整使用方法见 [EVALUATION.md](EVALUATION.md)。

| 能力 | 来源 | 数量 | 选择方式 |
|---|---|---:|---|
| 视觉数学推理 | MathLLMs/MathVision，testmini | 304 | 完整官方 mini |
| 多模态知识 | MMMU/MMMU，validation | 200 | 按 30 个学科比例分层 |
| 纯文本通识 | TIGER-Lab/MMLU-Pro，test | 300 | 按学科比例分层 |
| 纯文本编程 | livecodebench/code_generation_lite，v6 增量文件 test6.jsonl | 100 | 按难度 × 比赛月份比例分层 |
| 跨模态多跳问答 | allenai/multimodalqa，dev | 100 | 仅选真实跨模态题，按模态组合 × 题型比例分层 |
| 研究生级科学推理 | idavidrein/gpqa，GPQA Diamond | 198 | 完整官方 Diamond；选项按固定种子稳定打乱 |

MathVision testmini 和 GPQA Diamond 使用对应官方子集的全部题目；其余名称指**本项目自建子集**。不要把整套汇总分标为完整官方榜单成绩。LiveCodeBench 的 `v6` 增量与累计 `release_v6` 不同：本项目使用前者，确切时间范围记录在抽样清单中。

## 使用入口

- `data/mini_eval_v1/questions/all.jsonl`：1,202 道题的模型输入数据。
- `data/mini_eval_v1/questions/smoke_200.jsonl`：200 道快速检查题，来自主套件，不是独立测试集。
- `data/mini_eval_v1/questions/<benchmark>.jsonl`：按能力分开的题目。
- `data/mini_eval_v1/references/<benchmark>.jsonl`：通过 `id` 关联的答案、测试入口和原始记录。
- `data/mini_eval_v1/assets/`：实际图片文件，保持原始分辨率，不重新编码。
- `data/mini_eval_v1/references/livecodebench_tests/`：解码后的公开与隐藏测试用例。
- `data/mini_eval_v1/assets/multimodalqa/`：MultiModalQA 候选图片；100 题共引用 958 张不同图片。
- `data/mini_eval_v1/suite.json`：套件概况。
- `data/mini_eval_v1/manifests/`：源版本、选择的 ID、未使用 ID、分层统计、原始数据说明。
- `data/mini_eval_v1/checksums.json`：数据文件的 SHA-256 和字节数。
- `data/mini_eval_v1/validation_report.json`：执行校验脚本后生成的结果。

复制整个 `data/mini_eval_v1` 目录即可搬到 A100 服务器。内部路径统一使用相对于这个目录的路径，不依赖当前 Windows 盘符。

## 数据格式

每个 `questions` 记录包含：

| 字段 | 含义 |
|---|---|
| `id` | 全局唯一 ID：数据集名加原始 ID |
| `source_id` | 原始题号；LCB 使用平台加题号，避免平台间重名 |
| `source` | Hugging Face 仓库、固定 commit、split、config |
| `question` | 原题文本/原始指令，保留公式和图像占位符 |
| `options` | 按原顺序保留的选项；非选择题为空列表 |
| `images` | 本地输入图片路径列表 |
| `metadata` | 学科、难度、图像映射、starter code 等必要信息 |

这是一种通用数据格式，**并非已经注册到 VLMEvalKit 或 lmms-eval 的任务**。后续接入模型时需要按以下约定组装输入：

1. 只使用 `questions` 数据构造模型消息。`references` 和原始源数据不传给模型。
2. 选择题将 `options` 顺序标为 A、B、C……，不得重排。
3. MMMU 的 `<image 1>`、`<image 2>` 等位置可能出现在题目或选项中。用 `metadata.image_map` 映射到真实图片，保持图文顺序。
4. MathVision 官方图片有时已把多个子图拼成一张、并在图内标注 `<image1>`、`<image2>`。这些标记可能映射到同一张合成图；只传入 `images` 中的一张官方图，保留题目中的子图引用，不要重复插入整张图。不要把路径字符串当成图像输入。
5. LCB 要结合 `metadata.starter_code` 和 `metadata.execution_metadata` 确定函数式或标准输入输出式任务；测试用例保留了 `testtype`。
6. MultiModalQA 只选 `metadata.modalities` 至少包含两种模态的 dev 题。输入保留官方完整候选上下文：10 段文本、1 张表格和 3–15 张候选图片，包含官方干扰项；支持事实、推理链、中间答案等答案侧标注只保留在 `references`。
7. GPQA 使用完整 Diamond 子集。原始 CSV 把正确答案与三个错误答案分列；构建时按题目 ID 和固定种子稳定打乱选项，并在 reference 端重新计算 A-D 标签。不要恢复成固定的“正确答案在第一个字段”顺序。

最简单的本地读取：

```python
import json
from pathlib import Path

root = Path("data/mini_eval_v1")
with (root / "questions/smoke_200.jsonl").open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

first = rows[0]
image_paths = [root / relative for relative in first["images"]]
```

## 抽样与复现

- 固定种子：`20260916`。
- 每层数量按总体占比采用最大余数法分配；每层按 `SHA256(seed|namespace|source_id)` 排序取前若干题。
- 该方法不依赖源文件读取顺序，也不依赖 Python 随机数实现。
- 不根据模型是否答对、答案内容或题目长度筛题；抽样清单保留每层总体数和选择数。
- 四个自建子集的 `*.unused_ids.json` 记录未使用的原始题号，可用于后续独立验收。这里只存清单，没有额外制作或跑分。MultiModalQA 的未使用清单只统计符合真实跨模态条件的 dev 题。
- MathVision 使用完整 testmini，因此该来源内没有剩余题。后续独立验收需要从完整 test 中排除 testmini ID。
- 主套件用于反复选 checkpoint，应视为开发评测；smoke 子集不能当成独立验收。

固定的数据仓库版本见 `config/sources.lock.json`。下载脚本会优先使用锁定 commit，已有缓存与锁不一致时会报错。首次固定版本后不要在同一实验里移动到新的上游版本。

## 重建和校验

Python 3.13 环境中可在项目目录安装构建依赖：

```text
python -m pip install --target .deps -r requirements-data.txt
python scripts/fetch_sources.py
python scripts/build_mini_eval.py
python scripts/finalize_mini_eval.py
python scripts/validate_mini_eval.py
python scripts/package_mini_eval.py
```

下载需要网络；源文件保存在 `data/sources`，已下载的文件可复用。大文件使用可续传的分块下载，并核对每块的范围和长度。构建、归档清单和校验均离线进行。现有数据的本地校验仅需 Python 和 Pillow，不需要 GPU。打包脚本生成 `artifacts/mini_eval_v1.zip`，包含数据、脚本、版本锁及说明，不包含源文件缓存和本机依赖。

校验内容包括：数量与 ID 对齐、题号唯一性、图片可解码、图像占位符映射、选择题答案范围、MultiModalQA 跨模态条件与答案侧字段隔离、代码测试字段与数量、文件校验和、smoke 子集与主集一致性。

**校验不等于完成模型评测**：当前不执行代码题程序，也不验证模型正确率。运行代码评测时应使用隔离执行环境。

## 来源与使用说明

- MathVision：https://huggingface.co/datasets/MathLLMs/MathVision
- MMMU：https://huggingface.co/datasets/MMMU/MMMU
- MMLU-Pro：https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- LiveCodeBench：https://huggingface.co/datasets/livecodebench/code_generation_lite
- MultiModalQA：https://github.com/allenai/multimodalqa
- GPQA：https://github.com/idavidrein/gpqa

保留了固定版本的原始数据卡，见 `manifests/*.source_README.md`。各来源的许可分别适用；本项目没有把汇总数据重新授权。LCB 数据卡的许可标签为 `cc`，其加载脚本写为 `MIT License`，二者不一致，保留原始说明供后续公开发布前核对。MultiModalQA 官方仓库当前没有单独的 `LICENSE` 文件，公开或再分发前需要向上游确认授权。GPQA 官方要求避免在线公开数据样例以降低泄露风险，因此包含 GPQA 的本地数据目录和压缩包应保持私有。所有评测数据和答案都不应混入训练集。

Harness 的工具接口参考并修改自 Apache-2.0 的 [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL)。固定 revision、改动范围和完整许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 `licenses/OpenSearch-VL-Apache-2.0.txt`。
