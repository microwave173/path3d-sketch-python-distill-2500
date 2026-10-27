# Path3D sketch animation · Python-tool distillation · 2,500 tasks

两轮实验合并的数据与脚本：**1,000＋1,500 个任务**，用于 Qwen3.5-9B 的 3D 草图动画 SFT。
这里的 2,500 指任务数，每个任务产生多条模型回复训练样本；包含验证任务。

**训练任务2,375／验证任务125；训练样本43,292／验证样本2,252。**
类别：人物625、动物625、物理/几何500、交通375、日常/可变形375。

## Pipeline

1. 动画 prompt → 总计划，模型决定 **5–9 帧**。
2. prompt＋总计划＋上一帧 → 当前完整 Path3D 帧；首帧没有上一帧。
3. 每个阶段可调用 Python 计算器，批量计算坐标、变换等；同阶段保留工具上下文。
4. 下一帧重新建立上下文，保留总计划和上一帧，不带上一阶段的推理及工具历史。
5. DeepSeek low thinking 生成教师轨迹；按论文版压缩提示词压缩推理。
6. 用压缩推理替换原推理，保留原答案/工具调用；每次 assistant 回复为一个 SFT 样本，仅监督当前回复。

输出是带三维坐标的自定义 **Path3D JSON**，不是原生二维 SVG。

## 文件

| 路径 | 内容 |
|---|---|
| `data/tasks.json` | 全部任务、类别和原始 train/validation 划分 |
| `data/*.jsonl.gz` | 已准备好的 SFT 样本，含压缩推理、工具历史及目标回复 |
| `data/manifest.json` | 数量、分片大小和 SHA256 |
| `pipeline/` | 教师生成、Python 计算器、压缩、Path3D 渲染和 LoRA 测试 |
| `training/` | 数据准备、目标掩码、DDP LoRA 训练 |
| `configs/` | 两轮历史训练参数参考 |

数据中不包含 API key、SSH 配置、模型权重或优化器状态。
已清除 provenance 中的服务器文件位置。原始未压缩推理和全部 API 原始日志不在本次发布中。

## 运行

```bash
pip install -r requirements.txt
python prepare_data.py
export MODEL_PATH=/path/to/Qwen3.5-9B
# 默认是合并数据上从 base 新建 LoRA 的训练入口
torchrun --standalone --nproc_per_node=1 training/train_ddp.py
```

`prepare_data.py` 验证分片哈希、任务去重、样本 ID 和数据划分，然后解压到 `training/`。
模型须预先下载到 `MODEL_PATH`。训练使用 BF16、rank64/alpha128、上下文上限32,768；超长样本已整条剔除，没有截断。
多卡时调整 `training/config.json` 的梯度累积，使 `卡数 × micro_batch × gradient_accumulation` 等于所需有效 batch；默认64。

历史实验是先训1,000条，再加载其 LoRA＋AdamW 状态续训1,500条；这和默认合并重训是两种设置。
续训可在配置中指定 `initial_adapter` 和 `initial_checkpoint_manifest`，清单需含 `epoch`、`step`、`optimizer`。

生成新教师数据时，先将 `pipeline/tasks.json` 换成待生成任务，并同步 `pipeline/config.json` 中任务总数：

```bash
export DEEPSEEK_API_KEY=your_key
python pipeline/batch.py
# 生成并压缩完成后，可重新准备 SFT 数据
python training/prepare.py pipeline
```

测试已训练 LoRA：

```bash
export ADAPTER_PATH=/path/to/adapter
python pipeline/evaluate.py 0 1
```

测试任务见 `pipeline/evaluation_tasks.json`；生成结果在 `pipeline/evaluation3/`。
Python 计算器采用进程、资源限制和语法白名单，不是面向不可信输入的安全沙箱。
本仓库是实验数据和复现入口，未宣称已解决几何一致性或动画物理正确性。
