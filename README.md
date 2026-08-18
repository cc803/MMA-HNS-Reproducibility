# MMA-HNS：面向文本模态缺失的 DHNS

[![Preprint](https://img.shields.io/badge/Preprint-2025-EE4C2C)](https://arxiv.org/abs/2501.15393)
[![DASFAA](https://img.shields.io/badge/DASFAA-2025-B57EDC)](https://dasfaa2025.github.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

本仓库包含原始 **DHNS**（Diffusion-based Hierarchical Negative Sampling）以及面向文本模态缺失的扩展方法 **MMA-HNS**。当前 MMA-HNS 主实验采用 RotatE，在 **MKG-W** 和 **MKG-Y** 上进行多模态知识图谱补全（MMKGC）训练与 filtered link prediction 评测。

## MMA-HNS 是什么

原始 DHNS 使用 Diffusion-based Hierarchical Embedding Generation（DiffHEG）生成分层困难负样本，并使用 Negative Triple-Adaptive Training（NTAT）进行训练。MMA-HNS 在这条训练路径上进一步处理实体文本缺失问题，由三个递进组件组成：

| 组件 | 代码开关/入口 | 作用 |
|---|---|---|
| A：learnable missing-text token | `--use-soft-missing-text` | 用可学习 token 替代缺失实体的零文本表示。 |
| B：retrieval compensation | `--use-retrieval-missing-text` | 从具有文本的结构相似实体中检索文本原型，为 missing-text token 增加实体相关的补偿信息。 |
| C：HPSAC | `eval_hpsac.py` | 在验证集上按层级选择安全的检索强度和 A/B 分数组合；不增加新的训练网络。 |

因此，本仓库中的实验命名对应为：

- `DHNS`：原始 RotatE + DHNS 路径；
- `DHNS + A`：soft missing-text token；
- `DHNS + A+B`：soft token + structural KNN retrieval；
- `DHNS + A+B+C`：完整 MMA-HNS，即 A+B 加 HPSAC 校准。

当前主线研究的是**文本模态缺失**。代码也包含图像缺失和其他诊断开关，但它们不属于本文档中的默认 MMA-HNS 配置。

## 数据集与目录

项目使用 MKG-W 和 MKG-Y：

| 数据集 | 实体数 | 关系数 | Train | Valid | Test |
|---|---:|---:|---:|---:|---:|
| MKG-W | 15,000 | 169 | 34,196 | 4,276 | 4,274 |
| MKG-Y | 15,000 | 28 | 21,310 | 2,665 | 2,663 |

数据和预训练模态特征应按以下方式组织：

```text
DHNS-main/
├── benchmarks/
│   ├── MKG-W/
│   │   ├── entity2id.txt
│   │   ├── relation2id.txt
│   │   ├── train2id.txt
│   │   ├── valid2id.txt
│   │   ├── test2id.txt
│   │   └── type_constrain.txt
│   └── MKG-Y/
│       └── ...
├── embeddings/
│   ├── MKG-W-visual.pth
│   ├── MKG-W-textual.pth
│   ├── MKG-Y-visual.pth
│   └── MKG-Y-textual.pth
├── masks/
├── checkpoint/
└── results/
```

`.pth` 中实体特征的行顺序必须与对应数据集的 `entity2id.txt` 一致。原始视觉/文本特征可从 DHNS 使用的 [Google Drive](https://drive.google.com/drive/folders/1UJSfnb8DEx2s-k8zaQx1fWUw5f45GBpI?usp=sharing) 下载并放入 `embeddings/`。

## 运行环境

推荐环境：

- Ubuntu/Linux 或 Windows + WSL2；
- Python 3.8；
- PyTorch 2.4.1 + CUDA 12.1；
- NVIDIA GPU；CPU 可用 `--no-gpu`，但完整评测会很慢；
- `g++`，用于编译数据加载器的 `mmkgc/release/Base.so`。

创建环境：

```bash
conda create -n mma_hns python=3.8 -y
conda activate mma_hns
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

`requirements.txt` 中的核心版本为：

```text
numpy==1.23.3
scikit_learn==1.1.2
torch==2.4.1+cu121
tqdm==4.66.5
```

仓库已经包含 Linux 版 `mmkgc/release/Base.so`。如果它与当前机器不兼容或不存在，可重新编译：

```bash
cd mmkgc
bash make.sh
cd ..
```

## Missingness mask 如何组织

### 实体级统一协议

人工文本缺失使用一个长度等于实体总数的布尔 mask：`mask[e] = True` 表示实体 `e` 的文本被额外置零。协议具有以下约束：

- 粒度为 `entity`，不是 triple-level 或 query-level；
- 同一 mask 在 train/valid/test 中保持不变；
- 注入比例只针对**原本具有文本特征**的实体计算，而不是针对全部实体；
- 加载的 mask 不允许选择原本就没有文本的实体；
- 同一 dataset/rate/seed 下，DHNS、A、A+B 和 HPSAC 必须复用同一个 mask；
- 每次运行会在 `RESULT_JSON.injection_info.mask_checksum_sha256` 中记录校验值。

支持三种内置策略：

- `random`：从原本具有文本的实体中按 seed 无放回采样；
- `low_degree`：根据 `train2id.txt` 的实体度数，从低度实体开始 mask，同度实体使用 seed 打破并列；
- `high_degree`：从高度实体开始 mask。

支持 `.pt`、`.pth` 和 `.json`。推荐的 `.pt` 文件包含：

```text
mask                 # bool tensor，长度为实体数
masked_entity_ids    # 被 mask 的实体 ID
metadata             # dataset/rate/seed/strategy/checksum 等信息
```

推荐命名规则：

```text
masks/<experiment>/<DATASET>_<strategy>_text<RATE>_seed<SEED>.pt
```

例如：

```text
masks/mkgw_main_table/MKG-W_random_text30_seed0.pt
masks/revision/MKG-Y_low_degree_text30_seed0.pt
```

先由第一个方法生成并保存 mask：

```bash
python train_dhns_rotate.py \
  --dataset MKG-W \
  --seed 0 \
  --inject-text-missing-rate 0.3 \
  --text-missing-mask-strategy random \
  --save-text-missing-mask-path masks/MKG-W_random_text30_seed0.pt \
  --checkpoint-path checkpoint/MKG-W_dhns_text30_seed0.ckpt
```

其他方法读取同一文件：

```bash
python train_dhns_rotate.py \
  --dataset MKG-W \
  --seed 0 \
  --inject-text-missing-rate 0.3 \
  --text-missing-mask-path masks/MKG-W_random_text30_seed0.pt \
  --use-soft-missing-text \
  --checkpoint-path checkpoint/MKG-W_A_text30_seed0.ckpt
```

不要为每个方法分别随机生成 mask；即使 rate 和 seed 相同，也应显式加载已经保存的文件，并核对 checksum。

## 主实验协议（Main experimental protocol）

MMA-HNS 主实验在 MKG-W 与 MKG-Y 上统一采用以下配置。主实验的 A+B retrieval weight 保持 `0.25`，请勿在复现主表时将其改为 `1.0`：

| 项目 | 主实验值 |
|---|---|
| A+B retrieval weight（`--retrieval-mix-weight`） | `0.25` |
| HPSAC fallback（`λ`, `α`） | `(0.25, 0.0)` |
| λ grid（`--lambda-grid`） | `{0.10, 0.20, 0.25, 0.30, 0.40}` |

下面两节分别说明 retrieval 与 HPSAC 的参数语义。权重 `1.0` 仅用于后文「Additional same-scale low-degree analysis」这一独立补充分析，不属于主实验协议。

## Retrieval 设置

主实验的 A+B retrieval 配置为：

| 参数 | 主实验值 | 含义 |
|---|---:|---|
| `--retrieval-source` | `entity_embedding_knn` | 使用结构实体嵌入进行 KNN。 |
| `--retrieval-topk` | `5` | 每个目标实体取 5 个近邻。 |
| `--retrieval-pool-size` | `512` | 在 KNN 前将具有文本的候选实体池确定性限制到 512 个。设为 `0` 表示使用全部候选。 |
| `--retrieval-mix-weight` | `0.25` | A+B 训练时加入检索文本原型的权重。 |

默认 retrieval 流程如下：

1. 从当前 mask 下仍具有文本的实体构造候选集合；
2. 候选数超过 512 时，按实体 ID 顺序进行确定性等距取样；
3. 使用归一化结构实体嵌入的余弦相似度检索 top-5；
4. 对 top-5 相似度做 softmax，并加权聚合这些实体的投影文本特征；
5. 对缺失文本实体使用

   ```text
   missing_text = learnable_token + λ × retrieved_text_prototype
   ```

代码还提供 `random_text_pool` 作为确定性的随机近邻对照。默认 retrieval 不依赖外部向量数据库，索引只保存候选实体 ID，并复用模型中的结构和文本嵌入。

注意：`train_dhns_rotate.py` 的通用 `--retrieval-mix-weight` 默认值是 `1.0`，而 MMA-HNS 主实验脚本会**显式设置为 `0.25`**。复现实验时不要省略该参数。

## HPSAC 设置

HPSAC（Hierarchical Pareto-Safe Adaptive Calibration）是完整 MMA-HNS 的组件 C。它读取两个已经训练好的 checkpoint：

- checkpoint A：soft missing-text token；
- checkpoint B：soft token + retrieval。

对每个候选 `λ`，B 路径内部使用相应的 retrieval mix weight；随后使用 `α` 融合 A/B 的 link prediction 分数：

```text
score = (1 - α) × score_B(λ) + α × score_A
```

主实验配置：

| 符号 | 命令行参数 | 主实验值 | 作用 |
|---|---|---|---|
| λ | `--lambda-grid` | `{0.10, 0.20, 0.25, 0.30, 0.40}` | 控制 B 路径中的 retrieval 补偿强度。 |
| α | `--alpha-grid` | `{0.0, 0.1, 0.2, 0.3}` | 控制最终分数中 A 路径的占比。 |
| δ | `--safe-delta` | `0.0002` | 候选配置相对 fallback 的验证集 MRR 至少提高 δ 才能被接受。 |
| Nmin | `--min-group-queries` | `30` | 一个校准组至少需要 30 个验证查询；否则回退。 |
| C-lock | `--lock-missing-text` | 开启 | missing-text 组固定使用 fallback，不允许自适应搜索改变其配置。 |

默认 fallback 为 `λ=0.25, α=0.0`，即直接使用固定 retrieval 权重的 B 路径。项目实验中的 **C-lock** 对应代码开关 `--lock-missing-text`：只对 `both_have_text` 组开放自适应校准；只要 head 或 tail 缺少文本，就保持 fallback。因此开启后，missing-text 子集的 HPSAC 分数应与 B-v1 fallback 完全一致。

HPSAC 按以下层级从细到粗回退：

1. `relation × prediction side × text state`；
2. `prediction side × text state`；
3. `text state`；
4. 全局 fallback。

参数选择只使用 validation split，test split 仅用于最终报告。默认行为使用完整 validation 进行 HPSAC 校准；若需要严格拆分验证用途，可额外添加 `--separate-calibration-split --validation-split-seed 0`，将 validation 确定性拆为两半。

> `train_dhns_rotate.py --alpha 0.002` 中的 `alpha` 是训练优化器学习率；`eval_hpsac.py --alpha-grid ...` 中的 `α` 是 A/B 分数融合系数，二者不是同一个参数。

## 训练与测试

以下命令应在仓库根目录运行。

### 一键运行 MKG-W 或 MKG-Y 主流程

下列脚本分别运行 3 个 seed 的 DHNS、hard mask、A、A+B、A+B+C，并汇总结果：

```bash
bash run_phase1_mkgw.sh
bash run_phase1_mkgy.sh
```

默认训练配置包括 400 epochs、batch size 2000、每个正样本 64 个负实体、RotatE dimension 512、margin 6.0、模型学习率 0.002、生成器学习率 0.002、`mu=0.01` 和每 epoch 10 次生成器更新。

输出位置：

```text
checkpoint/phase1_mkgw/
checkpoint/phase1_mkgy/
results/phase1_mkgw/
results/phase1_mkgy/
```

### 完整的固定 mask 训练示例

下面以 MKG-Y、30% random missing、seed 0 为例。把 `DATASET` 改为 `MKG-W` 即可运行另一数据集。

```bash
DATASET=MKG-Y
SEED=0
RATE=0.3
RATE_TAG=text30
MASK=masks/${DATASET}_random_${RATE_TAG}_seed${SEED}.pt
OUT=checkpoint/manual_${DATASET}_${RATE_TAG}_seed${SEED}

mkdir -p masks checkpoint results
```

1. 训练 DHNS，并生成本组实验唯一的共享 mask：

```bash
python train_dhns_rotate.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --inject-text-missing-rate "$RATE" \
  --text-missing-mask-strategy random \
  --save-text-missing-mask-path "$MASK" \
  --subset-eval \
  --checkpoint-path "${OUT}_dhns.ckpt" \
  --result-json-output-path "results/manual_${DATASET}_${RATE_TAG}_dhns_seed${SEED}.json"
```

2. 训练组件 A：

```bash
python train_dhns_rotate.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --inject-text-missing-rate "$RATE" \
  --text-missing-mask-path "$MASK" \
  --use-soft-missing-text \
  --subset-eval \
  --checkpoint-path "${OUT}_A.ckpt" \
  --result-json-output-path "results/manual_${DATASET}_${RATE_TAG}_A_seed${SEED}.json"
```

3. 训练组件 A+B：

```bash
python train_dhns_rotate.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --inject-text-missing-rate "$RATE" \
  --text-missing-mask-path "$MASK" \
  --use-soft-missing-text \
  --use-retrieval-missing-text \
  --retrieval-source entity_embedding_knn \
  --retrieval-topk 5 \
  --retrieval-pool-size 512 \
  --retrieval-mix-weight 0.25 \
  --subset-eval \
  --checkpoint-path "${OUT}_AB.ckpt" \
  --result-json-output-path "results/manual_${DATASET}_${RATE_TAG}_AB_seed${SEED}.json"
```

4. 在 validation 上校准 HPSAC，并在 test 上评测完整 MMA-HNS：

```bash
python eval_hpsac.py \
  --dataset "$DATASET" \
  --checkpoint-a "${OUT}_A.ckpt" \
  --checkpoint-b "${OUT}_AB.ckpt" \
  --inject-text-missing-rate "$RATE" \
  --text-missing-mask-path "$MASK" \
  --retrieval-source entity_embedding_knn \
  --retrieval-topk 5 \
  --retrieval-pool-size 512 \
  --lambda-grid 0.10,0.20,0.25,0.30,0.40 \
  --alpha-grid 0.0,0.1,0.2,0.3 \
  --min-group-queries 30 \
  --safe-delta 0.0002 \
  --lock-missing-text \
  --subset-eval \
  2>&1 | tee "results/manual_${DATASET}_${RATE_TAG}_hpsac_seed${SEED}.log"
```

HPSAC 是评测期校准模块，因此不会另存一个 A+B+C checkpoint；复现完整 MMA-HNS 测试时，需要保留 A、A+B 两个 checkpoint 和对应 mask。

### 仅测试已有 A+B checkpoint

`--test` 会跳过训练。测试时必须重复训练该 checkpoint 时使用的模型开关，并加载同一 mask：

```bash
python train_dhns_rotate.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --test \
  --inject-text-missing-rate "$RATE" \
  --text-missing-mask-path "$MASK" \
  --use-soft-missing-text \
  --use-retrieval-missing-text \
  --retrieval-source entity_embedding_knn \
  --retrieval-topk 5 \
  --retrieval-pool-size 512 \
  --retrieval-mix-weight 0.25 \
  --subset-eval \
  --checkpoint-path "${OUT}_AB.ckpt"
```

### MKG-W 10%/30%/50% random missing 主表

`run_mkgw_main_table.py` 会运行 seeds 0/1/2，为同一 rate/seed 的所有方法复用固定 mask，并输出均值、标准差和配对 t-test：

```bash
python run_mkgw_main_table.py
```

快速连通性检查可减少到两个 seed、一个 missing rate 和一个 epoch：

```bash
python run_mkgw_main_table.py --seeds 0 1 --rates 0.3 --train-times 1
```

默认输出：

```text
masks/mkgw_main_table/
checkpoint/mkgw_main_table/
results/mkgw_main_table/main_table_runs.csv
results/mkgw_main_table/main_table_summary.csv
results/mkgw_main_table/main_table_result.json
results/mkgw_main_table/main_table.md
```

### Low-degree missingness sweep

该脚本默认在 MKG-W/MKG-Y、10%/30%/50%、seeds 0/1/2 上运行 low-degree missingness 实验：

```bash
bash run_revision_low_degree_missing_sweep.sh
```

例如只运行 MKG-Y、30%、seed 0：

```bash
DATASETS=MKG-Y RATES=0.3 SEEDS=0 bash run_revision_low_degree_missing_sweep.sh
```

## Additional same-scale low-degree analysis

在主实验协议之外，本仓库单独提供一个同规模的 low-degree 补充分析，用于在主实验 A+B retrieval weight `0.25` 之外对照更高检索权重下的表现：

| 项目 | 该分析取值 |
|---|---|
| Dataset | `MKG-Y` |
| Missingness | `30% low-degree`（`low_degree` 策略，seed 0/1/2） |
| A+B retrieval weight（`--retrieval-mix-weight`） | `1.0` |
| HPSAC fallback（`λ`, `α`） | `(1.0, 0.0)` |
| λ grid（`--lambda-grid`） | `{0.10, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80, 1.00}` |

复现要点：

- 复用已有的 mask：`masks/MKG-Y_low_degree_text30_seed<SEED>.pt`；
- A+B 训练时显式设置 `--retrieval-mix-weight 1.0`；
- HPSAC 校准时使用 `--lambda-grid 0.10,0.20,0.25,0.30,0.40,0.60,0.80,1.00`，fallback 为 `λ=1.0, α=0.0`（即 `--fallback-lambda 1.0`）；
- 其余参数（`--retrieval-source entity_embedding_knn`、`--retrieval-topk 5`、`--retrieval-pool-size 512`、`--safe-delta 0.0002`、`--min-group-queries 30`、`--lock-missing-text`）与主实验保持一致。

该分析仅改变 retrieval weight 与 λ grid 的取值，不影响主实验协议；主 README 其余部分仍以 A+B retrieval weight `0.25`、HPSAC fallback `(0.25, 0.0)` 为准。

## 结果与复现检查

训练和评测脚本都会在标准输出末尾打印一行：

```text
RESULT_JSON: {...}
```

训练入口还可以通过 `--result-json-output-path <path>` 直接保存 JSON。重要字段包括：

- `overall_metrics`：filtered MRR、MR、Hits@1/3/10；
- `subset_metrics`：missing-text 和 both-have-text 子集指标；
- `injection_info`：mask strategy、source、path、scope、实际缺失数和 SHA-256 checksum；
- `retrieval_*`：top-k、candidate pool、retrieval source 和统计信息；
- `runtime_cost`、`gpu_memory_cost`、`storage_cost`；
- HPSAC 的 `lambda_grid`、`alpha_grid`、`safe_delta`、`min_group_queries`、`lock_missing_text`、group records 和 fallback 统计。

公平比较 DHNS 与 MMA-HNS 前，应至少核对：dataset、missing rate、seed、mask path 和 `mask_checksum_sha256` 完全一致。

## 主要入口

| 文件 | 用途 |
|---|---|
| `train_dhns_rotate.py` | DHNS、A、A+B 的训练和单 checkpoint 测试。 |
| `eval_hpsac.py` | A+B+C/HPSAC 的验证集校准和测试集评测。 |
| `missing_text_protocol.py` | entity-level mask 的生成、加载、保存和 checksum。 |
| `run_phase1_mkgw.sh` / `run_phase1_mkgy.sh` | 两个数据集的 3-seed 主流程。 |
| `run_mkgw_main_table.py` | MKG-W random 10%/30%/50% 的 DHNS vs MMA-HNS 主表。 |
| `run_revision_low_degree_missing_sweep.sh` | MKG-W/MKG-Y low-degree missingness sweep。 |
| `summarize_results.py` | 汇总含 `RESULT_JSON` 的实验日志。 |

## Citation

如果使用原始 DHNS 代码或方法，请引用：

```bibtex
@misc{niu2025dhns,
  author        = {Guanglin Niu and Xiaowei Zhang},
  title         = {Diffusion-based Hierarchical Negative Sampling for Multimodal Knowledge Graph Completion},
  archivePrefix = {arXiv},
  year          = {2025},
  eprint        = {2501.15393},
  primaryClass  = {cs.AI}
}
```
