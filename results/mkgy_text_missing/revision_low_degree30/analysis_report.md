# MKG-Y Low-Degree Text Missing 30% Results Analysis

输入目录：

- `results/mkgy_text_missing/revision_low_degree30`
- `results/retrieval_stats`

## 1. 完整性检查

本轮 low-degree 30% 文本缺失实验共 15 个日志文件，均包含 `RESULT_JSON`：

- `A_seed0/1/2.log`
- `AB_seed0/1/2.log`
- `AB_random_seed0/1/2.log`
- `retrieval_quality_seed0/1/2.log`
- `hpsac_seed0/1/2.log`

固定 mask 已覆盖 seed 0/1/2。虽然部分加载 mask 的日志顶层字段显示 `text_missing_mask_strategy=random`，但 `injection_info.loaded_mask_metadata` 和 A 实验原始 mask 元数据均显示实际 mask 是 `low_degree`。

## 2. Low-Degree Mask 真实性

三组 seed 的 mask 均为 low-degree 策略，注入缺失实体数为 3695，对应 30.0016% 的原本有文本实体。

| Seed | Mask checksum | Available degree mean | Masked degree mean | Masked degree max |
|---|---|---:|---:|---:|
| 0 | `500e660692f0decd75ba57c71d0d5b31ecfc10b05756832f67ea40ff7ae3417f` | 2.7718 | 0.9291 | 2 |
| 1 | `c913edd722b7eb42d8cdbc248b7c16469a13a91c24ebe3ed19e2d6dd8d9b330b` | 2.7718 | 0.9291 | 2 |
| 2 | `db45462ffcb79e45f0cd6d726e223d8765a516db14262f9e3e4d286981497287` | 2.7718 | 0.9291 | 2 |

这可以直接回应审稿意见中“真实文本缺失更可能发生在长尾/低度实体，而不是完全随机缺失”的问题。

## 3. 三 Seed 主结果

| Method | MRR | Hits@1 | Hits@3 | Hits@10 | Missing-text MRR |
|---|---:|---:|---:|---:|---:|
| A: soft missing-text token | 0.3695 ± 0.0007 | 0.3228 ± 0.0013 | 0.3970 ± 0.0005 | 0.4436 ± 0.0012 | 0.3083 ± 0.0026 |
| A+B: KNN retrieval | 0.3680 ± 0.0002 | 0.3222 ± 0.0010 | 0.3940 ± 0.0016 | 0.4418 ± 0.0013 | 0.3065 ± 0.0026 |
| A+B: random-neighbor retrieval | 0.3688 ± 0.0010 | 0.3224 ± 0.0019 | 0.3953 ± 0.0011 | 0.4423 ± 0.0007 | 0.3080 ± 0.0028 |

关键差值：

- A+B KNN vs A：MRR 平均下降 `-0.0015`，missing-text MRR 平均下降 `-0.0018`。
- A+B random vs A：MRR 平均下降 `-0.0007`，missing-text MRR 平均下降 `-0.0003`。
- A+B KNN vs random-neighbor：MRR 平均低 `-0.0008`。

结论：在 MKG-Y low-degree 30% 缺失设定下，A 模块最稳定；B 检索模块没有带来端到端性能收益。论文中应如实表述为：结构检索邻居在结构相关性上明显优于随机，但在该场景下，直接将检索文本注入模型并不稳定，说明检索补偿需要可靠性约束或更强校准。

## 4. 检索质量对照

| Retrieval source | Top-k sim mean | Top-k sim max mean | Relation Jaccard | Relation overlap | Direct train neighbor |
|---|---:|---:|---:|---:|---:|
| Entity-embedding KNN | 0.2745 ± 0.0135 | 0.3778 ± 0.0148 | 0.4165 ± 0.0079 | 0.5892 ± 0.0074 | 0.0126 ± 0.0004 |
| Random text pool | -0.0013 ± 0.0066 | 0.0886 ± 0.0085 | 0.1426 ± 0.0049 | 0.2337 ± 0.0060 | 0.0001 ± 0.0001 |

结论：KNN 检索在结构相似度、关系重合率和直接邻居率上明显强于随机邻居。这能回应审稿人关于“检索邻居是否合理”的质疑。但由于端到端 A+B 性能没有提升，需要在论文中避免把 B 单独描述为稳定有效，应强调其作用依赖后续安全校准或限制。

## 5. HPSAC 统计

HPSAC 参数：

- `lambda_grid = {0.10, 0.20, 0.25, 0.30, 0.40}`
- `alpha_grid = {0.0, 0.1, 0.2, 0.3}`
- `N_min = 30`
- `delta = 0.0002`
- `lock_missing_text = False`

| Seed | Effective level1 groups | Group-specific effective | Fallback effective | Non-fallback both-have-text groups | Test MRR delta vs HPSAC fallback | Missing-text MRR delta vs HPSAC fallback |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 86 | 83 | 3 | 37 | +0.0481 | +0.0697 |
| 1 | 86 | 69 | 17 | 23 | +0.0513 | +0.0734 |
| 2 | 86 | 69 | 17 | 23 | +0.0495 | +0.0703 |

平均：

- Test MRR delta vs HPSAC fallback：`+0.0496 ± 0.0016`
- Missing-text MRR delta vs HPSAC fallback：`+0.0711 ± 0.0020`

注意：`eval_hpsac.py` 的 HPSAC 绝对 MRR 与 `train_dhns_rotate.py` 的直接测试 MRR 不在同一数值口径下，不能直接把 HPSAC 的绝对 MRR 与 A/A+B 表格混排。HPSAC 结果更适合报告为“相对于 HPSAC 内部 fallback policy 的增益”和“group-specific/fallback 统计”，用于回应审稿人对 `delta`、`N_min`、group 回退机制和验证集选择过程的质疑。

## 6. 对审稿意见的覆盖情况

已明显补强：

- 审稿意见 2：增加了 low-degree 实体缺失实验，证明不是只做 random missing。
- 审稿意见 3：固定 mask 已保存，mask checksum、mask scope、seed、缺失实体数均有记录。
- 审稿意见 4：增加了 KNN retrieval 与 random-neighbor retrieval 的检索质量对照。
- 审稿意见 5：HPSAC 的 `N_min`、`delta`、group-specific/fallback 统计可以从日志中报告。
- 审稿意见 6：有 seed 0/1/2 的 mean ± std，可用于统计稳定性呈现。
- 审稿意见 8：日志包含 runtime、GPU memory、checkpoint size 和 retrieval index info。

需要谨慎表述：

- A+B 检索模块没有提升 A，不能写成“检索补偿稳定提高性能”。
- HPSAC 当前应报告相对内部 fallback 的提升，不建议直接和 A/A+B 的 `train_dhns_rotate.py` 绝对指标混排。
- 部分加载 mask 的日志顶层字段显示 `text_missing_mask_strategy=random`，但 `loaded_mask_metadata.mask_strategy=low_degree`；论文或补充材料中应引用原始 mask metadata/checksum，避免误读。

