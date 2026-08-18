# Retrieval B-v1 — MKG-Y text missing 10% (KNN Retrieval-Augmented)

**命令**:
```bash
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --inject-text-missing-rate 0.1 \
  --text-missing-mask-path masks/MKG-Y_random_text10_seed0.pt \
  --use-soft-missing-text --use-retrieval-missing-text --subset-eval \
  --checkpoint-path checkpoint/revision_mkgy_retrieval_text10_seed0.ckpt
```

**说明**: 在 Soft Token A 基线基础上，使用 KNN 检索从结构相似实体聚合文本原型来增强缺失文本表示（B-v1 方法）。Mask 复用 Soft Token 实验的同一文件。

**运行时间**: 2026-07-25 | 训练约 2h17min | GPU 显存 ~10.2 GB 分配 / 15.5 GB 预留

---

## 整体指标

| 指标 | 值 |
|---|---|
| MRR | **0.3690** |
| MR | 1298.83 |
| hit@10 | **0.4487** |
| hit@3 | 0.3969 |
| hit@1 | 0.3201 |

## Subset 指标（按文本可用性）

| 子集 | 查询数 | MRR | hit@10 | hit@1 |
|---|---|---|---|---|
| head_or_tail_missing_text | 2610 | 0.3392 | 0.4146 | 0.2935 |
| head_missing_text | 1646 | 0.3125 | 0.3852 | 0.2667 |
| tail_missing_text | 1392 | 0.3769 | 0.4526 | 0.3326 |
| head_or_tail_injected_missing_text | 866 | 0.3578 | 0.4376 | 0.3106 |
| head_and_tail_have_text | 2716 | 0.3976 | 0.4816 | 0.3457 |

## Retrieval 统计

| 参数 | 值 |
|---|---|
| topk | 5 |
| pool_size | 512 / 11,084 |
| avg topk_similarity_mean | 0.3107 |
| avg topk_similarity_max | 0.4500 |
| prototype_text_agg_mean_norm | 0.9255 |
| fallback_ratio | 0.0 |

## 关键配置

| 参数 | 值 |
|---|---|
| 方法 | Retrieval-augmented missing text (B-v1) |
| 注入缺失比例 | 10% (1,232 / 12,316 entities) |
| 天然缺失文本 | 2,684 entities |
| 总计缺失文本 | 3,916 entities |
| Missing text token norm | 0.0922 |
| Checkpoint | `checkpoint/revision_mkgy_retrieval_text10_seed0.ckpt` |
| Mask | `masks/MKG-Y_random_text10_seed0.pt` (复用 Soft Token 实验) |
