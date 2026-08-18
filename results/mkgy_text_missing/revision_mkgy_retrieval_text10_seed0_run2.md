# Retrieval B-v1 Run 2 — MKG-Y text missing 10%

**命令**:
```bash
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --inject-text-missing-rate 0.1 \
  --text-missing-mask-path masks/MKG-Y_random_text10_seed0.pt \
  --use-soft-missing-text --use-retrieval-missing-text --subset-eval \
  --checkpoint-path checkpoint/revision_mkgy_retrieval_text10_seed0.ckpt
```

**说明**: Retrieval B-v1 第 2 次运行（与第 1 次相同命令，不同训练结果）。

**运行时间**: 2026-07-25 | 训练约 4h09min | GPU 显存 ~10.2 GB 分配 / 15.5 GB 预留

---

## 整体指标

| 指标 | Soft Token A | Retrieval Run1 | Retrieval Run2 (本实验) |
|---|---|---|---|
| MRR | 0.3715 | 0.3690 | **0.3716** |
| MR | 1338.77 | 1298.83 | 1289.04 |
| hit@10 | 0.4495 | 0.4487 | **0.4474** |
| hit@3 | 0.3997 | 0.3969 | 0.3967 |
| hit@1 | 0.3231 | 0.3201 | **0.3252** |

## Subset 指标（按文本可用性）

| 子集 | 查询数 | MRR | hit@10 | hit@1 |
|---|---|---|---|---|
| head_or_tail_missing_text | 2610 | 0.3423 | 0.4134 | 0.2989 |
| head_missing_text | 1646 | 0.3182 | 0.3858 | 0.2764 |
| tail_missing_text | 1392 | 0.3785 | 0.4526 | 0.3348 |
| head_or_tail_injected_missing_text | 866 | 0.3663 | 0.4353 | 0.3245 |
| head_and_tail_have_text | 2716 | 0.3999 | 0.4801 | 0.3505 |

## Retrieval 统计

| 参数 | 值 |
|---|---|
| topk | 5 |
| pool_size | 512 / 11,084 |
| avg topk_similarity_mean | 0.3106 |
| prototype_text_agg_mean_norm | 0.9391 |
| fallback_ratio | 0.0 |

## 关键配置

| 参数 | 值 |
|---|---|
| 方法 | Retrieval B-v1 (第 2 次) |
| 注入缺失 | 10% (1,232 + 2,684 天然 = 3,916) |
| missing_text_token_norm | 0.0915 |
| Checkpoint | `checkpoint/revision_mkgy_retrieval_text10_seed0.ckpt` |
| Mask | `masks/MKG-Y_random_text10_seed0.pt` |
