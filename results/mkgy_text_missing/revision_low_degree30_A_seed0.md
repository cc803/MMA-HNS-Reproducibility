# Soft Token A Baseline — MKG-Y low_degree text missing 30%

**命令**:
```bash
python -u train_dhns_rotate.py --dataset MKG-Y --seed 0 \
  --inject-text-missing-rate 0.3 \
  --text-missing-mask-strategy low_degree \
  --save-text-missing-mask-path masks/MKG-Y_low_degree_text30_seed0.pt \
  --use-soft-missing-text --subset-eval \
  --checkpoint-path checkpoint/revision_mkgy_low_degree_text30_A_seed0.ckpt
```

**说明**: Soft Token A 基线，**low_degree** 策略（优先遮蔽低度实体），30% 文本缺失。

**运行时间**: 2026-07-26 | 训练约 2h48min | GPU 显存 ~9.2 GB

---

## 整体指标

| 指标 | 值 |
|---|---|
| MRR | **0.3692** |
| MR | 1082.90 |
| hit@10 | **0.4431** |
| hit@3 | 0.3967 |
| hit@1 | 0.3220 |

## Subset 指标

| 子集 | 查询数 | MRR | hit@10 | hit@1 |
|---|---|---|---|---|
| head_or_tail_missing_text | 3712 | 0.3072 | 0.3658 | 0.2680 |
| head_missing_text | 2300 | 0.3080 | 0.3657 | 0.2691 |
| tail_missing_text | 2624 | 0.2793 | 0.3335 | 0.2431 |
| head_or_tail_injected_missing_text | 2396 | 0.2675 | 0.3159 | 0.2354 |
| head_and_tail_have_text | 1614 | **0.5120** | **0.6208** | **0.4461** |

## Mask 统计

| 参数 | 值 |
|---|---|
| 策略 | low_degree |
| 注入缺失 | 30% (3,695 entities) |
| 被遮蔽实体 degree | max=2, mean=0.93 |
| 全部有文本实体 degree | max=78, mean=2.77 |

## Random vs Low_Degree 30% 对比

| 指标 | Random 30% | Low_Degree 30% |
|---|---|---|
| MRR | **0.3711** | 0.3692 |
| hit@10 | **0.4461** | 0.4431 |
| missing_text MRR | **0.3609** | 0.3072 |
| have_text MRR | 0.3939 | **0.5120** |
| injected_missing MRR | **0.3743** | 0.2675 |

> Low_degree 遮蔽下，缺失文本子集显著恶化（0.3072 vs 0.3609），但有文本子集大幅提升（0.5120 vs 0.3939）。因为低度实体结构信息少，丢失文本后更难预测；而保留文本的高度实体预测更容易。

## 关键配置

| 参数 | 值 |
|---|---|
| missing_text_token_norm | 0.7880 |
| Checkpoint | `checkpoint/revision_mkgy_low_degree_text30_A_seed0.ckpt` |
| Mask | `masks/MKG-Y_low_degree_text30_seed0.pt` |
