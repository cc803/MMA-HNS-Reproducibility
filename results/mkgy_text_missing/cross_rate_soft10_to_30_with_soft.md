# Cross-Rate: Soft Token 10% Checkpoint → 30% Missing (WITH Soft Text)

**命令**:
```bash
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --inject-text-missing-rate 0.3 \
  --text-missing-mask-path masks/MKG-Y_random_text30_seed0.pt \
  --use-soft-missing-text --subset-eval --test \
  --checkpoint-path checkpoint/revision_mkgy_soft_text10_seed0.ckpt
```

**说明**: 10% 缺失训练的 Soft Token checkpoint，在 30% 缺失下启用 soft-missing-text 做 test-only 评估。

**运行时间**: 2026-07-25 | 仅评估（~32s）| GPU 显存 ~0.96 GB

---

## 整体指标

| 指标 | 10% 训练+评估 | 30% + soft token (本实验) | 30% legacy (无 soft) |
|---|---|---|---|
| MRR | 0.3715 | **0.3241** | 0.3589 |
| MR | 1338.77 | 1594.08 | 1392.41 |
| hit@10 | 0.4495 | **0.4183** | 0.4476 |
| hit@3 | 0.3997 | 0.3554 | 0.3898 |
| hit@1 | 0.3231 | **0.2685** | 0.3057 |

## Subset 指标（按文本可用性）

| 子集 | 查询数 | MRR | hit@10 | hit@1 |
|---|---|---|---|---|
| head_or_tail_missing_text | 3680 | 0.2904 | 0.3921 | 0.2313 |
| head_missing_text | 2470 | 0.2778 | 0.3753 | 0.2202 |
| tail_missing_text | 2256 | 0.3052 | 0.4078 | 0.2447 |
| head_or_tail_injected_missing_text | 2268 | 0.2570 | 0.3792 | 0.1874 |
| head_and_tail_have_text | 1646 | 0.3993 | 0.4769 | 0.3518 |

## 关键配置

| 参数 | 值 |
|---|---|
| 模式 | test-only |
| 训练缺失率 | 10% |
| 评估缺失率 | 30% (6,379 missing) |
| soft-missing-text | **启用** (text_mode=soft_token) |
| missing_text_token_norm | 0.8209 (训练所得) |
| Checkpoint | `checkpoint/revision_mkgy_soft_text10_seed0.ckpt` |
| Mask | `masks/MKG-Y_random_text30_seed0.pt` (loaded) |

> ⚠️ 启用 soft-missing-text 后 30% 缺失率下 MRR 反而比 legacy 模式更低（0.3241 vs 0.3589），说明在训练时未见的更高缺失比例下，训练到的 soft token 反而有负迁移效应。
