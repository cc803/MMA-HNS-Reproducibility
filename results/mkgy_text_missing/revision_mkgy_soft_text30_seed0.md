# Soft Token A Baseline — MKG-Y text missing 30%

**命令**:
```bash
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --inject-text-missing-rate 0.3 \
  --text-missing-mask-path masks/MKG-Y_random_text30_seed0.pt \
  --use-soft-missing-text --subset-eval \
  --checkpoint-path checkpoint/revision_mkgy_soft_text30_seed0.ckpt
```

**运行时间**: 2026-07-25 | 训练约 2h08min | GPU 显存 ~9.2 GB 分配 / 11.9 GB 预留

---

## 整体指标

| 指标 | 值 |
|---|---|
| MRR | **0.3711** |
| MR | 1350.41 |
| hit@10 | **0.4461** |
| hit@3 | 0.3979 |
| hit@1 | 0.3244 |

## Subset 指标（按文本可用性）

| 子集 | 查询数 | MRR | hit@10 | hit@1 |
|---|---|---|---|---|
| head_or_tail_missing_text | 3680 | 0.3609 | 0.4334 | 0.3147 |
| head_missing_text | 2470 | 0.3473 | 0.4178 | 0.3012 |
| tail_missing_text | 2256 | 0.3727 | 0.4464 | 0.3276 |
| head_or_tail_injected_missing_text | 2268 | 0.3743 | 0.4497 | 0.3258 |
| head_and_tail_have_text | 1646 | 0.3939 | 0.4745 | 0.3463 |

## 关键配置

| 参数 | 值 |
|---|---|
| 方法 | Soft missing-text token (A baseline) |
| 注入缺失比例 | 30% (3,695 + 2,684 天然 = 6,379 缺失) |
| missing_text_token_norm | 0.7407 |
| Checkpoint | `checkpoint/revision_mkgy_soft_text30_seed0.ckpt` |
| Mask | `masks/MKG-Y_random_text30_seed0.pt` (loaded) |

---

## 与 Soft Token 10% 对比

| 指标 | 10% 训练 | 30% 训练 | 10%→30% cross (soft) |
|---|---|---|---|
| MRR | 0.3715 | 0.3711 | 0.3241 |
| hit@10 | 0.4495 | 0.4461 | 0.4183 |
| hit@1 | 0.3231 | 0.3244 | 0.2685 |
| missing_text MRR | 0.3420 | **0.3609** | 0.2904 |
| injected_missing MRR | 0.3702 | **0.3743** | 0.2570 |

> 直接在 30% 下训练，整体 MRR 与 10% 基本持平（0.3711 vs 0.3715），但缺失文本子集 MRR 大幅优于跨缺失率泛化（0.3609 vs 0.2904），说明 soft token 在训练时见过的缺失率下有效，但不具备泛化到更高缺失率的能力。
