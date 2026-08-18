# Cross-Rate: Soft Token 10% Checkpoint → 50% Missing (WITH Soft Text)

**命令**:
```bash
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --inject-text-missing-rate 0.5 \
  --text-missing-mask-path masks/MKG-Y_random_text50_seed0.pt \
  --use-soft-missing-text --subset-eval --test \
  --checkpoint-path checkpoint/revision_mkgy_soft_text10_seed0.ckpt
```

**说明**: 10% 缺失训练的 Soft Token checkpoint，在 50% 缺失下启用 soft-missing-text 做 test-only 评估。

**运行时间**: 2026-07-25 | 仅评估（~32s）| GPU 显存 ~0.96 GB

---

## 整体指标

| 指标 | 10% 训练+评估 | 50% + soft (本实验) | 50% legacy |
|---|---|---|---|
| MRR | 0.3715 | **0.2893** | 0.3589 |
| MR | 1338.77 | 1856.37 | 1392.41 |
| hit@10 | 0.4495 | **0.3992** | 0.4476 |
| hit@3 | 0.3997 | 0.3205 | 0.3898 |
| hit@1 | 0.3231 | **0.2291** | 0.3057 |

## Subset 指标（按文本可用性）

| 子集 | 查询数 | MRR | hit@10 | hit@1 |
|---|---|---|---|---|
| head_or_tail_missing_text | 4474 | 0.2665 | 0.3840 | 0.2032 |
| head_missing_text | 3328 | 0.2625 | 0.3753 | 0.1992 |
| tail_missing_text | 3104 | 0.2862 | 0.4050 | 0.2223 |
| head_or_tail_injected_missing_text | 3430 | 0.2413 | 0.3746 | 0.1711 |
| head_and_tail_have_text | 852 | 0.4093 | 0.4789 | 0.3650 |

## 关键配置

| 参数 | 值 |
|---|---|
| 模式 | test-only |
| 训练缺失率 | 10% |
| 评估缺失率 | 50% (8,842 missing, 58.9%) |
| soft-missing-text | **启用** (text_mode=soft_token) |
| missing_text_token_norm | 0.8209 (训练所得) |
| Checkpoint | `checkpoint/revision_mkgy_soft_text10_seed0.ckpt` |
| Mask | `masks/MKG-Y_random_text50_seed0.pt` (loaded) |

---

## 跨缺失率完整对比（Soft Token 10% checkpoint）

| 设置 | MRR | hit@10 | hit@1 | missing_text MRR |
|---|---|---|---|---|
| 10% 训练+评估 (soft) | 0.3715 | 0.4495 | 0.3231 | 0.3420 |
| 10%→30% legacy | 0.3589 | 0.4476 | 0.3057 | — |
| 10%→50% legacy | 0.3589 | 0.4476 | 0.3057 | — |
| 10%→30% + soft | 0.3241 | 0.4183 | 0.2685 | 0.2904 |
| 10%→50% + soft | **0.2893** | 0.3992 | 0.2291 | 0.2665 |

> 趋势清晰：启用 soft token 后缺失率越高性能越差（30%=0.3241 → 50%=0.2893），legacy 模式下缺失率不影响（始终 0.3589）。这说明 10% 训练的 soft token 在更高缺失率下有严重的负迁移。
