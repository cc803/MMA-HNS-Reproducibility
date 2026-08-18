# Cross-Rate Generalization: Soft Token 10% Checkpoint → 50% Missing (No Soft Text)

**命令**:
```bash
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --inject-text-missing-rate 0.5 \
  --text-missing-mask-strategy random \
  --save-text-missing-mask-path masks/MKG-Y_random_text50_seed0.pt \
  --test --checkpoint-path checkpoint/revision_mkgy_soft_text10_seed0.ckpt
```

**说明**: 用 10% 缺失训练的 Soft Token A 基线 checkpoint，在 50% 缺失下做 test-only 评估（未启用 soft-missing-text 路径）。

**运行时间**: 2026-07-25 | 仅评估（~49s）| GPU 显存 ~0.96 GB

---

## 整体指标

| 指标 | 10% 训练+评估 | 30% 仅评估 | 50% 仅评估 (本实验) |
|---|---|---|---|
| MRR | 0.3715 | 0.3589 | 0.3589 |
| MR | 1338.77 | 1392.41 | 1392.41 |
| hit@10 | 0.4495 | 0.4476 | 0.4476 |
| hit@3 | 0.3997 | 0.3898 | 0.3898 |
| hit@1 | 0.3231 | 0.3057 | 0.3057 |

> ⚠️ 30% 与 50% 结果完全一致（legacy 模式下无 soft-missing-text 处理）。

## 关键配置

| 参数 | 值 |
|---|---|
| 模式 | test-only（无训练） |
| 训练缺失率 | 10% (1,232 injected) |
| 评估缺失率 | 50% (6,158 injected + 2,684 天然 = 8,842 缺失, 58.9%) |
| soft-missing-text | **未启用** (text_mode=legacy) |
| Checkpoint | `checkpoint/revision_mkgy_soft_text10_seed0.ckpt` |
| 新 Mask | `masks/MKG-Y_random_text50_seed0.pt` |
