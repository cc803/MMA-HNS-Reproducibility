# Cross-Rate Generalization: Soft Token 10% Checkpoint → 30% Missing (No Soft Text)

**命令**:
```bash
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --inject-text-missing-rate 0.3 \
  --text-missing-mask-strategy random \
  --save-text-missing-mask-path masks/MKG-Y_random_text30_seed0.pt \
  --test --checkpoint-path checkpoint/revision_mkgy_soft_text10_seed0.ckpt
```

**说明**: 用 10% 缺失训练的 Soft Token A 基线 checkpoint，在 30% 缺失下做 test-only 评估（未启用 soft-missing-text 路径）。

**运行时间**: 2026-07-25 | 仅评估（~51s）| GPU 显存 ~0.96 GB

---

## 整体指标

| 指标 | 10% 训练+评估 | 30% 仅评估 (本实验) | Δ |
|---|---|---|---|
| MRR | 0.3715 | **0.3589** | −0.0126 |
| MR | 1338.77 | 1392.41 | +53.64 |
| hit@10 | 0.4495 | **0.4476** | −0.0019 |
| hit@3 | 0.3997 | 0.3898 | −0.0100 |
| hit@1 | 0.3231 | 0.3057 | −0.0175 |

## 关键配置

| 参数 | 值 |
|---|---|
| 模式 | test-only（无训练） |
| 训练缺失率 | 10% (1,232 injected) |
| 评估缺失率 | 30% (3,695 injected + 2,684 天然 = 6,379 缺失) |
| soft-missing-text | **未启用** (text_mode=legacy) |
| Checkpoint | `checkpoint/revision_mkgy_soft_text10_seed0.ckpt` |
| 新 Mask | `masks/MKG-Y_random_text30_seed0.pt` |
