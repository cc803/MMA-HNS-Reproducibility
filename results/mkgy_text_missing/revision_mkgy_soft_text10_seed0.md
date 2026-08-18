# Soft Token A Baseline — MKG-Y text missing 10%

**命令**:
```bash
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --inject-text-missing-rate 0.1 \
  --text-missing-mask-strategy random \
  --save-text-missing-mask-path masks/MKG-Y_random_text10_seed0.pt \
  --use-soft-missing-text --subset-eval \
  --checkpoint-path checkpoint/revision_mkgy_soft_text10_seed0.ckpt
```

**运行时间**: 2026-07-25 | 训练约 2h47min | GPU 显存 ~9.2 GB 分配 / 11.9 GB 预留

---

## 整体指标

| 指标 | 值 |
|---|---|
| MRR | **0.3715** |
| MR | 1338.77 |
| hit@10 | **0.4495** |
| hit@3 | 0.3997 |
| hit@1 | 0.3231 |

## Subset 指标（按文本可用性）

| 子集 | 查询数 | MRR | hit@10 | hit@1 |
|---|---|---|---|---|
| head_or_tail_missing_text | 2610 | 0.3420 | 0.4138 | 0.2969 |
| head_missing_text | 1646 | 0.3185 | 0.3888 | 0.2764 |
| tail_missing_text | 1392 | 0.3805 | 0.4497 | 0.3341 |
| head_or_tail_injected_missing_text | 866 | 0.3702 | 0.4411 | 0.3279 |
| head_and_tail_have_text | 2716 | 0.3999 | 0.4838 | 0.3483 |

## 关键配置

| 参数 | 值 |
|---|---|
| 方法 | Soft missing-text token (A baseline) |
| 注入缺失比例 | 10% (1,232 / 12,316 entities) |
| 天然缺失文本 | 2,684 entities |
| 总计缺失文本 | 3,916 entities |
| Missing text token norm | 0.8209 |
| Checkpoint | `checkpoint/revision_mkgy_soft_text10_seed0.ckpt` |
| Mask | `masks/MKG-Y_random_text10_seed0.pt` |
