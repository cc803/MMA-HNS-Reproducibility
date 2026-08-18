# Phase 1 MKG-Y 3-Seed Main Experiment Manifest

Generated at: 2026-04-27 09:48:54 +08:00

Git commit hash: UNAVAILABLE (.git directory not found in this workspace; git command was unavailable during audit)

Dataset: MKG-Y

Seeds: 0, 1, 2

Results directory: `./results/phase1_mkgy`

Checkpoint directory: `./checkpoint/phase1_mkgy`

Summary script: `python summarize_results.py --results-dir ./results/phase1_mkgy`

## Frozen Settings

- Retrieval mix weight is fixed at `0.25`.
- Do not use `run_rotate_retrieval_weight_probe.sh` or any other test-set weight probe script.
- Do not adjust `retrieval_mix_weight`, `lambda-grid`, `alpha-grid`, or `safe-delta` after looking at test results.
- HPSAC alpha/lambda/safe-delta selection must use the validation set only.
- Test set is for final evaluation only.
- Existing files under `./results/` are not used for this phase-1 main table.

## Methods

### rotate_baseline

Seed 0:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --subset-eval --checkpoint-path ./checkpoint/phase1_mkgy/rotate_baseline_seed0.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/rotate_baseline_seed0.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_baseline_seed0.ckpt`

Log: `./results/phase1_mkgy/rotate_baseline_seed0.log`

Seed 1:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 1 --subset-eval --checkpoint-path ./checkpoint/phase1_mkgy/rotate_baseline_seed1.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/rotate_baseline_seed1.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_baseline_seed1.ckpt`

Log: `./results/phase1_mkgy/rotate_baseline_seed1.log`

Seed 2:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 2 --subset-eval --checkpoint-path ./checkpoint/phase1_mkgy/rotate_baseline_seed2.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/rotate_baseline_seed2.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_baseline_seed2.ckpt`

Log: `./results/phase1_mkgy/rotate_baseline_seed2.log`

### hard_mask

Seed 0:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --subset-eval --use-missing-mask --checkpoint-path ./checkpoint/phase1_mkgy/rotate_hardmask_seed0.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/hard_mask_seed0.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_hardmask_seed0.ckpt`

Log: `./results/phase1_mkgy/hard_mask_seed0.log`

Seed 1:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 1 --subset-eval --use-missing-mask --checkpoint-path ./checkpoint/phase1_mkgy/rotate_hardmask_seed1.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/hard_mask_seed1.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_hardmask_seed1.ckpt`

Log: `./results/phase1_mkgy/hard_mask_seed1.log`

Seed 2:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 2 --subset-eval --use-missing-mask --checkpoint-path ./checkpoint/phase1_mkgy/rotate_hardmask_seed2.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/hard_mask_seed2.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_hardmask_seed2.ckpt`

Log: `./results/phase1_mkgy/hard_mask_seed2.log`

### soft_token

Seed 0:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --subset-eval --use-soft-missing-text --checkpoint-path ./checkpoint/phase1_mkgy/rotate_softtoken_seed0.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_seed0.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_softtoken_seed0.ckpt`

Log: `./results/phase1_mkgy/soft_token_seed0.log`

Seed 1:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 1 --subset-eval --use-soft-missing-text --checkpoint-path ./checkpoint/phase1_mkgy/rotate_softtoken_seed1.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_seed1.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_softtoken_seed1.ckpt`

Log: `./results/phase1_mkgy/soft_token_seed1.log`

Seed 2:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 2 --subset-eval --use-soft-missing-text --checkpoint-path ./checkpoint/phase1_mkgy/rotate_softtoken_seed2.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_seed2.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_softtoken_seed2.ckpt`

Log: `./results/phase1_mkgy/soft_token_seed2.log`

### soft_token_retrieval

Seed 0:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 0 --subset-eval --use-soft-missing-text --use-retrieval-missing-text --retrieval-topk 5 --retrieval-pool-size 512 --retrieval-mix-weight 0.25 --checkpoint-path ./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed0.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_retrieval_seed0.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed0.ckpt`

Log: `./results/phase1_mkgy/soft_token_retrieval_seed0.log`

Seed 1:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 1 --subset-eval --use-soft-missing-text --use-retrieval-missing-text --retrieval-topk 5 --retrieval-pool-size 512 --retrieval-mix-weight 0.25 --checkpoint-path ./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed1.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_retrieval_seed1.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed1.ckpt`

Log: `./results/phase1_mkgy/soft_token_retrieval_seed1.log`

Seed 2:

```powershell
python train_dhns_rotate.py --dataset MKG-Y --seed 2 --subset-eval --use-soft-missing-text --use-retrieval-missing-text --retrieval-topk 5 --retrieval-pool-size 512 --retrieval-mix-weight 0.25 --checkpoint-path ./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed2.ckpt 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_retrieval_seed2.log
```

Checkpoint: `./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed2.ckpt`

Log: `./results/phase1_mkgy/soft_token_retrieval_seed2.log`

### soft_token_retrieval_guarded_hpsac

Seed 0:

```powershell
python eval_hpsac.py --dataset MKG-Y --checkpoint-a ./checkpoint/phase1_mkgy/rotate_softtoken_seed0.ckpt --checkpoint-b ./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed0.ckpt --lambda-grid 0.10,0.20,0.25,0.30,0.40 --alpha-grid 0.0,0.1,0.2,0.3 --min-group-queries 30 --safe-delta 0.0002 --lock-missing-text --subset-eval 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_retrieval_guarded_hpsac_seed0.log
```

Checkpoint: `checkpoint_a=./checkpoint/phase1_mkgy/rotate_softtoken_seed0.ckpt; checkpoint_b=./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed0.ckpt`

Log: `./results/phase1_mkgy/soft_token_retrieval_guarded_hpsac_seed0.log`

Seed 1:

```powershell
python eval_hpsac.py --dataset MKG-Y --checkpoint-a ./checkpoint/phase1_mkgy/rotate_softtoken_seed1.ckpt --checkpoint-b ./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed1.ckpt --lambda-grid 0.10,0.20,0.25,0.30,0.40 --alpha-grid 0.0,0.1,0.2,0.3 --min-group-queries 30 --safe-delta 0.0002 --lock-missing-text --subset-eval 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_retrieval_guarded_hpsac_seed1.log
```

Checkpoint: `checkpoint_a=./checkpoint/phase1_mkgy/rotate_softtoken_seed1.ckpt; checkpoint_b=./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed1.ckpt`

Log: `./results/phase1_mkgy/soft_token_retrieval_guarded_hpsac_seed1.log`

Seed 2:

```powershell
python eval_hpsac.py --dataset MKG-Y --checkpoint-a ./checkpoint/phase1_mkgy/rotate_softtoken_seed2.ckpt --checkpoint-b ./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed2.ckpt --lambda-grid 0.10,0.20,0.25,0.30,0.40 --alpha-grid 0.0,0.1,0.2,0.3 --min-group-queries 30 --safe-delta 0.0002 --lock-missing-text --subset-eval 2>&1 | Tee-Object ./results/phase1_mkgy/soft_token_retrieval_guarded_hpsac_seed2.log
```

Checkpoint: `checkpoint_a=./checkpoint/phase1_mkgy/rotate_softtoken_seed2.ckpt; checkpoint_b=./checkpoint/phase1_mkgy/rotate_retrieval_w025_seed2.ckpt`

Log: `./results/phase1_mkgy/soft_token_retrieval_guarded_hpsac_seed2.log`

## Summarization

After all logs are produced, run:

```powershell
python summarize_results.py --results-dir ./results/phase1_mkgy
```

Expected outputs:

- `./results/phase1_mkgy/main_runs.csv`
- `./results/phase1_mkgy/main_summary.csv`
- `./results/phase1_mkgy/main_results.json`
