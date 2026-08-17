#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
RESULT_DIR="./results/phase1_mkgw"
CHECKPOINT_DIR="./checkpoint/phase1_mkgw"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULT_DIR}" "${CHECKPOINT_DIR}"

CODE_SNAPSHOT="${RESULT_DIR}/code_snapshot_${RUN_TS}.tar.gz"
tar \
  --exclude="./checkpoint" \
  --exclude="./results" \
  --exclude="./embeddings" \
  --exclude="./__pycache__" \
  --exclude="*/__pycache__" \
  --exclude="*.pyc" \
  -czf "${CODE_SNAPSHOT}" .

echo "Code snapshot saved to: ${CODE_SNAPSHOT}"
echo "Phase1 MKG-W settings:"
echo "  seeds: 0 1 2"
echo "  retrieval_mix_weight: 0.25 (frozen; do not tune on test)"
echo "  HPSAC lambda-grid: 0.10,0.20,0.25,0.30,0.40"
echo "  HPSAC alpha-grid: 0.0,0.1,0.2,0.3"
echo "  HPSAC safe-delta: 0.0002"
echo "  HPSAC selection split: validation only"
echo "  test split: final evaluation only"

for S in 0 1 2; do
  "${PYTHON_EXE}" train_dhns_rotate.py \
    --dataset MKG-W \
    --seed "${S}" \
    --subset-eval \
    --checkpoint-path "${CHECKPOINT_DIR}/rotate_baseline_seed${S}.ckpt" \
    2>&1 | tee "${RESULT_DIR}/rotate_baseline_seed${S}.log"
done

for S in 0 1 2; do
  "${PYTHON_EXE}" train_dhns_rotate.py \
    --dataset MKG-W \
    --seed "${S}" \
    --subset-eval \
    --use-missing-mask \
    --checkpoint-path "${CHECKPOINT_DIR}/rotate_hardmask_seed${S}.ckpt" \
    2>&1 | tee "${RESULT_DIR}/hard_mask_seed${S}.log"
done

for S in 0 1 2; do
  "${PYTHON_EXE}" train_dhns_rotate.py \
    --dataset MKG-W \
    --seed "${S}" \
    --subset-eval \
    --use-soft-missing-text \
    --checkpoint-path "${CHECKPOINT_DIR}/rotate_softtoken_seed${S}.ckpt" \
    2>&1 | tee "${RESULT_DIR}/soft_token_seed${S}.log"
done

for S in 0 1 2; do
  "${PYTHON_EXE}" train_dhns_rotate.py \
    --dataset MKG-W \
    --seed "${S}" \
    --subset-eval \
    --use-soft-missing-text \
    --use-retrieval-missing-text \
    --retrieval-topk 5 \
    --retrieval-pool-size 512 \
    --retrieval-mix-weight 0.25 \
    --checkpoint-path "${CHECKPOINT_DIR}/rotate_retrieval_w025_seed${S}.ckpt" \
    2>&1 | tee "${RESULT_DIR}/soft_token_retrieval_seed${S}.log"
done

for S in 0 1 2; do
  "${PYTHON_EXE}" eval_hpsac.py \
    --dataset MKG-W \
    --checkpoint-a "${CHECKPOINT_DIR}/rotate_softtoken_seed${S}.ckpt" \
    --checkpoint-b "${CHECKPOINT_DIR}/rotate_retrieval_w025_seed${S}.ckpt" \
    --lambda-grid 0.10,0.20,0.25,0.30,0.40 \
    --alpha-grid 0.0,0.1,0.2,0.3 \
    --min-group-queries 30 \
    --safe-delta 0.0002 \
    --lock-missing-text \
    --subset-eval \
    2>&1 | tee "${RESULT_DIR}/soft_token_retrieval_guarded_hpsac_seed${S}.log"
done

"${PYTHON_EXE}" summarize_results.py --results-dir "${RESULT_DIR}"
