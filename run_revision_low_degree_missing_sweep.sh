#!/usr/bin/env bash
set -euo pipefail

# Run realistic low-degree missing-text revision experiments across datasets/rates.
#
# Defaults:
#   - datasets: MKG-Y and MKG-W
#   - low-degree text missing rates: 10%, 30%, 50%
#   - seeds: 0/1/2
#
# The underlying runner is resumable: completed logs with RESULT_JSON are skipped
# unless FORCE=1 is set.
#
# Typical usage:
#   bash run_revision_low_degree_missing_sweep.sh
#
# Useful overrides:
#   DATASETS="MKG-W" RATES="0.1 0.5" bash run_revision_low_degree_missing_sweep.sh
#   SEEDS="0" TRAIN_TIMES=200 bash run_revision_low_degree_missing_sweep.sh
#   PYTHON_EXE=/path/to/python bash run_revision_low_degree_missing_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATASETS="${DATASETS:-MKG-Y MKG-W}"
RATES="${RATES:-0.1 0.3 0.5}"
SEEDS="${SEEDS:-0 1 2}"
MASK_STRATEGY="${MASK_STRATEGY:-low_degree}"
PYTHON_EXE="${PYTHON_EXE:-python}"

dataset_tag() {
  case "$1" in
    MKG-Y) echo "mkgy" ;;
    MKG-W) echo "mkgw" ;;
    DB15K) echo "db15k" ;;
    *)
      echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_'
      ;;
  esac
}

rate_tag() {
  "${PYTHON_EXE}" - "$1" <<'PY'
import sys
rate = float(sys.argv[1])
print("text%d" % int(round(rate * 100)))
PY
}

echo "Low-degree missing-text revision sweep"
echo "  datasets: ${DATASETS}"
echo "  rates: ${RATES}"
echo "  seeds: ${SEEDS}"
echo "  mask_strategy: ${MASK_STRATEGY}"
echo "  python_exe: ${PYTHON_EXE}"

for DATASET_NAME in ${DATASETS}; do
  TAG="$(dataset_tag "${DATASET_NAME}")"
  for RATE in ${RATES}; do
    RATE_TAG="$(rate_tag "${RATE}")"
    RATE_PCT="${RATE_TAG#text}"
    RESULT_DIR_VALUE="results/${TAG}_text_missing/revision_low_degree${RATE_PCT}"
    if [[ "${DATASET_NAME}" == "MKG-Y" && "${RATE_TAG}" == "text30" ]]; then
      RESULT_DIR_VALUE="results/mkgy_text_missing/revision_low_degree30"
    fi

    echo
    echo "===== Sweep item: dataset=${DATASET_NAME} rate=${RATE} (${RATE_TAG}) ====="
    DATASET="${DATASET_NAME}" \
    DATASET_TAG="${TAG}" \
    PYTHON_EXE="${PYTHON_EXE}" \
    SEEDS="${SEEDS}" \
    INJECT_RATE="${RATE}" \
    RATE_TAG="${RATE_TAG}" \
    MASK_STRATEGY="${MASK_STRATEGY}" \
    RESULT_DIR="${RESULT_DIR_VALUE}" \
    SUMMARY_CSV="${RESULT_DIR_VALUE}/summary.csv" \
      bash run_revision_mkgy_low_degree30_remaining.sh
  done
done

echo
echo "Low-degree missing-text revision sweep done."
