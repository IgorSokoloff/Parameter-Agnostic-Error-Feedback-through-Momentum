#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Global switches — adjust to match your machine
###############################################################################
CONDA_ENV="${CONDA_ENV:-ef21-hess}"  # name of the environment to activate
PROJECT_NAME="${PROJECT_NAME:-EF21_SOM}"

METHODS=(
  "EF21_HM_NORM"
  "EF21_RHM_NORM"
  "EF21_MVR_NORM"
  "EF21_IGT_NORM"
  "EF21"
  "ECONTROL"
  "EF21_SGDM"
  "EF21_SGDM_NORM"
)

WORKERS=10
BATCH=64
EPOCHS=90
TOPK=0.1

###############################################################################

timestamp="$(date +%Y%m%d_%H%M%S)"

for method in "${METHODS[@]}"; do
  session="train_${method}_${timestamp}"
  log_file="${session}.log"

  tmux new-session -d -s "${session}" "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\" &&
    conda activate ${CONDA_ENV} &&
    python -m repro_training \
      --method ${method} \
      --project ${PROJECT_NAME} \
      --workers ${WORKERS} \
      --batch-size ${BATCH} \
      --epochs ${EPOCHS} \
      --topk ${TOPK} \
      2>&1 | tee ${log_file}
  "

  echo "Started tmux session '${session}' for method ${method}"

  while tmux has-session -t "${session}" 2>/dev/null; do
    sleep 300
  done

  echo "Finished ${method} (log: ${log_file})"
done
