#!/bin/bash
#SBATCH --job-name=bcql-l2-rollout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
: "${BCQ_RUN_DIR:?Set BCQ_RUN_DIR}"
OUTPUT_ROOT="${REPRO_OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

cd "$PROJECT_ROOT"
source "${VENV_DIR:-$PROJECT_ROOT/.venv}/bin/activate"
mkdir -p logs "$OUTPUT_ROOT/l2"

python src/bcql/l2_rollout_v2.py \
  --checkpoint "$BCQ_RUN_DIR/checkpoints/model.pt" \
  --l2-dir "$OUTPUT_ROOT/l2/offline" \
  --out "$OUTPUT_ROOT/l2/rollout" \
  --rollouts 5 \
  --horizon 0
