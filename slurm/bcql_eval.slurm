#!/bin/bash
#SBATCH --job-name=bcql-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
RUN_DIR="${1:?usage: sbatch bcql_eval.slurm <run_dir> [output_json]}"
OUT_JSON="${2:-$RUN_DIR/evaluation_100.json}"

cd "$PROJECT_ROOT"
source "${VENV_DIR:-$PROJECT_ROOT/.venv}/bin/activate"
mkdir -p logs

python src/bcql/evaluate_bcql_checkpoint.py \
  --checkpoint "$RUN_DIR/checkpoints/model.pt" \
  --config "$RUN_DIR/config.json" \
  --episodes 100 \
  --seed-start 149000 \
  --output "$OUT_JSON"
