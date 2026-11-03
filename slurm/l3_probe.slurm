#!/bin/bash
#SBATCH --job-name=bcql-l3-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
: "${BCQ_RUN_DIR:?Set BCQ_RUN_DIR}"
OUTPUT_ROOT="${REPRO_OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

cd "$PROJECT_ROOT"
source "${VENV_DIR:-$PROJECT_ROOT/.venv}/bin/activate"
mkdir -p logs "$OUTPUT_ROOT/l3"

python src/bcql/l3_static_probe.py \
  --checkpoint "$BCQ_RUN_DIR/checkpoints/model.pt" \
  --n 1024 \
  --output "$OUTPUT_ROOT/l3/probe.json"
