#!/bin/bash
#SBATCH --job-name=bcql-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
: "${BCQ_SEED:?Set BCQ_SEED}"
OUTPUT_ROOT="${REPRO_OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

cd "$PROJECT_ROOT"
source "${VENV_DIR:-$PROJECT_ROOT/.venv}/bin/activate"
mkdir -p logs "$OUTPUT_ROOT/bcql"

for VARIANT in A0 A1 A2 A3 A4 A5; do
  python src/bcql/train_bcql_ablation.py \
    --variant "$VARIANT" \
    --seed "$BCQ_SEED" \
    --steps 20000 \
    --output-dir "$OUTPUT_ROOT/bcql/${VARIANT}_${BCQ_SEED}"
done
