#!/bin/bash
#SBATCH --job-name=bcql-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
: "${BCQ_SEED:?Set BCQ_SEED}"
VARIANT="${BCQ_VARIANT:-base}"
STEPS="${BCQ_STEPS:-100000}"
OUTPUT_ROOT="${REPRO_OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

cd "$PROJECT_ROOT"
source "${VENV_DIR:-$PROJECT_ROOT/.venv}/bin/activate"
mkdir -p logs "$OUTPUT_ROOT/bcql"

if [[ "$VARIANT" == "base" ]]; then
  python src/bcql/train_bcql_cargoal1.py \
    --seed "$BCQ_SEED" \
    --steps "$STEPS" \
    --output-dir "$OUTPUT_ROOT/bcql/$BCQ_SEED"
elif [[ "$VARIANT" == "control" ]]; then
  python src/bcql/train_bcql_cargoal1.py \
    --seed "$BCQ_SEED" \
    --steps "$STEPS" \
    --disable-cost \
    --output-dir "$OUTPUT_ROOT/bcql/${BCQ_SEED}_control"
else
  python src/bcql/train_bcql_ablation.py \
    --variant "$VARIANT" \
    --seed "$BCQ_SEED" \
    --steps "$STEPS" \
    --output-dir "$OUTPUT_ROOT/bcql/${VARIANT}_${BCQ_SEED}"
fi
