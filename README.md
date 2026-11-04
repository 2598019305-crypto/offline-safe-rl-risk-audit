# Offline Safe RL Risk Audit

This repository contains the source code for the paper
**Diagnosing Risk-Penalization Actionability in Offline Safe Reinforcement
Learning**. It covers the four diagnostic layers used in the paper:

1. risk-score resolution;
2. finite-candidate feasibility;
3. weight, gradient, and actor response;
4. closed-loop realized cost and policy safety.

The repository includes the HalfCheetah and SafetySwimmerVelocity experiment
code, the Cost-Prediction audit, the CarCircle candidate audit, and the BCQ-Lag
diagnostic pipeline. Large HDF5/NPZ files and model weights are distributed
separately in the replication-data record.

## Citation

Use the version-specific DOI in the manuscript's Data Availability Statement
once the archival release has been assigned. Until then, cite this repository
by its tagged release.

## What is included

| Path | Contents |
| --- | --- |
| `src/shared` | Fixed-batch replay engine and common analysis utilities |
| `src/halfcheetah` | HalfCheetah data construction, L1-L4 analyses, IQL, NORM2, and sealed evaluation |
| `src/swimmer` | SafetySwimmerVelocity data construction and risk-proxy diagnostics |
| `src/carcircle` | Candidate-feasibility, gating, actor-transfer, and sealed-evaluation code |
| `src/costpred` | Strategy-level cost-prediction audit |
| `src/bcql` | BCQ-Lag training, ablation, candidate, gradient, and closed-loop evaluation code |
| `slurm` | Portable Slurm launch templates with no institution-specific paths or accounts |
| `configs` | Frozen configuration used by the BCQ-Lag experiment |
| `docs` | Data schema, source map, and reproduction protocol |
| `scripts` | Local workspace preparation and integrity checks |

## Environment

The experiments reported in the paper used:

- Python 3.9.25
- PyTorch 2.2.2 with CUDA 12.1
- Gymnasium 0.28.1
- Safety-Gymnasium 0.4.0
- DSRL 0.1.0
- OSRL commit `03a4586463690566ec8d7830ca4027b46d4ad02f`

The exact environment recorded during training is in
`docs/environment_versions.json`. GPU jobs used one NVIDIA G4/RTX 4090 with
64 GB host memory.

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MuJoCo and Safety-Gymnasium may require platform-specific system packages.
The archived experiment used `MUJOCO_GL=disabled` for headless training.

## Datasets and checkpoints

Download the replication-data archive referenced by the manuscript and place
it outside version control, for example:

```text
artifacts/
  datasets/
    halfcheetah_day21.npz
    safety_swimmer_velocity.npz
  checkpoints/
  evidence/
```

Create the directory layout expected by the original experiment scripts:

```bash
python scripts/prepare_local_workspace.py --artifacts /path/to/artifacts
```

Verify the released data before running any experiment:

```bash
python scripts/verify_release.py --artifacts /path/to/artifacts
```

Add `--check-all` to verify every file listed in the release manifest.

Expected SHA256:

```text
e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179  halfcheetah_day21.npz
4edffe8b0178c1220a3d158521cbeafccd2ca479dac07fa8e38430d6086202fa  safety_swimmer_velocity.npz
```

The BCQ-Lag experiments additionally require the public DSRL
`SafetyCarGoal1Gymnasium-v0-120-1671.hdf5` file. It is not redistributed in
this repository. Place it at `data/raw/SafetyCarGoal1Gymnasium-v0-120-1671.hdf5`
or set `BCQ_CARGOAL_HDF5` to its absolute path.

## Reproducing the four diagnostic layers

The original scripts are preserved under their experiment names to keep a
direct trace to the pre-registered protocols and recorded outputs. Most scripts
are intentionally small, single-purpose programs rather than hidden behind a
large framework.

### L1: risk-score resolution

For HalfCheetah:

```bash
cd src/halfcheetah
python 工具_Day21_D组_直接邻域上尾统计消融.py --help
python 工具_Day21_E组_直接邻域均值消融.py --help
python 工具_Day21_F组_上尾比例扫描.py --help
```

For SafetySwimmerVelocity:

```bash
cd src/swimmer
python 工具_E3e_Swimmer离线数据pilot.py
python 工具_E4_0_风险代理有效性诊断.py
```

### L2: finite-candidate feasibility

The primary analyses are in `src/carcircle`. The pipeline is ordered:

```text
p0_healthcheck.py
p1_bc_formal.py
p1_cpq_official.py
p1_unified_table.py
p2_gate_arms.py
p2_train.py
p3_sealed_eval.py
p3_merge.py
```

### L3: actor weights and gradients

The main scripts are:

```text
src/halfcheetah/工具_Day21_P3_稳定IQL_DE端点复验_v1_1.py
src/halfcheetah/工具_Day21_P3_稳定IQL_DE端点复验_扩种子v1.py
src/halfcheetah/工具_Day21_P4_动作平方范数公平对照_扩种子v1.py
src/halfcheetah/工具_Day21_E2b阶段II_多起点多种子.py
src/halfcheetah/工具_判据对照_同状态动作可解释度.py
```

### L4: closed-loop cost

HalfCheetah sealed evaluation:

```bash
cd src/halfcheetah
python 工具_P3_封存测试_仅评估.py --help
python 工具_P3_封存测试_汇总.py
```

BCQ-Lag 100-seed evaluation:

```bash
python src/bcql/evaluate_bcql_checkpoint.py \
  --checkpoint /path/to/model.pt \
  --config /path/to/config.json \
  --episodes 100 \
  --seed-start 149000 \
  --output /path/to/evaluation_100.json
```

## BCQ-Lag training

The BCQ-Lag scripts accept environment variables instead of assuming a remote
cluster layout:

```bash
export BCQ_DATA_DIR=/path/to/data/raw
export BCQ_CARGOAL_HDF5=/path/to/SafetyCarGoal1Gymnasium-v0-120-1671.hdf5

python src/bcql/train_bcql_cargoal1.py \
  --seed 20261216 \
  --steps 100000 \
  --output-dir outputs/bcql/20261216

python src/bcql/train_bcql_cargoal1.py \
  --seed 20261216 \
  --steps 100000 \
  --disable-cost \
  --output-dir outputs/bcql/20261216_control
```

The ablation entry point is `src/bcql/train_bcql_ablation.py`. A6 uses
`--variant A6`; A0-A5 provide the single-factor controls described in the
paper.

To reproduce the five-seed paired analysis from the released evaluation JSON
files, arrange the constrained and `--disable-cost` files as
`<seed>_evaluation_100.json` and `<seed>_control_evaluation_100.json`, and the
A6 files as `<a6-root>/<seed>/evaluation_100.json`. Then run:

```bash
python src/bcql/analyze_causal_intervention.py \
  --original-root /path/to/constrained-and-control-evaluations \
  --a6-root /path/to/a6-evaluations \
  --output-dir /path/to/analysis \
  --bootstrap-reps 20000 \
  --bootstrap-seed 20261031 \
  --seeds 20261216 20261217 20261218 20261219 20261220
```

The inference treats training seeds as the upper sampling unit and nests the
100 paired evaluation episodes within each training seed. It never treats all
500 episodes as independent training runs. The archived five-seed run gives
an A6-minus-constrained mean-cost difference of `-11.20` (95% paired cluster
bootstrap CI `[-15.80, -6.18]`) and a CVaR@25 difference of `-22.40`
(`[-34.76, -9.11]`).

## Reproduction scope

The archived files make it possible to:

- recompute every reported table from episode-level evidence;
- reload every released model and rerun evaluation;
- replay frozen HalfCheetah batches for gradient and ESS diagnostics;
- rebuild both self-generated datasets from their released generators.

Exact bitwise reproduction of GPU training across hardware and CUDA versions is
not guaranteed. The released software version, seeds, configs, evaluation
seeds, and deterministic replay protocol are the intended reproduction
boundary.

## Third-party code and data

The BCQ-Lag implementation is derived from OSRL at the pinned commit recorded
in `docs/environment_versions.json`. The Apache-2.0 license for that code is
retained in `third_party/osrl/LICENSE`.

The CarCircle experiments use the public DSRL release. Users must obtain that
data from its original source and comply with its license; this repository does
not redistribute it.

## License

Repository code is licensed under Apache-2.0. The self-generated datasets,
released checkpoints, and evidence tables are licensed under CC BY 4.0. See
`LICENSE` and the replication-data `LICENSE-DATA.md`.

## Contact

Open an issue for code-level reproduction questions. Do not open an issue to
request private credentials, internal cluster access, or files that are
excluded by a third-party license.
