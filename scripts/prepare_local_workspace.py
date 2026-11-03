#!/usr/bin/env python3
"""Create the directory layout expected by the preserved experiment scripts."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing release artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(args: argparse.Namespace) -> None:
    artifacts = args.artifacts.resolve()

    halfcheetah_data = artifacts / "datasets" / "halfcheetah_day21.npz"
    swimmer_data = artifacts / "datasets" / "safety_swimmer_velocity.npz"

    link_or_copy(
        halfcheetah_data,
        REPO_ROOT
        / "src"
        / "halfcheetah"
        / "每日结果"
        / "Day20_路径二状态相关代价"
        / "数据_HalfCheetah_异质.npz",
    )
    link_or_copy(
        swimmer_data,
        REPO_ROOT
        / "src"
        / "swimmer"
        / "论文核验输出_E3e_Swimmer离线数据pilot_20261013"
        / "数据_SwimmerVelocity_自采pilot.npz",
    )

    if args.cargoal:
        cargoal = args.cargoal.resolve()
        link_or_copy(
            cargoal,
            REPO_ROOT / "data" / "raw" / "SafetyCarGoal1Gymnasium-v0-120-1671.hdf5",
        )

    early_dir = (
        REPO_ROOT
        / "src"
        / "halfcheetah"
        / "论文核验输出_P3稳定IQL_DE端点"
    )
    extended_dir = (
        REPO_ROOT
        / "src"
        / "halfcheetah"
        / "论文核验输出_HC补充_20261016"
    )
    for arm in ("D10", "E100"):
        for seed in range(3):
            source = (
                artifacts
                / "checkpoints"
                / "halfcheetah"
                / "p3"
                / "early"
                / f"{arm}_seed{seed}.pt"
            )
            destination = early_dir / f"模型_P3_{arm}_种子{seed}_40000步.pt"
            if source.is_file():
                link_or_copy(source, destination)
        for seed in range(3, 8):
            source = (
                artifacts
                / "checkpoints"
                / "halfcheetah"
                / "p3"
                / "extended"
                / f"P3_{arm}_seed{seed}"
                / "model.pt"
            )
            destination = (
                extended_dir
                / f"P3_{arm}_seed{seed}"
                / f"模型_P3_{arm}_种子{seed}_40000步.pt"
            )
            if source.is_file():
                link_or_copy(source, destination)

    print(f"Prepared local workspace under {REPO_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Directory containing datasets/, checkpoints/, and evidence/.",
    )
    parser.add_argument(
        "--cargoal",
        type=Path,
        help="Optional path to the public DSRL CarGoal HDF5 file.",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
