#!/usr/bin/env python3
"""Verify released hashes, array schemas, and episode boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED = {
    "datasets/halfcheetah_day21.npz": (
        "e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179"
    ),
    "datasets/safety_swimmer_velocity.npz": (
        "4edffe8b0178c1220a3d158521cbeafccd2ca479dac07fa8e38430d6086202fa"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_halfcheetah(path: Path) -> dict:
    with np.load(path) as data:
        obs = data["obs"]
        act = data["act"]
        done = data["done"]
        cost = data["cost"]
        episodes = np.flatnonzero(done > 0.5) + 1
        assert obs.shape == (200000, 17)
        assert act.shape == (200000, 6)
        assert len(episodes) == 200
        assert np.isfinite(obs).all()
        assert np.isfinite(act).all()
        assert np.isfinite(cost).all()
        return {
            "transitions": int(len(obs)),
            "episodes": int(len(episodes)),
            "observation_dim": int(obs.shape[1]),
            "action_dim": int(act.shape[1]),
            "cost_mean": float(cost.mean()),
            "cost_positive_rate": float((cost > 0).mean()),
        }


def verify_swimmer(path: Path) -> dict:
    with np.load(path) as data:
        obs = data["obs"]
        act = data["act"]
        next_obs = data["next_obs"]
        episode = data["episode"]
        done = data["done"]
        assert obs.shape == (20000, 8)
        assert act.shape == (20000, 2)
        assert next_obs.shape == obs.shape
        assert int(episode.max()) + 1 == 20
        assert int(done.sum()) == 20
        for episode_id in np.unique(episode):
            idx = np.flatnonzero(episode == episode_id)
            assert np.array_equal(next_obs[idx[:-1]], obs[idx[1:]])
        return {
            "transitions": int(len(obs)),
            "episodes": int(episode.max()) + 1,
            "observation_dim": int(obs.shape[1]),
            "action_dim": int(act.shape[1]),
            "cost_positive_rate": float((data["cost"] > 0).mean()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Verify every file listed in manifests/SHA256SUMS.txt.",
    )
    args = parser.parse_args()

    for relative, expected in EXPECTED.items():
        path = args.artifacts / relative
        actual = sha256(path)
        if actual.lower() != expected:
            raise SystemExit(f"SHA256 mismatch for {relative}: {actual}")

    summary = {
        "halfcheetah_day21": verify_halfcheetah(
            args.artifacts / "datasets" / "halfcheetah_day21.npz"
        ),
        "safety_swimmer_velocity": verify_swimmer(
            args.artifacts / "datasets" / "safety_swimmer_velocity.npz"
        ),
    }
    if args.check_all:
        manifest_path = args.artifacts / "manifests" / "SHA256SUMS.txt"
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = args.artifacts / relative
            actual = sha256(path)
            if actual != expected:
                raise SystemExit(f"SHA256 mismatch for {relative}: {actual}")
        summary["full_manifest_files"] = len(
            manifest_path.read_text(encoding="utf-8").splitlines()
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
