import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

BASE = os.environ.get("HALFCHEETAH_WORKDIR", str(Path(__file__).resolve().parent))
EARLY = os.path.join(BASE, "论文核验输出_P3稳定IQL_DE端点")
HC = os.path.join(BASE, "论文核验输出_HC补充_20261016")
P3 = os.path.join(BASE, "工具_Day21_P3_稳定IQL_DE端点复验_v1_1.py")


def import_local(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_path(arm, seed):
    if seed <= 2:
        return os.path.join(EARLY, f"模型_P3_{arm}_种子{seed}_40000步.pt")
    return os.path.join(
        HC, f"P3_{arm}_seed{seed}", f"模型_P3_{arm}_种子{seed}_40000步.pt"
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


p3 = import_local(P3, "p3_frozen")
print("StablePolicy", p3.StablePolicy)
rows = []
for arm in ("D10", "E100"):
    for seed in range(8):
        path = model_path(arm, seed)
        exists = os.path.exists(path)
        if not exists:
            print("MISSING", arm, seed, path)
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = p3.StablePolicy(
            np.zeros((1, 17), dtype=np.float32), np.ones((1, 17), dtype=np.float32)
        )
        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            fingerprint = model.均值动作(torch.zeros((1, 17), dtype=torch.float32))
        meta = {
            "arm": ck.get("arm"),
            "seed": ck.get("seed"),
            "steps": ck.get("steps"),
        }
        ok_meta = meta == {"arm": arm, "seed": seed, "steps": 40000}
        obs_mean = model.obs_mean.detach().numpy().reshape(-1)
        obs_std = model.obs_std.detach().numpy().reshape(-1)
        ok_norm = (
            obs_mean.shape == (17,)
            and obs_std.shape == (17,)
            and np.isfinite(obs_mean).all()
            and np.isfinite(obs_std).all()
            and (obs_std > 0).all()
        )
        row = {
            "arm": arm,
            "seed": seed,
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": sha256(path),
            "meta": meta,
            "meta_ok": ok_meta,
            "normalization_ok": bool(ok_norm),
            "fingerprint_zero_state": fingerprint.numpy().reshape(-1).tolist(),
        }
        rows.append(row)
        print(
            arm,
            seed,
            "OK" if ok_meta and ok_norm else "CHECK",
            row["sha256"][:16],
            "fp=",
            np.round(row["fingerprint_zero_state"], 5).tolist(),
        )

print("\nTOTAL", len(rows), "all_ok", all(r["meta_ok"] and r["normalization_ok"] for r in rows))
if len(rows) != 16:
    raise SystemExit("model count != 16")
with open(
    os.path.join(BASE, "P3_FINAL_MODEL_VERIFICATION_20261020.json"),
    "w",
    encoding="utf-8",
) as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=2)
print("manifest written")
