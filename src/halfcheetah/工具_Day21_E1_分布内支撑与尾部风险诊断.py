#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day21 E1：分布内数据支撑与尾部风险诊断（只读分析，不训练策略）。

严格按 `Day21_E1_分布内支撑与尾部风险诊断_协议_v1.0.md` 执行：
  E1a 存在性：支撑度 10 档 × 动作范数 10 档 ⇒ 局部上尾成本 T 的网格与热图
  E1b 混杂  ：动作范数 ‖a‖² 能否解释"支撑度 ↔ 尾部成本"的关系
  E1c 信息性：状态 s 是否携带超出 ‖a‖² 的风险信息（两个标签、按轨迹分组交叉验证）

边界：不训练策略、不访问 125000–125099、不修改任何历史文件。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DATA = HERE / "每日结果" / "Day20_路径二状态相关代价" / "数据_HalfCheetah_异质.npz"
DATA_SHA = "e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179"
OUT = HERE / "论文核验输出_E1分布内支撑与尾部风险诊断"

TAU = 3.9079
K_NN = 50            # 支撑度用的近邻数
TAIL_FRAC = 0.10     # 邻居内取最高 10% 求均值（与 D10 语义一致）
N_BINS = 10          # 分位档数
OOD_FRAC = 0.10      # 后继状态落在 d50 最高 10% ⇒ 越界
CV_FOLDS = 5
SEED = 20261013


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def 载入():
    z = np.load(DATA, allow_pickle=False)
    obs = z["obs"].astype(np.float64)
    act = z["act"].astype(np.float64)
    done = z["done"].astype(np.float64)
    sq = np.sum(act ** 2, axis=1)
    cost = np.maximum(0.0, sq - TAU)
    # 轨迹 id：每遇一个 done 就 +1
    traj = np.concatenate([[0], np.cumsum(done > 0.5)[:-1]]).astype(np.int64)
    # 后继状态：跨轨迹边界与终点一律置为无效
    valid_next = np.zeros(len(obs), dtype=bool)
    valid_next[:-1] = (done[:-1] <= 0.5)
    nxt = np.full((len(obs), obs.shape[1]), np.nan)
    nxt[:-1][valid_next[:-1]] = obs[1:][valid_next[:-1]]
    return dict(obs=obs, act=act, done=done, sq=sq, cost=cost, traj=traj,
                nxt=nxt, valid_next=valid_next)


def 支撑度(obs, nxt, valid_next):
    from scipy.spatial import cKDTree
    mu, sd = obs.mean(0), np.maximum(obs.std(0), 1e-8)
    def z(x):
        return (x - mu) / sd
    tree = cKDTree(z(obs))
    d_self, _ = tree.query(z(obs), k=K_NN + 1, workers=-1)
    d50 = d_self[:, -1]
    # 后继状态的 d50：用同一棵树
    d_next = np.full(len(obs), np.nan)
    idx = np.where(valid_next)[0]
    dn, _ = tree.query(z(nxt[idx]), k=K_NN + 1, workers=-1)
    d_next[idx] = dn[:, -1]
    return d50, d_next, tree, z, mu, sd


def 局部上尾(obs, cost, tree, z):
    _, nn = tree.query(z(obs), k=K_NN, workers=-1)
    nc = cost[nn]                       # (N, K)
    k_tail = max(1, int(round(K_NN * TAIL_FRAC)))
    part = -np.sort(-nc, axis=1)[:, :k_tail]
    T = part.mean(1)
    p90 = np.percentile(nc, 90, axis=1)
    return T, p90, nn


def 分位档(x):
    qs = np.quantile(x, np.linspace(0, 1, N_BINS + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    # 去掉重复边界，保证 bin 数正确
    qs = np.unique(qs)
    b = np.digitize(x, qs[1:-1])
    return b.astype(np.int64), qs


def 偏相关(y, x, z):
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    return float(np.corrcoef(resid(y, z), resid(x, z))[0, 1])


def R2(y, X):
    X1 = np.column_stack([np.ones(len(y))] + list(X))
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def 分组交叉验证(X, y, groups, 名):
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1,
                                             random_state=SEED)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    ap = float(average_precision_score(y, oof))
    print(f"    {名:16s} AUC={auc:.4f}  PR-AUC={ap:.4f}  正例率={y.mean():.4f}", flush=True)
    return dict(features=名, auc=auc, pr_auc=ap, pos_rate=float(y.mean()))


def 主函数():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    h = sha256(DATA)
    print(f"[E1] 数据 SHA256 = {h}", flush=True)
    if h != DATA_SHA:
        raise SystemExit("数据 SHA256 与协议不一致，停止")

    d = 载入()
    obs, act, sq, cost, traj = d["obs"], d["act"], d["sq"], d["cost"], d["traj"]
    print(f"[E1] 转移 {len(obs):,} 条，轨迹 {traj.max()+1} 条，终点 {int((d['done']>0.5).sum())} 个",
          flush=True)
    print(f"[E1] 一步成本均值 {cost.mean():.5f}，动作范数均值 {sq.mean():.7f}，"
          f"有效后继 {int(d['valid_next'].sum()):,}", flush=True)

    d50, d_next, tree, z, mu, sd = 支撑度(obs, d["nxt"], d["valid_next"])
    T, p90, _ = 局部上尾(obs, cost, tree, z)
    print(f"[E1] d50 中位数 {np.median(d50):.4f}；局部上尾 T 均值 {T.mean():.4f}", flush=True)

    # 越界代理
    thr_ood = float(np.quantile(d50, 1 - OOD_FRAC))
    y_ood = ((d_next > thr_ood) & d["valid_next"]).astype(np.int64)
    y_cost = (cost > 0).astype(np.int64)
    print(f"[E1] 越界阈值 d50>{thr_ood:.4f}（最高的 {OOD_FRAC:.0%}）；"
          f"越界率 {y_ood[d['valid_next']].mean():.4f}", flush=True)

    A_bin, A_qs = 分位档(sq)
    S_bin, S_qs = 分位档(d50)

    # ---------- E1a ----------
    print("\n[E1a] 支撑度 × 动作范数 网格（T 均值）", flush=True)
    grid = np.full((N_BINS, N_BINS), np.nan)
    cnt = np.zeros((N_BINS, N_BINS), dtype=np.int64)
    for i in range(N_BINS):
        mi = S_bin == i
        for j in range(N_BINS):
            m = mi & (A_bin == j)
            cnt[i, j] = int(m.sum())
            if cnt[i, j] > 0:
                grid[i, j] = float(T[m].mean())
    rows = []
    for i in range(N_BINS):
        rows.append(dict(支撑档=i,
                         d50区间=[float(S_qs[i]), float(S_qs[i + 1])],
                         每档T均值=float(T[S_bin == i].mean()),
                         每档成本均值=float(cost[S_bin == i].mean()),
                         每档动作范数均值=float(sq[S_bin == i].mean()),
                         样本数=int((S_bin == i).sum())))
        print(f"  支撑档{i:2d} d50∈[{rows[-1]['d50区间'][0]:.3f},{rows[-1]['d50区间'][1]:.3f}) "
              f"T={rows[-1]['每档T均值']:.4f} c={rows[-1]['每档成本均值']:.4f} "
              f"|a|^2={rows[-1]['每档动作范数均值']:.3f}", flush=True)

    # 高支撑档（最好的 2 档）里最高 T 的格
    hi = grid[0:2, :]
    med_T = float(np.median(T))
    exist_hi_support_hi_tail = bool(np.nanmax(hi) > 1.5 * med_T)
    print(f"  全局 T 中位数 {med_T:.4f}；最好 2 档支撑内的最大格 {np.nanmax(hi):.4f} "
          f"⇒ 高支撑+高尾部 {'存在' if exist_hi_support_hi_tail else '未观察到'}", flush=True)

    # ---------- E1b ----------
    print("\n[E1b] 动作范数混杂检验", flush=True)
    r_T_d50 = float(np.corrcoef(T, d50)[0, 1])
    r_T_A = float(np.corrcoef(T, sq)[0, 1])
    pc_T_d50_given_A = 偏相关(T, d50, sq)
    r2_A = R2(T, [sq])
    r2_d = R2(T, [d50])
    r2_both = R2(T, [sq, d50])
    print(f"  corr(T,d50)={r_T_d50:+.4f}  corr(T,|a|^2)={r_T_A:+.4f}  "
          f"偏相关 corr(T,d50 | |a|^2)={pc_T_d50_given_A:+.4f}", flush=True)
    print(f"  R²: T~|a|²={r2_A:.4f}  T~d50={r2_d:.4f}  T~(两者)={r2_both:.4f}", flush=True)
    within = []
    for j in range(N_BINS):
        m = A_bin == j
        if m.sum() < 500:
            continue
        d_med = np.median(d50[m])
        lo = T[m & (d50 <= d_med)]
        hi2 = T[m & (d50 > d_med)]
        within.append(dict(动作范数档=j, T_高支撑组=float(lo.mean()),
                           T_低支撑组=float(hi2.mean()),
                           差=float(hi2.mean() - lo.mean()), n=int(m.sum())))
        print(f"  |a|^2 档{j:2d}（n={int(m.sum()):>7d}）T 支撑好={lo.mean():.4f} "
              f"支撑差={hi2.mean():.4f} 差={hi2.mean()-lo.mean():+.4f}", flush=True)
    ds = np.array([w["差"] for w in within])
    print(f"  各动作范数档内 (低支撑 − 高支撑) 的 T 差：均值 {ds.mean():+.4f}，"
          f"同号 {int((np.sign(ds) == np.sign(ds.mean())).sum())}/{len(ds)}", flush=True)

    # ---------- E1c ----------
    print("\n[E1c] 分组交叉验证（按轨迹分组，5 折）", flush=True)
    feats = {"仅 s（状态）": obs, "仅 A（‖a‖²）": sq.reshape(-1, 1),
             "s + A": np.column_stack([obs, sq])}
    cv_res = {}
    for 标签名, y in (("y_cost（一步成本>0）", y_cost), ("y_ood（后继越界）", y_ood)):
        print(f"  标签 {标签名}：", flush=True)
        cv_res[标签名] = [分组交叉验证(X, y, traj, 名) for 名, X in feats.items()]

    # ---------- 落盘 ----------
    res = dict(
        data_sha256=h, tau=TAU, k_nn=K_NN, tail_frac=TAIL_FRAC, n_bins=N_BINS,
        ood_frac=OOD_FRAC, ood_threshold=thr_ood, cv_folds=CV_FOLDS, seed=SEED,
        n_transitions=int(len(obs)), n_trajectories=int(traj.max() + 1),
        cost_mean=float(cost.mean()), action_norm_mean=float(sq.mean()),
        d50_median=float(np.median(d50)), T_mean=float(T.mean()), T_median=med_T,
        E1a=dict(grid_T=grid.tolist(), grid_n=cnt.tolist(), support_rows=rows,
                 exist_high_support_high_tail=exist_hi_support_hi_tail),
        E1b=dict(corr_T_d50=r_T_d50, corr_T_Anorm=r_T_A,
                 partial_corr_T_d50_given_Anorm=pc_T_d50_given_A,
                 R2_T_Anorm=r2_A, R2_T_d50=r2_d, R2_T_both=r2_both,
                 within_anorm_bins=within,
                 within_diff_mean=float(ds.mean()),
                 within_diff_same_sign=int((np.sign(ds) == np.sign(ds.mean())).sum()),
                 within_diff_n=len(ds)),
        E1c=cv_res,
        boundaries=dict(
            read_only=True, no_new_training=True, heldout_125000_untouched=True,
            note="Day21 成本是 max(0,‖a‖²−τ)，y_cost 一路天生无法排除动作范数解释；"
                 "能承担该任务的是 y_ood 一路。"),
        elapsed_sec=round(time.time() - t0, 1))
    with (OUT / "结果_E1分布内支撑与尾部风险诊断.json").open("w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    import csv
    with (OUT / "表_E1支撑度分层.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (OUT / "表_E1网格_T均值.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["支撑档\\动作范数档"] + [f"A{j}" for j in range(N_BINS)])
        for i in range(N_BINS):
            w.writerow([f"S{i}"] + [("" if np.isnan(v) else round(v, 6)) for v in grid[i]])

    # 图（英文标签，避免缺中文字体）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
    im = ax[0].imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax[0].set_xlabel("action-norm decile  $\\|a\\|^2$ (high $\\rightarrow$)")
    ax[0].set_ylabel("state-support decile (good support $\\downarrow$)")
    ax[0].set_title("local upper-tail cost $T$")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].plot([r["d50区间"][0] for r in rows], [r["每档T均值"] for r in rows],
               "o-", color="tab:red", label="$T$ (neighbour top-10%)")
    ax[1].plot([r["d50区间"][0] for r in rows], [r["每档成本均值"] for r in rows],
               "s-", color="tab:blue", label="one-step cost")
    ax[1].set_xlabel("$d_{50}$ (larger = worse support)")
    ax[1].set_ylabel("mean value")
    ax[1].legend(fontsize=7)
    ax[1].set_title("by support decile")
    # 图里只用 ASCII 标签：容器内没有 CJK 字体，中文会渲染成方块
    for 标签名, col in (("y_cost（一步成本>0）", "tab:blue"), ("y_ood（后继越界）", "tab:red")):
        图例名 = "y_cost (one-step cost>0)" if 标签名.startswith("y_cost") else "y_ood (next state off-support)"
        xs = [c["auc"] for c in cv_res[标签名]]
        ax[2].plot(range(3), xs, "o-", color=col, label=图例名)
        for i, c in enumerate(cv_res[标签名]):
            ax[2].annotate(f"{c['auc']:.3f}", (i, c["auc"]), fontsize=6,
                           textcoords="offset points", xytext=(0, 5))
    ax[2].axhline(0.5, ls=":", color="grey")
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(["s only", "A only", "s + A"], fontsize=8)
    ax[2].set_ylabel("AUC (grouped 5-fold)")
    ax[2].set_title("state carries risk info beyond $\\|a\\|^2$?")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "图_E1支撑与尾部风险.pdf")
    fig.savefig(OUT / "图_E1支撑与尾部风险.png", dpi=300)

    import shutil
    shutil.copyfile(HERE / "Day21_E1_分布内支撑与尾部风险诊断_协议_v1.0.md",
                    OUT / "协议副本_Day21_E1_v1.0.md")
    print(f"\n[E1] 完成，用时 {res['elapsed_sec']:.1f} 秒；输出 {OUT}", flush=True)


if __name__ == "__main__":
    主函数()
