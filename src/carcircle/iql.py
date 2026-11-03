#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-2 v2 night: standard IQL full policy baseline (P1) and mechanism arms (P2).

P1 standard IQL: Q target = r + gamma(1-d)V(s') with online V; V fitted by expectile(tau=0.7)
onto target Q; actor = behaviour cloning weighted by exp(beta*(min Q - V)), beta=3.0, cap 100.
P2 mechanism arms: freeze the same-seed critic, compare Reward-only / D10 / E100 with the
D10-paper operator (double in-batch z-score, beta=0.5, lambda=1, mean-normalised, cap 100).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import night_lib as L


def load_and_split(env_id: str) -> dict:
    import warnings
    warnings.filterwarnings('ignore')
    import gymnasium as gym
    import dsrl.offline_bullet_safety_gym  # noqa: F401
    env = gym.make(env_id)
    try:
        d = env.get_dataset(h5path=str(L.DATASET[env_id]))
    finally:
        env.close()
    bnd = L.load_trajectories(d)
    split = {'train': [], 'dev': [], 'critic_only': []}
    for tid in range(len(bnd)):
        h = L.traj_hash(L.RNG_SEED, tid)
        split['train' if h < 700 else ('dev' if h < 850 else 'critic_only')].append(tid)

    def idx(tids):
        if not tids:
            return np.array([], dtype=np.int64)
        return np.concatenate([np.arange(bnd[t][0], bnd[t][1]) for t in tids])

    out = dict(bounds=bnd, split=split, n_traj=len(bnd))
    out['train_idx'] = idx(split['train'])
    out['critic_idx'] = idx(split['critic_only'])
    out['S'] = d['observations'].astype(np.float32)
    out['A'] = d['actions'].astype(np.float32)
    out['R'] = d['rewards'].astype(np.float32)
    out['S2'] = d['next_observations'].astype(np.float32)
    out['D'] = ((d['terminals'] == 1) | (d['timeouts'] == 1)).astype(np.float32)
    out['C'] = d['costs'].astype(np.float32)
    tr = out['train_idx']
    out['obs_mean'] = out['S'][tr].mean(0)
    out['obs_std'] = np.maximum(out['S'][tr].std(0), 1e-6)
    return out


def mlp(i, o, h=256, act=nn.ReLU):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))


def expectile_loss(diff, tau):
    return (torch.where(diff > 0, tau, 1.0 - tau) * diff.square()).mean()


class Actor(nn.Module):
    """tanh Gaussian policy; log_std is a state-independent learnable parameter."""

    def __init__(self, no, na, log_std_init=-1.0):
        super().__init__()
        self.net = mlp(no, na, 256, nn.Tanh)
        self.log_std = nn.Parameter(torch.full((na,), float(log_std_init)))

    def forward(self, s):
        return self.net(s)

    def log_prob(self, s, a):
        mu = self.net(s)
        std = self.log_std.exp().clamp(1e-3, 1.0)
        return (-0.5 * ((a - mu) / std).square() - torch.log(std)
                - 0.5 * math.log(2 * math.pi)).sum(-1)


def _diag_critic(q1, q2, v, data, norm, S, A, R, S2, D):
    with torch.no_grad():
        ci = torch.from_numpy(data['critic_idx'].astype(np.int64))
        s, a = norm(S[ci]), A[ci]
        qv = torch.minimum(q1(torch.cat([s, a], -1)), q2(torch.cat([s, a], -1)))
        y = R[ci] + L.GAMMA * (1 - D[ci]) * v(norm(S2[ci]))
        td = float(torch.sqrt((y - qv).square().mean()))
        aq = []
        for off in range(0, len(S), 20000):
            ss = norm(S[off:off + 20000])
            aa = A[off:off + 20000]
            aq.append(torch.minimum(q1(torch.cat([ss, aa], -1)), q2(torch.cat([ss, aa], -1))))
        aq = torch.cat(aq)
        vv = v(norm(S[ci]))
    return dict(td_rmse_critic_holdout=td, q_absmax=float(aq.abs().max()),
                q_mean=float(aq.mean()), v_absmax=float(vv.abs().max()),
                finite=bool(torch.isfinite(aq).all()))


def stage_P1(log: L.Log, data: dict, seed: int):
    tag = f'P1_seed{seed}'
    ck = L.WORK / 'checkpoints' / tag
    ck.mkdir(parents=True, exist_ok=True)
    final = ck / 'critic_step40000.pt'
    if final.is_file():                      # 断点续跑：已有 40k 则跳过重训
        try:
            torch.load(final, map_location='cpu', weights_only=False)
            log('P1 critic seed%d 已存在 40k 检查点，跳过重训' % seed)
            return final
        except Exception:
            log('P1 critic seed%d 的 40k 检查点不可用，重新训练' % seed)
    metrics = []
    torch.manual_seed(seed)
    rng = np.random.RandomState(20261013 + seed)
    no, na = data['S'].shape[1], data['A'].shape[1]
    q1, q2, v = mlp(no + na, 1), mlp(no + na, 1), mlp(no, 1)
    q1t, q2t = mlp(no + na, 1), mlp(no + na, 1)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    for m in (q1t, q2t):
        for p in m.parameters():
            p.requires_grad_(False)
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=L.LR)
    ov = torch.optim.Adam(v.parameters(), lr=L.LR)
    tr = torch.from_numpy(data['train_idx'].astype(np.int64))
    om = torch.from_numpy(data['obs_mean'])
    osd = torch.from_numpy(data['obs_std'])
    S = torch.from_numpy(data['S'])
    A = torch.from_numpy(data['A'])
    R = torch.from_numpy(data['R']).reshape(-1, 1)
    S2 = torch.from_numpy(data['S2'])
    D = torch.from_numpy(data['D']).reshape(-1, 1)
    norm = lambda x: (x - om) / osd

    metrics.append({'arm': 'standard_iql_critic', 'train_seed': seed, 'step': 0,
                    **_diag_critic(q1, q2, v, data, norm, S, A, R, S2, D)})
    torch.save(dict(q1=q1.state_dict(), q2=q2.state_dict(), v=v.state_dict(),
                    obs_mean=data['obs_mean'], obs_std=data['obs_std'], step=0),
               ck / 'critic_step0.pt')
    steps = 0
    t0 = time.time()
    for target in (2000, 10000, 20000, 40000):
        while steps < target:
            j = tr[torch.from_numpy(rng.choice(len(tr), L.BATCH))]
            with torch.no_grad():
                tv = R[j] + L.GAMMA * (1 - D[j]) * v(norm(S2[j]))
                tq = torch.minimum(q1t(torch.cat([norm(S[j]), A[j]], -1)),
                                   q2t(torch.cat([norm(S[j]), A[j]], -1)))
            qa = torch.cat([norm(S[j]), A[j]], -1)
            lq = (q1(qa) - tv).square().mean() + (q2(qa) - tv).square().mean()
            lv = expectile_loss(tq - v(norm(S[j])), L.EXPECTILE_TAU)
            loss = lq + lv
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite critic loss seed=%d step=%d' % (seed, steps))
            oq.zero_grad()
            ov.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.0)
            nn.utils.clip_grad_norm_(v.parameters(), 10.0)
            oq.step()
            ov.step()
            with torch.no_grad():
                for a_, b_ in ((q1, q1t), (q2, q2t)):
                    for pa, pb in zip(a_.parameters(), b_.parameters()):
                        pb.mul_(1 - L.POLYAK).add_(pa, alpha=L.POLYAK)
            steps += 1
        d_ = _diag_critic(q1, q2, v, data, norm, S, A, R, S2, D)
        metrics.append({'arm': 'standard_iql_critic', 'train_seed': seed, 'step': steps, **d_})
        torch.save(dict(q1=q1.state_dict(), q2=q2.state_dict(), v=v.state_dict(),
                        obs_mean=data['obs_mean'], obs_std=data['obs_std'], step=steps,
                        cfg=dict(gamma=L.GAMMA, tau=L.EXPECTILE_TAU, polyak=L.POLYAK, lr=L.LR,
                                 batch=L.BATCH, done_mask='terminals|timeouts', v_target='online')),
                   ck / ('critic_step%d.pt' % steps))
        log('P1 critic seed%d step %d done | TD(holdout)=%.4f |Q|max=%.2f elapsed %.0fs'
            % (seed, steps, d_['td_rmse_critic_holdout'], d_['q_absmax'], time.time() - t0))
        L.copy_to_win('checkpoints/%s/critic_step%d.pt' % (tag, steps))
    L.dump_csv(L.WORK / 'results' / ('train_metrics_critic_seed%d.csv' % seed), metrics)
    L.copy_to_win('results/train_metrics_critic_seed%d.csv' % seed)
    return ck / 'critic_step40000.pt'


def _safez(a):
    a = np.asarray(a, np.float64)
    # AG-RW 协议定义：分母为 std.clamp_min(1e-6)（不是 std + 1e-6）
    return (a - a.mean()) / max(float(a.std()), 1e-6)


def weights_standard_iql(adv):
    # 形状安全：一律拉平成一维，避免与 log_prob (n,) 相乘时广播成 (n,n)
    a = adv.reshape(-1) if torch.is_tensor(adv) else torch.from_numpy(np.asarray(adv, np.float32).reshape(-1))
    return torch.clamp(torch.exp(L.IQL_TEMPERATURE * a), max=L.WEIGHT_CAP)


def weights_mechanism(adv, risk, beta=L.MECH_BETA, lam=L.MECH_LAMBDA, gate=None):
    """AG-RW 加权算子。

    u = z( z(A) - lam * g * z(P) ),  w = clip(exp(beta*u)/mean, 0, cap)

    `gate` 为逐样本 0/1 门控（协议里的 g）。省略时等价于 g≡1（即常规风险臂）。
    注意：门控必须作用在**已标准化**的 z(P) 上；若把 g 乘在原 P 上再标准化，
    结果与协议不同（AG-RW 实现闸门 A0 已量化该差异）。
    """
    adv = np.asarray(adv, np.float64).reshape(-1)
    base = _safez(adv)
    if risk is not None:
        zr = _safez(np.asarray(risk, np.float64).reshape(-1))
        if gate is not None:
            zr = zr * np.asarray(gate, np.float64).reshape(-1)
        base = base - lam * zr
    u = _safez(base)
    x = np.exp(beta * u - (beta * u).max())
    x = x / x.mean()
    return np.minimum(x, L.WEIGHT_CAP)


def train_actor(critic_path, data, seed, arm, log: L.Log, risk=None,
                updates=L.ACTOR_UPDATES, operator='iql', eval_cb=None, ckpt_root=None,
                gate_full=None, extra_log=None, gate_shuffle_seed=None,
                lam=None, beta=None):
    # lam/beta 为 P2 探索性对照新增的可选覆盖（默认 None = 沿用 MECH_LAMBDA/MECH_BETA 的
    # 定义期默认值），不传时行为与改动前逐位一致。
    # 自产检查点含 numpy 数组；torch>=2.6 默认 weights_only=True 会拒绝，显式关闭
    ck = torch.load(critic_path, map_location='cpu', weights_only=False)
    no, na = data['S'].shape[1], data['A'].shape[1]
    q1, q2, v = mlp(no + na, 1), mlp(no + na, 1), mlp(no, 1)
    q1.load_state_dict(ck['q1'])
    q2.load_state_dict(ck['q2'])
    v.load_state_dict(ck['v'])
    for m in (q1, q2, v):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    torch.manual_seed(1000 + seed)
    actor = Actor(no, na)
    opt = torch.optim.Adam(actor.parameters(), lr=L.LR)
    tr = data['train_idx'].astype(np.int64)
    om = torch.from_numpy(data['obs_mean'])
    osd = torch.from_numpy(data['obs_std'])
    S = torch.from_numpy(data['S'])
    A = torch.from_numpy(data['A'])
    norm = lambda x: (x - om) / osd
    streams = np.random.RandomState(777 + seed).choice(len(tr), L.BATCH * updates).reshape(updates, L.BATCH)
    root = Path(ckpt_root) if ckpt_root else (L.WORK / 'checkpoints')
    out_dir = root / ('%s_seed%d' % (arm, seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(updates + 1):
        if i in (0, 2000, 5000, 10000):
            torch.save(dict(actor=actor.state_dict(), arm=arm, seed=seed, step=i,
                            cfg=dict(operator=operator,
                                     beta=(L.IQL_TEMPERATURE if operator == 'iql' else L.MECH_BETA),
                                     lam=(None if operator == 'iql' else L.MECH_LAMBDA),
                                     cap=L.WEIGHT_CAP, lr=L.LR, batch=L.BATCH)),
                       out_dir / ('actor_step%d.pt' % i))
            L.copy_to_win('checkpoints/%s_seed%d/actor_step%d.pt' % (arm, seed, i))
            log('  actor[%s seed%d] checkpoint step%d' % (arm, seed, i))
            if eval_cb is not None:
                eval_cb(actor, i)
        if i == updates:
            break
        j = tr[streams[i]]
        sj, aj = norm(S[j]), A[j]
        with torch.no_grad():
            adv = torch.minimum(q1(torch.cat([sj, aj], -1)),
                                q2(torch.cat([sj, aj], -1))) - v(sj)
            if operator == 'iql':
                w = weights_standard_iql(adv)
                g_used = None
            else:
                g = None if gate_full is None else gate_full[j].astype(np.float64)
                if g is not None and gate_shuffle_seed is not None:
                    # 批内置乱：保留该批的门控开启数量，只随机化位置
                    # 注意 numpy 的 seed 上限为 2**32-1，需取模避免 ValueError
                    sd_ = (int(gate_shuffle_seed) * 100003 + i) % (2 ** 31 - 1)
                    g = np.random.RandomState(sd_).permutation(g)
                kw = {}
                if lam is not None:
                    kw['lam'] = lam
                if beta is not None:
                    kw['beta'] = beta
                w = torch.from_numpy(weights_mechanism(adv.numpy(),
                                                       None if risk is None else risk[j],
                                                       gate=g, **kw))
                g_used = g
        loss = -(w * actor.log_prob(sj, aj)).mean()
        if w.shape != actor.log_prob(sj, aj).shape:      # 防御：形状必须逐样本对齐
            raise RuntimeError('权重形状与 log_prob 不匹配: %s vs %s'
                               % (tuple(w.shape), tuple(actor.log_prob(sj, aj).shape)))
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite actor loss %s seed%d step%d' % (arm, seed, i))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        opt.step()
        if i % 100 == 0 or i < 20:
            p = (w / w.sum()).numpy()
            cj = data['C'][j] > 0
            row = dict(arm=arm, train_seed=seed, updates=i + 1,
                       ess=float(1 / np.square(p).sum()), top_mass=float(p.max()),
                       clip_frac=float((w >= L.WEIGHT_CAP - 1e-7).float().mean()),
                       weight_mean=float(w.mean()), loss=float(loss.detach()),
                       cost_sample_share=float(cj.mean()),
                       cost_weight_share=float(p[cj].sum()) if cj.any() else 0.0)
            if extra_log is not None:
                row.update(extra_log(j, p, g_used))
            rows.append(row)
        if (i + 1) % 2000 == 0:
            log('  actor[%s seed%d] %d/%d loss=%.5f' % (arm, seed, i + 1, updates,
                                                        float(loss.detach())))
    L.dump_csv(L.WORK / 'results' / ('actor_diag_%s_seed%d.csv' % (arm, seed)), rows)
    L.copy_to_win('results/actor_diag_%s_seed%d.csv' % (arm, seed))
    return actor


def rollout(actor, env, seed, arm, train_seed, milestone, steps_out):
    # CarCircle(bullet_safety_gym) 的 reset(seed=) 被忽略：builder.py:452 接收 seed 但函数体不使用，
    # 任务随机化依赖全局 numpy RNG。因此在 reset 前显式播种，才能让"环境种子"真正决定回合。
    np.random.seed(int(seed) % (2 ** 31 - 1))
    obs, _ = env.reset(seed=seed)
    tot_r = tot_c = 0.0
    t = 0
    done = False
    while not done:
        with torch.no_grad():
            a = actor.net(torch.from_numpy(np.asarray(obs, np.float32)[None, :])).numpy()[0]
        a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
        obs, r, c, done, _ = L.step_env(env, a)
        tot_r += r
        tot_c += c
        steps_out.append(dict(arm=arm, train_seed=train_seed, milestone=milestone,
                              env_seed=seed, t=t, action_0=float(a[0]), action_1=float(a[1]),
                              reward=r, cost=c))
        t += 1
    return tot_r, tot_c, t


def evaluate(actor, env, arm, train_seed, milestone, steps_out):
    rows = []
    for sd in L.DEV_SEEDS:
        n0 = len(steps_out)
        r, c, n = rollout(actor, env, sd, arm, train_seed, milestone, steps_out)
        cuts = [x['t'] for x in steps_out[n0:] if x['cost'] > 0]
        cut = int(np.floor(0.1 * n))
        rows.append(dict(arm=arm, train_seed=train_seed, milestone=milestone, env_seed=sd,
                         steps=n, reward_sum=r, cost_sum=c,
                         cost_after_10pct=float(sum(1 for i in cuts if i >= cut)),
                         cost_events=float(len(cuts))))
    return rows


def compute_risk_proxies(data, log: L.Log, k=500, tail=0.10):
    """D10 / E100 on the train split with a frozen reference pool and radius."""
    from scipy.spatial import cKDTree
    tr = data['train_idx'].astype(np.int64)
    sm, ss = data['S'][tr].mean(0), np.maximum(data['S'][tr].std(0), 1e-8)
    am, asd = data['A'][tr].mean(0), np.maximum(data['A'][tr].std(0), 1e-8)
    feats = np.concatenate([(data['S'][tr] - sm) / ss, 20.0 * (data['A'][tr] - am) / asd], axis=-1)
    t0 = time.time()
    tree = cKDTree(feats)
    n = len(feats)
    j = min(k, len(feats) - 1)
    # 半径校准：从参考池中抽 5000 个查询点估计 k 近邻距离中位数（全量 30 万点会占用 GB 级内存）
    calib = np.random.RandomState(L.RNG_SEED).choice(n, size=min(5000, n), replace=False)
    dist, _ = tree.query(feats[calib], k=j + 1, workers=-1)
    radius = float(np.median(dist[:, -1]))
    cost = data['C'][tr]
    d10 = np.full(n, np.nan)
    e100 = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=np.int32)
    for lo in range(0, n, 256):
        nb = tree.query_ball_point(feats[lo:lo + 256], radius, workers=-1)
        for i2, neigh in enumerate(nb):
            idx = lo + i2
            cnt[idx] = len(neigh)
            if len(neigh):
                cc = cost[np.asarray(neigh, dtype=np.intp)]
                take = max(1, int(math.ceil(tail * len(cc))))
                d10[idx] = float(np.partition(cc, -take)[-take:].mean())
                e100[idx] = float(cc.mean())
    meta = dict(k=k, tail=tail, radius=radius, pool_size=n, query='train split (self-inclusive)',
                radius_calibration='5000 个随机池内查询点的 k 近邻距离中位数（RNG=%d）' % L.RNG_SEED,
                empty_fraction=float((cnt == 0).mean()), seconds=round(time.time() - t0, 1),
                d10_nonzero_fraction=float(np.nan_to_num(d10) > 0).mean() if False else
                float((np.nan_to_num(d10, nan=0.0) > 0).mean()),
                geometry='(z(s), 20 z(a))')
    log('风险代理：池=%d 半径=%.5g 空邻域比例=%.4f D10正值比例=%.4f 用时%.0fs'
        % (n, radius, meta['empty_fraction'], meta['d10_nonzero_fraction'], meta['seconds']))
    # 展开成"全局转移下标"可直接索引的数组（池外位置为 0，不参与训练采样）
    n_all = len(data['S'])
    full_d10 = np.zeros(n_all, np.float64)
    full_e100 = np.zeros(n_all, np.float64)
    full_empty = np.ones(n_all, bool)
    full_d10[tr] = np.nan_to_num(d10, nan=0.0)
    full_e100[tr] = np.nan_to_num(e100, nan=0.0)
    full_empty[tr] = (cnt == 0)
    return dict(d10=full_d10, e100=full_e100, empty=full_empty, neighbor_count=cnt,
                meta=meta, train_idx=tr)
