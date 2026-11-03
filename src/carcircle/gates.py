#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-2 v2 夜间：G0 / G1 / G2 三道预注册闸门。"""
from __future__ import annotations
import inspect, json, sys, time
from pathlib import Path
import numpy as np

import night_lib as L


# ---------------------------------------------------------------- G0
def stage_A0(log: L.Log, status: L.Status, env_id: str) -> dict:
    log('=== A0 / G0：环境与数据技术核验 ===')
    import warnings
    warnings.filterwarnings('ignore')
    ev = {'env_id': env_id, 'at': L.now()}
    ev['versions'] = L.env_versions()
    (L.OUT / 'env_versions.txt').write_text(
        json.dumps({k: v for k, v in ev['versions'].items() if k != 'pip_freeze'},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    (L.OUT / 'env_versions.txt').open('a', encoding='utf-8').write(
        '\n\n=== pip freeze ===\n' + str(ev['versions'].get('pip_freeze', '')))

    import gymnasium as gym
    import dsrl.offline_bullet_safety_gym  # noqa: F401
    env = gym.make(env_id)
    try:
        sig = str(inspect.signature(env.get_dataset))
        env.reset(seed=510)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        out = env.step(a)
        ev['get_dataset_signature'] = sig
        ev['obs_space'] = list(env.observation_space.shape)
        ev['act_space'] = list(env.action_space.shape)
        ev['max_episode_steps'] = getattr(env.spec, 'max_episode_steps', None)
        ev['step_return_len'] = len(out)
        ev['step_return_types'] = [type(x).__name__ for x in out]
        ev['info_keys'] = sorted(out[4].keys()) if len(out) >= 5 and isinstance(out[4], dict) else None
        ev['step_api'] = ('5-tuple (obs, reward, terminated, truncated, info); cost in info["cost"]'
                          if len(out) == 5 else f'{len(out)}-tuple')
        ev['cost_source'] = "info['cost']"
        ev['positions_2_3_are_bool'] = bool(isinstance(out[2], (bool, np.bool_))
                                            and isinstance(out[3], (bool, np.bool_)))
        ev['cost_value_at_zero_action'] = float(out[4].get('cost', 0.0)) if isinstance(out[4], dict) else None
        for k in ('terminated', 'truncated'):
            ev[f'has_{k}'] = k in out[4] if isinstance(out[4], dict) else None
    finally:
        env.close()

    # 数据：只用本地文件
    p = L.DATASET[env_id]
    ev['dataset_path'] = str(p)
    ev['dataset_sha256'] = L.sha256_file(p)
    ev['dataset_bytes'] = p.stat().st_size
    ev['hf_repo'] = L.HF_REPO
    ev['hf_commit'] = L.HF_COMMIT
    env = gym.make(env_id)
    try:
        d = env.get_dataset(h5path=str(p))
    finally:
        env.close()
    need = ['observations', 'next_observations', 'actions', 'rewards', 'costs', 'terminals', 'timeouts']
    ev['dataset_keys'] = sorted(d.keys())
    ev['missing_keys'] = [k for k in need if k not in d]
    ev['n_transitions'] = int(d['observations'].shape[0])
    ev['obs_dim'] = int(d['observations'].shape[1])
    ev['act_dim'] = int(d['actions'].shape[1])
    ev['finite_all'] = bool(all(np.isfinite(d[k]).all() for k in
                                ['observations', 'next_observations', 'actions', 'rewards', 'costs']))
    ev['cost_unique'] = sorted(set(np.unique(d['costs']).tolist()))
    ev['cost_binary'] = bool(set(np.unique(d['costs']).tolist()) <= {0.0, 1.0})
    ev['timeouts_sum'] = int(np.sum(d['timeouts']))
    ev['terminals_sum'] = int(np.sum(d['terminals']))
    bnd = L.load_trajectories(d)
    ev['n_trajectories'] = len(bnd)
    ev['trajectory_lengths'] = sorted({b - a for a, b in bnd})
    ev['obs_dim_matches_env'] = bool(ev['obs_dim'] == ev['obs_space'][0]
                                     and ev['act_dim'] == ev['act_space'][0])
    # 硬断言：API 映射必须明确（第 2/3 位为 bool，且 info 中存在 cost），否则 G0 直接 FAIL
    ev['api_mapping_assert'] = bool(ev['step_return_len'] == 5 and ev['positions_2_3_are_bool']
                                    and 'cost' in (ev['info_keys'] or []))
    lhs = ev['obs_dim_matches_env'] and ev['finite_all'] and not ev['missing_keys']
    ev['G0_PASS'] = bool(lhs and ev['api_mapping_assert'])
    ev['note'] = ('第三方副本，内容指纹与 DSRL infos.py 一致；无官方字节级认证；'
                  '本次一律以 h5path 读取本地文件，未隐式访问原服务器')
    L.write_json_atomic(L.WORK / 'results' / f'G0_证据_{env_id}.json', ev)
    L.copy_to_win(f'results/G0_证据_{env_id}.json')
    log(f'G0: obs={ev["obs_space"]} act={ev["act_space"]} step={ev["step_api"]} '
        f'轨迹={ev["n_trajectories"]} 转移={ev["n_transitions"]} 维度匹配={ev["obs_dim_matches_env"]} '
        f'→ {"PASS" if ev["G0_PASS"] else "FAIL"}')
    return ev


# ---------------------------------------------------------------- G1
def stage_A1(log: L.Log, env_id: str) -> dict:
    log('=== A1 / G1：离线风险持续性闸门（最多 64 条完整轨迹，RNG=20261013）===')
    import gymnasium as gym
    import dsrl.offline_bullet_safety_gym  # noqa: F401
    env = gym.make(env_id)
    try:
        d = env.get_dataset(h5path=str(L.DATASET[env_id]))
    finally:
        env.close()
    cost, rew = d['costs'], d['rewards']
    bnd = L.load_trajectories(d)
    n_traj = len(bnd)
    rng = np.random.RandomState(L.RNG_SEED)
    n_pick = min(64, n_traj)
    picked = sorted(rng.choice(n_traj, size=n_pick, replace=False).tolist())

    rows, timing, per_traj_cost_steps = [], [], []
    for tid in picked:
        s, e = bnd[tid]
        c = cost[s:e]
        H = int(e - s)
        n_ev = int(c.sum())
        idx = np.flatnonzero(c > 0)
        cut = int(np.floor(0.1 * H))
        rows.append(dict(traj_id=tid, length=H, reward_return=float(rew[s:e].sum()),
                         cost_return=float(c.sum()), n_cost_events=n_ev,
                         first_cost_step=(int(idx[0]) if len(idx) else -1),
                         last_cost_step=(int(idx[-1]) if len(idx) else -1),
                         cost_events_after_10pct=int((idx >= cut).sum()),
                         has_any_cost_event=bool(n_ev > 0),
                         has_cost_event_after_10pct=bool((idx >= cut).any()),
                         has_non_event_window=bool(n_ev < H)))
        per_traj_cost_steps.append((tid, idx, H))
        q = [0, 0, 0, 0]
        for i in idx:
            q[min(3, int(4 * i / H))] += 1
        timing.append(dict(traj_id=tid, H=H, q0_25=q[0], q25_50=q[1], q50_75=q[2], q75_100=q[3]))

    tot_events = int(sum(r['n_cost_events'] for r in rows))
    after10 = int(sum(r['cost_events_after_10pct'] for r in rows))
    traj_with_cost = [r for r in rows if r['has_any_cost_event']]
    traj_cost_after10 = [r for r in rows if r['has_cost_event_after_10pct']]
    share_after = (after10 / tot_events) if tot_events else 0.0
    frac_traj_after = (len(traj_cost_after10) / len(traj_with_cost)) if traj_with_cost else 0.0
    qs = [sum(t[k] for t in timing) for k in ('q0_25', 'q25_50', 'q50_75', 'q75_100')]
    res = dict(
        env_id=env_id, n_trajectories_total=n_traj, n_picked=n_pick, rng_seed=L.RNG_SEED,
        picked_traj_ids=picked,
        c1_total_traj_ge_24=bool(n_traj >= 24),
        c2_traj_with_cost_ge_8=bool(len(traj_with_cost) >= 8),
        c3_traj_with_non_event_window_ge_8=bool(sum(1 for r in rows if r['has_non_event_window']) >= 8),
        c4_total_cost_events_ge_100=bool(tot_events >= 100),
        total_cost_events=tot_events,
        c5_share_events_after_10pct=round(share_after, 6),
        c5_ge_20pct=bool(share_after >= 0.20),
        c6_frac_cost_traj_with_event_after_10pct=round(frac_traj_after, 6),
        c6_ge_20pct=bool(frac_traj_after >= 0.20),
        quartile_events=dict(zip(['0-25%', '25-50%', '50-75%', '75-100%'], qs)),
        quartile_shares=[round(x / tot_events, 4) for x in qs] if tot_events else None,
        reward_return_stats=dict(mean=float(np.mean([r['reward_return'] for r in rows])),
                                 min=float(np.min([r['reward_return'] for r in rows])),
                                 max=float(np.max([r['reward_return'] for r in rows]))),
        cost_return_stats=dict(mean=float(np.mean([r['cost_return'] for r in rows])),
                               min=float(np.min([r['cost_return'] for r in rows])),
                               max=float(np.max([r['cost_return'] for r in rows]))),
        note='只读 pilot；未计算任何风险代理成绩用于选择环境')
    res['G1_PASS'] = bool(res['c1_total_traj_ge_24'] and res['c2_traj_with_cost_ge_8']
                          and res['c3_traj_with_non_event_window_ge_8']
                          and res['c4_total_cost_events_ge_100'] and res['c5_ge_20pct']
                          and res['c6_ge_20pct'])
    L.dump_csv(L.WORK / 'results' / 'pilot_trajectory.csv', rows)
    L.dump_csv(L.WORK / 'results' / 'cost_timing.csv', timing)
    L.write_json_atomic(L.WORK / 'results' / f'G1_{env_id}.json', res)
    L.copy_to_win('results/pilot_trajectory.csv')
    L.copy_to_win('results/cost_timing.csv')
    L.copy_to_win(f'results/G1_{env_id}.json')
    log('G1 判定: ' + ', '.join(f'{k}={res[k]}' for k in
        ['c1_total_traj_ge_24', 'c2_traj_with_cost_ge_8', 'c3_traj_with_non_event_window_ge_8',
         'c4_total_cost_events_ge_100', 'c5_ge_20pct', 'c6_ge_20pct']))
    log(f'G1 总事件={tot_events} 初始10%之后占比={share_after:.4f} '
        f'有成本轨迹中10%后仍发生占比={frac_traj_after:.4f} → {"PASS" if res["G1_PASS"] else "FAIL"}')
    return res


# ---------------------------------------------------------------- G2
def _bc_train(X, Y, log, tag, updates=L.BC_UPDATES, seed=0):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    na = Y.shape[1]
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, na), nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=L.LR)
    n = X.shape[0]
    rng = np.random.RandomState(1000 + seed)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    last = float('nan')
    for i in range(updates):
        j = torch.from_numpy(rng.choice(n, L.BATCH))
        loss = nn.functional.mse_loss(net(Xt[j]), Yt[j])
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if (i + 1) % 500 == 0:
            log(f'  BC[{tag}] {i+1}/{updates} loss={last:.5f}')
    return net


def _rollout(net, env, seed, arm, steps_records):
    import torch
    # Bullet-Safety-Gym 的 reset(seed=) 被忽略（builder.py:452），必须显式播种全局 numpy RNG
    np.random.seed(int(seed) % (2 ** 31 - 1))
    obs, _ = env.reset(seed=seed)
    tot_r = tot_c = 0.0
    t = 0
    done = False
    while not done:
        with torch.no_grad():
            a = net(torch.from_numpy(np.asarray(obs, np.float32)[None, :])).numpy()[0]
        a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
        obs, r, c, done, _ = L.step_env(env, a)
        tot_r += r; tot_c += c
        steps_records.append(dict(arm=arm, env_seed=seed, t=t, action_0=float(a[0]),
                                  action_1=float(a[1]), reward=r, cost=c))
        t += 1
    return tot_r, tot_c, t


def stage_A2(log: L.Log, env_id: str) -> dict:
    log('=== A2 / G2：策略级安全可行动性闸门（BC-All vs BC-Safe）===')
    import torch
    from torch import nn  # noqa: F401
    import gymnasium as gym
    import dsrl.offline_bullet_safety_gym  # noqa: F401
    env = gym.make(env_id)
    try:
        d = env.get_dataset(h5path=str(L.DATASET[env_id]))
    finally:
        env.close()
    obs, act, cost = d['observations'], d['actions'], d['costs']
    bnd = L.load_trajectories(d)
    split = {'train': [], 'dev': [], 'critic_only': []}
    for tid, (s, e) in enumerate(bnd):
        h = L.traj_hash(L.RNG_SEED, tid)
        split['train' if h < 700 else ('dev' if h < 850 else 'critic_only')].append(tid)
    cost_ret = {tid: float(cost[bnd[tid][0]:bnd[tid][1]].sum()) for tid in split['train']}
    order = sorted(split['train'], key=lambda t: (cost_ret[t], L.traj_hash(7, t)))
    safe_half = set(order[:max(1, len(order) // 2)])
    log(f'划分：train={len(split["train"])} dev={len(split["dev"])} '
        f'critic_only={len(split["critic_only"])}；BC-Safe 用 {len(safe_half)} 条最低成本轨迹')
    L.write_json_atomic(L.WORK / 'results' / 'split_A2.json',
                        {k: v for k, v in split.items()} | {'bc_safe_ids': sorted(safe_half),
                                                            'train_cost_returns': cost_ret})
    L.copy_to_win('results/split_A2.json')

    def stack(tids):
        X = np.concatenate([obs[bnd[t][0]:bnd[t][1]] for t in tids]).astype(np.float32)
        Y = np.concatenate([act[bnd[t][0]:bnd[t][1]] for t in tids]).astype(np.float32)
        return X, Y

    Xa, Ya = stack(split['train'])
    Xs, Ys = stack(sorted(safe_half))
    net_all = _bc_train(Xa, Ya, log, 'All')
    net_safe = _bc_train(Xs, Ys, log, 'Safe')

    env = gym.make(env_id)
    eps, steps = [], []
    try:
        for arm, net in (('BC-All', net_all), ('BC-Safe', net_safe)):
            for sd in L.DEV_SEEDS:
                r, c, n = _rollout(net, env, sd, arm, steps)
                H = n
                idx = np.flatnonzero([x['cost'] for x in steps[-n:]])
                cut = int(np.floor(0.1 * H))
                eps.append(dict(arm=arm, env_seed=sd, steps=n, reward_sum=r, cost_sum=c,
                                cost_after_10pct=float(sum(1 for i in idx if i >= cut)),
                                has_cost_after_10pct=bool(any(i >= cut for i in idx))))
    finally:
        env.close()
    by = {(e['arm'], e['env_seed']): e for e in eps}
    paired = []
    for sd in L.DEV_SEEDS:
        a, b = by[('BC-All', sd)], by[('BC-Safe', sd)]
        paired.append(dict(env_seed=sd, cost_all=a['cost_sum'], cost_safe=b['cost_sum'],
                           delta_cost=a['cost_sum'] - b['cost_sum'],
                           nonzero_delta=bool(abs(a['cost_sum'] - b['cost_sum']) > 1e-9),
                           any_arm_has_cost_after_10pct=bool(a['has_cost_after_10pct']
                                                             or b['has_cost_after_10pct'])))
    n_nonzero = sum(1 for p in paired if p['nonzero_delta'])
    n_after = sum(1 for p in paired if p['any_arm_has_cost_after_10pct'])
    res = dict(env_id=env_id, definition='配对口径：每个开发环境种子比较两臂；'
                                         '“初始 10% 之后仍有成本”记为该配对中任一臂满足',
               n_env_seeds=len(L.DEV_SEEDS), nonzero_cost_diff_seeds=n_nonzero,
               need_nonzero=4, pass_nonzero=bool(n_nonzero >= 4),
               seeds_with_cost_after_10pct=n_after, need_after=4, pass_after=bool(n_after >= 4),
               paired=paired, bc_updates=L.BC_UPDATES, bc_seed=0, split_sizes={k: len(v) for k, v in split.items()})
    res['G2_PASS'] = bool(res['pass_nonzero'] and res['pass_after'])
    L.dump_csv(L.WORK / 'results' / 'g2_episode.csv', eps)
    L.dump_csv(L.WORK / 'results' / 'g2_step.csv', steps)
    L.write_json_atomic(L.WORK / 'results' / f'G2_{env_id}.json', res)
    L.copy_to_win('results/g2_episode.csv')
    L.copy_to_win('results/g2_step.csv')
    L.copy_to_win(f'results/G2_{env_id}.json')
    log(f'G2：非零成本差种子 {n_nonzero}/8（需 ≥4）；10% 后仍有成本的配对 {n_after}/8（需 ≥4）'
        f' → {"PASS" if res["G2_PASS"] else "FAIL"}')
    return res
