#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：官方 CPQ（OSRL）在 OfflineCarCircle-v0 上的参考基线。

**算法与训练配方逐字节使用官方实现**（OSRL commit 03a4586…，osrl/algorithms/cpq.py
SHA a77ec391…），本文件只做三件事：
  1) 提供官方 FSRL 的日志替身（见 vendor/fsrl_stub，差异已登记）；
  2) 用本地 HDF5 作为数据源（官方脚本默认从数据服务器下载，本机数据服务不可达）；
  3) 用本项目的**已审计评估协议**（每回合新建环境 + reset 前 np.random.seed(env_seed)）
     在开发种子 510–517 上评估，使结果与其余所有臂可直接比较。

超参数取自官方 examples/configs/cpq_configs.py 的 CPQCarCircleConfig（未改动）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR = SCRIPT_DIR / 'vendor'
sys.path.insert(0, str(VENDOR / 'fsrl_stub'))
sys.path.insert(0, str(VENDOR / 'OSRL'))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import bullet_safety_gym                # noqa: E402,F401
import dsrl                             # noqa: E402,F401
import gymnasium as gym                 # noqa: E402
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa: E402
from fsrl.utils import DummyLogger      # noqa: E402
from osrl.algorithms import CPQ, CPQTrainer  # noqa: E402
from osrl.common import TransitionDataset    # noqa: E402
from osrl.common.exp_util import seed_all    # noqa: E402

import night_lib as L                   # noqa: E402

OUT = L.WORK / 'results_p1_cpq'
CK = L.WORK / 'checkpoints_p1_cpq'
H5 = str(L.DATASET['OfflineCarCircle-v0'])
ENVSEEDS = list(range(510, 518))

# 官方 CPQCarCircleConfig 默认值（未改动）
CFG = dict(task='OfflineCarCircle-v0', device='cpu', threads=4, reward_scale=0.1,
           cost_scale=1.0, actor_lr=1e-4, critic_lr=1e-3, alpha_lr=1e-4, vae_lr=1e-3,
           cost_limit=10, episode_len=300, batch_size=512, update_steps=100_000,
           a_hidden_sizes=[256, 256], c_hidden_sizes=[256, 256], vae_hidden_sizes=400,
           sample_action_num=10, gamma=0.99, tau=0.005, beta=0.5, num_q=2, num_qc=2,
           qc_scalar=1.5, eval_every=2500)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def evaluate_audited(model, seeds, tag, rows):
    """本项目已审计协议：每回合新建环境 + reset 前播种。"""
    for es in seeds:
        env = gym.make(CFG['task'])
        try:
            np.random.seed(int(es) % (2 ** 31 - 1))
            obs, _ = env.reset(seed=es)
            r_tot = c_tot = 0.0
            events, t, done = [], 0, False
            while not done:
                # 官方 CPQ.act() 在 with_logprob=False（默认）时会对 None 调 .data 而崩溃
                # （osrl/algorithms/cpq.py:250）。这里走**公共 API 的安全分支** with_logprob=True，
                # 算法模块本身未做任何修改。
                a, _ = model.act(obs.astype(np.float32), deterministic=True, with_logprob=True)
                a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
                obs, r, c, done, _ = L.step_env(env, a)
                # 官方用 reward_scale 缩放奖励；成本不缩放（cost_scale=1）
                r_tot += r * CFG['reward_scale']
                c_tot += c
                if c > 0:
                    events.append(t)
                t += 1
            cut = int(np.floor(0.1 * t))
            rows.append(dict(arm='CPQ', train_seed=tag, env_seed=es, steps=t,
                             reward_sum=r_tot, cost_sum=c_tot, cost_events=len(events),
                             cost_after_10pct=sum(1 for i in events if i >= cut)))
        finally:
            env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--update-steps', type=int, default=None)
    ap.add_argument('--eval-only-from', default=None, help='从已有检查点只做评估')
    ap.add_argument('--skip-eval', action='store_true')
    ap.add_argument('--threads', type=int, default=0,
                    help='覆盖 torch 线程数（资源参数；0=用官方配置里的 4）')
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CK.mkdir(parents=True, exist_ok=True)
    steps = args.update_steps or CFG['update_steps']
    seed_all(args.seed)
    if CFG['device'] == 'cpu':
        torch.set_num_threads(args.threads or CFG['threads'])
    log = L.Log('p1_cpq_seed%d' % args.seed)
    log('CPQ 官方配方启动 seed=%d steps=%d（数据源=本地 HDF5）' % (args.seed, steps))

    env = gym.make(CFG['task'])
    data = env.get_dataset(h5path=H5)          # ← 唯一的数据来源适配（登记在案）
    env.set_target_cost(CFG['cost_limit'])
    data = env.pre_process_data(data, None, None, None, None, 1.0,
                                cbins=None, rbins=None, max_npb=None, min_npb=None)
    env = wrap_env(env=env, reward_scale=CFG['reward_scale'])
    env = OfflineEnvWrapper(env)
    log(' 数据就绪：transitions=%d obs=%d act=%d' %
        (len(data['observations']), env.observation_space.shape[0], env.action_space.shape[0]))

    model = CPQ(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                a_hidden_sizes=CFG['a_hidden_sizes'], c_hidden_sizes=CFG['c_hidden_sizes'],
                vae_hidden_sizes=CFG['vae_hidden_sizes'],
                sample_action_num=CFG['sample_action_num'], gamma=CFG['gamma'],
                tau=CFG['tau'], beta=CFG['beta'], num_q=CFG['num_q'], num_qc=CFG['num_qc'],
                qc_scalar=CFG['qc_scalar'], cost_limit=CFG['cost_limit'],
                episode_len=CFG['episode_len'], device=CFG['device'])
    log(' 模型参数 %d' % sum(p.numel() for p in model.parameters()))
    ckpt = CK / ('cpq_seed%d.pt' % args.seed)
    if args.eval_only_from and Path(args.eval_only_from).is_file():
        model.load_state_dict(torch.load(args.eval_only_from, map_location='cpu')['model_state'])
        log(' 从检查点载入：%s' % args.eval_only_from)
        steps = 0
    else:
        trainer = CPQTrainer(model, env, logger=DummyLogger(), actor_lr=CFG['actor_lr'],
                             critic_lr=CFG['critic_lr'], alpha_lr=CFG['alpha_lr'],
                             vae_lr=CFG['vae_lr'], reward_scale=CFG['reward_scale'],
                             cost_scale=CFG['cost_scale'], device=CFG['device'])
        dataset = TransitionDataset(data, reward_scale=CFG['reward_scale'],
                                    cost_scale=CFG['cost_scale'])
        # 官方脚本同样不使用 shuffle（TransitionDataset 是 IterableDataset）
        loader = DataLoader(dataset, batch_size=CFG['batch_size'],
                            num_workers=0, pin_memory=False)
        it = iter(loader)
        t1 = time.time()
        hist = []
        for step in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            obs, nobs, act_, rew, cost, done = [b.to(CFG['device']) for b in batch]
            trainer.train_one_step(obs, nobs, act_, rew, cost, done)
            if (step + 1) % 500 == 0:
                rate = (time.time() - t1) / (step + 1)
                log('  step %d/%d  %.1f ms/step  预计剩余 %.0f 分钟'
                    % (step + 1, steps, rate * 1000, rate * (steps - step - 1) / 60))
            if (step + 1) % CFG['eval_every'] == 0:
                torch.save(dict(model_state=model.state_dict(), step=step + 1, seed=args.seed,
                                cfg=CFG), ckpt)
                log('  已保存检查点 %s' % ckpt)
                # 周期性审计评估（开发种子 510-517），用于判断缩减预算是否足够
                if not args.skip_eval:
                    rows_p = []
                    evaluate_audited(model, ENVSEEDS, args.seed, rows_p)
                    mc = float(np.mean([r['cost_sum'] for r in rows_p]))
                    mr = float(np.mean([r['reward_sum'] for r in rows_p]))
                    hist.append(dict(step=step + 1, cost_mean=mc, reward_mean=mr))
                    log('  [曲线] step %d 成本 %.2f 回报 %.2f' % (step + 1, mc, mr))
        torch.save(dict(model_state=model.state_dict(), step=steps, seed=args.seed, cfg=CFG), ckpt)
        log(' 训练完成，用时 %.0f 秒' % (time.time() - t1))
        if hist:
            (OUT / ('curve_seed%d.json' % args.seed)).write_text(
                json.dumps(hist, ensure_ascii=False, indent=2), encoding='utf-8')

    if not args.skip_eval:
        rows = []
        evaluate_audited(model, ENVSEEDS, args.seed, rows)
        ep = OUT / ('eval_episode_seed%d.csv' % args.seed)
        with ep.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        mc = float(np.mean([r['cost_sum'] for r in rows]))
        mr = float(np.mean([r['reward_sum'] for r in rows]))
        cv = float(np.mean(sorted(r['cost_sum'] for r in rows)[:2]))
        log(' 评估（开发种子 510-517）：成本均值 %.2f 回报均值 %.2f CVaR25 %.2f' % (mc, mr, cv))
        summ = OUT / ('summary_seed%d.json' % args.seed)
        summ.write_text(json.dumps(dict(seed=args.seed, update_steps=steps, cost_mean=mc,
                                        reward_mean=mr, cvar25=cv,
                                        ckpt=ckpt.name, ckpt_sha256=sha(ckpt) if ckpt.is_file() else None,
                                        osrl_commit='03a4586463690566ec8d7830ca4027b46d4ad02f',
                                        cpq_sha256='a77ec39148c5ae151142897fb9f2f362f4220592f875536abcc38ee63089187b',
                                        data_h5path=H5,
                                        seconds=round(time.time() - t0, 1)), ensure_ascii=False,
                                       indent=2), encoding='utf-8')
    print('CPQ seed%d 完成，用时 %.0f 秒' % (args.seed, time.time() - t0))


if __name__ == '__main__':
    main()
