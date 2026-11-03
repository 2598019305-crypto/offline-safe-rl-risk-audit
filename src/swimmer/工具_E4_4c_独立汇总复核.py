#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-4c 独立汇总复核（只读原始归档，另开目录写汇总）。

不改动 E4-4c 原始结果目录；输出汇总 JSON/CSV 到 --out。
统计单位：训练批次流(0,1,2) × 开发环境种子(410..417) 配对；
报告中同时给出“按环境种子配对”的差值均值与“按批次流”的三条独立均值，
避免把 24 个格子当作独立样本。
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def rows(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f(x):
    return float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Output exists, refusing overwrite: {args.out}')
    args.out.mkdir(parents=True)
    R = args.results

    ep = rows(R / '逐回合_配对闭环.csv')
    we = rows(R / '训练期权重与ESS.csv')
    gr = rows(R / '演员初始梯度对照.csv')
    ac = rows(R / '固定状态动作变化.csv')

    MILESTONES = (100, 500, 2000)
    dev = sorted({int(r['env_seed']) for r in ep})
    streams = sorted({int(r['actor_seed']) for r in ep if int(r['actor_seed']) >= 0})
    arms = sorted({r['arm'] for r in ep if r['arm'] != 'PPO_frozen'})

    idx = {(int(r['actor_seed']), r['arm'], int(r['updates']), int(r['env_seed'])): r for r in ep}
    frozen = {int(r['env_seed']): r for r in ep if r['arm'] == 'PPO_frozen'}

    def agg(sel):
        return dict(
            n=len(sel),
            return_sum=float(np.mean([f(r['return_sum']) for r in sel])),
            cost_sum=float(np.mean([f(r['cost_sum']) for r in sel])),
            reward_t0_39=float(np.mean([f(r['reward_t0_39']) for r in sel])),
            reward_t100_plus=float(np.mean([f(r['reward_t100_plus']) for r in sel])),
            cost_t0_39=float(np.mean([f(r['cost_t0_39']) for r in sel])),
            cost_t40_99=float(np.mean([f(r['cost_t40_99']) for r in sel])),
            event_starts=float(np.mean([f(r['event_starts']) for r in sel])),
        )

    out = {'evaluation_seeds': dev, 'actor_streams': streams, 'arms': arms}
    out['PPO_frozen'] = agg(list(frozen.values()))

    # 各臂：原始绝对水平 + 相对冻结起点 / 相对同 beta reward-only 的配对差
    per_arm = []
    for arm in arms:
        beta = float(arm.split('_')[0][4:])
        base = f'beta{beta:g}_NONE'
        for ms in MILESTONES:
            sel = [idx[(s, arm, ms, d)] for s in streams for d in dev]
            rec = dict(arm=arm, beta=beta, updates=ms, **agg(sel))
            d_frozen = [f(idx[(s, arm, ms, d)]['return_sum']) - f(frozen[d]['return_sum']) for s in streams for d in dev]
            c_frozen = [f(idx[(s, arm, ms, d)]['cost_sum']) - f(frozen[d]['cost_sum']) for s in streams for d in dev]
            rec['delta_return_vs_PPO_frozen'] = float(np.mean(d_frozen))
            rec['delta_cost_vs_PPO_frozen'] = float(np.mean(c_frozen))
            rec['delta_return_vs_frozen_sd_over_env_seeds'] = float(
                np.mean([np.std([f(idx[(s, arm, ms, d)]['return_sum']) - f(frozen[d]['return_sum']) for d in dev], ddof=1)
                         for s in streams]))
            rec['delta_cost_vs_frozen_sd_over_env_seeds'] = float(
                np.mean([np.std([f(idx[(s, arm, ms, d)]['cost_sum']) - f(frozen[d]['cost_sum']) for d in dev], ddof=1)
                         for s in streams]))
            rec['per_stream_delta_return_vs_frozen'] = [
                float(np.mean([f(idx[(s, arm, ms, d)]['return_sum']) - f(frozen[d]['return_sum']) for d in dev]))
                for s in streams]
            rec['per_stream_delta_cost_vs_frozen'] = [
                float(np.mean([f(idx[(s, arm, ms, d)]['cost_sum']) - f(frozen[d]['cost_sum']) for d in dev]))
                for s in streams]
            if arm != base:
                dr = [f(idx[(s, arm, ms, d)]['return_sum']) - f(idx[(s, base, ms, d)]['return_sum']) for s in streams for d in dev]
                dc = [f(idx[(s, arm, ms, d)]['cost_sum']) - f(idx[(s, base, ms, d)]['cost_sum']) for s in streams for d in dev]
                rec['delta_return_vs_same_beta_reward_only'] = float(np.mean(dr))
                rec['delta_cost_vs_same_beta_reward_only'] = float(np.mean(dc))
                rec['per_stream_delta_return_vs_reward_only'] = [
                    float(np.mean([f(idx[(s, arm, ms, d)]['return_sum']) - f(idx[(s, base, ms, d)]['return_sum']) for d in dev]))
                    for s in streams]
                rec['per_stream_delta_cost_vs_reward_only'] = [
                    float(np.mean([f(idx[(s, arm, ms, d)]['cost_sum']) - f(idx[(s, base, ms, d)]['cost_sum']) for d in dev]))
                    for s in streams]
            per_arm.append(rec)
    out['per_arm'] = per_arm

    # 权重/ESS：按臂取平均（update 1 / 100 / 500 / 2000 与全部）
    wsum = []
    for arm in arms:
        for ms in sorted({int(r['updates']) for r in we}):
            sel = [r for r in we if r['arm'] == arm and int(r['updates']) == ms]
            if not sel:
                continue
            wsum.append(dict(arm=arm, updates=ms, n=len(sel),
                             batch_ess=float(np.mean([f(r['batch_ess']) for r in sel])),
                             top_sample_mass=float(np.mean([f(r['top_sample_mass']) for r in sel])),
                             early40_weight_share=float(np.mean([f(r['early40_weight_share']) for r in sel])),
                             clip_frac=float(np.mean([f(r['clip_frac']) for r in sel]))))
    out['weights'] = wsum

    out['initial_gradients'] = [{k: v for k, v in r.items()} for r in gr]
    out['action_distance'] = [
        dict(arm=r['arm'], actor_seed=int(r['actor_seed']), updates=int(r['updates']),
             mean_action_distance_to_ppo=f(r['mean_action_distance_to_ppo']),
             max_abs_action=f(r['max_abs_action'])) for r in ac]
    out['action_distance_by_arm_2000'] = [
        dict(arm=arm,
             mean_distance=float(np.mean([f(r['mean_action_distance_to_ppo']) for r in ac
                                          if r['arm'] == arm and int(r['updates']) == 2000])))
        for arm in arms]

    (args.out / '独立复核汇总_E4_4c.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    # 控制台表格
    print('PPO_frozen 8 回合均值: return %.3f cost %.3f (t0-39 cost %.3f, t40-99 %.3f)'
          % (out['PPO_frozen']['return_sum'], out['PPO_frozen']['cost_sum'],
             out['PPO_frozen']['cost_t0_39'], out['PPO_frozen']['cost_t40_99']))
    print()
    print('%-16s %6s %9s %9s %11s %10s %11s %10s' % (
        'arm', 'upd', 'return', 'cost', 'dRet_vs0', 'dCost_vs0', 'dRet_vsNONE', 'dCost_vsNONE'))
    for r in per_arm:
        print('%-16s %6d %9.3f %9.3f %11.3f %10.3f %11s %10s' % (
            r['arm'], r['updates'], r['return_sum'], r['cost_sum'],
            r['delta_return_vs_PPO_frozen'], r['delta_cost_vs_PPO_frozen'],
            ('%.3f' % r['delta_return_vs_same_beta_reward_only']) if 'delta_return_vs_same_beta_reward_only' in r else '-',
            ('%.3f' % r['delta_cost_vs_same_beta_reward_only']) if 'delta_cost_vs_same_beta_reward_only' in r else '-'))
    print()
    for r in wsum:
        if r['updates'] in (1, 100, 2000):
            print('权重 %-16s update=%-5d ESS=%.3f 前40步份额=%.4f 单样本最大=%.3f 触上限比例=%.4f' % (
                r['arm'], r['updates'], r['batch_ess'], r['early40_weight_share'],
                r['top_sample_mass'], r['clip_frac']))
    print()
    for r in gr:
        print('初始梯度 %-16s 与同 beta reward-only 的余弦=%.4f  |g|=%.4g' % (
            r['arm'], f(r['gradient_cosine_to_same_beta_reward_only']), f(r['gradient_l2'])))
    print()
    for r in out['action_distance_by_arm_2000']:
        print('2000 步动作位移 %-16s %.4f' % (r['arm'], r['mean_distance']))


if __name__ == '__main__':
    main()
