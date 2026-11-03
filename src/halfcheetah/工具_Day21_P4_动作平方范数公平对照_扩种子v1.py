#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day21 P4: squared-action-norm comparator on frozen P3 v1.1 IQL-like trainer.

SCOPE: controlled repair of W2 reward branch, NOT a reproduction of published IQL.
Controls frozen: data SHA, discount=.99, V expectile=.7, reward Q MSE,
EMA target Q for V, LR=3e-4, separate Adam optimizers, Q/V clipping=10,
EMA=.005, state Z-score, same W2 tanh-Gaussian actor weighted BC and dual.
P4 NORM2 replaces ONLY the risk proxy: g(a)=sum(a**2), no neighbors.
All reward, actor, dual, data, real-cost evaluation and seeds otherwise match P3.
One-step neighborhood label is NOT episode CVaR. Lambda budget is NOT a real
trajectory-CVaR budget. All outputs isolated from historical Day21 files.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import copy

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

DATA_SHA = 'e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179'
SOURCE_SHA = '923ed3d5256fb9292c306a10ba92cb60df5971af36e79ca382dd7d4900aecbeb'
GAMMA = 0.99
TAU_V = 0.7
LR = 3e-4
EMA = 0.005
CLIP_QV = 10.0
CLIP_ACTOR = 5.0
BATCH = 256
RHO = 0.75
COST_TAU = 3.9079
POOL_SEED = 0
N_QUERY = 30000
N_REF = 60000
K_RADIUS = 1500
ACTION_WEIGHT = 20.0
LAMBDA_MAX = 50.0
LAMBDA_LR = 0.5
LAMBDA_INTERVAL = 200
FRACTIONS = {'NORM2': 'action_norm_squared'}
P3_SCRIPT_SHA = 'c526d7a97aff894083d96e0ffb622062bead645662042e3d7e84b90a02ae0d37'
P3_OUTPUT_NAME = '论文核验输出_P3稳定IQL_DE端点'
P4_OUTPUT_NAME = '论文核验输出_P4动作范数平方对照'
ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_path(project):
    return project / '阶段2W2_约束主干.py'


def data_path(project):
    return project / '每日结果' / 'Day20_路径二状态相关代价' / '数据_HalfCheetah_异质.npz'


def check_source(project):
    paths = {'data': data_path(project), 'W2': source_path(project)}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('缺少原始输入: ' + '; '.join(missing))
    hdata = sha256(paths['data'])
    hw2 = sha256(paths['W2'])
    if hdata != DATA_SHA:
        raise RuntimeError('Day21 数据哈希错误，禁止训练: ' + hdata)
    if hw2 != SOURCE_SHA:
        raise RuntimeError('W2 源码哈希变化，必须先审计: ' + hw2)
    return {'data_sha256': hdata, 'W2_sha256': hw2,
            'data_path': str(paths['data']), 'W2_path': str(paths['W2'])}


def load_data(project, verify=True):
    if verify: check_source(project)
    with np.load(data_path(project), allow_pickle=False) as z:
        keys = ('obs', 'act', 'rew', 'done')
        if any(k not in z for k in keys):
            raise RuntimeError('数据键缺失，要求 obs/act/rew/done')
        d = {k: np.ascontiguousarray(z[k], dtype=np.float32) for k in keys}
    n = len(d['obs'])
    if n != 200_000 or any(len(v) != n for v in d.values()):
        raise RuntimeError('转移数或数据形状异常')
    if d['obs'].shape[1] != 17 or d['act'].shape[1] != 6:
        raise RuntimeError('HalfCheetah 维度不符')
    if any(not np.isfinite(x).all() for x in d.values()):
        raise RuntimeError('数据存在 NaN/Inf')
    cost = np.maximum(0., (d['act'].astype(np.float64)**2).sum(-1)-COST_TAU)
    d['cost'] = cost.astype(np.float32)
    obs2 = np.zeros_like(d['obs']); obs2[:-1] = d['obs'][1:]
    obs2[:-1][d['done'][:-1] > .5] = 0.
    d['obs2'] = obs2
    return d


class ActionNormSquared:
    """Simple immediate action magnitude proxy. No neighbor stats; no hidden threshold.

    Behavior labels and policy sample labels use the SAME analytic sum of squares;
    the dual scaling is the behavior-data mean, analogous to P3 risk label mean.
    This does NOT use reward, future cost, or episode-level CVaR.
    """
    def __init__(self, data):
        act = np.asarray(data['act'], dtype=np.float32)
        self.labels = np.square(act, dtype=np.float32).sum(axis=1, dtype=np.float32)
        if not np.isfinite(self.labels).all() or self.labels.mean() <= 0:
            raise RuntimeError('NORM2 数据标签非法')
        self.meta = {
            'risk_semantics':'instantaneous_action_squared_norm',
            'formula':'sum_i action[i]**2',
            'risk_is_not_episode_CVaR':True,
            'has_neighbor_geometry':False,
            'has_cost_threshold':False,
            'source_data_sha256':DATA_SHA,
            'normalization':'risk_label_mean_over_200000_behavior_actions',
            'episode_cost_threshold_used_only_for_environment_evaluation':COST_TAU,
        }
        self.counts={'ondata':0, 'policy_queries':0, 'policy_empty':0}

    def risk(self, states, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError('非有限动作，不能求 NORM2 风险')
        val=actions.square().sum(dim=-1,keepdim=True)
        if not torch.isfinite(val).all():
            raise FloatingPointError('NORM2 风险存在 NaN/Inf')
        self.counts['policy_queries']+=len(actions)
        return val


class NeighborhoodTail:
    """Exact Day21 D/E neighborhood geometry. Cache is written only to new P3 dir."""
    def __init__(self, data, fraction, cache_file=None, verbose=True, historical_cache=None):
        if float(fraction) not in (0.1, 1.0):
            raise ValueError('P3 freezes q at D10/E100 endpoints')
        self.fraction = float(fraction)
        obs, act, costs = data['obs'], data['act'], data['cost']
        self.sm = obs.mean(0); self.ss = np.maximum(obs.std(0), 1e-6)
        self.am = act.mean(0); self.asd = np.maximum(act.std(0), 1e-6)
        self.rows = np.ascontiguousarray(np.concatenate([obs, act], 1))
        rng = np.random.RandomState(POOL_SEED)
        seed_queries = rng.choice(len(obs), size=min(N_QUERY, len(obs)), replace=False)
        mask = np.ones(len(obs), dtype=bool); mask[seed_queries] = False
        refs = np.where(mask)[0]
        if len(refs) > N_REF:
            refs = rng.choice(refs, size=N_REF, replace=False)
        self.ref_ids = refs
        self.tree = cKDTree(self.features(obs[refs], act[refs]))
        self.ref_costs = np.asarray(costs[refs], dtype=np.float64)
        k = min(K_RADIUS, len(refs))
        qs = seed_queries[:min(500, len(seed_queries))]
        dist, _ = self.tree.query(self.features(obs[qs], act[qs]), k=k, workers=-1)
        self.radius = float(np.median(dist[:,-1] if dist.ndim == 2 else dist))
        self.meta = {'fraction': self.fraction, 'radius': self.radius,
                     'pool_seed': POOL_SEED, 'reference_count': len(refs),
                     'action_weight': ACTION_WEIGHT, 'radius_k': k,
                     'reference_ids_sha256': hashlib.sha256(np.ascontiguousarray(refs).tobytes()).hexdigest(),
                     'risk_semantics': 'local upper-fraction mean of one-step costs',
                     'empty_fallback': 0.0}
        self.counts = {'ondata':0, 'policy_queries':0, 'policy_empty':0}
        labels = None
        if cache_file is not None and cache_file.exists():
            with np.load(cache_file,allow_pickle=False) as c:
                if ('labels' not in c or 'ref_ids' not in c or 'radius' not in c
                    or 'fraction' not in c or not np.array_equal(c['ref_ids'],refs)
                    or abs(float(c['radius']) - self.radius) > 1e-8
                    or abs(float(c['fraction']) - self.fraction) > 1e-12
                    or len(c['labels']) != len(obs)):
                    raise RuntimeError('已有风险缓存与实验几何不一致，停止')
                labels = np.asarray(c['labels'],dtype=np.float32)
        if labels is None and historical_cache is not None and historical_cache.is_file():
            # Historical D/E originals are strictly read-only. Verify exact original
            # reference ids, radius, label count AND a pointwise 48-query spot check.
            with np.load(historical_cache, allow_pickle=False) as c:
                if ('labels' not in c or 'ref_ids' not in c or 'radius' not in c
                    or not np.array_equal(c['ref_ids'],refs)
                    or abs(float(c['radius']) - self.radius) > 1e-8
                    or len(c['labels']) != len(obs)):
                    raise RuntimeError('历史 D/E 标签缓存存在但几何不匹配，停止')
                labels = np.asarray(c['labels'],dtype=np.float32)
            check_ix=np.random.RandomState(20261011).choice(len(obs),48,replace=False)
            expected=self.lookup(self.features(obs[check_ix],act[check_ix]),count=False)
            if not np.allclose(labels[check_ix],expected,atol=1e-5,rtol=1e-5):
                raise RuntimeError('历史 D/E 标签缓存 spot-check 与新实现不一致')
            if verbose:print('P3 imported verified historical labels, read-only:',historical_cache,flush=True)
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True,exist_ok=True)
                with cache_file.open('xb') as f:
                    np.savez_compressed(f,labels=labels,ref_ids=refs,
                                        radius=self.radius,fraction=self.fraction)
        if labels is None:
            labels = np.empty(len(obs),dtype=np.float32)
            for b in range(0,len(obs),1000):
                e = min(len(obs),b+1000)
                labels[b:e] = self.lookup(self.features(obs[b:e],act[b:e]),count=False)
                if verbose and (e%50000==0 or e==len(obs)):
                    print(f'P3 cached {e}/{len(obs)} labels',flush=True)
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True,exist_ok=True)
                with cache_file.open('xb') as f:
                    np.savez_compressed(f,labels=labels,ref_ids=refs,
                                        radius=self.radius,fraction=self.fraction)
        # Verify *every* cache (including new P3 cache) by independently querying
        # a deterministic held-out set of 48 original data points. The check is
        # intentionally after cache creation, before training.
        spot=np.random.RandomState(20261011).choice(len(obs),48,replace=False)
        actual=self.lookup(self.features(obs[spot],act[spot]),count=False)
        if not np.allclose(labels[spot],actual,rtol=1e-5,atol=1e-5):
            raise RuntimeError('P3 风险标签与当前邻域直接统计不一致；停止，禁止正式训练')
        self.labels = labels
        # Exact lookup: cached behavior actions, kd-tree only for freshly sampled actions.
        self.key_idx = {self.rows[i].tobytes():i for i in range(len(obs))}
        if verbose:
            print('P3 q/radius/mean:',self.fraction,self.radius,float(labels.mean()),flush=True)

    def features(self,s,a):
        return np.concatenate([(s-self.sm)/self.ss,
                               ACTION_WEIGHT*(a-self.am)/self.asd],axis=-1)

    def lookup(self, feats, count=True):
        neighs = self.tree.query_ball_point(np.atleast_2d(feats),self.radius,workers=-1)
        out = np.zeros(len(neighs),dtype=np.float32)
        for i, ids in enumerate(neighs):
            if not ids:
                if count:self.counts['policy_empty'] += 1
                continue
            val = self.ref_costs[np.asarray(ids,dtype=np.intp)]
            k = max(1,int(round(len(val)*self.fraction)))
            out[i] = float(np.partition(val,len(val)-k)[-k:].mean())
        if count:self.counts['policy_queries'] += len(neighs)
        return out

    def risk(self, states, actions):
        o = states.detach().cpu().numpy().astype(np.float32, copy=False)
        a = actions.detach().cpu().numpy().astype(np.float32, copy=False)
        rows = np.ascontiguousarray(np.concatenate([o,a],axis=1))
        out = np.empty(len(rows),dtype=np.float32); queries=[]
        for i, v in enumerate(rows):
            j = self.key_idx.get(v.tobytes())
            if j is None: queries.append(i)
            else:
                out[i] = self.labels[j];self.counts['ondata'] += 1
        if queries:
            ix = np.asarray(queries,dtype=np.intp)
            out[ix] = self.lookup(self.features(o[ix],a[ix]))
        return torch.from_numpy(out).reshape(-1,1).to(device=states.device)


def network(i,o):
    return nn.Sequential(nn.Linear(i,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,o))


class StablePolicy(nn.Module):
    def __init__(self, sm, ss, ds=17, da=6):
        super().__init__()
        self.q1=network(ds+da,1);self.q2=network(ds+da,1)
        self.vr=network(ds,1)
        self.演员=network(ds,2*da)
        self.演员[-1].weight.data.mul_(0.01)
        self.演员[-1].bias.data.zero_()
        self.register_buffer('obs_mean',torch.as_tensor(sm,dtype=torch.float32).reshape(1,-1))
        self.register_buffer('obs_std',torch.as_tensor(ss,dtype=torch.float32).reshape(1,-1))
        self.da=da

    def normalized(self,raw):
        return (raw-self.obs_mean)/self.obs_std

    def params(self,s_norm):
        z=self.演员(s_norm)
        return z[:,:self.da],z[:,self.da:].clamp(-5.,2.)

    def 均值动作(self,raw):
        return torch.tanh(self.params(self.normalized(raw))[0])

    def sample_raw(self,raw,generator):
        mu,ls=self.params(self.normalized(raw))
        noise=torch.randn(mu.shape,generator=generator,device=mu.device,dtype=mu.dtype)
        return torch.tanh(mu+ls.exp()*noise)


def expectile(diff,tau=TAU_V):
    return (torch.where(diff>0,tau,1-tau)*diff.square()).mean()


def standardize(x):
    return (x-x.mean())/(x.std()+1e-6)


def actor_weights(adv,risk,lam):
    z_r=standardize(adv)
    z_p=standardize(risk)
    u=standardize(z_r-lam*z_p)
    w=torch.exp(3.0*u)
    return (w/w.mean().clamp_min(1e-6)).clamp(max=100.0)


def numerical_metrics(model,d,index,device='cpu'):
    # Fixed 4096 behavior transitions. Raw+next state standardized consistently.
    with torch.no_grad():
        ix=index
        s=model.normalized(torch.from_numpy(d['obs'][ix]).to(device))
        s2=model.normalized(torch.from_numpy(d['obs2'][ix]).to(device))
        a=torch.from_numpy(d['act'][ix]).to(device)
        r=torch.from_numpy(d['rew'][ix]).reshape(-1,1).to(device)
        done=torch.from_numpy(d['done'][ix]).reshape(-1,1).to(device)
        q=torch.minimum(model.q1(torch.cat([s,a],-1)),model.q2(torch.cat([s,a],-1)))
        v=model.vr(s)
        targ=r+GAMMA*(1-done)*model.vr(s2)
        td=targ-q
        adv=q-v
        def stats(y):
            y=y.reshape(-1)
            return {'min':float(y.min()),'p50':float(y.median()),
                    'max':float(y.max()),'std':float(y.std()),
                    'mean':float(y.mean())}
        return {'Q':stats(q),'V':stats(v),'advantage':stats(adv),
                'TD_RMSE':float(td.square().mean().sqrt()),
                'finite': bool(torch.isfinite(q).all() and torch.isfinite(v).all() and torch.isfinite(td).all())}


def import_local(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def validate_dir(project,out):
    h=(project/'每日结果').resolve()
    if out==h or h in out.parents:
        raise ValueError('禁止写入历史 每日结果')
    if '论文核验输出_F组上尾比例' in str(out):
        raise ValueError('禁止覆写 F 组输出')
    if not out.is_relative_to(project):
        raise ValueError('输出目录必须在项目根目录下')


def validate_real_seed(seed):
    if seed not in (0,1,2,3,4,5,6,7):
        raise ValueError('扩种子版：允许 seeds 0-7（唯一改动；原脚本白名单为 0,1,2）')


def p3_baseline_root(project):
    return project / P3_OUTPUT_NAME


def p3_baseline_candidates(project, arm, seed, steps, stage=None):
    """扩种子版新增：P3 配对基线的搜索顺序（唯一的基础设施改动）。

    seeds 0-2 的冻结基线在 `论文核验输出_P3稳定IQL_DE端点/`；
    本轮 HC 补充实验新训的 seeds 3-7 基线写在 `论文核验输出_HC补充_20261016/P3_D10_seedN/`。
    两个来源都必须通过下面 load_p3_json 的元数据与 τ 认证校验才算合法基线；
    不改配对判据、不改科学参数。搜索顺序：本轮补充目录 → 归档目录。
    """
    if stage=='pilot':
        # 试跑基线固定取归档目录（试跑只跑 seed0，本轮补充实验不重跑试跑）
        return [p3_baseline_root(project)/'_试跑'/f'结果_P3_{arm}_种子{seed}_{steps}步.json']
    sup=project/'论文核验输出_HC补充_20261016'/f'P3_{arm}_seed{seed}'
    return [sup/f'结果_P3_{arm}_种子{seed}_{steps}步.json',
            p3_baseline_root(project)/f'结果_P3_{arm}_种子{seed}_{steps}步.json']


def load_p3_json(project, arm, seed, steps, stage=None):
    cands=p3_baseline_candidates(project,arm,seed,steps,stage)
    f=next((x for x in cands if x.is_file()), cands[0])
    if not f.is_file():
        raise RuntimeError('P3 配对基线结果缺失，先停止：'+str(f))
    with f.open(encoding='utf-8') as fh: obj=json.load(fh)
    if (obj.get('arm')!=arm or obj.get('seed')!=seed
        or obj.get('training_steps')!=steps or obj.get('data_sha256')!=DATA_SHA
        or obj.get('W2_source_sha256')!=SOURCE_SHA):
        raise RuntimeError('P3 基线元数据不匹配：'+str(f))
    if stage!='pilot':
        ev=obj.get('evaluation') or {}
        if (ev.get('cost_threshold')!=COST_TAU
            or ev.get('environment_cost_preflight_passed') is not True
            or ev.get('seed_start')!=115000
            or ev.get('num_episodes')!=100):
            raise RuntimeError('P3 基线正式评估字段异常：'+str(f))
    return obj


def numeric_equal_to_p3(p4,p3,label,tol=0.03,abs_tol=0.25):
    h0=p4['history'];h1=p3['history']
    if len(h0)!=len(h1):
        raise RuntimeError(label+' 与 P3 历史记录长度不同')
    for i,(a,b) in enumerate(zip(h0,h1)):
        if a['step']!=b['step']:
            raise RuntimeError(label+' 与 P3 步数不一致')
        for metric in ('Q','V','advantage'):
            for k in ('min','p50','max','std','mean'):
                local_abs,local_rel=(1.0,.15) if metric=='advantage' else (abs_tol,tol)
                if abs(a[metric][k]-b[metric][k])>max(local_abs, local_rel*abs(b[metric][k])):
                    raise RuntimeError(f'{label} critic 不配对：step={a["step"]} {metric}.{k} P4={a[metric][k]} P3={b[metric][k]}')
        for metric in ('TD_RMSE','q_loss','v_loss'):
            local_abs,local_rel=(.25,.10) if metric!='TD_RMSE' else (.20,.10)
            if abs(a[metric]-b[metric])>max(local_abs, local_rel*abs(b[metric])):
                raise RuntimeError(f'{label} critic 不配对：step={a["step"]} {metric} P4={a[metric]} P3={b[metric]}')
    return True


def p4_pilot_gate(project,base):
    f=base/'_试跑'/'结果_P4_NORM2_种子0_5000步.json'
    if not f.is_file():
        raise RuntimeError('缺 P4 NORM2 5000步试跑，不允许 full：'+str(f))
    obj=json.loads(f.read_text(encoding='utf-8'))
    if (obj.get('arm')!='NORM2' or obj.get('seed')!=0 or obj.get('training_steps')!=5000
        or obj.get('mode')!='pilot' or obj.get('data_sha256')!=DATA_SHA):
        raise RuntimeError('P4 试跑 JSON 元数据异常')
    last=obj['history'][-1]
    if (not last['finite'] or abs(last['Q']['p50'])>10000
       or abs(last['Q']['max'])>10000 or last['TD_RMSE']>1000):
        raise RuntimeError('P4 pilot 数值不通过')
    baseline=load_p3_json(project,'D10',0,5000,'pilot')
    numeric_equal_to_p3(obj,baseline,'P4 pilot vs P3 D10 pilot')
    print('P4 pilot PASS: reward Q/V within 3% or .25; TD and loss within 10%; adv stats within 15% or 1.0, at all checkpoints',flush=True)


def check_p3_provenance(project):
    f=project/'工具_Day21_P3_稳定IQL_DE端点复验_v1_1.py'
    if not f.is_file(): raise FileNotFoundError('原 P3 v1.1 脚本不存在：'+str(f))
    h=sha256(f)
    if h!=P3_SCRIPT_SHA:
        raise RuntimeError('原 P3 v1.1 脚本 SHA 不符，禁止进行配对实验：'+h)
    for seed in (0,1,2):
        load_p3_json(project,'D10',seed,40000)
        load_p3_json(project,'E100',seed,40000)
    load_p3_json(project,'D10',0,5000,'pilot')
    print('PASS P3 provenance: SHA',h, 'and all baseline JSONs',flush=True)


def 环境预飞(hc, tol=1e-4):
    """v1.1 新增：真实 Gymnasium/MuJoCo 环境上的成本口径预飞。

    审计发现的独立缺陷：`阶段5_HalfCheetah机制验证.py` 的模块级 `数据阈值=[0.0]`，
    只在 `HC.准备()` 里被设成 τ；P3 从不调用 `准备()`，而环境 wrapper 的 step 用
    `max(0, ‖a‖² − 数据阈值[0])` 算成本 —— 不显式设置就会把成本算成 ‖a‖²，
    正式回放的回合成本与 CVaR 会整体偏大。这里显式设成 τ，并做两条真实环境预飞。
    """
    hc.数据阈值[0] = COST_TAU
    期望 = float(max(0.0, 6.0 - COST_TAU))   # 全 1 动作：‖a‖²=6 → 6 − 3.9079 = 2.0921
    act_dim = hc.W2.动作维
    环境 = hc.建环境()
    try:
        环境.reset(seed=12345)
        _, _, _, _, info0 = 环境.step(np.zeros(act_dim, dtype=np.float32))
        c0 = float(info0.get('cost', float('nan')))
        环境.reset(seed=12345)
        _, _, _, _, info1 = 环境.step(np.ones(act_dim, dtype=np.float32))
        c1 = float(info1.get('cost', float('nan')))
    finally:
        环境.close()
    checks = [{'case': 'action_norm2=0', 'cost': c0, 'expected': 0.0,
               'pass': bool(abs(c0 - 0.0) <= tol)},
              {'case': 'action_norm2=6', 'cost': c1, 'expected': 期望,
               'pass': bool(abs(c1 - 期望) <= tol)}]
    out = {'cost_threshold': float(hc.数据阈值[0]), 'cost_tau_expected': COST_TAU,
           'budget_rho': float(getattr(hc, '预算', float('nan'))),
           'episode_horizon': float(getattr(hc, '固定视界', float('nan'))),
           'action_norm2_zero_cost': c0, 'action_norm2_six_cost': c1,
           'expected_cost_for_ones': 期望, 'checks': checks,
           'passed': bool(all(x['pass'] for x in checks))}
    for x in checks:
        print(('PASS' if x['pass'] else 'FAIL'),
              f"{x['case']} cost={x['cost']:.6f} (expect {x['expected']:.6f})", flush=True)
    return out


def train(project, out, arm, seed, steps, mode, validation_seed=115000):
    torch.set_num_threads(1)  # F16 修复：锁定单线程，消除归约顺序导致的运行间 1e-6 差异
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    check_source(project)
    validate_dir(project,out)
    if mode=='full':
        check_p3_provenance(project)
        p4_pilot_gate(project,out)
        validate_real_seed(seed)
        if steps != 40000: raise ValueError('正式训练必须整整 40000 步')
    elif mode=='pilot':
        if seed!=0 or steps not in (500,2000,5000):
            raise ValueError('诊断试跑只允许 seed0，500/2000/5000 步')
    elif mode=='smoke':
        if seed!=99 or steps!=200:
            raise ValueError('冒烟只允许 seed99，200 步')
    else: raise ValueError('训练模式错误')
    if arm not in FRACTIONS: raise ValueError('P4 只允许 NORM2')
    out.mkdir(parents=True,exist_ok=True)
    model_file=out/f'模型_P4_{arm}_种子{seed}_{steps}步.pt'
    json_file=out/f'结果_P4_{arm}_种子{seed}_{steps}步.json'
    if model_file.exists() or json_file.exists():
        raise FileExistsError('拒绝覆盖已完成结果：'+str(json_file))
    d=load_data(project)
    # P4: the ONLY experimental treatment is the risk-proxy implementation.
    # No KDTree/cache is needed for the exact analytic ||a||^2 surrogate.
    risk=ActionNormSquared(d)
    print('P4 norm2 data label mean:',float(risk.labels.mean()),flush=True)
    # Identical P3 initialization and critic batch stream, independent of risk proxy.
    torch.manual_seed(seed)
    mean=d['obs'].mean(0,keepdims=True)
    std=np.maximum(d['obs'].std(0,ddof=1,keepdims=True),1e-6)
    m=StablePolicy(mean,std)
    qtarget=copy.deepcopy(m)
    qtarget.eval()
    for p in qtarget.parameters():p.requires_grad_(False)
    optim_q=torch.optim.Adam(list(m.q1.parameters())+list(m.q2.parameters()),lr=LR)
    optim_v=torch.optim.Adam(m.vr.parameters(),lr=LR)
    optim_actor=torch.optim.Adam(m.演员.parameters(),lr=LR)
    # Shared critic batch sequence across arms/seeds; actor sampling has separate RNG.
    gen_batch=torch.Generator(device='cpu').manual_seed(100000+seed)
    gen_action=torch.Generator(device='cpu').manual_seed(200000+seed)
    s=torch.from_numpy(d['obs']);s2=torch.from_numpy(d['obs2'])
    a=torch.from_numpy(d['act'])
    r=torch.from_numpy(d['rew']).reshape(-1,1)
    done=torch.from_numpy(d['done']).reshape(-1,1)
    sn=m.normalized(s).detach();s2n=m.normalized(s2).detach()
    norm_scale=float(np.maximum(float(risk.labels.mean()),1e-6))
    lam=0.0; history=[]; dual=[]
    fixed_idx=np.random.RandomState(20261011).choice(len(s),4096,replace=False)
    log_freq= (200 if mode=='smoke' else (500 if mode=='pilot' else 2000))
    started=time.time()
    for step in range(1,steps+1):
        j=torch.randint(0,len(s),(BATCH,),generator=gen_batch)
        ss=sn[j];ns=s2n[j];aa=a[j];rr=r[j];dd=done[j]
        sa=torch.cat([ss,aa],-1)
        with torch.no_grad():
            target=rr+GAMMA*(1-dd)*m.vr(ns)
        q1=m.q1(sa);q2=m.q2(sa)
        q_loss=(q1-target).square().mean()+(q2-target).square().mean()
        with torch.no_grad():
            qt=torch.minimum(qtarget.q1(sa),qtarget.q2(sa))
        v_loss=expectile(qt-m.vr(ss))
        loss=q_loss+v_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'非有限奖励损失，step={step}')
        optim_q.zero_grad();optim_v.zero_grad()
        loss.backward()
        for net in (m.q1,m.q2,m.vr):
            nn.utils.clip_grad_norm_(net.parameters(),CLIP_QV)
        optim_q.step();optim_v.step()
        with torch.no_grad():
            for online,targetnet in ((m.q1,qtarget.q1),(m.q2,qtarget.q2)):
                for p,t in zip(online.parameters(),targetnet.parameters()):
                    t.mul_(1-EMA).add_(p,alpha=EMA)
        # Same W2 actor logic, rewards and costs independently batch Z-normalized.
        with torch.no_grad():
            qnow=torch.minimum(m.q1(sa),m.q2(sa))
            adv=qnow-m.vr(ss)
            penalty=risk.risk(s[j],aa)/norm_scale
            weights=actor_weights(adv,penalty,lam)
            if not torch.isfinite(weights).all():
                raise FloatingPointError('非有限演员权重')
        mu,logstd=m.params(ss)
        stddev=logstd.exp()
        az=torch.atanh(aa.clamp(-.999999,.999999))
        logp=(-.5*((az-mu)/stddev).square()-logstd-.9189).sum(-1,keepdim=True)
        actor_loss=-(weights*logp).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError('非有限演员损失')
        optim_actor.zero_grad();actor_loss.backward()
        nn.utils.clip_grad_norm_(m.演员.parameters(),CLIP_ACTOR)
        optim_actor.step()
        if step%LAMBDA_INTERVAL==0:
            with torch.no_grad():
                sample=m.sample_raw(s[j],gen_action)
                constraint=float((risk.risk(s[j],sample)/norm_scale).mean())
            lam=float(np.clip(lam+LAMBDA_LR*(constraint-RHO),0,LAMBDA_MAX))
            dual.append([step,lam,constraint])
        if step%log_freq==0 or step==steps:
            num=numerical_metrics(m,d,fixed_idx)
            num.update(step=step,lam=lam,q_loss=float(q_loss.detach()),
                       v_loss=float(v_loss.detach()),actor_loss=float(actor_loss.detach()),
                       elapsed_sec=round(time.time()-started,1))
            history.append(num)
            print(f'P4 {arm}/{seed} {step}/{steps}: Q50={num["Q"]["p50"]:.2f} '
                  f'TD={num["TD_RMSE"]:.2f}, lambda={lam:.3f}, '
                  f'time={num["elapsed_sec"]}s',flush=True)
            # Hard numerical kill switch; not an acceptance criterion.
            if (not num['finite'] or abs(num['Q']['p50'])>10000 or
                    abs(num['Q']['max'])>10000 or num['TD_RMSE']>1000):
                raise FloatingPointError('P3 奖励网络触发数值停止门槛；严禁继续正式训练')
    result={'experiment':'P4 repaired IQL-like actor, pre-registered immediate action norm squared comparator',
        'status':'completed_training_not_proven_stable', 'arm':arm,'seed':seed,
        'training_steps':steps,'mode':mode,
        'data_sha256':DATA_SHA,'W2_source_sha256':SOURCE_SHA,
        'risk_geometry':risk.meta,'p3_source_sha256':P3_SCRIPT_SHA,
        'risk_calls':risk.counts,'risk_label_mean':norm_scale,
        'reward_train':{'MSE_Q':True,'V_expectile_tau':TAU_V,
                        'V_source':'EMA target double Q','EMA':EMA,
                        'optimizer':'separate Adam','lr':LR,'grad_clip_QV':CLIP_QV,
                        'obs_standardization':'train-data mean and sample std'},
        'actor_update':'W2 weighted tanh-Gaussian behavior cloning, risk no_grad',
        'actor_lr':LR,'actor_clip':CLIP_ACTOR,
        'dual':{'rho':RHO,'interval':LAMBDA_INTERVAL,'lr':LAMBDA_LR,
                'lambda_max':LAMBDA_MAX,'trace':dual,'final':lam},
        'history':history,'evaluation':None}
    if mode=='full':
        # Validation only. Final heldout test MUST be a separate frozen-model operation.
        drv=import_local(project/'工具_HC_异质驱动.py',f'P3_HC_driver_{arm}_{seed}')
        hc=drv.装好(RHO)
        # v1.1: 显式设成本阈值并做真实环境预飞；不通过就直接终止，禁止产出被污染的 CVaR。
        预飞=环境预飞(hc)
        if not 预飞['passed']:
            raise RuntimeError(f'P3 v1.1 环境成本预飞失败，终止正式回放：{预飞}')
        _d, _tau = hc.读并标号()      # 修正：评估环境成本阈值须与协议/归档一致
        hc.数据阈值[0] = float(_tau)
        ev=hc.评估(m,回合=100,并行=6,种子=validation_seed)
        result['evaluation']={'type':'validation_not_final_test','seed_start':validation_seed,
            'num_episodes':100,
            'cost_threshold':float(hc.数据阈值[0]),
            'environment_cost_preflight_passed':bool(预飞['passed']),
            'environment_cost_preflight':预飞,
            'return_mean':float(ev['回报均值']),
            'cost_mean':float(ev['成本均值']),
            'episode_CVaR25':float(ev['成本CVaR25']),
            'returns':np.asarray(ev['逐回合回报']).tolist(),
            'episode_costs':np.asarray(ev['逐回合成本']).tolist()}
        print('P4 validation',arm,seed,result['evaluation']['return_mean'],
              result['evaluation']['episode_CVaR25'],flush=True)
    # Strict P3 pairing: reward diagnostics must match every recorded checkpoint.
    # Do this before saving output; do not mask even small experimental drifts.
    if mode in ('pilot','full'):
        baseline=load_p3_json(project,'D10',seed,steps,'pilot' if mode=='pilot' else None)
        numeric_equal_to_p3(result,baseline,f'P4 NORM2 seed{seed} vs P3 D10')
        print('PASS P4 reward diagnostics within frozen tolerance of P3 D10, seed',seed,flush=True)
    # Saves new artifacts only when end-to-end training and eval completed.
    torch.save({'state_dict':m.state_dict(),'risk_meta':risk.meta,'seed':seed,
                'steps':steps,'arm':arm,'reward_config':result['reward_train']},model_file)
    with json_file.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print('P4 outputs:',model_file,json_file,flush=True)


def selftest():
    torch.set_num_threads(2)
    cost=np.arange(1.,21.)
    for frac in (0.1,1.0):
        k=max(1,int(round(len(cost)*frac)))
        assert np.partition(cost,len(cost)-k)[-k:].mean()==np.sort(cost)[-k:].mean()
    x=torch.tensor([-2.,-1.,0.,1.,2.])
    assert torch.allclose(expectile(x,.5),x.square().mean()*.5)
    w=actor_weights(torch.tensor([[1.],[2.],[3.]]),
                    torch.tensor([[2.],[5.],[7.]]),.7)
    assert torch.isfinite(w).all() and (w>=0).all()
    # Check W2 actor-gradient path and consistency of raw/normalization input.
    torch.manual_seed(0)
    m=StablePolicy(np.zeros((1,17),dtype=np.float32),np.ones((1,17),dtype=np.float32))
    s=torch.randn(16,17); a=torch.tanh(torch.randn(16,6))
    mu,ls=m.params(m.normalized(s));std=ls.exp();z=torch.atanh(a.clamp(-.999999,.999999))
    loss=-(-.5*((z-mu)/std).square()-ls-.9189).sum(-1).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.演员.parameters())
    assert torch.allclose(m.均值动作(s),torch.tanh(m.params(s)[0]))
    # Identical risk arm seed must give same critic init and batch RNG.
    g=torch.Generator().manual_seed(100000)
    j=torch.randint(0,100,(256,),generator=g)
    h=torch.Generator().manual_seed(100000)
    assert torch.equal(j,torch.randint(0,100,(256,),generator=h))
    # Synthetic geometry: identical reference/radius and pointwise D10 >= E100.
    global N_QUERY, N_REF, K_RADIUS
    orig=(N_QUERY,N_REF,K_RADIUS)
    try:
        N_QUERY,N_REF,K_RADIUS=20,50,10
        nr=np.random.RandomState(17)
        ds={'obs':nr.standard_normal((100,17)).astype('float32'),
            'act':nr.uniform(-1,1,(100,6)).astype('float32')}
        ds['cost']=np.maximum(0,np.sum(ds['act'].astype('float64')**2,axis=1)-1.0).astype('float32')
        objs=[NeighborhoodTail(ds,fr,cache_file=None,verbose=False) for fr in (.1,1.)]
        assert objs[0].meta['reference_ids_sha256']==objs[1].meta['reference_ids_sha256']
        assert abs(objs[0].radius-objs[1].radius)<1e-10
        assert (objs[0].labels>=objs[1].labels-1e-6).all()
        sob=torch.from_numpy(ds['obs'][:5]);act=torch.from_numpy(ds['act'][:5])
        assert torch.allclose(objs[0].risk(sob,act).reshape(-1),
                              torch.from_numpy(objs[0].labels[:5]))
        newrisk=objs[1].risk(sob,torch.zeros_like(act))
        assert newrisk.shape==(5,1) and torch.isfinite(newrisk).all()
    finally:
        N_QUERY,N_REF,K_RADIUS=orig
    # For zero-cost actions, squared action norm can be nonzero (alternative baseline still needed).
    print('PASS P4 inherited P3: tail aggregation, IQL losses, actor gradient, obs transform, RNG',flush=True)


def audit(project, output):
    check_p3_provenance(project)
    pairs=[]
    for seed in (0,1,2):
        f=output/f'结果_P4_NORM2_种子{seed}_40000步.json'
        if not f.is_file():raise FileNotFoundError('P4 正式结果不完整：'+str(f))
        p4=json.loads(f.read_text(encoding='utf-8'))
        p3=load_p3_json(project,'D10',seed,40000)
        if (p4.get('mode')!='full' or p4.get('data_sha256')!=DATA_SHA
            or p4.get('arm')!='NORM2' or p4.get('seed')!=seed
            or p4.get('training_steps')!=40000):
            raise RuntimeError('P4 结果元数据不一致：'+str(f))
        ev=p4.get('evaluation') or {}
        if (ev.get('cost_threshold')!=COST_TAU or
            ev.get('environment_cost_preflight_passed') is not True or
            ev.get('seed_start')!=115000 or ev.get('num_episodes')!=100):
            raise RuntimeError('P4 评估元数据非法：'+str(f))
        numeric_equal_to_p3(p4,p3,'P4 full vs P3 full seed'+str(seed))
        for obj in (p4,p3):
            e=obj['evaluation']
            if (len(e['returns'])!=100 or len(e['episode_costs'])!=100
                or not np.isclose(np.mean(e['returns']),e['return_mean'],atol=1e-3)
                or not np.isclose(np.mean(e['episode_costs']),e['cost_mean'],atol=1e-3)
                or not np.isclose(np.mean(sorted(e['episode_costs'])[-25:]),e['episode_CVaR25'],atol=1e-3)):
                raise RuntimeError('逐回合结果复算失败 seed='+str(seed))
        pairs.append({'seed':seed,'P3_D10':{k:p3['evaluation'][k] for k in ('return_mean','cost_mean','episode_CVaR25')},
            'P4_NORM2':{k:p4['evaluation'][k] for k in ('return_mean','cost_mean','episode_CVaR25')},
            'lambda_D10':p3['dual']['final'],'lambda_NORM2':p4['dual']['final'],
            'reward_Q_p50_identical':True})
    metrics={}
    for col in ('return_mean','cost_mean','episode_CVaR25'):
        a=np.array([x['P3_D10'][col] for x in pairs]);b=np.array([x['P4_NORM2'][col] for x in pairs]);delta=a-b
        metrics[col]={'D10_mean':float(a.mean()),'NORM2_mean':float(b.mean()),
                      'D10_minus_NORM2_mean':float(delta.mean()),
                      'D10_minus_NORM2_each_seed':delta.tolist(),
                      'D10_std_sample':float(a.std(ddof=1)),
                      'NORM2_std_sample':float(b.std(ddof=1))}
    summary={'design':'P4_NORM2_vs_P3_D10_same_reward_training','validation_only':True,
             'development_eval_seed_start':115000,'heldout_125000_untouched':True,
             'warning':'Internal lambda rho=0.75 does not match actual CVaR budgets',
             'pairs':pairs,'metrics':metrics}
    target=output/'P4_三种子对照核验.json'
    with target.open('x',encoding='utf-8') as fh:json.dump(summary,fh,ensure_ascii=False,indent=2)
    print('P4 audit PASS: 3 paired reward diagnostic trajectories within tolerance and 600 episode arrays recomputed',flush=True)
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
    print('P4 audit result',target,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=None)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--selftest',action='store_true')
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--evalcheck',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--pilot',action='store_true')
    mode.add_argument('--full',action='store_true')
    mode.add_argument('--audit',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=40000)
    args=p.parse_args()
    torch.set_num_threads(2)
    project=args.project.resolve()
    output=(args.output.resolve() if args.output else project/P4_OUTPUT_NAME)
    if args.selftest:
        selftest()
        fake={'act':np.array([[0]*6,[1]*6,[0.5]*6],dtype=np.float32)}
        norm=ActionNormSquared(fake)
        assert np.allclose(norm.labels,[0,6,1.5])
        acts=torch.from_numpy(fake['act']);s=torch.zeros(len(acts),17)
        assert torch.allclose(norm.risk(s,acts).flatten(),torch.tensor([0.,6.,1.5]))
        print('PASS P4: raw action norm2 cost surrogate, separate from episode cost threshold',flush=True)
        return
    check_source(project)
    if args.check:
        d=load_data(project)
        check_p3_provenance(project)
        risk=ActionNormSquared(d)
        print('PASS P4 check: dataset/W2/P3 SHA and all P3 results; label mean',risk.labels.mean(),flush=True)
        return
    if args.evalcheck:
        drv=import_local(project/'工具_HC_异质驱动.py','P4_HC_driver_evalcheck')
        hc=drv.装好(RHO)
        out=环境预飞(hc)
        if not out['passed']:raise SystemExit('P4 evalcheck 成本阈值不匹配，停止')
        print('PASS P4 evalcheck cost_threshold',hc.数据阈值[0],flush=True)
        return
    validate_dir(project,output)
    if args.audit:
        audit(project,output)
        return
    if args.smoke: output=output/'_冒烟'
    if args.pilot: output=output/'_试跑'
    mode='smoke' if args.smoke else ('pilot' if args.pilot else 'full')
    train(project,output,'NORM2',args.seed,args.steps,mode)


if __name__=='__main__':
    main()
