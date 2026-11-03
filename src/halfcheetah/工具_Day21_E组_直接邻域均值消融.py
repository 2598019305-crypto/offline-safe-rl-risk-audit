#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day21 / IQL: E group, direct local NEIGHBORHOOD MEAN one-step cost, without neural distillation.

Intended scientific comparison:
 A: learned supervised one-step cost critic
 B: analytic pointwise one-step cost
 C: NN-distilled local upper-tail (W4, cost_discount=0)
 D: direct reference-set upper-tail statistic, no NN.
 E: SAME reference set and radius as D; use all neighbors' mean cost instead of upper 10%.

E is NOT trajectory CVaR. E vs D tests tail selection conditional on fixed neighborhood geometry;
 different penalty normalization and dual feedback can still confound end-policy comparisons.

Original training source W2.训练一臂 is imported unchanged. This method is valid
 for the *IQL actor branch* because W2 computes risk signals only under no_grad.
 The direct statistic is nondifferentiable; do NOT use this class as a TD3+BC
 risk critic (there the actor directly differentiates through risk).

Dependencies: original Day21 project in --project, dsrl-env with torch/gymnasium.
 In the original project folder:
    python 工具_Day21_E组_直接邻域均值消融.py --selftest
    python 工具_Day21_E组_直接邻域均值消融.py --check
    python 工具_Day21_E组_直接邻域均值消融.py --cache-only
    python 工具_Day21_E组_直接邻域均值消融.py --seeds 99 --steps 200 --smoke
    python 工具_Day21_E组_直接邻域均值消融.py --seeds 0,1,2 --steps 40000

ALL output stays outside 每日结果; source datasets and model caches are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TAU = 3.9079
DATA_SHA = 'e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179'
POOL_SEED = 0
N_QUERY = 30000
N_REFERENCE = 60000
RADIUS_K = 1500
ACTION_WEIGHT = 20.0
# E uses all costs in the neighborhood; D uses the highest 10%.


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def import_local(path, name):
    s = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class DirectNeighborhoodMean:
    """Reference-statistic lookup, matching D's geometry for Day21.

    For data (s,a), return a precomputed label based on the same reference set.
    For freshly sampled policy actions, calculate the same neighborhood MEAN
    on demand. An empty neighborhood returns zero, matching W4's label builder;
    count such misses (OOD/generalization issue) and disclose them.
    """
    def __init__(self, data, *, cache_path=None, verbose=True):
        obs = np.asarray(data['obs'], dtype=np.float32)
        act = np.asarray(data['act'], dtype=np.float32)
        costs = np.asarray(data['cost'], dtype=np.float64)
        if len(obs) != len(act) or len(obs) != len(costs):
            raise ValueError('obs, act and cost lengths must match')
        self.n = len(obs)
        self.sm = obs.mean(0)
        self.ss = np.maximum(obs.std(0), 1e-6)
        self.am = act.mean(0)
        self.asd = np.maximum(act.std(0), 1e-6)
        self._counts = {'cache_hit': 0, 'query': 0, 'empty': 0}
        self._data_rows = np.ascontiguousarray(np.concatenate([obs,act], axis=1))
        self._data_dim = obs.shape[1]
        if not np.all(np.isfinite(self._data_rows)):
            raise ValueError('Non-finite data state or action')
        rng = np.random.RandomState(POOL_SEED)
        query_ids = rng.choice(self.n, size=min(N_QUERY,self.n), replace=False)
        mask = np.ones(self.n, dtype=bool); mask[query_ids] = False
        refs = np.where(mask)[0]
        if len(refs) > N_REFERENCE:
            refs = rng.choice(refs, size=N_REFERENCE, replace=False)
        self.ref_ids = refs
        feat_ref = self._features(obs[refs], act[refs])
        self.tree = cKDTree(feat_ref)
        self.ref_cost = costs[refs]
        k = min(RADIUS_K, len(refs))
        reference_queries = query_ids[:min(500,len(query_ids))]
        distances, _ = self.tree.query(self._features(obs[reference_queries],act[reference_queries]),k=k,workers=-1)
        self.radius = float(np.median(distances[:, -1] if distances.ndim == 2 else distances))
        self.meta = dict(kind='direct local all-neighbor MEAN one-step cost',
              radius=self.radius, ref_count=len(refs), n=self.n,
              radius_k=k, action_weight=ACTION_WEIGHT, statistic='arithmetic mean of ALL radius neighbors; no top-k selection',
              pool_seed=POOL_SEED, n_query_seed=min(N_QUERY,self.n),
              convention='empty neighborhood -> 0 (original W4 definition)',
              cost_discount=0.0)
        if verbose:
            print('[E] pool radius:', round(self.radius,6),
                  'references:',len(refs),'computing/loading data labels...',flush=True)
        labels = None
        if cache_path is not None and Path(cache_path).exists():
            z = np.load(str(cache_path),allow_pickle=False)
            if ('labels' in z and 'ref_ids' in z and 'radius' in z
                and int(z['labels'].shape[0])==self.n
                and np.array_equal(z['ref_ids'],self.ref_ids)
                and abs(float(z['radius'])-self.radius) < 1e-8):
                labels = np.asarray(z['labels'],dtype=np.float32)
                if verbose: print('[E] data-label cache loaded',flush=True)
            else:
                raise RuntimeError('E cache geometry mismatch; stop instead of silently reusing')
        if labels is None:
            labels = np.empty(self.n,dtype=np.float32)
            for start in range(0,self.n,1000):
                end = min(self.n,start+1000)
                labels[start:end] = self._lookup(self._features(obs[start:end],act[start:end]), count=False)
                if verbose and (end%25000==0 or end==self.n):
                    print(f'[E] labels {end}/{self.n}',flush=True)
            if cache_path is not None:
                target = Path(cache_path)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    raise FileExistsError(f'Refuse overwrite: {target}')
                np.savez_compressed(str(target),labels=labels,ref_ids=self.ref_ids,radius=self.radius)
        self.labels = labels
        # Exact float32 match: train batches are slices of the same data, so the
        # cached signal can replace 10m+ repeated KDTree queries without changing it.
        self.key_to_idx = {self._data_rows[i].tobytes():i for i in range(self.n)}
        if verbose:
            print('[E] label mean/std/range:',float(labels.mean()),float(labels.std()),
                  float(labels.min()),float(labels.max()),flush=True)

    def _features(self, obs, act):
        return np.concatenate([(obs-self.sm)/self.ss,
                               ACTION_WEIGHT*(act-self.am)/self.asd],axis=-1)

    def _lookup(self, feats, *, count=True):
        batches = np.atleast_2d(feats)
        neighbors = self.tree.query_ball_point(batches,self.radius,workers=-1)
        y = np.zeros(len(batches),dtype=np.float32)
        for i, ids in enumerate(neighbors):
            if len(ids)==0:
                if count: self._counts['empty'] += 1
                continue
            vals = self.ref_cost[np.asarray(ids,dtype=np.intp)]
            y[i] = float(vals.mean())
        if count: self._counts['query'] += len(batches)
        return y

    def 风险值(self, s, a):
        # IQL calls this under torch.no_grad() -- no gradients are required.
        import torch
        o=s.detach().cpu().numpy().astype(np.float32,copy=False)
        ac=a.detach().cpu().numpy().astype(np.float32,copy=False)
        joint=np.ascontiguousarray(np.concatenate([o,ac],axis=1))
        y=np.empty(len(joint),dtype=np.float32)
        new_positions=[]
        for i,row in enumerate(joint):
            index=self.key_to_idx.get(row.tobytes())
            if index is not None:
                y[i]=self.labels[index]
                self._counts['cache_hit']+=1
            else:
                new_positions.append(i)
        if new_positions:
            ii=np.asarray(new_positions,dtype=np.intp)
            y[ii]=self._lookup(self._features(o[ii],ac[ii]))
        return torch.as_tensor(y,dtype=torch.float32,device=s.device)

    def diagnostic(self):
        return dict(self._counts,**self.meta)


def get_day21(project):
    import_path=project/'工具_HC_异质驱动.py'
    data_path=project/'每日结果'/'Day20_路径二状态相关代价'/'数据_HalfCheetah_异质.npz'
    model_path=project/'每日结果'/'Day21_异质数据主表'
    required=[import_path, data_path, model_path/'模型_HalfCheetah_风险critic.pt',
              model_path/'模型_HalfCheetah_均值成本critic.pt',
              project/'阶段5_HalfCheetah机制验证.py',
              project/'阶段2W2_约束主干.py',
              project/'阶段2W4_动作决定风险任务.py',
              project/'公共_定批回放引擎.py']
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError('Missing original Day21 inputs: '+'; '.join(missing))
    sh=sha256(data_path)
    if sh!=DATA_SHA: raise RuntimeError(f'Day21 dataset version mismatch: {sh} != {DATA_SHA}')
    drv=import_local(import_path,'Day21_E_direct_driver')
    hc=drv.装好(0.75)
    if abs(float(drv.固定τ)-TAU)>1e-8 or abs(float(hc.预算)-.75)>1e-8:
        raise ValueError('Original driver budget/tau mismatch')
    return hc, data_path, model_path


def check_data_cost(data):
    a=np.asarray(data['act'],dtype=np.float64)
    direct=np.maximum(0,np.square(a).sum(-1)-TAU)
    difference=float(np.max(np.abs(direct-np.asarray(data['cost']))))
    if difference>1e-5: raise ValueError('Data cost/analytic action-cost mismatch '+str(difference))
    return difference


def selftest():
    # Mean of all neighbors differs sharply from top-10% mean.
    v=np.arange(1.,11.)
    assert abs(v.mean()-5.5) < 1e-12
    assert abs(v.mean()-v.max()) > 1
    rng=np.random.RandomState(0)
    for n in [1,2,5,9,10,11,19,20,100,300]:
        vv=rng.randn(n)
        assert abs(vv.mean()-float(np.sum(vv)/len(vv))) < 1e-12
    print('[PASS] neighborhood mean statistic, distinct from D top-10% mean.')


def run(project,seeds,steps,output,smoke=False,check=False,cache_only=False):
    hc,data_path,model_dir=get_day21(project)
    base= (project/'每日结果').resolve()
    if output==base or base in output.parents:
        raise ValueError('Never write to historical 每日结果')
    print('[E] input SHA256',sha256(data_path),flush=True)
    print('[E] critic caches read-only',
          {f.name:sha256(f) for f in [model_dir/'模型_HalfCheetah_风险critic.pt',model_dir/'模型_HalfCheetah_均值成本critic.pt']},flush=True)
    if check: print('[E] file/budget check passed; no training.'); return
    data, _c, meancritic=hc.准备(打印=True)
    check_data_cost(data)
    output.mkdir(parents=True,exist_ok=True)
    direct=DirectNeighborhoodMean(data,cache_path=output/'E_固定邻域均值标签_cache.npz')
    with (output/'E_邻域定义.json').open('w',encoding='utf-8') as f:
        json.dump(direct.meta,f,ensure_ascii=False,indent=2)
    if cache_only:
        print('[E] cached labels; no policy training.',flush=True);return
    import torch
    hc.日目录=str(output) # output only, keep critic cache at original model folder
    for seed in seeds:
        destination=output/f'结果_Day21_IQL_E邻域均值_种子{seed}.json'
        policy_path=output/f'模型_Day21_IQL_E邻域均值_种子{seed}.pt'
        if destination.exists() or policy_path.exists():
            raise FileExistsError(f'Refusing to overwrite existing E result/checkpoint for seed {seed}')
        initial=dict(direct._counts)
        start=time.time()
        policy,lam,log=hc.W2.训练一臂('IQL','尾部约束',data,direct,meancritic,
            步数=steps,种子=seed,打印间隔=max(200,steps//8),预算=.75)
        eval_result=hc.评估(policy,回合=hc.评估回合数,并行=6,种子=hc.评估初始种子)
        diagnostics={k:direct._counts[k]-initial[k] for k in direct._counts}
        if diagnostics['query']:
            diagnostics['policy_lookup_empty_fraction']=diagnostics['empty']/diagnostics['query']
        result=dict(实验='E组: 同一邻域的全部本步成本均值，不训练风险神经网络',
           原始模型训练函数='W2.训练一臂 IQL 尾部约束 [仅选用风险接口分支，统计量为邻域均值]',
           机制限定='与D组仅替换固定邻域标签统计量；自身尺度、lambda反馈仍会变化，不能声称绝对预算相同',
           种子=seed,训练步数=steps,预算rho=.75,tau=TAU,
           数据SHA256=DATA_SHA,邻域定义=direct.meta,邻域调用=diagnostics,
           评估=dict(回报均值=float(eval_result['回报均值']),
              成本均值=float(eval_result['成本均值']),
              成本CVaR25=float(eval_result['成本CVaR25'])),
           逐回合回报=np.asarray(eval_result['逐回合回报']).tolist(),
           逐回合成本=np.asarray(eval_result['逐回合成本']).tolist(),
           最终对偶变量=float(lam.值),对偶轨迹=list(lam.轨迹),训练日志=log,
           运行秒数=round(time.time()-start,1),冒烟=bool(smoke))
        torch.save(policy.state_dict(),policy_path)
        with destination.open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
        print('[E] seed',seed,'return',result['评估']['回报均值'],
              'cost',result['评估']['成本均值'],
              'CVaR25',result['评估']['成本CVaR25'],
              'lambda',result['最终对偶变量'],
              'empty neighborhood rate',diagnostics.get('policy_lookup_empty_fraction'),flush=True)
    print('[E] DONE. Inspect per-seed JSON and compare to historical A/B/C/D on the SAME eval protocol.')


def main():
    ap=argparse.ArgumentParser(description='Day21 E group: direct nonparametric neighborhood MEAN statistic')
    ap.add_argument('--project',default='.')
    ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--steps',type=int,default=40000)
    ap.add_argument('--output',default=None)
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--cache-only',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.selftest: selftest();return
    if args.steps<=0: raise ValueError('steps must be >0')
    project=Path(args.project).resolve()
    output=(Path(args.output).resolve() if args.output else
            project/'论文核验输出_E组直接邻域均值')
    if args.smoke:
        if args.steps>500: raise ValueError('Use --steps 200 with --smoke')
        output=output/'_冒烟'
    if not args.smoke and args.steps!=40000 and not args.check and not args.cache_only:
        raise ValueError('Main E group must use exactly 40000 steps')
    seeds=[int(s) for s in args.seeds.split(',') if s.strip()]
    if not seeds:raise ValueError('No seeds specified')
    run(project,seeds,args.steps,output,smoke=args.smoke,check=args.check,cache_only=args.cache_only)

if __name__=='__main__':
    main()
