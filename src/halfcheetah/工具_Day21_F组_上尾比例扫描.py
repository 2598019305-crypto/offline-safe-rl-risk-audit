#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day21 / IQL: pre-registered q=0.25/0.50 neighborhood upper-cost-fraction scan.

Intended scientific comparison:
 D: same reference-sample local top-10% statistic (q=0.10).
 E: all-neighborhood mean statistic (q=1.00).
 This script trains ONLY intermediate q=0.25 and q=0.50, without NN.

The labels are upper-fraction means of ONE-STEP NEIGHBORHOOD COSTS, not trajectory CVaR.
This tool holds reference geometry and training protocol fixed, varying only q.

Original training source W2.训练一臂 is imported unchanged. This method is valid
 for the *IQL actor branch* because W2 computes risk signals only under no_grad.
 The direct statistic is nondifferentiable; do NOT use this class as a TD3+BC
 risk critic (there the actor directly differentiates through risk).

Dependencies: original Day21 project in --project, dsrl-env with torch/gymnasium.
 In the original project folder:
    python 工具_Day21_F组_上尾比例扫描.py --selftest
    python 工具_Day21_F组_上尾比例扫描.py --q 0.25 --check
    python 工具_Day21_F组_上尾比例扫描.py --q 0.25 --cache-only
    python 工具_Day21_F组_上尾比例扫描.py --q 0.25 --seeds 99 --steps 200 --smoke
    python 工具_Day21_F组_上尾比例扫描.py --q 0.25 --seeds 0,1,2 --steps 40000
    (repeat for --q 0.50 after success)

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
EXPERIMENT_FRACTIONS = (0.25, 0.50)


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


class DirectNeighborhoodFractionTail:
    """Reference-statistic lookup, matching W4's defaults for Day21.

    For data (s,a), return a precomputed label based on the same reference set.
    For freshly sampled policy actions, calculate the same neighborhood statistic
    on demand. An empty neighborhood returns zero, matching W4's label builder;
    count such misses (OOD/generalization issue) and disclose them.
    """
    def __init__(self, data, *, tail_fraction, cache_path=None, verbose=True):
        self.tail_fraction = float(tail_fraction)
        if self.tail_fraction not in EXPERIMENT_FRACTIONS:
            raise ValueError("Only pre-registered fractions q=0.25 and q=0.50 allowed")
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
        self.meta = dict(kind='direct local upper-fraction-mean one-step cost',
              radius=self.radius, ref_count=len(refs), n=self.n,
              radius_k=k, action_weight=ACTION_WEIGHT, tail_fraction=self.tail_fraction,
              pool_seed=POOL_SEED, n_query_seed=min(N_QUERY,self.n),
              convention='empty neighborhood -> 0 (original W4 definition)',
              cost_discount=0.0)
        if verbose:
            print('[F-scan] pool radius:', round(self.radius,6),
                  'references:',len(refs),'computing/loading data labels...',flush=True)
        labels = None
        if cache_path is not None and Path(cache_path).exists():
            z = np.load(str(cache_path),allow_pickle=False)
            if ('labels' in z and 'ref_ids' in z and 'radius' in z and 'tail_fraction' in z
                and int(z['labels'].shape[0])==self.n
                and np.array_equal(z['ref_ids'],self.ref_ids)
                and abs(float(z['radius'])-self.radius) < 1e-8
                and abs(float(z['tail_fraction'])-self.tail_fraction) < 1e-12):
                labels = np.asarray(z['labels'],dtype=np.float32)
                if verbose: print('[F-scan] data-label cache loaded',flush=True)
            else:
                raise RuntimeError('F cache geometry or tail-fraction mismatch; STOP instead of silently reusing')
        if labels is None:
            labels = np.empty(self.n,dtype=np.float32)
            for start in range(0,self.n,1000):
                end = min(self.n,start+1000)
                labels[start:end] = self._lookup(self._features(obs[start:end],act[start:end]), count=False)
                if verbose and (end%25000==0 or end==self.n):
                    print(f'[F-scan] labels {end}/{self.n}',flush=True)
            if cache_path is not None:
                target = Path(cache_path)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    raise FileExistsError(f'Refuse overwrite: {target}')
                np.savez_compressed(str(target),labels=labels,ref_ids=self.ref_ids,radius=self.radius,tail_fraction=self.tail_fraction)
        self.labels = labels
        # Exact float32 match: train batches are slices of the same data, so the
        # cached signal can replace 10m+ repeated KDTree queries without changing it.
        self.key_to_idx = {self._data_rows[i].tobytes():i for i in range(self.n)}
        if verbose:
            print('[F-scan] label mean/std/range:',float(labels.mean()),float(labels.std()),
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
            topk=max(1,int(round(len(vals)*self.tail_fraction)))
            y[i] = float(np.partition(vals,len(vals)-topk)[-topk:].mean())
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
    drv=import_local(import_path,'Day21_F_fraction_driver')
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
    # Validate D/E boundary cases and q=0.25/0.50 intermediate variants.
    rng=np.random.RandomState(0)
    for n in [1,2,5,9,10,11,19,20,100,300]:
        vv=rng.uniform(-2,4,size=n)
        for q in [0.10,0.25,0.50,1.0]:
            k=max(1,int(round(n*q)))
            selected=np.partition(vv,n-k)[-k:]
            expected=np.sort(vv)[-k:]
            assert abs(float(selected.mean())-float(expected.mean())) < 1e-12
            if q==1.0:
                assert abs(float(selected.mean())-float(vv.mean())) < 1e-12
    v=np.arange(1.,101.)
    assert abs(np.partition(v,75)[-25:].mean()-88.0)<1e-12
    assert abs(np.partition(v,50)[-50:].mean()-75.5)<1e-12
    assert abs(v.mean()-50.5)<1e-12
    assert abs(np.partition(v,90)[-10:].mean()-95.5)<1e-12
    print('[PASS] q scan follows D original rounding, top-fraction mean; E/D endpoints reproduced mathematically.')


def run(project,seeds,steps,output,q,smoke=False,check=False,cache_only=False):
    hc,data_path,model_dir=get_day21(project)
    base= (project/'每日结果').resolve()
    if output==base or base in output.parents:
        raise ValueError('Never write to historical 每日结果')
    forbidden=[project/'论文核验输出_D组直接邻域上尾',project/'论文核验输出_E组直接邻域均值']
    if any(output==old or old in output.parents for old in forbidden):
        raise ValueError('Never overwrite historical D/E experiment outputs')
    print('[F-scan] input SHA256',sha256(data_path),flush=True)
    print('[F-scan] critic caches read-only',
          {f.name:sha256(f) for f in [model_dir/'模型_HalfCheetah_风险critic.pt',model_dir/'模型_HalfCheetah_均值成本critic.pt']},flush=True)
    if check: print('[F-scan] file/budget check passed; q='+str(q)+'; no training.'); return
    data, _c, meancritic=hc.准备(打印=True)
    check_data_cost(data)
    output.mkdir(parents=True,exist_ok=True)
    direct=DirectNeighborhoodFractionTail(data,tail_fraction=q,cache_path=output/f'F_q{int(q*100):02d}_固定邻域标签_cache.npz')
    definition_path=output/f'F_q{int(q*100):02d}_邻域定义.json'
    if definition_path.exists():
        with definition_path.open('r',encoding='utf-8') as f:
            existing=json.load(f)
        if existing!=direct.meta:
            raise RuntimeError('F metadata mismatch; do not reuse this output directory')
    else:
        with definition_path.open('x',encoding='utf-8') as f:
            json.dump(direct.meta,f,ensure_ascii=False,indent=2)
    if cache_only:
        print('[F-scan] cached labels; no policy training.',flush=True);return
    import torch
    hc.日目录=str(output) # output only, keep critic cache at original model folder
    for seed in seeds:
        destination=output/f'结果_Day21_IQL_F上尾{int(q*100):02d}_种子{seed}.json'
        policy_path=output/f'模型_Day21_IQL_F上尾{int(q*100):02d}_种子{seed}.pt'
        if destination.exists() or policy_path.exists():
            raise FileExistsError(f'Refusing to overwrite existing F result/checkpoint for seed {seed}')
        initial=dict(direct._counts)
        start=time.time()
        policy,lam,log=hc.W2.训练一臂('IQL','尾部约束',data,direct,meancritic,
            步数=steps,种子=seed,打印间隔=max(200,steps//8),预算=.75)
        eval_result=hc.评估(policy,回合=hc.评估回合数,并行=6,种子=hc.评估初始种子)
        diagnostics={k:direct._counts[k]-initial[k] for k in direct._counts}
        if diagnostics['query']:
            diagnostics['policy_lookup_empty_fraction']=diagnostics['empty']/diagnostics['query']
        result=dict(实验=f'F组扫描: 邻域最高{int(q*100)}%一步成本均值，不训练风险神经网络',
           原始模型训练函数='W2.训练一臂 IQL 尾部约束',
           机制限定='与D/E同一邻域、同IQL训练，仅切换邻域一步成本上尾比例；不是轨迹CVaR',
           种子=seed,训练步数=steps,预算rho=.75,tau=TAU,上尾比例=q,
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
        print('[F-scan] seed',seed,'return',result['评估']['回报均值'],
              'cost',result['评估']['成本均值'],
              'CVaR25',result['评估']['成本CVaR25'],
              'lambda',result['最终对偶变量'],
              'empty neighborhood rate',diagnostics.get('policy_lookup_empty_fraction'),flush=True)
    print('[F-scan] DONE. Compare paired seeds to q=1.0(E) and q=0.1(D); same eval protocol.')


def main():
    ap=argparse.ArgumentParser(description='Day21 F group: q=0.25/0.50 nonparametric upper-cost-fraction mean scan')
    ap.add_argument('--project',default='.')
    ap.add_argument('--q',type=float,default=None,help='Required for experiments: 0.25 or 0.50')
    ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--steps',type=int,default=40000)
    ap.add_argument('--output',default=None)
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--cache-only',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.selftest: selftest();return
    if args.q not in EXPERIMENT_FRACTIONS:
        raise ValueError('--q must be 0.25 or 0.50; D=0.10 and E=1.0 are historical and must not be overwritten')
    if args.steps<=0: raise ValueError('steps must be >0')
    project=Path(args.project).resolve()
    output=(Path(args.output).resolve() if args.output else
            project/f'论文核验输出_F组上尾比例_q{int(args.q*100):02d}')
    if args.smoke:
        if args.steps>500: raise ValueError('Use --steps 200 with --smoke')
        output=output/'_冒烟'
    if not args.smoke and args.steps!=40000 and not args.check and not args.cache_only:
        raise ValueError('Main F q-scan must use exactly 40000 steps')
    seeds=[int(s) for s in args.seeds.split(',') if s.strip()]
    if not seeds:raise ValueError('No seeds specified')
    run(project,seeds,args.steps,output,args.q,smoke=args.smoke,check=args.check,cache_only=args.cache_only)

if __name__=='__main__':
    main()
