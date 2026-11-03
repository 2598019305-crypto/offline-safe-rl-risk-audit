#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day21 D/E six-policy on-policy neighborhood statistics, READ-ONLY inputs.

Run from the ORIGINAL Windows project root (with Day21 D/E models already present):
    python 工具_Day21_DE六策略_只读回放风险诊断.py --selftest
    python 工具_Day21_DE六策略_只读回放风险诊断.py --check
    python 工具_Day21_DE六策略_只读回放风险诊断.py --episodes 6 --pilot
    python 工具_Day21_DE六策略_只读回放风险诊断.py --episodes 100

No policy training; no edits to 每日结果, D or E caches/models. The original 100-episode
fixed-batch evaluation is reproduced explicitly, with (s,a) interception BEFORE env.step.
The two statistics use ONE IDENTICAL KDTree query, radius and reference costs.
Results are independently written to 论文核验输出_DE六策略只读诊断 (or _冒烟).

Scientific boundaries:
- A decrease in on-policy D/E Spearman correlation alone does NOT establish OOD;
  compare nearest-reference distance/neighbor support with an independent data-state baseline.
- Pearson correlation is invariant to Z-scoring when measured on the same batch;
  reporting both doesn't independently test weighting. Also report z-score deviations/top-k overlap.
- Mean/cost/CVaR are evaluated with actual trajectory cost, NOT either proxy.
- Must reproduce each policy's previously recorded 100-episode result before interpreting.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

TAU = 3.9079
DATA_SHA = 'e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179'
POOL_SEED, N_QUERY, N_REF, K_RADIUS, ACTION_W = 0, 30000, 60000, 1500, 20.0
EP_START, PARALLEL, EP_LEN = 95000, 6, 1000
GROUP_CONFIG = {
    'D': ('论文核验输出_D组直接邻域上尾', 'D直接上尾'),
    'E': ('论文核验输出_E组直接邻域均值', 'E邻域均值'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''): h.update(x)
    return h.hexdigest()


def import_local(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if not len(finite): return dict(n=len(x), valid=0, mean=None, sd=None, cv=None, median=None, p90=None, p99=None)
    avg, sd = float(finite.mean()), float(finite.std())
    return dict(n=len(x), valid=len(finite), mean=avg, sd=sd,
                cv=(sd/abs(avg) if abs(avg)>1e-10 else None),
                median=float(np.median(finite)), p90=float(np.percentile(finite, 90)),
                p99=float(np.percentile(finite, 99)), zero_fraction=float(np.mean(finite == 0)))


def safe_spearman(x, y):
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10: return None
    value = float(spearmanr(x,y).statistic)
    return value if np.isfinite(value) else None


def safe_pearson(x,y):
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10: return None
    value = float(np.corrcoef(x,y)[0,1])
    return value if np.isfinite(value) else None


class OneGeometryTwoStatistics:
    def __init__(self, data, verbose=True):
        obs, act = np.asarray(data['obs'], np.float32), np.asarray(data['act'], np.float32)
        costs = np.asarray(data['cost'], np.float64)
        self.sm, self.ss = obs.mean(0), np.maximum(obs.std(0),1e-6)
        self.am, self.asd = act.mean(0), np.maximum(act.std(0),1e-6)
        self.num = len(obs)
        rng = np.random.RandomState(POOL_SEED)
        query_ids = rng.choice(len(obs),size=min(N_QUERY,len(obs)),replace=False)
        mask = np.ones(len(obs), dtype=bool); mask[query_ids] = False
        refs = np.where(mask)[0]
        if len(refs)>N_REF: refs = rng.choice(refs,size=N_REF,replace=False)
        self.ref_ids = refs
        self.tree = cKDTree(self.feature(obs[refs],act[refs]))
        self.ref_cost = costs[refs]
        self.query_ids = query_ids
        k = min(K_RADIUS,len(refs))
        q = query_ids[:min(500,len(query_ids))]
        dd,_ = self.tree.query(self.feature(obs[q],act[q]),k=k,workers=-1)
        self.radius = float(np.median(dd[:,-1] if dd.ndim == 2 else dd))
        self.data_obs, self.data_act = obs,act
        if verbose:
            print(f'[geometry] n={self.num}, refs={len(refs)}, radius={self.radius:.9f}, k={k}',flush=True)
        if not np.isfinite(self.radius) or self.radius<=0: raise RuntimeError('Invalid reference radius')

    def feature(self,s,a):
        return np.concatenate([(s-self.sm)/self.ss, ACTION_W*(a-self.am)/self.asd],axis=-1)

    def compute(self,obs,act,chunk=512,workers=1,progress_label=''):
        # Values are computed by exactly the D/E scripts' aggregation conventions:
        # upper 10%: k=max(1, round(0.1*len(neighbors))); all-neighbor mean.
        obs = np.asarray(obs,dtype=np.float32); act = np.asarray(act,dtype=np.float32)
        if obs.ndim != 2 or act.ndim != 2 or len(obs)!=len(act): raise ValueError('Invalid (s,a) shapes')
        n=len(obs)
        d=np.zeros(n,np.float32); e=np.zeros(n,np.float32)
        counts=np.zeros(n,np.int32); nearest=np.zeros(n,np.float32)
        t0=time.time()
        for start in range(0,n,chunk):
            end=min(start+chunk,n)
            f=self.feature(obs[start:end],act[start:end])
            neighbor_lists=self.tree.query_ball_point(f,self.radius,workers=workers)
            dn,_=self.tree.query(f,k=1,workers=workers)
            nearest[start:end]=np.asarray(dn,dtype=np.float32).reshape(-1)
            for offset, ids in enumerate(neighbor_lists):
                i=start+offset
                counts[i]=len(ids)
                if not ids: continue
                vals=self.ref_cost[np.asarray(ids,dtype=np.intp)]
                e[i]=float(vals.mean())
                k=max(1,int(round(len(vals)*0.10)))
                d[i]=float(np.partition(vals,len(vals)-k)[-k:].mean())
            if progress_label and (end==n or (end//chunk)%max(1, 5000//chunk)==0):
                print(f'  [{progress_label}] evaluated {end}/{n}, {time.time()-t0:.0f}s',flush=True)
        if np.any(d + 1e-5 < e): raise RuntimeError('D< E: invalid tail aggregation')
        return dict(D=d,E=e,ncount=counts,nn_dist=nearest)


def baseline(geometry, sample_count=4096,chunk=512,workers=1):
    ids = geometry.query_ids[:min(sample_count,len(geometry.query_ids))]
    # query_ids are excluded from the reference set, avoiding artificial NN distance=0.
    v=geometry.compute(geometry.data_obs[ids], geometry.data_act[ids], chunk,workers,'data baseline')
    return v


def z_and_batch_analysis(d,e,batch=256,perm_seed=20261010):
    d,e=np.asarray(d,np.float64),np.asarray(e,np.float64)
    rng=np.random.RandomState(perm_seed)
    order=rng.permutation(len(d))
    pears=[]; spears=[]; zmae=[]; top_overlap=[]
    for begin in range(0,len(order)-batch+1,batch):
        ids=order[begin:begin+batch]
        a,b=d[ids],e[ids]
        if np.std(a)<1e-10 or np.std(b)<1e-10: continue
        # Match W2 torch.std() (unbiased=True) and +1e-6 convention.
        za=(a-a.mean())/(a.std(ddof=1)+1e-6)
        zb=(b-b.mean())/(b.std(ddof=1)+1e-6)
        pears.append(safe_pearson(za,zb))
        spears.append(safe_spearman(a,b))
        zmae.append(float(np.mean(np.abs(za-zb))))
        k=max(1,int(round(batch*.10)))
        ia=set(np.argpartition(a,-k)[-k:]); ib=set(np.argpartition(b,-k)[-k:])
        top_overlap.append(len(ia&ib)/k)
    pears=[i for i in pears if i is not None]; spears=[i for i in spears if i is not None]
    return dict(batch=batch,complete_batches=len(order)//batch,valid_batches=len(zmae),
                pearson_of_batch_z=summary(pears),spearman_batch=summary(spears),
                mean_abs_z_difference=summary(zmae),top10pct_overlap=summary(top_overlap),
                note='Pearson(zD,zE) equals Pearson(D,E) within each nonconstant batch; z-difference and top-k overlap test different aspects.')


def rollout(hc,policy,episodes):
    # Mirrors 公共_定批回放引擎.定批回放() with fixed batch size, plus read-only (s,a) logging.
    import torch
    if episodes<1: raise ValueError('episodes must be positive')
    W=min(PARALLEL,episodes)
    envs=[hc.建环境() for _ in range(W)]
    obs=np.zeros((W,hc.W2.观察维),np.float32)
    alive=np.ones(W,bool)
    owners=np.arange(W,dtype=np.int32)
    episode_steps=np.zeros(W,np.int32)
    next_idx=W
    ep_ret=np.zeros(episodes,np.float64);ep_cost=np.zeros(episodes,np.float64)
    cur_ret=np.zeros(W,np.float64);cur_cost=np.zeros(W,np.float64)
    s_parts=[]; a_parts=[]; ep_parts=[]; t_parts=[]
    try:
        for i in range(W):
            obs[i],_=envs[i].reset(seed=EP_START+i)
        iterations=0
        finished_episodes=0
        while alive.any():
            batch=np.zeros((W,hc.W2.观察维),np.float32)
            batch[alive]=obs[alive]
            with torch.no_grad():
                action=policy.均值动作(torch.tensor(batch,dtype=torch.float32)).numpy().astype(np.float32)
            live=np.where(alive)[0]
            s_parts.append(batch[live].copy()); a_parts.append(action[live].copy())
            ep_parts.append(owners[live].copy()); t_parts.append(episode_steps[live].copy())
            for i in live:
                oo,rr,te,tr,info=envs[i].step(action[i])
                cur_ret[i]+=float(rr); cur_cost[i]+=float(info.get('cost',0.0))
                obs[i]=oo
                episode_steps[i]+=1
                if te or tr:
                    finished_episodes+=1
                    ep_ret[owners[i]]=cur_ret[i]; ep_cost[owners[i]]=cur_cost[i]
                    cur_ret[i]=0.;cur_cost[i]=0.
                    if next_idx<episodes:
                        obs[i],_=envs[i].reset(seed=EP_START+next_idx)
                        owners[i]=next_idx
                        episode_steps[i]=0
                        next_idx+=1
                    else:
                        alive[i]=False; owners[i]=-1
            iterations+=1
            if iterations%5000==0:
                print(f'  [rollout] rounds {iterations}, finished episodes {finished_episodes}',flush=True)
    finally:
        for env in envs: env.close()
    return dict(obs=np.concatenate(s_parts),act=np.concatenate(a_parts),
                episode=np.concatenate(ep_parts),timestep=np.concatenate(t_parts),
                episode_return=ep_ret,episode_cost=ep_cost,
                mean_return=float(ep_ret.mean()),mean_cost=float(ep_cost.mean()),
                cvar25=float(np.sort(ep_cost)[-max(1,episodes//4):].mean()))


def verify_rollout(roll,ref_json,episodes,pilot):
    if pilot or episodes!=100: return dict(checked=False,reason='pilot or non-100 episode evaluation')
    with open(ref_json,'r',encoding='utf-8') as f: prior=json.load(f)
    r,c = np.asarray(prior['逐回合回报'],np.float64),np.asarray(prior['逐回合成本'],np.float64)
    if len(r)!=100 or len(c)!=100: raise RuntimeError('Historical JSON lacks 100 episodes')
    dr=np.abs(roll['episode_return']-r); dc=np.abs(roll['episode_cost']-c)
    match=(np.allclose(roll['episode_return'],r,rtol=1e-7,atol=1e-3) and
           np.allclose(roll['episode_cost'],c,rtol=1e-7,atol=1e-3))
    info=dict(checked=True,match=bool(match),max_abs_return=float(dr.max()),max_abs_cost=float(dc.max()),
              historical_return=float(r.mean()),historical_cost=float(c.mean()))
    if not match:
        raise RuntimeError(f'Fixed-batch 100-episode replay mismatch! {info}. Stop: inspect model/environment/version before interpreting risk statistics.')
    return info


def policy_metrics(roll,v,base):
    c=np.maximum(0.,np.square(roll['act'].astype(np.float64)).sum(1)-TAU)
    nz=np.abs(c)<1e-10
    nd=v['nn_dist']; bn=base['nn_dist']
    threshold=float(np.quantile(bn,0.95))
    analysis=dict(
       transitions=int(len(c)),
       one_step_analytic=summary(c),
       D_tail=summary(v['D']),E_mean=summary(v['E']),
       neighbor_count=summary(v['ncount']),nearest_distance=summary(nd),
       empty_fraction=float(np.mean(v['ncount']==0)),
       above_data_nn_distance_p95=float(np.mean(nd>threshold)),
       data_nn_distance_p95=threshold,
       spearman_D_E=safe_spearman(v['D'],v['E']),
       pearson_D_E=safe_pearson(v['D'],v['E']),
       zero_immediate_count=int(nz.sum()),
       zero_immediate_spearman_D_E=safe_spearman(v['D'][nz],v['E'][nz]),
       zero_immediate_D=summary(v['D'][nz]),zero_immediate_E=summary(v['E'][nz]),
       nonzero_indicator_agreement=float(np.mean((v['D']>0)==(v['E']>0))),
       batch_z=z_and_batch_analysis(v['D'],v['E']),
       rollout=dict(mean_return=roll['mean_return'],mean_cost=roll['mean_cost'],cvar25=roll['cvar25']))
    return analysis


def selftest():
    x=np.arange(1,11,dtype=np.float64)
    top=float(np.partition(x,9)[-1:].mean())
    assert top==10 and float(x.mean())==5.5
    assert np.isclose(safe_spearman(x,2*x),1.0,rtol=1e-12,atol=1e-12)
    assert np.isclose(z_and_batch_analysis(np.arange(512),2*np.arange(512))['mean_abs_z_difference']['mean'],0)
    rng=np.random.RandomState(3)
    assert summary(rng.normal(size=100))['valid']==100
    print('[PASS] tail, correlation, z-score invariance and summary selftests')


def main():
    ap=argparse.ArgumentParser(description='Day21 D/E six-policy fixed-batch risk diagnostic: strictly no training')
    ap.add_argument('--project',default='.',help='Windows Day21 project root')
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--seeds',default='0,1,2',help='Same seed numbers for D and E')
    ap.add_argument('--chunk',type=int,default=512)
    ap.add_argument('--workers',type=int,default=1,help='KDTree query threads, 1 is conservative')
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--selftest',action='store_true')
    args=ap.parse_args()
    if args.selftest: selftest();return
    if args.episodes<1 or args.chunk<1 or args.workers<1: ap.error('episodes, chunk and workers must be positive')
    if not args.pilot and args.episodes!=100: ap.error('formal diagnostic must be exactly 100 episodes; otherwise add --pilot')
    project=Path(args.project).resolve()
    data_file=project/'每日结果'/'Day20_路径二状态相关代价'/'数据_HalfCheetah_异质.npz'
    drv_file=project/'工具_HC_异质驱动.py'
    if not data_file.is_file() or not drv_file.is_file():
        raise FileNotFoundError(f'Day21 data/driver missing: {data_file}; {drv_file}')
    sh=sha256(data_file)
    if sh != DATA_SHA: raise RuntimeError(f'Dataset SHA mismatch: got {sh}; expected {DATA_SHA}')
    seeds=[int(v) for v in args.seeds.split(',') if v.strip()]
    if not seeds or len(seeds)!=len(set(seeds)): raise ValueError('Bad or duplicate seeds')
    configs=[]
    for group,(dirname,stem) in GROUP_CONFIG.items():
        for seed in seeds:
            root=project/dirname
            model=root/f'模型_Day21_IQL_{stem}_种子{seed}.pt'
            result=root/f'结果_Day21_IQL_{stem}_种子{seed}.json'
            if not model.is_file() or not result.is_file():
                raise FileNotFoundError(f'Missing {group} seed{seed} checkpoint/result: {model}; {result}')
            configs.append((group,seed,model,result))
    print(f'[CHECK] dataset SHA OK; {len(configs)} D/E checkpoint+JSON pairs exist; no source files will be written',flush=True)
    if args.check: return

    drv=import_local(drv_file,'day21_de_readonly_driver')
    if abs(float(drv.固定τ)-TAU)>1e-8: raise RuntimeError('Driver TAU mismatch')
    hc=drv.装好(.75)
    data,tau=hc.读并标号() # Does not call 准备(), never creates/trains critics.
    if abs(float(tau)-TAU)>1e-8: raise RuntimeError('Cost threshold mismatch')
    hc.数据阈值[0]=TAU  # Important: same wrapper analytic cost as original official evaluation.
    geometry=OneGeometryTwoStatistics(data)
    base=baseline(geometry,chunk=args.chunk,workers=args.workers)
    baseline_stats=dict(data_spearman_D_E=safe_spearman(base['D'],base['E']),
                        data_pearson_D_E=safe_pearson(base['D'],base['E']),
                        data_nn_distance=summary(base['nn_dist']),
                        data_neighbor_count=summary(base['ncount']),
                        data_batch_z=z_and_batch_analysis(base['D'],base['E']))
    print('[baseline] Spearman:',baseline_stats['data_spearman_D_E'],
          'Pearson:',baseline_stats['data_pearson_D_E'],flush=True)
    outfile=project/'论文核验输出_DE六策略只读诊断'
    if args.pilot: outfile=outfile/'_冒烟'
    history=project/'每日结果'
    if outfile.resolve()==history.resolve() or history.resolve() in outfile.resolve().parents:
        raise ValueError('Refuse writing to 每日结果')
    outfile.mkdir(parents=True,exist_ok=True)
    manifest=outfile/'DE诊断_逐种子汇总.json'
    reports=[]
    import torch
    for group,seed,model_path,ref_path in configs:
        ident=f'{group}_seed{seed}_episodes{args.episodes}'
        jpath=outfile/f'{ident}_统计.json'
        npzpath=outfile/f'{ident}_逐转移.npz'
        if jpath.exists() and npzpath.exists():
            print(f'[SKIP] {ident}: complete output exists',flush=True)
            with jpath.open(encoding='utf-8') as f: reports.append(json.load(f))
            continue
        if jpath.exists() or npzpath.exists(): raise FileExistsError(f'Partial existing output for {ident}: inspect manually')
        print(f'[{ident}] replay starting',flush=True)
        start=time.time()
        model=hc.W2.约束IQL()
        weights=torch.load(str(model_path),map_location='cpu',weights_only=True)
        model.load_state_dict(weights,strict=True)
        model.eval()
        with torch.no_grad(): r=rollout(hc,model,args.episodes)
        consistency=verify_rollout(r,ref_path,args.episodes,args.pilot)
        print(f'[{ident}] rollout {len(r["obs"])} transitions; return={r["mean_return"]:.1f}, cost={r["mean_cost"]:.2f}, CVaR={r["cvar25"]:.2f}; replay consistency={consistency}',flush=True)
        v=geometry.compute(r['obs'],r['act'],chunk=args.chunk,workers=args.workers,progress_label=ident)
        stat=policy_metrics(r,v,base)
        rep=dict(group=group,seed=seed,episodes=args.episodes,data_SHA256=sh,
                 model_SHA256=sha256(model_path),model_file=str(model_path),
                 reference_result_file=str(ref_path),radius=geometry.radius,
                 pool_seed=POOL_SEED,references=len(geometry.ref_ids),
                 eval_start_seed=EP_START,parallel=PARALLEL,deterministic=True,
                 verified_replay=consistency,baseline=baseline_stats,
                 metrics=stat,elapsed_sec=round(time.time()-start,2))
        tempnpz=npzpath.with_suffix('.npz.tmp')
        with open(tempnpz,'wb') as f:
            np.savez_compressed(f,s=r['obs'],a=r['act'],episode=r['episode'],
                timestep=r['timestep'],analytic_cost=np.maximum(0,np.square(r['act'].astype(np.float64)).sum(1)-TAU).astype(np.float32),
                D_tail=v['D'],E_mean=v['E'],neighbor_count=v['ncount'],nearest_reference_distance=v['nn_dist'],
                episode_return=r['episode_return'],episode_cost=r['episode_cost'])
        os.replace(str(tempnpz),str(npzpath))
        # Write stats atomically, last. Existing policy files are NEVER changed.
        tmpj=jpath.with_suffix('.json.tmp')
        with tmpj.open('w',encoding='utf-8') as f: json.dump(rep,f,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(str(tmpj),str(jpath))
        reports.append(rep)
        print(f'[DONE] {ident}: Spearman D/E={stat["spearman_D_E"]}; empty={stat["empty_fraction"]:.4%}; data-distance-p95-exceedance={stat["above_data_nn_distance_p95"]:.2%}',flush=True)
    if manifest.exists():
        print(f'[SKIP] aggregate already exists; verified {len(reports)} saved policy reports: {manifest}',flush=True)
        return
    with manifest.open('x',encoding='utf-8') as f:
        json.dump(dict(protocol='Day21 six-policy read-only rollout risk diagnostic',
             data_sha256=sh,episode_seed=EP_START,episodes=args.episodes,parallel=PARALLEL,
             baseline=baseline_stats,policies=reports,
             interpretation='A Spearman decline plus larger nearest-distance/more empty neighborhoods supports, but does not prove, an OOD-related proxy divergence.'),
             f,ensure_ascii=False,indent=2,allow_nan=False)
    print(f'[COMPLETE] {len(reports)} policies; summary: {manifest}',flush=True)


if __name__=='__main__':
    main()
