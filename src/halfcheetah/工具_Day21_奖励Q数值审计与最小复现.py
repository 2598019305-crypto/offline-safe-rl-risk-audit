#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day21 reward-Q numerical audit and minimal reward-subsystem probe.

NO MuJoCo, NO existing model/data modification. `--audit` loads six F models;
`--probe` trains separate Q/V diagnostic networks only, NEVER full policies.
Uses original W2 reward-Q / V loss definitions with independent paired batches.
Not a bit-for-bit recreation of W2 joint actor/dual training.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

SHA='e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179'
GAMMA=0.99
ROOT=Path(__file__).resolve().parent

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def project_data(p):
    candidates=[p/'每日结果'/'Day20_路径二状态相关代价'/'数据_HalfCheetah_异质.npz',
                p/'F_scan_audit'/'数据_HalfCheetah_异质.npz']
    for f in candidates:
        if f.exists():return f
    raise FileNotFoundError('缺少 Day21 200k 异质数据：'+str(candidates[0]))

def model_path(p,q,seed):
    name=f'模型_Day21_IQL_F上尾{q}_种子{seed}.pt'
    candidates=[p/f'论文核验输出_F组上尾比例_q{q}'/name,
                p/'F_scan_audit'/f'输出_q{q}'/name]
    for f in candidates:
        if f.exists():return f
    raise FileNotFoundError(f'找不到 F{q} seed{seed} 模型：{candidates[0]}')

def load_dataset(p):
    f=project_data(p); h=sha256(f)
    if h!=SHA:raise RuntimeError(f'Day21 数据 SHA 不一致：{h}; 要求 {SHA}')
    with np.load(f,allow_pickle=False) as z:
        d={k:np.asarray(z[k],dtype=np.float32) for k in ('obs','act','rew','done')}
        cost=np.asarray(z['cost'],dtype=np.float32)
    n=len(d['obs']);assert n==200000 and d['obs'].shape[1]==17 and d['act'].shape[1]==6
    assert all(len(x)==n for x in d.values())
    if any(not np.isfinite(x).all() for x in d.values()):raise ValueError('数据含非有限值')
    # EXACT reconstruction used in 阶段5_HalfCheetah机制验证.py:读并标号
    s2=np.zeros_like(d['obs']);s2[:-1]=d['obs'][1:];s2[:-1][d['done'][:-1]>.5]=0.0
    d['obs2']=s2
    sq=np.sum(d['act'].astype(np.float64)**2,axis=1)
    cost_err=float(np.max(np.abs(np.maximum(0,sq-3.9079)-cost)))
    if cost_err>1e-4:raise RuntimeError(f'成本阈值/数据不一致，最大误差 {cost_err}')
    meta={'path':str(f),'sha256':h,'n':n,'r_mean':float(d['rew'].mean()),
          'r_min':float(d['rew'].min()),'r_max':float(d['rew'].max()),
          'done_fraction':float(d['done'].mean()),'cost_max_abs_error':cost_err}
    return d,meta

def network(i,o):
    return nn.Sequential(nn.Linear(i,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,o))

class OriginalIQL(nn.Module):
    def __init__(self,dim_s=17,dim_a=6):
        super().__init__()
        self.q1=network(dim_s+dim_a,1);self.q2=network(dim_s+dim_a,1)
        self.vr=network(dim_s,1);self.演员=network(dim_s,2*dim_a)
        # Must match W2 actor initialization, even though actor is not trained here.
        self.演员[-1].weight.data.mul_(0.01);self.演员[-1].bias.data.zero_()

def vals(x):
    z=np.asarray(x,dtype=np.float64).reshape(-1)
    if len(z)==0:return None
    return {'min':float(np.min(z)),'p01':float(np.percentile(z,1)),
            'p50':float(np.median(z)),'p99':float(np.percentile(z,99)),
            'max':float(np.max(z)),'mean':float(z.mean()),'std':float(z.std()),
            'finite_fraction':float(np.isfinite(z).mean())}

@torch.no_grad()
def eval_fixed(m,d,ids):
    s=torch.from_numpy(d['obs'][ids]);a=torch.from_numpy(d['act'][ids]);
    s2=torch.from_numpy(d['obs2'][ids]);r=torch.from_numpy(d['rew'][ids]).reshape(-1,1)
    done=torch.from_numpy(d['done'][ids]).reshape(-1,1)
    q1=m.q1(torch.cat([s,a],-1));q2=m.q2(torch.cat([s,a],-1));v=m.vr(s)
    target=r+GAMMA*(1-done)*m.vr(s2)
    mini=torch.minimum(q1,q2)
    td=target-mini
    adv=mini-v
    return {'Qmin':vals(mini.numpy()),'Q1':vals(q1.numpy()),'Q2':vals(q2.numpy()),
            'V':vals(v.numpy()),'bellman_target':vals(target.numpy()),
            'bellman_residual':vals(td.numpy()),'bellman_rmse':float(td.square().mean().sqrt()),
            'advantage':vals(adv.numpy()),'abs_Q_gt_10000':float((mini.abs()>10000).float().mean()),
            'abs_V_gt_10000':float((v.abs()>10000).float().mean()),
            'corr_Q_V':float(np.corrcoef(mini.flatten().numpy(),v.flatten().numpy())[0,1])}

def save_new(directory,name,obj):
    directory.mkdir(parents=True,exist_ok=True)
    dest=directory/name
    with dest.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    print('输出:',dest,flush=True)
    return dest

def check(p):
    d,meta=load_dataset(p)
    files=[]
    for q in (25,50):
        for seed in (0,1,2):
            f=model_path(p,q,seed)
            w=torch.load(f,map_location='cpu',weights_only=True)
            m=OriginalIQL();m.load_state_dict(w,strict=True)
            files.append({'q':q,'seed':seed,'name':str(f),'bytes':f.stat().st_size,'sha256':sha256(f)})
    print('数据校验通过；模型 6/6 可加载；数据奖励范围:',meta['r_min'],meta['r_max'])
    return d,meta,files

def audit(p,out):
    d,meta,files=check(p)
    ids=np.random.RandomState(20261011).choice(len(d['obs']),4096,replace=False)
    checks=[]
    torch.set_num_threads(2)
    for entry in files:
        m=OriginalIQL();m.load_state_dict(torch.load(entry['name'],map_location='cpu',weights_only=True));m.eval()
        e=eval_fixed(m,d,ids)
        checks.append({**entry,'metrics':e})
        print(f"F{entry['q']} s{entry['seed']} Q50={e['Qmin']['p50']:.5g} V50={e['V']['p50']:.5g} TD_RMSE={e['bellman_rmse']:.5g} adv50={e['advantage']['p50']:.5g}",flush=True)
    obj={'experiment':'Day21 F checkpoint read-only reward Q/V + Bellman residual audit',
         'scope':'4096 random fixed data state-action transitions; no environment rollout / no training',
         'input':meta,'checks':checks}
    save_new(out,'P0_六模型数值审计.json',obj)

def expectile(diff,tau=.7):
    return (torch.where(diff>0,tau,1-tau)*diff.pow(2)).mean()

def selftest():
    d=torch.tensor([-2.,-1.,0.,1.,2.]);a=expectile(d,.5)
    assert torch.allclose(a,0.5*d.square().mean())
    assert math.isclose(float(expectile(d,.7)),float(((torch.where(d>0,.7,.3))*d.square()).mean()),rel_tol=1e-7)
    a=np.arange(10,dtype=np.float32); done=np.array([0,0,1,0,0,0,0,0,0,1],dtype=np.float32)
    nxt=np.zeros_like(a);nxt[:-1]=a[1:];nxt[:-1][done[:-1]>.5]=0
    assert nxt[2]==0 and nxt[1]==2 and nxt[-1]==0
    print('PASS: expectile 方向和 obs2 终止掩码自检')

def probe(p,out,seed,steps,interval,ceil):
    if steps<1 or steps>10000:raise ValueError('短程诊断限制 steps 1..10000；不得当正式 40k 策略训练')
    d,meta=load_dataset(p)
    torch.set_num_threads(2)
    states=torch.from_numpy(d['obs']);acts=torch.from_numpy(d['act'])
    nxt=torch.from_numpy(d['obs2']); rew=torch.from_numpy(d['rew']).reshape(-1,1)
    done=torch.from_numpy(d['done']).reshape(-1,1)
    ids=np.random.RandomState(20261011).choice(len(d['obs']),4096,replace=False)
    # Paired batches are independent of original training RNG: controlled loss comparison,
    # NOT bitwise replay of original joint actor/critic trainer.
    torch.manual_seed(seed)
    initial=OriginalIQL().state_dict()
    models={}; opt={}
    for name in ('legacy_expectile_Q','standard_MSE_Q'):
        m=OriginalIQL();m.load_state_dict(initial);models[name]=m
        opt[name]=torch.optim.Adam(m.parameters(),lr=3e-4)
    gen=torch.Generator(device='cpu').manual_seed(100000+seed)
    records=[];started=time.time();stopped={}
    def record(step):
        row={'step':step,'elapsed_sec':round(time.time()-started,1)}
        for name,m in models.items():
            row[name]=eval_fixed(m,d,ids)
        records.append(row)
        print('Q-only probe',step,' '.join(f"{k}:Q50={row[k]['Qmin']['p50']:.3g},TD={row[k]['bellman_rmse']:.3g}" for k in models),flush=True)
    record(0)
    for step in range(1,steps+1):
        j=torch.randint(0,len(states),(256,),generator=gen)
        for name,m in models.items():
            if name in stopped:continue
            with torch.no_grad():target=rew[j]+GAMMA*(1-done[j])*m.vr(nxt[j])
            q1=m.q1(torch.cat([states[j],acts[j]],-1));q2=m.q2(torch.cat([states[j],acts[j]],-1))
            if name=='legacy_expectile_Q':lq=expectile(target-q1)+expectile(target-q2)
            else:lq=(target-q1).pow(2).mean()+(target-q2).pow(2).mean()
            lv=expectile(torch.min(q1,q2).detach()-m.vr(states[j]))
            loss=lq+lv
            if not bool(torch.isfinite(loss)):
                stopped[name]={'step':step,'reason':'loss not finite'};continue
            opt[name].zero_grad();loss.backward();opt[name].step()
        if step%interval==0 or step==steps:
            record(step)
            for name,m in models.items():
                if name not in stopped:
                    z=records[-1][name]
                    if (not math.isfinite(z['Qmin']['p50'])) or abs(z['Qmin']['p50'])>ceil or z['bellman_rmse']>ceil:
                        stopped[name]={'step':step,'reason':f'oversized median-Q or TD > {ceil:g}'}
                        print('STOP variant',name,stopped[name],flush=True)
        if len(stopped)==len(models):break
    obj={'experiment':'Day21 reward-only paired loss-function probe',
         'scope':'Only reward Q/V updated; paired index batches; no actor, dual, risk, or MuJoCo. NOT bitwise replication of full W2 training.',
         'data':meta,'seed':seed,'steps_requested':steps,'steps_finished':step,
         'batch':256,'gamma':GAMMA,'lr':.0003,'value_expectile':.7,
         'legacy_q_expectile':.7,'standard_q_loss':'MSE',
         'stopped':stopped,'records':records}
    save_new(out,f'P1_短程奖励Q双损失_种子{seed}_{steps}步.json',obj)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=ROOT,help='原工程根目录（默认脚本目录）')
    ap.add_argument('--output',type=Path,default=None,help='独立输出目录；默认 <project>/论文核验输出_奖励Q数值核验')
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--selftest',action='store_true');g.add_argument('--check',action='store_true')
    g.add_argument('--audit',action='store_true');g.add_argument('--probe',action='store_true')
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--interval',type=int,default=500);ap.add_argument('--ceiling',type=float,default=1e7)
    args=ap.parse_args();p=args.project.resolve()
    out=args.output.resolve() if args.output else (p/'论文核验输出_奖励Q数值核验').resolve()
    if '每日结果' in out.parts:raise ValueError('严禁在 每日结果 历史目录输出')
    if args.selftest:selftest();return
    if args.check:check(p);return
    if args.audit:audit(p,out);return
    probe(p,out,args.seed,args.steps,args.interval,args.ceiling)
if __name__=='__main__':main()
