"""Collect the full existing numerical matrix without changing its tolerances."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;T=ROOT/'exact_tuning_20261105'
if not T.exists():T=ROOT.parent/'exact_tuning_20261105'
sys.path.insert(0,str(T))
import numerical as n

def check(name,actual,expected,low,checks):
 a,b=actual.detach().double(),expected.detach().double();delta=a-b
 rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-6));maximum=float(delta.abs().max())/max(1.,float(b.abs().max()))
 rt,mt=(.025,.03) if low else (.005,.005)
 finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
 checks.append({'name':name,'finite':finite,'rms_relative':rms,'normalized_max':maximum,'rms_tolerance':rt,'max_tolerance':mt,'passed':finite and rms<=rt and maximum<=mt})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 candidate=n.tune.configuration('baseline').copy();candidate['projection_dtype']='float32'
 n.check=check;checks=[];start=time.monotonic()
 for dtype in (torch.float32,torch.bfloat16):
  for pattern in ('random','near','duplicate','strong'):n.run_case(candidate,pattern,dtype,checks)
 report={'status':'completed','verdict':'PASSED' if all(x['passed'] for x in checks) else 'FAILED','task':a.task,'sources':n.tune.sources(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidate':candidate,'seed':917,'shape':{'B':1,'T':65,'H':1,'M':32,'K':64,'V':64},'checks':checks,'passed_checks':sum(x['passed'] for x in checks),'total_checks':len(checks),'seconds':time.monotonic()-start,'scope':'Diagnostic only; inherited numerical tolerances unchanged. The inherited run_case PASS line denotes execution completion; per-metric passed/verdict here is authoritative.'}
 n.tune.d.base.atomic_json(ROOT/'evidence'/f'fp32-projection-diagnostic-{a.task}.json',report)
 print('FP32_PROJECTION_VERDICT',report['verdict'],report['passed_checks'],report['total_checks'],flush=True)
