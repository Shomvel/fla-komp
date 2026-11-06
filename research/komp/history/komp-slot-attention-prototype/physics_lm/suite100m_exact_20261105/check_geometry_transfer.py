"""Fixed-checkpoint FP64/FP32 geometry comparison; no optimizer or training steps."""
import argparse,datetime,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parent
TUNING=ROOT/'exact_tuning_20261105'
if not TUNING.exists():TUNING=ROOT.parent/'exact_tuning_20261105'
sys.path.insert(0,str(TUNING))
import tune as t

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def projection(candidate,dtype):
    def operator(*args,**kwargs):
        kwargs.update(temperature=candidate['temperature'],ridge=candidate['ridge'],projection_dtype=dtype)
        return t.ORIGINAL_EXACT(*args,**kwargs)
    t.models.exact_komp=operator

def main(a):
    rank=dist.get_rank();world=dist.get_world_size();assert world==8
    plan=json.loads((TUNING/'initial-screen.json').read_text());summaries={}
    for candidate in plan['screen']:
        path=TUNING/'results'/a.task/candidate['name']/'summary.json'
        if not path.exists() and a.selection_scope=='current_completed':continue
        item=json.loads(path.read_text());assert item['status']=='completed'
        summaries[candidate['name']]=item
    assert summaries,'No completed checkpoint for comparison'
    best=min(summaries,key=lambda k:summaries[k]['val']['ce']);assert best==a.candidate,(best,a.candidate)
    dest=TUNING/'results'/a.task/best;checkpoint=dest/'checkpoint.pt'
    if rank==0:assert sha(checkpoint)==a.checkpoint_sha256,'Checkpoint changed since selection'
    dist.barrier()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=saved['config'];assert config['sources']==t.sources() and config['world_size']==world
    candidate=config['candidate'];assert candidate['name']==best
    model=t.configure(candidate);model.load_state_dict(saved['model'],strict=True);del saved
    weights_before=torch.stack([p.detach().double().sum() for p in model.parameters()])
    corpus=t.d.base.Corpus(a.task,'val','tune')
    rows=[corpus.row(i) for i in range(rank,16,world)]
    ids,mask=t.d.base.batch(rows)
    count=mask[:,1:].sum().to(torch.float64);dist.all_reduce(count)
    names=('embed.weight','head.weight','layers.0.mix.keys','layers.14.mix.keys','layers.0.mix.k.weight','layers.0.mix.decay.weight','layers.0.mix.decay.bias','layers.0.mix.beta.weight','layers.0.mix.out.weight')
    reference={};result={}
    for dtype in ('float64','float32'):
        projection(candidate,dtype)
        torch.cuda.synchronize();started=time.perf_counter()
        val=t.evaluate(model,corpus)
        torch.cuda.synchronize();elapsed=torch.tensor(time.perf_counter()-started,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        model.zero_grad(set_to_none=True)
        with t.d.amp('bf16'):
            loss,n=t.d.base.loss_sum(model,ids,mask)
            normalized=loss/count
        normalized.backward()
        with torch.no_grad(),t.d.amp('bf16'):
            hidden=model(ids)[:,:-1][mask[:,1:]].float()
        if dtype=='float64':
            reference_hidden=hidden
            hidden_error=None
        else:
            delta=hidden-reference_hidden
            sums=torch.stack([delta.double().square().sum(),reference_hidden.double().square().sum()]);dist.all_reduce(sums)
            maxima=torch.stack([delta.abs().max(),reference_hidden.abs().max()]);dist.all_reduce(maxima,op=dist.ReduceOp.MAX)
            hidden_error={'finite':bool(torch.isfinite(sums).all()),'rms_relative':float((sums[0]/sums[1].clamp_min(1e-16)).sqrt()),'normalized_max':float(maxima[0]/maxima[1].clamp_min(1e-8))}
        total=loss.detach().double();dist.all_reduce(total)
        gradients={}
        for name,p in model.named_parameters():
            if name not in names:continue
            assert p.grad is not None,name
            grad=p.grad.detach().clone();dist.all_reduce(grad)
            if dtype=='float64':
                reference[name]=grad
                gradients[name]={'finite':bool(torch.isfinite(grad).all())}
            else:
                ref=reference[name];delta=grad-ref
                finite=bool(torch.isfinite(grad).all() and torch.isfinite(ref).all())
                gradients[name]={'finite':finite,'rms_relative':(delta.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8)).item() if finite else None,'normalized_max':(delta.abs().max()/ref.abs().max().clamp_min(1e-8)).item() if finite else None}
        result[dtype]={'validation':val,'validation_seconds_single_pass_including_compile':elapsed.item(),'validation_finite':math.isfinite(val['ce']),'fixed_batch_loss':float(total/count),'fixed_batch_supervised':int(count),'gradient_comparison':gradients,'supervised_hidden_comparison':hidden_error}
        if rank==0:print('GEOMETRY_TRANSFER',dtype,json.dumps(result[dtype]),flush=True)
    weights_after=torch.stack([p.detach().double().sum() for p in model.parameters()])
    assert torch.equal(weights_before,weights_after),'Evaluation changed model weights'
    fp64=result['float64']['validation']['ce'];fp32=result['float32']['validation']['ce']
    assert abs(fp64-summaries[best]['val']['ce'])<.005,'FP64 checkpoint validation failed to reproduce screen'
    report={'status':'completed','task':a.task,'selection_scope':a.selection_scope,'completed_candidates_at_selection':len(summaries),'candidate':candidate,'checkpoint':str(checkpoint),'checkpoint_sha256':a.checkpoint_sha256,'sources':t.sources(),'script_sha256':sha(Path(__file__)),'world_size':world,'training_steps':0,'training_tokens':0,'precision_paths':result,'validation_ce_delta_fp32_minus_fp64':fp32-fp64,'validation_ce_abs_delta':abs(fp32-fp64),'original_screen_ce':summaries[best]['val']['ce'],'weights_unchanged':True,'scope':'Fixed trained checkpoint inference and gradient comparison only; this does not establish equivalent FP32 training trajectories.'}
    if rank==0:t.d.base.atomic_json(ROOT/'evidence'/f'geometry-transfer-{a.task}.json',report)
    dist.barrier()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['depo2','brevo2']);p.add_argument('--candidate',required=True);p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--selection-scope',choices=['all_completed','current_completed'],default='all_completed');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=10),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    try:main(a)
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    else:dist.destroy_process_group()
