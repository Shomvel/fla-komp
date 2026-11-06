"""Compare unchanged frozen Block checkpoint on exact and right-padded prefixes."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot/suite100m'))
from models import LM
RUN='brevo2-komp-final-lr0.001-wd0.1-seed731'
folder=ROOT/'runs-ddp/results'/RUN
saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
model=LM('komp',731).cuda().eval();model.load_state_dict(saved['model']);del saved
records=json.loads((folder/'eval-progress.json').read_text())['records'][:24]
assert len(records)==24
assert all(hashlib.sha256((ROOT/'snapshot'/name).read_bytes()).hexdigest()==wanted for name,wanted in json.loads((ROOT/'evidence/snapshot-sha256.json').read_text()).items())
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
report={'status':'running','run':RUN,'checkpoint_stat':{'size':(folder/'checkpoint.pt').stat().st_size,'mtime_ns':(folder/'checkpoint.pt').stat().st_mtime_ns},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'unchanged frozen BF16 Block model; right pad only to128boundary; fixed recorded generated prefixes, no new generation samples or optimizer updates','cases':[]}
dest=ROOT/'evidence/block-eval-padding-diagnostic.json'
for start in (0,8,16):
 for step in (0,1,31,63,127,200):
    group=records[start:start+8]
    seqs=[r['probe']['prompt']+r['guess'][:step] for r in group]
    length=max(map(len,seqs));padded=math.ceil(length/128)*128
    ids=torch.zeros(8,length,dtype=torch.long,device='cuda')
    for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.tensor(seq,device='cuda')
    pad=torch.nn.functional.pad(ids,(0,padded-length))
    positions=torch.tensor([len(seq)-1 for seq in seqs],device='cuda')
    valid=torch.arange(length,device='cuda')[None]<torch.tensor([len(seq) for seq in seqs],device='cuda')[:,None]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=model(ids);b=model(pad)[:,:length]
        la=model.head(a[torch.arange(8,device='cuda'),positions]);lb=model.head(b[torch.arange(8,device='cuda'),positions])
    a,b=a[valid].float(),b[valid].float();la,lb=la.float(),lb.float()
    result={'batch_start':start,'generated_step':step,'length':length,'padded_length':padded,'hidden_max_abs':(a-b).abs().max().item(),'hidden_rms_relative':((a-b).square().mean().sqrt()/a.square().mean().sqrt()).item(),'last_logits_max_abs':(la-lb).abs().max().item(),'greedy_equal':bool(torch.equal(la.argmax(-1),lb.argmax(-1))),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
    report['cases'].append(result);print('PADDING_CASE',json.dumps(result),flush=True);dest.write_text(json.dumps(report,indent=2)+'\n')
report['status']='completed';report['all_bitwise_equal']=all(x['hidden_max_abs']==x['last_logits_max_abs']==0 for x in report['cases']);report['all_greedy_equal']=all(x['greedy_equal'] for x in report['cases'])
dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
