"""New-source forward-layout and real BF16 DDP witnesses."""
import torch


def forward_layout(d, model):
    rows=[d.base.Corpus('depo2','train','tune').row(i) for i in range(17)]
    ids,_=d.base.batch(rows)
    valid=torch.arange(ids.shape[1])[None]<torch.tensor([len(r['input_ids']) for r in rows])[:,None]
    def forward(indices,training):
        model.train(training);outputs=[]
        hooks=[layer.register_forward_hook(lambda m,i,o:outputs.append(o.detach().float().cpu())) for layer in model.layers]
        local,_=d.base.batch([rows[i] for i in indices])
        with torch.no_grad(),d.amp('bf16'):outputs.append(model(local).detach().float().cpu())
        for hook in hooks:hook.remove()
        return outputs
    baseline=forward(list(range(17)),True)
    evaluation=forward(list(range(17)),False)
    combined=[torch.zeros_like(x) for x in baseline]
    for rank in range(8):
        indices=list(range(rank,17,8))
        for target,value in zip(combined,forward(indices,True)):target[indices,:value.shape[1]]=value
    checks={}
    for name,outputs in [('whole_vs_rank_partition',combined),('train_vs_eval',evaluation)]:
        values=[]
        for index,(reference,actual) in enumerate(zip(baseline,outputs)):
            a,b=reference[valid],actual[valid]
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            error=(a-b).abs().max().item()
            assert error==0,(name,index,error)
            values.append({'layer':index if index<15 else 'final_norm','max_abs':error})
        checks[name]=values
    model.train()
    return {'shape':list(ids.shape),'documents':17,'precision':'BF16','checks':checks,'scope':'Seed731 real Depo 17-document initial-model witness; not all checkpoints or arbitrary lengths'}


def real_reference(d, model, rows, selected):
    reference=d.LM('gsa2',731).cuda();reference.load_state_dict(model.state_dict())
    optimizer=d.optimizer(reference,1e-4,.01)
    nvalid=sum(int(d.np.sum(r['loss_mask'][1:])) for r in rows)
    padded=d.math.ceil(max(len(r['input_ids']) for r in rows)/16)*16
    micro=min(d.math.ceil(len(rows)/8),2**int(d.math.log2(max(1,8192/padded))))
    value=0.
    for rank in range(8):
        local=rows[rank::8]
        for start in range(0,len(local),micro):
            ids,mask=d.base.batch(local[start:start+micro])
            with d.amp('bf16'):loss,_=d.base.loss_sum(reference,ids,mask);normalized=loss/nvalid
            normalized.backward();value+=normalized.item()
    gradients={n:p.grad.detach().clone() for n,p in reference.named_parameters() if n in selected}
    torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
    # The live model has two Adam steps. Match its existing optimizer in the
    # caller rather than using a fresh Adam state for the update comparison.
    return reference,optimizer,gradients,value
