"""FP64 research only: causal local-RKHS KAPA writes + Block-KOMP compression.

No production kernel is changed. Single batch/head, FIFO, normalized Gaussian.
Write normalization (1+lambda) makes p=0 exactly the original Block-KOMP rule.
Anchor history p<=M is in the FIFO carried dictionary; learned eviction needs an
explicit raw-anchor cache or reserved anchors and is NOT implemented here.
"""
import torch
import torch.nn.functional as F


def kernel(x,y,tau=.5):
    return torch.exp((x@y.T-1)/tau)


def compress(u,n,m,p,ridge,protect,eta):
    kept=u[n:n+m];eye=torch.eye(m,dtype=u.dtype,device=u.device)
    metric=kernel(kept,kept)+ridge*eye
    base=torch.cat((torch.linalg.solve(metric,kernel(kept,u[:n])),eye),1)
    if protect and p:
        # p most recent raw keys exist in FIFO kept dictionary (p<=M).
        anchor=u[-p:];a=kernel(anchor,kept);target=kernel(anchor,u)
        z=torch.linalg.solve(metric,a.T)
        base=base+z@torch.linalg.solve(a@z+eta*torch.eye(p,dtype=u.dtype,device=u.device),target-a@base)
    return base


def geometry(keys,new,q,history,*,m,start,g,beta,p,lam,ridge,protect_write,protect_compress,eta):
    n=len(new);u=torch.cat((keys,new));N=len(u)
    columns=[]
    for i in range(n):
        if protect_write and p:
            # Global history contains actual raw token keys, including previous chunks.
            count=min(p,start+i)
            akeys=torch.cat((new[i:i+1],history[start+i-count:start+i].flip(0)))
            gram=kernel(akeys,akeys)
            rhs=torch.zeros(count+1,dtype=u.dtype,device=u.device);rhs[0]=1+lam
            coeff=torch.linalg.solve(gram+lam*torch.eye(count+1,dtype=u.dtype,device=u.device),rhs)
            # In FIFO, preceding raw keys are immediately before current in U.
            index=torch.arange(m+i,m+i-count-1,-1,device=u.device)
            e=F.one_hot(index,N).to(u.dtype)
            columns.append(e.T@coeff)
        else:
            columns.append(F.one_hot(torch.tensor(m+i,device=u.device),N).to(u.dtype))
    w=torch.stack(columns,1)
    h=kernel(new,u);r=kernel(q,u)
    if g.ndim==1:a=g.exp()[:,None].expand(n,N)
    else:
        # Atom chronology maps to physical ring slots; inherited from baseline.
        slots=torch.arange(start-m,start+n,device=u.device)%m
        a=g.exp()[:,slots]
    # Initial dictionary atoms are not real history: protect only seen tokens.
    j=compress(u,n,m,min(p,start+n),ridge,protect_compress,eta)
    return u,w,h,r,a,j


def forward(q,k,v,g,beta,keys0,state0,*,chunk=4,p=2,lam=.05,ridge=1e-4,
            protect_write=True,protect_compress=False,eta=.05,mode='token'):
    m=len(keys0);assert 0<=p<=m and chunk>0
    assert lam>0 and ridge>0 and eta>0
    q,k,keys=map(lambda x:F.normalize(x,dim=-1,eps=1e-6),(q,k,keys0))
    state=state0;outs=[];maps=[]
    for start in range(0,len(q),chunk):
        n=min(chunk,len(q)-start)
        u,w,h,r,a,j=geometry(keys,k[start:start+n],q[start:start+n],k,m=m,start=start,
            g=g[start:start+n],beta=beta[start:start+n],p=p,lam=lam,ridge=ridge,
            protect_write=protect_write,protect_compress=protect_compress,eta=eta)
        bv=beta[start:start+n];vv=v[start:start+n];N=len(u)
        if mode=='token':
            s=torch.cat((state,torch.zeros(n,v.shape[-1],dtype=v.dtype,device=v.device)))
            for i in range(n):
                pre=a[i,:,None]*s;error=vv[i]-h[i]@pre
                s=pre+bv[i]*w[:,i,None]*error[None,:]
                outs.append(r[i]@s)
            state=j@s
        else:
            # U_i = beta_i(v_i-h_i D_i state_{i-1}); W_i is causal/key-only.
            prefix=torch.cumprod(a,dim=0)
            lower=[];response=[]
            for i in range(n):
                ls=[];rs=[]
                for t in range(n):
                    if t<i:
                        window=a[t+1:i+1].prod(0)
                        ls.append(-bv[i]*(h[i]*window)@w[:,t])
                    else:ls.append(bv.new_zeros(()))
                    if t<=i:
                        window=a[t+1:i+1].prod(0)
                        rs.append((r[i]*window)@w[:,t])
                    else:rs.append(bv.new_zeros(()))
                lower.append(torch.stack(ls));response.append(torch.stack(rs))
            lower,response=torch.stack(lower),torch.stack(response)
            inv=torch.linalg.solve_triangular(torch.eye(n,dtype=v.dtype,device=v.device)-lower,
                torch.eye(n,dtype=v.dtype,device=v.device),upper=False,unitriangular=True)
            us=inv@(-bv[:,None]*h[:,:m]*prefix[:,:m]);uv=inv@torch.diag(bv)
            endw=torch.stack([a[t+1:].prod(0)*w[:,t] for t in range(n)],1)
            initial_map=torch.cat((torch.diag(prefix[-1,:m]),torch.zeros(n,m,dtype=v.dtype,device=v.device)))
            os=prefix[:,:m]*r[:,:m]+response@us;ov=response@uv
            fs=j@(initial_map+endw@us);fv=j@endw@uv
            outs.extend(os@state+ov@vv);state=fs@state+fv@vv
            maps.append((os,ov,fs,fv))
        keys=u[n:n+m]
    return torch.stack(outs),keys,state
