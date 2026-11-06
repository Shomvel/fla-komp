"""Execute the actual Triton kernels under TRITON_INTERPRET=1, on CPU tensors.

This checks arithmetic and masks, not CUDA execution, races or performance.
"""

import os
os.environ['TRITON_INTERPRET'] = '1'

import json
from pathlib import Path

import torch
import triton

from komp_slot.kernels import slot_bwd_kernel, slot_fwd_kernel
from komp_slot.reference import recurrent_reference
from validate import REPORT, check, core_inputs


def main():
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    cases = [(1,1,1,1,1), (2,7,3,5,37), (1,33,2,16,64), (1,65,1,32,17), (1,17,1,64,65)]
    for number, shape in enumerate(cases):
        batch,t,heads,m,width=shape
        xs,slots=core_inputs(shape,'cpu',number)
        if number == 0:
            xs[4].zero_()
            xs[5].fill_(1)
        if number == 1:
            xs[4][:,0].zero_()
            xs[4][:,-1].fill_(1)
            xs[5][:,0].zero_()
            slots.fill_(m-1)
        h,p,r,v,a,b,initial=xs
        bm,bv=triton.next_power_of_2(m),32
        nv=triton.cdiv(width,bv)
        output=torch.empty_like(v)
        final=torch.empty_like(initial)
        tape=v.new_empty(batch,t+1,heads,m,width)
        slot_fwd_kernel[(batch*heads*nv,)](
            h,p,r,v,a,b,slots,initial,output,final,tape,t,heads,m,width,bm,bv,True,
        )
        refs=[x.double().requires_grad_() for x in xs]
        ro,rf=recurrent_reference(*refs[:6],slots,refs[6])
        check(f'interpreter{number}/output',output,ro)
        check(f'interpreter{number}/final',final,rf)
        go,gf=torch.randn_like(output),torch.randn_like(final)
        dh,dp,dr=[h.new_empty(nv,*h.shape) for _ in range(3)]
        da,db=[a.new_empty(nv,*a.shape) for _ in range(2)]
        dv,dinitial=torch.empty_like(v),torch.empty_like(initial)
        slot_bwd_kernel[(batch*heads*nv,)](
            h,p,r,v,a,b,slots,tape,go,gf,dh,dp,dr,dv,da,db,dinitial,
            t,batch,heads,m,width,bm,bv,
        )
        expected=torch.autograd.grad((ro,rf),refs,(go.double(),gf.double()))
        actual=[dh.sum(0),dp.sum(0),dr.sum(0),dv,da.sum(0),db.sum(0),dinitial]
        for name,got,ref in zip(('h','p','read','value','alpha','beta','initial'),actual,expected):
            check(f'interpreter{number}/grad_{name}',got,ref,6e-5,6e-4)
        # no training tape is allocated in inference
        out2,fin2=torch.empty_like(v),torch.empty_like(initial)
        slot_fwd_kernel[(batch*heads*nv,)](
            h,p,r,v,a,b,slots,initial,out2,fin2,v.new_empty(1),t,heads,m,width,bm,bv,False,
        )
        check(f'interpreter{number}/inference',out2,output,0,0)
        check(f'interpreter{number}/inference_final',fin2,final,0,0)
        print('PASS interpreted Triton',shape,flush=True)
    REPORT.update(status='passed',backend='Triton CPU interpreter',triton=triton.__version__,
                  limitation='No CUDA execution, allocator/race validation, or timing evidence')
    destination=Path(__file__).parent/'evidence'/'triton-interpreter.json'
    destination.write_text(json.dumps(REPORT,indent=2)+'\n')
    print('PASS',len(REPORT['checks']),'checks',flush=True)


if __name__=='__main__':
    main()
