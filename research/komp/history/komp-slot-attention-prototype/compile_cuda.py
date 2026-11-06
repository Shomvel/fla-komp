"""Compile both Triton kernels to sm80/sm90 cubins without a CUDA device."""

import hashlib
import inspect
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from komp_slot.kernels import slot_bwd_kernel, slot_fwd_kernel


def main():
    destination=Path(__file__).parent/'evidence'/'offline-cuda'
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for arch in (80,90):
        for m,width in ((5,37),(16,64),(64,65)):
            for kernel in (slot_fwd_kernel,slot_bwd_kernel):
                flags=(False,True) if kernel is slot_fwd_kernel else (None,)
                for save in flags:
                    constants=dict(B=2,H=3,M=m,V=width,BM=triton.next_power_of_2(m),BV=32)
                    if save is not None:
                        constants['SAVE']=save
                    constants={k:v for k,v in constants.items() if k in kernel.arg_names}
                    signature={name:('i32' if name=='T' else '*i64' if name=='Slot' else '*fp32')
                               for name in kernel.arg_names if name not in constants}
                    parameter = 'constexprs' if 'constexprs' in inspect.signature(ASTSource).parameters else 'constants'
                    source=ASTSource(fn=kernel,signature=signature,**{parameter:constants})
                    compiled=triton.compile(source,target=GPUTarget('cuda',arch,32),options={'num_warps':4})
                    name=f'{kernel.__name__}-sm{arch}-m{m}-v{width}-save{save}'
                    binary=compiled.asm['cubin']
                    (destination/(name+'.cubin')).write_bytes(binary)
                    rows.append(dict(name=name,bytes=len(binary),sha256=hashlib.sha256(binary).hexdigest()))
                    print('COMPILED',name,len(binary),flush=True)
    report=dict(status='passed',triton=triton.__version__,artifacts=rows,
                limitation='Offline compilation only; kernels were not launched on a GPU')
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
