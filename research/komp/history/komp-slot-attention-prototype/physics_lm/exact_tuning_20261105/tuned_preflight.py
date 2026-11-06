"""Run the accepted full-model DDP checks under the tuning geometry precision."""
import argparse
import datetime
import faulthandler
import os
import sys
import traceback
import tune
import torch
import torch.distributed as dist

p=argparse.ArgumentParser();p.add_argument('--task',required=True);args=p.parse_args()
d=tune.d
source=tune.sources()
candidate=tune.configuration('baseline')
# Isolate all tickets/checkpoints from the frozen seven-model preflight.
d.ROOT=tune.HERE/'preflight'/args.task
d.digest=lambda: source
d.LM=lambda kind,seed: tune.configure(candidate)
original_metadata=d.metadata
d.metadata=lambda model: {**original_metadata(model),'projection_dtype':candidate['projection_dtype'],'candidate':candidate}
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
os.environ['DDP_PREFLIGHT']='1'
timeout=datetime.timedelta(minutes=10)
dist.init_process_group('nccl',timeout=timeout,device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
d.CONTROL=dist.new_group(backend='gloo',timeout=timeout)
assert dist.get_world_size()==8
faulthandler.dump_traceback_later(90,repeat=True)
try:
 d.preflight('exact')
except BaseException:
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:
 faulthandler.cancel_dump_traceback_later();dist.destroy_process_group()
