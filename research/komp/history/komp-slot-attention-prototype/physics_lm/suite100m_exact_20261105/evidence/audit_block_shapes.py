import numpy as np,json,math,hashlib
from pathlib import Path
report={};suite=Path('/gpfs/users/linlizhi/physics100m-exact-20261105')
for task in ['depo2','brevo2']:
 root=Path('/gpfs/users/linlizhi/physics100m-20261101/data')/task
 offsets=np.memmap(root/'train.offsets.u64',dtype='<u8',mode='r');lengths=np.diff(offsets)
 rng=np.random.default_rng(12345);order=rng.permutation(len(lengths));groups=[]
 for start in range(0,len(order),4096):
  pool=order[start:start+4096];pool=pool[np.argsort(lengths[pool],kind='stable')];part=[pool[j:j+128] for j in range(0,len(pool),128)];rng.shuffle(part);groups.extend(part)
 order_sha=hashlib.sha256(b''.join(np.asarray(g,dtype='<i8').tobytes() for g in groups)).hexdigest()
 config=json.loads(next((suite/'runs-ddp/results').glob(task+'-komp-*/config.json')).read_text());assert order_sha==config['data_order_sha256']
 seen=set();steps=[]
 for si,g in enumerate(groups):
  padded=math.ceil(int(max(lengths[g]))/16)*16;maxlocal=math.ceil(len(g)/8);micro=min(maxlocal,2**int(math.log2(max(1,8192/padded))));nsteps=math.ceil(maxlocal/micro);shapes=[]
  for rank in range(8):
   local=g[rank::8]
   for ms in range(nsteps):
    selected=local[ms*micro:(ms+1)*micro]
    if not len(selected):selected=g[:1]
    shapes.append((len(selected),math.ceil(int(max(lengths[selected]))/16)*16))
  new=set(shapes)-seen;seen.update(shapes)
  if si<25:steps.append({'step':si+1,'positions':int(sum(lengths[g])-len(g)),'microsteps':nsteps,'shapes':sorted(set(shapes)),'new_shapes':len(new),'cumulative_unique_shapes':len(seen)})
 report[task]={'data_order_sha256':order_sha,'steps':steps,'total_unique_shapes':len(seen),'all_shapes':sorted(seen)}
print(json.dumps(report,indent=2))
