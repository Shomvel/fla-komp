import json,torch
from models import ROOT,KOMPConfig,KOMPBlock,LM,legacy,map_weights,AutoConfig,AutoModelForCausalLM,KOMPForCausalLM

torch.set_num_threads(2)
config=KOMPConfig();assert isinstance(AutoConfig.for_model('komp'),KOMPConfig)
assert AutoModelForCausalLM._model_mapping[KOMPConfig] is KOMPForCausalLM
old=legacy.LM('exact');new=LM('exact');new.load_state_dict(map_weights(old.state_dict()),strict=True)
assert sum(p.numel() for p in new.parameters())==sum(p.numel() for p in old.parameters())==339966848
assert not new.model.gradient_checkpointing
assert all(type(x).__name__=='KOMPBlock' for x in new.model.layers)
(ROOT/'evidence/cpu-contract.json').write_text(json.dumps(dict(status='passed',parameters=339966848,auto_registration=True,strict_state_mapping=True,native_block=True),indent=2))
print('CPU_CONTRACT_PASS',flush=True)
