"""CPU filesystem regression of the actual cross-revision resume function."""
import ast,hashlib,json,pickle,tempfile,types
from pathlib import Path
ROOT_REAL=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT_REAL/'train.py').read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resume')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);old=root/'old';new=root/'new';folder=old/'checkpoint';folder.mkdir(parents=True);(new/'configs').mkdir(parents=True);out=new/'runs';out.mkdir()
 manifest={'files':{'models.py':'same','data.py':'same'}}
 for d in [old,new]:(d/'source-manifest.json').write_text(json.dumps(manifest))
 source=sha(old/'source-manifest.json');new_source='new-source';data='same-data'
 model_state={'weights':42};opt_state={'moments':43};rank_state={'stream':{'rank':0},'rng':{'seed':42}}
 checkpoint={'model':model_state,'optimizer':opt_state,'step':1068,'tokens':1068*1048576,'source_sha':source,'data_sha':data}
 (folder/'model.pt').write_bytes(pickle.dumps(checkpoint));(folder/'rank0.pt').write_bytes(pickle.dumps(rank_state));(folder/'complete.json').write_text(json.dumps({'world_size':32}));(out/'latest.json').write_text(json.dumps({'path':str(folder)}))
 transition={'parent_root':str(old),'checkpoint_path':str(folder),'source_sha':source,'data_sha':data,'step':1068,'tokens':1068*1048576,'world_size':32,'unchanged_files':['models.py','data.py'],'checkpoint_sha256':sha(folder/'model.pt'),'rank_sha256':{'0':sha(folder/'rank0.pt')}}
 config=new/'configs/resume-parent.json';config.write_text(json.dumps(transition))
 received={};receiver=lambda key:types.SimpleNamespace(load_state_dict=lambda state:received.__setitem__(key,state))
 env={'ROOT':new,'Path':Path,'json':json,'hashlib':hashlib,'WORLD':32,'GLOBAL':1048576,'torch':types.SimpleNamespace(load=lambda path,**kwargs:pickle.loads(Path(path).read_bytes())),'restore_rng':lambda state:received.__setitem__('rng',state)}
 exec(compile(ast.Module(body=[function],type_ignores=[]),'actual-resume','exec'),env)
 def call():return env['resume'](out,receiver('model'),receiver('opt'),receiver('stream'),0,new_source,data)
 assert call()==(1068,1068*1048576)
 assert received=={'model':model_state,'opt':opt_state,'stream':rank_state['stream'],'rng':rank_state['rng']}
 rejected=[]
 for key,bad in [('checkpoint_path',str(old/'wrong')),('source_sha','wrong'),('data_sha','wrong'),('step',1069),('tokens',0),('world_size',8),('checkpoint_sha256','wrong'),('rank_sha256',{'0':'wrong'})]:
  mutated=dict(transition);mutated[key]=bad;config.write_text(json.dumps(mutated))
  try:call()
  except AssertionError:rejected.append(key)
  else:raise AssertionError('Invalid migration accepted: '+key)
 config.write_text(json.dumps(transition));(new/'source-manifest.json').write_text(json.dumps({'files':{'models.py':'changed','data.py':'same'}}))
 try:call()
 except AssertionError:rejected.append('model-source-change')
 else:raise AssertionError('Changed model source accepted')
 result={'status':'passed','actual_resume_function':True,'exact_state_restored':['model','optimizer','rank_stream','rank_rng'],'rejected':rejected}
(ROOT_REAL/'evidence/resume-contract-regression.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
