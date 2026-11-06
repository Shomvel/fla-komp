"""Read-only Northjob project/queue check. Credentials stay remote."""
import json,pathlib,urllib.request,urllib.parse
origin='http://10.1.2.7:31000';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
creds=json.loads((pathlib.Path.home()/'.config/northjob/userinfo.conf').read_text())
def req(path,params=None,body=None,token=None):
 if params:path+='?'+urllib.parse.urlencode(params)
 headers={'Content-Type':'application/json'}
 if token:headers['Authorization']='Bearer '+token
 r=urllib.request.Request(origin+path,data=json.dumps(body).encode() if body else None,headers=headers)
 with opener.open(r,timeout=30) as f:return json.load(f)
token=req('/api/auth/get_token',body={'username':creds['username'],'password':creds['password'],'expireTime':3600})['data']
project=req('/api/v1/projectdetail',{'projectname':'WeaverRSI'},token=token)
ps=project['data']['project_list'];assert any(p['name']=='WeaverRSI' and p['id']==32 for p in ps)
jobs=req('/api/get_jobs',token=token)['data'];selected=[]
for j in jobs:
 if j.get('user_def_name') not in ('komp340m-exact-fla-mb16-20261106','komp340m-block-fla-mb16-20261106'):continue
 data=req('/api/task/get_task_overview',{'uuid':j['name'],'kind':'Job','namespace':'qiji'},token=token)['data']
 base=data['base_info'];selected.append(dict(name=j['name'],alias=j['user_def_name'],project=base.get('project'),phase=data.get('phase')))
queues=req('/api/k8s/queues',token=token)
# Queue response may be empty. Confirm live job queue through the jobs API if present.
queue_values=[]
def collect(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if 'queue' in k.lower() and isinstance(v,(str,int)):queue_values.append({k:v})
   else:collect(v)
 elif isinstance(x,list):
  for v in x:collect(v)
for j in jobs:
 if j.get('user_def_name')=='komp340m-exact-fla-mb16-20261106':collect(j)
report=dict(project=[dict(id=p['id'],name=p['name']) for p in ps],formal_jobs=selected,live_job_queue_fields=queue_values,queue_api=queues)
text=json.dumps(report)
for secret in (token,creds['password']):text=text.replace(secret,'[REDACTED]')
print(text)
