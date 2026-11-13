import json,pathlib,urllib.request,urllib.parse,datetime
origin='http://10.1.2.7:31000';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
creds=json.loads((pathlib.Path.home()/'.config/northjob/userinfo.conf').read_text())
def req(path,params=None,body=None,token=None):
 if params:path+='?'+urllib.parse.urlencode(params)
 headers={'Content-Type':'application/json'}
 if token:headers['Authorization']='Bearer '+token
 r=urllib.request.Request(origin+path,data=json.dumps(body).encode() if body else None,headers=headers)
 with opener.open(r,timeout=30) as f:return json.load(f)
token=req('/api/auth/get_token',body={'username':creds['username'],'password':creds['password'],'expireTime':3600})['data']
report={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':{}}
jobs=req('/api/get_jobs',token=token)['data']
for identity in ['job-b975c4a6b8da-20261108013315','job-ab970a5f32c2-20261108013328']:
 report.setdefault('raw_job_rows',[]).extend([j for j in jobs if j.get('name')==identity])
 r={}
 for label,path,params in [('overview','/api/task/get_task_overview',{'uuid':identity,'kind':'Job','namespace':'qiji'}),('events','/api/task/get_task_events',{'involvedObject':identity,'namespace':'qiji','type':''}),('scheduler','/api/k8s/scheduler/schedulerlogs',{'jobname':identity,'namespace':'qiji','type':''})]:
  try:r[label]=req(path,params,token=token)
  except Exception as e:r[label]={'error':type(e).__name__}
 report['jobs'][identity]=r
report['project']=req('/api/v1/projectdetail',{'projectname':'WeaverRSI'},token=token)
def clean(x):
 if isinstance(x,dict):return {k:('[REDACTED]' if any(a in k.lower() for a in ['secret','token','password','authorization']) else clean(v)) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,str):return x.replace(token,'[REDACTED]').replace(creds['password'],'[REDACTED]')
 return x
if isinstance(report.get('queues'),list):report['queues']=[q for q in report['queues'] if q.get('name')=='proj-32']
print(json.dumps(clean(report)))
