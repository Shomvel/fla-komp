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
import sys,time
sys.path.insert(0,'/gpfs/users/linlizhi/physics100m-exact-20261105/evidence/diagnostic-python')
import websocket
pod='job-880a5c43fba6-20261105221543-runner-0'
url='ws://10.1.2.7:31000/cloudshell?'+urllib.parse.urlencode({'podname':pod,'username':creds['username'],'containername':'runner'})
ws=websocket.create_connection(url,timeout=10,http_no_proxy=['10.1.2.7'],header=['Authorization: Bearer '+token])
ws.send(json.dumps({'msg_type':'resize','rows':40,'cols':180}))
ws.send(json.dumps({'msg_type':'input','data':"for n in 1 2 3; do echo SAMPLE=$n; ps -p 1 -o pid,pcpu,time,stat; ps -eo pid,ppid,pcpu,etimes,time,comm,args | awk '$2==1'; ls -l /proc/1/fd 2>/dev/null | grep -E '\\.ptx|tmp.*log'; sleep 5; done; echo DIAGNOSTIC_DONE\n"}))
start=time.time();chunks=[]
while time.time()-start<25:
 try:
  data=ws.recv();chunks.append(data)
  if '\r\nDIAGNOSTIC_DONE\r\n' in ''.join(chunks):break
 except websocket.WebSocketTimeoutException:break
ws.close()
out=''.join(chunks).replace(token,'[REDACTED]').replace(creds['password'],'[REDACTED]')
print(out)
