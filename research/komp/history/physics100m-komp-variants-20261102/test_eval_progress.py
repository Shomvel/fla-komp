import ast,json,tempfile,types,unittest
from contextlib import nullcontext,redirect_stdout
from io import StringIO
from pathlib import Path
from eval_progress import load_progress,save_progress

class Factory:
    def __init__(self,task):pass
    def grade(self,p,g):return g==p['answer']
class Tensor:
    def __init__(self,n):self.n=n
    def __setitem__(self,key,value):pass
    def __getitem__(self,key):return self
    def argmax(self,*args):return self
    def tolist(self):return [5001]*self.n
class Model:
    def eval(self):pass
    def __call__(self,ids):return ids
    def head(self,h):return h

def generator():
    tree=ast.parse(Path('evaluate.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    fn.decorator_list=[]
    torch=types.SimpleNamespace(zeros=lambda n,t,**kw:Tensor(n),tensor=lambda x,**kw:x,arange=lambda n,**kw:list(range(n)),autocast=lambda *args,**kw:nullcontext(),long=None,bfloat16=None)
    env=dict(torch=torch,Factory=Factory,EOS=9998)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'generate','exec'),env)
    return env['generate']

class ResumeTest(unittest.TestCase):
    def test_interrupted_generation_matches_uninterrupted(self):
        probes=[dict(prompt=[9999,i],answer=[5001]) for i in range(24)]
        gen=generator();grade=Factory('mano').grade
        with tempfile.TemporaryDirectory() as td,redirect_stdout(StringIO()):
            path=Path(td)/'progress.json';identity={'run':'test'}
            expected=gen(Model(),'mano',probes)
            def interrupt(records):
                save_progress(path,identity,{'ce':.5},records,3)
                raise InterruptedError()
            with self.assertRaises(InterruptedError):gen(Model(),'mano',probes,on_batch=interrupt)
            saved=load_progress(path,identity,probes,grade)
            self.assertEqual(len(saved['records']),8)
            (Path(td)/'progress.json.tmp').write_text('{broken')
            self.assertEqual(load_progress(path,identity,probes,grade),saved)
            batches=[]
            actual=gen(Model(),'mano',probes,saved['records'],lambda rs:batches.append(len(rs)))
            self.assertEqual(actual,expected);self.assertEqual(batches,[16,24])
            with self.assertRaises(AssertionError):load_progress(path,{'run':'wrong'},probes,grade)
            with self.assertRaises(AssertionError):load_progress(path,identity,list(reversed(probes)),grade)
            bad=json.loads(path.read_text());bad['records'][0]['valid']=False;path.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):load_progress(path,identity,probes,grade)

if __name__=='__main__':unittest.main()
