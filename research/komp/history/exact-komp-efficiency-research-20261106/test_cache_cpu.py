"""Exercise actual PyTorch checkpoint/autograd lifecycle without CUDA or Triton."""
import gc, json, sys, unittest, weakref
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
sys.path.insert(0,str(Path(__file__).resolve().parent/'candidate'))
from geometry_cache import cached_geometry, geometry_contexts

class CacheTests(unittest.TestCase):
    def test_custom_autograd_replay_and_lifetime(self):
        builds=[]; refs=[]
        class ExactSurrogate(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x):
                def build():
                    builds.append(1); z=x*x;refs.append(weakref.ref(z));return (z,)
                (z,)=cached_geometry((x.shape,x.dtype,x.device),build)
                ctx.save_for_backward(x,z)
                return x*z
            @staticmethod
            def backward(ctx,dy):
                x,z=ctx.saved_tensors
                return dy*3*z
        x=torch.randn(9,dtype=torch.float64,requires_grad=True)
        def mixer(x):return ExactSurrogate.apply(x).sin()
        a=checkpoint(mixer,x,use_reentrant=False,context_fn=geometry_contexts)
        b=checkpoint(mixer,x+1,use_reentrant=False,context_fn=geometry_contexts)
        loss=(a+b).sum()
        expect=3*x.detach().square()*x.detach().pow(3).cos()+3*(x.detach()+1).square()*(x.detach()+1).pow(3).cos()
        loss.backward(retain_graph=True);torch.testing.assert_close(x.grad,expect,rtol=1e-14,atol=1e-14)
        self.assertEqual(len(builds),2)
        x.grad=None;loss.backward();torch.testing.assert_close(x.grad,expect,rtol=1e-14,atol=1e-14)
        self.assertEqual(len(builds),2)
        del loss,a,b;gc.collect();self.assertTrue(all(r() is None for r in refs))
    def test_contract_exception_and_scope_restore(self):
        f,r=geometry_contexts()
        with f:self.assertEqual(cached_geometry(('a',),lambda:13),13)
        with self.assertRaisesRegex(RuntimeError,'contract changed'):
            with r:cached_geometry(('b',),lambda:19)
        self.assertEqual(cached_geometry(('a',),lambda:17),17)
        with r:
            self.assertEqual(cached_geometry(('a',),lambda:19),13)
            with self.assertRaisesRegex(RuntimeError,'extra geometry'):
                cached_geometry(('a',),lambda:23)
    def test_nested_scopes(self):
        f,r=geometry_contexts();g,s=geometry_contexts()
        with f:
            cached_geometry(('outer',),lambda:1)
            with g:cached_geometry(('inner',),lambda:2)
        with r:
            self.assertEqual(cached_geometry(('outer',),lambda:0),1)
            with s:self.assertEqual(cached_geometry(('inner',),lambda:0),2)

if __name__=='__main__':unittest.main()
