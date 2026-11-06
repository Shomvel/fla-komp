import unittest
import torch
from compare import row

class ComparisonTests(unittest.TestCase):
 def test_equal_and_nearzero(self):
  self.assertTrue(row('x',torch.ones(3),torch.ones(3))['passed'])
  self.assertTrue(row('x',torch.full((3,),1e-8),torch.zeros(3))['passed'])
 def test_real_failure_retained(self):
  self.assertFalse(row('x',torch.tensor([1.,1.2]),torch.ones(2))['passed'])
  self.assertFalse(row('x',torch.full((2,),float('nan')),torch.ones(2))['passed'])
  self.assertFalse(row('x',torch.full((2,),float('inf')),torch.ones(2))['passed'])
if __name__=='__main__':unittest.main()
