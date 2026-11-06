"""Bitwise comparison of saved outputs/gradients between two code versions (fifo only; learned uses atomics)."""
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
print(sys.argv[1], sys.argv[2], 'bitwise', all(torch.equal(x, y) for x, y in zip(a, b)),
      'max abs diff', max(float((x.double()-y.double()).abs().max()) for x, y in zip(a, b)))
