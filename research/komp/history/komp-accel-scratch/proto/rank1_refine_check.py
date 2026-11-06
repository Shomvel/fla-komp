"""float32 CPU check: standard sweep vs rank-1 sweep vs rank-1 sweep + one Newton step, near-duplicate Grams."""
import numpy as np


def rank1(a):
    a = a.copy()
    for j in range(len(a)):
        col = a[:, j].copy(); p = col[j]; x = col.copy(); x[j] -= 1
        a -= np.outer(x, x/p)
    return (2*np.eye(len(a), dtype=a.dtype)-a).astype(a.dtype)


def sweep(a):
    a = a.copy()
    for j in range(len(a)):
        col = a[:, j].copy(); p = col[j]; c = col/p
        a -= np.outer(col, c); a[j, :] = c; a[:, j] = c; a[j, j] = -1/p
    return -a


rng = np.random.default_rng(0)
for spread in (1.0, .05, .02, .01):
    errs = {'sweep': [], 'rank1': [], 'rank1+newton': [], 'sweep+newton': []}
    for trial in range(20):
        center = rng.standard_normal(64)
        k = center+spread*rng.standard_normal((64, 64)) if spread < 1 else rng.standard_normal((64, 64))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        g = np.exp((k@k.T-1)/.5)+1e-4*np.eye(64)
        e = np.exp((k@rng.standard_normal((64, 64)).T*.0-1)/.5)  # placeholder rhs not used
        ref = np.linalg.inv(g)
        g32 = g.astype(np.float32)
        for name in errs:
            x = (rank1 if name.startswith('rank1') else sweep)(g32)
            if name.endswith('newton'):
                x = x+x@(np.eye(64, dtype=np.float32)-g32@x)
            errs[name].append(np.linalg.norm(x-ref)/np.linalg.norm(ref))
    print(f'spread {spread:5}', ' '.join(f'{n}={np.median(v):.1e}/{np.max(v):.1e}' for n, v in errs.items()),
          f'cond~{np.linalg.cond(g):.0e}')
