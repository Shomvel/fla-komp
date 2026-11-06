"""CPU check of the rank-1 sweep identity against numpy inverse (float64 and float32)."""
import numpy as np

def rank1(a, m):
    a = a.copy()
    for j in range(m):
        col = a[:, j].copy()
        p = col[j]
        x = col.copy(); x[j] -= 1
        a -= np.outer(x, x/p)
    return 2*np.eye(len(a), dtype=a.dtype)-a

def sweep(a, m):
    a = a.copy()
    for j in range(m):
        col = a[:, j].copy(); p = col[j]; c = col/p
        a -= np.outer(col, c)
        a[j, :] = c; a[:, j] = c; a[j, j] = -1/p
    return -a

rng = np.random.default_rng(0)
for dup in (False, True):
    k = rng.standard_normal((64, 128))
    if dup:
        k[1::2] = k[::2]+1e-3*rng.standard_normal((32, 128))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    g = np.exp((k@k.T-1)/.5)+1e-4*np.eye(64)
    ref = np.linalg.inv(g)
    for dt in (np.float64, np.float32):
        for name, f in (('sweep', sweep), ('rank1', rank1)):
            e = np.abs(f(g.astype(dt), 64)-ref).max()/np.abs(ref).max()
            print(f'dup={dup} {dt.__name__:8s} {name:6s} max rel err {e:.2e}')
