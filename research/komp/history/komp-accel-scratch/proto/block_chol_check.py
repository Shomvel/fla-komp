"""float32 CPU emulation: 64x64 SPD kernel-Gram inverse by (a) the scalar sweep, (b) explicit-inverse block
LDL (round 2's failed variant), (c) blocked Cholesky (16x16 blocks) + blocked triangular inverse,
Pinv = Linv^T Linv. Error vs float64 inverse, and of P = Pinv K (what the operator uses)."""
import numpy as np

f32 = np.float32


def sweep(a):
    a = a.copy()
    for j in range(len(a)):
        col = a[:, j].copy(); p = col[j]; c = col/p
        a -= np.outer(col, c); a[j, :] = c; a[:, j] = c; a[j, j] = f32(-1)/p
    return -a


def chol16(a):
    # unblocked right-looking Cholesky of a 16x16 tile (float32)
    a = a.copy(); n = len(a); L = np.zeros_like(a)
    for j in range(n):
        d = np.sqrt(a[j, j]); l = a[:, j]/d; l[:j] = 0
        L[:, j] = l
        a -= np.outer(l, l)
    return L


def trinv16(L):
    # forward substitution inverse of a lower-triangular 16x16 tile
    n = len(L); X = np.zeros_like(L)
    for i in range(n):
        X[i, :] = (np.eye(n, dtype=L.dtype)[i]-L[i, :i]@X[:i, :])/L[i, i]
    return X


def block_chol_inv(g, b=16):
    n = len(g); nb = n//b
    B = lambda X, i, j: X[i*b:(i+1)*b, j*b:(j+1)*b]
    L = np.zeros_like(g); Li = np.zeros_like(g)
    for k in range(nb):
        akk = B(g, k, k)-sum((B(L, k, j)@B(L, k, j).T for j in range(k)), np.zeros((b, b), f32))
        lkk = chol16(akk); ikk = trinv16(lkk)
        L[k*b:(k+1)*b, k*b:(k+1)*b] = lkk; Li[k*b:(k+1)*b, k*b:(k+1)*b] = ikk
        for i in range(k+1, nb):
            aik = B(g, i, k)-sum((B(L, i, j)@B(L, k, j).T for j in range(k)), np.zeros((b, b), f32))
            L[i*b:(i+1)*b, k*b:(k+1)*b] = aik@ikk.T
    for k in range(nb):
        for i in range(k+1, nb):
            s = sum((B(L, i, j)@B(Li, j, k) for j in range(k, i)), np.zeros((b, b), f32))
            Li[i*b:(i+1)*b, k*b:(k+1)*b] = -B(Li, i, i)@s
    return Li.T@Li


def block_ldl_explicit(g):
    h = 32
    g11, g12, g22 = g[:h, :h], g[:h, h:], g[h:, h:]
    p11 = sweep(g11); x = p11@g12; s = g22-g12.T@x
    p22 = sweep(s); p12 = -x@p22; p11 = p11-p12@x.T
    return np.block([[p11, p12], [p12.T, p22]])


rng = np.random.default_rng(0)
for spread in (1.0, .05, .02, .01):
    errs = {k: [] for k in ('sweep', 'block_ldl_explicit', 'block_chol')}
    perr = {k: [] for k in errs}
    for trial in range(20):
        center = rng.standard_normal(128)
        k = center+spread*rng.standard_normal((64, 128)) if spread < 1 else rng.standard_normal((64, 128))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        kold = center+spread*rng.standard_normal((64, 128)) if spread < 1 else rng.standard_normal((64, 128))
        kold /= np.linalg.norm(kold, axis=1, keepdims=True)
        g = np.exp((k@k.T-1)/.5)+1e-4*np.eye(64)
        kc = np.exp((k@kold.T-1)/.5)
        ref = np.linalg.inv(g); pref = ref@kc
        for name, fn in (('sweep', sweep), ('block_ldl_explicit', block_ldl_explicit), ('block_chol', block_chol_inv)):
            x = fn(g.astype(f32)).astype(np.float64)
            errs[name].append(np.linalg.norm(x-ref)/np.linalg.norm(ref))
            perr[name].append(np.linalg.norm(x.astype(f32)@kc.astype(f32)-pref)/np.linalg.norm(pref))
    print(f'spread {spread:5}', ' | '.join(f'{n}: inv {np.median(errs[n]):.1e}/{np.max(errs[n]):.1e} P {np.median(perr[n]):.1e}/{np.max(perr[n]):.1e}' for n in errs))
