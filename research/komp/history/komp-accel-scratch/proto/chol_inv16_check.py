"""Exact structure of the planned Triton factorization, in float64 and float32:
merged 16x16 Cholesky + inverse (X <- M_j^-1 X), left-looking 16-blocked Cholesky with L_kk^-1 written into
the consumed G(k,k) slot, block forward substitution for L^-1, Pinv = Linv^T Linv; padding (M < BM) by identity."""
import numpy as np


def chol_inv16(A):
    n = len(A); r = np.arange(n); a = A.copy(); L = np.zeros_like(A); X = np.eye(n, dtype=A.dtype)
    for j in range(n):
        col = a[:, j].copy(); d = np.sqrt(col[j])
        l = np.where(r >= j, col/d, 0).astype(A.dtype)
        L[:, j] = l
        a = a-np.outer(l, l)
        xj = X[j, :].copy(); u = l.copy(); u[j] -= 1
        X = X-np.outer(u/d, xj)
    return L, X


def chol_pinv(G, M, BM=64, dt=np.float64):
    nb = BM//16
    S = np.eye(BM, dtype=dt); S[:M, :M] = G[:M, :M]          # padded with identity
    L = np.zeros((BM, BM), dt)
    blk = lambda X, i, j: X[i*16:(i+1)*16, j*16:(j+1)*16]
    def put(X, i, j, v): X[i*16:(i+1)*16, j*16:(j+1)*16] = v
    for k in range(nb):
        akk = blk(S, k, k).copy()
        for j in range(k):
            akk -= blk(L, k, j)@blk(L, k, j).T
        lkk, ikk = chol_inv16(akk)
        put(L, k, k, lkk); put(S, k, k, ikk)
        for i in range(k+1, nb):
            aik = blk(S, i, k).copy()
            for j in range(k):
                aik -= blk(L, i, j)@blk(L, k, j).T
            put(L, i, k, aik@ikk.T)
    for i in range(1, nb):
        for k in range(i):
            s = sum(blk(L, i, j)@blk(S, j, k) for j in range(k, i))
            put(S, i, k, -blk(S, i, i)@s)
    rb = np.arange(BM)//16
    Li = np.where(rb[:, None] >= rb[None, :], S, 0)
    P = Li.T@Li
    return P[:M, :M]


rng = np.random.default_rng(1)
for M in (64, 48, 32, 16):
    for spread in (1.0, .02):
        center = rng.standard_normal(128)
        k = center+spread*rng.standard_normal((M, 128)) if spread < 1 else rng.standard_normal((M, 128))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        g = np.exp((k@k.T-1)/.5)+1e-4*np.eye(M)
        ref = np.linalg.inv(g)
        bm = max(16, 1 << (M-1).bit_length())
        e64 = np.linalg.norm(chol_pinv(g, M, bm)-ref)/np.linalg.norm(ref)
        e32 = np.linalg.norm(chol_pinv(g.astype(np.float32), M, bm, np.float32)-ref)/np.linalg.norm(ref)
        print(f'M={M} BM={bm} spread={spread}: float64 rel err {e64:.1e}, float32 {e32:.1e}')
