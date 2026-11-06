"""Chunked kernel geometry: one program per (chunk, batch*head) carries the inverse Gram.

The chunk's first post-write Gram is inverted with an in-register sweep. Each later
token replaces one dictionary row, a symmetric rank-two change, so the inverse is
carried by a Woodbury update and every projection gets one step of iterative
refinement against the exactly maintained Gram. The backward pass replays the same
chain from the stored chunk-start inverse; z=P dp replaces both triangular solves, and
dictionary-key gradients are accumulated per slot in registers, flushed only when a
slot is overwritten and at the end of the chunk.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from .geometry import normalize_kernel, routes_kernel, routes_parallel_into, final_keys_kernel, final_bwd_kernel, normalize_bwd_kernel
from .utils import zero

# num_warps per kernel; single-warp loops avoid cross-warp reduction barriers.
WARPS = {'start': 1, 'fwd': 4, 'z': 1, 'grad': 1, 'finish': 1}


@tr.jit
def _sweep_inverse(gram, M: tl.constexpr, BM: tl.constexpr):
    # Symmetric sweep operator without pivoting; after M sweeps a = -inverse.
    r = tl.arange(0, BM)
    a = gram
    for j in range(M):
        col = tl.sum(tl.where(r[None, :] == j, a, 0.), 1)
        piv = tl.sum(tl.where(r == j, col, 0.), 0)
        c = col/piv
        a = a-col[:, None]*c[None, :]
        rowj = r[:, None] == j
        colj = r[None, :] == j
        a = tl.where(rowj, c[None, :], tl.where(colj, c[:, None], a))
        a = tl.where(rowj & colj, -1./piv, a)
    valid = (r[:, None] < M) & (r[None, :] < M)
    return tl.where(valid, -a, 0.)


@tr.jit
def _woodbury(P, G, gnew, gold, j, BM: tl.constexpr):
    # G' = G + u e^T + e u^T with u = (gnew-gold) and half the diagonal change.
    r = tl.arange(0, BM)
    d = gnew-gold
    u = tl.where(r == j, .5*d, d)
    pu = tl.sum(P*u[None, :], 1)
    pe = tl.sum(tl.where(r[None, :] == j, P, 0.), 1)
    a11 = 1.+tl.sum(tl.where(r == j, pu, 0.), 0)
    a12 = tl.sum(tl.where(r == j, pe, 0.), 0)
    a21 = tl.sum(u*pu, 0)
    det = a11*a11-a12*a21
    c11 = a11/det
    c12 = -a12/det
    c21 = -a21/det
    P = P-(c12*pu[:, None]*pu[None, :]+c21*pe[:, None]*pe[None, :]+c11*(pu[:, None]*pe[None, :]+pe[:, None]*pu[None, :]))
    G = tl.where(r[:, None] == j, gnew[None, :], tl.where(r[None, :] == j, gnew[:, None], G))
    return P, G


@tr.jit
def geometry_chunk_fwd_kernel(QN, Pool, Source, Removed, Slot, Hist, Proj, Read, P0,
                              T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                              L: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                              TAU: tl.constexpr, RIDGE: tl.constexpr, SAVE: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    r = tl.arange(0, BM)
    d = tl.arange(0, BD)
    vm = r < M
    vd = d < D
    valid = vm[:, None] & vm[None, :]
    eye = (r[:, None] == r[None, :]) & valid
    base = bh*(T+M)
    t0 = chunk*L
    tok0 = (b*T+t0)*H+head
    src = tl.load(Source+tok0*M+r, vm, 0)
    keys = tl.load(Pool+(base+src[:, None])*D+d[None, :], vm[:, None] & vd[None, :], 0)
    gram = tl.exp((tl.dot(keys, tl.trans(keys), input_precision='ieee')-1.)/TAU)
    G = tl.where(valid, gram, 0.)+tl.where(eye, RIDGE, 0.)
    P = _sweep_inverse(G, M, BM)
    if SAVE:
        tl.store(P0+(bh*NC+chunk)*M*M+r[:, None]*M+r[None, :], P, valid)
    n = tl.minimum(L, T-t0)
    for s in range(n):
        t = t0+s
        tok = (b*T+t)*H+head
        j = tl.load(Slot+tok)
        rem = tl.load(Removed+tok)
        o = tl.load(Pool+(base+rem)*D+d, vd, 0)
        k = tl.load(Pool+(base+M+t)*D+d, vd, 0)
        q = tl.load(QN+tok*D+d, vd, 0)
        keys = tl.where(r[:, None] == j, k[None, :], keys)
        cn = tl.exp((tl.sum(keys*k[None, :], 1)-1.)/TAU)
        co = tl.exp((tl.sum(keys*o[None, :], 1)-1.)/TAU)
        rd = tl.exp((tl.sum(keys*q[None, :], 1)-1.)/TAU)
        cn = tl.where(vm, cn, 0.)
        co = tl.where(vm, co, 0.)
        if s > 0:
            gnew = cn+tl.where(r == j, RIDGE, 0.)
            gold = tl.sum(tl.where(r[None, :] == j, G, 0.), 1)
            P, G = _woodbury(P, G, gnew, gold, j, BM)
        p = tl.sum(P*co[None, :], 1)
        res = co-tl.sum(G*p[None, :], 1)
        p += tl.sum(P*res[None, :], 1)
        hist = tl.where(r == j, co, cn)
        tl.store(Hist+tok*M+r, hist, vm)
        tl.store(Proj+tok*M+r, p, vm)
        tl.store(Read+tok*M+r, rd, vm)


@tr.jit
def geometry_chunk_bwd_kernel(QN, Pool, Source, Removed, Slot, Hist, Proj, Read, P0, DH, DP, DR, DQN, DPool,
                              T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                              L: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                              TAU: tl.constexpr, RIDGE: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    r = tl.arange(0, BM)
    d = tl.arange(0, BD)
    vm = r < M
    vd = d < D
    valid = vm[:, None] & vm[None, :]
    eye = (r[:, None] == r[None, :]) & valid
    base = bh*(T+M)
    t0 = chunk*L
    tok0 = (b*T+t0)*H+head
    src = tl.load(Source+tok0*M+r, vm, 0)
    keys = tl.load(Pool+(base+src[:, None])*D+d[None, :], vm[:, None] & vd[None, :], 0)
    gram = tl.exp((tl.dot(keys, tl.trans(keys), input_precision='ieee')-1.)/TAU)
    G = tl.where(valid, gram, 0.)+tl.where(eye, RIDGE, 0.)
    P = tl.load(P0+(bh*NC+chunk)*M*M+r[:, None]*M+r[None, :], valid, 0)
    acc = tl.zeros((BM, BD), tl.float32)
    dacc = tl.zeros((BM, BM), tl.float32)
    n = tl.minimum(L, T-t0)
    for s in range(n):
        t = t0+s
        tok = (b*T+t)*H+head
        j = tl.load(Slot+tok)
        rem = tl.load(Removed+tok)
        o = tl.load(Pool+(base+rem)*D+d, vd, 0)
        k = tl.load(Pool+(base+M+t)*D+d, vd, 0)
        q = tl.load(QN+tok*D+d, vd, 0)
        rowj = r[:, None] == j
        colj = r[None, :] == j
        gout = tl.zeros((BD,), tl.float32)
        if s > 0:
            # Flush gradients that belong to the atom leaving slot j.
            dcol = tl.sum(tl.where(colj, dacc, 0.), 1)
            acc += dcol[:, None]*o[None, :]
            dacc = tl.where(colj, 0., dacc)
            drow = tl.sum(tl.where(rowj, dacc, 0.), 0)
            gout = tl.sum(tl.where(rowj, acc, 0.), 0)+tl.sum(drow[:, None]*keys, 0)
            dacc = tl.where(rowj, 0., dacc)
            acc = tl.where(rowj, 0., acc)
            keys = tl.where(rowj, k[None, :], keys)
        cn = tl.exp((tl.sum(keys*k[None, :], 1)-1.)/TAU)
        co = tl.exp((tl.sum(keys*o[None, :], 1)-1.)/TAU)
        cn = tl.where(vm, cn, 0.)
        co = tl.where(vm, co, 0.)
        if s > 0:
            gnew = cn+tl.where(r == j, RIDGE, 0.)
            gold = tl.sum(tl.where(r[None, :] == j, G, 0.), 1)
            P, G = _woodbury(P, G, gnew, gold, j, BM)
        p = tl.load(Proj+tok*M+r, vm, 0)
        dp = tl.load(DP+tok*M+r, vm, 0)
        dh = tl.load(DH+tok*M+r, vm, 0)*tl.load(Hist+tok*M+r, vm, 0)/TAU
        dr = tl.load(DR+tok*M+r, vm, 0)*tl.load(Read+tok*M+r, vm, 0)/TAU
        z = tl.sum(P*dp[None, :], 1)
        res = dp-tl.sum(G*z[None, :], 1)
        z += tl.sum(P*res[None, :], 1)
        dacc += -(z[:, None]*p[None, :]+p[:, None]*z[None, :])*tl.where(eye, G-RIDGE, G)/TAU
        db = z*co/TAU
        dhj = tl.sum(tl.where(r == j, dh, 0.), 0)
        dhm = tl.where(r == j, 0., dh)
        acc += db[:, None]*o[None, :]+dr[:, None]*q[None, :]+dhm[:, None]*k[None, :]
        gout += tl.sum(db[:, None]*keys, 0)+dhj*k
        dk = tl.sum(dhm[:, None]*keys, 0)+dhj*o
        acc += tl.where(rowj, dk[None, :], 0.)
        tl.store(DQN+tok*D+d, tl.sum(dr[:, None]*keys, 0), vd)
        tl.atomic_add(DPool+(base+rem)*D+d, gout, vd, sem='relaxed')
        src = tl.where(r == j, (M+t).to(tl.int32), src)
    acc += tl.dot(dacc, keys, input_precision='ieee')
    tl.atomic_add(DPool+(base+src[:, None])*D+d[None, :], acc, vm[:, None] & vd[None, :], sem='relaxed')


@tr.jit
def _woodbury_p(P, gnew, gold, j, BM: tl.constexpr):
    r = tl.arange(0, BM)
    d = gnew-gold
    u = tl.where(r == j, .5*d, d)
    pu = tl.sum(P*u[None, :], 1)
    pe = tl.sum(tl.where(r[None, :] == j, P, 0.), 1)
    a11 = 1.+tl.sum(tl.where(r == j, pu, 0.), 0)
    a12 = tl.sum(tl.where(r == j, pe, 0.), 0)
    a21 = tl.sum(u*pu, 0)
    det = a11*a11-a12*a21
    c11 = a11/det
    c12 = -a12/det
    c21 = -a21/det
    return P-(c12*pu[:, None]*pu[None, :]+c21*pe[:, None]*pe[None, :]+c11*(pu[:, None]*pe[None, :]+pe[:, None]*pu[None, :]))


@tr.jit
def geometry_chunk_z_kernel(Pool, Source, Removed, Slot, P0, DP, Z,
                            T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                            L: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                            TAU: tl.constexpr, RIDGE: tl.constexpr):
    # Replays the inverse chain only; z = P dp. The old Gram column comes from fresh kernel values.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    r = tl.arange(0, BM)
    d = tl.arange(0, BD)
    vm = r < M
    vd = d < D
    valid = vm[:, None] & vm[None, :]
    base = bh*(T+M)
    t0 = chunk*L
    tok0 = (b*T+t0)*H+head
    src = tl.load(Source+tok0*M+r, vm, 0)
    keys = tl.load(Pool+(base+src[:, None])*D+d[None, :], vm[:, None] & vd[None, :], 0)
    P = tl.load(P0+(bh*NC+chunk)*M*M+r[:, None]*M+r[None, :], valid, 0)
    n = tl.minimum(L, T-t0)
    for s in range(n):
        t = t0+s
        tok = (b*T+t)*H+head
        if s > 0:
            j = tl.load(Slot+tok)
            rem = tl.load(Removed+tok)
            o = tl.load(Pool+(base+rem)*D+d, vd, 0)
            k = tl.load(Pool+(base+M+t)*D+d, vd, 0)
            keys = tl.where(r[:, None] == j, k[None, :], keys)
            cn = tl.where(vm, tl.exp((tl.sum(keys*k[None, :], 1)-1.)/TAU), 0.)
            co = tl.where(vm, tl.exp((tl.sum(keys*o[None, :], 1)-1.)/TAU), 0.)
            oo = tl.exp((tl.sum(o*o, 0)-1.)/TAU)
            P = _woodbury_p(P, cn+tl.where(r == j, RIDGE, 0.), tl.where(r == j, oo+RIDGE, co), j, BM)
        dp = tl.load(DP+tok*M+r, vm, 0)
        tl.store(Z+tok*M+r, tl.sum(P*dp[None, :], 1), vm)


@tr.jit
def geometry_chunk_grad_kernel(QN, Pool, Source, Removed, Slot, Hist, Proj, Read, Z, DH, DR, DQN, DPool, Acc, Dacc,
                               T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                               L: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                               TAU: tl.constexpr, RIDGE: tl.constexpr):
    # Key/query gradients from z. dacc holds -(z p^T + p z^T)/tau summed while both atoms of an
    # entry stay in place; their Gram entry is constant over that window, so it is applied only
    # when a slot is overwritten (using the outgoing atom's kernel column) and at the chunk end.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    r = tl.arange(0, BM)
    d = tl.arange(0, BD)
    vm = r < M
    vd = d < D
    valid = vm[:, None] & vm[None, :]
    base = bh*(T+M)
    t0 = chunk*L
    tok0 = (b*T+t0)*H+head
    src = tl.load(Source+tok0*M+r, vm, 0)
    keys = tl.load(Pool+(base+src[:, None])*D+d[None, :], vm[:, None] & vd[None, :], 0)
    acc = tl.zeros((BM, BD), tl.float32)
    dacc = tl.zeros((BM, BM), tl.float32)
    n = tl.minimum(L, T-t0)
    for s in range(n):
        t = t0+s
        tok = (b*T+t)*H+head
        j = tl.load(Slot+tok)
        rem = tl.load(Removed+tok)
        o = tl.load(Pool+(base+rem)*D+d, vd, 0)
        k = tl.load(Pool+(base+M+t)*D+d, vd, 0)
        q = tl.load(QN+tok*D+d, vd, 0)
        rowj = r[:, None] == j
        colj = r[None, :] == j
        gout = tl.zeros((BD,), tl.float32)
        if s > 0:
            # keys still hold the outgoing atom o in slot j: cpre = Gram column j.
            cpre = tl.where(vm, tl.exp((tl.sum(keys*o[None, :], 1)-1.)/TAU), 0.)
            dcol = tl.sum(tl.where(colj, dacc, 0.), 1)*cpre
            acc += dcol[:, None]*o[None, :]
            drow = tl.sum(tl.where(rowj & ~colj, dacc, 0.), 0)*cpre
            gout = tl.sum(tl.where(rowj, acc, 0.), 0)+tl.sum(drow[:, None]*keys, 0)
            dacc = tl.where(rowj | colj, 0., dacc)
            acc = tl.where(rowj, 0., acc)
            keys = tl.where(rowj, k[None, :], keys)
            co = tl.where(r == j, tl.exp((tl.sum(k*o, 0)-1.)/TAU), cpre)
        else:
            co = tl.where(vm, tl.exp((tl.sum(keys*o[None, :], 1)-1.)/TAU), 0.)
        z = tl.load(Z+tok*M+r, vm, 0)
        p = tl.load(Proj+tok*M+r, vm, 0)
        dh = tl.load(DH+tok*M+r, vm, 0)*tl.load(Hist+tok*M+r, vm, 0)/TAU
        dr = tl.load(DR+tok*M+r, vm, 0)*tl.load(Read+tok*M+r, vm, 0)/TAU
        dacc -= (z[:, None]*p[None, :]+p[:, None]*z[None, :])/TAU
        db = z*co/TAU
        dhj = tl.sum(tl.where(r == j, dh, 0.), 0)
        dhm = tl.where(r == j, 0., dh)
        acc += db[:, None]*o[None, :]+dr[:, None]*q[None, :]+dhm[:, None]*k[None, :]
        gout += tl.sum(db[:, None]*keys, 0)+dhj*k
        dk = tl.sum(dhm[:, None]*keys, 0)+dhj*o
        acc += tl.where(rowj, dk[None, :], 0.)
        tl.store(DQN+tok*D+d, tl.sum(dr[:, None]*keys, 0), vd)
        tl.atomic_add(DPool+(base+rem)*D+d, gout, vd, sem='relaxed')
        src = tl.where(r == j, (M+t).to(tl.int32), src)
    idx = bh*NC+chunk
    tl.store(Dacc+idx*M*M+r[:, None]*M+r[None, :], dacc, valid)
    tl.store(Acc+idx*M*D+r[:, None]*D+d[None, :], acc, vm[:, None] & vd[None, :])


@tr.jit
def chunk_start_kernel(Pool, Source, G0, P0, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                       L: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                       TAU: tl.constexpr, RIDGE: tl.constexpr):
    # Post-write Gram of each chunk's first token and its sweep inverse.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    r = tl.arange(0, BM)
    d = tl.arange(0, BD)
    vm = r < M
    valid = vm[:, None] & vm[None, :]
    eye = (r[:, None] == r[None, :]) & valid
    tok0 = (b*T+chunk*L)*H+head
    src = tl.load(Source+tok0*M+r, vm, 0)
    keys = tl.load(Pool+(bh*(T+M)+src[:, None])*D+d[None, :], vm[:, None] & (d[None, :] < D), 0)
    G = tl.where(valid, tl.exp((tl.dot(keys, tl.trans(keys), input_precision='ieee')-1.)/TAU), 0.)+tl.where(eye, RIDGE, 0.)
    out = (bh*NC+chunk)*M*M+r[:, None]*M+r[None, :]
    tl.store(G0+out, G, valid)
    tl.store(P0+out, _sweep_inverse(G, M, BM), valid)


@tr.jit
def geometry_loop_fwd_kernel(QN, Pool, Source, Removed, Slot, Hist, Proj, Read, G0, P0,
                             T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                             L: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                             TAU: tl.constexpr, RIDGE: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    r = tl.arange(0, BM)
    d = tl.arange(0, BD)
    vm = r < M
    vd = d < D
    valid = vm[:, None] & vm[None, :]
    base = bh*(T+M)
    t0 = chunk*L
    tok0 = (b*T+t0)*H+head
    src = tl.load(Source+tok0*M+r, vm, 0)
    keys = tl.load(Pool+(base+src[:, None])*D+d[None, :], vm[:, None] & vd[None, :], 0)
    off = (bh*NC+chunk)*M*M+r[:, None]*M+r[None, :]
    G = tl.load(G0+off, valid, 0)
    P = tl.load(P0+off, valid, 0)
    n = tl.minimum(L, T-t0)
    for s in range(n):
        t = t0+s
        tok = (b*T+t)*H+head
        j = tl.load(Slot+tok)
        rem = tl.load(Removed+tok)
        o = tl.load(Pool+(base+rem)*D+d, vd, 0)
        k = tl.load(Pool+(base+M+t)*D+d, vd, 0)
        q = tl.load(QN+tok*D+d, vd, 0)
        keys = tl.where(r[:, None] == j, k[None, :], keys)
        cn = tl.where(vm, tl.exp((tl.sum(keys*k[None, :], 1)-1.)/TAU), 0.)
        co = tl.where(vm, tl.exp((tl.sum(keys*o[None, :], 1)-1.)/TAU), 0.)
        rd = tl.exp((tl.sum(keys*q[None, :], 1)-1.)/TAU)
        if s > 0:
            gnew = cn+tl.where(r == j, RIDGE, 0.)
            gold = tl.sum(tl.where(r[None, :] == j, G, 0.), 1)
            P, G = _woodbury(P, G, gnew, gold, j, BM)
        pr = tl.sum(P*co[None, :], 1)
        res = co-tl.sum(G*pr[None, :], 1)
        pr += tl.sum(P*res[None, :], 1)
        tl.store(Hist+tok*M+r, tl.where(r == j, co, cn), vm)
        tl.store(Proj+tok*M+r, pr, vm)
        tl.store(Read+tok*M+r, rd, vm)


@tr.jit
def grad_finish_kernel(Pool, Source, Acc, Dacc, DPool, T: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                       L: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr, TAU: tl.constexpr, H: tl.constexpr):
    # Apply the final Gram to the pending pair coefficients and flush every slot of the chunk.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    r = tl.arange(0, BM)
    d = tl.arange(0, BD)
    vm = r < M
    vd = d < D
    valid = vm[:, None] & vm[None, :]
    idx = bh*NC+chunk
    last = tl.minimum((chunk+1)*L, T)-1
    tok = ((bh//H)*T+last)*H+bh % H
    src = tl.load(Source+tok*M+r, vm, 0)
    base = bh*(T+M)
    keys = tl.load(Pool+(base+src[:, None])*D+d[None, :], vm[:, None] & vd[None, :], 0)
    gram = tl.where(valid, tl.exp((tl.dot(keys, tl.trans(keys), input_precision='ieee')-1.)/TAU), 0.)
    dacc = tl.load(Dacc+idx*M*M+r[:, None]*M+r[None, :], valid, 0)
    acc = tl.load(Acc+idx*M*D+r[:, None]*D+d[None, :], vm[:, None] & vd[None, :], 0)
    acc += tl.dot(dacc*gram, keys, input_precision='ieee')
    tl.atomic_add(DPool+(base+src[:, None])*D+d[None, :], acc, vm[:, None] & vd[None, :], sem='relaxed')


class _ChunkGeometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, slots, initial, tau, ridge, save, block, parallel_routes, warps):
        b, t, h, d = q.shape
        m = initial.shape[-2]
        bm = max(16, tr.next_power_of_2(m))
        bd = max(16, tr.next_power_of_2(d))
        nc = tr.cdiv(t, block)
        alloc = lambda *s: torch.empty(s, device=q.device, dtype=torch.float32)
        qn = alloc(b, t, h, d)
        pool = alloc(b, h, t+m, d)
        sources = torch.empty((b, t, h, m), device=q.device, dtype=torch.int32)
        removed = torch.empty((b, t, h), device=q.device, dtype=torch.int32)
        last = torch.empty((b, h, m), device=q.device, dtype=torch.int32)
        hist, proj, read = [alloc(b, t, h, m) for _ in range(3)]
        g0, p0 = alloc(b, h, nc, m, m), alloc(b, h, nc, m, m)
        final = alloc(b, h, m, d)
        normalize_kernel[(t+m, b*h)](q, k, initial, qn, pool, t, h, m, d, bd)
        if parallel_routes:
            routes_parallel_into(slots, sources, removed, last, m)
        else:
            routes_kernel[(b*h,)](slots, sources, removed, last, t, h, m, tr.next_power_of_2(m))
        chunk_start_kernel[(nc, b*h)](pool, sources, g0, p0, t, h, m, d, block, nc, bm, bd, tau, ridge, num_warps=WARPS['start'])
        geometry_loop_fwd_kernel[(nc, b*h)](qn, pool, sources, removed, slots, hist, proj, read, g0, p0,
                                            t, h, m, d, block, nc, bm, bd, tau, ridge, num_warps=WARPS['fwd'])
        final_keys_kernel[(b*h, m)](pool, last, final, t, m, d, bd)
        ctx.save_for_backward(q, k, initial, slots, qn, pool, sources, removed, last, hist, proj, read, p0)
        ctx.cfg = (tau, ridge, block)
        return hist, proj, read, final

    @staticmethod
    @once_differentiable
    def backward(ctx, dh, dp, dr, df):
        q, k, initial, slots, qn, pool, sources, removed, last, hist, proj, read, p0 = ctx.saved_tensors
        tau, ridge, block = ctx.cfg
        b, t, h, d = q.shape
        m = initial.shape[-2]
        bm = max(16, tr.next_power_of_2(m))
        bd = max(16, tr.next_power_of_2(d))
        nc = tr.cdiv(t, block)
        dpool = zero(torch.empty_like(pool))
        dqn = torch.empty_like(qn)
        dq, dk, di = [torch.empty_like(x) for x in (q, k, initial)]
        dh, dp, dr = dh.contiguous(), dp.contiguous(), dr.contiguous()
        z = torch.empty_like(proj)
        acc = torch.empty((b, h, nc, m, d), device=q.device, dtype=torch.float32)
        dacc = torch.empty((b, h, nc, m, m), device=q.device, dtype=torch.float32)
        geometry_chunk_z_kernel[(nc, b*h)](pool, sources, removed, slots, p0, dp, z,
                                           t, h, m, d, block, nc, bm, bd, tau, ridge, num_warps=WARPS['z'])
        geometry_chunk_grad_kernel[(nc, b*h)](qn, pool, sources, removed, slots, hist, proj, read, z, dh, dr, dqn, dpool, acc, dacc,
                                              t, h, m, d, block, nc, bm, bd, tau, ridge, num_warps=WARPS['grad'])
        grad_finish_kernel[(nc, b*h)](pool, sources, acc, dacc, dpool, t, m, d, block, nc, bm, bd, tau, h, num_warps=WARPS['finish'])
        final_bwd_kernel[(b*h, m)](last, df.contiguous(), dpool, t, m, d, bd)
        normalize_bwd_kernel[(t+m, b*h)](q, k, initial, dqn, dpool, dq, dk, di, t, h, m, d, bd)
        return dq, dk, None, di, None, None, None, None, None, None


def triton_geometry_chunked(q, k, slots, initial, temperature=.5, ridge=1e-4, block=16, warps=4):
    """Same outputs and gradients as triton_geometry; inverse carried within `block` tokens."""
    save = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, initial))
    parallel = q.shape[1] >= 192 and initial.shape[-2] >= 16
    with torch.cuda.device(q.device):
        return _ChunkGeometry.apply(q.contiguous(), k.contiguous(), slots.contiguous(), initial.contiguous(),
                                    temperature, ridge, save, block, parallel, warps)
