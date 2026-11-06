"""Block-KOMP reference: per-token kernel delta writes into fresh slots, FIFO compression at chunk ends.

Within a chunk of C tokens the dictionary is the M carried atoms plus the chunk's new keys. Token t
decays all coefficients by exp(g_t), predicts with the current dictionary, and writes beta_t * err_t
into its own new slot. After the chunk the newest M atoms are kept and every evicted atom is jointly
projected onto them with the ridge-regularized kernel Gram (kept coefficients are not shrunk).
"""
import torch
import torch.nn.functional as F


def kernel(x, y, temperature):
    return torch.exp((x @ y.transpose(-1, -2) - 1) / temperature)


def block_komp_reference(q, k, v, g, beta, initial_state, chunk_size=64, temperature=0.5, ridge=1e-4):
    """q,k:[B,T,H,K], v:[B,T,H,V], g log decay <= 0 ([B,T,H] or per ring slot [B,T,H,M]), beta:[B,T,H].

    state = ([B,H,M,K], [B,H,M,V]).
    """
    initial_keys, initial_values = initial_state
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k = [F.normalize(x.to(dtype), dim=-1, eps=1e-6) for x in (q, k)]
    keys = F.normalize(initial_keys.to(dtype), dim=-1, eps=1e-6)
    coef = initial_values.to(dtype)
    v, g, beta = v.to(dtype), g.to(dtype), beta.to(dtype)
    t_len, m = q.shape[1], keys.shape[-2]
    eye = torch.eye(m, dtype=dtype, device=q.device)
    # Per-slot decay: the atom at sequence position p (initial atoms -M..-1) lives in slot p mod M.
    slot_decay = g.ndim == 4
    slots = torch.arange(m, device=q.device)
    outs = []
    for c0 in range(0, t_len, chunk_size):
        n = min(chunk_size, t_len - c0)
        dictionary, state, active = keys, coef, slots
        for s in range(n):
            t = c0 + s
            if slot_decay:
                state = torch.exp(g[:, t][..., active])[..., None] * state
            else:
                state = torch.exp(g[:, t])[..., None, None] * state
            kt = k[:, t, :, None, :]
            err = v[:, t] - (state * kernel(dictionary, kt, temperature)).sum(-2)
            dictionary = torch.cat((dictionary, kt), -2)
            state = torch.cat((state, (beta[:, t, :, None] * err)[:, :, None]), -2)
            active = torch.cat((active, active.new_tensor([t % m])))
            outs.append((state * kernel(dictionary, q[:, t, :, None, :], temperature)).sum(-2))
        kept, evicted = dictionary[..., n:n+m, :], dictionary[..., :n, :]
        proj = torch.linalg.solve(kernel(kept, kept, temperature) + ridge * eye, kernel(kept, evicted, temperature))
        keys, coef, slots = kept, state[..., n:n+m, :] + proj @ state[..., :n, :], active[n:n+m]
    return torch.stack(outs, 1), (keys, coef)


EVICTION = ('fifo', 'residual', 'coef_norm', 'learned')


def _assign_slots(keep_temp_tslot, freed):
    """Kept temporaries take their positional slot when it was freed; the rest fill free slots in order."""
    free = sorted(freed)
    taken = {}
    pending = []
    for i, slot in enumerate(keep_temp_tslot):
        if slot in free and slot not in taken.values():
            taken[i] = slot
        else:
            pending.append(i)
    rest = [r for r in free if r not in taken.values()]
    for i, r in zip(pending, rest):
        taken[i] = r
    return [taken[i] for i in range(len(keep_temp_tslot))]


def block_komp_token_reference(q, k, v, g, beta, initial_state, chunk_size=64, temperature=0.5, ridge=1e-4,
                               evict='fifo', projectron=False, scores=None, initial_scores=None, coherence=None,
                               dedupe=None, project=True):
    """Token-level Block-KOMP with eviction/projectron flags; loops over batch*head (validation only).

    Dictionary is held by physical slot. g: [B,T,H] or per physical slot [B,T,H,M]. New atom of token t
    decays with slot t mod M inside its chunk. evict: fifo (newest M), residual (largest projection
    residual 1/[(G+ridge)^-1]_aa kept), coef_norm (largest coefficient norm kept), learned (largest score
    kept; scores [B,T,H], initial_scores [B,H,M]); ties go to the newer atom. projectron scales beta by the novelty of k_t: its
    ridge-regularized projection residual on the current dictionary (no gradient through novelty).
    coherence (cosine threshold, None = off; fifo/learned only): k_t becomes a new atom only if its largest cosine with the
    current dictionary (old atoms and this chunk's admitted atoms) is <= coherence; otherwise its write goes to that
    most similar atom, whose key and slot are unchanged (coherence criterion with QKLMS-style merging). Only admitted
    atoms are eviction candidates. No gradient through the admission decision.
    dedupe (cosine threshold, None = off; fifo/learned): chunk-end cleanup. Candidates that have a newer candidate
    with cosine > dedupe are evicted first (older first among them); the joint projection then moves their
    coefficients onto the kept atoms. In-chunk writes are unchanged.
    project=False (ablation): evicted atoms are dropped with their coefficients (no joint projection), i.e. a
    delta-rule sliding window over the kept keys.
    """
    assert evict in EVICTION
    assert coherence is None or (evict in ('fifo', 'learned') and not projectron)
    assert dedupe is None or (evict in ('fifo', 'learned') and not projectron and coherence is None)
    keys0, values0 = initial_state
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    qn, kn = [F.normalize(x.to(dtype), dim=-1, eps=1e-6) for x in (q, k)]
    keys0 = F.normalize(keys0.to(dtype), dim=-1, eps=1e-6)
    v, g, beta, values0 = v.to(dtype), g.to(dtype), beta.to(dtype), values0.to(dtype)
    b_, t_len, h_, _ = q.shape
    m = keys0.shape[-2]
    kern = lambda x, y: kernel(x, y, temperature)
    out = torch.zeros(b_, t_len, h_, v.shape[-1], dtype=dtype, device=q.device)
    out = [[[None]*t_len for _ in range(h_)] for _ in range(b_)]
    final_k, final_v = [[None]*h_ for _ in range(b_)], [[None]*h_ for _ in range(b_)]
    for bi in range(b_):
        for hi in range(h_):
            K, S = keys0[bi, hi], values0[bi, hi]
            born = [float(-m+r) for r in range(m)]
            score = [float(initial_scores[bi, hi, r]) for r in range(m)] if evict == 'learned' else None
            for c0 in range(0, t_len, chunk_size):
                n = min(chunk_size, t_len-c0)
                tk, tw, tslot, tborn, tscore = [], [], [], [], []
                for s in range(n):
                    t = c0+s
                    gt = g[bi, t, hi]
                    decay_slots = torch.exp(gt) if g.ndim == 4 else torch.exp(gt).expand(m)
                    S = decay_slots[:, None]*S
                    tw = [torch.exp(gt[tslot[i]] if g.ndim == 4 else gt)*w for i, w in enumerate(tw)]
                    U = torch.cat([K]+[x[None] for x in tk]) if tk else K
                    coef = torch.cat([S]+[w[None] for w in tw]) if tw else S
                    h = kern(U, kn[bi, t, hi][None])[:, 0]
                    err = v[bi, t, hi]-(coef*h[:, None]).sum(0)
                    bt = beta[bi, t, hi]
                    if projectron:
                        G = kern(U, U)+ridge*torch.eye(U.shape[0], dtype=dtype, device=q.device)
                        novelty = 1.+ridge-h@torch.linalg.solve(G, h)
                        bt = bt*novelty.detach()
                    target = None
                    if coherence is not None:
                        cos = (U.detach()@kn[bi, t, hi].detach())
                        if float(cos.max()) > coherence:
                            target = int(cos.argmax())
                    if target is None:
                        tk.append(kn[bi, t, hi])
                        tw.append(bt*err)
                        tslot.append(t % m)
                        tborn.append(float(t))
                        if evict == 'learned':
                            tscore.append(float(scores[bi, t, hi]))
                    elif target < m:
                        S = torch.cat((S[:target], S[target:target+1]+bt*err[None], S[target+1:]))
                    else:
                        tw[target-m] = tw[target-m]+bt*err
                    U = torch.cat([K]+[x[None] for x in tk])
                    coef = torch.cat([S]+[w[None] for w in tw])
                    out[bi][hi][t] = (coef*kern(U, qn[bi, t, hi][None])).sum(0)
                U = torch.cat([K]+[x[None] for x in tk])
                coef = torch.cat([S]+[w[None] for w in tw])
                if evict == 'fifo':
                    rank = torch.tensor(born+tborn, dtype=dtype)
                elif evict == 'residual':
                    G = kern(U, U)+ridge*torch.eye(U.shape[0], dtype=dtype, device=q.device)
                    rank = (1./torch.linalg.inv(G).diagonal()).detach().cpu()
                elif evict == 'coef_norm':
                    rank = coef.norm(dim=-1).detach().cpu()
                else:
                    rank = torch.tensor(score+tscore, dtype=dtype)
                allb = torch.tensor(born+tborn, dtype=dtype)
                recent = torch.argsort(allb, descending=True)
                order = recent[torch.argsort(rank[recent], descending=True, stable=True)]
                if dedupe is not None:
                    cos = U.detach()@U.detach().T
                    dup = ((cos > dedupe) & (allb[None, :] > allb[:, None])).any(1)
                    order = order[torch.argsort(dup[order].to(torch.int8), stable=True)]
                keep = sorted(order[:m].tolist())
                drop = [i for i in range(m+len(tk)) if i not in keep]
                Kk, Ke = U[keep], U[drop]
                if project:
                    P = torch.linalg.solve(kern(Kk, Kk)+ridge*torch.eye(m, dtype=dtype, device=q.device), kern(Kk, Ke))
                    Snew = coef[keep]+P@coef[drop]
                else:
                    Snew = coef[keep]
                kept_slots = [i for i in keep if i < m]
                kept_temp = [i-m for i in keep if i >= m]
                freed = [r for r in range(m) if r not in kept_slots]
                assigned = _assign_slots([tslot[i] for i in kept_temp], freed)
                order = {i: i for i in kept_slots}
                for i, slot in zip(kept_temp, assigned):
                    order[i+m] = slot
                perm = [None]*m
                for pos, i in enumerate(keep):
                    perm[order[i]] = pos
                K, S = Kk[perm], Snew[perm]
                allborn, allscore = born+tborn, (score+tscore) if evict == 'learned' else None
                born = [allborn[keep[p]] for p in perm]
                if evict == 'learned':
                    score = [allscore[keep[p]] for p in perm]
            final_k[bi][hi], final_v[bi][hi] = K, S
    o = torch.stack([torch.stack([torch.stack(out[bi][hi]) for hi in range(h_)], 1) for bi in range(b_)])
    fk = torch.stack([torch.stack(final_k[bi]) for bi in range(b_)])
    fv = torch.stack([torch.stack(final_v[bi]) for bi in range(b_)])
    return o, (fk, fv)
