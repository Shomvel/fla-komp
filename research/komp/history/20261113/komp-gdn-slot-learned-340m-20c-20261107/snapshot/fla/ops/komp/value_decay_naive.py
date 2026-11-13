# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Differentiable token oracle for value-coordinate FIFO KOMP."""

import torch
from torch.nn import functional as F

from fla.ops.komp.block_reference import kernel


def naive_value_decay_komp(q, k, v, g, beta, initial_state, *, chunk_size=128, temperature=0.5, ridge=1e-4):
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in (q, k, v, g, beta, *initial_state)) else torch.float32
    output_dtype = v.dtype
    with torch.autocast(q.device.type, enabled=False):
        q, k = [F.normalize(x.to(dtype), dim=-1, eps=1e-6) for x in (q, k)]
        v, g, beta = [x.to(dtype) for x in (v, g, beta)]
        keys = F.normalize(initial_state[0].to(dtype), dim=-1, eps=1e-6)
        coef = initial_state[1].to(dtype)
        m = keys.shape[-2]
        eye = torch.eye(m, device=q.device, dtype=dtype)
        outputs = []
        for start in range(0, q.shape[1], chunk_size):
            end = min(start + chunk_size, q.shape[1])
            dictionary, state = keys, coef
            for t in range(start, end):
                state = state * g[:, t].exp()[..., None, :]
                kt = k[:, t, :, None, :]
                error = v[:, t] - (state * kernel(dictionary, kt, temperature)).sum(-2)
                dictionary = torch.cat((dictionary, kt), -2)
                state = torch.cat((state, (beta[:, t, :, None] * error)[:, :, None]), -2)
                outputs.append((state * kernel(dictionary, q[:, t, :, None, :], temperature)).sum(-2))
            n = end - start
            kept, evicted = dictionary[..., n:, :], dictionary[..., :n, :]
            projection = torch.linalg.solve(kernel(kept, kept, temperature) + ridge * eye, kernel(kept, evicted, temperature))
            keys, coef = kept, state[..., n:, :] + projection @ state[..., :n, :]
        return torch.stack(outputs, 1).to(output_dtype), (keys, coef)
