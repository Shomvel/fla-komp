# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deterministic rank-sharded batches from a flat uint16 token file."""

import hashlib
from pathlib import Path

import numpy as np
import torch


class TokenBatches:
    def __init__(self, path, sequence_length, global_batch, rank=0, world_size=1):
        if sequence_length < 1 or global_batch < 1 or world_size < 1:
            raise ValueError("Sequence length, batch and world size must be positive")
        if global_batch % world_size or not 0 <= rank < world_size:
            raise ValueError("Global batch must divide evenly across valid ranks")
        self.path = Path(path)
        if self.path.stat().st_size % 2:
            raise ValueError("Token file must contain complete uint16 values")
        self.tokens = np.memmap(self.path, dtype="<u2", mode="r")
        self.sequence_length = sequence_length
        self.global_batch = global_batch
        self.local_batch = global_batch // world_size
        self.rank = rank
        self.steps = (len(self.tokens) - 1) // (sequence_length * global_batch)
        if self.steps < 1:
            raise ValueError("Token file must hold at least one full global batch plus its boundary token")

    def batch(self, step):
        # Keep sequence boundary targets exactly once; repeat complete epochs deterministically.
        first = ((step % self.steps) * self.global_batch + self.rank * self.local_batch) * self.sequence_length
        rows = [
            np.array(self.tokens[start: start + self.sequence_length + 1], dtype=np.int64)
            for start in range(first, first + self.local_batch * self.sequence_length, self.sequence_length)
        ]
        ids = torch.from_numpy(np.stack(rows))
        return ids[:, :-1], ids[:, 1:]

    def fingerprint(self):
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
