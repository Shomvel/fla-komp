# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Cache immutable geometry outputs for one eager non-reentrant checkpoint.

No approximate math and no autograd replacement: the original Exact Function
still saves its normal tape and executes its original analytic backward.
The cache lives with the checkpoint graph, supports repeated backward, and is
never shared across layers, microbatches, or calls. torch.compile is unsupported.
"""
from contextvars import ContextVar

_active = ContextVar('exact_geometry_cache', default=None)


class _Scope:
    def __init__(self, entries, replay):
        self.entries, self.replay = entries, replay
        self.tokens = []

    def __enter__(self):
        # Cursor belongs to each entry into the context, including retained graphs.
        self.tokens.append(_active.set([self.entries, self.replay, 0]))
        return self

    def __exit__(self, exc_type, exc, tb):
        _active.reset(self.tokens.pop())
        return False


def geometry_contexts():
    entries = []
    return _Scope(entries, False), _Scope(entries, True)


def cached_geometry(signature, build):
    active = _active.get()
    if active is None:
        return build()
    entries, replay, cursor = active
    active[2] += 1
    if replay:
        if cursor >= len(entries):
            raise RuntimeError('Exact checkpoint replay has an extra geometry call')
        expected, values = entries[cursor]
        if signature != expected:
            raise RuntimeError('Exact checkpoint geometry contract changed on replay')
        return values
    if cursor != len(entries):
        raise RuntimeError('Exact checkpoint forward context was reused')
    values = build()
    entries.append((signature, values))
    return values
