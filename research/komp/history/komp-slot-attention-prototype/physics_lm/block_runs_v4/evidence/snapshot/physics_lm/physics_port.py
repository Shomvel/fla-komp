#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline synthetic dataset generator for PhysicsLM4 Brevo/Depo tasks.

Outputs JSONL shards with:
  - input_ids: List[int]
  - loss_mask: List[bool]   (True = compute LM loss on this token)
  - task: str
  - meta: dict (small, optional)

Designed to be loaded via HuggingFace datasets:
  load_dataset("json", data_files=".../*.jsonl", streaming=True)
and consumed by flame after adding pretokenized support (input_ids/loss_mask).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple


BREVO_BOS = 9999
BREVO_EOS = 9998

DEPO_BOS = 9999
DEPO_EOS = 9998


def _mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    tmp.replace(path)


def generate_multi_token_words(
    rng: random.Random,
    n: int,
    mini_vocab: int = 3,
    min_tlen: int = 5,
    max_tlen: int = 7,
) -> List[List[int]]:
    """Ported from PhysicsLM4; last token +mini_vocab acts as end-of-word marker."""

    def my_sample(length: int) -> Tuple[int, ...]:
        toks = [rng.randint(1, mini_vocab) for _ in range(length)]
        toks[-1] += mini_vocab
        return tuple(toks)

    words = set()
    while len(words) < n:
        length = rng.randint(min_tlen, max_tlen)
        words.add(my_sample(length))
    return [list(w) for w in words]


class TopoSortDepthStats:
    """Minimal port of Brevo's graph/toposort task."""

    def __init__(self, n: int, vocab_size: int = 125, max_in: int = 4):
        self.n = n
        self.vocab_size = vocab_size
        self.max_in = max_in
        self.degree_constraint = "A_dep_B_for_at_most_4_Bs_and_4_As_with_leaves_on_left"

    def generate_dag(self, rng: random.Random):
        nodes = rng.sample(range(1, self.vocab_size + 1), self.n)
        dag = defaultdict(list)
        if self.degree_constraint != "A_dep_B_for_at_most_4_Bs_and_4_As_with_leaves_on_left":
            raise AssertionError("Unsupported degree_constraint")

        out_degree = defaultdict(int)
        leaves = rng.randint(1, (len(nodes) - 1) // 4 + 1)
        for i in range(leaves, len(nodes)):
            tgt = nodes[i]
            possible_parents = [src for src in nodes[:i] if out_degree[src] < 4]
            if not possible_parents:
                continue
            num_parents = rng.randint(1, min(len(possible_parents), 4))
            parents = rng.sample(possible_parents, num_parents)
            for parent in parents:
                dag[tgt].append(parent)
                out_degree[parent] += 1
        return nodes, dag

    @staticmethod
    def subtree_from_query(dag, query):
        visited = set()
        stack = [query]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for parent in dag.get(node, []):
                    if parent not in visited:
                        stack.append(parent)
        filtered = defaultdict(list)
        for node in visited:
            for parent in dag.get(node, []):
                if parent in visited:
                    filtered[node].append(parent)
        for node in visited:
            _ = filtered[node]
        return filtered

    @staticmethod
    def topological_sort(dag, rng: random.Random):
        indegree = {node: 0 for node in dag}
        for node in dag:
            for parent in dag[node]:
                indegree[parent] += 1
        queue = [node for node in dag if indegree[node] == 0]
        order = []
        while queue:
            node = queue.pop(rng.randint(0, len(queue) - 1))
            order.append(node)
            for parent in dag[node]:
                indegree[parent] -= 1
                if indegree[parent] == 0:
                    queue.append(parent)
        order.reverse()
        return order

    @staticmethod
    def compute_graph_depth(dag, query):
        distance = {query: 0}
        queue = deque([query])
        while queue:
            node = queue.popleft()
            for parent in dag[node]:
                if parent not in distance:
                    distance[parent] = distance[node] + 1
                    queue.append(parent)
        leaves = [node for node in dag if len(dag[node]) == 0]
        if not leaves:
            return 0
        return min(distance.get(leaf, float("inf")) for leaf in leaves if leaf in distance)

    def generate_sample(self, rng: random.Random):
        nodes, dag = self.generate_dag(rng)

        start_index = max(len(nodes) * 3 // 4, len(nodes) - 1)
        candidate_nodes = nodes[start_index:]
        nonzero_degree_nodes = [node for node in candidate_nodes if len(dag[node]) > 0]
        query = rng.choice(nonzero_degree_nodes)

        subdag = self.subtree_from_query(dag, query)
        topo = self.topological_sort(subdag, rng)
        depth = self.compute_graph_depth(subdag, query)
        return dag, topo, depth

    def generate_tokens(self, rng: random.Random, *, multi: bool = False, bos: int = BREVO_BOS, eos: int = BREVO_EOS):
        dag, topo, depth = self.generate_sample(rng)
        query = topo[-1]

        if multi:
            all_node_ids = sorted(set(dag.keys()) | {p for ps in dag.values() for p in ps})
            id_to_index = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
            word_list = generate_multi_token_words(rng, n=len(all_node_ids), mini_vocab=4, min_tlen=2, max_tlen=4)
            word_map = {node_id: word_list[id_to_index[node_id]] for node_id in all_node_ids}

        edges = [(p, c) for c, ps in dag.items() for p in ps]
        rng.shuffle(edges)

        tokens: List[int] = [bos]
        for p, c in edges:
            if multi:
                tokens += word_map[p] + word_map[c]
            else:
                tokens += [p, c]

        # Query delimiter + answer delimiter (matches original code; note bos-1 == eos for their chosen IDs)
        if multi:
            tokens += [bos - 2] + word_map[query] + [bos - 1]
        else:
            tokens += [bos - 2, query, bos - 1]

        # loss labels: 1 for "answer region". Original sets 1 starting from the *answer delimiter* token.
        # Implementation detail: their code creates a [0]*(len(prefix)-1) then fills the remainder with ones.
        loss_labels = [0] * (len(tokens) - 1)

        if multi:
            for node in topo:
                tokens += word_map[node]
        else:
            tokens += topo
        tokens += [eos]

        loss_labels += [1] * (len(tokens) - len(loss_labels))
        assert len(tokens) == len(loss_labels)
        loss_mask = [bool(x) for x in loss_labels]
        return tokens, loss_mask, depth


def _zipf_like_choice(rng: random.Random, low: int, high: int) -> int:
    """Match PhysicsLM4's sampling: p(i) ∝ 1/(i + sqrt(high)) over i∈[low..high]."""
    xs = list(range(low, high + 1))
    power, bias = 1.0, math.sqrt(high)
    ps = [1.0 / ((x**power) + bias + 1e-12) for x in xs]
    z = sum(ps)
    ps = [p / z for p in ps]
    return rng.choices(xs, weights=ps, k=1)[0]


@dataclass(frozen=True)
class BrevoSpec:
    name: str
    N: int
    multi: bool


def gen_brevo_one(rng: random.Random, spec: BrevoSpec) -> Dict:
    n = _zipf_like_choice(rng, 3, spec.N)
    topo = TopoSortDepthStats(n, vocab_size=spec.N)
    input_ids, loss_mask, depth = topo.generate_tokens(rng, multi=spec.multi, bos=BREVO_BOS, eos=BREVO_EOS)
    return {
        "task": "brevo",
        "variant": spec.name,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "meta": {"N": spec.N, "n": n, "multi": spec.multi, "depth": depth},
    }


@dataclass(frozen=True)
class DepoSpec:
    name: str
    N: int
    K: int
    M: int = 10
    separator: bool = False
    mini_vocab: int = 50
    min_tlen: int = 1
    max_tlen: int = 2


def gen_depo_one(rng: random.Random, spec: DepoSpec) -> Dict:
    n = _zipf_like_choice(rng, 3, spec.N)

    vals = generate_multi_token_words(rng, n, mini_vocab=spec.mini_vocab, min_tlen=spec.min_tlen, max_tlen=spec.max_tlen)
    rng.shuffle(vals)
    order = rng.sample(range(n), n)

    text: List[int] = [DEPO_BOS]
    for i in range(n):
        if spec.separator:
            text += [9700]
        v1 = vals[order[i]]
        v2 = vals[(order[i] + 1) % n]
        text += v1 + v2
    loss = [0] * len(text)

    # Queries: [9000+k] + v1 + [9500] + v2
    m = spec.M
    for idx in rng.sample(range(n), k=min(n, m)):
        k = rng.randint(1, spec.K)
        v1, v2 = vals[idx], vals[(idx + k) % n]
        text += [9000 + k] + v1 + [9500] + v2
        loss += [0] * (len(v1) + 1) + [1] * (len(v2) + 1)

    # Add EOS to make boundaries explicit when packing many docs.
    text += [DEPO_EOS]
    loss += [0]

    assert len(text) == len(loss)
    loss_mask = [bool(x) for x in loss]
    return {
        "task": "depo",
        "variant": spec.name,
        "input_ids": text,
        "loss_mask": loss_mask,
        "meta": {"N": spec.N, "n": n, "K": spec.K, "M": spec.M},
    }


def _gen_shard(args) -> str:
    out_dir, task, variant, seed, shard_id, n_samples = args
    rng = random.Random(seed + shard_id * 1000003)

    if task == "brevo":
        if variant == "brevo1":
            spec = BrevoSpec(name="brevo1", N=110, multi=False)
        elif variant == "brevo2":
            spec = BrevoSpec(name="brevo2", N=50, multi=True)
        else:
            raise ValueError(f"Unknown brevo variant: {variant}")
        rows = [gen_brevo_one(rng, spec) for _ in range(n_samples)]
    elif task == "depo":
        if variant == "depo1":
            spec = DepoSpec(name="depo1", N=375, K=8, mini_vocab=50, min_tlen=1, max_tlen=2)
        elif variant == "depo2":
            spec = DepoSpec(name="depo2", N=125, K=16, mini_vocab=3, min_tlen=5, max_tlen=7)
        else:
            raise ValueError(f"Unknown depo variant: {variant}")
        rows = [gen_depo_one(rng, spec) for _ in range(n_samples)]
    else:
        raise ValueError(f"Unknown task: {task}")

    out_path = Path(out_dir) / task / variant / f"train-{shard_id:05d}.jsonl"
    _mkdir(out_path.parent)
    _write_jsonl(out_path, rows)
    return str(out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tasks", type=str, default="brevo:brevo1,depo:depo1")
    p.add_argument("--num_shards", type=int, default=64)
    p.add_argument("--samples_per_shard", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument(
        "--backend",
        type=str,
        default="subprocess",
        choices=["thread", "process", "subprocess"],
        help="Parallelism backend. Use 'subprocess' if your environment blocks multiprocessing semaphores.",
    )
    # Internal: worker mode for --backend subprocess
    p.add_argument("--_worker_jobs", type=str, default=None)
    p.add_argument("--_worker_out", type=str, default=None)
    args = p.parse_args()

    if args._worker_jobs is not None:
        jobs = json.loads(Path(args._worker_jobs).read_text(encoding="utf-8"))
        outs = [_gen_shard(tuple(j)) for j in jobs]
        if args._worker_out is not None:
            Path(args._worker_out).write_text("\n".join(outs) + "\n", encoding="utf-8")
        return

    task_variants: List[Tuple[str, str]] = []
    for tv in args.tasks.split(","):
        tv = tv.strip()
        if not tv:
            continue
        if ":" not in tv:
            raise ValueError(f"Bad --tasks entry: {tv} (expected task:variant)")
        task, variant = tv.split(":", 1)
        task_variants.append((task.strip(), variant.strip()))

    jobs = []
    for task, variant in task_variants:
        for shard_id in range(args.num_shards):
            jobs.append((args.out_dir, task, variant, args.seed, shard_id, args.samples_per_shard))

    _mkdir(args.out_dir)
    manifest = []
    if args.backend == "thread":
        # NOTE: Some HPC/container envs forbid posix semaphores -> multiprocessing pools fail.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.num_proc) as ex:
            futs = [ex.submit(_gen_shard, j) for j in jobs]
            for fut in as_completed(futs):
                manifest.append(fut.result())
    else:
        if args.backend == "process":
            with Pool(processes=args.num_proc) as pool:
                for out_path in pool.imap_unordered(_gen_shard, jobs, chunksize=1):
                    manifest.append(out_path)
        else:
            # args.backend == "subprocess": spawn independent Python workers to avoid multiprocessing SemLock.
            tmp_dir = Path(args.out_dir) / ".job_tmp"
            _mkdir(tmp_dir)
            chunks: List[List[Tuple]] = [[] for _ in range(args.num_proc)]
            for i, j in enumerate(jobs):
                chunks[i % args.num_proc].append(j)

            procs = []
            out_files = []
            script = Path(__file__).resolve()
            for wid, chunk in enumerate(chunks):
                if not chunk:
                    continue
                jobs_path = tmp_dir / f"jobs-{wid:03d}.json"
                out_path = tmp_dir / f"outs-{wid:03d}.txt"
                jobs_path.write_text(json.dumps(chunk), encoding="utf-8")
                out_files.append(out_path)
                procs.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(script),
                            "--out_dir",
                            args.out_dir,
                            "--_worker_jobs",
                            str(jobs_path),
                            "--_worker_out",
                            str(out_path),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )

            for p_ in procs:
                _, err = p_.communicate()
                if p_.returncode != 0:
                    raise RuntimeError(f"worker failed (rc={p_.returncode}):\n{err}")

            for f in out_files:
                if f.exists():
                    manifest.extend([ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()])

    manifest_path = Path(args.out_dir) / "MANIFEST.txt"
    manifest_path.write_text("\n".join(sorted(manifest)) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} shard files. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
