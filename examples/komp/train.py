# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Single-node or torchrun KOMP340M training on pre-tokenized uint16 data."""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from examples.komp.data import TokenBatches
from fla.models.komp import KOMPConfig
from fla.models.komp.conversion import map_weights, optimizer
from fla.models.komp.legacy import LM as LegacyLM
from fla.models.komp.modeling_komp import TrainingLM
from fla.models.komp_value_decay import KOMPValueDecayConfig
from fla.models.komp_value_decay.modeling_komp_value_decay import ValueDecayTrainingLM


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["exact", "block"], required=True)
    parser.add_argument("--decay-axis", choices=["slot", "value"], default="slot")
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12970)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--init-checkpoint", type=Path, help="Trusted native model state or {model: state}; weights only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-geometry", action="store_true")
    parser.add_argument("--no-checkpoint-mixer", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.microbatch, args.save_every) <= 0 or args.warmup_steps < 0 or args.learning_rate <= 0:
        parser.error("Steps, microbatch, save interval and learning rate must be positive; warmup must be nonnegative")
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    return args


def main():
    args = arguments()
    config_class = KOMPConfig if args.decay_axis == "slot" else KOMPValueDecayConfig
    model_class = TrainingLM if args.decay_axis == "slot" else ValueDecayTrainingLM
    config = config_class(kind=args.kind, checkpoint_mixer=not args.no_checkpoint_mixer, cache_geometry=args.cache_geometry)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    batches = TokenBatches(args.tokens, args.sequence_length, args.global_batch, rank, world)
    if batches.local_batch % args.microbatch:
        raise ValueError("Per-rank batch must be divisible by microbatch")
    contract = dict(
        config=config.to_dict(),
        data_sha256=batches.fingerprint(),
        world_size=world,
        sequence_length=args.sequence_length,
        global_batch=args.global_batch,
        microbatch=args.microbatch,
        seed=args.seed,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        prune_gc=os.environ.get("EXACT_KOMP_PRUNE_GC", "0"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "latest.pt"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError("Output already has a checkpoint; use --resume or a new output directory")
    torch.manual_seed(args.seed)
    model = model_class(config)
    if not args.resume:
        if args.init_checkpoint:
            initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(initial.get("model", initial), strict=True)
        elif args.decay_axis == "slot":
            # Reproduce the frozen recipe's seeded initialization, including decay/beta biases.
            legacy = LegacyLM(args.kind, seed=args.seed)
            model.load_state_dict(map_weights(legacy.state_dict()), strict=True)
            del legacy
    model.cuda()
    opt = optimizer(model)
    step = 0
    if args.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["contract"] != contract:
            raise ValueError("Resume requires identical model, data, schedule, batching and world size")
        model.load_state_dict(saved["model"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        step = saved["step"]
        torch.set_rng_state(saved["rng"][rank]["cpu"])
        torch.cuda.set_rng_state(saved["rng"][rank]["cuda"])
        del saved
    if rank == 0:
        (args.output / "config.json").write_text(json.dumps(contract, indent=2) + "\n")
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    wrapped.train()
    start_step = step
    for step in range(start_step, args.steps):
        begin = time.perf_counter()
        inputs, labels = batches.batch(step)
        if labels.max() >= config.vocab_size or inputs.max() >= config.vocab_size:
            raise ValueError("Token ID outside model vocabulary")
        local_tokens = inputs.numel()
        lr_ratio = min(1.0, (step + 1) / max(1, args.warmup_steps))
        if step + 1 > args.warmup_steps:
            phase = (step + 1 - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr_ratio = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * phase))
        for group in opt.param_groups:
            group["lr"] = args.learning_rate * lr_ratio
        opt.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
        for start in range(0, batches.local_batch, args.microbatch):
            stop = start + args.microbatch
            sync = wrapped.no_sync() if world > 1 and stop < batches.local_batch else nullcontext()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                loss = wrapped(inputs[start:stop].cuda(), labels[start:stop].cuda())
                loss_sum += loss.detach().double()
                (loss / local_tokens).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if world > 1:
            dist.all_reduce(loss_sum)
        torch.cuda.synchronize()
        if rank == 0:
            row = dict(
                step=step + 1,
                supervised_tokens=(step + 1) * local_tokens * world,
                loss=float(loss_sum / (local_tokens * world)),
                lr=opt.param_groups[0]["lr"],
                grad_norm=float(norm),
                seconds=time.perf_counter() - begin,
            )
            print(json.dumps(row), flush=True)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng)
            else:
                states[0] = rng
            if rank == 0:
                temporary = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(
                    dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step + 1, contract=contract, rng=states),
                    temporary,
                )
                temporary.replace(checkpoint_path)
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
