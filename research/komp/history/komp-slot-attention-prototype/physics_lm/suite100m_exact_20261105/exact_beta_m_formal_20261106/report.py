"""Audit the 2x2 beta/M comparison without using heldout scores to select cells."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'snapshot/suite100m'))
from tasks import Factory


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan, manifest = read(ROOT / 'plan.json'), read(ROOT / 'source-manifest.json')
for name, wanted in manifest['files'].items():
    assert sha(ROOT / name) == wanted, name
binding = read(ROOT / 'evidence/baseline-reuse.json')
rows = []
for task in plan['tasks']:
    baseline_entry = binding['tasks'][task]
    baseline = Path(baseline_entry['directory'])
    if not baseline.exists():
        baseline = ROOT.parent / 'exact_alpha995_lr_formal_20261105/runs-ddp/results' / baseline.name
    for name, wanted in baseline_entry['artifacts'].items():
        assert sha(baseline / name) == wanted, (task, name)
    original = read(baseline / 'summary.json')['config']
    for candidate in [plan['reused_candidate'], *plan['new_candidates']]:
        beta, slots = candidate['beta0'], candidate['slots']
        reused = candidate == plan['reused_candidate']
        variant = ROOT / 'variants' / f'{task}-beta{beta:g}-m{slots}'
        name = f'{task}-exact-beta{beta:g}-m{slots}-final-lr{plan["lr"][task]:g}-wd0.1-seed731'
        directory = baseline if reused else variant / 'runs-ddp/results' / name
        row = dict(task=task, beta0=beta, slots=slots, lr=plan['lr'][task], reused=reused,
                   parameters=plan['parameters_by_slots'][str(slots)], training='pending', heldout='pending')
        if (directory / 'summary.json').exists():
            summary = read(directory / 'summary.json')
            cfg = summary['config']
            assert summary['status'] == 'completed' and summary['tokens'] == plan['positions_per_run'][task]
            for key in ('task', 'phase', 'seed', 'lr', 'wd', 'batch', 'world_size', 'steps', 'data_order_sha256'):
                assert cfg[key] == original[key], (task, beta, slots, key)
            for key in ('width', 'layers', 'ff', 'heads', 'head_dim', 'exact_compute_chunk', 'projection_dtype', 'eviction', 'alpha0'):
                assert cfg['model'][key] == original['model'][key], (task, beta, slots, key)
            assert cfg['model']['parameters'] == row['parameters'] and cfg['model']['block_slots'] == slots
            if not reused:
                assert cfg['model']['beta0'] == beta
                assert cfg['sources'] == dict(snapshot=manifest['files']['evidence/snapshot-sha256.json'],
                                              ddp_train=manifest['files']['ddp_train.py'],
                                              beta_m_models=manifest['files']['beta_m_models.py'])
            metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            assert metrics[-1]['step'] == cfg['steps'] and metrics[-1]['tokens'] == summary['tokens']
            assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in metrics)
            row.update(training='completed', validation_ce=summary['val']['ce'], positions=summary['tokens'],
                       train_seconds_including_cold_compile_and_final_validation=summary['train_seconds'],
                       peak_bytes_rank0=summary['peak_bytes_rank0'], summary_sha256=sha(directory / 'summary.json'))
            warm = [x for x in metrics if x['step'] >= 25]
            if len(warm) >= 2:
                row['observed_training_positions_per_second_after_step25'] = (warm[-1]['tokens'] - warm[0]['tokens']) / (warm[-1]['seconds'] - warm[0]['seconds'])
        elif (directory / 'metrics.jsonl').exists():
            lines = (directory / 'metrics.jsonl').read_text().splitlines()
            if lines:
                row.update(training='running', latest=json.loads(lines[-1]))
        if (directory / 'eval-progress.json').exists():
            progress = read(directory / 'eval-progress.json')
            row.update(heldout='running', test_ce=progress['test']['ce'], generation_probes=len(progress['records']))
        if (directory / 'heldout.json').exists():
            heldout, records = read(directory / 'heldout.json'), read(directory / 'generations.json')
            grades = [bool(Factory(task).grade(x['probe'], x['guess'])) for x in records]
            assert len(grades) == 256 and all(a == bool(r['valid']) for a, r in zip(grades, records))
            assert sum(grades) / 256 == heldout['generation_accuracy']
            row.update(heldout='completed', test_ce=heldout['test']['ce'], generation_correct=sum(grades), generation_accuracy=sum(grades) / 256, heldout_seconds_including_cold_compile=heldout['seconds'])
        benchmark_dir = ROOT / 'variants' / f'{task}-beta0.1-m32' if reused else variant
        benchmark = benchmark_dir / f'evidence/benchmark-beta{beta:g}-m{slots}.json'
        if benchmark.exists():
            row['warm_benchmark'] = read(benchmark)
        rows.append(row)
winners = {}
for task in plan['tasks']:
    candidates = [r for r in rows if r['task'] == task]
    if all(r['training'] == 'completed' for r in candidates):
        winner = min(candidates, key=lambda r: r['validation_ce'])
        winners[task] = {k: winner[k] for k in ('beta0', 'slots', 'validation_ce')}
report = dict(new_training_complete=sum(not r['reused'] and r['training'] == 'completed' for r in rows),
              new_heldout_complete=sum(not r['reused'] and r['heldout'] == 'completed' for r in rows),
              expected_new=6, reused_baselines=2, runs=rows, validation_only_winners=winners,
              limitation=plan['precision_limit'], scope='One seed; fixed backbone (M64 adds parameters); fixed best prior LR; warm synthetic fwd/bwd and observed training throughput (which may still include shape compilation) reported separately.')
(ROOT / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['# Exact beta × M formal comparison', '',
         f'New training {report["new_training_complete"]}/6; new heldout {report["new_heldout_complete"]}/6; completed baselines reused 2/2.', '',
         '| Task | beta0 | M | Parameters | Validation CE | Test CE | Generation |',
         '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    val = f'{row["validation_ce"]:.6f}' if 'validation_ce' in row else row['training']
    test = f'{row["test_ce"]:.6f}' if 'test_ce' in row else 'pending'
    gen = f'{row["generation_correct"]}/256' if 'generation_correct' in row else 'pending'
    lines.append(f'| {row["task"]} | {row["beta0"]} | {row["slots"]} | {row["parameters"]:,} | {val} | {test} | {gen} |')
lines += ['', 'Alpha .995; FP32 projection/state; BF16 network; seed 731; WD .1. Depo2 LR .001 and Brevo2 LR .002. Width512 / 15 layers / FF1984 remain fixed. Only final validation CE selects cells. M64 is 102.74M parameters, compared with M32 at 100.53M.', '', plan['precision_limit']]
lines += ['', '## Measured cost', '',
          'Warm benchmark: one GPU, batch1 × 1024 tokens, full-model forward/backward with activation checkpointing, no optimizer/DDP. Training wall time includes cold shape compilation and final validation; rank0 training peak includes optimizer state. These are different measurements.', '',
          '| Task | beta0 | M | Training wall minutes | Training peak GiB | Warm fwd+bwd ms | Warm peak GiB | Heldout wall minutes |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    wall = f'{row["train_seconds_including_cold_compile_and_final_validation"] / 60:.2f}' if 'train_seconds_including_cold_compile_and_final_validation' in row else 'pending'
    peak = f'{row["peak_bytes_rank0"] / 2**30:.3f}' if 'peak_bytes_rank0' in row else 'pending'
    benchmark = row.get('warm_benchmark', {})
    milliseconds = f'{benchmark["median_ms"]:.2f}' if benchmark else 'pending'
    bench_peak = f'{benchmark["peak_allocated_bytes"] / 2**30:.3f}' if benchmark else 'pending'
    eval_wall = f'{row["heldout_seconds_including_cold_compile"] / 60:.2f}' if 'heldout_seconds_including_cold_compile' in row else 'pending'
    lines.append(f'| {row["task"]} | {row["beta0"]} | {row["slots"]} | {wall} | {peak} | {milliseconds} | {bench_peak} | {eval_wall} |')
lines += ['', 'The first M64 candidate pays new-shape compilation; the later M64 candidate reuses the shared cache. Native unpadded generation also pays per-length M64 compilation: heldout wall time was about44minutes on Depo2 and111minutes on Brevo2. These are observed cold-cache evaluation costs, not steady-state decoding throughput. Wall-time differences are not pure arithmetic comparisons. M64 doubles dictionary key/coefficient state and quadruples M×M map storage; fixed-backbone parameters rise by 2,215,680 (+2.20%).', '', f'Validation-only winners after all four cells finish: {winners}.']
(ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'runs'}))
