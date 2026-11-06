"""Run the added native GSA2 arm with the same search and scale-up budget."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'gsa2_runs'
DEST.mkdir(parents=True, exist_ok=True)
COMMAND = [sys.executable, str(ROOT / 'gsa2_experiment.py')]


def run(args, tag):
    (DEST / 'status.json').write_text(json.dumps({'stage': tag, 'status': 'running'}, indent=2))
    with (DEST / (tag + '.log')).open('x') as log:
        subprocess.run(COMMAND + args, check=True, stdout=log, stderr=subprocess.STDOUT)
    print('COMPLETE', tag, flush=True)


def main():
    run(['--action', 'preflight'], 'preflight')
    for lr in (2.5e-4, 5e-4, 1e-3):
        for wd in (0., .01, .1):
            run(['--action', 'train', '--lr', str(lr), '--wd', str(wd)], f'search-lr{lr:g}-wd{wd:g}')
    run(['--action', 'select'], 'selection')
    best = json.loads((DEST / 'selection.json').read_text())['models']['gsa2']['selected']
    selected = ['--lr', str(best['lr']), '--wd', str(best['wd'])]
    run(['--action', 'train', '--seed', '732'] + selected, 'replication')
    run(['--action', 'evaluate', '--scale', 'small'], 'small-heldout')
    run(['--action', 'train', '--scale', 'large'] + selected, 'large')
    run(['--action', 'evaluate', '--scale', 'large'], 'large-heldout')
    (DEST / 'completed.json').write_text(json.dumps({'status': 'completed', 'runs': 11, 'selected': best}, indent=2))
    (DEST / 'status.json').write_text(json.dumps({'stage': 'complete', 'status': 'completed'}, indent=2))
    print('ALL GSA2 RUNS COMPLETED', flush=True)


if __name__ == '__main__':
    main()
