"""Build a four-arm report from separately frozen selections and evidence."""
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'comparison_runs'
DEST.mkdir(exist_ok=True)
combined = {'criterion': 'minimum final full validation mean content CE at seed731', 'models': {}, 'source_selections': {}}
for study in ('lrwd_runs', 'gsa2_runs'):
    source = ROOT / study
    path = source / 'selection.json'
    selection = json.loads(path.read_text())
    assert not (combined['models'].keys() & selection['models'].keys())
    combined['models'].update(selection['models'])
    combined['source_selections'][study] = hashlib.sha256(path.read_bytes()).hexdigest()
    for trial in (source / 'results').iterdir():
        if not trial.is_dir():
            continue
        target = DEST / 'results' / trial.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.symlink_to(trial.resolve(), target_is_directory=True)
        else:
            assert target.resolve() == trial.resolve()
(DEST / 'selection.json').write_text(json.dumps(combined, indent=2))
subprocess.run([sys.executable, str(ROOT / 'lrwd_report.py')], check=True,
               env=dict(os.environ, STUDY_REPORT_ROOT=str(DEST), STUDY_REPORT_KINDS='komp_slot,gsa,gdn,gsa2'))
