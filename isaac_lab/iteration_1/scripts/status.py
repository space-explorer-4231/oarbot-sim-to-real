"""Read training status without opening Isaac Sim or using the GPU."""
import json
from pathlib import Path
from processes import ROOT, active_tasks

print('Active tasks:', active_tasks() or 'none')
record = ROOT/'logs/launches/last_launch.json'
if record.exists():
    launch=json.loads(record.read_text())
    print('Last background launch:', launch)
    log=Path(launch['log'])
    if log.exists():
        print('\nLatest console output:')
        print('\n'.join(log.read_text(errors='replace').splitlines()[-15:]))
runs=ROOT/'logs/rsl_rl/oarbot_iteration_1'
if runs.exists():
    paths=sorted((p for p in runs.iterdir() if p.is_dir()), key=lambda p:p.stat().st_mtime)
    if paths:
        latest=paths[-1]
        checkpoints=sorted(latest.glob('model_*.pt'), key=lambda p:p.stat().st_mtime)
        print('Latest run:',latest)
        print('Latest checkpoint:',checkpoints[-1] if checkpoints else 'not saved yet')
        print('Episode measurements:',latest/'episodes.csv')
