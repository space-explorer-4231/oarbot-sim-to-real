"""Locate only this user's Python simulation tasks belonging to this project."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {ROOT/'scripts'/name for name in (
    'view.py', 'check_setup.py', 'evaluate.py', 'rsl_rl/train.py', 'rsl_rl/play.py')}

def active_tasks():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try:
            if entry.stat().st_uid != os.getuid(): continue
            args = (entry/'cmdline').read_bytes().split(b'\0')
            if len(args)<2 or not Path(args[0].decode()).name.startswith('python'): continue
            # isaaclab.sh may use `python -c ... -p SCRIPT`, rather than `python SCRIPT`.
            for argument in args[1:]:
                value = argument.decode(errors='replace')
                if not value.endswith('.py'): continue
                path = Path(value)
                if not path.is_absolute(): path = (entry/'cwd').resolve()/path
                path = path.resolve()
                if path in ALLOWED:
                    found.append((int(entry.name), str(path)))
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    # The CLI parent waits for its Python child; interrupt only the child so it
    # can finish saving without a second signal forwarded by the parent.
    matched = {pid for pid, _ in found}
    parents = set()
    for pid in matched:
        try:
            ancestor = pid
            for _ in range(64):
                lines = (Path('/proc')/str(ancestor)/'status').read_text().splitlines()
                ancestor = int(next(line.split()[1] for line in lines if line.startswith('PPid:')))
                if ancestor <= 1: break
                parents.add(ancestor)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return [(pid, path) for pid, path in found if pid not in parents]

if __name__ == '__main__':
    tasks = active_tasks()
    if '--guard' in sys.argv and tasks:
        raise SystemExit('An Iteration 1 task is already running: '+str(tasks)+
                         '. Use python3 scripts/stop.py before starting another.')
    if '--guard' not in sys.argv:
        print('Active Iteration 1 tasks:', tasks or 'none')
