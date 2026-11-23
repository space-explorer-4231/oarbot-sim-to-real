"""Gracefully stop only this project's tasks; train.py saves on SIGINT."""
import os
import signal
import time
from pathlib import Path
from processes import active_tasks

tasks = active_tasks()
if not tasks: print('No Iteration 1 simulation or trainer is running.')
for pid, path in tasks:
    try:
        os.kill(pid, signal.SIGINT)
        print(f'Requested stop: {pid} {path}')
    except ProcessLookupError:
        continue
deadline = time.monotonic()+15
while active_tasks() and time.monotonic()<deadline:
    time.sleep(.25)
remaining = active_tasks()
if remaining:
    print('Still shutting down:', remaining)
    raise SystemExit(1)
print('Iteration 1 tasks stopped.')
