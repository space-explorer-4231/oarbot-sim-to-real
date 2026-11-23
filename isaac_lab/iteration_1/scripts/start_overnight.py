"""Start bounded training detached from SSH, with a log and launch record."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from processes import ROOT, active_tasks

parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=101)
parser.add_argument('--max_iterations', type=int, default=1000)
parser.add_argument('--hours', type=float, default=3.)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--watch', action='store_true', help='Enable the live WebRTC viewer while background training runs.')
args = parser.parse_args()
if args.num_envs < 1 or args.max_iterations < 1 or args.hours <= 0:
    raise SystemExit('Environment count, iterations, and hours must be positive.')
if active_tasks(): raise SystemExit('An Iteration 1 task is already running; stop it first.')
folder = ROOT/'logs/launches'; folder.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log = folder/f'overnight_{stamp}.log'
command = ['bash', str(ROOT/'scripts/train.sh'), '--num_envs', str(args.num_envs),
    '--max_iterations', str(args.max_iterations), '--hours', str(args.hours),
    '--seed', str(args.seed), '--run_name', 'overnight', '--logger', 'tensorboard']
if args.watch:
    command.append('--watch')
with log.open('w') as output:
    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
time.sleep(2)
if child.poll() is not None:
    raise SystemExit(f'Training launcher exited early. Read {log}')
record = dict(pid=child.pid, command=command, log=str(log), started_local=datetime.now().isoformat(),
              maximum_training_hours=args.hours, maximum_iterations=args.max_iterations,
              num_envs=args.num_envs, long_run_success_not_yet_measured=True)
record['live_viewer'] = args.watch
(folder/'last_launch.json').write_text(json.dumps(record, indent=2)+'\n')
print('Background training launched. You can close SSH.')
print('Stops after the iteration limit or training-time limit, whichever comes first.')
print('Log:', log)
if args.watch:
    print('Watch: open Isaac Sim WebRTC Streaming Client and connect to 128.113.120.75.')
print('Status: python3 scripts/status.py')
print('Stop and save: python3 scripts/stop.py')
