"""Copy the existing SVR2 robot asset into a clone without changing its source."""
import argparse
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '3c7d378de54fed97a7267bb8032d5c3fd4be1a5f952dc0b6eec0e650cb28bc8f'
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=Path(
    '/home/zalewm3/isaac_lab/iteration_1/assets/oarbot_iteration_1.usdc'))
args = parser.parse_args()
source = args.source.expanduser().resolve()
destination = ROOT/'assets/oarbot_iteration_1.usdc'
if not source.is_file():
    raise SystemExit('Provide the exported robot with --source PATH, or use prepare_asset.py with access to the saved scene.')
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if digest != EXPECTED:
    raise SystemExit('Asset differs from the tested snapshot. Keep it separate and validate its calibration/configuration before using it.')
destination.parent.mkdir(exist_ok=True)
if destination.resolve() != source:
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise SystemExit('Destination exists with different content; refusing to overwrite it.')
    else:
        shutil.copy2(source, destination)
print('Asset ready:', destination)
print('SHA-256:', digest)
