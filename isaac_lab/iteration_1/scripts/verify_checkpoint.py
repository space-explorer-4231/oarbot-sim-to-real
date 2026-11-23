"""Verify a short PPO run saved finite network weights and changed them during updates."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
first = torch.load(args.run/'model_0.pt', map_location='cpu', weights_only=False)
last = torch.load(args.run/'model_2.pt', map_location='cpu', weights_only=False)
result = dict(run=str(args.run), initial_saved_iteration=first['iter'], final_saved_iteration=last['iter'])
for model in ('actor','critic'):
    original = first[model+'_state_dict']
    current = last[model+'_state_dict']
    assert all(torch.isfinite(v).all() for v in current.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
    weight_keys = [k for k,v in current.items() if isinstance(v,torch.Tensor) and k.endswith('weight')]
    assert weight_keys, 'No neural-network layer weights found'
    changed = sum(int(torch.count_nonzero(current[k]-original[k])) for k in weight_keys)
    assert changed > 0, f'{model} network did not update'
    result[model] = dict(finite=True, changed_layer_weight_values=changed)
root = Path(__file__).resolve().parents[1]
(root/'verification/checkpoint_check.json').write_text(json.dumps(result,indent=2)+'\n')
print('CHECKPOINT_CHECK_PASS '+json.dumps(result))
