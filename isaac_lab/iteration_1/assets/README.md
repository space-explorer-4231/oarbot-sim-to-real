# External robot asset

`oarbot_iteration_1.usdc` is an 18,007,950-byte isolated robot export, not included
in the public source-code folder. SHA-256:

```text
3c7d378de54fed97a7267bb8032d5c3fd4be1a5f952dc0b6eec0e650cb28bc8f
```

On SVR2, run `python3 scripts/setup_assets.py` from the project root to copy the
tested asset from `/home/zalewm3/isaac_lab/iteration_1/assets/` into a GitHub clone.
The included `sensor_mounts.json` contains the corresponding saved static optical
configuration. Alternatively, `prepare_asset.py` exports the robot and calibration
from the author's saved `/home/zalewm3/nursebot_sim/ai_bridge/scenes/Iteration_1.usd`.
Neither command downloads or installs Isaac Sim/Lab. Hardware meshes and SDK
components remain subject to their original terms; see [credits](../CREDITS.md).
