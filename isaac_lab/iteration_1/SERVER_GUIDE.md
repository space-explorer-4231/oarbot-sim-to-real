# OARBot: Iteration 1 â€” camera-based whole-body PPO

This external Isaac Lab project trains one policy to drive the mobile base,
move the Kinova arm, and close three fingers. Each arena has a table, blue cube,
and one obstacle. An instruction selects lifting or pushing towards a visible
green patch. All robot copies contribute experience to the same policy.

## Start from a Windows PC

1. Connect the RPI VPN and check the shared server reservation.
2. Open PowerShell and run `ssh zalewm3@svr2.cscms.rpi.edu`.
3. Run the following commands **inside that SSH terminal**.

```bash
cd /home/zalewm3/isaac_lab/iteration_1
bash scripts/view.sh --num_envs 8
```

Open **Isaac Sim WebRTC Streaming Client** on Windows, connect to
**128.113.120.75**, and wait for `OARBOT_PREVIEW_READY` in the terminal. The
preview holds still and resets episodes; it does not learn. Isaac Lab uses the
Isaac Sim viewerâ€”there is no separate Lab viewer application.

## Train while watching

Stop the preview with **Ctrl+C**, then:

```bash
cd /home/zalewm3/isaac_lab/iteration_1
bash scripts/train.sh --watch --num_envs 101
```

Disconnect/reconnect the WebRTC client after the new application starts.
`Learning iteration ...` lines confirm PPO updates. Early movements are
exploration, not a trained skill. Keep this SSH terminal open for this foreground
run. Ctrl+C saves an interrupted checkpoint and stops. To change a setting,
stop the run, edit the configuration, and restart; compatible checkpoints can
be resumed. Changing the action/observation/network shapes requires a new model.

A small smoke run, rather than a full training session:

```bash
bash scripts/train.sh --watch --num_envs 8 --max_iterations 3 --run_name viewer_check
```

The SDK Python environment is selected by the wrapper; no Windows `conda
activate` or Linux `env` command needs to be run in Windows PowerShell.
Do not run the old standalone `start-oarbot.sh` alongside this viewer on the same
streaming port. Closing the WebRTC window alone does not stop the server job.

## Background training, playback, and stopping

Stop any preview/foreground trainer first. This command survives an SSH disconnect:

```bash
python3 scripts/start_overnight.py --num_envs 101 --hours 3 --max_iterations 1000
```

Add `--watch` to that command to keep the live WebRTC viewer available during
background training, including after closing SSH.

The time limit applies to learning, after application loading. The run ends at
the iteration limit or time limit, whichever occurs first. Cameras still render
when the WebRTC viewer is disabled.

```bash
python3 scripts/status.py
python3 scripts/stop.py
bash scripts/play.sh --num_envs 8
```

`play.sh` loads the latest saved checkpoint on held-out cube positions. Playback
runs the learned weights without updating them. To resume a particular run:

```bash
bash scripts/train.sh --watch --num_envs 101 --resume --load_run RUN_FOLDER --checkpoint model_20.pt
```

Replace `RUN_FOLDER` with a directory name under `logs/rsl_rl/oarbot_iteration_1`.
Checkpoints save every 10 updates and at completion/interruption. TensorBoard
events and `episodes.csv` live in the same run directory. Episode records flag
the initial randomized-length warm starts; exclude those from success-rate
measurements. Checkpoints and short setup runs are not evidence of reliable
grasping or safe physical deployment.

## What is implemented

- **12 actions:** body forward/sideways/turn, six arm target increments, three
  finger closures. Wheel kinematics map body commands to all four mecanum wheels;
  36 rollers remain passive. Finger-tip targets are coupled to proximal closure.
- **Actor input:** two camera devices with separate RGB/depth optical channels,
  depth point samples, color features, motor encoders, previous actions, task
  instruction, and four observations of history. No true object/base/obstacle
  world coordinates or contact forces enter the policy or critic. Geometry and
  filtered cube-contact forces are used only for reward/scoring.
- **Randomization:** floor friction 0.3â€“1.2, arm calibration offsets Â±0.01 rad,
  encoder/RGB/depth noise, missing depth pixels, randomized robot/cube/obstacle
  positions. Curriculum widens starting positions as experience accumulates.
- **Task:** approach, reach, close fingers, lift the cube 8 cm with two distinct
  fingers contacting it for 0.5 s; or push to within 2.5 cm of a visible patch
  and settle for 0.5 s. Collisions, falling, and leaving the arena incur penalties.
- **Timing:** 240 Hz physics, 60 Hz control, 15 Hz camera rendering. These are
  simulation rates; actual wall-clock performance depends on the run.

## PPO and literature

[Kim et al., DOI 10.1007/s12541-026-01595-5](https://doi.org/10.1007/s12541-026-01595-5)
provides the Table 3 PPO baseline: discount 0.99, GAE 0.95, initial learning rate
0.001, clipping 0.2, entropy coefficient 0.01, 8 epochs, 4 minibatches, gradient
norm limit 1, 64 steps per environment, and 1,000 updates.

With 101 environments, each update collects 6,464 transitions. Our actor and
critic each use hidden layers 256/128/64 with ELU, observation normalization,
and an initial actor Gaussian standard deviation of 0.35. The learning rate is
adapted using a desired KL of 0.01. RSL-RL is our implementation choice: the
paper does not identify its exact RL library. Whole-body control, visual input,
network architecture, pushing, obstacle, and curriculum are our adaptations;
the paper's arm-reaching task is not reproduced unchanged.

This project implements PPO. SuperMimic-style world-model/BPTT training and the
latent safety filter are future work, not included in these scripts.

## Files and limits

- `source/iteration_1/iteration_1/tasks/manager_based/iteration_1/iteration_1_env_cfg.py`:
  scene, sensors, timing, events, rewards, termination settings.
- `whole_body_actions.py`, `sensor_observations.py`, `task_logic.py` in that folder:
  controls, camera/internal observations, and scoring/resets.
- `agents/rsl_rl_ppo_cfg.py` in that folder: PPO and neural-network configuration.
- `assets/oarbot_iteration_1.usdc`: isolated robot exported from the user's saved
  `/home/zalewm3/nursebot_sim/ai_bridge/scenes/Iteration_1.usd`; source is preserved.
- `assets/sensor_mounts.json`: copied static optical calibration and limitations.
- `verification/asset_export.json`: provenance/hashes; `scripts/check_setup.py`:
  finite-state, sensor visibility, conservative driving, and observation-source checks.

Training uses reduced RGB 320Ã—180 (room), 160Ã—90 (wrist), depth 160Ã—144 (room),
128Ã—128 (wrist). Separate RGB/depth frusta and registration are retained. On this
SDK, both RGB and depth render outputs are enabled at each optical channel to
avoid black RGB buffers; observations select the appropriate channel. The camera
model uses ideal pinholes, approximate depth noise, and the saved scene's mounts;
factory distortion and measured Kinect RGB/depth baseline are not calibrated.
The installed renderer assumes square pixels and ignores vertical aperture:
the measured room depth frustum is 75Â°Ã—69.26Â° at 160Ã—144, rather than the Kinect's
nominal 75Â°Ã—65Â°. Wrist depth is 120Â°Ã—120Â°, and RGB is about 90Â°Ã—58.72Â°. Matching
factory intrinsics/distortion exactly requires a later calibrated camera model.
Mass distribution, friction, coupled fingertip mechanics, and simulated contact
remain approximations. No MoveIt/ROS hardware driver is connected by these runs.
1,000 camera-equipped environments and real-world transfer require separate
validation. Start with the verified environment count before increasing it.
