# Iteration 1: OARBot whole-body PPO

**What you are watching:** copies of the same mobile robot practice in separate
arenas. Every copy has a table, blue cube, one orange obstacle, two RGB-D camera
devices, and its own episode. All copies contribute to **one shared policy**.
The policy controls the mobile base, six Kinova arm joints, and three fingers.

This is an OARBot adaptation of published methods, not an exact reproduction of
an arm-only paper. See [credits](CREDITS.md), [every setting](PARAMETERS.md),
[the implemented math](MATH.md), and [verification evidence](EVIDENCE.md).

## 1. What is the goal?

Each episode gives the robot one of two instructions:

| Instruction | Successful result |
| --- | --- |
| Lift | Raise the blue cube more than 8 cm, with at least two distinct fingers contacting it and the tool nearby, continuously for 0.5 simulated seconds. |
| Push | Move the cube within 2.5 cm of the visible green patch, keep it on the table, and settle below 0.08 m/s for 0.5 simulated seconds. |

Episodes end on success, failure, or a 20-second simulated time limit. A fresh
episode samples robot/cube/obstacle positions and physical/sensor variations.
Early training starts nearer the table, then widens starting positions. A shaping
reward encourages a base-to-cube distance near 0.7 m; this is not a separate
navigation planner. Collision penalties discourage impacts but do not guarantee
collision avoidance.

## 2. How does learning work?

1. Cameras render RGB and depth. Python registers color to depth and computes
   small image grids, sampled 3D points, and blue/green color features.
2. These features, joint encoders, previous actions, and the task instruction go
   into the **actor**, a neural network that proposes 12 control values.
3. Joint controllers and wheel kinematics apply the commands in simulated physics.
4. The simulator scores progress and stores observations, actions, and rewards.
5. After 64 control steps from every robot, PPO updates the actor and a **critic**
   network, which predicts future reward. The cycle repeats.

The networks already contain initialized weights; training changes their values,
not their number. The actor and critic are not separate base/gripper policies.
This implementation has no LLM controller, MoveIt motion planner, learned world
model, BPTT training, or latent safety filter.

The actor and critic receive sensor-derived observations. True cube/obstacle/base
world coordinates and contact forces are used for training scores, not as policy
inputs. An external viewer camera is only for the human watching.

## 3. Start on the existing SVR2 installation

The working server project is `/home/zalewm3/isaac_lab/iteration_1`. Connect the
RPI VPN, check the shared server reservation, then open **Windows PowerShell**:

```powershell
ssh zalewm3@svr2.cscms.rpi.edu
```

The following commands belong in the **Linux SSH terminal**:

```bash
cd /home/zalewm3/isaac_lab/iteration_1
python3 scripts/stop.py
bash scripts/train.sh --watch --num_envs 100
```

`stop.py` stops only tasks identified as belonging to this project. Wait for
`OARBOT_TRAINING_SCENE_READY`, then open the **Isaac Sim WebRTC Streaming Client**
on Windows and connect to `128.113.120.75`. `Learning iteration` messages confirm
PPO updates. No separate Isaac Lab viewer or manual USD opening is needed.

For a stationary preview instead of learning:

```bash
bash scripts/view.sh --num_envs 8
```

Stop a preview before starting training. When replacing the server application,
fully close and reopen the WebRTC client. The launcher fixes the stream at
1280×720, requests 30 FPS, and disables dynamic resizing; actual delivery can be
slower. It refuses to reuse an occupied streaming port.

## 4. Run overnight, stop, resume, or view a saved policy

The foreground command requires its SSH session to remain open. **Ctrl+C saves
the current weights and stops learning**, once the trainer is initialized.
For a run that survives closing SSH, use this instead:

```bash
python3 scripts/start_overnight.py --watch --num_envs 100 --hours 3 --max_iterations 1000
```

```bash
python3 scripts/status.py
python3 scripts/stop.py
bash scripts/train.sh --watch --num_envs 100 --resume
bash scripts/play.sh --num_envs 8
```

Run these commands sequentially as needed, not simultaneously. `--resume`
continues from the latest compatible checkpoint; `play.sh` runs a saved policy
on held-out cube positions without updating its weights. Without `--resume`,
training starts fresh. Record a specific run/checkpoint before changing settings.
Changing action/observation dimensions or the network structure requires a new
model or an explicit conversion.

Logs, TensorBoard events, `params/agent.yaml`, `params/env.yaml`, checkpoints, and
`episodes.csv` are stored under `logs/rsl_rl/oarbot_iteration_1/<run>/`.
Checkpoints save every 10 iterations and on normal completion/interruption.
Evaluation episode CSVs go in a separate `evaluation/` subfolder.

## 5. How long, and why is the viewer laggy?

The run stops at **1,000 PPO iterations or three hours of learning**, whichever
comes first. Loading time is additional. This is a compute budget, not a promise
of grasping success. The current run began learning around 4:56 PM Eastern on
November 23, 2026, so its time cap is approximately 7:56 PM Eastern if uninterrupted.

At 100 robots, one PPO iteration collects 6,400 transitions and runs 32 minibatch
optimizer steps. The first measured iteration took 16.13 seconds, giving an
initial estimate of 4.48 hours for all 1,000; the three-hour cap therefore matters.
Later iteration times vary with contacts, rendering, and the viewer.

There are **400 rendered optical views**: 100 robots × two devices × RGB/depth
optics. There are also many wheel-roller colliders. The first rollout simulated
1.067 seconds per robot in 15.538 real seconds, so simulated motion appeared
about 14.6 times slower than real time. A 30-FPS stream request cannot make the
simulation itself advance faster. Camera sensing is enabled even without a viewer.

## 6. Recreate from this GitHub folder

This folder contains the task, controls, sensors, PPO configuration, SDK-derived
training/playback scripts, launch/stop helpers, and asset export/setup scripts.
**The 18 MB robot USD, Isaac Sim/Lab installation, and 96 MB checkpoints are
external dependencies**, not bundled here. Model geometry retains its upstream
terms; this repository does not relicense the robot meshes.

On SVR2, clone into a separate directory while leaving the running project alone:

```bash
git clone https://github.com/space-explorer-4231/oarbot-sim-to-real.git
cd oarbot-sim-to-real/isaac_lab/iteration_1
python3 scripts/setup_assets.py
# Only after stopping the other viewer/trainer:
bash scripts/view.sh --num_envs 8
```

`setup_assets.py` copies and checks the existing server asset; it does not change
the original scene. Alternatively, with access to the saved full scene, run
`/opt/isaaclab/isaaclab.sh -p scripts/prepare_asset.py`. This exports `/oarbot` and
the saved optical calibration from `Iteration_1.usd`. On another machine, install
compatible SDK dependencies and provide that asset first; this is not a complete
cross-version installation script. The tested server uses Ubuntu 24.04/aarch64,
GB10, Isaac Sim 6.0.1rc7, an Isaac Lab 3.0 development checkout, and RSL-RL 5.0.1.

## 7. Where is the important code?

All task files below are under
`source/iteration_1/iteration_1/tasks/manager_based/iteration_1/`.

| File | Purpose |
| --- | --- |
| `iteration_1_env_cfg.py` | Assemble observations, actions, rewards, resets, termination rules, and timing. |
| `scene_cfg.py` | Create the robot/table/cube/obstacle arenas. |
| `robot_cfg.py`, `mobile_robot_cfg.py` | Load the USD; unfix the base; configure wheel, arm, and finger actuators. |
| `whole_body_actions.py` | Convert the actor's 12 outputs into wheel speeds and arm/finger targets. |
| `sensor_observations.py` | Build policy inputs from RGB/depth, static calibration, encoders, and instruction. |
| `task_logic.py`, `tool_frame.py` | Curriculum, randomized resets, success/failure tests, and reward calculations. |
| `agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters and actor/critic architecture. |

`scripts/rsl_rl/train.py` runs the RSL-RL trainer; `play.py` loads saved weights.
`scripts/check_setup.py` checks sensing and physical controls;
`scripts/verify_checkpoint.py` checks that finite weights actually changed.
The generated `mdp/` and example UI module are retained scaffold, not the active
whole-body reward implementation.

## Current limits

Short checks verify execution, not reliable task success. The current task has
not established real-robot transfer, reliable obstacle avoidance, or 1,000-camera
training. Factory Kinect intrinsics/distortion and hardware extrinsics remain
approximate. Finger-tip coupling, contact friction, and mass distributions are
models to validate. The camera feature extractor is fixed preprocessing feeding
an MLP, not an end-to-end trained vision encoder. [Evidence and measured camera
fields of view](EVIDENCE.md) make those distinctions explicit.
