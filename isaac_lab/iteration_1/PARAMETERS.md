# Current configuration and where it came from

The saved `params/agent.yaml` and `params/env.yaml` in a run are authoritative
for that run. Editing a source file does not change an already running process.

## PPO baseline

These starting values follow **Table 3 of Kim et al. (2026)**
([paper](https://doi.org/10.1007/s12541-026-01595-5)). The whole-body task and
network below are OARBot adaptations, not the authors' original experiment.

| Parameter | Value | Plain meaning |
| --- | ---: | --- |
| Discount factor, γ | 0.99 | How future reward is discounted. |
| GAE factor, λ | 0.95 | How advantage estimates mix shorter and longer prediction errors. |
| Initial learning rate | 0.001 | Starting Adam optimizer step size; adapts during training. |
| PPO clipping, ε | 0.2 | Clip the policy probability ratio in the surrogate objective. |
| Entropy coefficient | 0.01 | Encourage exploration. |
| Learning epochs | 8 | Reuse each rollout batch for eight passes. |
| Minibatches per epoch | 4 | Split the rollout into four parts per pass. |
| Maximum gradient norm | 1.0 | Rescale gradients that exceed this global norm. |
| Steps per environment | 64 | Collect 64 control transitions per robot before updating. |
| Maximum iterations | 1,000 | Maximum rollout/update cycles per launch. |

## Additional implementation choices

| Setting | Value |
| --- | --- |
| RL implementation | RSL-RL 5.0.1, PyTorch; our choice, not an exact library attributed to Kim et al. |
| Actor | MLP 16,056 → 256 → 128 → 64 → 12, ELU hidden activations |
| Critic | MLP 16,056 → 256 → 128 → 64 → 1, ELU hidden activations |
| Observation normalization | Enabled for both networks |
| Initial Gaussian action standard deviation | 0.35, learnable; state-independent distribution parameters |
| Action clipping | Each normalized action clipped to [−1, 1] for physical commands |
| Optimizer | Adam |
| Value loss coefficient | 1.0, clipped value loss enabled |
| KL target / learning-rate schedule | 0.01 / adaptive |
| Advantage normalization | Whole rollout, not separately inside each minibatch |
| Seed | 42 for training; evaluation environment default 1042 |
| Checkpoint interval | Every 10 iterations, plus completion/interruption |
| Default environment count in code | 101; current user run explicitly selected 100 |
| Learning-time budget | 3 hours by default; configurable with `--hours` |

At 100 environments: batch = 100 × 64 = **6,400** transitions; minibatch =
6,400 / 4 = **1,600**; optimizer steps per iteration = 8 × 4 = **32**.
One thousand iterations would collect 6.4 million transitions, if the time cap
does not stop the run first.

## Robot, sensor, and task settings

| Setting | Value |
| --- | --- |
| Physics / policy control / camera update | 240 / 60 / 15 Hz in simulated time |
| Episode limit | 20 simulated seconds = 1,200 control steps |
| Observation history | Current plus three previous control observations; repeated camera images are possible |
| Maximum commanded base x/y speed | ±0.20 m/s on each axis; diagonal magnitude can exceed 0.20 m/s |
| Maximum commanded yaw speed | ±0.40 rad/s |
| Arm command speed scale | 0.5 × [0.628, 0.628, 0.628, 0.838, 0.838, 0.838] rad/s |
| Proximal/distal finger closure targets | 1.45 / 0.40 rad at full normalized closure |
| Wheel radius / rotational lever parameter | 0.049 / 0.4954 m; imported joint sign convention is preserved |
| Floor static/dynamic friction | Uniform [0.3, 1.2], 32 material buckets; dynamic ≤ static |
| Arm calibration bias | Uniform [−0.01, 0.01] rad per joint, per episode |
| Encoder angle noise | Gaussian σ = 0.001 rad |
| RGB noise | Gaussian σ = 0.005 on normalized [0, 1] intensity |
| Depth noise | Gaussian σ(z) = 0.005 + 0.001z² metres, z in metres |
| Missing depth probability | 0.02 per pixel/sample |
| Room RGB / depth image | 320×180 / 160×144 |
| Wrist RGB / depth image | 160×90 / 128×128 |
| Accepted room / wrist depth | 0.5–3.86 / 0.25–2.88 m |
| Lift success | >8 cm lift, ≥2 fingers touching cube, tool within 12 cm of shaped target, hold 0.5 s |
| Push success | <2.5 cm XY target error, cube stays on table, speed <0.08 m/s, hold 0.5 s |
| Environment spacing | 5.2 m |
| Streaming | 1280×720, target 30 FPS, dynamic resizing disabled |

Friction and calibration ranges are inspired by SuperMimic's domain-randomization
table. Their application to a mecanum mobile manipulator, other sensor-noise
values, network architecture, curriculum, grasp/push criteria, and action scaling
are this project's choices. No physically measured noise distribution is claimed.

## Curriculum

The implementation computes total experience as `common_step_counter × num_envs`.
It uses a reset function rather than named Curriculum Manager terms, so the SDK
can print **“0 active curriculum terms”** while these stages are still in use.

| Experience collected | Stage | Initial robot positions | Push instruction probability |
| --- | ---: | --- | ---: |
| <250,000 transitions | 0 | x = 0.60–0.82 m, y ±0.15 m, yaw ±0.15 rad | 10% |
| 250,000–999,999 | 1 | x = −0.15–0.65 m, y ±0.75 m, yaw ±0.50 rad | 25% |
| ≥1,000,000 | 2 | Same broader starts; obstacle can come closer to the route | 50% |

Training cube centers: x = 1.50–1.72 m, y ±0.12 m. Held-out evaluation uses
x = 1.76–1.84 m, y ±0.18 m. Push target offsets are +0.12 m in training and
−0.12 m in evaluation to keep the marker on the table. Evaluation is therefore
a deliberate position/direction generalization test, not an identical reset distribution.
