# OARBot Sim-to-Real Learning: PPO, World Models, and Safety

This project studies how to train the OARBot mobile manipulator for tabletop manipulation and evaluate transfer from simulation to physical hardware. OARBot combines a Clearpath Dingo base with a six-joint Kinova Jaco arm.

The current Isaac Lab task focuses on reaching a pre-grasp pose above a tabletop cube with the base held stationary. The literature below guides the existing PPO setup and the proposed world-model and safety experiments.

## Literature

### 1. Sim-to-Real Transfer of Deep Reinforcement Learning for Robotic Pick-and-Place Tasks

Kim, M., Yuk, D.-G., Lee, J., & Sohn, J. W. (2026). *Sim-to-Real Transfer of Deep Reinforcement Learning for Robotic Pick-and-Place Tasks*. International Journal of Precision Engineering and Manufacturing, 27, 3745–3763.

[Read the paper](https://doi.org/10.1007/s12541-026-01595-5)

- The authors trained a reaching policy using Proximal Policy Optimization (PPO) in Isaac Lab and Isaac Sim. Randomized target positions provided practice across different locations.
- Lightweight color segmentation and RGB-D depth estimated the target's 3D position. The learned policy controlled arm reaching, with separate logic handling the gripper.
- Their physical-robot experiments reported 100% success in the fixed static test, 93.3% in the dynamic test, and 60% in the multi-object test. These are the authors' results, not OARBot results.

**Relevance to OARBot:** Provides the starting approach for randomized reaching and PPO joint-position control. Table 3 informed the initial PPO settings; evaluation will measure reach success, tool-position error, and time to reach on held-out cube positions.

### 2. SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model

Liao, Q., Shao, Y., Long, J., Zakka, K., Horowitz, R., Abbeel, P., & Sreenath, K. (2026). *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*. Conference on Robot Learning (CoRL 2026).

[Read the paper](https://arxiv.org/abs/2610.28480) · [Project and demonstrations](https://supermimic.github.io/) · [Previously uploaded PDF](https://github.com/user-attachments/files/35247318/Super.Mimic.Paper.pdf)

- A differentiable world model learns robot dynamics from simulator transitions. The policy improves through short imagined rollouts using backpropagation through time (BPTT).
- Simulator data collection, world-model fitting, and policy improvement alternate. Gradients pass through the learned model rather than the physics simulator.
- The authors demonstrated humanoid motion tracking and transfer to a physical Unitree G1. Their experiments use MuJoCo Warp; this is not an existing OARBot or Isaac Lab implementation.

**Relevance to OARBot:** Motivates a future comparison between model-free PPO and policy learning through a world model trained on OARBot transitions. Prediction accuracy and training time must be measured on our task and hardware.

### 3. Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis

Nakamura, K., Peters, L., & Bajcsy, A. (2025). *Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis*. Robotics: Science and Systems (RSS 2025).

[Read the paper](https://arxiv.org/abs/2502.00935) · [Project, code, and demonstrations](https://kensukenk.github.io/latent-safety/) · [Previously uploaded PDF](https://github.com/user-attachments/files/35247323/2000.demention.safty.fetures.pdf)

- Latent Safety Filters use a learned world model and approximate Hamilton–Jacobi reachability analysis to predict and mitigate failures from high-dimensional observations.
- A separate safety policy can override task-policy actions. Demonstrated hazards include toppling nearby objects and spilling the contents of a bag.
- Learned dynamics and approximate safety values introduce errors. The reported experiments have failure cases and do not establish universal safety guarantees.

**Relevance to OARBot:** Motivates a future safety-filter experiment for tabletop manipulation, beginning with observable hazards such as knocking over a nearby object. Evaluation must compare task success and safety violations with and without the filter.

## Current OARBot implementation

Status checked on November 22, 2026.

- Custom manager-based Isaac Lab reach task using the OARBot USD asset from the Isaac Sim tabletop scene.
- Six Kinova joint-position actions with a fixed base.
- Randomized cube/target placement within a small reachable tabletop region.
- PPO actor and critic configurations, short training checks, and saved checkpoints.
- Observation noise is currently disabled. Learned grasping, base control, world-model training, and latent safety filtering remain future work.

The existing scripted stacking demonstrations and short PPO checks are separate results. Reliable learned pick-and-place and physical sim-to-real transfer have not yet been established.

## Planned experiments

1. Add explicit collision measurements, a sustained reach-success criterion, and fixed held-out evaluation targets.
2. Run longer PPO training with 1,000 parallel environments sharing one policy. Compare nominal training against controlled noise, delay, and physical randomization.
3. Train and validate an OARBot world model, then compare BPTT-based policy learning against PPO on the same task.
4. Extend to grasping and a camera-based safety-filter experiment. Train mobile-base behavior as an additional task.
5. Plot success, reach error, collisions, prediction error, and training time before evaluating transfer to physical hardware.

Paper findings above are attributed to their authors. Proposed OARBot adaptations are experiments to implement and test.
