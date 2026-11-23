# OARBot Whole-Body Learning: PPO and Sim-to-Real Research

This project studies how to train the OARBot mobile manipulator for tabletop manipulation and evaluate transfer from simulation to physical hardware. OARBot combines a Clearpath Dingo base with a six-joint Kinova Jaco arm.

**November 23, 2026:** camera-based whole-body PPO is running with 100 parallel OARBots. One shared policy commands the mobile base, six arm joints, and three fingers. The task mixes cube lifting and pushing, with randomized starts, friction, calibration offsets, sensor noise, and one obstacle per arena.

**Start here:** [working code and beginner guide](isaac_lab/iteration_1/README.md) · [PPO/control/reward math](isaac_lab/iteration_1/MATH.md) · [all parameters](isaac_lab/iteration_1/PARAMETERS.md) · [credits](isaac_lab/iteration_1/CREDITS.md) · [verification and limits](isaac_lab/iteration_1/EVIDENCE.md).

The source code is published; the robot USD and trained checkpoints remain external assets on SVR2, with setup/export instructions included. Reliable learned grasping and real-robot transfer are not established results. The literature below explains the baseline and future research directions.

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

The [Iteration 1 project](isaac_lab/iteration_1/) uses two RGB-D camera devices and internal encoders as policy input. Simulator geometry/contact values are reserved for reward and measurement. A 12-output actor controls body motion, six arm targets, and three finger closures; all copies share its weights. A critic helps PPO estimate future return.

A 101-environment check completed three PPO updates, saved changed finite weights, and reloaded a checkpoint. The user then started a longer 100-environment run and confirmed live viewing. This verifies execution, not convergence or pick-and-place reliability. The initial stationary reach task is earlier work; scripted stacking demonstrations are separate from learned-policy results.

## Next research steps

1. Evaluate the current saved policy on held-out cube positions and report lift/push success, collisions, and time; exclude randomized-length warm starts.
2. Benchmark increasing camera-equipped environment counts before targeting 1,000. Compare physical/sensor randomization through controlled runs.
3. Fit and validate an OARBot world model, then implement a BPTT-based policy learner for a fair comparison with PPO. No world-model/BPTT training is implemented yet.
4. Add and evaluate a learned safety filter, then build and validate the physical robot interface and camera calibration before sim-to-real trials.

Paper findings above are attributed to their authors. Proposed OARBot adaptations are experiments to implement and test.

