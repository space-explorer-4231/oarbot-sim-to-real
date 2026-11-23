# Credits and provenance

**Project implementation and experiment design:** Mateusz Zalewski
([space-explorer-4231](https://github.com/space-explorer-4231)), with AI coding
assistance from OpenAI Codex. The OARBot task, control/sensor adapters, reward
extensions, curriculum, and integration are project adaptations. They are not
the invention of PPO or a reproduction of the authors' entire systems.

| Source | Credit and role |
| --- | --- |
| [Schulman, Wolski, Dhariwal, Radford, and Klimov (2017), Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | Original PPO algorithm and clipped surrogate objective. |
| [Schulman et al. (2015), High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | GAE advantage-estimation method. |
| [Kim, Yuk, Lee, and Sohn (2026), Sim-to-Real Transfer of Deep Reinforcement Learning for Robotic Pick-and-Place Tasks](https://doi.org/10.1007/s12541-026-01595-5) | Table 3 PPO starting hyperparameters; position/fine-position/orientation reward inspiration. Their learned arm-reaching policy and separate gripper logic differ from this whole-body camera task. Their reported results are not our results. |
| [Liao et al. (2026), SuperMimic](https://arxiv.org/abs/2610.28480) | Friction U[0.3,1.2] and joint-calibration U[−0.01,0.01] randomization inspiration; future world-model/BPTT comparison. Its world model and BPTT are not implemented here. |
| [Nakamura, Peters, and Bajcsy (2025), Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis](https://arxiv.org/abs/2502.00935) | Future learned safety-filter research direction. No such filter is implemented in this task. |
| [Isaac Lab Project Developers](https://github.com/isaac-sim/IsaacLab), [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) | Environment framework, physics/rendering integration, generated task scaffold, and adapted RSL-RL launch scripts. Preserve the upstream copyright/SPDX headers. |
| [RSL-RL contributors, ETH Zurich and NVIDIA](https://github.com/leggedrobotics/rsl_rl) | Actual PPO trainer, MLP actor/critic, rollout storage, optimizer, and checkpoint implementation. The library is a dependency, not copied wholesale into this folder. |
| [RPI Robotics OARBot documentation](https://rpirobotics.github.io/oarbots/docs/getting-started/hardware-overview), [oarbots](https://github.com/rpiRobotics/oarbots), [Assistive-Robotics](https://github.com/rpiRobotics/Assistive-Robotics) | OARBot hardware/URDF references and robot integration context. This training launch does not connect their ROS/MoveIt drivers to physical hardware. |
| Clearpath Robotics, Kinova Robotics, Microsoft | Dingo base, Jaco Gen2 arm/gripper, and Azure Kinect DK hardware/design references. Hardware model meshes and SDKs retain their own terms. |

RSL-RL, PyTorch, and this 256/128/64 MLP architecture are implementation choices;
the architecture is not claimed to come from Kim et al. The 100-copy visual
mobile task, pushing goal, obstacle, noise model, curriculum, and rewards are
adaptations requiring independent evaluation.

## Source-code notices

SDK-derived files retain their BSD-3-Clause headers. The full Isaac Lab notice
is in [THIRD_PARTY_LICENSES/IsaacLab.txt](THIRD_PARTY_LICENSES/IsaacLab.txt).
Other dependencies and robot assets are obtained separately and retain their
upstream licenses; this project does not relicense them. Research papers are
cited and linked, not redistributed here. New project integration code is
copyright 2026 Mateusz Zalewski; no blanket license grant for external robot
meshes or proprietary SDK components is implied by publication of this folder.
