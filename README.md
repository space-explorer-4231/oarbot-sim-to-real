# OARBot Sim-to-Real Research

This project explores how to train OARBot's Kinova Jaco arm in NVIDIA Isaac Lab. The first experiment focuses on reaching a pose above a tabletop cube with the Clearpath Dingo base held stationary.

## Literature

Kim, M., Yuk, D.-G., Lee, J., & Sohn, J. W. (2026). *Sim-to-Real Transfer of Deep Reinforcement Learning for Robotic Pick-and-Place Tasks*. International Journal of Precision Engineering and Manufacturing, 27, 3745–3763. [Read the paper](https://doi.org/10.1007/s12541-026-01595-5).

- The authors trained a reaching policy using Proximal Policy Optimization (PPO) in Isaac Lab and Isaac Sim. They sampled target positions during training so the arm could practice reaching different locations.
- They used color segmentation and RGB-D depth to estimate the target's 3D position. The learned policy controlled arm reaching, and separate logic handled the gripper.
- Their physical UR5e tests reported 100% success with a fixed target, 93.3% for pick-and-place with real-time tracking in a dynamic test, and 60% in the multi-object test.

## How it informs OARBot

The OARBot Reach setup adapts the paper's target randomization and PPO joint-position control for the six Kinova arm joints. Table 3 of the paper provides the starting PPO settings. The OARBot evaluation will measure reach success, gripper position error, and time to reach on cube positions reserved for testing.


also used [2000 demention safty fetures.pdf](https://github.com/user-attachments/files/35247323/2000.demention.safty.fetures.pdf)


[Super Mimic Paper.pdf](https://github.com/user-attachments/files/35247318/Super.Mimic.Paper.pdf)
