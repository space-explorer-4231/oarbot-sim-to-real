# What has actually been checked

November 23, 2026. These checks were performed on SVR2. They establish that the
pipeline executes; they do not establish reliable grasping or physical deployment.

| Check | Observed result |
| --- | --- |
| Four- and 101-environment diagnostics | Finite observations/rewards, visible blue cube and push marker in the room camera, valid depth, randomized reset images, and forward movement without failure. |
| Whole-body control diagnostic | All six arm joints moved; all fingers closed and reopened; lateral base travel ≈4.10 cm and yaw ≈0.0621 rad under conservative diagnostic commands. |
| Actual 101-environment PPO run | Three PPO updates, 19,392 transitions, 49.56 seconds of learning after loading, streaming enabled. |
| Checkpoint verification | Actor and critic layer weights changed between saved iterations 0 and 2 and remained finite. |
| Saved-policy playback | Loaded the checkpoint and executed 32 finite-action steps on eight evaluation environments. This was not a full success-rate evaluation. |
| Interrupted training | Saved and successfully reloaded an interrupted checkpoint. |
| Background watched launch | Started detached from SSH and completed a short bounded training check. |
| PC viewer | User confirmed eight-robot preview, then 100-robot training visible through WebRTC after fixed-resolution stream recovery. |

The current 100-environment run began on November 23 at about 4:56 PM Eastern.
Its first PPO update collected 6,400 steps in 16.13 seconds. A later snapshot at
5:05 PM Eastern showed TensorBoard iteration index 28 (29 updates), with no lift
or push success reported in the latest termination metrics. It remained an early
training run. Rewards and successful-looking movement are not success rates.

The source of that snapshot is [live_status_20261123.json](evidence/live_status_20261123.json).
It is historical, not a live dashboard. The full server-side check report is
`verification/setup_summary.json`; large logs and model checkpoints are not uploaded.

## Camera fidelity

Measured ideal pinhole frusta in the installed SDK:

| Optical channel | Training resolution | Measured horizontal × vertical field of view |
| --- | --- | --- |
| Room RGB | 320×180 | 90° × 58.72° |
| Room depth | 160×144 | 75° × 69.26° |
| Wrist RGB | 160×90 | 90° × 58.72° |
| Wrist depth | 128×128 | 120° × 120° |

The renderer assumes square pixels and ignores vertical aperture for these
intrinsics. Room depth therefore does not exactly match Kinect's nominal
75°×65° NFOV. Factory distortion, measured camera-to-robot extrinsics, RGB/depth
baseline, physical contact, fingertip coupling, and mass distributions require
further validation. A thousand camera-equipped environments, learned safety,
world-model/BPTT training, and real-robot transfer have not been demonstrated here.
