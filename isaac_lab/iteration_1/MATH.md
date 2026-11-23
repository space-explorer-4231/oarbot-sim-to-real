# Math implemented in the current OARBot PPO task

This describes this repository's code and the installed RSL-RL PPO implementation.
**PPO is due to Schulman et al. (2017), not this project.** Kim et al. (2026)
informed the starting hyperparameters and reaching rewards. See [credits](CREDITS.md).

## 1. One policy, many independent practice episodes

For robot copy i at control step t:

$$
o_t^{(i)}\ \xrightarrow{\pi_\theta}\ a_t^{(i)}\
\xrightarrow{\text{controllers + simulated physics}}\
(o_{t+1}^{(i)},r_t^{(i)},d_t^{(i)}).
$$

Here o is a sensor observation, a an action, r a reward, d an episode-end flag,
and θ the shared actor weights. All 100 copies share θ while keeping separate
states, actions, and episodes. The environment is reset when an episode ends.

Each update collects B = N × T = 100 × 64 = 6,400 transitions. PPO performs
8 epochs × 4 minibatches = 32 optimizer steps over that batch; these are not
32 new simulator rollouts.

## 2. Depth pixels become camera-frame points

For an accepted depth pixel (u,v), measured optical depth z, and intrinsic matrix K:

$$
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix},\qquad
p_D=zK^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix}
=\begin{bmatrix}(u-c_x)z/f_x\\(v-c_y)z/f_y\\z\end{bmatrix}.
$$

Static saved calibration registers a depth point into the RGB optical frame:

$$
p_R=R_{RD}p_D+t_{RD},\qquad
u_R=f_x^R p_{R,x}/p_{R,z}+c_x^R,\quad
v_R=f_y^R p_{R,y}/p_{R,z}+c_y^R.
$$

RGB is sampled at the projected pixel, with validity masks for missing depth,
out-of-range depth, and points outside RGB coverage. USD OpenGL axes are converted
to ROS optical axes with C = diag(1,−1,−1): R_ROS = C R_USD C, t_ROS = C t_USD.
The source calibration currently has approximately coincident RGB/depth centers;
it is not measured Azure Kinect factory calibration.

The code adds z′ = z + ε, with ε ~ Normal(0, (0.005 + 0.001z²)²) in metres and
2% dropout, then forms features. At each valid point, the cloud vector is
[x/4, y/4, z/4, R, G, B, depth-valid, color-valid]. It samples a 12×16 grid.
A 9×16 RGB grid and blue/green color centroids are included as well. A color
“confidence” feature is min(pixel_count/8, 1); it is mask support, not a calibrated
probability of correct detection. No semantic instance IDs are supplied.

Per camera device: 12×16×8 + 9×16×3 + 2×(3+1) = **1,976 features**.
Internal readings are sin/cos of 16 joint angles (32), 16 scaled velocities,
12 previous actions, and a two-value instruction: **62 features**.

$$
D_{\text{one control observation}}=62+2(1976)=4014,\qquad
D_{\text{history}}=4(4014)=16056.
$$

History is four control observations, not necessarily four distinct camera
frames: control is 60 Hz and cameras are 15 Hz in simulated time.

## 3. Neural networks and action distribution

The actor and critic each have three hidden layers of 256, 128, and 64 units:

$$
h_1=\operatorname{ELU}(W_1\bar o+b_1),\quad
h_2=\operatorname{ELU}(W_2h_1+b_2),\quad
h_3=\operatorname{ELU}(W_3h_2+b_3).
$$

Observation normalization produces bar(o). The actor outputs μθ(o) in R¹²,
with a learned state-independent diagonal Gaussian scale initially 0.35:

$$
a\sim\pi_\theta(\cdot\mid o)=\mathcal N(\mu_\theta(o),\operatorname{diag}(\sigma^2)),
\qquad \tilde a_j=\operatorname{clip}(a_j,-1,1).
$$

The critic outputs one number Vφ(o), an estimate of future discounted reward.
It helps train the actor; it does not issue another robot command. These are
MLPs, not recurrent/world-model networks. Four-frame history is input data.

## 4. Twelve actions become wheel, arm, and finger commands

$$
\tilde a=[a_x,a_y,a_\omega,a_{q_1},\ldots,a_{q_6},a_{f_1},a_{f_2},a_{f_3}].
$$

Body speeds are vx = 0.20 ax m/s, vy = 0.20 ay m/s, and ω = 0.40 aω rad/s.
For wheel radius r = 0.049 m and lever parameter L = 0.4954 m, the imported
wheel-joint sign convention gives:

$$
\begin{bmatrix}\dot q_{FL}\\\dot q_{FR}\\\dot q_{RL}\\\dot q_{RR}\end{bmatrix}
=\frac1r\begin{bmatrix}-1&1&L\\-1&-1&-L\\-1&-1&L\\-1&1&-L\end{bmatrix}
\begin{bmatrix}v_x\\v_y\\\omega\end{bmatrix}.
$$

Arm targets integrate bounded increments, with Δt = 1/60 s:

$$
q^{cmd}_{t+1}=\operatorname{clip}(q^{cmd}_t+\Delta t\,s\odot a_q,q_{min},q_{max}),
\quad s=0.5[0.628,0.628,0.628,0.838,0.838,0.838].
$$

Per-episode calibration bias b ~ Uniform(−0.01,0.01) rad is added to commanded
arm targets and encoder angles in this model. Encoder angle noise has σ = 0.001 rad.
This is a calibration-perturbation model, not measured hardware calibration.

For finger i, ci = (afi + 1)/2, proximal target = 1.45ci rad and distal target =
0.40ci rad. Distal coupling is an initial approximation. Physics joint drives
apply these targets; the network does not directly teleport the robot or output
motor torques. Resets do place the robot at a new sampled start.

There are 16 actuated joints (4 wheels + 6 arm + 6 finger joints), 36 passive
roller joints, and 12 independent policy actions. Thus the SDK warning “16 != 52”
is expected for this articulation.

## 5. Exact reward terms

Let p be the tool position, c the cube center, g = c + [0,0,0.045] m, d = ||p−g||,
ρ the base-to-cube XY distance, and θdown the angle between tool +Z and world −Z.
Let h = clip(cz−floor_z−0.671, 0, 0.20) m. Let G be true when at least two distinct
fingers have filtered cube contact above 0.02 N, n = 1[d<0.12], P = 1[push mode],
and O mean the cube remains within 3 cm of its initial table height.

For push target gpush, e = ||cXY−gpush,XY||. The environment's reward rate is:

| Component | Weighted contribution |
| --- | --- |
| Base approach | 0.5[1−tanh(|ρ−0.70|/0.45)] |
| Tool position | −0.2d |
| Fine reaching | 1−tanh(d/0.10) |
| Coarse reaching | 1−tanh(d/0.35) |
| Tool orientation | −0.1 θdown |
| Grasp | G n (1−P) |
| Lift | 6(h/0.15) G n (1−P) |
| Push | 4[1−tanh(e/0.08)] P O |
| Collision | −2 max(Fbody−15,0)/50, force in N |
| Action change | −α(t)||at−at−1||² |
| Arm/finger speed | −β(t)||q̇arm,finger||² |
| Command magnitude (`effort` label) | −0.001||at||²; this is an action penalty, not measured motor energy |

With u = min(t/4500,1), α(t) = 10⁻⁴ + u(0.005−10⁻⁴) and
β(t) = 10⁻⁴ + u(0.001−10⁻⁴). Here t is the shared control-step counter, not
the sum of all parallel transitions. The interpretation and added reward terms
are OARBot adaptations; only the position/fine/orientation baseline is attributed
directly to the paper's reaching formulation.

$$
r_t=\Delta t\,r_{rate,t}+50\,1[\text{success}]-20\,1[\text{failure}].
$$

The code divides terminal event terms by Δt before the Reward Manager multiplies
all terms by Δt, preserving the +50/−20 event values. Lift success requires h>0.08,
G, and n for 0.5 s. Push success requires e<0.025, O, and cube speed<0.08 m/s for
0.5 s. A cube fall, excessive body contact (>250 N), non-finite physical state,
or leaving the allowed arena triggers failure. Timeouts end episodes separately.

True geometry/contact data produce training supervision only. They are not
deployed observation inputs, nor an existing learned safety controller.

## 6. Returns, critic, and GAE

The discounted objective is J(θ) = E[Σk γᵏ rt+k], with γ = 0.99. A reward 100
control steps later has weight 0.99¹⁰⁰ ≈ 0.366. Discounting is per control step,
not per real-world second or PPO update.

For a nonterminal transition mask mt and a frozen rollout value estimate:

$$
\delta_t=r_t+\gamma m_tV_{old}(o_{t+1})-V_{old}(o_t),\qquad
\hat A_t=\delta_t+\gamma\lambda m_t\hat A_{t+1},\qquad
\hat R_t=\hat A_t+V_{old}(o_t).
$$

Here λ = 0.95. RSL-RL handles time-limit bootstrapping separately through its
timeout reward handling. Advantages are standardized over the rollout using
(A−mean(A))/(std(A)+10⁻⁸).

## 7. PPO update and loss

For stored action a sampled by the old policy, let
rt(θ) = exp(log πθ(a|o) − log πold(a|o)). This rt is a probability ratio,
distinct from the environment reward rt above.

$$
L^{clip}(\theta)=\mathbb E_t\left[\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),0.8,1.2)\hat A_t\right)\right].
$$

For positive advantage, increasing an action's probability beyond the clipped
range no longer improves this clipped term. The clip is not a hard guarantee
that every policy probability ratio stays in that range.

The critic's clipped prediction is Vclip = Vold + clip(Vφ−Vold, −0.2, 0.2):

$$
L_V=\mathbb E\left[\max\left((V_\phi-\hat R)^2,
(V_{clip}-\hat R)^2\right)\right],\qquad
\mathcal L=-L^{clip}+1.0L_V-0.01\,\mathbb E[H(\pi_\theta)].
$$

Adam minimizes this loss, starting at learning rate 0.001. The adaptive schedule
uses a target KL divergence of 0.01; the learning rate is not held at 0.001 forever.
The concatenated gradient norm is clipped to 1.0 before each optimizer step.
Entropy encourages continued exploration. The saved checkpoint contains actor,
critic, optimizer, normalization, and iteration state.

This PPO implementation uses backpropagation through the neural networks during
optimization. It **does not differentiate through simulated trajectories**.
SuperMimic-style BPTT through a learned world model is a separate future method.

## 8. Wall time versus simulation time

One rollout contains 64/60 = 1.067 simulated seconds per robot. The first measured
collection took 15.538 real seconds: 1.067/15.538 ≈ 0.0687 times real time.
Its optimizer took 0.592 s, giving 16.13 s per PPO iteration. Initial completion
estimate: 1,000 × 16.13 / 3600 ≈ 4.48 real hours. The launcher's three-hour learning
cap takes precedence. Neither the cap nor 1,000 iterations establishes convergence
or sim-to-real reliability; those require separate evaluation.
