# DRS-gatekeeper: racecar

Racecar experiment for *Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach* (Cherenson, Lee, Kim, Panagou, IEEE RA-L 2026). [Project page](https://drs-gk.github.io/) · [arXiv](https://arxiv.org/pdf/2605.12974)

An MPPI controller drives a Dallara F317 around the Barcelona-Catalunya GP circuit at high speed. DRS-gatekeeper runs on top of it to intervene whenever the MPPI controller leads the racecar to go off-track to provide probabilistic guarantee with high confidence.

## Structure

```
clean_universal_simulator.py   main script: settings, episode loop, statistics
drs_gatekeeper/
  drs_gatekeeper.py            DRS-gatekeeper (Algorithm 1 of the paper)
  track_bounds.py              distribution of the track-boundary parameters
uncertain_racecar_gym/
  jax_env.py                   JAX environment and dynamics
  dynamics.py, track.py        NumPy bicycle model and track geometry
  scenario.py                  scenario YAML loader
  controllers.py               backup controllers (centerline, full brake)
  mppi_jax/, mppi_support.py   MPPI controller (nominal policy)
  uncertainty.py               empirical uncertainty model
  uncertainty_jax.py           JAX sampler for that model, used in gatekeeper rollouts
  features.py                  features the uncertainty model is conditioned on
  rendering.py, replay.py      PyBullet video rendering, replay export
  assets/                      scenarios, track centrelines, car URDF
output/
  demo_empirical_model.pkl     fitted empirical uncertainty model
```

## Setup

Python 3.11–3.13.

```bash
uv sync                  # CPU
uv sync --extra cuda     # GPU
```

## Usage

All settings are constants at the top of `clean_universal_simulator.py`. Run from the repository root:

```bash
uv run python clean_universal_simulator.py
```

| Setting | Meaning |
|---|---|
| `UNCERTAINTY` | noise in the true dynamics: `none`, `gaussian`, `empirical` |
| `USE_GATEKEEPER` | turn the safety filter on or off |
| `GK_M`, `GK_T`, `GK_N` | candidate switching times, rollout horizon, rollouts per candidate |
| `GK_DELTA`, `GK_EPSILON`, `GK_BETA`, `GK_ALPHA` | confidence $\delta$, failure probability $\varepsilon$, Wasserstein radius $\beta$, backup failure probability $\alpha$ |
| `LIPSCHITZ_CONSTANT` | Lipschitz constant $L_H$ |
| `BACKUP_CONTROLLER` | `centerline` (slows to $6\,\mathrm{m/s}$) or `full_brake` |
| `CONTROLLER_LATERAL_NOISE_STD` | noise on the lane distance seen by the controllers |
| `RENDER_MODE`, `SAVE_VIDEOS` | `rgb_array_follow` / `rgb_array_birds_eye` / `rgb_array_cinematic`; save one MP4 per episode |
| `ITERATIONS`, `SEED` | number of episodes; episode $i$ uses seed `SEED` $+\,i$ |

An episode ends when the car reaches the half-lap mark, leaves the track, or hits `MAX_STEPS`. The script prints each episode's outcome, backup ratio, average speed and computation time, then a summary.

Paper settings (empirical case): `UNCERTAINTY = "empirical"`, `USE_GATEKEEPER = True`, $M = 10$, $T = 10$, $N = 1000$, $\delta = 0.1$, $\varepsilon = 0.1$, $\beta = 0.25$, $L_H = 2.0$, `CONTROLLER_LATERAL_NOISE_STD = 0.5`.

## Origin

The environment, dynamics, track assets, MPPI controller and empirical uncertainty model come from [uncertain-racecar-gym](https://github.com/tkkim-robot/uncertain-racecar-gym). This repository keeps only what the simulator needs. The uncertainty model is fitted in that repository and loaded here from `output/demo_empirical_model.pkl`.

### Dynamics model

A dynamic bicycle model with linear tyre forces, stepped at $\Delta t = 0.05\,\mathrm{s}$ (`step_nominal` in `jax_env.py`):

- State: position $(x, y)$ and yaw $\psi$, velocities $v_x$, $v_y$ and yaw rate $r$, steering $\delta_s$, and the car's progress, lateral error $e_y$ and heading error $e_\psi$ relative to the track centreline.
- Action: $u = (u_{\mathrm{steer}}, u_{\mathrm{throttle}}, u_{\mathrm{brake}}) \in [-1, 1] \times [0, 1] \times [0, 1]$.
  These are scaled to $\pm 0.32\,\mathrm{rad}$ steering, $12\,\mathrm{m/s^2}$ acceleration and $18\,\mathrm{m/s^2}$ braking.
- Steering follows the command with a first-order lag. Longitudinal acceleration comes from throttle, brake and quadratic drag; lateral forces are $F_y = C_\alpha \, \alpha$ (cornering stiffness times slip angle).
- Model error enters as an additive residual $w = (\Delta v_x, \Delta v_y, \Delta r)$.

The F317 parameters are in `assets/scenarios/ks_barcelona_layout_gp_dallara_f317_rl_long.yaml`.

### Data

The empirical model is built from offline laps of the Dallara F317 on Barcelona (`ks_barcelona_layout_gp`) from the human driving dataset of [assetto_corsa_gym](https://github.com/dasGringuen/assetto_corsa_gym) (Remonda et al., NeurIPS 2024), the same data used in its *Train SAC from demonstrations* example. For every recorded step, the model stores the one-step residual between the recorded telemetry and the bicycle model's prediction of $v_x$, $v_y$ and $r$, together with features: state, track curvature, controls, recent actions, and telemetry such as slip and gear.

At runtime it samples a residual by nearest neighbours on these features, within the same driving regime and section of track, and follows the recorded sequence for a short block. The resulting noise is non-Gaussian and depends on the state.

## How DRS-gatekeeper runs

At every step:

1. MPPI plans from a noisy observation of the car.
2. The gatekeeper takes the true state and MPPI's plan. For each candidate switching time $s \in \lbrace t, \dots, t+M-1 \rbrace$ it rolls out $N$ trajectories that follow the plan until $s$ and the backup controller afterwards. A rollout fails if the lane distance $|e_y|$ exceeds $3.0\,\mathrm{m}$ at any step, or if it does not end with $|e_y| \le 2.75\,\mathrm{m}$, the set where the backup can keep the car.
3. It selects the latest $s$ whose upper bound $\bar{q}^s$ on the failure probability, made robust to distributions within $\beta$ of the sampled one, satisfies $\bar{q}^s \le (\varepsilon - \alpha)/(1 - \alpha)$. Once $s$ is reached, the backup controller drives.

The gatekeeper's rollouts deliberately differ from the true system in two ways, and $\beta$ has to cover the gap:

- **Dynamics mismatch.** The environment samples residuals from the full empirical model, using the full action history and following recorded sequences. The gatekeeper samples from a JAX version conditioned only on the current control, with each step drawn independently.
- **Perception mismatch.** The controllers see the lane distance (lateral error) corrupted by Gaussian noise with $\sigma = 0.5\,\mathrm{m}$, clipped at $\pm 4\sigma$. The gatekeeper's safety function uses the lane distance without this noise.

## Citation

```bibtex
@article{cherenson2026drs,
  title   = {Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach},
  author  = {Cherenson, Daniel M. and Lee, Haejoon and Kim, Taekyung and Panagou, Dimitra},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```
