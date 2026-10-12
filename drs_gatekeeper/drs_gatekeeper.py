"""Distributionally Robust Stochastic Gatekeeper (DRS-gatekeeper).

Implements Algorithm 1 of

    D. M. Cherenson, H. Lee, T. Kim, and D. Panagou, "Distributionally
    Robust Safety Under Arbitrary Uncertainties: A Safety Filtering
    Approach," IEEE Robotics and Automation Letters, 2026.

The filter decides when to hand control from the nominal policy pi_N to
the backup policy pi_B. The switching policy pi_S(x, t; s) (Eq. 7) runs
pi_N while t < s and pi_B from s onward. At each time t, for every
candidate switching time s in {t, ..., t+M-1}, the filter:

  1. samples N noise trajectories w_T ~ P_hat_T and rolls out pi_S(.; s)
     for T steps (Alg. 1, lines 1-4),
  2. evaluates the finite-horizon safety value H_t^{x,s}(w_T) (Def. 4),
  3. counts failures k^s = #{i : H(w_i) <= L_H * beta}, where L_H * beta
     is the Wasserstein buffer from Lemma 1 (line 6),
  4. bounds the failure probability with the Clopper-Pearson bound
     q_bar^s = q_bar(k^s, N, rho) of Eq. 10, using the Sidak level
     rho = 1 - (1 - delta)^(1/M) (line 7),

and returns the largest s with q_bar^s <= (eps - alpha) / (1 - alpha)
(lines 8-10). If no candidate qualifies, the previous switching time
s*_{t-1} is kept (lines 11-12).

Implementation notes
--------------------
- Rollouts are vectorized with jax.vmap over the N samples and M
  candidates, and use lax.scan over the T steps.
- q_bar is computed on the host with scipy, since betaincinv is not
  JAX-traceable.
- Noise comes from one of two sources: i.i.d. Gaussian residuals sampled
  before the rollouts, or the empirical uncertainty model, sampled inside
  each rollout step from the current rollout state.
- States are flat vectors of length STATE_DIM (layout below) and actions
  are [steer, throttle, brake].
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, NamedTuple, Optional, Tuple
import time


import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import scipy.special as sc_special

from uncertain_racecar_gym.uncertainty import EmpiricalUncertaintyModel
from uncertain_racecar_gym.uncertainty_jax import JAXEmpiricalData, sample_empirical_jax

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Array = jax.Array

# States are flat float32 vectors inside the vmapped/scanned rollouts.
# Layout: [x, y, yaw, vx, vy, yaw_rate, steer, throttle, brake, progress, lateral_error, heading_error]
STATE_DIM = 12

# Action layout: [steer_cmd, throttle_cmd, brake_cmd]
ACTION_DIM = 3


def _normalize_gaussian_vector(
    values: Optional[Any],
    default: np.ndarray,
    nonnegative: bool = False,
) -> np.ndarray:
    """Return `values` (or `default` if None) as a float vector of length 4.

    The four entries are the residual channels
    [delta_vx, delta_vy, delta_yaw_rate, delta_steer]. With nonnegative=True,
    negative entries are clipped to 0 (used for standard deviations).
    """
    if values is None:
        vector = np.asarray(default, dtype=float).copy()
    else:
        vector = np.asarray(values, dtype=float).reshape(-1)
    if len(vector) != 4:
        raise ValueError(f"Gaussian noise vectors must have length 4, got {len(vector)}")
    if nonnegative:
        vector = np.clip(vector, 0.0, np.inf)
    return vector.astype(float)


# ---------------------------------------------------------------------------
# Parameters and runtime state
# ---------------------------------------------------------------------------


class GatekeeperParams(NamedTuple):
    """Hyperparameters of Algorithm 1. Symbols follow the paper.

    Attributes
    ----------
    M : int
        Number of candidate switching times, s in {t, ..., t+M-1}.
    T : int
        Rollout horizon in steps.
    N : int
        Number of sampled rollouts per candidate.
    delta : float
        Confidence parameter delta. When a switching time is certified, it
        satisfies the chance constraint with probability >= 1 - delta (Thm. 1).
    epsilon : float
        Allowed failure probability epsilon in the chance constraint (CC).
    beta : float
        Radius of the infinity-Wasserstein ambiguity set around the nominal
        noise distribution (Assumption 2). beta = 0 certifies against the
        nominal distribution only.
    alpha : float
        Failure probability of the backup policy's alpha-PCIS (Assumption 1,
        alpha < epsilon). Enters the threshold (epsilon - alpha) / (1 - alpha).
    p : int
        Exponent in the failure buffer L_H * beta^(1/p). This is not a paper
        parameter: the paper uses the infinity-Wasserstein distance, whose
        buffer is L_H * beta (Lemma 1). Keep p = 1.
    lipschitz_mode : str
        'fixed' uses `lipschitz_constant` as L_H. 'autodiff' sets L_H to the
        largest ||dH/dw||_2 over the sampled rollouts; this is an empirical
        estimate, not a guaranteed Lipschitz constant (see Remark 1).
    lipschitz_constant : float
        L_H for lipschitz_mode='fixed'. The paper's racecar uses 2.0.
    debug_timing : bool
        Print a per-update timing breakdown.
    """

    M: int
    T: int
    N: int
    delta: float
    epsilon: float
    beta: float
    alpha: float
    p: int = 1
    lipschitz_mode: str = "fixed"   # 'fixed' | 'autodiff'
    lipschitz_constant: float = 0.0
    debug_timing: bool = False


class GatekeeperState(NamedTuple):
    """Runtime state of the gatekeeper between updates.

    Attributes
    ----------
    t : int
        Current time step.
    s_prev : int
        Current switching time s*_t (absolute time step). It is kept as
        s*_{t-1} when an update certifies no candidate.
    using_backup : bool
        True when t >= s_prev, i.e. the backup policy should be applied.
    predicted_trajectories : np.ndarray or None
        [N, T+1, 2] x-y positions of the rollouts for the selected candidate.
        Visualisation only.
    failure_mask : np.ndarray or None
        [N] bool, which of those rollouts counted as failures (using the fixed
        L_H). Visualisation only.
    m_star : int
        Offset s*_t - t of the current switching time.
    q_bar_star : float
        q_bar of the candidate shown in the visualisation.
    is_reverting : bool
        True when the last update certified nothing and s_prev was kept.
    plan_start_t : int
        Time step at which the displayed rollouts were computed.
    """

    t: int
    s_prev: int
    using_backup: bool = False
    predicted_trajectories: Optional[np.ndarray] = None
    failure_mask: Optional[np.ndarray] = None
    m_star: int = 0
    q_bar_star: float = 0.0
    is_reverting: bool = False
    plan_start_t: int = 0

# ---------------------------------------------------------------------------
# Distribution of the unsafe-set parameters theta (track boundaries); see
# track_bounds.py.
# ---------------------------------------------------------------------------
from drs_gatekeeper.track_bounds import TrackBoundsEstimate  # noqa: F401


# ---------------------------------------------------------------------------
# State helpers and the JAX feature extractor
# ---------------------------------------------------------------------------


def pack_state(x, y, yaw, vx, vy, yaw_rate, steer, throttle, brake,
               progress, lateral_error, heading_error) -> Array:
    """Pack state components into a flat float32 vector (layout of STATE_DIM)."""
    return jnp.array(
        [x, y, yaw, vx, vy, yaw_rate, steer, throttle, brake,
         progress, lateral_error, heading_error],
        dtype=jnp.float32,
    )


def unpack_state(s: Array):
    """Unpack a flat state vector into a dict of named components."""
    return dict(
        x=s[0], y=s[1], yaw=s[2],
        vx=s[3], vy=s[4], yaw_rate=s[5],
        steer=s[6], throttle=s[7], brake=s[8],
        progress=s[9], lateral_error=s[10], heading_error=s[11],
    )


def jax_feature_extractor(
    state: Array,
    curvature: float,
    feature_dim: int,
    lr: float = 0.16,
) -> Array:
    """Approximate the empirical model's feature vector from a flat state, in JAX.

    Used to condition the empirical noise sampler inside rollouts. It fills
    only part of the feature layout defined in features.py:

    - 0-7: curvature, progress, vx, vy, yaw_rate, steer, throttle, brake
    - 13: rear slip angle, from vx, vy, yaw_rate and the rear axle distance lr
    - 14: gear_norm, approximated by clip(vx / 50, 0, 1)
    - 18+: action history, filled by repeating the current
      [steer, throttle, brake]

    All other entries stay 0. Because the history repeats the current action,
    the rollout noise is conditioned on the current input only, not on the
    full history (model mismatch (ii) in the paper's racecar experiment).
    """
    vx = state[3]
    vy = state[4]
    yaw_rate = state[5]
    steer = state[6]
    throttle = state[7]
    brake = state[8]
    progress = state[9]

    # accel_y is computed but not written to the feature vector (index 9 stays 0).
    accel_y = vx * yaw_rate
    rear_slip_angle = jnp.arctan2(vy - lr * yaw_rate, jnp.maximum(jnp.abs(vx), 0.5))
    
    feat = jnp.zeros((feature_dim,), dtype=jnp.float32)

    def _set_if(index: int, value: Array, arr: Array) -> Array:
        if feature_dim > index:
            return arr.at[index].set(value)
        return arr

    feat = _set_if(0, curvature, feat)
    feat = _set_if(1, progress, feat)
    feat = _set_if(2, vx, feat)
    feat = _set_if(3, vy, feat)
    feat = _set_if(4, yaw_rate, feat)
    feat = _set_if(5, steer, feat)
    feat = _set_if(6, throttle, feat)
    feat = _set_if(7, brake, feat)
    feat = _set_if(13, rear_slip_angle, feat)
    feat = _set_if(14, jnp.clip(vx / 50.0, 0.0, 1.0), feat)

    if feature_dim > 18:
        history_width = feature_dim - 18
        history_template = jnp.tile(
            jnp.asarray([steer, throttle, brake], dtype=jnp.float32),
            (history_width + 2) // 3,
        )[:history_width]
        feat = feat.at[18:feature_dim].set(history_template)
    
    return feat


def _zero_curvature_fn(progress: Array) -> Array:
    """Curvature fallback used when no track samples are available."""
    del progress
    return jnp.asarray(0.0, dtype=jnp.float32)


def _build_track_curvature_fn(track: Optional[Any]) -> Callable[[Array], Array]:
    """Return a JAX function mapping track progress in [0, 1) to curvature.

    It interpolates the track's arc-length curvature samples, extended by one
    lap so the lookup wraps around. Returns `_zero_curvature_fn` if the track
    does not provide samples.
    """
    if track is None:
        return _zero_curvature_fn

    arc_samples = getattr(track, "_arc_samples", None)
    curvature_samples = getattr(track, "_curvature_samples", None)
    length = getattr(track, "length", None)
    if arc_samples is None or curvature_samples is None or length is None:
        return _zero_curvature_fn

    try:
        arc_np = np.asarray(arc_samples, dtype=np.float32)
        curv_np = np.asarray(curvature_samples, dtype=np.float32)
        length_f = float(length)
    except Exception:
        return _zero_curvature_fn

    if arc_np.size < 2 or curv_np.size < 2:
        return _zero_curvature_fn

    interp_arc = np.concatenate([arc_np, arc_np[1:] + length_f], axis=0)
    interp_values = np.concatenate([curv_np, curv_np[1:]], axis=0)
    interp_arc_jax = jnp.asarray(interp_arc, dtype=jnp.float32)
    interp_values_jax = jnp.asarray(interp_values, dtype=jnp.float32)
    length_jax = jnp.asarray(length_f, dtype=jnp.float32)

    def _curvature_fn(progress: Array) -> Array:
        arc = jnp.mod(progress, 1.0) * length_jax
        return jnp.interp(arc, interp_arc_jax, interp_values_jax)

    return _curvature_fn


def rollout_single(
    x0: Array,
    switching_offset: Array,
    noise_traj: Array,
    env_params_traj: Array,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    nominal_trajectory: Optional[Array] = None,
) -> Array:
    """Roll out pi_S(.; s) for T steps with a given noise trajectory and return H.

    The nominal action is used while step < switching_offset and the backup
    action afterwards, so offset 0 means "switch to backup now". If
    `nominal_trajectory` is given (e.g. an MPPI action plan with at least T
    rows), the nominal action at step k is nominal_trajectory[k] (open loop);
    otherwise it is nominal_policy_fn(state).

    Parameters
    ----------
    x0 : [STATE_DIM]
    switching_offset : int
        s - t.
    noise_traj : [T, noise_dim]
        Process noise passed to dynamics_fn at each step.
    env_params_traj : [T, theta_dim]
        Unsafe-set parameters theta passed to safety_fn at each step.

    Returns
    -------
    H : scalar
        min( min_{1<=k<=T} h(x_k, theta_k), h_c(x_T) ), the safety value of
        Def. 4 evaluated on the states after each step. The current state x_0
        is not included.
    xy_traj : [T+1, 2]
        Planar positions along the rollout, for visualisation.
    """
    def step_fn(carry, inputs):
        state, step_idx = carry
        noise_step, env_param_step = inputs
        use_nominal = step_idx < switching_offset
        
        # pi_S(.; s): nominal before the switching offset, backup from it on.
        # The nominal action is the open-loop plan (MPPI) if given, else pi_N(x).
        if nominal_trajectory is not None:
             action_nominal = nominal_trajectory[step_idx]
        else:
             action_nominal = nominal_policy_fn(state)

        action_backup = backup_policy_fn(state)
        action = jnp.where(use_nominal, action_nominal, action_backup)
        next_state = dynamics_fn(state, action, noise_step)
        h_val = safety_fn(next_state, env_param_step)
        # x-y position for visualisation
        xy = jnp.stack([state[0], state[1]], axis=0)
        return (next_state, step_idx + 1), (h_val, xy)

    final_carry, (h_values, xy_traj) = lax.scan(
        step_fn, (x0, 0), (noise_traj, env_params_traj), length=T
    )
    final_xy = jnp.stack([final_carry[0][0], final_carry[0][1]], axis=0)
    xy_traj = jnp.concatenate([xy_traj, final_xy[None, :]], axis=0)
    
    # H = min(min_k h(x_k), h_c(x_T))  (Def. 4)
    h_c_terminal = pcis_fn(final_carry[0])
    return jnp.minimum(h_c_terminal, jnp.min(h_values)), xy_traj


def rollout_single_empirical(
    x0: Array,
    switching_offset: Array,
    rng_key: Array,
    env_params: Array,
    uncertainty_data: JAXEmpiricalData,
    gate_id: int,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    curvature_fn: Callable[[Array], Array],
    dynamic_gate_from_progress: bool = False,
    lr: float = 0.16,
    nominal_trajectory: Optional[Array] = None,
) -> Array:
    """Like `rollout_single`, but samples the noise online from the empirical model.

    At each step the noise is drawn with `sample_empirical_jax`, conditioned
    on the current rollout state and the track curvature. theta is static
    over the rollout. Returns (H, noise_traj [T, noise_dim], xy_traj); the
    realized noise is needed for the autodiff Lipschitz estimate.
    """
    def step_fn(carry, subkey):
        state, step_idx = carry
        use_nominal = step_idx < switching_offset
        
        # pi_S(.; s): nominal before the switching offset, backup from it on.
        if nominal_trajectory is not None:
             action_nominal = nominal_trajectory[step_idx]
        else:
             action_nominal = nominal_policy_fn(state)

        action_backup = backup_policy_fn(state)
        action = jnp.where(use_nominal, action_nominal, action_backup)
        
        # Sample this step's noise from the empirical model, conditioned on the
        # current rollout state. With a single-gate model the gate follows
        # track progress; otherwise gate_id is fixed.
        curv = curvature_fn(state[9])
        feat = jax_feature_extractor(
            state,
            curv,
            feature_dim=int(uncertainty_data.feature_mean.shape[0]),
            lr=lr,
        )
        if dynamic_gate_from_progress:
            gate_count = uncertainty_data.lookup_gate_start.shape[0]
            gate_step = jnp.clip(
                jnp.floor(jnp.mod(state[9], 1.0) * gate_count).astype(jnp.int32),
                0,
                gate_count - 1,
            )
        else:
            gate_step = jnp.asarray(gate_id, dtype=jnp.int32)
        noise_step = sample_empirical_jax(feat, gate_step, subkey, uncertainty_data)

        next_state = dynamics_fn(state, action, noise_step)
        h_val = safety_fn(next_state, env_params)
        xy = jnp.stack([state[0], state[1]], axis=0)
        return (next_state, step_idx + 1), (h_val, noise_step, xy)

    keys = jax.random.split(rng_key, T)
    final_carry, (h_values, noise_traj, xy_traj) = lax.scan(step_fn, (x0, 0), keys, length=T)
    final_xy = jnp.stack([final_carry[0][0], final_carry[0][1]], axis=0)
    xy_traj = jnp.concatenate([xy_traj, final_xy[None, :]], axis=0)

    h_c_terminal = pcis_fn(final_carry[0])
    return jnp.minimum(h_c_terminal, jnp.min(h_values)), noise_traj, xy_traj


def rollout_single_empirical_value(
    x0: Array,
    switching_offset: Array,
    rng_key: Array,
    env_params: Array,
    uncertainty_data: JAXEmpiricalData,
    gate_id: int,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    curvature_fn: Callable[[Array], Array],
    dynamic_gate_from_progress: bool = False,
    lr: float = 0.16,
    nominal_trajectory: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Value-only version of `rollout_single_empirical`.

    Does not return the sampled noise, which saves memory when L_H is fixed.
    Returns (H, xy_traj).
    """
    def step_fn(carry, subkey):
        state, step_idx = carry
        use_nominal = step_idx < switching_offset

        if nominal_trajectory is not None:
            action_nominal = nominal_trajectory[step_idx]
        else:
            action_nominal = nominal_policy_fn(state)

        action_backup = backup_policy_fn(state)
        action = jnp.where(use_nominal, action_nominal, action_backup)

        curv = curvature_fn(state[9])
        feat = jax_feature_extractor(
            state,
            curv,
            feature_dim=int(uncertainty_data.feature_mean.shape[0]),
            lr=lr,
        )
        if dynamic_gate_from_progress:
            gate_count = uncertainty_data.lookup_gate_start.shape[0]
            gate_step = jnp.clip(
                jnp.floor(jnp.mod(state[9], 1.0) * gate_count).astype(jnp.int32),
                0,
                gate_count - 1,
            )
        else:
            gate_step = jnp.asarray(gate_id, dtype=jnp.int32)
        noise_step = sample_empirical_jax(feat, gate_step, subkey, uncertainty_data)

        next_state = dynamics_fn(state, action, noise_step)
        h_val = safety_fn(next_state, env_params)
        xy = jnp.stack([state[0], state[1]], axis=0)
        return (next_state, step_idx + 1), (h_val, xy)

    keys = jax.random.split(rng_key, T)
    final_carry, (h_values, xy_traj) = lax.scan(step_fn, (x0, 0), keys, length=T)
    final_xy = jnp.stack([final_carry[0][0], final_carry[0][1]], axis=0)
    xy_traj = jnp.concatenate([xy_traj, final_xy[None, :]], axis=0)

    h_c_terminal = pcis_fn(final_carry[0])
    return jnp.minimum(h_c_terminal, jnp.min(h_values)), xy_traj


def rollout_with_grad(
    x0: Array,
    switching_offset: Array,
    noise_traj: Array,
    env_params_traj: Array,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    nominal_trajectory: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Return ((H, xy_traj), dH/dw) for a fixed noise trajectory w.

    Used to estimate L_H when lipschitz_mode='autodiff'.
    """
    def h_only(w):
        return rollout_single(
            x0, switching_offset, w, env_params_traj,
            dynamics_fn, nominal_policy_fn, backup_policy_fn,
            safety_fn, pcis_fn, T,
            nominal_trajectory=nominal_trajectory
        )
    # has_aux: h_only returns (H, xy_traj); only H is differentiated.
    (val, aux), grad = jax.value_and_grad(h_only, has_aux=True)(noise_traj)
    return (val, aux), grad


# ---------------------------------------------------------------------------
# Batched rollouts over M candidates x N samples
# ---------------------------------------------------------------------------


def build_rollout_all(
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
) -> Callable:
    """Build the batched rollout for pre-sampled (Gaussian) noise.

    The returned function has the signature

        rollout_all(x0, switching_offsets, noise_batch, env_params_batch,
                    compute_grad=False, nominal_trajectory=None)
            -> (H [M, N], grad [M, N, T, noise_dim] or None, xy [M, N, T+1, 2])

    with noise_batch [M, N, T, noise_dim] and env_params_batch
    [M, N, T, theta_dim]. It is vmapped over the N samples (inner) and the M
    candidates (outer). compute_grad must be static under jax.jit.
    """

    def rollout_all(
        x0: Array,              # [STATE_DIM]
        switching_offsets: Array,  # [M] int32
        noise_batch: Array,     # [M, N, T, noise_dim]
        env_params_batch: Array,   # [M, N, T, theta_dim]
        compute_grad: bool = False,
        nominal_trajectory: Optional[Array] = None,
    ) -> Tuple[Array, Optional[Array], Array]:

        def _single_val(m, w, th):
            return rollout_single(
                x0, m, w, th, dynamics_fn, nominal_policy_fn,
                backup_policy_fn, safety_fn, pcis_fn, T,
                nominal_trajectory=nominal_trajectory
            )

        def _single_grad(m, w, th):
            return rollout_with_grad(
                x0, m, w, th, dynamics_fn, nominal_policy_fn,
                backup_policy_fn, safety_fn, pcis_fn, T,
                nominal_trajectory=nominal_trajectory
            )

        # vmap over the N samples (inner), then the M candidates (outer).
        if not compute_grad:
            single_N = jax.vmap(_single_val, in_axes=(None, 0, 0))
            all_MN = jax.vmap(single_N, in_axes=(0, 0, 0))
            H_matrix, X_all = all_MN(switching_offsets, noise_batch, env_params_batch)
            return H_matrix, None, X_all
        else:
            single_N = jax.vmap(_single_grad, in_axes=(None, 0, 0))
            all_MN = jax.vmap(single_N, in_axes=(0, 0, 0))
            (H_matrix, X_all), Grad_matrix = all_MN(switching_offsets, noise_batch, env_params_batch)
            return H_matrix, Grad_matrix, X_all

    return rollout_all


def build_rollout_all_empirical(
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    uncertainty_data: JAXEmpiricalData,
    gate_id: int,
    curvature_fn: Callable[[Array], Array],
    dynamic_gate_from_progress: bool = False,
) -> Callable:
    """Build the batched rollout for empirical noise sampled inside the rollouts.

    Same output as `build_rollout_all`, but takes PRNG keys rng_matrix
    [M, N, 2] instead of a noise batch, and theta of shape [M, N, theta_dim]
    (static per rollout). With compute_grad=True it first samples and stores
    the noise, then differentiates H with respect to that realized noise.
    """
    def rollout_all(
        x0: Array,
        switching_offsets: Array,
        rng_matrix: Array,     # [M, N, 2] PRNG keys
        env_params_batch: Array, # [M, N, theta_dim]
        compute_grad: bool = False,
        nominal_trajectory: Optional[Array] = None,
    ) -> Tuple[Array, Optional[Array], Array]:
        if not compute_grad:
            # Value-only path: the sampled noise is not stored.
            def _single_val(m, key, th):
                return rollout_single_empirical_value(
                    x0, m, key, th, uncertainty_data, gate_id,
                    dynamics_fn, nominal_policy_fn, backup_policy_fn,
                    safety_fn, pcis_fn, T,
                    curvature_fn=curvature_fn,
                    dynamic_gate_from_progress=dynamic_gate_from_progress,
                    nominal_trajectory=nominal_trajectory,
                )

            v_val = jax.vmap(_single_val, in_axes=(None, 0, 0))
            all_MN_val = jax.vmap(v_val, in_axes=(0, 0, 0))
            H_matrix, X_all = all_MN_val(switching_offsets, rng_matrix, env_params_batch)
            return H_matrix, None, X_all

        # Gradient path: sample and store the noise, then differentiate H
        # with respect to that realized noise, treating it as fixed.
        def _single_sample(m, key, th):
            return rollout_single_empirical(
                x0, m, key, th, uncertainty_data, gate_id,
                dynamics_fn, nominal_policy_fn, backup_policy_fn,
                safety_fn, pcis_fn, T,
                curvature_fn=curvature_fn,
                dynamic_gate_from_progress=dynamic_gate_from_progress,
                nominal_trajectory=nominal_trajectory,
            )

        v_sample = jax.vmap(_single_sample, in_axes=(None, 0, 0))
        all_MN_sample = jax.vmap(v_sample, in_axes=(0, 0, 0))
        H_matrix, sampled_noise_batch, X_all = all_MN_sample(switching_offsets, rng_matrix, env_params_batch)

        def _single_grad(m, w, th):
            return rollout_with_grad(
                x0, m, w, th, dynamics_fn, nominal_policy_fn,
                backup_policy_fn, safety_fn, pcis_fn, T,
                nominal_trajectory=nominal_trajectory
            )

        v_grad = jax.vmap(_single_grad, in_axes=(None, 0, 0))
        all_MN_grad = jax.vmap(v_grad, in_axes=(0, 0, 0))

        _, Grad_matrix = all_MN_grad(switching_offsets, sampled_noise_batch, env_params_batch)
        return H_matrix, Grad_matrix, X_all

    return rollout_all



# ---------------------------------------------------------------------------
# Clopper-Pearson bound (Eq. 10)
# ---------------------------------------------------------------------------


def compute_q_bar(
    k: int,
    N: int,
    rho: float,
) -> float:
    """Upper confidence bound q_bar on a failure probability (Eq. 10).

        q_bar(k, N, rho) = max{ q in [0, 1] : Bin(k; N, q) >= rho },

    where Bin(k; N, q) is the binomial CDF, i.e. the probability of at most k
    failures in N trials. 

    Parameters
    ----------
    k : int
        Number of failures among the N rollouts.
    N : int
        Number of rollouts.
    rho : float
        Per-candidate level from the Sidak correction,
        rho = 1 - (1 - delta)^(1/M).
    """
    k = int(k)
    b = N - k
    if b <= 0:
        return 1.0
    a = k + 1.0
    return float(sc_special.betaincinv(float(a), float(b), float(1.0 - rho)))


def compute_q_bar_vec(k_vec: np.ndarray, N: int, rho: float) -> np.ndarray:
    """Apply `compute_q_bar` to each failure count in k_vec. Returns float32 [M]."""
    return np.array(
        [compute_q_bar(int(k), N, rho) for k in k_vec],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# One iteration of Algorithm 1
# ---------------------------------------------------------------------------


def run_gatekeeper(
    x0: Array,                  # [STATE_DIM] current flat state
    s_prev: int,                # previous switching time (absolute, Python int)
    t: int,                     # current absolute timestep (Python int)
    switching_offsets: Array,   # [M] int32 — candidate relative offsets {1,...,M}
    noise_batch: Optional[Array], # [M, N, T, noise_dim] (None if empirical)
    env_params_batch: Array,    # [M, N, T, theta_dim]  or  [M, N, theta_dim]
    params: GatekeeperParams,
    rollout_all_fn: Callable,   # standard offline rollout
    uncertainty_data: Optional[JAXEmpiricalData] = None,
    rollout_all_empirical_fn: Optional[Callable] = None,
    rng_key: Optional[Array] = None,
    use_empirical_noise: bool = False,
    nominal_trajectory: Optional[Array] = None,
) -> Tuple[int, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, int, float, dict]:
    """Run one iteration of Algorithm 1 and return the selected switching time.

    Steps: roll out all M x N samples, count failures with the buffer
    L_H * beta^(1/p), compute q_bar for each candidate, and pick the largest
    certified switching time (or keep s_prev if none is certified).

    Parameters
    ----------
    x0 : [STATE_DIM]
        Current state x_t.
    s_prev : int
        Previous switching time s*_{t-1} (absolute time step).
    t : int
        Current time step.
    switching_offsets : [M] int32
        Candidate offsets s - t, in increasing order (0, ..., M-1).
    noise_batch : [M, N, T, noise_dim] or None
        Pre-sampled noise for Gaussian mode; None in empirical mode.
    env_params_batch : [M, N, T, theta_dim] or [M, N, theta_dim]
        theta samples (per step in Gaussian mode, per rollout in empirical
        mode).
    params : GatekeeperParams
    rollout_all_fn, rollout_all_empirical_fn : callable
        Batched rollouts from build_rollout_all / build_rollout_all_empirical.
    uncertainty_data : JAXEmpiricalData or None
        Unused here; the empirical data is bound into rollout_all_empirical_fn.
    rng_key : PRNG key or None
        Required in empirical mode.
    use_empirical_noise : bool
        Use the empirical rollout instead of the pre-sampled one.
    nominal_trajectory : [>=T, ACTION_DIM] or None
        Open-loop nominal actions (e.g. the MPPI plan).

    Returns
    -------
    s_t : int
        Selected switching time (absolute time step).
    I_valid : [M] bool
        Certified candidates.
    q_bars : [M] float32
        Upper bounds on the failure probability per candidate.
    L_H : float
        Lipschitz constant used.
    trajectories : [N, T+1, 2]
        x-y rollouts of the selected candidate (offset 0 if none certified).
    mask : [N] bool
        Failures among those rollouts, using the fixed L_H.
    m_star : int
        Offset of that candidate.
    q_bar_star : float
        q_bar of that candidate.
    timings : dict
        Per-stage timings, filled when params.debug_timing is True.
    """
    timings = {}
    compute_grad = (params.lipschitz_mode == "autodiff")
    
    L_H_fixed = float(params.lipschitz_constant)
    # Wasserstein failure buffer L_H * beta^(1/p); equals L_H * beta (Lemma 1) for p = 1.
    dr_buffer_init = L_H_fixed * (float(params.beta) ** (1.0 / float(params.p)))

    t0 = time.perf_counter() if params.debug_timing else None
    if use_empirical_noise:
        if rollout_all_empirical_fn is None or rng_key is None:
            raise ValueError("Empirical Gatekeeper mode requires empirical rollout function and rng_key.")
        # Rollouts (Alg. 1, lines 1-4). Empirical noise: one PRNG key per
        # (candidate, sample), so all M*N rollouts are independent.
        rng_matrix = jax.random.split(rng_key, params.M * params.N).reshape(params.M, params.N, 2)
        H_matrix, Grad_matrix, X_all = rollout_all_empirical_fn(
            x0, switching_offsets, rng_matrix, env_params_batch,
            compute_grad, nominal_trajectory
        )
    else:
        # Rollouts (Alg. 1, lines 1-4) with the pre-sampled Gaussian noise.
        H_matrix, Grad_matrix, X_all = rollout_all_fn(
            x0, switching_offsets, noise_batch, env_params_batch,
            compute_grad, nominal_trajectory
        )

    # Wait for the device so the debug timings are accurate.
    if params.debug_timing:
        H_matrix.block_until_ready()
        if Grad_matrix is not None:
            Grad_matrix.block_until_ready()
        X_all.block_until_ready()
        timings["jax_rollout"] = time.perf_counter() - t0

    # Failure counts k^s = #{i : H_i <= buffer} (line 6). Failures are counted
    # with <=, so H exactly at the buffer counts as a failure.
    t1 = time.perf_counter() if params.debug_timing else None
    failure_mask_init = H_matrix <= jnp.float32(dr_buffer_init)
    k_vec_jax = jnp.sum(failure_mask_init, axis=1).astype(jnp.int32)
    
    if compute_grad and Grad_matrix is not None:
        # Empirical Lipschitz estimate: L_H = max over (m, i) of ||dH/dw||_2,
        # with the gradient flattened over the T x noise_dim noise trajectory.
        grad_norms = jnp.linalg.norm(Grad_matrix.reshape(params.M, params.N, -1), axis=-1)
        L_H = float(jnp.max(grad_norms))
        
        # Recount the failures with the estimated L_H.
        dr_buffer = L_H * (float(params.beta) ** (1.0 / float(params.p)))
        failure_mask = H_matrix <= jnp.float32(dr_buffer)
        k_vec_jax = jnp.sum(failure_mask, axis=1).astype(jnp.int32)
    else:
        L_H = L_H_fixed

    k_vec = np.array(k_vec_jax)  # [M], copied to the host
    if params.debug_timing:
        timings["lipschitz_failure_counting"] = time.perf_counter() - t1

    # Sidak-corrected level and Clopper-Pearson bounds (lines 7-8).
    t2 = time.perf_counter() if params.debug_timing else None
    rho = 1- (1-params.delta)**(1/float(params.M))
    q_bars = compute_q_bar_vec(k_vec, params.N, rho)

    # Certified set I = {s : q_bar^s <= (eps - alpha) / (1 - alpha)} (line 8).
    feasibility_threshold = (params.epsilon - params.alpha) / (1.0 - params.alpha)
    I_valid = q_bars <= float(feasibility_threshold)

    # s*_t = max I, else keep s*_{t-1} (lines 9-12). Offsets are increasing,
    # so the last certified index is the largest switching time.
    abs_times = t + np.asarray(switching_offsets, dtype=int)
    if np.any(I_valid):
        m_idx_star = int(np.where(I_valid)[0][-1])
        s_t = int(abs_times[m_idx_star])
    else:
        m_idx_star = 0  # only selects which rollouts to display
        s_t = int(s_prev)

    # Rollouts of the selected candidate, for visualisation. X_all is
    # [M, N, T+1, 2]; the mask uses the fixed L_H.
    trajectories = np.asarray(X_all[m_idx_star], dtype=np.float32)
    mask = np.asarray(failure_mask_init[m_idx_star], dtype=bool)
    m_star = int(switching_offsets[m_idx_star])
    q_bar_star = float(q_bars[m_idx_star])

    if params.debug_timing:
        timings["cpu_selection"] = time.perf_counter() - t2

    return s_t, I_valid, q_bars, L_H, trajectories, mask, m_star, q_bar_star, timings


# ---------------------------------------------------------------------------
# Stateful wrapper
# ---------------------------------------------------------------------------


class DRSGatekeeper:
    """Stateful DRS-gatekeeper: samples noise, runs Algorithm 1, tracks s*_t.

    Typical use (as in clean_universal_simulator.py)::

        gk = DRSGatekeeper(params, dynamics_fn, nominal_fn, backup_fn,
                           safety_fn, pcis_fn, noise_dim=4,
                           uncertainty_model=env.uncertainty_model)
        gk.reset(x0_flat, t=0)
        for each step:
            state = gk.update(x_flat, nominal_trajectory=mppi_plan)
            action = backup_action if state.using_backup else nominal_action
            gk.tick()

    `step()` combines update, act and tick when the JAX policy functions are
    also the controllers being executed.

    Noise model
    -----------
    If `uncertainty_model` is given, the rollouts sample noise online from the
    empirical model (3 channels: delta_vx, delta_vy, delta_yaw_rate).
    Otherwise they use i.i.d. Gaussian residuals with `gaussian_noise_mean`
    and `gaussian_noise_std` (4 channels, adding delta_steer). Unlike the
    environment, these Gaussian samples are not clipped to +-4 sigma.

    Parameters
    ----------
    params : GatekeeperParams
    dynamics_fn : callable
        (state [STATE_DIM], action [ACTION_DIM], noise) -> next_state. JAX.
    nominal_policy_fn : callable
        (state,) -> action. Used in rollouts when no nominal_trajectory is
        passed, and by act().
    backup_policy_fn : callable
        (state,) -> action. The backup policy pi_B used in rollouts and act().
    safety_fn : callable
        h(state, theta) -> scalar, >= 0 when safe.
    pcis_fn : callable
        h_c(state) -> scalar, >= 0 inside the backup's invariant set C.
    noise_dim : int
        Stored for reference; the Gaussian sampler always draws 4 channels.
    theta_dim : int
        Size of the theta samples passed to safety_fn.
    noise_sampler : callable or None
        Optional replacement for the Gaussian sampler:
        noise_sampler(M, N, T, rng) -> (noise_batch, env_batch).
    uncertainty_model : EmpiricalUncertaintyModel or None
        Enables empirical-noise rollouts (see above).
    gaussian_noise_mean, gaussian_noise_std : sequence of 4 floats or None
        Gaussian residual parameters. Default to the environment's
        zero mean and std [0.45, 0.45, 0.25, 0.015].
    calibration_model : Any
        Unused; kept for compatibility.
    track : TrackModel or None
        Used for the curvature lookup in empirical-noise rollouts.
    vehicle_config : VehicleConfig or None
        Stored; not used by the filter.
    initial_track_bounds : TrackBoundsEstimate or None
        Distribution of theta. Defaults to TrackBoundsEstimate.unknown().
    seed : int
        Seed for noise and theta sampling.
    """

    def __init__(
        self,
        params: GatekeeperParams,
        dynamics_fn: Callable,
        nominal_policy_fn: Callable,
        backup_policy_fn: Callable,
        safety_fn: Callable,
        pcis_fn: Callable,
        noise_dim: int,
        theta_dim: int = 2,
        noise_sampler: Optional[Callable] = None,
        uncertainty_model: Optional[EmpiricalUncertaintyModel] = None,
        gaussian_noise_mean: Optional[Any] = None,
        gaussian_noise_std: Optional[Any] = None,
        calibration_model: Optional[Any] = None,
        track: Optional[Any] = None,
        vehicle_config: Optional[Any] = None,
        initial_track_bounds: Optional["TrackBoundsEstimate"] = None,
        seed: int = 0,
    ):
        """Set up sampling, JIT-compile the rollouts and fix the candidate offsets. See the class docstring for the arguments."""
        self.params = params
        self.dynamics_fn = dynamics_fn
        self.nominal_policy_fn = nominal_policy_fn
        self.backup_policy_fn = backup_policy_fn
        self.noise_sampler = noise_sampler
        self.uncertainty_model = uncertainty_model
        self.calibration_model = calibration_model
        self.track = track
        self.vehicle_config = vehicle_config
        self.noise_dim = noise_dim
        self.theta_dim = theta_dim
        self.gaussian_noise_mean = _normalize_gaussian_vector(
            gaussian_noise_mean,
            default=np.zeros(4, dtype=float),
        )
        self.gaussian_noise_std = _normalize_gaussian_vector(
            gaussian_noise_std,
            default=np.array([0.45, 0.45, 0.25, 0.015], dtype=float),
            nonnegative=True,
        )
        self._rng = np.random.default_rng(seed)
        self._key = jax.random.PRNGKey(seed)
        self.last_L_H = float(params.lipschitz_constant)
        self._curvature_fn = _build_track_curvature_fn(track)
        self._dynamic_empirical_gate = bool(
            uncertainty_model is not None
            and hasattr(uncertainty_model, "gate_prefixes")
            and len(uncertainty_model.gate_prefixes) == 1
        )

        # Distribution of theta (track boundaries).
        self._track_bounds: TrackBoundsEstimate = (
            initial_track_bounds
            if initial_track_bounds is not None
            else TrackBoundsEstimate.unknown()
        )

        # Empirical model as device arrays, for sampling inside the rollouts.
        self.uncertainty_data: Optional[JAXEmpiricalData] = None
        if uncertainty_model is not None:
            self.uncertainty_data = uncertainty_model.to_jax()

        # JIT-compiled batched rollouts: one for pre-sampled Gaussian noise and
        # one for empirical noise. compute_grad (argument index 4) is static.
        self._rollout_all = jax.jit(
            build_rollout_all(
                dynamics_fn=dynamics_fn,
                nominal_policy_fn=nominal_policy_fn,
                backup_policy_fn=backup_policy_fn,
                safety_fn=safety_fn,
                pcis_fn=pcis_fn,
                T=params.T,
            ),
            static_argnums=4,
        )
        
        if self.uncertainty_data is not None:
            # gate_id is fixed at 0 unless the model has a single gate prefix,
            # in which case the gate follows track progress
            # (self._dynamic_empirical_gate).
            self._rollout_all_empirical = jax.jit(
                build_rollout_all_empirical(
                    dynamics_fn=dynamics_fn,
                    nominal_policy_fn=nominal_policy_fn,
                    backup_policy_fn=backup_policy_fn,
                    safety_fn=safety_fn,
                    pcis_fn=pcis_fn,
                    T=params.T,
                    uncertainty_data=self.uncertainty_data,
                    gate_id=0, # Default, can be overridden if needed
                    curvature_fn=self._curvature_fn,
                    dynamic_gate_from_progress=self._dynamic_empirical_gate,
                ),
                static_argnums=4,
            )

        # Candidate offsets s - t in {0, ..., M-1}, i.e. s in {t, ..., t+M-1}.
        self._switching_offsets = jnp.arange(0, params.M, dtype=jnp.int32)

        # Runtime state
        self.state: Optional[GatekeeperState] = None
        self.last_timings: dict = {}

    # ------------------------------------------------------------------
    # Noise sampling
    # ------------------------------------------------------------------

    def _sample_noise(
        self,
        track_bounds: Optional["TrackBoundsEstimate"] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the Gaussian-mode inputs for one update.

        Returns
        -------
        noise_batch : [M, N, T, 4] float32
            i.i.d. N(gaussian_noise_mean, gaussian_noise_std^2) residuals on
            [delta_vx, delta_vy, delta_yaw_rate, delta_steer] (not clipped).
        env_batch : [M, N, T, theta_dim] float32
            theta drawn from `track_bounds` (or the stored estimate), one sample
            per rollout repeated over the T steps.

        If a custom `noise_sampler` was given, its output is returned instead.
        """
        p = self.params
        M, N, T = p.M, p.N, p.T
        bounds = track_bounds if track_bounds is not None else self._track_bounds

        if self.noise_sampler is not None:
            # A custom sampler replaces both the noise and theta sampling.
            return self.noise_sampler(M, N, T, self._rng)

        # Process noise w^f: Gaussian residuals on
        # [delta_vx, delta_vy, delta_yaw_rate, delta_steer]. jax_env.py clips
        # its samples to +-4 sigma; these are not clipped.
        self._key, subkey_f = jax.random.split(self._key)
        noise_batch = np.asarray(
            jax.random.normal(subkey_f, shape=(M, N, T, 4), dtype=jnp.float32)
            * jnp.asarray(self.gaussian_noise_std, dtype=jnp.float32)
            + jnp.asarray(self.gaussian_noise_mean, dtype=jnp.float32)
        )

        # Unsafe-set parameters w^theta from the track-bounds estimate.
        env_batch = bounds.sample_env_params(M, N, T, self._rng)  # [M,N,T,2]

        # If theta_dim != 2, pad or truncate to match
        if self.theta_dim != 2:
            if self.theta_dim == 0:
                env_batch = np.zeros((M, N, T, 1), dtype=np.float32)
            elif self.theta_dim > 2:
                pad = np.zeros((M, N, T, self.theta_dim - 2), dtype=np.float32)
                env_batch = np.concatenate([env_batch, pad], axis=-1)
            else:
                env_batch = env_batch[..., :self.theta_dim]

        return noise_batch, env_batch

    def _sample_env_params_only(
        self,
        track_bounds: Optional["TrackBoundsEstimate"] = None,
    ) -> np.ndarray:
        """Sample theta only, for empirical mode (noise is sampled inside the rollouts).

        Returns [M, N, theta_dim]: one static theta per rollout.
        """
        p = self.params
        M, N = p.M, p.N
        bounds = track_bounds if track_bounds is not None else self._track_bounds

        env_batch = bounds.sample_env_params(M, N, 1, self._rng)  # [M, N, 1, 2]
        env_batch = env_batch[:, :, 0, :]  # [M, N, 2]

        # If theta_dim != 2, pad or truncate to match
        if self.theta_dim != 2:
            if self.theta_dim == 0:
                env_batch = np.zeros((M, N, 1), dtype=np.float32)
            elif self.theta_dim > 2:
                pad = np.zeros((M, N, self.theta_dim - 2), dtype=np.float32)
                env_batch = np.concatenate([env_batch, pad], axis=-1)
            else:
                env_batch = env_batch[..., :self.theta_dim]

        return env_batch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, x0_flat: Array, t: int = 0) -> None:
        """Start a new episode at time t.

        s_prev is initialized to t, so if the first update certifies no
        candidate the backup policy is used immediately (the paper initialises
        s*_{-1} = 0).
        """
        # s_prev = t: the backup is used until some switching time is certified.
        self.state = GatekeeperState(t=t, s_prev=t, using_backup=False, plan_start_t=t)

    def update(
        self,
        x_flat: Array,
        track_bounds: Optional["TrackBoundsEstimate"] = None,
        nominal_trajectory: Optional[Array] = None,
        max_steps: Optional[int] = None,
    ) -> GatekeeperState:
        """Run Algorithm 1 at the current time and return the updated state.

        Does not advance time; call `tick()` after applying the action.

        Parameters
        ----------
        x_flat : [STATE_DIM]
            Current state x_t.
        track_bounds : TrackBoundsEstimate or None
            New theta distribution; None keeps the stored one.
        nominal_trajectory : [>=T, ACTION_DIM] or None
            Open-loop nominal actions for the rollouts (e.g. the MPPI plan).
        max_steps : int or None
            Only used in the debug timing printout.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before update().")

        if track_bounds is not None:
            self._track_bounds = track_bounds

        use_empirical = self.uncertainty_data is not None
        # Sample this update's random inputs.
        t_sample_start = time.perf_counter()

        noise_batch_jax = None
        rng_key_jax = None
        rollout_emp_fn = None

        if use_empirical:
            if self.uncertainty_data is None:
                raise ValueError(
                    "Gatekeeper empirical mode requested, but no empirical uncertainty model is available."
                )
            env_batch = self._sample_env_params_only(self._track_bounds)
            env_batch_jax = jnp.asarray(env_batch)
            # Empirical mode: noise is sampled inside the rollouts, so only
            # theta and a PRNG key are prepared here.
            self._key, rng_key_jax = jax.random.split(self._key)
            rollout_emp_fn = self._rollout_all_empirical
        else:
            # Gaussian mode: pre-sample noise and theta.
            noise_batch, env_batch = self._sample_noise(self._track_bounds)
            noise_batch_jax = jnp.asarray(noise_batch)
            env_batch_jax = jnp.asarray(env_batch)

        t_sample = time.perf_counter() - t_sample_start

        s_t, I_valid, q_bars, L_H, trajectories, mask, m_star, q_bar_star, timings = run_gatekeeper(
            x0=x_flat,
            s_prev=self.state.s_prev,
            t=self.state.t,
            switching_offsets=self._switching_offsets,
            noise_batch=noise_batch_jax,
            env_params_batch=env_batch_jax,
            params=self.params,
            rollout_all_fn=self._rollout_all,
            uncertainty_data=self.uncertainty_data,
            rollout_all_empirical_fn=rollout_emp_fn,
            rng_key=rng_key_jax,
            use_empirical_noise=use_empirical,
            nominal_trajectory=nominal_trajectory,
        )

        timings["sampling"] = t_sample
        total_time = sum(timings.values())

        if self.params.debug_timing:
            progress_str = f"Step {self.state.t}"
            if max_steps is not None:
                progress_str += f"/{max_steps}"
            print(f"\n--- DRS-GK Timing [{progress_str}] ---")
            print(f"  Sampling:          {timings['sampling']*1000:7.2f} ms")
            print(f"  JAX Rollout:       {timings['jax_rollout']*1000:7.2f} ms")
            print(f"  Lipschitz/Fail:    {timings['lipschitz_failure_counting']*1000:7.2f} ms")
            print(f"  CPU Selection:     {timings['cpu_selection']*1000:7.2f} ms")
            print(f"  --------------------------")
            print(f"  Total Update:      {total_time*1000:7.2f} ms")

        self.last_L_H = L_H
        self.last_timings = timings
        
        # Was any candidate certified, and is the backup active now?
        found_new = np.any(I_valid)
        using_backup = self.state.t >= s_t

        # Visualisation bookkeeping: keep the rollouts from when the current
        # switching time was first chosen until a different one is certified.
        # Note that tick() clears these fields, so in the simulator loop the
        # latest rollouts are shown.
        is_new_commitment = found_new and (s_t != self.state.s_prev)

        if is_new_commitment:
            # New switching time: store this update's rollouts.
            self.state = GatekeeperState(
                t=self.state.t,
                s_prev=s_t,
                using_backup=using_backup,
                predicted_trajectories=trajectories,
                failure_mask=mask,
                m_star=m_star,
                q_bar_star=q_bar_star, # Bound from when the switch was chosen
                is_reverting=False,
                plan_start_t=self.state.t, # Mark the birth of this plan
            )
        else:
            # Same switching time, or nothing certified: keep s_prev and the
            # stored rollouts.
            current_m_star = self.state.s_prev - self.state.t
            
            retained_trajectores = self.state.predicted_trajectories
            retained_mask = self.state.failure_mask
            retained_q_bar = self.state.q_bar_star
            retained_start_t = self.state.plan_start_t
            
            # Nothing stored yet (first update, or after tick()): show the
            # current rollouts.
            if retained_trajectores is None:
                retained_trajectores = trajectories
                retained_mask = mask
                retained_q_bar = q_bar_star
                retained_start_t = self.state.t

            self.state = GatekeeperState(
                t=self.state.t,
                s_prev=self.state.s_prev,
                using_backup=using_backup,
                predicted_trajectories=retained_trajectores,
                failure_mask=retained_mask,
                m_star=current_m_star,
                q_bar_star=retained_q_bar,
                is_reverting=(not found_new),
                plan_start_t=retained_start_t,
            )
        return self.state

    def act(self, x_flat: Array) -> Array:
        """Return pi_B(x) if the backup is active (t >= s*_t), else pi_N(x).

        Uses the JAX policy functions given to the constructor. The simulator
        instead applies its own controllers based on `state.using_backup`.
        """
        if self.state is None:
            raise RuntimeError("Call reset() and update() before act().")

        if self.state.using_backup:
            return self.backup_policy_fn(x_flat)
        else:
            return self.nominal_policy_fn(x_flat)

    def tick(self) -> None:
        """Advance time by one step and update `using_backup`. Resets the visualisation fields."""
        if self.state is None:
            raise RuntimeError("Call reset() before tick().")
        t_new = self.state.t + 1
        using_backup = t_new >= self.state.s_prev
        self.state = GatekeeperState(
            t=t_new,
            s_prev=self.state.s_prev,
            using_backup=using_backup,
        )

    def step(
        self,
        x_flat: Array,
        track_bounds: Optional["TrackBoundsEstimate"] = None,
        nominal_trajectory: Optional[Array] = None,
    ) -> Array:
        """Run update(), act() and tick() in one call and return the action.

        Only appropriate when the constructor's policy functions are the
        controllers actually being executed.
        """
        self.update(x_flat, track_bounds=track_bounds, nominal_trajectory=nominal_trajectory)
        action = self.act(x_flat)
        self.tick()
        return action

    # ------------------------------------------------------------------
    # Conversions to the flat state vector
    # ------------------------------------------------------------------

    @staticmethod
    def state_from_jax_racecar(jax_state) -> Array:
        """Convert a JaxRacecarState into the flat STATE_DIM vector."""
        return jnp.array(
            [
                jax_state.x,
                jax_state.y,
                jax_state.yaw,
                jax_state.vx,
                jax_state.vy,
                jax_state.yaw_rate,
                jax_state.steer,
                jax_state.throttle,
                jax_state.brake,
                jax_state.progress,
                jax_state.lateral_error,
                jax_state.heading_error,
            ],
            dtype=jnp.float32,
        )

    @staticmethod
    def state_from_vehicle_state(vehicle_state) -> Array:
        """Convert a VehicleState (uncertain_racecar_gym.dynamics) into the flat STATE_DIM vector."""
        return jnp.array(
            [
                vehicle_state.x,
                vehicle_state.y,
                vehicle_state.yaw,
                vehicle_state.vx,
                vehicle_state.vy,
                vehicle_state.yaw_rate,
                vehicle_state.steer,
                vehicle_state.throttle,
                vehicle_state.brake,
                vehicle_state.progress,
                vehicle_state.lateral_error,
                vehicle_state.heading_error,
            ],
            dtype=jnp.float32,
        )


# ---------------------------------------------------------------------------
# Racecar safety function h and invariant-set function h_c
# ---------------------------------------------------------------------------


def make_track_safety_fn(road_half_width: float) -> Callable:
    """Return the racecar safety function h(state, theta) = road_half_width - |lateral_error|.

    h >= 0 inside the track. theta is ignored: the track boundaries are taken
    as known, so uncertainty in the boundaries is not part of the
    certificate.
    """
    half_w = jnp.float32(road_half_width)

    def safety_fn(state: Array, env_param: Array) -> Array:
        lateral_error = state[10]
        return half_w - jnp.abs(lateral_error)

    return safety_fn


def make_track_pcis_fn(pcis_half_width: float) -> Callable:
    """Return h_c(state) = pcis_half_width - |lateral_error|.

    C = {h_c >= 0} is the set the backup policy is assumed to keep the car in
    (the alpha-PCIS of Assumption 1). The paper's racecar uses 2.75 m with a
    3.0 m track half-width.
    """
    half_w = jnp.float32(pcis_half_width)

    def pcis_fn(state: Array) -> Array:
        lateral_error = state[10]
        return half_w - jnp.abs(lateral_error)

    return pcis_fn
