"""DRS-gatekeeper: a distributionally robust, backup-based safety filter.

Implements Algorithm 1 of Cherenson et al., "Distributionally Robust
Safety Under Arbitrary Uncertainties: A Safety Filtering Approach",
IEEE RA-L 2026, for the racecar in uncertain_racecar_gym.

Public API
----------
DRSGatekeeper
    Main entry point. Call reset(x0, t=0) once per episode, then update(x)
    and tick() every step; state.using_backup says which policy to apply.
GatekeeperParams
    Hyperparameters M, T, N, delta, epsilon, beta, alpha and the Lipschitz
    settings.
GatekeeperState
    Runtime state: current time and switching time, plus visualisation data.
run_gatekeeper
    One iteration of Algorithm 1 (rollouts, failure counts, selection).
build_rollout_all, rollout_single
    Batched and single rollouts of the switching policy.
compute_q_bar
    Clopper-Pearson upper bound on a failure probability (Eq. 10).
make_track_safety_fn, make_track_pcis_fn
    Racecar safety function h and backup invariant-set function h_c,
    both based on |lateral_error|.
TrackBoundsEstimate
    Distribution of the track-boundary parameters theta (track_bounds.py).
pack_state, unpack_state, STATE_DIM, ACTION_DIM
    Flat state helpers.

State layout (STATE_DIM = 12)
-----------------------------
[x, y, yaw, vx, vy, yaw_rate, steer, throttle, brake,
 progress, lateral_error, heading_error]

Actions (ACTION_DIM = 3): [steer, throttle, brake].
"""

from drs_gatekeeper.track_bounds import TrackBoundsEstimate
from drs_gatekeeper.drs_gatekeeper import (
    STATE_DIM,
    ACTION_DIM,
    GatekeeperParams,
    GatekeeperState,
    DRSGatekeeper,
    run_gatekeeper,
    build_rollout_all,
    rollout_single,
    compute_q_bar,
    pack_state,
    unpack_state,
    make_track_safety_fn,
    make_track_pcis_fn,
)

__all__ = [
    "STATE_DIM",
    "ACTION_DIM",
    "GatekeeperParams",
    "GatekeeperState",
    "TrackBoundsEstimate",
    "DRSGatekeeper",
    "run_gatekeeper",
    "build_rollout_all",
    "rollout_single",
    "compute_q_bar",
    "pack_state",
    "unpack_state",
    "make_track_safety_fn",
    "make_track_pcis_fn",
]

