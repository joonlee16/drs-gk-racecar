"""Helpers for the JAX MPPI controller: cost rollouts, warm-start plans and conversions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from uncertain_racecar_gym.controllers import CenterlineDriver, ProfiledCenterlineDriver, build_speed_profile
from uncertain_racecar_gym.dynamics import DynamicBicycleModel, VehicleState
from uncertain_racecar_gym.jax_env import JaxRacecarState, NominalJaxEnvParams, build_nominal_jax_params, step_nominal
from uncertain_racecar_gym.scenario import Scenario
from uncertain_racecar_gym.track import TrackModel


Array = jax.Array


@dataclass(slots=True)
class MPPICostConfig:
    """MPPI temperature and stage-cost weights."""
    lambda_: float = 8.0
    gamma_: float = 0.02
    offtrack_penalty: float = 2500.0
    action_l2_weight: float = 0.02
    action_diff_weight: float = 0.4
    progress_weight: float = 2500.0
    lateral_weight: float = 20.0
    heading_weight: float = 10.0
    speed_tracking_weight: float = 0.25
    target_speed: float = 18.0


def clip_action(action: Array) -> Array:
    """Clip actions to their ranges and keep only the larger of throttle and brake."""
    steer = jnp.clip(action[..., 0], -1.0, 1.0)
    throttle = jnp.clip(action[..., 1], 0.0, 1.0)
    brake = jnp.clip(action[..., 2], 0.0, 1.0)
    throttle_dominant = throttle >= brake
    throttle = jnp.where(throttle_dominant, throttle, 0.0)
    brake = jnp.where(throttle_dominant, 0.0, brake)
    return jnp.stack([steer, throttle, brake], axis=-1)


def padded_action_history(history: deque[np.ndarray] | list[np.ndarray], history_length: int) -> np.ndarray:
    """Last `history_length` actions, zero-padded at the front."""
    items = [np.asarray(item, dtype=np.float32) for item in history]
    while len(items) < history_length:
        items.insert(0, np.zeros(3, dtype=np.float32))
    return np.asarray(items[-history_length:], dtype=np.float32)


def vehicle_state_to_jax(state: VehicleState, action_history: np.ndarray) -> JaxRacecarState:
    """Convert a VehicleState and action history into a JaxRacecarState."""
    return JaxRacecarState(
        x=jnp.asarray(state.x, dtype=jnp.float32),
        y=jnp.asarray(state.y, dtype=jnp.float32),
        yaw=jnp.asarray(state.yaw, dtype=jnp.float32),
        progress=jnp.asarray(state.progress, dtype=jnp.float32),
        lateral_error=jnp.asarray(state.lateral_error, dtype=jnp.float32),
        heading_error=jnp.asarray(state.heading_error, dtype=jnp.float32),
        vx=jnp.asarray(state.vx, dtype=jnp.float32),
        vy=jnp.asarray(state.vy, dtype=jnp.float32),
        yaw_rate=jnp.asarray(state.yaw_rate, dtype=jnp.float32),
        steer=jnp.asarray(state.steer, dtype=jnp.float32),
        throttle=jnp.asarray(state.throttle, dtype=jnp.float32),
        brake=jnp.asarray(state.brake, dtype=jnp.float32),
        wheel_rotation=jnp.asarray(state.wheel_rotation, dtype=jnp.float32),
        lap_count=jnp.asarray(state.lap_count, dtype=jnp.int32),
        step_count=jnp.asarray(state.step_count, dtype=jnp.int32),
        action_history=jnp.asarray(action_history, dtype=jnp.float32),
    )


def build_heuristic_driver(
    *,
    driver_dataset: str | None = None,
    speed_profile_quantile: float = 0.65,
    speed_profile_scale: float = 0.55,
    target_speed: float = 18.0,
    min_speed: float = 8.0,
) -> CenterlineDriver | ProfiledCenterlineDriver:
    """Driver for warm starts: a speed-profile driver if driver_dataset is given, else CenterlineDriver."""
    if driver_dataset is not None:
        canonical = pd.read_parquet(driver_dataset)
        return ProfiledCenterlineDriver.from_canonical_dataframe(
            canonical,
            speed_quantile=speed_profile_quantile,
            speed_scale=speed_profile_scale,
            min_speed=min_speed,
        )
    return CenterlineDriver(target_speed=target_speed, min_speed=min_speed)


def build_speed_profile_arrays(
    *,
    driver_dataset: str | None = None,
    speed_profile_quantile: float = 0.65,
    speed_profile_scale: float = 0.55,
    min_speed: float = 8.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Speed profile (progress, speed) from driver_dataset for the MPPI cost, or None."""
    if driver_dataset is None:
        return None
    canonical = pd.read_parquet(driver_dataset)
    progress_points, target_speeds = build_speed_profile(
        canonical,
        progress_bins=160,
        speed_quantile=speed_profile_quantile,
        speed_scale=speed_profile_scale,
        min_speed=min_speed,
    )
    progress_points = np.concatenate([progress_points, np.array([1.0], dtype=float)])
    target_speeds = np.concatenate([target_speeds, np.array([target_speeds[0]], dtype=float)])
    return progress_points.astype(np.float32), target_speeds.astype(np.float32)


def rollout_heuristic_plan(
    *,
    state: VehicleState,
    track: TrackModel,
    scenario: Scenario,
    horizon: int,
    driver: CenterlineDriver | ProfiledCenterlineDriver,
) -> np.ndarray:
    """Roll out the heuristic driver with the NumPy model (no noise) to get an initial action plan."""
    dynamics = DynamicBicycleModel(scenario.vehicle)
    simulated = VehicleState(
        x=state.x,
        y=state.y,
        yaw=state.yaw,
        progress=state.progress,
        lateral_error=state.lateral_error,
        heading_error=state.heading_error,
        vx=state.vx,
        vy=state.vy,
        yaw_rate=state.yaw_rate,
        steer=state.steer,
        throttle=state.throttle,
        brake=state.brake,
        wheel_rotation=state.wheel_rotation,
        lap_count=state.lap_count,
        step_count=state.step_count,
    )
    actions: list[np.ndarray] = []
    for _ in range(horizon):
        action = np.asarray(driver.act(simulated, track), dtype=np.float32)
        action[1:] = np.clip(action[1:], 0.0, 1.0)
        if action[1] >= action[2]:
            action[2] = 0.0
        else:
            action[1] = 0.0
        actions.append(action)
        simulated = dynamics.step(simulated, action, track, scenario.simulation.dt, residual=None)
    return np.asarray(actions, dtype=np.float32)


def action_plan_tail(
    *,
    state: VehicleState,
    track: TrackModel,
    driver: CenterlineDriver | ProfiledCenterlineDriver,
) -> np.ndarray:
    """Heuristic action appended to the end of the shifted plan."""
    action = np.asarray(driver.act(state, track), dtype=np.float32)
    action[1:] = np.clip(action[1:], 0.0, 1.0)
    if action[1] >= action[2]:
        action[2] = 0.0
    else:
        action[1] = 0.0
    return action


def build_rollout_cost_fn(
    params: NominalJaxEnvParams,
    cost_config: MPPICostConfig,
    speed_profile: tuple[np.ndarray, np.ndarray] | None = None,
) -> Any:
    """Return a JIT-compiled, vmapped function giving the cost of each action sequence.

    Rollouts use step_nominal without noise and stop accumulating cost after
    leaving the track (plus offtrack_penalty).
    """
    speed_profile_points = None
    speed_profile_values = None
    if speed_profile is not None:
        speed_profile_points = jnp.asarray(speed_profile[0], dtype=jnp.float32)
        speed_profile_values = jnp.asarray(speed_profile[1], dtype=jnp.float32)

    def single_rollout_cost(initial_state: JaxRacecarState, action_sequence: Array) -> Array:
        initial_progress = initial_state.progress
        initial_prev_action = initial_state.action_history[-1]

        def body(carry, action):
            state, active, prev_action, cumulative_cost = carry
            bounded_action = clip_action(action)
            step = step_nominal(params, state, bounded_action)
            delta_progress = step.state.progress - state.progress
            delta_progress = jnp.where(delta_progress < -0.5, delta_progress + 1.0, delta_progress)
            target_speed = cost_config.target_speed
            if speed_profile_points is not None and speed_profile_values is not None:
                target_speed = jnp.interp(step.state.progress, speed_profile_points, speed_profile_values)
            stage_cost = (
                -cost_config.progress_weight * delta_progress
                + cost_config.lateral_weight * jnp.square(step.state.lateral_error)
                + cost_config.heading_weight * jnp.square(step.state.heading_error)
                + cost_config.speed_tracking_weight * jnp.square(step.state.vx - target_speed)
                + cost_config.action_l2_weight * jnp.sum(jnp.square(bounded_action))
                + cost_config.action_diff_weight * jnp.sum(jnp.square(bounded_action - prev_action))
                + jnp.where(step.terminated, cost_config.offtrack_penalty, 0.0)
            )
            stage_cost = jnp.where(active, stage_cost, 0.0)
            next_state = jax.tree.map(lambda new, old: jnp.where(active, new, old), step.state, state)
            next_active = jnp.logical_and(active, jnp.logical_not(jnp.logical_or(step.terminated, step.truncated)))
            next_cost = cumulative_cost + stage_cost
            return (next_state, next_active, bounded_action, next_cost), stage_cost

        carry, _ = jax.lax.scan(
            body,
            (
                initial_state,
                jnp.asarray(True),
                clip_action(initial_prev_action),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            action_sequence,
        )
        final_state, _, _, cumulative_cost = carry
        terminal_progress_bonus = -(final_state.lap_count.astype(jnp.float32) + final_state.progress - initial_progress) * cost_config.progress_weight * 0.25
        terminal_heading = 0.5 * jnp.abs(final_state.heading_error)
        terminal_lateral = 0.5 * jnp.abs(final_state.lateral_error)
        return cumulative_cost + terminal_progress_bonus + terminal_heading + terminal_lateral

    return jax.jit(jax.vmap(single_rollout_cost, in_axes=(None, 0)))


def build_rollout_positions_fn(params: NominalJaxEnvParams) -> Any:
    """Return a vmapped function giving the x-y path of each action sequence (for rendering)."""
    def single_rollout(initial_state: JaxRacecarState, action_sequence: Array) -> Array:
        def body(carry, action):
            state, active = carry
            bounded_action = clip_action(action)
            step = step_nominal(params, state, bounded_action)
            next_state = jax.tree.map(lambda new, old: jnp.where(active, new, old), step.state, state)
            next_active = jnp.logical_and(active, jnp.logical_not(jnp.logical_or(step.terminated, step.truncated)))
            xy = jnp.stack([next_state.x, next_state.y], axis=0)
            return (next_state, next_active), xy

        (_, _), xy_points = jax.lax.scan(
            body,
            (initial_state, jnp.asarray(True)),
            action_sequence,
        )
        start_xy = jnp.stack([initial_state.x, initial_state.y], axis=0)[None, :]
        return jnp.concatenate([start_xy, xy_points], axis=0)

    return jax.jit(jax.vmap(single_rollout, in_axes=(None, 0)))


def softmax_weights(costs: Array, temperature: float) -> Array:
    """MPPI weights exp(-(cost - min cost) / temperature), normalised."""
    minimum = jnp.min(costs)
    logits = -(costs - minimum) / jnp.maximum(temperature, 1e-6)
    logits = logits - jnp.max(logits)
    weights = jnp.exp(logits)
    return weights / jnp.sum(weights)


def build_nominal_params(scenario_path: str | Scenario) -> tuple[NominalJaxEnvParams, Scenario]:
    """Alias of build_nominal_jax_params."""
    return build_nominal_jax_params(scenario_path)
