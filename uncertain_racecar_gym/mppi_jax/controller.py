"""Model Predictive Path Integral (MPPI) controller in JAX, used as the nominal policy pi_N."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from uncertain_racecar_gym.controllers import BenchmarkController, ControllerStepContext
from uncertain_racecar_gym.mppi_support import (
    MPPICostConfig,
    action_plan_tail,
    build_heuristic_driver,
    build_nominal_params,
    build_rollout_cost_fn,
    build_rollout_positions_fn,
    build_speed_profile_arrays,
    clip_action,
    padded_action_history,
    rollout_heuristic_plan,
    softmax_weights,
    vehicle_state_to_jax,
)


@dataclass(slots=True)
class JaxMPPIConfig:
    """MPPI settings: horizon, samples, iterations, replanning interval, exploration noise and cost weights."""
    horizon: int = 24
    num_samples: int = 512
    optimization_steps: int = 2
    replan_interval: int = 2
    lambda_: float = 12.0
    gamma_: float = 0.015
    action_noise_std: tuple[float, float, float] = (0.24725, 0.225, 0.15)
    action_l2_weight: float = 0.01
    action_diff_weight: float = 0.10
    offtrack_penalty: float = 4000.0
    progress_weight: float = 3200.0
    lateral_weight: float = 28.0
    heading_weight: float = 18.0
    speed_tracking_weight: float = 0.30
    target_speed: float = 60.0
    min_speed: float = 8.0
    speed_profile_quantile: float = 0.68
    speed_profile_scale: float = 0.62
    driver_dataset: str | None = None
    debug_render_plans: bool = True
    debug_num_trajectories: int = 100
    seed: int = 0


class JaxMPPIController(BenchmarkController):
    """MPPI controller over the noise-free JAX dynamics.

    Every `replan_interval` steps it shifts the previous plan, perturbs it
    with Gaussian noise, and updates it with cost-weighted averages
    (`optimization_steps` times). Between replans it repeats the last action
    and the plan is not shifted, so `_action_plan[0]` is always the action
    being applied. The gatekeeper uses `_action_plan` as the nominal
    trajectory.
    """
    def __init__(self, scenario: str, config: JaxMPPIConfig | None = None):
        self.config = config or JaxMPPIConfig()
        self.name = "mppi_jax"
        self.params, self.scenario = build_nominal_params(scenario)
        self.driver = build_heuristic_driver(
            driver_dataset=self.config.driver_dataset,
            speed_profile_quantile=self.config.speed_profile_quantile,
            speed_profile_scale=self.config.speed_profile_scale,
            target_speed=self.config.target_speed,
            min_speed=self.config.min_speed,
        )
        self.cost_config = MPPICostConfig(
            lambda_=self.config.lambda_,
            gamma_=self.config.gamma_,
            offtrack_penalty=self.config.offtrack_penalty,
            action_l2_weight=self.config.action_l2_weight,
            action_diff_weight=self.config.action_diff_weight,
            progress_weight=self.config.progress_weight,
            lateral_weight=self.config.lateral_weight,
            heading_weight=self.config.heading_weight,
            speed_tracking_weight=self.config.speed_tracking_weight,
            target_speed=self.config.target_speed,
        )
        self.speed_profile = build_speed_profile_arrays(
            driver_dataset=self.config.driver_dataset,
            speed_profile_quantile=self.config.speed_profile_quantile,
            speed_profile_scale=self.config.speed_profile_scale,
            min_speed=self.config.min_speed,
        )
        self._rollout_costs = build_rollout_cost_fn(self.params, self.cost_config, speed_profile=self.speed_profile)
        self._rollout_positions = build_rollout_positions_fn(self.params)
        self._key = jax.random.PRNGKey(self.config.seed)
        self._action_plan = jnp.zeros((self.config.horizon, 3), dtype=jnp.float32)
        self._initialized = False
        self._last_action = np.zeros(3, dtype=np.float32)
        self._cached_action = np.zeros(3, dtype=np.float32)
        self._last_replan_step = -10**9
        self._latest_render_debug: dict[str, np.ndarray] | None = None

    def reset(self, *, seed: int | None = None, case_id: str | None = None) -> None:
        """Clear the plan and reseed for a new episode."""
        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed) + self.config.seed)
        self._action_plan = jnp.zeros((self.config.horizon, 3), dtype=jnp.float32)
        self._initialized = False
        self._last_action = np.zeros(3, dtype=np.float32)
        self._cached_action = np.zeros(3, dtype=np.float32)
        self._last_replan_step = -10**9
        self._latest_render_debug = None

    def _ensure_initialized(self, context: ControllerStepContext) -> None:
        """Warm-start the plan by rolling out the heuristic driver on the first call."""
        if self._initialized:
            return
        if context.env is None or getattr(context.env, "_state", None) is None:
            raise ValueError("JaxMPPIController requires a live env with `_state` for warm-start planning.")
        heuristic_plan = rollout_heuristic_plan(
            state=context.env._state,
            track=context.env.track,
            scenario=context.env.scenario,
            horizon=self.config.horizon,
            driver=self.driver,
        )
        self._action_plan = jnp.asarray(heuristic_plan, dtype=jnp.float32)
        self._cached_action = np.asarray(heuristic_plan[0], dtype=np.float32)
        self._initialized = True

    def _current_state(self, context: ControllerStepContext):
        assert context.env is not None and context.env._state is not None
        history = padded_action_history(context.env._history, self.scenario.uncertainty.history_length)
        return vehicle_state_to_jax(context.env._state, history)

    def _shift_plan(self, context: ControllerStepContext) -> None:
        """Drop the first action of the plan and append a heuristic action."""
        assert context.env is not None and context.env._state is not None
        tail = action_plan_tail(state=context.env._state, track=context.env.track, driver=self.driver)
        shifted = np.roll(np.asarray(self._action_plan, dtype=np.float32), shift=-1, axis=0)
        shifted[-1] = tail
        self._action_plan = jnp.asarray(shifted, dtype=jnp.float32)

    def _optimize(self, state, context: ControllerStepContext) -> np.ndarray:
        """Run the MPPI updates from `state`, store the new plan and return its first action."""
        sigma = jnp.asarray(self.config.action_noise_std, dtype=jnp.float32)
        base_plan = self._action_plan
        last_candidates = None
        for _ in range(self.config.optimization_steps):
            self._key, noise_key = jax.random.split(self._key)
            noise = jax.random.normal(noise_key, shape=(self.config.num_samples, self.config.horizon, 3), dtype=jnp.float32) * sigma
            candidate_actions = clip_action(base_plan[None, :, :] + noise)
            costs = self._rollout_costs(state, candidate_actions)
            perturbation_cost = self.cost_config.gamma_ * jnp.sum(jnp.square(noise / jnp.maximum(sigma, 1e-6)), axis=(1, 2))
            total_costs = costs + perturbation_cost
            weights = softmax_weights(total_costs, self.config.lambda_)
            weighted_noise = jnp.tensordot(weights, noise, axes=(0, 0))
            base_plan = clip_action(base_plan + weighted_noise)
            last_candidates = candidate_actions
        self._action_plan = base_plan
        if self.config.debug_render_plans and last_candidates is not None:
            limit = min(int(self.config.debug_num_trajectories), int(last_candidates.shape[0]))
            sampled = np.asarray(self._rollout_positions(state, last_candidates[:limit]), dtype=np.float32)
            final_plan = np.asarray(self._rollout_positions(state, base_plan[None, :, :])[0], dtype=np.float32)
            self._latest_render_debug = {
                "candidate_xy": sampled,
                "final_xy": final_plan,
            }
        action = np.asarray(base_plan[0], dtype=np.float32)
        self._last_action = action
        self._cached_action = action
        return action

    def get_render_debug(self) -> dict[str, np.ndarray] | None:
        """Sampled and final plan paths from the last optimisation, for the video overlay."""
        return self._latest_render_debug

    def act(self, observation: np.ndarray, *, context: ControllerStepContext) -> np.ndarray:
        """Return the action for this step, replanning every `replan_interval` steps."""
        self._ensure_initialized(context)
        if context.step_index - self._last_replan_step < self.config.replan_interval:
            return self._cached_action
        self._shift_plan(context)
        state = self._current_state(context)
        action = self._optimize(state, context)
        self._last_replan_step = context.step_index
        return action
