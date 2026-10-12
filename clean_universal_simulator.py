#!/usr/bin/env python3
"""Racecar simulator with an optional DRS-gatekeeper safety filter.

Runs ITERATIONS episodes on the scenario's track and prints per-episode
and aggregate statistics. Each step:

  1. the nominal controller (MPPI or centerline) acts on a noisy copy of
     the state,
  2. if USE_GATEKEEPER, the DRS-gatekeeper (drs_gatekeeper package)
     checks the nominal plan and switches to the backup controller when
     needed,
  3. the JAX environment applies the action with the selected uncertainty
     model (none, gaussian or empirical).

Optionally renders each episode with PyBullet and saves an MP4 and/or a
replay bundle. Edit the constants in the configuration section below and
run `python clean_universal_simulator.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import time

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from drs_gatekeeper import (
    DRSGatekeeper,
    GatekeeperParams,
    TrackBoundsEstimate,
    make_track_pcis_fn,
    make_track_safety_fn,
)
from uncertain_racecar_gym.controllers import (
    CenterlineDriver,
    ControllerStepContext,
    FullBrakeDriver,
)
from uncertain_racecar_gym.dynamics import VehicleState
from uncertain_racecar_gym.jax_env import JaxRacecarState, NominalJaxRacecarEnv, _observation, step_nominal
from uncertain_racecar_gym.mppi_jax import JaxMPPIConfig, JaxMPPIController
from uncertain_racecar_gym.rendering import PyBulletMirrorRenderer
from uncertain_racecar_gym.replay import export_replay_bundle
from uncertain_racecar_gym.scenario import load_scenario
from uncertain_racecar_gym.track import TrackModel


# ============================================================
# Configuration
# ============================================================

SCENARIO = "package://scenarios/ks_barcelona_layout_gp_dallara_f317_rl_long.yaml"
CONTROLLER = "mppi_jax"                # nominal policy: centerline | mppi_jax
MAX_STEPS = 500000
SEED = 42                              # episode i uses seed SEED + i
ITERATIONS = 50                        # number of episodes
START_MODE = "grid"                    # grid | random; used only if START_PROGRESS is None
START_PROGRESS = 0.0                   # start position as a fraction of the lap

CONTROLLER_KWARGS = {
    "target_speed": 55.0,
}

UNCERTAINTY = "empirical"              # none | gaussian | empirical

# Empirical model file. The environment loads it whenever a path is set,
# whatever UNCERTAINTY is (a missing file is an error), and the gatekeeper
# then samples its rollout noise from it. With None, the environment still
# falls back to output/demo_empirical_model.pkl if that file exists.
UNCERTAINTY_ARTIFACT = "output/demo_empirical_model.pkl"

# --- DRS-gatekeeper (Algorithm 1); symbols follow the paper ---
USE_GATEKEEPER = True
BACKUP_CONTROLLER = "centerline"       # backup policy: centerline (slow to 6 m/s) | full_brake

GK_M = 10                              # M: number of candidate switching times
GK_T = 10                              # T: rollout horizon [steps]
GK_N = 1000                            # N: rollouts per candidate
GK_DELTA = 0.1                         # delta: certified with confidence 1 - delta
GK_EPSILON = 0.1                       # epsilon: allowed failure probability
GK_BETA = 0.25                         # beta: Wasserstein radius (paper: 0 gaussian, 0.25 empirical)
GK_ALPHA = 0.0                         # alpha: failure probability of the backup's invariant set
GK_P = 1                               # exponent in the buffer L_H * beta^(1/p); keep 1
LIPSCHITZ_MODE = "fixed"               # fixed | autodiff
LIPSCHITZ_CONSTANT = 2.0               # L_H for "fixed" (paper: 2.0)
GK_DEBUG_TIMING = False                # print per-update timings

# --- Rendering and output ---
RENDER_MODE = "none"                   # rgb_array_follow | rgb_array_birds_eye | rgb_array_cinematic | none
SAVE_VIDEOS = False                    # needs an rgb_array_* RENDER_MODE
SAVE_REPLAY_BUNDLES = False            # write trajectory + scene files per episode
OUTPUT = "highway_demo.mp4"            # base name: <stem>_iterNNN_seedS.mp4 / _bundle
FPS = 20                               # video frame rate
QUALITY = 8                            # imageio video quality (0-10)

# --- Track geometry [m] ---
TRACK_HALF_WIDTH = 3.0                 # gatekeeper safety function h: |lateral_error| <= this
PCIS_HALF_WIDTH = 2.75                 # backup invariant set C: |lateral_error| <= this
TRACK_WIDTH = 6.0                      # full width in the environment (off-track check)
ACTUAL_TRACK_HALF_WIDTH = None         # used only if TRACK_WIDTH is None

# --- Noise on the state seen by the controllers (not the gatekeeper) ---
CONTROLLER_PROGRESS_NOISE_STD = 0.0
CONTROLLER_LATERAL_NOISE_STD = 0.5     # [m]; paper: 0.5 empirical, 0.0 gaussian
CONTROLLER_HEADING_NOISE_STD = 0.0     # [rad]
CONTROLLER_SPEED_NOISE_STD = 0.0       # currently unused

# --- Residual scaling in the environment ---
# When enabled, the environment multiplies its residual by (1 + RESIDUAL_GAIN_STD).
RESIDUAL_GAIN_NOISE = False
RESIDUAL_GAIN_STD = 0.0


@dataclass
class SimulatorConfig:
    """All run settings; built from the module constants by build_config()."""
    scenario: str
    controller: str
    max_steps: int
    seed: int
    iterations: int
    start_mode: str
    start_progress: Optional[float]

    controller_kwargs: dict

    uncertainty: Optional[str]
    uncertainty_artifact: Optional[str]

    use_gatekeeper: bool
    backup_controller: str

    gk_m: int
    gk_t: int
    gk_n: int
    gk_delta: float
    gk_epsilon: float
    gk_beta: float
    gk_alpha: float
    gk_p: int
    lipschitz_mode: str
    lipschitz_constant: float
    gk_debug_timing: bool

    render_mode: str
    save_videos: bool
    output: str
    fps: int
    quality: int

    track_half_width: float
    pcis_half_width: Optional[float]
    track_width: Optional[float]
    actual_track_half_width: Optional[float]

    controller_progress_noise_std: float
    controller_lateral_noise_std: float
    controller_heading_noise_std: float
    controller_speed_noise_std: float

    residual_gain_noise: bool
    residual_gain_std: float


class CleanUniversalSimulator:
    """One environment, controller set and (optional) gatekeeper for one episode.

    BatchExperiment creates a new instance per episode.
    """
    def __init__(self, config: SimulatorConfig):
        self.cfg = config
        self.scenario_obj = load_scenario(config.scenario)
        self.actual_track_half_width = self._resolve_actual_track_half_width()
        if self.actual_track_half_width is not None:
            self.scenario_obj.track.width = float(self.actual_track_half_width) * 2.0

        self.is_jax = True
        self.env = NominalJaxRacecarEnv(
            self.scenario_obj,
            uncertainty_mode=self._normalized_uncertainty_mode(),
            uncertainty_artifact=self.cfg.uncertainty_artifact,
            residual_gain_noise=self.cfg.residual_gain_noise,
            residual_gain_std=self.cfg.residual_gain_std,
        )
        self.track = self.env.track if hasattr(self.env, "track") else TrackModel.from_config(self.scenario_obj.track)

        self.controller = None
        self.backup_controller = None
        self.gatekeeper: Optional[DRSGatekeeper] = None

        self.renderer: Optional[PyBulletMirrorRenderer] = None
        self.frames: list[np.ndarray] = []
        self.trajectory: list[dict] = []
        self._controller_noise_rng = np.random.default_rng(self.cfg.seed)

        # Template state reused when building observations.
        self._jax_state_template = JaxRacecarState(
            x=jnp.float32(0), y=jnp.float32(0), yaw=jnp.float32(0),
            vx=jnp.float32(0), vy=jnp.float32(0), yaw_rate=jnp.float32(0),
            steer=jnp.float32(0), throttle=jnp.float32(0), brake=jnp.float32(0),
            progress=jnp.float32(0), lateral_error=jnp.float32(0), heading_error=jnp.float32(0),
            wheel_rotation=jnp.float32(0), lap_count=jnp.int32(0), step_count=jnp.int32(0),
            action_history=self.env.params.simulation.history_template,
        )
        self._jitted_observation = jax.jit(lambda jax_state: _observation(self.env.params, jax_state))
        # Last observation, reused while the state barely changes (see _controller_action).
        self._last_obs_cache: dict[int, np.ndarray] = {}
        self._last_noisy_state: Optional[VehicleState] = None

        self._configure_controller()
        if self.cfg.use_gatekeeper:
            self._configure_gatekeeper()

    def _normalized_uncertainty_mode(self) -> Optional[str]:
        mode = "none" if self.cfg.uncertainty is None else str(self.cfg.uncertainty).strip().lower()
        if mode in {"", "none", "nominal"}:
            return None
        if mode not in {"gaussian", "empirical"}:
            raise ValueError(f"Unsupported uncertainty mode: {self.cfg.uncertainty}")
        return mode

    def _resolve_actual_track_half_width(self) -> Optional[float]:
        if self.cfg.track_width is not None:
            return float(self.cfg.track_width) / 2.0
        return self.cfg.actual_track_half_width

    def _configure_controller(self) -> None:
        kwargs = dict(self.cfg.controller_kwargs or {})
        if self.cfg.controller == "centerline":
            target_speed = kwargs.pop("target_speed", 12.0)
            min_speed = kwargs.pop("min_speed", 6.0)
            self.controller = CenterlineDriver(target_speed=target_speed, min_speed=min_speed)
        elif self.cfg.controller == "mppi_jax":
            self.controller = JaxMPPIController(self.cfg.scenario, JaxMPPIConfig(**kwargs))
        else:
            raise ValueError(f"Unknown controller: {self.cfg.controller}")

        if self.cfg.use_gatekeeper:
            if self.cfg.backup_controller == "centerline":
                self.backup_controller = CenterlineDriver(target_speed=6.0, min_speed=3.0)
            elif self.cfg.backup_controller == "full_brake":
                self.backup_controller = FullBrakeDriver()
            else:
                raise ValueError(f"Unknown backup controller: {self.cfg.backup_controller}")

    def _state_to_vehicle_state(self, state) -> VehicleState:
        return VehicleState(
            x=float(state.x),
            y=float(state.y),
            vx=float(state.vx),
            vy=float(state.vy),
            yaw=float(state.yaw),
            yaw_rate=float(state.yaw_rate),
            steer=float(state.steer),
            throttle=float(getattr(state, "throttle", 0.0)),
            brake=float(getattr(state, "brake", 0.0)),
            progress=float(state.progress),
            lateral_error=float(state.lateral_error),
            heading_error=float(state.heading_error),
            lap_count=int(state.lap_count),
            step_count=int(state.step_count),
            wheel_rotation=float(state.wheel_rotation),
        )

    def _noise_state(self, state: VehicleState) -> VehicleState:
        """Return a copy of the state with Gaussian noise added to what the controllers see.

        Noise (clipped to +-4 sigma) is added to progress, lateral_error and
        heading_error using the CONTROLLER_*_NOISE_STD settings. The true state,
        used by the environment and the gatekeeper, is unchanged.
        """
        rng = self._controller_noise_rng
        progress_noise_std = self.cfg.controller_progress_noise_std
        later_noise_std =  self.cfg.controller_lateral_noise_std
        heading_noise_std = self.cfg.controller_heading_noise_std
        return VehicleState(
            x=float(state.x),
            y=float(state.y),
            vx=max(0.0, float(state.vx)),
            vy=float(state.vy),
            yaw=float(state.yaw),
            yaw_rate=float(state.yaw_rate),
            steer=float(state.steer),
            throttle=float(getattr(state, "throttle", 0.0)),
            brake=float(getattr(state, "brake", 0.0)),
            progress=(float(state.progress) + jnp.clip(rng.normal(0.0, progress_noise_std), -4.0 * progress_noise_std, 4.0 * progress_noise_std)),
            lateral_error=float(state.lateral_error) + jnp.clip(rng.normal(0.0, later_noise_std), -4.0 * later_noise_std, 4.0 * later_noise_std),
            heading_error=float(state.heading_error) + jnp.clip(rng.normal(0.0, heading_noise_std), -4.0 * heading_noise_std, 4.0 * heading_noise_std),
            lap_count=int(state.lap_count),
            step_count=int(state.step_count),
            wheel_rotation=float(state.wheel_rotation),
        )

    def _observation_from_state(self, state: VehicleState) -> np.ndarray:
        """Build the environment observation vector for a VehicleState (JIT-compiled)."""
        jax_state = self._jax_state_template._replace(
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
        )
        return np.asarray(self._jitted_observation(jax_state), dtype=np.float32)

    def _controller_action(self, controller, state: VehicleState, *, step_index: int, mode: str) -> np.ndarray:
        """Return the action of `controller` for the current step.

        The controller sees the noisy state from _noise_state. MPPI ("Jax"
        controllers) gets an observation and a context with the environment;
        the centerline and full-brake drivers get the noisy VehicleState and the
        track directly.
        """
        noisy_state = self._noise_state(state)
        
        # Reuse the cached observation if x and vx moved by less than 0.01 m
        # and 0.1 m/s since it was computed (an approximation).
        controller_id = id(controller)
        if controller_id not in self._last_obs_cache or self._last_noisy_state is None or \
           (abs(self._last_noisy_state.x - noisy_state.x) > 0.01 or 
            abs(self._last_noisy_state.vx - noisy_state.vx) > 0.1):
            obs = self._observation_from_state(noisy_state)
            self._last_obs_cache[controller_id] = obs
            self._last_noisy_state = noisy_state
        else:
            obs = self._last_obs_cache[controller_id]

        # MPPI reads the (noisy) state, track and history from the environment.
        if isinstance(controller, (JaxMPPIController)):
            self.env._state = noisy_state
            if not hasattr(self.env, "_history"):
                self.env._history = []
            self.env.track = self.track
            self.env.scenario = self.scenario_obj

        if hasattr(controller, "act"):
            if "Jax" in controller.__class__.__name__:
                ctx = ControllerStepContext(
                    observation=obs,
                    step_index=step_index,
                    episode_index=0,
                    case_id="clean_sim",
                    mode=mode,
                    env=self.env,
                )
                action = controller.act(obs, context=ctx)
            else:
                action = controller.act(noisy_state, self.track)
        else:
            raise RuntimeError("Controller must implement act")

        return np.asarray(action, dtype=np.float32)

    def _configure_gatekeeper(self) -> None:
        """Build the DRS-gatekeeper for this episode.

        The rollout policies mirror the executed controllers: the backup is the
        centerline driver (6 m/s, brake only) or full brake, and an MPPI nominal
        is represented by its action plan, passed in each update.
        """
        if self.controller is None or self.backup_controller is None:
            raise RuntimeError("Controller and backup must be configured before gatekeeper")

        params = GatekeeperParams(
            M=self.cfg.gk_m,
            T=self.cfg.gk_t,
            N=self.cfg.gk_n,
            delta=self.cfg.gk_delta,
            epsilon=self.cfg.gk_epsilon,
            beta=self.cfg.gk_beta,
            alpha=self.cfg.gk_alpha,
            p=self.cfg.gk_p,
            lipschitz_mode=self.cfg.lipschitz_mode,
            lipschitz_constant=self.cfg.lipschitz_constant,
            debug_timing=self.cfg.gk_debug_timing,
        )

        safety_fn = make_track_safety_fn(road_half_width=self.cfg.track_half_width)
        pcis_fn = make_track_pcis_fn(
            pcis_half_width=self.cfg.pcis_half_width if self.cfg.pcis_half_width is not None else self.cfg.track_half_width
        )

        env_unwrapped = self.env
        uncertainty_model = getattr(env_unwrapped, "uncertainty_model", None)
        gaussian_noise_mean = getattr(env_unwrapped, "gaussian_noise_mean", None)
        gaussian_noise_std = getattr(env_unwrapped, "gaussian_noise_std", None)
        jax_params = self.env.params

        def centerline_rollout_policy(s):
            lat_err = s[10]
            head_err = s[11]
            vx = s[3]
            speed_error = jnp.float32(6.0) - vx
            steer = jnp.clip(-(0.2 * lat_err + 0.85 * head_err), -1.0, 1.0)
            brake = jnp.clip(-speed_error * 0.30, 0.0, 1.0)
            return jnp.array([steer, 0.0, brake], dtype=jnp.float32)

        def full_brake_rollout_policy(s):
            lat_err = s[10]
            head_err = s[11]
            steer = jnp.clip(-(0.18 * lat_err + 0.85 * head_err), -1.0, 1.0)
            return jnp.array([steer, 0.0, 1.0], dtype=jnp.float32)

        def nominal_policy_fn(s):
            if isinstance(self.controller, CenterlineDriver):
                return centerline_rollout_policy(s)
            # Not used for MPPI: its action plan is passed to update() instead.
            return centerline_rollout_policy(s)

        def backup_policy_fn(s):
            if isinstance(self.backup_controller, FullBrakeDriver):
                return full_brake_rollout_policy(s)
            return centerline_rollout_policy(s)

        def dynamics_fn(s, action, noise):
            jax_state = JaxRacecarState(
                x=s[0], y=s[1], yaw=s[2], vx=s[3], vy=s[4], yaw_rate=s[5],
                steer=s[6], throttle=s[7], brake=s[8],
                progress=s[9], lateral_error=s[10], heading_error=s[11],
                wheel_rotation=jnp.float32(0), lap_count=jnp.int32(0), step_count=jnp.int32(0),
                action_history=jax_params.simulation.history_template,
            )
            res = step_nominal(jax_params, jax_state, action, residual=noise)
            return DRSGatekeeper.state_from_jax_racecar(res.state)

        initial_bounds = TrackBoundsEstimate.from_track_width(
            half_width=self.cfg.track_half_width,
            relative_uncertainty=0.0,
        )

        # Note: uncertainty_model is a model object (or None), so this is always 4.
        # The gatekeeper stores noise_dim but does not use it.
        noise_dim = 3 if uncertainty_model == "empirical" else 4
        self.gatekeeper = DRSGatekeeper(
            params=params,
            dynamics_fn=dynamics_fn,
            nominal_policy_fn=nominal_policy_fn,
            backup_policy_fn=backup_policy_fn,
            safety_fn=safety_fn,
            pcis_fn=pcis_fn,
            noise_dim=noise_dim,
            theta_dim=2,
            uncertainty_model=uncertainty_model,
            gaussian_noise_mean=gaussian_noise_mean,
            gaussian_noise_std=gaussian_noise_std,
            track=self.track,
            vehicle_config=self.scenario_obj.vehicle,
            initial_track_bounds=initial_bounds,
        )

    def _maybe_init_renderer(self) -> None:
        mode = str(self.cfg.render_mode).strip().lower()
        if mode in {"", "none"}:
            return
        if self.renderer is None:
            self.renderer = PyBulletMirrorRenderer(
                scenario=self.scenario_obj,
                track=self.track,
                render_mode=mode,
                width=1280,
                height=720,
            )

    def _render_frame(self, state, step_index: int, total_steps: int) -> None:
        if self.renderer is None:
            return
        render_state = {
            "x": float(state.x),
            "y": float(state.y),
            "vx": float(state.vx),
            "yaw": float(state.yaw),
            "steering_angle": float(state.steer * self.scenario_obj.vehicle.max_steer_rad),
            "wheel_rotation": float(state.wheel_rotation),
            "frame_index": step_index,
            "total_frames": total_steps,
        }
        frame = self.renderer.render(render_state)
        if frame is not None:
            self.frames.append(frame)

    def _record_trajectory_step(self, state, reward: float, step_index: int) -> None:
        self.trajectory.append(
            {
                "x": float(state.x),
                "y": float(state.y),
                "yaw": float(state.yaw),
                "steering_angle": float(state.steer * self.scenario_obj.vehicle.max_steer_rad),
                "wheel_rotation": float(state.wheel_rotation),
                "progress": float(state.progress),
                "frame_index": int(step_index),
                "speed": float(state.vx),
                "reward": float(reward),
                "uncertainty": {"mode": self._normalized_uncertainty_mode() or "none"},
            }
        )

    def _mppi_debug_payload(self, controller) -> dict | None:
        if controller is None or not hasattr(controller, "get_render_debug"):
            return None

        render_debug = controller.get_render_debug()
        if not render_debug:
            return None

        payload: dict[str, object] = {
            "candidate_xy": np.asarray(render_debug.get("candidate_xy", []), dtype=float),
            "final_xy": np.asarray(render_debug.get("final_xy", []), dtype=float),
        }
        return payload

    def _gatekeeper_debug_payload(self, gatekeeper_state, action: np.ndarray | None = None) -> dict | None:
        if gatekeeper_state is None:
            return None

        payload: dict[str, object] = {
            "using_backup": bool(gatekeeper_state.using_backup),
            "is_reverting": bool(gatekeeper_state.is_reverting),
            "s_t": int(gatekeeper_state.s_prev),
            "m_star": int(gatekeeper_state.m_star),
            "q_bar_star": float(gatekeeper_state.q_bar_star),
            "plan_start_t": int(gatekeeper_state.plan_start_t),
        }
        if action is not None:
            payload["action"] = np.asarray(action, dtype=float)
        if gatekeeper_state.predicted_trajectories is not None:
            payload["gk_trajectories"] = np.asarray(gatekeeper_state.predicted_trajectories, dtype=float)
        if gatekeeper_state.failure_mask is not None:
            payload["failure_mask"] = np.asarray(gatekeeper_state.failure_mask, dtype=bool)
        return payload

    def save_video(self, output_path: str) -> None:
        if not self.frames:
            return
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), self.frames, fps=self.cfg.fps, quality=self.cfg.quality)

    def save_replay_bundle(self, output_dir: str, *, video_path: str | None = None) -> Path:
        return export_replay_bundle(self.trajectory, self.scenario_obj, output_dir, video_path=video_path)

    def cleanup(self) -> None:
        try:
            import pybullet as p
            if self.renderer is not None:
                p.disconnect(self.renderer.client_id)
        except Exception:
            pass

    def run_episode(self, run_seed: int) -> dict:
        """Simulate one episode and return its statistics.

        The episode ends when the car first crosses the half-lap mark
        (progress 0.5), completes a lap, leaves the track
        (|lateral_error| > actual half-width) or reaches MAX_STEPS.
        """
        self.frames = []
        self.trajectory = []
        self._maybe_init_renderer()

        episode_key = jax.random.PRNGKey(run_seed)
        reset_key, controller_key = jax.random.split(episode_key)
        controller_seed = int(np.asarray(controller_key[0], dtype=np.uint32))
        self._controller_noise_rng = np.random.default_rng(controller_seed)
        self._last_obs_cache.clear()
        self._last_noisy_state = None

        if self.is_jax:
            if self.cfg.start_progress is not None:
                reset_out = self.env.reset_custom(key=reset_key, progress=self.cfg.start_progress)
            else:
                reset_out = self.env.reset(reset_key, start_mode=self.cfg.start_mode)
            state = reset_out.state
            self.env._state = state
            self.env._history = []
            self.env.track = self.track
            self.env.scenario = self.scenario_obj
        else:
            raise RuntimeError("CleanUniversalSimulator currently uses JAX env only")

        if hasattr(self.controller, "reset"):
            self.controller.reset(seed=controller_seed, case_id="clean_sim")
        if self.backup_controller is not None and hasattr(self.backup_controller, "reset"):
            self.backup_controller.reset(seed=controller_seed, case_id="clean_sim")

        if self.gatekeeper is not None:
            x0 = DRSGatekeeper.state_from_vehicle_state(self._state_to_vehicle_state(state))
            self.gatekeeper.reset(x0, t=0)

        self._record_trajectory_step(state, reward=0.0, step_index=0)

        stats = {
            "total_steps": 0,
            "termination_reason": None,
            "total_computation_nominal": 0.0,
            "total_computation_time": 0.0,
            "backup_steps_count": 0,
            "progress_ge_0_5_crossings": 0,
            "total_vx": 0.0,
        }

        was_ge_half = float(state.progress) >= 0.5
        for step in range(self.cfg.max_steps):
            vehicle_state = self._state_to_vehicle_state(state)

            t0 = time.perf_counter()
            nominal_action = self._controller_action(self.controller, vehicle_state, step_index=step, mode="nominal")
            nominal_trajectory = getattr(self.controller, "_action_plan", None)
            stats["total_computation_nominal"] += (time.perf_counter() - t0)

            if self.gatekeeper is not None:
                if isinstance(self.controller, (JaxMPPIController)):
                    if nominal_trajectory is None:
                        raise RuntimeError("MPPI is nominal, but _action_plan is missing")
                    if np.asarray(nominal_trajectory).shape[0] < self.gatekeeper.params.T:
                        raise RuntimeError("MPPI _action_plan shorter than gatekeeper horizon")

                x_flat = DRSGatekeeper.state_from_vehicle_state(vehicle_state)
                t1 = time.perf_counter()
                gk_state = self.gatekeeper.update(x_flat, nominal_trajectory=nominal_trajectory, max_steps=self.cfg.max_steps)
                stats["total_computation_time"] += (time.perf_counter() - t1)

                if gk_state.using_backup:
                    stats["backup_steps_count"] += 1
                    action = self._controller_action(self.backup_controller, vehicle_state, step_index=step, mode="backup")
                else:
                    action = nominal_action
                self.gatekeeper.tick()
            else:
                action = nominal_action

            step_out = self.env.step(state, jnp.asarray(action, dtype=jnp.float32))
            state = step_out.state
            self.env._state = state
            self.env._history.append(action)
            if len(self.env._history) > 20:
                self.env._history.pop(0)

            self._record_trajectory_step(state, reward=float(step_out.reward), step_index=step + 1)
            if self.trajectory:
                if self.cfg.use_gatekeeper:
                    planner_debug = self._gatekeeper_debug_payload(gk_state, action=action)
                else:
                    planner_debug = self._mppi_debug_payload(self.controller)
                if planner_debug is not None:
                    self.trajectory[-1]["planner_debug"] = planner_debug

            stats["total_steps"] += 1
            stats["total_vx"] += float(state.vx)
            self._render_frame(state, step_index=step, total_steps=self.cfg.max_steps)

            # Stop at the first half-lap crossing (the paper's racecar success
            # metric). From START_PROGRESS = 0 this comes before a full lap, so
            # "lap_complete" is not reached in that setting.
            now_ge_half = float(state.progress) >= 0.5
            if now_ge_half and not was_ge_half:
                stats["progress_ge_0_5_crossings"] += 1
                break
            was_ge_half = now_ge_half

            if int(state.lap_count) >= 1:
                stats["termination_reason"] = "lap_complete"
                break


            track_half_width = self.actual_track_half_width if self.actual_track_half_width is not None else self.track.width / 2.0
            if abs(float(state.lateral_error)) > track_half_width:
                stats["termination_reason"] = "off-track"
                break

        if stats["termination_reason"] is None:
            stats["termination_reason"] = "horizon"

        steps = max(1, int(stats["total_steps"]))
        stats["avg_nominal_ms_per_step"] = (stats["total_computation_nominal"] / steps) * 1000.0
        stats["avg_gatekeeper_ms_per_step"] = (stats["total_computation_time"] / steps) * 1000.0
        stats["backup_ratio"] = float(stats["backup_steps_count"]) / float(steps)
        stats["avg_speed"] = float(stats["total_vx"]) / float(steps)

        return stats


class BatchExperiment:
    """Run ITERATIONS episodes (seeds SEED, SEED+1, ...) and aggregate the results."""
    def __init__(self, config: SimulatorConfig):
        self.cfg = config

    def _video_path(self, iteration: int, seed: int) -> str:
        output = Path(self.cfg.output)
        suffix = output.suffix if output.suffix else ".mp4"
        stem = output.stem if output.stem else "episode"
        parent = output.parent if str(output.parent) != "" else Path(".")
        return str(parent / f"{stem}_iter{iteration:03d}_seed{seed}{suffix}")

    def _bundle_dir(self, iteration: int, seed: int) -> str:
        output = Path(self.cfg.output)
        stem = output.stem if output.stem else "episode"
        parent = output.parent if str(output.parent) != "" else Path(".")
        return str(parent / f"{stem}_iter{iteration:03d}_seed{seed}_bundle")

    def run(self) -> dict:
        """Run all episodes, save videos/bundles if enabled, print a summary and return the aggregate dict."""
        aggregate = {
            "total_runs": 0,
            "full_lap_finishes": 0,
            "half_lap_finishes": 0,
            "total_steps": 0,
            "total_backup_steps": 0,
            "total_nominal_time_sec": 0.0,
            "total_gatekeeper_time_sec": 0.0,
            "total_vx": 0.0,
            "runs": [],
        }

        print(f"Gatekeeper: {USE_GATEKEEPER} | Controller: {self.cfg.controller} | Uncertainty: {self.cfg.uncertainty}")
        print(f"Track half-width: {self.cfg.track_half_width} | PCIS half-width: {self.cfg.pcis_half_width}")
        print(f"epsilon: {GK_EPSILON} | beta: {GK_BETA} | alpha: {GK_ALPHA} | p: {GK_P} | lipschitz_mode: {LIPSCHITZ_MODE} | lipschitz_constant: {LIPSCHITZ_CONSTANT}")
        
        for i in range(self.cfg.iterations):
            seed = self.cfg.seed + i
            print(f"\n=== Episode {i+1}/{self.cfg.iterations} | seed={seed} ===")

            sim = CleanUniversalSimulator(self.cfg)
            stats = sim.run_episode(seed)

            video_path = None
            if self.cfg.save_videos and str(self.cfg.render_mode).lower() != "none":
                video_path = self._video_path(i + 1, seed)
                sim.save_video(video_path)
            if SAVE_REPLAY_BUNDLES:
                sim.save_replay_bundle(self._bundle_dir(i + 1, seed), video_path=video_path)
            sim.cleanup()

            aggregate["total_runs"] += 1
            aggregate["total_steps"] += int(stats["total_steps"])
            aggregate["total_backup_steps"] += int(stats["backup_steps_count"])
            aggregate["total_nominal_time_sec"] += float(stats["total_computation_nominal"])
            aggregate["total_gatekeeper_time_sec"] += float(stats["total_computation_time"])
            aggregate["total_vx"] += float(stats["total_vx"])

            if int(stats["progress_ge_0_5_crossings"]) > 0:
                aggregate["half_lap_finishes"] += 1
            if stats["termination_reason"] == "lap_complete":
                aggregate["full_lap_finishes"] += 1

            aggregate["runs"].append(stats)

            print(
                f"termination={stats['termination_reason']} | steps={stats['total_steps']} | "
                f"backup_ratio={stats['backup_ratio']:.5f} | "
                f"avg_speed={stats['avg_speed']:.2f} m/s | "
                f"nominal_ms/step={stats['avg_nominal_ms_per_step']:.3f} | "
                f"gk_ms/step={stats['avg_gatekeeper_ms_per_step']:.3f}"
            )
            
            # Running success rates
            full_lap_pct = (aggregate["full_lap_finishes"] / aggregate["total_runs"]) * 100.0
            half_lap_pct = (aggregate["half_lap_finishes"] / aggregate["total_runs"]) * 100.0
            print(
                f"  ✓ Success ratio so far: "
                f"Half lap {aggregate['half_lap_finishes']}/{aggregate['total_runs']} ({half_lap_pct:.1f}%)"
            )

        total_steps = max(1, aggregate["total_steps"])
        aggregate["overall_backup_ratio"] = aggregate["total_backup_steps"] / total_steps
        aggregate["overall_nominal_ms_per_step"] = (aggregate["total_nominal_time_sec"] / total_steps) * 1000.0
        aggregate["overall_gatekeeper_ms_per_step"] = (aggregate["total_gatekeeper_time_sec"] / total_steps) * 1000.0

        print("\n" + "=" * 72)
        print("BATCH SUMMARY")
        print(f"Distribution Uncertainty: {self.cfg.uncertainty} | ")
        print("=" * 72)
        print(f"Total runs: {aggregate['total_runs']}")
        print(f"Full lap finishes: {aggregate['full_lap_finishes']}")
        print(f"Half lap finishes: {aggregate['half_lap_finishes']}")
        print(f"Backup ratio (global): {aggregate['overall_backup_ratio']:.5f}")
        print(f"Avg nominal computation: {aggregate['overall_nominal_ms_per_step']:.3f} ms/step")
        print(f"Avg gatekeeper computation: {aggregate['overall_gatekeeper_ms_per_step']:.3f} ms/step")
        
        overall_avg_speed = aggregate["total_vx"] / max(1, aggregate["total_steps"])
        print(f"Overall average speed: {overall_avg_speed:.2f} m/s")

        return aggregate


def build_config() -> SimulatorConfig:
    """Collect the module-level constants into a SimulatorConfig."""
    return SimulatorConfig(
        scenario=SCENARIO,
        controller=CONTROLLER,
        max_steps=MAX_STEPS,
        seed=SEED,
        iterations=ITERATIONS,
        start_mode=START_MODE,
        start_progress=START_PROGRESS,
        controller_kwargs=CONTROLLER_KWARGS,
        uncertainty=UNCERTAINTY,
        uncertainty_artifact=UNCERTAINTY_ARTIFACT,
        use_gatekeeper=USE_GATEKEEPER,
        backup_controller=BACKUP_CONTROLLER,
        gk_m=GK_M,
        gk_t=GK_T,
        gk_n=GK_N,
        gk_delta=GK_DELTA,
        gk_epsilon=GK_EPSILON,
        gk_beta=GK_BETA,
        gk_alpha=GK_ALPHA,
        gk_p=GK_P,
        lipschitz_mode=LIPSCHITZ_MODE,
        lipschitz_constant=LIPSCHITZ_CONSTANT,
        gk_debug_timing=GK_DEBUG_TIMING,
        render_mode=RENDER_MODE,
        save_videos=SAVE_VIDEOS,
        output=OUTPUT,
        fps=FPS,
        quality=QUALITY,
        track_half_width=TRACK_HALF_WIDTH,
        pcis_half_width=PCIS_HALF_WIDTH,
        track_width=TRACK_WIDTH,
        actual_track_half_width=ACTUAL_TRACK_HALF_WIDTH,
        controller_progress_noise_std=CONTROLLER_PROGRESS_NOISE_STD,
        controller_lateral_noise_std=CONTROLLER_LATERAL_NOISE_STD,
        controller_heading_noise_std=CONTROLLER_HEADING_NOISE_STD,
        controller_speed_noise_std=CONTROLLER_SPEED_NOISE_STD,
        residual_gain_noise=RESIDUAL_GAIN_NOISE,
        residual_gain_std=RESIDUAL_GAIN_STD,
    )


if __name__ == "__main__":
    cfg = build_config()
    experiment = BatchExperiment(cfg)
    experiment.run()
