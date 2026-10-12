"""Rule-based drivers and the controller interface.

CenterlineDriver and FullBrakeDriver are the backup policies used by the
simulator and the gatekeeper. ProfiledCenterlineDriver and
build_speed_profile follow a speed profile built from recorded laps; they
are only used to warm-start MPPI when a driver_dataset is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from uncertain_racecar_gym.dynamics import VehicleState
from uncertain_racecar_gym.track import TrackModel


class CenterlineDriver:
    """Slow centreline tracker used as the backup policy.

    Steers on lateral and heading error and brakes towards 6 m/s; it never
    applies throttle. The target_speed and min_speed arguments are stored but
    not used.
    """
    def __init__(self, target_speed: float = 14.0, min_speed: float = 6.0):
        self.target_speed = target_speed
        self.min_speed = min_speed

    def act(self, state: VehicleState, track: TrackModel) -> np.ndarray:
        curvature = abs(track.sample(state.progress).curvature)
        # speed_target = max(self.min_speed, self.target_speed / (1.0 + curvature * 50.0))
        speed_target = 6.0
        speed_error = speed_target - state.vx

        steer = np.clip(-(0.2 * state.lateral_error + 0.85 * state.heading_error), -1.0, 1.0)
        # throttle = float(np.clip(speed_error * 0.22, 0.0, 1.0))
        brake = float(np.clip(-speed_error * 0.30, 0.0, 1.0))
        return np.array([steer, 0.0, brake], dtype=np.float32)


class FullBrakeDriver:
    """Backup policy that tracks the centreline with full brake applied."""
    def __init__(self, steer_gain_lateral: float = 0.18, steer_gain_heading: float = 0.85):
        self.steer_gain_lateral = float(steer_gain_lateral)
        self.steer_gain_heading = float(steer_gain_heading)

    def act(self, state: VehicleState, track: TrackModel) -> np.ndarray:
        steer = np.clip(
            -(self.steer_gain_lateral * state.lateral_error + self.steer_gain_heading * state.heading_error),
            -1.0,
            1.0,
        )
        return np.array([steer, 0.0, 1.0], dtype=np.float32)


def build_speed_profile(
    canonical: pd.DataFrame,
    progress_bins: int = 160,
    speed_quantile: float = 0.65,
    speed_scale: float = 0.55,
    min_speed: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Target speed per progress bin from recorded laps.

    Takes a quantile of vx in each bin, scales it, applies a minimum speed and
    smooths it around the lap. Returns (bin centres, speeds).
    """
    frame = canonical[["progress", "vx"]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame["progress_bin"] = np.floor(frame["progress"].to_numpy(dtype=float) * progress_bins).astype(int) % progress_bins
    grouped = frame.groupby("progress_bin", observed=False)["vx"].quantile(speed_quantile)
    profile = np.full(progress_bins, float(min_speed), dtype=float)
    for bin_index, value in grouped.items():
        profile[int(bin_index)] = max(float(min_speed), float(value) * speed_scale)

    valid = np.flatnonzero(np.isfinite(profile))
    if len(valid) and len(valid) < progress_bins:
        profile = np.interp(np.arange(progress_bins), valid, profile[valid], period=progress_bins)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel = kernel / kernel.sum()
    padded = np.pad(profile, (2, 2), mode="wrap")
    smoothed = np.convolve(padded, kernel, mode="valid")
    progress = (np.arange(progress_bins, dtype=float) + 0.5) / float(progress_bins)
    return progress, smoothed


class ProfiledCenterlineDriver:
    """Centreline tracker that follows a progress-dependent speed profile."""
    def __init__(self, progress_points: np.ndarray, target_speeds: np.ndarray, min_speed: float = 8.0):
        self.progress_points = np.asarray(progress_points, dtype=float)
        self.target_speeds = np.asarray(target_speeds, dtype=float)
        self.min_speed = float(min_speed)

    @classmethod
    def from_canonical_dataframe(
        cls,
        canonical: pd.DataFrame,
        progress_bins: int = 160,
        speed_quantile: float = 0.65,
        speed_scale: float = 0.55,
        min_speed: float = 8.0,
    ) -> "ProfiledCenterlineDriver":
        progress_points, target_speeds = build_speed_profile(
            canonical,
            progress_bins=progress_bins,
            speed_quantile=speed_quantile,
            speed_scale=speed_scale,
            min_speed=min_speed,
        )
        return cls(progress_points=progress_points, target_speeds=target_speeds, min_speed=min_speed)

    def target_speed(self, progress: float) -> float:
        clipped = float(progress % 1.0)
        return float(max(self.min_speed, np.interp(clipped, self.progress_points, self.target_speeds, period=1.0)))

    def act(self, state: VehicleState, track: TrackModel) -> np.ndarray:
        speed_target = self.target_speed(state.progress)
        speed_error = speed_target - state.vx
        steer = np.clip(-(0.18 * state.lateral_error + 0.85 * state.heading_error), -1.0, 1.0)
        throttle = float(np.clip(speed_error * 0.18, 0.0, 1.0))
        brake = float(np.clip(-speed_error * 0.18, 0.0, 1.0))
        return np.array([steer, throttle, brake], dtype=np.float32)


@dataclass(slots=True)
class ControllerStepContext:
    """Per-step information passed to controllers that need more than the observation (e.g. MPPI reads env)."""
    observation: np.ndarray
    step_index: int
    episode_index: int
    case_id: str
    mode: str | None
    env: Any | None = None
    info: Mapping[str, Any] | None = None


class BenchmarkController(Protocol):
    """Interface for controllers with reset() and act(observation, context=...)."""
    name: str

    def reset(self, *, seed: int | None = None, case_id: str | None = None) -> None: ...

    def act(self, observation: np.ndarray, *, context: ControllerStepContext) -> np.ndarray: ...


