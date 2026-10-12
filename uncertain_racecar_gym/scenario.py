"""Scenario configuration: dataclasses for the track, vehicle and simulation settings, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from uncertain_racecar_gym.common import resolve_resource_path


@dataclass(slots=True)
class TrackConfig:
    """Track centreline CSV, full width [m], number of progress bins, and whether the track is closed."""
    csv: str
    width: float
    progress_bins: int = 24
    closed: bool = True


@dataclass(slots=True)
class VehicleConfig:
    """Dynamic bicycle model parameters (SI units; max_steer_rad is the steering angle at steer = 1)."""
    wheelbase: float
    lf: float
    lr: float
    mass: float
    inertia_z: float
    cornering_stiffness_front: float
    cornering_stiffness_rear: float
    max_steer_rad: float
    max_accel: float
    max_brake: float
    drag_coefficient: float
    wheel_radius: float
    chassis_size: tuple[float, float, float]


@dataclass(slots=True)
class SimulationConfig:
    """Time step [s], episode step limit, and the observation's curvature look-ahead points."""
    dt: float
    max_steps: int
    lookahead_points: int = 5
    lookahead_spacing_m: float = 5.0


@dataclass(slots=True)
class UncertaintyConfig:
    """Settings for the empirical uncertainty model; history_length is also the length of the action history in the state."""
    history_length: int = 5
    neighbor_count: int = 64
    block_length: int = 25


@dataclass(slots=True)
class RewardConfig:
    """Weights of the environment reward (progress, speed, lateral and heading error)."""
    progress_coef: float = 100.0
    speed_coef: float = 0.05
    lateral_error_coef: float = 0.03
    heading_error_coef: float = 0.01


@dataclass(slots=True)
class Scenario:
    """A complete scenario loaded from YAML."""
    name: str
    track: TrackConfig
    vehicle: VehicleConfig
    simulation: SimulationConfig
    uncertainty: UncertaintyConfig
    reward: RewardConfig
    source_path: Path


DEFAULT_SCENARIO = "package://scenarios/sample_oval.yaml"


def load_scenario(path: str | Path | None = None) -> Scenario:
    """Load a scenario YAML file (default: the sample oval).

    Accepts 'package://...' paths. The track CSV path in the file is resolved
    relative to the YAML file.
    """
    scenario_path = resolve_resource_path(path or DEFAULT_SCENARIO)
    with scenario_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    track = TrackConfig(**raw["track"])
    track.csv = str(resolve_resource_path(track.csv, scenario_path))
    vehicle = VehicleConfig(
        chassis_size=tuple(raw["vehicle"]["chassis_size"]),
        **{k: v for k, v in raw["vehicle"].items() if k != "chassis_size"},
    )
    simulation = SimulationConfig(**raw["simulation"])
    uncertainty = UncertaintyConfig(**raw.get("uncertainty", {}))
    reward = RewardConfig(**raw.get("reward", {}))
    return Scenario(
        name=raw["name"],
        track=track,
        vehicle=vehicle,
        simulation=simulation,
        uncertainty=uncertainty,
        reward=reward,
        source_path=scenario_path,
    )
