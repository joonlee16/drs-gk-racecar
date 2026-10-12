"""Feature vector used to condition the empirical uncertainty model.

Layout (FEATURE_NAMES): 8 core state features, 10 telemetry-like
features, then the flattened action history [steer, throttle, brake] x
history_length. In simulation the telemetry features are approximated
from the state by runtime_telemetry_proxy.
"""

from __future__ import annotations

from typing import Any

import numpy as np


CORE_FEATURE_NAMES = [
    "curvature",
    "progress",
    "vx",
    "vy",
    "yaw_rate",
    "steer",
    "throttle",
    "brake",
]

TELEMETRY_CANONICAL_COLUMNS = [
    "accel_x",
    "accel_y",
    "drive_train_speed",
    "rpm",
    "gear",
    "rear_slip_ratio_mean",
    "rear_slip_angle_mean",
    "tc_active",
    "abs_active",
]

EXTRA_FEATURE_NAMES = [
    "accel_x",
    "accel_y",
    "drive_train_speed",
    "speed_gap",
    "rear_slip_ratio_mean",
    "rear_slip_angle_mean",
    "gear_norm",
    "rpm_norm",
    "tc_active",
    "abs_active",
]

FEATURE_NAMES = CORE_FEATURE_NAMES + EXTRA_FEATURE_NAMES
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
ACTION_HISTORY_WIDTH = 3


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def estimate_gear_from_speed(speed_mps: float) -> float:
    """Rough gear (1-7) from speed using fixed thresholds."""
    thresholds = [6.0, 13.0, 21.0, 30.0, 40.0, 51.0]
    for index, threshold in enumerate(thresholds, start=1):
        if speed_mps < threshold:
            return float(index)
    return 7.0


def normalize_gear(gear: float) -> float:
    """Gear scaled to [0, 1]."""
    return float(np.clip(_safe_float(gear, 0.0) / 7.0, 0.0, 1.0))


def estimate_rpm_from_drive_speed(drive_speed_mps: float, gear: float, wheel_radius: float) -> float:
    """Rough engine rpm from drive speed, gear and wheel radius."""
    drive_speed = max(abs(_safe_float(drive_speed_mps, 0.0)), 0.0)
    gear_value = max(1.0, _safe_float(gear, 1.0))
    gear_ratio = {
        1.0: 13.5,
        2.0: 9.6,
        3.0: 7.2,
        4.0: 5.8,
        5.0: 4.8,
        6.0: 4.0,
        7.0: 3.5,
    }.get(float(round(gear_value)), 4.2)
    wheel_rad_s = drive_speed / max(_safe_float(wheel_radius, 0.33), 1e-6)
    rpm = wheel_rad_s * gear_ratio * 60.0 / (2.0 * np.pi) + 1000.0
    return float(np.clip(rpm, 900.0, 9000.0))


def normalize_rpm(rpm: float) -> float:
    """rpm scaled to [0, 1]."""
    return float(np.clip((_safe_float(rpm, 1000.0) - 1000.0) / 8000.0, 0.0, 1.0))


def rear_slip_angle_proxy(vx: float, vy: float, yaw_rate: float, lr: float) -> float:
    """Rear slip angle approximation atan2(vy - lr * yaw_rate, |vx|)."""
    vx_safe = max(abs(_safe_float(vx, 0.0)), 1.0)
    return float(np.arctan2(_safe_float(vy, 0.0) - _safe_float(lr, 0.0) * _safe_float(yaw_rate, 0.0), vx_safe))


def runtime_telemetry_proxy(
    state: Any,
    previous_state: Any | None,
    dt: float,
    vehicle_config: Any,
) -> dict[str, float]:
    """Approximate the telemetry features from the current and previous simulator state."""
    vx = _safe_float(getattr(state, "vx", 0.0))
    vy = _safe_float(getattr(state, "vy", 0.0))
    yaw_rate = _safe_float(getattr(state, "yaw_rate", 0.0))
    throttle = _safe_float(getattr(state, "throttle", 0.0))
    brake = _safe_float(getattr(state, "brake", 0.0))
    wheel_rotation = _safe_float(getattr(state, "wheel_rotation", 0.0))

    if previous_state is not None:
        previous_vx = _safe_float(getattr(previous_state, "vx", vx))
        previous_wheel_rotation = _safe_float(getattr(previous_state, "wheel_rotation", wheel_rotation))
        accel_x = (vx - previous_vx) / max(_safe_float(dt, 0.05), 1e-6)
        drive_train_speed = (
            (wheel_rotation - previous_wheel_rotation)
            * _safe_float(getattr(vehicle_config, "wheel_radius", 0.33), 0.33)
            / max(_safe_float(dt, 0.05), 1e-6)
        )
    else:
        accel_x = throttle * _safe_float(getattr(vehicle_config, "max_accel", 0.0)) - brake * _safe_float(getattr(vehicle_config, "max_brake", 0.0))
        drive_train_speed = vx

    accel_y = vx * yaw_rate
    speed_gap = drive_train_speed - vx
    rear_slip_ratio = speed_gap / max(abs(vx), 1.0)
    rear_slip_angle = rear_slip_angle_proxy(vx, vy, yaw_rate, _safe_float(getattr(vehicle_config, "lr", 0.0), 0.0))
    gear = estimate_gear_from_speed(vx)
    rpm = estimate_rpm_from_drive_speed(drive_train_speed, gear, _safe_float(getattr(vehicle_config, "wheel_radius", 0.33), 0.33))
    tc_active = float(throttle > 0.35 and rear_slip_ratio > 0.08)
    abs_active = float(brake > 0.25 and rear_slip_ratio < -0.08)

    return {
        "accel_x": float(accel_x),
        "accel_y": float(accel_y),
        "drive_train_speed": float(drive_train_speed),
        "speed_gap": float(speed_gap),
        "rear_slip_ratio_mean": float(rear_slip_ratio),
        "rear_slip_angle_mean": float(rear_slip_angle),
        "gear_norm": normalize_gear(gear),
        "rpm_norm": normalize_rpm(rpm),
        "tc_active": float(tc_active),
        "abs_active": float(abs_active),
    }


def build_feature_vector_from_state(
    state: Any,
    curvature: float,
    action_history: list[np.ndarray] | np.ndarray,
    vehicle_config: Any,
    dt: float,
    previous_state: Any | None = None,
) -> np.ndarray:
    """Full feature vector (core + telemetry proxies + action history) for a simulator state."""
    telemetry = runtime_telemetry_proxy(state, previous_state=previous_state, dt=dt, vehicle_config=vehicle_config)
    feature = [
        _safe_float(curvature),
        _safe_float(getattr(state, "progress", 0.0)),
        _safe_float(getattr(state, "vx", 0.0)),
        _safe_float(getattr(state, "vy", 0.0)),
        _safe_float(getattr(state, "yaw_rate", 0.0)),
        _safe_float(getattr(state, "steer", 0.0)),
        _safe_float(getattr(state, "throttle", 0.0)),
        _safe_float(getattr(state, "brake", 0.0)),
        telemetry["accel_x"],
        telemetry["accel_y"],
        telemetry["drive_train_speed"],
        telemetry["speed_gap"],
        telemetry["rear_slip_ratio_mean"],
        telemetry["rear_slip_angle_mean"],
        telemetry["gear_norm"],
        telemetry["rpm_norm"],
        telemetry["tc_active"],
        telemetry["abs_active"],
    ]
    history = np.asarray(action_history, dtype=float).reshape(-1)
    return np.concatenate([np.asarray(feature, dtype=float), history])


def history_means_from_feature_vector(feature_vector: np.ndarray) -> tuple[float, float, float]:
    """Mean steer, throttle and brake over the action history in a feature vector."""
    values = np.asarray(feature_vector, dtype=float)
    if len(values) <= len(FEATURE_NAMES):
        return 0.0, 0.0, 0.0
    history = values[len(FEATURE_NAMES) :].reshape(-1, ACTION_HISTORY_WIDTH)
    return float(history[:, 0].mean()), float(history[:, 1].mean()), float(history[:, 2].mean())


def feature_value(feature_vector: np.ndarray, name: str, default: float = 0.0) -> float:
    """Value of a named feature, or `default` if it is missing."""
    index = FEATURE_INDEX.get(name)
    if index is None or len(feature_vector) <= index:
        return float(default)
    return _safe_float(np.asarray(feature_vector, dtype=float)[index], default=default)
