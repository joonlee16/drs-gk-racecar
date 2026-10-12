"""Small shared helpers: angle wrapping, paths to package assets, and softmax weights."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if needed and return it."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def package_asset_path(relative_path: str) -> Path:
    """Absolute path of a file under uncertain_racecar_gym/assets/."""
    return Path(str(files("uncertain_racecar_gym").joinpath("assets").joinpath(relative_path)))


def resolve_resource_path(path: str | Path, base_path: str | Path | None = None) -> Path:
    """Resolve a resource path.

    'package://<rel>' maps to the package assets folder. A relative path is
    resolved against the folder of `base_path` if given, otherwise against
    the current directory.
    """
    candidate = Path(path)
    if isinstance(path, str) and path.startswith("package://"):
        return package_asset_path(path.replace("package://", "", 1))
    if candidate.is_absolute():
        return candidate
    if base_path is None:
        return candidate.resolve()
    return (Path(base_path).parent / candidate).resolve()


def softmax_sample_weights(distances: np.ndarray) -> np.ndarray:
    """Sampling weights that favour small distances: softmax(-d / (mean(d) + 1e-6))."""
    if distances.size == 0:
        return distances
    scale = float(np.mean(distances)) + 1e-6
    logits = -distances / scale
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)
