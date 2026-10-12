"""Export an episode as a replay bundle (trajectory, scene metadata, camera plan) for offline rendering."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np

from uncertain_racecar_gym.common import ensure_dir
from uncertain_racecar_gym.scenario import Scenario


def _json_ready(value):
    """Convert numpy values and paths to JSON-serialisable types."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def export_replay_bundle(
    history: list[dict],
    scenario: Scenario,
    output_dir: str | Path,
    video_path: str | Path | None = None,
) -> Path:
    """Write the replay bundle for one episode to `output_dir` and return the folder.

    Files: trajectory.json, scene_manifest.json, camera_script.json,
    track_centerline.csv, scenario.yaml and README.md. The generated README
    still refers to a Blender script that is no longer in this repository.
    """
    bundle_dir = ensure_dir(output_dir)

    trajectory_path = bundle_dir / "trajectory.json"
    manifest_path = bundle_dir / "scene_manifest.json"
    camera_path = bundle_dir / "camera_script.json"
    readme_path = bundle_dir / "README.md"
    track_path = bundle_dir / "track_centerline.csv"
    scenario_path = bundle_dir / "scenario.yaml"

    trajectory_path.write_text(json.dumps(_json_ready(history), indent=2), encoding="utf-8")
    shutil.copyfile(Path(scenario.track.csv), track_path)
    shutil.copyfile(scenario.source_path, scenario_path)
    manifest_path.write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "track_csv": track_path.name,
                "scenario_yaml": scenario_path.name,
                "vehicle_asset": "package://vehicles/simple_racecar.urdf",
                "video_path": str(video_path) if video_path else None,
                "frame_count": len(history),
                "simulation_dt": float(scenario.simulation.dt),
                "fps": int(round(1.0 / scenario.simulation.dt)),
                "track": asdict(scenario.track),
                "vehicle": asdict(scenario.vehicle),
                "simulation": asdict(scenario.simulation),
                "uncertainty": asdict(scenario.uncertainty),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    camera_path.write_text(
        json.dumps(
            {
                "shots": [
                    {"name": "follow", "type": "follow", "start": 0, "end": max(0, len(history) // 3)},
                    {"name": "orbit", "type": "cinematic", "start": max(0, len(history) // 3), "end": max(0, 2 * len(history) // 3)},
                    {"name": "trackside", "type": "birds_eye", "start": max(0, 2 * len(history) // 3), "end": len(history)},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    readme_path.write_text(
        "\n".join(
            [
                "# Blender Replay Bundle",
                "",
                "This bundle contains the simulator trajectory, camera script, and scene metadata for offline rendering.",
                "",
                "Suggested workflow:",
                "1. Build the track mesh from `track_centerline.csv`.",
                "2. Reconstruct the vehicle using the dimensions in `scene_manifest.json`.",
                "3. Animate the vehicle using `trajectory.json`.",
                "4. Apply the shot plan from `camera_script.json`.",
                "5. Render with the Blender automation in `uncertain_racecar_gym/assets/blender/render_replay.py`.",
            ]
        ),
        encoding="utf-8",
    )
    return bundle_dir
