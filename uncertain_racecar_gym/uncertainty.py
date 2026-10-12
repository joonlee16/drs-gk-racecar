"""Empirical uncertainty model: data-driven residuals for the racecar dynamics.

The model stores one-step residuals [delta_vx, delta_vy, delta_yaw_rate]
between recorded laps and the nominal model, grouped into buckets by
(track, car, progress bin). To sample a residual it finds the k nearest
recorded states to the current feature vector, preferring states in the
same driving regime, and picks one with softmax weights. It then keeps
following that recording for a random block of steps.

This module only loads and samples the model; it is fitted in
https://github.com/tkkim-robot/uncertain-racecar-gym and loaded from a
.pkl file. `to_jax` exports it for the JAX sampler in uncertainty_jax.py.
"""

from __future__ import annotations

import pickle
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from uncertain_racecar_gym.common import softmax_sample_weights
from uncertain_racecar_gym.features import FEATURE_NAMES, feature_value, history_means_from_feature_vector

RESIDUAL_NAMES = ["delta_vx", "delta_vy", "delta_yaw_rate"]


def regime_key_from_feature_vector(feature_vector: np.ndarray) -> str:
    """Classify a feature vector into a driving regime string.

    The key combines control (braking, power, coasting, traction limited),
    speed band, corner or straight, direction and rear slip state.
    """
    curvature = feature_value(feature_vector, "curvature")
    vx = feature_value(feature_vector, "vx")
    vy = feature_value(feature_vector, "vy")
    yaw_rate = feature_value(feature_vector, "yaw_rate")
    steer = feature_value(feature_vector, "steer")
    throttle = feature_value(feature_vector, "throttle")
    brake = feature_value(feature_vector, "brake")
    speed_gap = feature_value(feature_vector, "speed_gap")
    rear_slip_ratio = feature_value(feature_vector, "rear_slip_ratio_mean")
    rear_slip_angle = feature_value(feature_vector, "rear_slip_angle_mean")
    tc_active = feature_value(feature_vector, "tc_active")
    abs_active = feature_value(feature_vector, "abs_active")
    gear_norm = feature_value(feature_vector, "gear_norm")
    history_steer, history_throttle, history_brake = history_means_from_feature_vector(feature_vector)

    effective_throttle = max(float(throttle), history_throttle)
    effective_brake = max(float(brake), history_brake)
    if abs_active > 0.5 or effective_brake > 0.20 or rear_slip_ratio < -0.14:
        control = "brake_hard"
    elif effective_brake > 0.06:
        control = "brake"
    elif tc_active > 0.5 or rear_slip_ratio > 0.14 or speed_gap > 2.0:
        control = "traction_limited"
    elif effective_throttle > 0.75:
        control = "power_high"
    elif effective_throttle > 0.22:
        control = "power_mid"
    else:
        control = "coast"

    if vx < 12.0:
        speed = "slow"
    elif vx < 25.0:
        speed = "medium"
    elif vx < 40.0:
        speed = "fast"
    else:
        speed = "very_fast"
    if gear_norm > 0.82 and speed in {"fast", "very_fast"}:
        speed = "top_gear"

    turn_metric = max(
        abs(float(steer)),
        abs(history_steer),
        min(1.0, abs(float(curvature)) * 120.0),
        min(1.0, abs(float(vy)) / 3.0),
        min(1.0, abs(float(yaw_rate)) / 1.5),
        min(1.0, abs(float(rear_slip_angle)) / 0.16),
    )
    corner = "corner" if turn_metric > 0.10 else "straight"
    if float(steer) < -0.03 or float(curvature) < -0.004:
        direction = "left"
    elif float(steer) > 0.03 or float(curvature) > 0.004:
        direction = "right"
    else:
        direction = "straight"

    if rear_slip_ratio > 0.12:
        slip = "rear_push"
    elif rear_slip_ratio < -0.12:
        slip = "rear_drag"
    elif abs(rear_slip_angle) > 0.08:
        slip = "rear_angle"
    else:
        slip = "rear_settled"

    return f"{control}|{speed}|{corner}|{direction}|{slip}"


@dataclass(slots=True)
class SamplerRuntimeState:
    """Per-episode sampler state: the recording being followed and how many steps remain in the block."""
    active_row_id: int | None = None
    remaining_block: int = 0
    active_mode_key: str | None = None


@dataclass(slots=True)
class Bucket:
    """Recorded samples for one (track, car, progress bin) gate, with k-d trees per mode and regime."""
    gate: tuple[str, str, int]
    features: np.ndarray
    residuals: np.ndarray
    row_ids: np.ndarray
    trajectory_ids: np.ndarray
    frame_indices: np.ndarray
    mode_keys: np.ndarray
    regime_keys: np.ndarray
    tree: cKDTree | None = None
    mode_indices: dict[str, np.ndarray] | None = None
    mode_trees: dict[str, cKDTree] | None = None
    regime_indices: dict[str, np.ndarray] | None = None
    regime_trees: dict[str, cKDTree] | None = None


def _mode_key_from_trajectory_id(trajectory_id: str) -> str:
    text = str(trajectory_id)
    if "&_&" in text:
        parts = [part.strip() for part in text.split("&_&") if part.strip()]
        if len(parts) >= 3:
            return parts[2]
    tokens = [token for token in text.replace("/", "_").split("_") if token]
    return tokens[0] if tokens else text


class EmpiricalUncertaintyModel:

    """k-nearest-neighbour residual sampler over recorded driving data.

    Bucket features are stored normalised with feature_mean and feature_std.
    """
    def __init__(
        self,
        history_length: int,
        neighbor_count: int,
        progress_bins: int,
        block_length: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        buckets: dict[tuple[str, str, int], Bucket],
        continuation_map: dict[int, int],
        global_channel_mask: np.ndarray | None = None,
        regime_channel_masks: dict[str, np.ndarray] | None = None,
    ):
        self.history_length = history_length
        self.neighbor_count = neighbor_count
        self.progress_bins = progress_bins
        self.block_length = block_length
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.buckets = buckets
        self.continuation_map = continuation_map
        self.global_channel_mask = (
            np.asarray(global_channel_mask, dtype=bool).copy()
            if global_channel_mask is not None
            else np.ones(len(RESIDUAL_NAMES), dtype=bool)
        )
        self.regime_channel_masks = {
            str(key): np.asarray(value, dtype=bool).copy()
            for key, value in (regime_channel_masks or {}).items()
        }
        self._gate_prefixes = sorted({(gate[0], gate[1]) for gate in buckets})
        self._row_lookup = {}
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._row_lookup = {}
        for gate, bucket in self.buckets.items():
            bucket.tree = cKDTree(bucket.features) if len(bucket.features) else None
            bucket.mode_indices = {}
            bucket.mode_trees = {}
            bucket.regime_indices = {}
            bucket.regime_trees = {}
            for mode_key in sorted(set(bucket.mode_keys.tolist())):
                local_indices = np.flatnonzero(bucket.mode_keys == mode_key)
                bucket.mode_indices[str(mode_key)] = local_indices
                if len(local_indices):
                    bucket.mode_trees[str(mode_key)] = cKDTree(bucket.features[local_indices])
            for regime_key in sorted(set(bucket.regime_keys.tolist())):
                local_indices = np.flatnonzero(bucket.regime_keys == regime_key)
                bucket.regime_indices[str(regime_key)] = local_indices
                if len(local_indices):
                    bucket.regime_trees[str(regime_key)] = cKDTree(bucket.features[local_indices])
            for index, row_id in enumerate(bucket.row_ids):
                self._row_lookup[int(row_id)] = (gate, index)

    def make_runtime_state(self) -> SamplerRuntimeState:
        return SamplerRuntimeState()

    @property
    def gate_prefixes(self) -> list[tuple[str, str]]:
        """Distinct (track, car) pairs in the model."""
        return list(self._gate_prefixes)

    def resolve_gate_key(
        self,
        progress_bin: int,
        track_id: str | None = None,
        car_id: str | None = None,
    ) -> tuple[str, str, int]:
        """Pick the (track, car, progress bin) gate to sample from, falling back to the closest available."""
        progress_bin = int(progress_bin) % max(self.progress_bins, 1)
        requested = (str(track_id or ""), str(car_id or ""))
        if requested in self._gate_prefixes:
            return requested[0], requested[1], progress_bin
        if len(self._gate_prefixes) == 1:
            only_track_id, only_car_id = self._gate_prefixes[0]
            return only_track_id, only_car_id, progress_bin
        matching_track = [prefix for prefix in self._gate_prefixes if prefix[0] == requested[0]]
        if len(matching_track) == 1:
            return matching_track[0][0], matching_track[0][1], progress_bin
        matching_car = [prefix for prefix in self._gate_prefixes if prefix[1] == requested[1]]
        if len(matching_car) == 1:
            return matching_car[0][0], matching_car[0][1], progress_bin
        first_track_id, first_car_id = self._gate_prefixes[0]
        return first_track_id, first_car_id, progress_bin

    def _select_bucket(self, track_id: str, car_id: str, progress_bin: int) -> Bucket | None:
        exact = (track_id, car_id, progress_bin)
        if exact in self.buckets:
            return self.buckets[exact]
        for offset in range(1, self.progress_bins):
            for candidate in (
                (track_id, car_id, (progress_bin - offset) % self.progress_bins),
                (track_id, car_id, (progress_bin + offset) % self.progress_bins),
            ):
                if candidate in self.buckets:
                    return self.buckets[candidate]
        for gate, bucket in self.buckets.items():
            if gate[0] == track_id and gate[1] == car_id:
                return bucket
        return next(iter(self.buckets.values()), None)

    def _candidate_pool(
        self,
        bucket: Bucket,
        feature_vector: np.ndarray,
        runtime_state: SamplerRuntimeState | None = None,
    ) -> tuple[np.ndarray, cKDTree | None, str, str]:
        regime_key = regime_key_from_feature_vector(feature_vector)
        candidate_indices = np.arange(len(bucket.features), dtype=int)
        candidate_tree = bucket.tree
        scope = "full"

        regime_indices = None if bucket.regime_indices is None else bucket.regime_indices.get(regime_key)
        mode_indices = None
        if runtime_state is not None and runtime_state.active_mode_key is not None and bucket.mode_indices is not None:
            mode_indices = bucket.mode_indices.get(runtime_state.active_mode_key)

        minimum_subset = max(8, min(self.neighbor_count, 16))
        if regime_indices is not None and mode_indices is not None:
            combo = np.intersect1d(regime_indices, mode_indices, assume_unique=False)
            if len(combo) >= minimum_subset:
                return combo, cKDTree(bucket.features[combo]), regime_key, "mode_regime"
        if regime_indices is not None and len(regime_indices) >= minimum_subset and bucket.regime_trees is not None:
            return regime_indices, bucket.regime_trees[regime_key], regime_key, "regime"
        if mode_indices is not None and len(mode_indices) >= minimum_subset and bucket.mode_trees is not None:
            return mode_indices, bucket.mode_trees[runtime_state.active_mode_key], regime_key, "mode"
        return candidate_indices, candidate_tree, regime_key, scope

    def _channel_mask(self, regime_key: str) -> np.ndarray:
        return self.regime_channel_masks.get(regime_key, self.global_channel_mask).astype(float)

    def sample(
        self,
        feature_vector: np.ndarray,
        gate_key: tuple[str, str, int],
        rng: np.random.Generator,
        runtime_state: SamplerRuntimeState,
    ) -> tuple[np.ndarray, dict]:
        """Sample one residual [delta_vx, delta_vy, delta_yaw_rate].

        Continues the current block if one is active; otherwise draws a new
        neighbour by k-NN in the selected bucket. Channels can be masked per
        regime. Returns (residual, info dict).
        """
        if not self.buckets:
            return np.zeros(3, dtype=float), {"mode": "empty"}
        current_regime_key = regime_key_from_feature_vector(feature_vector)

        if runtime_state.remaining_block > 0 and runtime_state.active_row_id in self.continuation_map:
            next_row_id = self.continuation_map[runtime_state.active_row_id]
            if next_row_id in self._row_lookup:
                gate, local_index = self._row_lookup[next_row_id]
                bucket = self.buckets[gate]
                runtime_state.active_row_id = next_row_id
                runtime_state.remaining_block -= 1
                runtime_state.active_mode_key = str(bucket.mode_keys[local_index])
                return bucket.residuals[local_index] * self._channel_mask(current_regime_key), {
                    "mode": "block",
                    "row_id": int(next_row_id),
                    "gate": gate,
                    "mode_key": runtime_state.active_mode_key,
                    "regime_key": current_regime_key,
                    "candidate_scope": "block",
                    "channel_mask": self._channel_mask(current_regime_key).astype(int).tolist(),
                }

        normalized = (feature_vector - self.feature_mean) / self.feature_std
        bucket = self._select_bucket(*gate_key)
        if bucket is None or bucket.tree is None:
            return np.zeros(3, dtype=float), {"mode": "fallback"}

        candidate_indices, candidate_tree, current_regime_key, candidate_scope = self._candidate_pool(
            bucket,
            feature_vector,
            runtime_state=runtime_state,
        )

        query_k = min(self.neighbor_count, len(candidate_indices))
        distances, local_indices = candidate_tree.query(normalized, k=query_k)
        distances = np.atleast_1d(distances).astype(float)
        local_indices = np.atleast_1d(local_indices).astype(int)
        indices = candidate_indices[local_indices]
        weights = softmax_sample_weights(distances)
        choice_index = int(rng.choice(indices, p=weights))
        row_id = int(bucket.row_ids[choice_index])
        runtime_state.active_row_id = row_id
        runtime_state.remaining_block = int(rng.integers(1, self.block_length + 1))
        runtime_state.active_mode_key = str(bucket.mode_keys[choice_index])
        sampled = bucket.residuals[choice_index] * self._channel_mask(current_regime_key)
        return sampled, {
            "mode": "knn",
            "row_id": row_id,
            "gate": bucket.gate,
            "mode_key": runtime_state.active_mode_key,
            "regime_key": current_regime_key,
            "candidate_scope": candidate_scope,
            "distance_mean": float(distances.mean()),
            "sample_weight": float(weights[np.where(indices == choice_index)[0][0]]),
            "channel_mask": self._channel_mask(current_regime_key).astype(int).tolist(),
        }

    def copy(self) -> "EmpiricalUncertaintyModel":
        return copy.deepcopy(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "EmpiricalUncertaintyModel":
        """Rebuild the model from the dict stored in the .pkl file."""
        buckets = {
            gate: Bucket(
                gate=gate,
                features=data["features"],
                residuals=data["residuals"],
                row_ids=data["row_ids"],
                trajectory_ids=data["trajectory_ids"],
                frame_indices=data["frame_indices"],
                mode_keys=data.get(
                    "mode_keys",
                    np.asarray([_mode_key_from_trajectory_id(value) for value in data["trajectory_ids"]], dtype=object),
                ),
                regime_keys=data.get(
                    "regime_keys",
                    np.asarray(["unknown"] * len(data["row_ids"]), dtype=object),
                ),
            )
            for gate, data in payload["buckets"].items()
        }
        return cls(
            history_length=payload["history_length"],
            neighbor_count=payload["neighbor_count"],
            progress_bins=payload["progress_bins"],
            block_length=payload["block_length"],
            feature_mean=payload["feature_mean"],
            feature_std=payload["feature_std"],
            buckets=buckets,
            continuation_map=payload["continuation_map"],
            global_channel_mask=payload.get("global_channel_mask"),
            regime_channel_masks=payload.get("regime_channel_masks"),
        )

    @classmethod
    def load(cls, input_path: str | Path) -> "EmpiricalUncertaintyModel":
        """Load a model saved as a pickled payload."""
        with Path(input_path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls.from_payload(payload)

    def to_jax(self) -> "JAXEmpiricalData":
        """Export the model as JAXEmpiricalData for sampling inside JAX code.

        Samples from all buckets are flattened and sorted by (gate, regime) so
        each (gate, regime) pool is a contiguous slice found through a lookup
        table. Block continuation is not exported: the JAX sampler draws each
        step independently.
        """
        import jax.numpy as jnp
        from .uncertainty_jax import JAXEmpiricalData, compute_regime_id_jax

        # 1. Flatten the samples of all buckets.
        all_features = []
        all_residuals = []
        all_mode_keys = []
        all_regime_ids = []
        all_gate_ids = []
        
        gate_to_id = {gate: i for i, gate in enumerate(sorted(self.buckets.keys()))}
        mode_to_id = {}
        
        for gate, bucket in self.buckets.items():
            g_id = gate_to_id[gate]
            for i in range(len(bucket.features)):
                feat = bucket.features[i]
                res = bucket.residuals[i]
                mode = str(bucket.mode_keys[i])
                if mode not in mode_to_id:
                    mode_to_id[mode] = len(mode_to_id)
                m_id = mode_to_id[mode]
                
                # Regime ids come from the JAX classifier on the de-normalised
                # features, so export and sampling use the same regimes.
                regime_id = int(compute_regime_id_jax(jnp.asarray(feat * self.feature_std + self.feature_mean)))

                all_features.append(feat)
                all_residuals.append(res)
                all_mode_keys.append(m_id)
                all_regime_ids.append(regime_id)
                all_gate_ids.append(g_id)

        features = jnp.asarray(all_features, dtype=jnp.float32)
        residuals = jnp.asarray(all_residuals, dtype=jnp.float32)
        mode_keys = jnp.asarray(all_mode_keys, dtype=jnp.int32)
        regime_ids = np.asarray(all_regime_ids, dtype=int)
        gate_ids = np.asarray(all_gate_ids, dtype=int)

        # 2. Sort by (gate, regime) so each pool is a contiguous slice.
        sort_idx = np.lexsort((regime_ids, gate_ids))
        features = features[sort_idx]
        residuals = residuals[sort_idx]
        mode_keys = mode_keys[sort_idx]
        regime_ids = regime_ids[sort_idx]
        gate_ids = gate_ids[sort_idx]

        # 3. Lookup tables: (gate, regime) -> (start, count), plus per-gate
        #    ranges used when a regime pool is empty. 720 = number of regime ids.
        num_gates = len(gate_to_id)
        num_regimes = 720
        lookup_start = np.zeros(num_gates * num_regimes, dtype=int)
        lookup_count = np.zeros(num_gates * num_regimes, dtype=int)
        
        lookup_gate_start = np.zeros(num_gates, dtype=int)
        lookup_gate_count = np.zeros(num_gates, dtype=int)

        current_gate = -1
        for i in range(len(gate_ids)):
            g_id = gate_ids[i]
            if g_id != current_gate:
                lookup_gate_start[g_id] = i
                current_gate = g_id
            lookup_gate_count[g_id] += 1
            
            idx = g_id * num_regimes + regime_ids[i]
            if lookup_count[idx] == 0:
                lookup_start[idx] = i
            lookup_count[idx] += 1

        max_pool_size = int(
            max(
                int(np.max(lookup_count)) if lookup_count.size > 0 else 0,
                int(np.max(lookup_gate_count)) if lookup_gate_count.size > 0 else 0,
            )
        )
        max_pool_size = max(max_pool_size, self.neighbor_count, 1)

        return JAXEmpiricalData(
            features=features,
            residuals=residuals,
            mode_keys=mode_keys,
            lookup_start=jnp.asarray(lookup_start, dtype=jnp.int32),
            lookup_count=jnp.asarray(lookup_count, dtype=jnp.int32),
            lookup_gate_start=jnp.asarray(lookup_gate_start, dtype=jnp.int32),
            lookup_gate_count=jnp.asarray(lookup_gate_count, dtype=jnp.int32),
            feature_mean=jnp.asarray(self.feature_mean, dtype=jnp.float32),
            feature_std=jnp.asarray(self.feature_std, dtype=jnp.float32),
            continuation_map=jnp.zeros(len(features), dtype=jnp.int32),
            neighbor_count=self.neighbor_count,
            block_length=self.block_length,
            max_pool_size=max_pool_size,
        )
