"""JAX version of the empirical residual sampler, used inside gatekeeper rollouts.

Differs from EmpiricalUncertaintyModel.sample in three ways: each call
draws independently (no block continuation), the regime uses the first
action-history entry instead of the history mean, and the softmax weights
use squared distances.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

# Residual channels: delta_vx, delta_vy, delta_yaw_rate.
RESIDUAL_DIM = 3

class JAXEmpiricalData(NamedTuple):
    """Device arrays for the JAX empirical sampler (built by EmpiricalUncertaintyModel.to_jax).

    Samples are sorted by (gate, regime); lookup_start / lookup_count give
    the slice for each (gate, regime) pair and lookup_gate_start /
    lookup_gate_count the slice for each gate.
    """
    features: jax.Array      # [TotalSamples, FeatureDim]
    residuals: jax.Array     # [TotalSamples, ResidualDim]
    mode_keys: jax.Array     # [TotalSamples] (int)
    
    # Lookup table for bucketing: [GateID, RegimeID] -> (start, count)
    lookup_start: jax.Array  # [GateCount * RegimeCount]
    lookup_count: jax.Array  # [GateCount * RegimeCount]
    
    # Gate-level fallback: [GateID] -> (start, count)
    lookup_gate_start: jax.Array # [GateCount]
    lookup_gate_count: jax.Array # [GateCount]
    
    feature_mean: jax.Array  # [FeatureDim]
    feature_std: jax.Array   # [FeatureDim]
    
    # Not used: to_jax fills it with zeros (no block continuation in JAX).
    continuation_map: jax.Array # [TotalSamples]

    neighbor_count: int
    block_length: int
    max_pool_size: int


def compute_regime_id_jax(feature_vector: jax.Array) -> jax.Array:
    """JAX port of regime_key_from_feature_vector, returning an integer id in [0, 720).

    id = control + 6 * (speed + 5 * (corner + 2 * (direction + 3 * slip))).
    The history terms use the first history entry (indices 18-20) instead of
    the history mean, so the result can differ from the Python version.
    """
    # Feature indices (from features.py constants)
    # 0: curvature, 1: progress, 2: vx, 3: vy, 4: yaw_rate, 
    # 5: steer, 6: throttle, 7: brake, 11: speed_gap, 
    # 12: rear_slip_ratio_mean, 13: rear_slip_angle_mean,
    # 16: tc_active, 17: abs_active, 14: gear_norm
    
    curvature = feature_vector[0]
    vx = feature_vector[2]
    vy = feature_vector[3]
    yaw_rate = feature_vector[4]
    steer = feature_vector[5]
    throttle = feature_vector[6]
    brake = feature_vector[7]
    speed_gap = feature_vector[11]
    rear_slip_ratio = feature_vector[12]
    rear_slip_angle = feature_vector[13]
    tc_active = feature_vector[16]
    abs_active = feature_vector[17]
    gear_norm = feature_vector[14]
    
    # The Python version uses the mean of the action history; this uses the
    # first history entry (indices 18-20) as an approximation.
    history_steer = feature_vector[18]
    history_throttle = feature_vector[19]
    history_brake = feature_vector[20]
    
    eff_throttle = jnp.maximum(throttle, history_throttle)
    eff_brake = jnp.maximum(brake, history_brake)
    
    # Control Status (6 levels)
    # 0: brake_hard, 1: brake, 2: traction_limited, 3: power_high, 4: power_mid, 5: coast
    control = jnp.where(jnp.logical_or(jnp.logical_or(abs_active > 0.5, eff_brake > 0.20), rear_slip_ratio < -0.14), 0,
              jnp.where(eff_brake > 0.06, 1,
              jnp.where(jnp.logical_or(jnp.logical_or(tc_active > 0.5, rear_slip_ratio > 0.14), speed_gap > 2.0), 2,
              jnp.where(eff_throttle > 0.75, 3,
              jnp.where(eff_throttle > 0.22, 4, 5)))))
    
    # Speed Status (5 levels)
    # 0: slow, 1: medium, 2: fast, 3: very_fast, 4: top_gear
    speed = jnp.where(vx < 12.0, 0,
            jnp.where(vx < 25.0, 1,
            jnp.where(vx < 40.0, 2, 3)))
    speed = jnp.where(jnp.logical_and(gear_norm > 0.82, speed >= 2), 4, speed)
    
    # Corner Status (2 levels)
    turn_metric = jnp.maximum(jnp.abs(steer), jnp.abs(history_steer))
    turn_metric = jnp.maximum(turn_metric, jnp.minimum(1.0, jnp.abs(curvature) * 120.0))
    turn_metric = jnp.maximum(turn_metric, jnp.minimum(1.0, jnp.abs(vy) / 3.0))
    turn_metric = jnp.maximum(turn_metric, jnp.minimum(1.0, jnp.abs(yaw_rate) / 1.5))
    turn_metric = jnp.maximum(turn_metric, jnp.minimum(1.0, jnp.abs(rear_slip_angle) / 0.16))
    corner = jnp.where(turn_metric > 0.10, 1, 0)
    
    # Direction Status (3 levels)
    # 0: left, 1: right, 2: straight
    direction = jnp.where(jnp.logical_or(steer < -0.03, curvature < -0.004), 0,
                jnp.where(jnp.logical_or(steer > 0.03, curvature > 0.004), 1, 2))
    
    # Slip Status (4 levels)
    # 0: rear_push, 1: rear_drag, 2: rear_angle, 3: rear_settled
    slip = jnp.where(rear_slip_ratio > 0.12, 0,
           jnp.where(rear_slip_ratio < -0.12, 1,
           jnp.where(jnp.abs(rear_slip_angle) > 0.08, 2, 3)))
    
    # Unique ID: control (6) * speed (5) * corner (2) * direction (3) * slip (4) = 720
    # id = control + 6*(speed + 5*(corner + 2*(direction + 3*slip)))
    regime_id = control + 6 * (speed + 5 * (corner + 2 * (direction + 3 * slip)))
    return regime_id


def sample_empirical_jax(
    feature_vector: jax.Array,
    gate_id: jax.Array,        # concrete int
    rng_key: jax.Array,
    data: JAXEmpiricalData,
) -> jax.Array:
    """Sample one residual [delta_vx, delta_vy, delta_yaw_rate] for a feature vector.

    Finds the (gate, regime) pool (or the whole gate if that pool is empty),
    takes the neighbor_count nearest samples in normalised feature space,
    and picks one with softmax weights on the squared distances. The search
    window has size max_pool_size, which covers the largest pool.
    """
    # 1. Compute regime
    regime_id = compute_regime_id_jax(feature_vector)
    
    # 2. Find range in the flattened database
    lookup_idx = gate_id * 720 + regime_id
    start_idx = data.lookup_start[lookup_idx]
    count = data.lookup_count[lookup_idx]
    
    # Fallback to gate pool if regime is empty
    use_gate_pool = (count == 0)
    start_idx = jnp.where(use_gate_pool, data.lookup_gate_start[gate_id], start_idx)
    count = jnp.where(use_gate_pool, data.lookup_gate_count[gate_id], count)
    
    # 3. k-NN over a fixed-size window starting at the pool, so shapes are
    #    static under jit. Entries beyond `count` are masked out.
    normalized_q = (feature_vector - data.feature_mean) / data.feature_std

    total_samples = data.features.shape[0]
    pool_offsets = jnp.arange(data.max_pool_size, dtype=jnp.int32)
    pool_indices = start_idx + pool_offsets
    pool_indices_clipped = jnp.clip(pool_indices, 0, total_samples - 1)

    # Dynamic-valid part of the pool is [0, count).
    valid_mask = pool_offsets < count

    pool_features = data.features[pool_indices_clipped]
    diffs = pool_features - normalized_q
    dists_sq = jnp.sum(jnp.square(diffs), axis=-1)

    # Penalize points outside the active dynamic count.
    penalized_dists = jnp.where(valid_mask, dists_sq, 1e12)
    
    # Find top-k smallest distances
    # jax.lax.top_k returns largest, so we negate
    neg_dists = -penalized_dists
    vals, local_pool_idxs = jax.lax.top_k(neg_dists, data.neighbor_count)
    
    distances = -vals
    
    # 4. Softmax weights, as in common.softmax_sample_weights.
    scale = jnp.mean(distances) + 1e-6
    logits = -distances / scale
    weights = jax.nn.softmax(logits)
    
    # 5. Random choice
    choice_pool_idx = jax.random.choice(rng_key, local_pool_idxs, p=weights)
    choice_idx = pool_indices_clipped[choice_pool_idx]
    
    # 6. Retrieve residual
    return data.residuals[choice_idx]

