"""Distribution of the unsafe-set parameters theta for the racecar.

In the paper, the unsafe set O(theta) depends on a parameter theta that
is perceived with noise; the gatekeeper samples theta from
P_hat^theta_t = P(theta | x_t, z_t) in each rollout. Here theta describes
the track boundaries, and `TrackBoundsEstimate` provides those samples.

Current use: the simulator builds a fixed estimate with no uncertainty,
and `make_track_safety_fn` ignores theta, so the boundaries are treated
as known. To model boundary uncertainty, use a safety function that reads
theta and provide an estimate with nonzero std.

The gatekeeper only needs
    estimate.sample_env_params(M, N, T, rng) -> np.ndarray [M, N, T, theta_dim]
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TrackBoundsEstimate:
    """Gaussian (or fixed) estimate of the left and right track boundaries.

    theta = (left_offset, right_offset): distances from the centreline to the
    left and right boundaries at the current position, in metres. One theta
    is drawn per rollout and held fixed over the horizon (theta is a static
    parameter in the paper).

    Attributes
    ----------
    left_bound_mean, left_bound_std : float
        Mean and standard deviation of the left offset [m].
    right_bound_mean, right_bound_std : float
        Mean and standard deviation of the right offset [m].
    half_width_fallback : float
        Half-width used by the fixed placeholder estimate [m].
    distribution : str
        'gaussian': sample each offset from N(mean, std^2), clipped below at
        0.5 m. 'fixed': always return the means.

    The ambiguity radius beta and the Lipschitz constant L_H are set in
    GatekeeperParams and apply to the whole noise vector, including theta.
    """

    def __init__(
        self,
        left_bound_mean: float,
        left_bound_std: float,
        right_bound_mean: float,
        right_bound_std: float,
        half_width_fallback: float = 4.0,
        distribution: str = "gaussian",
    ):
        self.left_bound_mean = float(left_bound_mean)
        self.left_bound_std = float(left_bound_std)
        self.right_bound_mean = float(right_bound_mean)
        self.right_bound_std = float(right_bound_std)
        self.half_width_fallback = float(half_width_fallback)
        self.distribution = distribution

    # ------------------------------------------------------------------
    # Sampling interface used by the gatekeeper
    # ------------------------------------------------------------------

    def sample_env_params(
        self,
        M: int,
        N: int,
        T: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw theta samples for M candidates x N rollouts.

        Each (m, i) pair gets an independent sample, repeated over the T steps.

        Parameters
        ----------
        M, N, T : int
            Number of candidates, rollouts per candidate, and horizon.
        rng : np.random.Generator

        Returns
        -------
        np.ndarray [M, N, T, 2] float32
            Last axis: [left_offset, right_offset], both positive.
        """
        if self.distribution == "fixed":
            left_samples = np.full(
                (M, N, T), self.left_bound_mean, dtype=np.float32
            )
            right_samples = np.full(
                (M, N, T), self.right_bound_mean, dtype=np.float32
            )

        elif self.distribution == "gaussian":
            # One sample per (m, i) pair, repeated over the T steps: [M, N, T].
            left_samples = (
                rng.normal(
                    self.left_bound_mean,
                    max(self.left_bound_std, 1e-6),
                    size=(M, N, 1),
                )
                .astype(np.float32)
                .repeat(T, axis=2)
            )
            right_samples = (
                rng.normal(
                    self.right_bound_mean,
                    max(self.right_bound_std, 1e-6),
                    size=(M, N, 1),
                )
                .astype(np.float32)
                .repeat(T, axis=2)
            )

            # Keep each boundary at least 0.5 m from the centreline.
            left_samples = np.clip(left_samples, 0.5, None)
            right_samples = np.clip(right_samples, 0.5, None)

        else:
            raise ValueError(
                f"Unknown distribution type {self.distribution!r}. "
                "Expected 'gaussian' or 'fixed'."
            )

        return np.stack([left_samples, right_samples], axis=-1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def theta_dim(self) -> int:
        """Size of theta (2: left and right offsets)."""
        return 2  # [left_offset, right_offset]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_track_width(
        cls,
        half_width: float,
        relative_uncertainty: float = 0.0,
    ) -> "TrackBoundsEstimate":
        """Symmetric estimate from a known half-width.

        relative_uncertainty is the standard deviation as a fraction of
        half_width; 0 gives a 'fixed' estimate (no sampling).
        """
        std = half_width * relative_uncertainty
        return cls(
            left_bound_mean=half_width,
            left_bound_std=std,
            right_bound_mean=half_width,
            right_bound_std=std,
            half_width_fallback=half_width,
            distribution="gaussian" if std > 0 else "fixed",
        )

    @classmethod
    def unknown(cls, half_width_fallback: float = 4.0) -> "TrackBoundsEstimate":
        """Fixed placeholder with both offsets equal to half_width_fallback.

        No sampling, so the gatekeeper performs a non-robust check against this
        width.
        """
        return cls(
            left_bound_mean=half_width_fallback,
            left_bound_std=0.0,
            right_bound_mean=half_width_fallback,
            right_bound_std=0.0,
            half_width_fallback=half_width_fallback,
            distribution="fixed",
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TrackBoundsEstimate("
            f"left={self.left_bound_mean:.2f}±{self.left_bound_std:.2f} m, "
            f"right={self.right_bound_mean:.2f}±{self.right_bound_std:.2f} m, "
            f"dist={self.distribution!r})"
        )
