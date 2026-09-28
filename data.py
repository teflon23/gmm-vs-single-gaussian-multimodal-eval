"""Synthetic 3-D bimodal dataset for GMM-vs-single-Gaussian generation experiments.

The dataset is a mixture of two 3-D Gaussian clusters with distinct means and
covariance structures.  ``y`` is the binary component label (0 or 1).  All
values are deterministic for a given seed; no disk I/O is performed.
"""

from __future__ import annotations

import numpy as np


def make_dataset(seed: int = 42, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return a 3-D bimodal synthetic dataset.

    Parameters
    ----------
    seed : int
        Random seed controlling both the cluster assignments and the sampled
        coordinates.  Different seeds produce different data.
    n_samples : int
        Total number of samples.  Must be >= 32.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 3)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Binary component labels (0 or 1).

    Raises
    ------
    ValueError
        If ``n_samples`` < 32.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)

    # Split samples roughly 50/50 between the two components.
    n0 = n_samples // 2
    n1 = n_samples - n0

    # Component 0: mean near origin, moderate anisotropic covariance.
    mean0 = np.array([0.0, 0.0, 0.0])
    cov0 = np.array(
        [
            [1.0, 0.3, 0.1],
            [0.3, 0.8, 0.2],
            [0.1, 0.2, 0.6],
        ]
    )

    # Component 1: shifted mean, different covariance orientation.
    mean1 = np.array([4.0, -2.0, 3.0])
    cov1 = np.array(
        [
            [0.9, -0.2, 0.4],
            [-0.2, 1.2, 0.1],
            [0.4, 0.1, 0.7],
        ]
    )

    X0 = rng.multivariate_normal(mean0, cov0, size=n0)
    X1 = rng.multivariate_normal(mean1, cov1, size=n1)

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    # Shuffle so rows are not grouped by label.
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]
