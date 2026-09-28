"""Tests for the GMM vs Single-Gaussian multimodal generation project."""

from __future__ import annotations

import numpy as np
import pytest

from app import run_experiment
from data import make_dataset


class TestDataset:
    """Tests for the synthetic data generation."""

    def test_different_seed_changes_data(self) -> None:
        """Different seeds must produce different feature matrices."""
        X1, y1 = make_dataset(seed=42, n_samples=256)
        X2, y2 = make_dataset(seed=99, n_samples=256)
        assert not np.array_equal(X1, X2), "Different seeds should yield different X"

    def test_invalid_n_samples_raises(self) -> None:
        """n_samples < 32 must raise ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 32"):
            make_dataset(seed=42, n_samples=16)

    def test_output_shapes(self) -> None:
        """X and y must have the correct shapes."""
        X, y = make_dataset(seed=42, n_samples=256)
        assert X.shape == (256, 3)
        assert y.shape == (256,)
        assert set(np.unique(y)).issubset({0, 1})


class TestExperiment:
    """Tests for the GMM vs single-Gaussian experiment."""

    def test_gmm_ks_pvalue_beats_baseline(self) -> None:
        """GMM ks_pvalue_mean must be strictly greater than baseline."""
        result = run_experiment(seed=42, n_samples=256)
        gmm_ks = result["metrics"]["ks_pvalue_mean"]
        base_ks = result["baseline_metrics"]["ks_pvalue_mean"]
        assert gmm_ks > base_ks, (
            f"GMM KS p-value mean ({gmm_ks:.4f}) should exceed "
            f"baseline ({base_ks:.4f})"
        )

    def test_gmm_corr_mae_beats_baseline(self) -> None:
        """GMM corr_mae must be strictly lower than baseline."""
        result = run_experiment(seed=42, n_samples=256)
        gmm_corr = result["metrics"]["corr_mae"]
        base_corr = result["baseline_metrics"]["corr_mae"]
        assert gmm_corr < base_corr, (
            f"GMM corr MAE ({gmm_corr:.4f}) should be lower than "
            f"baseline ({base_corr:.4f})"
        )

    def test_gmm_knn_accuracy_above_threshold(self) -> None:
        """GMM utility_knn_acc must exceed 0.85."""
        result = run_experiment(seed=42, n_samples=256)
        knn_acc = result["metrics"]["utility_knn_acc"]
        assert knn_acc > 0.85, (
            f"GMM KNN accuracy ({knn_acc:.4f}) should exceed 0.85"
        )

    def test_reproducibility(self) -> None:
        """Same inputs must give exactly the same output."""
        r1 = run_experiment(seed=42, n_samples=256)
        r2 = run_experiment(seed=42, n_samples=256)
        assert r1["metrics"] == r2["metrics"]
        assert r1["baseline_metrics"] == r2["baseline_metrics"]
        assert r1["n_samples"] == r2["n_samples"]

    def test_respects_n_samples(self) -> None:
        """Different n_samples must be respected in the output."""
        r_small = run_experiment(seed=42, n_samples=64)
        r_large = run_experiment(seed=42, n_samples=512)
        assert r_small["n_samples"] == 64
        assert r_large["n_samples"] == 512

    def test_metrics_are_finite(self) -> None:
        """All metric values must be finite numbers."""
        result = run_experiment(seed=42, n_samples=256)
        for name, metrics in [("metrics", result["metrics"]),
                              ("baseline_metrics", result["baseline_metrics"])]:
            for key, val in metrics.items():
                assert isinstance(val, (int, float)), f"{name}[{key}] not numeric"
                assert np.isfinite(val), f"{name}[{key}] is not finite"
