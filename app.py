"""GMM vs Single-Gaussian synthetic generation on bimodal 3-D data.

Compares a 2-component Gaussian Mixture Model generator against a
single-Gaussian baseline, scoring marginal fidelity (KS), correlation
fidelity (MAE), and downstream KNN classification utility.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from data import make_dataset


def _ks_pvalue_mean(X_syn: np.ndarray, X_real: np.ndarray) -> float:
    """Mean two-sample KS p-value across features (higher = more similar)."""
    pvals = []
    for j in range(X_syn.shape[1]):
        _, p = stats.ks_2samp(X_syn[:, j], X_real[:, j])
        pvals.append(p)
    return float(np.mean(pvals))


def _corr_mae(X_syn: np.ndarray, X_real: np.ndarray) -> float:
    """Mean absolute error over pairwise Pearson correlations (lower = better)."""
    corr_syn = np.corrcoef(X_syn, rowvar=False)
    corr_real = np.corrcoef(X_real, rowvar=False)
    # Compare off-diagonal entries (3 pairwise correlations for 3 features).
    mask = ~np.eye(X_syn.shape[1], dtype=bool)
    return float(np.mean(np.abs(corr_syn[mask] - corr_real[mask])))


def _knn_accuracy(X_syn: np.ndarray, y_syn: np.ndarray,
                  X_real: np.ndarray, y_real: np.ndarray) -> float:
    """KNN accuracy: train on synthetic, evaluate on real test set."""
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_syn, y_syn)
    return float(knn.score(X_real, y_real))


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict:
    """Run the GMM vs single-Gaussian generation comparison.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Total number of samples. Must be >= 32.

    Returns
    -------
    dict
        JSON-serializable results with metrics, baseline_metrics, and explanation.
    """
    X, y = make_dataset(seed=seed, n_samples=n_samples)

    # 70/30 stratified split on the binary component label.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    n_train = len(X_train)

    # --- 2-component GMM generator ---
    gmm = GaussianMixture(
        n_components=2, covariance_type="full", random_state=seed, n_init=1
    )
    gmm.fit(X_train)
    X_syn_gmm, comp_labels = gmm.sample(n_train)

    # Map GMM component indices to the original binary labels (0/1) by
    # checking which component mean is closer to the true component means.
    # We estimate true component means from the training data.
    true_mean_0 = X_train[y_train == 0].mean(axis=0)
    true_mean_1 = X_train[y_train == 1].mean(axis=0)
    
    # Determine mapping: gmm.means_[0] is closer to true_mean_0 or true_mean_1?
    d00 = np.linalg.norm(gmm.means_[0] - true_mean_0)
    d01 = np.linalg.norm(gmm.means_[0] - true_mean_1)
    if d00 < d01:
        # GMM component 0 corresponds to label 0
        y_syn_gmm = comp_labels.copy()
    else:
        # GMM component 0 corresponds to label 1, so flip labels
        y_syn_gmm = 1 - comp_labels

    # --- Single-Gaussian baseline ---
    mean_pooled = X_train.mean(axis=0)
    cov_pooled = np.cov(X_train, rowvar=False)
    # Ensure positive semi-definite for sampling.
    eigvals, eigvecs = np.linalg.eigh(cov_pooled)
    eigvals = np.clip(eigvals, 1e-8, None)
    cov_pooled = eigvecs @ np.diag(eigvals) @ eigvecs.T
    rng = np.random.default_rng(seed + 1)
    X_syn_single = rng.multivariate_normal(mean_pooled, cov_pooled, size=n_train)
    # Assign labels by nearest component mean for KNN training.
    # Use the GMM means as reference for nearest-component assignment.
    comp_means = gmm.means_
    dists = np.linalg.norm(X_syn_single[:, None, :] - comp_means[None, :, :], axis=2)
    y_syn_single = np.argmin(dists, axis=1)
    
    # Map baseline synthetic labels to original binary labels using the same mapping logic
    # Actually, the baseline labels are assigned based on distance to gmm.means_.
    # If gmm.means_[0] is closer to true_mean_0, then argmin=0 means label 0.
    # If gmm.means_[0] is closer to true_mean_1, then argmin=0 means label 1.
    if d00 < d01:
        # GMM component 0 is label 0, so argmin=0 -> label 0, argmin=1 -> label 1
        pass # y_syn_single is already 0/1 corresponding to gmm components
    else:
        # GMM component 0 is label 1, so argmin=0 -> label 1, argmin=1 -> label 0
        y_syn_single = 1 - y_syn_single

    # --- Fidelity metrics (synthetic vs held-out real test) ---
    gmm_metrics = {
        "ks_pvalue_mean": _ks_pvalue_mean(X_syn_gmm, X_test),
        "corr_mae": _corr_mae(X_syn_gmm, X_test),
        "utility_knn_acc": _knn_accuracy(X_syn_gmm, y_syn_gmm, X_test, y_test),
    }
    baseline_metrics = {
        "ks_pvalue_mean": _ks_pvalue_mean(X_syn_single, X_test),
        "corr_mae": _corr_mae(X_syn_single, X_test),
        "utility_knn_acc": _knn_accuracy(X_syn_single, y_syn_single, X_test, y_test),
    }

    return {
        "n_samples": int(n_samples),
        "metrics": gmm_metrics,
        "baseline_metrics": baseline_metrics,
        "explanation": (
            "70/30 stratified split on binary component label (i.i.d. data, no "
            "temporal ordering). GMM (2 components, full covariance) preserves "
            "bimodal marginal structure and per-cluster correlations; the single "
            "Gaussian baseline collapses to one ellipsoid. KS p-value mean "
            "(higher=better) and correlation MAE (lower=better) compare synthetic "
            "vs held-out real test marginals/correlations. KNN accuracy (higher="
            "better) measures whether cluster separability survives generation."
        ),
    }
