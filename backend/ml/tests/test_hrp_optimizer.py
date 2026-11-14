"""Unit tests for ml.hrp_optimizer (Hierarchical Risk Parity)."""

import numpy as np
import pandas as pd
import pytest

from ml import hrp_optimizer as hrp


def test_compute_correlation_distance_properties():
    # Symmetric correlation matrix with negative and positive correlation
    corr = np.array([
        [1.0, 0.8, -0.6],
        [0.8, 1.0, 0.2],
        [-0.6, 0.2, 1.0],
    ])
    dist = hrp.compute_correlation_distance(corr)

    # 1. Diagonal must be zero
    np.testing.assert_allclose(np.diag(dist), 0.0)
    # 2. Symmetry
    np.testing.assert_allclose(dist, dist.T)
    # 3. Distance values strictly within [0, 1]
    assert np.all((dist >= 0.0) & (dist <= 1.0))
    # 4. Perfect positive correlation (0.8) should yield smaller distance than negative (-0.6)
    assert dist[0, 1] < dist[0, 2]
    # Exact metric check: d = sqrt(0.5 * (1 - 0.8)) = sqrt(0.1)
    assert dist[0, 1] == pytest.approx(np.sqrt(0.1), abs=1e-8)


def test_tree_clustering_and_quasi_diagonalize():
    corr = np.array([
        [1.0, 0.95, 0.1, 0.05],
        [0.95, 1.0, 0.15, 0.08],
        [0.1, 0.15, 1.0, 0.90],
        [0.05, 0.08, 0.90, 1.0],
    ])
    dist = hrp.compute_correlation_distance(corr)
    link = hrp.tree_clustering(dist, method="single")
    assert link.shape == (3, 4)

    ordered = hrp.quasi_diagonalize(link)
    assert len(ordered) == 4
    assert set(ordered) == {0, 1, 2, 3}

    # Highly correlated assets (0 & 1) and (2 & 3) should be placed adjacently in the tree
    pos_0, pos_1 = ordered.index(0), ordered.index(1)
    pos_2, pos_3 = ordered.index(2), ordered.index(3)
    assert abs(pos_0 - pos_1) == 1
    assert abs(pos_2 - pos_3) == 1


def test_compute_cluster_variance_positive():
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])
    var = hrp.compute_cluster_variance(cov, [0, 1])
    assert var > 0.0
    # Inverse variance weights: w_0 = (1/0.04)/(1/0.04 + 1/0.09) = 25 / (25 + 11.111) = 0.6923
    assert var < 0.04  # Diversification reduces total cluster variance below the lower variance asset


def test_hrp_optimize_synthetic_returns():
    rng = np.random.default_rng(42)
    # 4 assets: asset_0 and asset_1 are highly volatile; asset_3 is low volatility
    rets = pd.DataFrame({
        "HIGH_VOL_1": rng.normal(0.001, 0.03, size=500),
        "HIGH_VOL_2": rng.normal(0.001, 0.025, size=500),
        "MED_VOL": rng.normal(0.0005, 0.015, size=500),
        "LOW_VOL": rng.normal(0.0003, 0.005, size=500),
    })

    weights = hrp.hrp_optimize(rets)
    assert set(weights.keys()) == set(rets.columns)
    # Weights must sum to 1.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
    # All weights non-negative (long only)
    assert all(w >= 0.0 for w in weights.values())
    # Lowest volatility asset should receive higher allocation than highest volatility
    assert weights["LOW_VOL"] > weights["HIGH_VOL_1"]


def test_hrp_optimize_direct_covariance_matrix():
    cov = np.array([
        [0.10, 0.02, 0.01],
        [0.02, 0.08, 0.015],
        [0.01, 0.015, 0.02],
    ])
    names = ["EQUITY", "HYBRID", "GSEC"]
    weights = hrp.hrp_optimize(cov, asset_names=names)

    assert set(weights.keys()) == set(names)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
    assert weights["GSEC"] > weights["EQUITY"]


def test_hrp_optimize_edge_cases():
    # 1. Single asset
    single_res = hrp.hrp_optimize(pd.DataFrame({"ONLY_ONE": [0.01, -0.02, 0.03]}))
    assert single_res == {"ONLY_ONE": 1.0}

    # 2. Empty input
    empty_res = hrp.hrp_optimize(pd.DataFrame())
    assert empty_res == {}

    # 3. Insufficient observations
    with pytest.raises(ValueError, match="Insufficient observations"):
        hrp.hrp_optimize(pd.DataFrame({"A": [0.01], "B": [0.02]}))

