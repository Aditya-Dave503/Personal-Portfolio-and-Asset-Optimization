"""
Hierarchical Risk Parity (HRP) Portfolio Optimizer.

Implements Marcos López de Prado's Hierarchical Risk Parity algorithm:
1. Correlation & Angular Distance Matrix Calculation:
       D(i, j) = sqrt(0.5 * (1 - rho_{i, j}))
2. Tree Clustering using hierarchical Ward/single linkage.
3. Quasi-Diagonalization to reorder the covariance matrix along the dendrogram leaves.
4. Recursive Bisection to allocate weights inversely proportional to cluster variance,
   completely bypassing covariance matrix inversion (ill-conditioning / singularity robust).

Reference:
    López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
    The Journal of Portfolio Management, 42(4), 59-69.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


def compute_correlation_distance(corr: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Compute López de Prado's correlation distance metric:
        d_{i, j} = sqrt(0.5 * (1 - rho_{i, j}))

    Properties:
        - d_{i, i} = 0
        - d_{i, j} in [0, 1] for rho in [-1, 1]
        - Satisfies non-negativity, symmetry, and triangle inequality.
    """
    corr_arr = np.asarray(corr, dtype=float)
    # Numerical clip to handle floating-point approximations like 1.0000000000000002
    corr_clipped = np.clip(corr_arr, -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_clipped), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def tree_clustering(
    dist_matrix: np.ndarray,
    method: str = "single"
) -> np.ndarray:
    """
    Perform hierarchical agglomerative clustering on the distance matrix.

    Parameters:
        dist_matrix: (N, N) symmetric distance matrix.
        method: Linkage method ('single', 'ward', 'complete', 'average').
                López de Prado's canonical HRP specification defaults to 'single'.

    Returns:
        (N - 1, 4) linkage matrix.
    """
    # squareform extracts the upper triangle vector required by scipy.cluster.hierarchy.linkage
    condensed_dist = squareform(dist_matrix, checks=False)
    return linkage(condensed_dist, method=method)


def quasi_diagonalize(linkage_matrix: np.ndarray) -> List[int]:
    """
    Sort cluster leaves recursively so that correlated assets are placed next to each other.
    Replaces the matrix permutation step in López de Prado (2016).

    Parameters:
        linkage_matrix: (N - 1, 4) linkage matrix produced by tree_clustering.

    Returns:
        List of integer indices representing the quasi-diagonalized leaf order.
    """
    root_node = to_tree(linkage_matrix, rd=False)

    def _extract_leaves(node) -> List[int]:
        if node.is_leaf():
            return [node.id]
        # Traverse left child then right child
        return _extract_leaves(node.left) + _extract_leaves(node.right)

    return _extract_leaves(root_node)


def compute_cluster_variance(cov: np.ndarray, cluster_indices: List[int]) -> float:
    """
    Compute the inverse-variance portfolio variance for a subset cluster of assets:
        V_cluster = w^T * Cov_sub * w
    where w_i = (1 / sigma_i^2) / sum(1 / sigma_j^2).
    """
    cov_sub = cov[np.ix_(cluster_indices, cluster_indices)]
    inv_diag = 1.0 / np.diag(cov_sub)
    w = inv_diag / np.sum(inv_diag)
    cluster_var = float(w @ cov_sub @ w)
    return max(cluster_var, 1e-12)


def recursive_bisection(cov: np.ndarray, ordered_indices: List[int]) -> pd.Series:
    """
    Recursively bisect ordered assets and split allocation weights inversely
    proportional to cluster variance.

    Parameters:
        cov: (N, N) full covariance matrix.
        ordered_indices: Quasi-diagonalized sequence of asset indices.

    Returns:
        pd.Series of asset weights indexed by the original asset integers.
    """
    weights = pd.Series(1.0, index=ordered_indices, dtype=float)
    clusters = [ordered_indices]

    while len(clusters) > 0:
        # Bisect all clusters in the current queue with length > 1
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            split_point = len(cluster) // 2
            cluster_left = cluster[:split_point]
            cluster_right = cluster[split_point:]

            var_left = compute_cluster_variance(cov, cluster_left)
            var_right = compute_cluster_variance(cov, cluster_right)

            # Weight factor for left cluster: alpha = 1 - (var_left / (var_left + var_right))
            alpha = 1.0 - (var_left / (var_left + var_right))

            # Scale child weights
            weights.loc[cluster_left] *= alpha
            weights.loc[cluster_right] *= (1.0 - alpha)

            if len(cluster_left) > 1:
                new_clusters.append(cluster_left)
            if len(cluster_right) > 1:
                new_clusters.append(cluster_right)

        clusters = new_clusters

    return weights


def hrp_optimize(
    returns_or_cov: Union[pd.DataFrame, np.ndarray],
    asset_names: Optional[List[str]] = None,
    linkage_method: str = "single"
) -> Dict[str, float]:
    """
    Execute full Hierarchical Risk Parity (HRP) portfolio optimization.

    Parameters:
        returns_or_cov: Either:
            1. (T, N) DataFrame or 2D array of historical asset returns, OR
            2. (N, N) symmetric positive semi-definite covariance matrix.
        asset_names: Optional list of N asset identifiers. If returns_or_cov is a DataFrame,
                     its column names are used by default.
        linkage_method: Hierarchical clustering linkage method ('single', 'ward', etc.).

    Returns:
        Dictionary mapping asset name to its optimal portfolio weight in [0, 1], summing to 1.0.
    """
    if isinstance(returns_or_cov, pd.DataFrame):
        names = list(returns_or_cov.columns) if asset_names is None else asset_names
        # If square and symmetric, treat as covariance matrix, else compute cov from returns
        if (
            returns_or_cov.shape[0] == returns_or_cov.shape[1]
            and np.allclose(returns_or_cov.values, returns_or_cov.values.T, atol=1e-6)
        ):
            cov_df = returns_or_cov.copy()
            # Deriving correlation from covariance
            std_diag = np.sqrt(np.diag(cov_df.values))
            std_diag = np.where(std_diag == 0, 1e-8, std_diag)
            corr_mat = cov_df.values / np.outer(std_diag, std_diag)
            corr_df = pd.DataFrame(corr_mat, index=names, columns=names)
        else:
            returns_clean = returns_or_cov.dropna()
            if len(returns_clean) < 2:
                raise ValueError("Insufficient observations to compute covariance matrix.")
            cov_df = returns_clean.cov()
            corr_df = returns_clean.corr()
    else:
        arr = np.asarray(returns_or_cov, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array or DataFrame.")
        n_rows, n_cols = arr.shape
        if n_rows == n_cols and np.allclose(arr, arr.T, atol=1e-6):
            # Square covariance matrix
            cov_df = pd.DataFrame(arr)
            std_diag = np.sqrt(np.diag(arr))
            std_diag = np.where(std_diag == 0, 1e-8, std_diag)
            corr_mat = arr / np.outer(std_diag, std_diag)
            corr_df = pd.DataFrame(corr_mat)
            names = [f"asset_{i}" for i in range(n_cols)] if asset_names is None else asset_names
        else:
            # Rectangular return series (T, N)
            df = pd.DataFrame(arr)
            cov_df = df.cov()
            corr_df = df.corr()
            names = [f"asset_{i}" for i in range(n_cols)] if asset_names is None else asset_names

    if len(names) != cov_df.shape[0]:
        raise ValueError(f"Mismatch: {len(names)} asset names for {cov_df.shape[0]}x{cov_df.shape[1]} covariance.")

    # Edge cases
    n_assets = len(names)
    if n_assets == 0:
        return {}
    if n_assets == 1:
        return {names[0]: 1.0}

    cov = cov_df.values
    corr = corr_df.values

    # Step 1: Distance Matrix
    dist = compute_correlation_distance(corr)

    # Step 2: Tree Clustering
    link = tree_clustering(dist, method=linkage_method)

    # Step 3: Quasi-Diagonalization
    ordered_idx = quasi_diagonalize(link)

    # Step 4: Recursive Bisection
    weights_series = recursive_bisection(cov, ordered_idx)

    # Normalize and sort back to input asset name order
    weights_series = weights_series.sort_index()
    raw_weights = weights_series.to_numpy()
    sum_w = raw_weights.sum()
    if sum_w <= 0:
        normalized = np.repeat(1.0 / n_assets, n_assets)
    else:
        normalized = raw_weights / sum_w

    return {name: float(round(w, 6)) for name, w in zip(names, normalized)}

