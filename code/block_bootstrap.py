# Copyright: Juan-Juan Cai, Yicong Lin, Julia Schaumburg and Chenhui Wang
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# ==================================================#
# Researchers are free to use the code.
# Please cite the following paper:
# Juan-Juan Cai, Yicong Lin, Julia Schaumburg and Chenhui Wang (2026). Estimation and inference for the persistence of extremely high temperatures.
#
# Purpose: block (runs-based) bootstrap resampling primitives shared by
#          2_empirics.py (via getSample) and 3_simulation_supp.py
#          (via build_cluster_library / fXb). Functions are reproduced verbatim
#          from the authors' original code.
######################################################

import numpy as np



def runs_clusters(S, r):
    '''
    Partition exceedance times into clusters.

    Parameters
    ----------
    S : array-like of int
        Sorted indices (times) of exceedances.
    r : int
        Run parameter (minimum separation defining different clusters).
        Two exceedances belong to the same cluster if (s_cur - s_prev) < r.
    Returns
    -------
    clusters : list of list of int
        List of clusters, each cluster is a list of exceedance indices.
    '''
    clusters = []
    cur = [S[0]]
    for s_prev, s_cur in zip(S[:-1], S[1:]):
        if (s_cur - s_prev) < r:
            cur.append(s_cur)
        else:
            clusters.append(cur)
            cur = [s_cur]
    clusters.append(cur)
    return clusters


def build_cluster_library(x, k, r):
    '''
    Build a library of clusters and inter-cluster gaps for runs-based bootstrap.
    Parameters
    ----------
    x : array-like, shape (n,)
        Univariate time series.

    k : int
        Tail parameter determining the threshold.
    r : int
        Parameter used to define clusters.
    Returns
    -------
    lib : dict or None
        If there is at least one exceedance, returns a dict with keys:
          - "clusters": list of 1D ndarrays (cluster segments)
          - "gaps": list of 1D ndarrays (gap segments between clusters)
          - "u": float, threshold used
        If there are no exceedances, returns None.
    '''

    # 1. Threshold and exceedance indicator
    u = np.sort(x)[-k]
    I = (x > u).astype(int)
    times = np.flatnonzero(I)
    if times.size == 0:
        return None
    # 2. Identify cluster indices via run rule
    clusters_idx = runs_clusters(times, r)
    clusters, gaps = [], []
    boundaries = []
    # 3. Store cluster boundaries
    for cl in clusters_idx:
        start, end = cl[0], cl[-1]
        boundaries.append((start, end))
    # 4. Extract inter-cluster gaps
    for (s1, e1), (s2, e2) in zip(boundaries[:-1], boundaries[1:]):
        gap_vals = x[e1 + 1: s2]
        gaps.append(gap_vals.copy())
    # 4b. Handle possible leading and trailing gaps
    if boundaries:
        first_start = boundaries[0][0]
        last_end = boundaries[-1][1]
        # Leading gap before the first cluster
        if first_start > 0:
            gaps.insert(0, x[:first_start].copy())
        # Trailing gap after the last cluster
        if last_end < len(x) - 1:
            gaps.append(x[last_end + 1:].copy())
    # 5. Extract clusters (actual values)
    for (s, e) in boundaries:
        clusters.append(x[s: e + 1].copy())
    return {"clusters": clusters, "gaps": gaps, "u": u}


def fXb(n, clusters, gaps, rng):
    '''
    Construct one bootstrap series by alternating resampled gaps and clusters.

    Parameters
    ----------
    n : int
        Desired length of the bootstrap series.

    clusters : list of 1D ndarrays
        Cluster segments extracted from the original series.

    gaps : list of 1D ndarrays
        Gap segments between clusters extracted from the original series.

    rng : numpy random generator-like

    Returns
    -------
    x_b : ndarray, shape (n,)
        Bootstrap sample series of length n.
    '''
    x_b = []
    total_len = 0
    while total_len < n:
        G = gaps[rng.integers(len(gaps))]
        C = clusters[rng.integers(len(clusters))]

        x_b.extend(G.tolist())
        x_b.extend(C.tolist())
        total_len = len(x_b)
        if total_len >= n:
            break
    x_b = np.array(x_b[:n])
    return x_b


def getSample(vX, k, d_L,rng):
    '''
    Generate one bootstrap sample from a time series.

    Parameters
    ----------
    vX : array-like, shape (n,)
        Original time series.

    k : int
        Tail parameter used to define the exceedance threshold in `build_cluster_library`.

    d_L : int
        Parameter used to define clusters in the bootstrap.

    rng : numpy random generator-like
        Randomness source used in resampling.

    Returns
    -------
    vXb : ndarray, shape (n,)
        Bootstrap sample series. If there are no exceedances, returns `vX` unchanged.
    '''
    lib = build_cluster_library(vX, k, d_L)
    if lib is None:
        return vX

    clusters, gaps = lib["clusters"], lib["gaps"]
    vXb = fXb(len(vX), clusters, gaps, rng)
    return vXb
