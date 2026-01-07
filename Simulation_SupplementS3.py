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
# Juan-Juan Cai, Yicong Lin, Julia Schaumburg and Chenhui Wang (2026). Estimation and inference for the persistence of extreme high temperatures.

# Purpose: Simulation studies in Section S3.
######################################################


### Imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import genpareto
from scipy.integrate import quad

###
### ================ DGPs ================
###
def fAR_C(N, z, rng):
    X = np.zeros(N)
    X[0] = rng.standard_cauchy()
    # eps_i: Cauchy(0, 1 - |z|)
    scale = 1.0 - abs(z)
    eps = scale * rng.standard_cauchy(size=N)
    for i in range(1, N):
        X[i] = z * X[i-1] + eps[i]
    return X

def fARC_GPD(N, z, gamma, x_F, mu, rng):
    Y = fAR_C(N, z, rng)        # Step 1
    U = 0.5 + np.arctan(Y)/np.pi   # Step 2
    X = x_F + (mu - x_F) * (1 - U)**(-gamma)   # Step 3 
    return X

def fS0_ARC(T, z, gamma, x_F, mu):
    '''

    Compute S0(T) for the ARC model under a finite-endpoint GPD marginal (gamma < 0).

    Parameters
    ----------
    T : float
        Threshold level at which S0(T) is evaluated.

    z : float
        Dependence parameter.
    gamma : float
        GPD shape parameter. This routine is intended for gamma < 0 (finite endpoint).
    x_F : float
        Finite right endpoint of the marginal distribution.
    mu : float
        Lower bound/location parameter defining the start of support for the finite-endpoint
    Returns
    -------
    s0 : float
        The value of S0(T) computed by numerical integration.

    '''

    # GPD finite-endpoint CDF (gamma < 0)
    def gpd_finite_cdf(x):
        if x >= x_F: return 1.0
        if x < mu: return 0.0
        return 1 - ((x - x_F) / (mu - x_F)) ** (1 / (-gamma))

    # Cauchy(0, 1-z) CDF
    def eps_cdf(x):
        return 0.5 + (1 / np.pi) * np.arctan(x / (1 - z))

    u_T = gpd_finite_cdf(T)
    t_T = np.tan(np.pi * (u_T - 0.5))

    def integrand(x):
        p = 1 - eps_cdf(t_T - z * x)
        fz = 1 / (np.pi * (1 + x * x))
        return p * fz

    s0, _ = quad(integrand, t_T, np.inf, limit=1000)
    return s0


###
### ================ Estimation ================
###
def Theta_ck(data, d, k):
    '''
    Compute our extremal index estimator for a grid of (d, k).

    Parameters
    ----------
    data : array-like, shape (n,)
        Univariate sample or time series.

    d : int or array-like of int
        
    k : int or array-like of int
        Exceedances are defined as the k largest observations.

    Returns
    -------
    result : ndarray, shape (len(k), len(d))
        Matrix of \\hat{\\theta}(d, k) values. Row i corresponds to k[i], column l corresponds to d[l].
    '''
    if np.isscalar(d):
        d = np.array([d])
    if np.isscalar(k):
        k = np.array([k])
    n = len(data)
    R = n + 1 - np.argsort(np.argsort(data, kind="mergesort"))
    result = np.zeros((len(k), len(d)))
    for i, ki in enumerate(k):
        for l, dl in enumerate(d):
            c = 0
            ind = np.where(R <= ki)[0]
            ind = ind[ind <= n - dl]
            for j in ind:
                MR = np.min(R[(j + 1):(j + dl)])
                if MR > ki:
                    c += 1
            result[i, l] = c / ki
    return result

def estimate_theta(data, k, du=20):
    '''
    Estimate the extremal index using our method.
    Parameters
    ----------
    data : array-like, shape (n,)
        Univariate time series (or sample) used for estimation.

    k : int or array-like of int
        Tail index (or indices). Exceedances are defined as the k largest observations.
    du : int, default=20
        Maximum run parameter considered in the grid d = 2,...,du.


    Returns
    -------
    tht_ck : ndarray, shape (len(k),)
        Estimated extremal index values for each entry of `k`, computed at the selected parameter `de`.

    de : ndarray, shape (len(k),)
        Selected parameter d for each k. 
    '''
    k = np.atleast_1d(k)  # if k is a integer, ensure k is an array
    d = np.arange(2, du + 1)
    nd = len(d)
    de = np.full(len(k), du, dtype=float)
    tht_ck = np.zeros(len(k))
    for j, kk in enumerate(k):
        thrd = 1 / np.sqrt(kk)
        thtck = Theta_ck(data=data, d=d, k=kk)
        deltak = np.column_stack((1 - thtck[:, 0], thtck[:, :-1] - thtck[:, 1:]))
        # Find minimal l such that max(deltak[:, l:nd]) < thrd
        for l in range(nd - 1):
            if np.max(deltak[:, l:nd]) < thrd:
                de[j] = l + 1
                break
        # Compute theta depending on de[j]
        if de[j] == 1:
            tht_ck[j] = 1.0
        elif de[j] >= 2:
            tht_ck[j] = Theta_ck(data=data, d= int(de[j]), k=kk)[0,0]
    return tht_ck, de

def estimate_gamma_mle(data, k):
    '''
    Estimate the GPD shape parameter (gamma) by maximum likelihood on threshold excesses.

    Parameters
    ----------
    data : array-like, shape (n,)
        Univariate sample (or time series) of observations.
    k : int
        Number of upper order statistics used as exceedances.
    Returns
    -------
    gamma_hat : float
        Maximum likelihood estimate of the GPD shape parameter gamma.
    '''
    x_sorted = np.sort(data)
    excess = x_sorted[-k:] - x_sorted[-k - 1]
    gamma_hat, _, _ = genpareto.fit(excess, floc=0.0)
    return gamma_hat

def getmarginal(X, T, k_gamma):
    '''

    Estimate the marginal exceedance probability P(X > T) under a finite-endpoint GPD tail.
    Parameters
    ----------
    X : array-like, shape (n,)
        Univariate sample or time series.

    T : float
        Threshold level at which to compute the exceedance probability.

    k_gamma : int
        Number of upper order statistics used to fit the GPD tail.

    Returns
    -------
    p_T : float
        Estimated exceedance probability P(X > T).
    '''
    n = len(X)
    # marginal
    X_sorted = np.sort(X)
    u = X_sorted[-k_gamma - 1]
    excess = X_sorted[-k_gamma:] - u
    p_u = len(excess) / n
    gamma_hat, _, sigma_hat = genpareto.fit(excess, floc=0.0)  # fit pareto
    if gamma_hat >= 0:
        p_T = np.nan  # not a finite-endpoint fit

    xF_hat = u - sigma_hat / gamma_hat
    if T <= u:
        p_T = np.mean(X > T)
    elif T >= xF_hat:
        p_T = 0.0
    else:
        p_T = p_u * (1 + gamma_hat * (T - u) / sigma_hat) ** (-1 / gamma_hat)
    return p_T


###
### ================ Bootstrap ================
###
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

def bootstrapx(data, r,T, k_theta, k_gamma, B, rng):
    '''
    cluster bootstrap for tail parameters and S(T) .

    Parameters
    ----------
    data : array-like, shape (n,)
        Univariate time series (or sample) to bootstrap.

    r : int
        Parameter used to define clusters in `build_cluster_library`. 

    T : float
        Threshold level used in `getmarginal` to estimate the marginal exceedance probability p_T = P(X > T).
    k_theta : int
        Tail parameter used to define exceedances/clusters for the extremal index
        estimation and for building the cluster library.

    k_gamma : int
        Tail sample size used for GPD fitting in `estimate_gamma_mle` and `getmarginal`.

    B : int
        Number of bootstrap replications.

    rng : numpy random generator-like
        Randomness source.

    Returns
    -------
    thetas : ndarray, shape (B,)
        Bootstrap extremal index estimates.

    gammas : ndarray, shape (B,)
        Bootstrap GPD shape parameter estimates.

    Ss : ndarray, shape (B,)
        Bootstrap values of S(T) = p_T * (1 - theta).

    '''
    lib = build_cluster_library(data, k_theta, r)
    thetas = np.zeros(B)
    gammas = np.zeros(B)
    Ss = np.zeros(B)
    if lib is None:
        ts = np.repeat(estimate_theta(data, k_theta)[0], B)
        gs = np.repeat(estimate_gamma_mle(data, k_gamma), B)
        return ts, gs
    clusters, gaps = lib["clusters"], lib["gaps"]
    for b in range(B):
        rng_b = np.random.default_rng(rng.integers(1e9))
        x_b = fXb(len(data), clusters, gaps, rng_b)
        thetas[b], _ = estimate_theta(x_b, k_theta)
        gammas[b] = estimate_gamma_mle(x_b, k_gamma)
        Ss[b] = getmarginal(x_b,T, k_gamma)* (1 - float(thetas[b]))
    return thetas, gammas, Ss

def getci(hat, boot, true, alpha):
    '''
    Construct a basic bootstrap confidence interval and compute coverage of the true value.

    Parameters
    ----------
    hat : float
        Point estimate from the original sample.

    boot : array-like
        Bootstrap estimates of the same statistic.

    true : float
        The true parameter value (used only to compute coverage).

    alpha : float
        Significance level (e.g., 0.05 for a 95% confidence interval).

    Returns
    -------
    out : dict
        Dictionary with:
          - "ci" : list of float
              Two-element list [lower, upper] for the (1-alpha) confidence interval.
          - "cover" : int
              1 if true is inside the interval, else 0.

    '''
    diff = boot - hat
    C_l, C_u = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    ci_lower = hat - C_u
    ci_upper = hat - C_l
    ci = [ci_lower, ci_upper]
    if ci_lower <= true <= ci_upper:
        cover = 1
    else:
        cover = 0
    return {'ci': ci, 'cover': cover}

###
### ================ One iteration of MC ================
###
def one_mc(label, N, m, gamma, x_F, mu, k_theta, k_gamma, B, rng,T, alpha=0.05):
    if label == 'ARC':
        x = fARC_GPD(N, m, gamma, x_F, mu, rng=rng)
        s0 = fS0_ARC(T, m, gamma, x_F, mu)
        theta0 = 1-m
    gamma_hat = estimate_gamma_mle(x, k_gamma)
    # --- Estimate theta_hat and d_L for bootstrap ---
    theta_hat, d_L = estimate_theta(x, k_theta)
    theta_hat = float(theta_hat[0])
    d_L = int(d_L[0])
    s_hat =  getmarginal(x,T, k_gamma)* (1 - theta_hat)
    # -------bootstrap using r_t---------
    r_t = int(max(1, int(1 / theta_hat)))
    
    thetab_rt, gammab_rt, sb_rt = bootstrapx(x, r_t, T,  k_theta, k_gamma, B, rng)
    ci_res_theta_t = getci(theta_hat, thetab_rt, theta0, alpha)
    ci_res_gamma_t = getci(gamma_hat, gammab_rt, gamma, alpha)
    ci_res_s_t = getci(s_hat, sb_rt,s0,alpha)

    # -------bootstrap using r_dL---------
    thetab_rd, gammab_rd, sb_rd = bootstrapx(x, d_L, T, k_theta, k_gamma, B, rng)
    ci_res_theta_d = getci(theta_hat, thetab_rd, theta0, alpha)
    ci_res_gamma_d = getci(gamma_hat, gammab_rd, gamma, alpha)
    ci_res_s_d = getci(s_hat, sb_rd,s0,alpha)

    results = {
        "r_t": r_t,
        "r_d": d_L,
        "theta_hat": theta_hat,
        'gamma_hat': gamma_hat,
        's_hat': s_hat,
        # use 1/theta
        "ci_theta_t": ci_res_theta_t['ci'],
        'cover_theta_t': ci_res_theta_t['cover'],
        "ci_gamma_t": ci_res_gamma_t['ci'],
        "cover_gamma_t": ci_res_gamma_t['cover'],
        'ci_s_t': ci_res_s_t['ci'],
        'cover_s_t': ci_res_s_t['cover'],

        # use d_L
        "ci_theta_d": ci_res_theta_d['ci'],
        'cover_theta_d': ci_res_theta_d['cover'],
        "ci_gamma_d": ci_res_gamma_d['ci'],
        "cover_gamma_d": ci_res_gamma_d['cover'],
        'ci_s_d': ci_res_s_d['ci'],
        'cover_s_d': ci_res_s_d['cover'],
    }
    return results

###
### ================ Plots ================
###
def fPlotcomparer(res, title, save =False):
    # coverage 
    palette = sns.color_palette("Dark2", n_colors=3)
    x = res['frac_theta']
    y1 = res['coverage_theta_d']
    y2 = res['coverage_theta_t']
    # ci
    yc_d = res['mean_cu_theta_d'] - res['mean_cl_theta_d']
    yc_t = res['mean_cu_theta_t'] - res['mean_cl_theta_t']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(x, y1, color=palette[0], linewidth=3, linestyle='-', label = r"$r = \hat{d}_L$")
    ax.plot(x, y2, color=palette[0], linewidth=3, linestyle='dotted',label = r"$r = 1/\hat{\theta}$")
    ax.axhline(0.95, color='gray', linestyle='--', linewidth=2)
    ax.set_ylim(0.6,1)
    ax.set_xlabel(r"$k/n$", fontsize=20)
    ax.set_ylabel(r"Empirical Coverage", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=22, loc ='lower left', frameon=False)  
    
    ax = axes[1]
    ax.plot(x, yc_d, color=palette[0], linewidth=3, linestyle='-')
    ax.plot(x, yc_t, color=palette[0], linewidth=3, linestyle='dotted')
    ax.set_xlabel(r"$k/n$", fontsize=20)
    ax.set_ylabel(r"Empirical Length", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        os.makedirs("boot_plots", exist_ok=True)
        plt.savefig(f"boot_plots/{title}.png", dpi=300) 
    plt.show()
def fPlotcompareN(res,res_l, title, save =False):
    # coverage 
    palette = sns.color_palette("Dark2", n_colors=3)
    x = res['frac_theta']
    ys = res['coverage_theta_d']
    yl = res_l['coverage_theta_d']
    # ci
    yc_s = res['mean_cu_theta_d'] - res['mean_cl_theta_d']
    yc_l = res_l['mean_cu_theta_d'] - res_l['mean_cl_theta_d']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(x, ys, color=palette[2], linewidth=3, linestyle='dashed',label = r"$n = 5000$")
    ax.plot(x, yl, color=palette[0], linewidth=3, linestyle='-',label = r"$n = 15000$")
    ax.axhline(0.95, color='gray', linestyle='--', linewidth=2)

    ax.set_xlabel(r"$k/n$", fontsize=20)
    ax.set_ylabel(r"Empirical Coverage", fontsize=20)
    ax.set_ylim(0.6,1)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)
    
    ax.legend(fontsize=22, loc ='lower left', frameon=False) 
    
    ax = axes[1]
    ax.plot(x, yc_s, color=palette[2], linewidth=3, linestyle='dashed')
    ax.plot(x, yc_l, color=palette[0], linewidth=3, linestyle='-')
    ax.set_xlabel(r"$k/n$", fontsize=20)
    ax.set_ylabel(r"Empirical Length", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        os.makedirs("boot_plots", exist_ok=True)
        plt.savefig(f"boot_plots/{title}.png", dpi=300) 
    plt.show()

###
### ================ Main Functions ================
###
def main():
    # ---- experiment setup ----
    label = 'ARC'
    vm = [0.2, 0.5, 0.8]
    gamma = -0.3
    x_F = 45
    T =38
    mu = 20
    vn = [5000, 15000]
    vfrac = np.arange(0.01, 0.1, 0.001)
    B = 199 
    MC = 1000
    alpha = 0.05
    master_seed = 20251112
    frac_gamma = 0.05
    n_jobs = cpu_count() - 1

    results = []
    for m in vm:
        for n in vn:            
            for frac in vfrac:
                k_theta = int(n*frac)
                ss = np.random.SeedSequence(master_seed + int(k_theta))
                child_ss = ss.spawn(MC)
                rng_list = [np.random.default_rng(s) for s in child_ss]
                # run MC in parallel
                k_gamma = int(frac_gamma*n)
                mc_out = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes", verbose=10)(
                    delayed(one_mc)(label, n, m, gamma, x_F, mu, k_theta, k_gamma, B, rng_list[i], T, alpha=0.05)
                    for i in range(MC))
                mc_df = pd.DataFrame(mc_out)
                # unpack CI columns for means
                ci_theta_t = np.vstack(mc_df["ci_theta_t"].to_numpy())
                ci_theta_d = np.vstack(mc_df["ci_theta_d"].to_numpy())
                ci_gamma_t = np.vstack(mc_df["ci_gamma_t"].to_numpy())
                ci_gamma_d = np.vstack(mc_df["ci_gamma_d"].to_numpy())
                ci_s_t = np.vstack(mc_df["ci_s_t"].to_numpy())
                ci_s_d = np.vstack(mc_df["ci_s_d"].to_numpy())
                results.append({
                    "distribution": label,
                    "MC": MC,
                    "B": B,
                    "n": n,
                    'z': m,
                    'x_F': x_F,
                    'mu': mu,
                    'gamma': gamma,
                    "frac_theta": frac,
                    "frac_gamma": frac_gamma,
                    "theta_hat": mc_df["theta_hat"].mean(),
                    'gamma_hat': mc_df["gamma_hat"].mean(),
                    's_hat': mc_df["s_hat"].mean(),
                    # for r = 1/theta
                    'mean_rt': mc_df["r_t"].mean(),
                    # theta
                    "coverage_theta_t": mc_df["cover_theta_t"].mean(),
                    "mean_cl_theta_t": ci_theta_t[:, 0].mean(),
                    "mean_cu_theta_t": ci_theta_t[:, 1].mean(),
                    # gamma
                    "coverage_gamma_t": mc_df["cover_gamma_t"].mean(),
                    "mean_cl_gamma_t": ci_gamma_t[:, 0].mean(),
                    "mean_cu_gamma_t": ci_gamma_t[:, 1].mean(),
                    # S_T
                    "coverage_s_t": mc_df["cover_s_t"].mean(),
                    "mean_cl_s_t": ci_s_t[:, 0].mean(),
                    "mean_cu_s_t": ci_s_t[:, 1].mean(),
                    # for r = d_L
                    'mean_dL': mc_df["r_d"].mean(),
                    #theta
                    "coverage_theta_d": mc_df["cover_theta_d"].mean(),
                    "mean_cl_theta_d": ci_theta_d[:, 0].mean(),
                    "mean_cu_theta_d": ci_theta_d[:, 1].mean(),
                    # gamma
                    "coverage_gamma_d": mc_df["cover_gamma_d"].mean(),
                    "mean_cl_gamma_d": ci_gamma_d[:, 0].mean(),
                    "mean_cu_gamma_d": ci_gamma_d[:, 1].mean(),
                    # S_T
                    "coverage_s_d": mc_df["cover_s_d"].mean(),
                    "mean_cl_s_d": ci_s_d[:, 0].mean(),
                    "mean_cu_s_d": ci_s_d[:, 1].mean(),
                })
                
    out_df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    fname = f"ARC_S_n{n}_B{B}_MC{MC}_a{alpha:.2f}_{ts}.csv"
    out_df.to_csv(fname, index=False)
    print(f"Saved: {os.path.abspath(fname)}")
main() # Heavy computing, do not suggest.
### Read the result directly instead by the following codes:

allres = pd.read_csv('ARC_S_n15000_B199_MC1000_a0.05_20251119-014335.csv')

z8 = allres[allres['z']==0.8]
z8_small =  z8[z8['n']==5000]
z8_large = z8[z8['n']==15000]

fPlotcomparer(z8_large, title = 'ARC_GPDz8_comparer', save=False) ####### Figure S.1
fPlotcompareN(z8_small,z8_large, title='ARC_GPDz8_comparen', save=False)