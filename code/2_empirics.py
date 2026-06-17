# Copyright: Juan-Juan Cai, Yicong Lin, Julia Schaumburg and Chenhui Wang
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# Purpose: Section 4 and supplement, daily maximum apparent temperature application in Europe.
######################################################


### Imports
import os
import numpy as np
import pandas as pd
from scipy.stats import genpareto
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # headless: write figures to files
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from statsmodels.tsa.stattools import adfuller
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import math

# Shared block-bootstrap resampling primitives (same folder).
from block_bootstrap import runs_clusters, build_cluster_library, fXb, getSample

# Silence harmless, non-fatal warnings for clean console output.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # matplotlib non-GUI backend notice
np.seterr(divide="ignore", invalid="ignore")              # log(0)/0-division in bootstrap CIs

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
            tht_ck[j] = Theta_ck(data=data, d=int(de[j]), k=kk)[0, 0]
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
    Estimate the marginal exceedance probability P(X > T) using a GPD tail model.

    Parameters
    ----------
    X : array-like, shape (n,)
        Univariate sample / time series.

    T : float
        Threshold level at which to estimate the exceedance probability P(X > T).

    k_gamma : int
        Number of upper order statistics used to fit the GPD tail (tail sample size).

    Returns
    -------
    out : dict
        Dictionary with keys:
          - "p_T" : float
              Estimated marginal exceedance probability P(X > T).
          - "u" : float
              Threshold used for the GPD fit.
          - "gamma_hat" : float
              Estimated GPD shape parameter.
          - "sigma_hat" : float
              Estimated GPD scale parameter.
    '''
    from scipy.stats import genpareto
    n = len(X)
    X_sorted = np.sort(X)
    # threshold u
    u = X_sorted[-k_gamma - 1]
    excess = X_sorted[-k_gamma:] - u
    # empirical exceedance prob
    p_u = k_gamma / n
    # fit GPD (shape=gamma_hat, scale=sigma_hat)
    gamma_hat, _, sigma_hat = genpareto.fit(excess, floc=0.0)
    # ------------------------------------------------------------------
    # CASE 1:T below threshold
    # ------------------------------------------------------------------
    if T <= u:
        return {
            "p_T": np.mean(X > T),
            "u": u,
            "gamma_hat": gamma_hat,
            "sigma_hat": sigma_hat
        }
    # ------------------------------------------------------------------
    # CASE 2: Finite endpoint if gamma < 0
    # ------------------------------------------------------------------
    if gamma_hat < 0:
        xF_hat = u - sigma_hat / gamma_hat     # theoretical endpoint
        if T >= xF_hat:
            return {
                "p_T": 1e-12,    # extremely small tail probability
                "u": u,
                "gamma_hat": gamma_hat,
                "sigma_hat": sigma_hat
            }
    # ------------------------------------------------------------------
    # CASE 3: Exponential limit (gamma ≈ 0)
    # ------------------------------------------------------------------
    if abs(gamma_hat) < 1e-8:
        p_T = p_u * np.exp(-(T - u) / sigma_hat)
        return {
            "p_T": p_T,
            "u": u,
            "gamma_hat": gamma_hat,
            "sigma_hat": sigma_hat
        }

    # ------------------------------------------------------------------
    # CASE 4: General GPD tail
    # ------------------------------------------------------------------
    z = 1 + gamma_hat * (T - u) / sigma_hat

    # SUPPORT CHECK
    if z <= 0:
        # outside support of GPD
        p_T = 1e-12
    else:
        p_T = p_u * z ** (-1 / gamma_hat)

    return {
        "p_T": p_T,
        "u": u,
        "gamma_hat": gamma_hat,
        "sigma_hat": sigma_hat
    }

def p_joint_exceedance(series, T):
    x = np.asarray(series)
    indicator = (x[:-1] > T) & (x[1:] > T)
    return indicator.mean()

###
### ================ Bootstrap ================
###
def fBootstrap(City, df_all, B=999, vfrac = np.linspace(0.02,0.08,7), p_S_list = [97, 98, 99.9]):
    '''
    Cluster bootstrap for city-level daily maximum apparent temperature series.

    Parameters
    ----------
    City : sequence of str
        City names to process.

    df_all : pandas.DataFrame
        DataFrame containing the full panel time series.

    B : int, default=999
        Number of bootstrap replications.
    vfrac : array-like, default=np.linspace(0.02, 0.08, 7)
        Fractions used to set tail sample size k in each segment.
    p_S_list : list of float, default=[97, 98, 99.9]
        Percentile levels used to define the temperature threshold on the full series.

    Returns
    -------
    boot_res : pandas.DataFrame
        Long-format results table with one row per (city, frac, p_S) combination.
        Each row contains:
          - point estimates for p1 and p2: theta, S, and GPD tail parameters (gamma, u, sigma)
          - bootstrap arrays of length B for the corresponding quantities
    '''
    rng = np.random.default_rng(12345)
    boot_res = pd.DataFrame(columns=[
        "city", "frac", "k", "p_S", "threshold",
        "tht_p1", "S_p1", "gamma_p1", "u_p1", "sigma_p1",
        "tht_p1_boot", "S_p1_boot", "gamma_p1_boot", "u_p1_boot", "sigma_p1_boot",
        "tht_p2", "S_p2", "gamma_p2", "u_p2", "sigma_p2",
        "tht_p2_boot", "S_p2_boot", "gamma_p2_boot", "u_p2_boot", "sigma_p2_boot"
    ])

    for city in City:
        df_city = df_all[df_all["city"] == city].copy()
        data_full = df_city['tappmax'].values
        vX_p1 = data_full[:ndays]
        gap = data_full[ndays:-1*ndays]
        vX_p2 = data_full[-1*ndays:]
        for frac in vfrac:
            k = int(frac * len(vX_p1))
            kg = int(frac * len(gap))
            for p_S in p_S_list:
                threshold = np.percentile(data_full, p_S)
                # --------------------- p1 -----------------------------
                tht_p1, d_p1 = estimate_theta(vX_p1, k)
                tht_p1 = tht_p1[0]
                d_p1 = d_p1[0]
                res_p1 = getmarginal(vX_p1, threshold, k_gamma=k)
                S_p1 = res_p1['p_T'] * (1 - tht_p1)
                gamma_p1 = res_p1['gamma_hat']
                u_p1 = res_p1['u']
                sigma_p1 = res_p1['sigma_hat']
                # --------------------- gap -----------------------------
                tht_g, d_g = estimate_theta(gap, kg)
                tht_g = tht_g[0]
                d_g = d_g[0]
                # --------------------- p2 -----------------------------
                tht_p2, d_p2 = estimate_theta(vX_p2, k)
                tht_p2 = tht_p2[0]
                d_p2 = d_p2[0]
                res_p2 = getmarginal(vX_p2, threshold, k_gamma=k)
                S_p2 = res_p2['p_T'] * (1 - tht_p2)

                gamma_p2 = res_p2['gamma_hat']
                u_p2 = res_p2['u']
                sigma_p2 = res_p2['sigma_hat']
                ## bootstrap array
                tht_p1_boot = np.zeros(B)
                S_p1_boot = np.zeros(B)
                gamma_p1_boot = np.zeros(B)
                u_p1_boot = np.zeros(B)
                sigma_p1_boot = np.zeros(B)

                tht_p2_boot = np.zeros(B)
                S_p2_boot = np.zeros(B)
                gamma_p2_boot = np.zeros(B)
                u_p2_boot = np.zeros(B)
                sigma_p2_boot = np.zeros(B)

                # BOOTSTRAP
                for b in range(B):
                    rng_b = np.random.default_rng(rng.integers(1e9))
                    # resample each block
                    vX_p1b = getSample(vX_p1, k, d_p1, rng_b)
                    vX_p2b = getSample(vX_p2, k, d_p2, rng_b)
                    gap_b   = getSample(gap, kg, d_g, rng_b)
                    # recombine structure
                    vXb = np.hstack([vX_p1b, gap_b, vX_p2b])

                    # bootstrap thresholds
                    threshold_b = np.percentile(vXb, p_S)
                    # p1 bootstrap
                    tht_p1_b, d_p1_b = estimate_theta(vX_p1b, k)
                    tht_p1_boot[b] = tht_p1_b[0]
                    res_boot_p1 = getmarginal( vX_p1b, threshold_b, k_gamma=k)

                    S_p1_boot[b] = res_boot_p1['p_T']* (1 - tht_p1_boot[b])
                    gamma_p1_boot[b] = res_boot_p1['gamma_hat']
                    u_p1_boot[b] = res_boot_p1['u']
                    sigma_p1_boot[b] = res_boot_p1['sigma_hat']

                    # p2 bootstrap
                    tht_p2_b, d_p2_b = estimate_theta(vX_p2b, k)
                    tht_p2_boot[b] = tht_p2_b[0]
                    res_boot_p2 = getmarginal( vX_p2b, threshold_b, k_gamma=k)

                    S_p2_boot[b] = res_boot_p2['p_T']* (1 - tht_p2_boot[b])
                    gamma_p2_boot[b] = res_boot_p2['gamma_hat']
                    u_p2_boot[b] = res_boot_p2['u']
                    sigma_p2_boot[b] = res_boot_p2['sigma_hat']

                boot_res.loc[len(boot_res)] = {
                    "city": city,
                    "frac": frac,
                    "k": k,
                    "p_S": p_S,
                    "threshold": threshold,

                    "tht_p1": tht_p1,
                    "S_p1": S_p1,
                    "gamma_p1": gamma_p1,
                    "u_p1": u_p1,
                    "sigma_p1": sigma_p1,

                    "tht_p1_boot": tht_p1_boot,
                    "S_p1_boot": S_p1_boot,
                    "gamma_p1_boot": gamma_p1_boot,
                    "u_p1_boot": u_p1_boot,
                    "sigma_p1_boot": sigma_p1_boot,

                    "tht_p2": tht_p2,
                    "S_p2": S_p2,
                    "gamma_p2": gamma_p2,
                    "u_p2": u_p2,
                    "sigma_p2": sigma_p2,

                    "tht_p2_boot": tht_p2_boot,
                    "S_p2_boot": S_p2_boot,
                    "gamma_p2_boot": gamma_p2_boot,
                    "u_p2_boot": u_p2_boot,
                    "sigma_p2_boot": sigma_p2_boot
                }

    return boot_res
### Stationarity Test ###
def fAdf_test(df: pd.DataFrame, ndays: int = 35 * 92) -> pd.DataFrame:
    """
    Compute ADF p-values for each city.
    """
    samples = {
        "full": lambda x: x,
        "p1":   lambda x: x[:ndays],
        "p2":   lambda x: x[-ndays:],
    }
    regressions = ("ct", "c", "n")

    def adf_p(x, reg):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return adfuller(x, regression=reg, autolag="AIC")[1]
    rows = []
    for city, g in df.groupby("city", sort=True):
        x_all = g["tappmax"].to_numpy()
        for sample_name, slicer in samples.items():
            x = slicer(x_all)
            for reg in regressions:
                rows.append({
                    "city": city,
                    "sample": sample_name,
                    "regression": reg,
                    "p_value": adf_p(x, reg),
                })
    return pd.DataFrame(rows)  #####

### empirical diagnostic for Condition D(u_n) ###
def dun_delta(series, kn, max_lag=30, block_size=5):
    """
    Empirical diagnostic for Condition D(u_n).

    Parameters
    ----------
    series : pandas Series
        Observed time series X_t.
    kn : float
        Tail probability. For example, kn=0.05 gives u_n as the 95% quantile.
    max_lag : int
        Maximum separation h.
    block_size : int
        Length of each block used to approximate max_{1 <= t <= q} U_t <= u_n.

    Returns
    -------
    lags : np.ndarray
        h = 1, ..., max_lag.
    delta : np.ndarray
        Absolute probability difference at each h.
    u_n : float
        Empirical threshold.
    """

    arr = series.dropna().to_numpy()
    n = len(arr)

    # High threshold u_n
    u_n = np.quantile(arr, 1 - kn)

    lags = np.arange(1, max_lag + 1)
    delta = np.full(max_lag, np.nan)

    for h in lags:
        # Need two blocks of length block_size separated by h
        # left block:  X[t : t + block_size]
        # right block: X[t + block_size + h : t + 2*block_size + h]
        n_windows = n - (2 * block_size + h) + 1

        if n_windows <= 0:
            continue

        A = np.zeros(n_windows, dtype=bool)
        B = np.zeros(n_windows, dtype=bool)

        for t in range(n_windows):
            left_block = arr[t : t + block_size]
            right_block = arr[t + block_size + h : t + 2 * block_size + h]

            A[t] = np.max(left_block) <= u_n
            B[t] = np.max(right_block) <= u_n

        p_A = A.mean()
        p_B = B.mean()
        p_AB = (A & B).mean()

        delta[h - 1] = abs(p_AB - p_A * p_B)

    return lags, delta, u_n

###
### ================ Confidence Intervals ================
###
def getci(hat, boot, alpha):
    '''
    Compute a basic bootstrap confidence interval using the bootstrap differences.

    Parameters
    ----------
    hat : float
        Point estimate from the original sample.
    boot : array-like
        Bootstrap estimates computed on resampled data.

    alpha : float
        Significance level (e.g., 0.05 for a 95% confidence interval).
    Returns
    -------
    ci : list of float
        Two-element list [lower, upper] giving the (1-alpha) bootstrap CI.
        The upper endpoint is truncated at 1.
    '''
    # for theta, cut at 1
    diff = boot - hat
    C_l, C_u = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    ci_lower = hat - C_u
    ci_upper = hat - C_l
    ci = [ci_lower, min(ci_upper,1)]
    return ci

def logs(s_hat, sboot, alpha):
    '''
    Compute a bootstrap confidence interval on the -log scale for a probability.

    Parameters
    ----------
    s_hat : float
        Original estimate of S(T).

    sboot : array-like
        Bootstrap estimates of S(T).

    alpha : float
        Significance level (e.g., 0.05 for a 95% confidence interval).

    Returns
    -------
    ci : list of float
        Two-element list [lower, upper] for the CI on the -log(S) scale.
        The lower endpoint is truncated at 0.
    '''
    # negative logS
    s_l, s_u = np.quantile(-1 * np.log(sboot) - (-1 * np.log(s_hat)), [alpha / 2, 1 - alpha / 2])
    ci_l = -1 * np.log(s_hat) - s_u
    ci_u = -1 * np.log(s_hat) - s_l
    ci = [max(ci_l, 0), ci_u]
    return ci

def fProcessTheta(boot_res, alpha_list=[0.01, 0.05, 0.10]):
    """
    Build confidence intervals from bootstrap results for theta.

    Parameters
    ----------
    boot_res : pandas.DataFrame
        Output from fBootstrap.

    alpha_list : list of float, default=[0.01, 0.05, 0.10]
        Significance levels for confidence intervals.

    Returns
    -------
    df_thetaci : pandas.DataFrame
        Long-format CI table.
    """
    df_thetaci = pd.DataFrame(columns=[
        "city", "frac", "k", "alpha", "p_S", "threshold", "period", "theta", "ci_theta"
    ])

    for _, row in boot_res.iterrows():
        city = row["city"]
        frac = row["frac"]
        k = row["k"]
        p_S = row["p_S"]
        threshold = row["threshold"]

        # PERIOD 1
        theta_p1 = row["tht_p1"]
        boot_p1 = row["tht_p1_boot"]
        for alpha in alpha_list:
            ci_theta = getci(theta_p1, boot_p1, alpha)
            df_thetaci.loc[len(df_thetaci)] = {
                "city": city, "frac": frac, "k": k, "alpha": alpha,
                "p_S": p_S, "threshold": threshold, "period": 1,
                "theta": theta_p1, "ci_theta": ci_theta
            }

        # PERIOD 2
        theta_p2 = row["tht_p2"]
        boot_p2 = row["tht_p2_boot"]
        for alpha in alpha_list:
            ci_theta = getci(theta_p2, boot_p2, alpha)
            df_thetaci.loc[len(df_thetaci)] = {
                "city": city, "frac": frac, "k": k, "alpha": alpha,
                "p_S": p_S, "threshold": threshold, "period": 2,
                "theta": theta_p2, "ci_theta": ci_theta
            }

    return df_thetaci

def fProcessS(boot_res, alpha_list=[0.01, 0.05, 0.10]):
    """
    Build confidence intervals from bootstrap results for S, using -log(S) bootstrap intervals.
    Parameters
    ----------
    boot_res : pandas.DataFrame
        Output from fBootstrap.

    alpha_list : list of float, default=[0.01, 0.05, 0.10]
        Significance levels for confidence intervals.

    Returns
    -------
    df_Sci : pandas.DataFrame
        Long-format CI table .
    """
    df_Sci = pd.DataFrame(columns=[
        "city", "frac", "k", "alpha", "p_S", "threshold",
        "period", "S", "ci_S", "logS", "ci_logS"
    ])

    for _, row in boot_res.iterrows():
        city = row["city"]
        frac = row["frac"]
        k = row["k"]
        p_S = row["p_S"]
        threshold = row["threshold"]

        # PERIOD 1
        S_p1 = row["S_p1"]
        boot_S_p1 = row["S_p1_boot"]
        for alpha in alpha_list:
            ci_logS = logs(S_p1, boot_S_p1, alpha)

            # convert back: careful with interval ordering
            ci_S = np.exp(-1 * np.array(ci_logS))
            ci_S = ci_S[::-1]  # reverse to get [lower, upper]

            df_Sci.loc[len(df_Sci)] = {
                "city": city, "frac": frac, "k": k, "alpha": alpha,
                "p_S": p_S, "threshold": threshold, "period": 1,
                "S": S_p1, "ci_S": ci_S,
                "logS": -np.log(S_p1), "ci_logS": ci_logS
            }

        # PERIOD 2
        S_p2 = row["S_p2"]
        boot_S_p2 = row["S_p2_boot"]
        for alpha in alpha_list:
            ci_logS = logs(S_p2, boot_S_p2, alpha)

            ci_S = np.exp(-1 * np.array(ci_logS))
            ci_S = ci_S[::-1]

            df_Sci.loc[len(df_Sci)] = {
                "city": city, "frac": frac, "k": k, "alpha": alpha,
                "p_S": p_S, "threshold": threshold, "period": 2,
                "S": S_p2, "ci_S": ci_S,
                "logS": -np.log(S_p2), "ci_logS": ci_logS
            }

    return df_Sci

###
### ======================== Plots ========================
###
### Figure 2 ###
def fPlotStation_info(cities, save=False, folder=None):
    # plot the location of the selected cities
    df = pd.DataFrame([{"STANAME": name, "LAT": lat, "LON": lon} for name, (lat, lon) in cities.items()])
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 12),
                            subplot_kw={"projection": proj},
                            dpi=300)
    # Base map
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,alpha = 0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha = 0.8)
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.add_feature(cfeature.RIVERS, alpha=0.4)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    # Cities
    ax.scatter(df["LON"], df["LAT"], color='red', s=50, transform=ccrs.PlateCarree(), zorder=5)
    for _, row in df.iterrows():
        ax.text(row["LON"] + 0.2, row["LAT"] + 0.2, row["STANAME"], fontsize=20, transform=ccrs.PlateCarree(), zorder=6)
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent([-5, 30, 30, 60])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if save:
        if folder is None:
            folder = "Results/Figures/Figure2"

        os.makedirs(folder, exist_ok=True)

        plt.savefig(
            os.path.join(folder, "Figure2_Map.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()
def fPlotTS(df, city,legend = False, save=False, folder=None):
    #plot the time series overview from 1940-2025 for a certain city
    sub_df = df[df["city"] == city].copy()
    data = sub_df['tappmax'].values
    years = sub_df['date'].dt.year.values
    # ===== Segments =====
    seg1 = (years >= 1940) & (years <= 1974)
    seg2 = (years >= 1975) & (years <= 1990)
    seg3 = (years >= 1991) & (years <= 2025)
    # ===== Segment-specific 95% thresholds =====
    p97_seg1 = np.percentile(data[seg1], 95)
    p97_seg3 = np.percentile(data[seg3], 95)
    palette = sns.color_palette("Dark2", n_colors=3)
    clr_line = palette[0]
    clr_pts = palette[2]
    fig = plt.figure(figsize=(12, 4), dpi=300)
    sns.set_style("white")
    plt.grid(False)
    # ===== Segment 1 =====
    plt.plot( np.where(seg1)[0], data[seg1], color=clr_line, linewidth=1.4, label=r"$T^{\max}_{\text{app}}$")
    # ===== Gap =====
    plt.plot( np.where(seg2)[0], data[seg2], color="gray", linestyle=":", linewidth=1.2, label=r"$T^{\max}_{\text{app}}$ (Gap)")
    # ===== Segment 3 =====
    plt.plot( np.where(seg3)[0], data[seg3], color=clr_line, linewidth=1.4, label="_nolegend_")
    # ===== Period-specific exceedances =====
    mask = np.zeros_like(data, dtype=bool)
    mask[seg1] = data[seg1] > p97_seg1
    mask[seg3] = data[seg3] > p97_seg3

    plt.scatter(
        np.where(seg1 & mask)[0], data[seg1 & mask],
        color=clr_pts, s=20,
        label=r"$T^{\max}_{\text{app}} > \mathcal{T}_{95}$"
    )
    plt.scatter(
        np.where(seg2 & mask)[0], data[seg2 & mask],
        color=clr_pts, s=20, alpha=0.1,
        label="_nolegend_"
    )
    plt.scatter(
        np.where(seg3 & mask)[0], data[seg3 & mask],
        color=clr_pts, s=20,
        label="_nolegend_"
    )

    # ===== Segmented threshold lines =====
    plt.plot(
        np.where(seg1)[0],
        np.full(seg1.sum(), p97_seg1),
        color="red", linestyle="--", linewidth=2
    )

    plt.plot(
        np.where(seg3)[0],
        np.full(seg3.sum(), p97_seg3),
        color="red", linestyle="--", linewidth=2
    )

    # ===== threshold labels =====
    x1 = np.where(seg2)[0][0]   # start point
    x3 = np.where(seg3)[0][-1]

    plt.text(
    x1, p97_seg1 - 2,
    fr"$\mathcal{{T}}_{{95}} = {p97_seg1:.2f}^\circ\!C$",
    color="red", fontsize=16, fontweight="bold",
    ha="right",
    va="top",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0))

    plt.text(
        x3, p97_seg3 - 2,
        fr"$\mathcal{{T}}_{{95}} = {p97_seg3:.2f}^\circ\!C$",
        color="red", fontsize=16, fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0)
    )
    # ===== X ticks =====
    x_ticks = [
        np.where(seg1)[0][0],
        np.where(seg1)[0][-1],
        np.where(seg2)[0][-1],
        np.where(seg3)[0][-1]
    ]
    x_labels = ["1940", "1974", "1991", "2025"]
    plt.xticks(x_ticks, x_labels, fontsize=22)
    plt.yticks([20, 30, 40], fontsize=22)
    plt.ylim(5, 45)

    # ===== Axis styling =====
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color("#AAAAAA")
        spine.set_linewidth(0.8)

    plt.title(city, fontsize=26)

    # ===== Clean simplified legend =====
    legend_elements = [
        Line2D([0], [0], color=clr_line, lw=2,
               label=r"$X_t$"),

        Line2D([0], [0], color="gray", lw=1.5, ls=":",
               label=r"$X_t$ (Gap)"),

        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=clr_pts, markersize=8,
               label=r"$ X_t > \mathcal{T}_{95}$"),
        Line2D([0], [0], color="red", lw=2, ls="--",
           label=r"$\mathcal{T}_{95}$"),
    ]
    if legend:
        plt.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=4,
            frameon=False,
            fontsize=22
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    if save:

        if folder is None:
            folder = os.path.join("Results", "Figures", "Figure2")

        os.makedirs(folder, exist_ok=True)

        path = os.path.join(
            folder,
            f"Figure2_TimeSeries_{city}.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Figure saved to {path}")

    plt.show()
    plt.close()

### Figure 3, Figure S.4 ###
def fSelectdL(df_all, city, d=np.arange(2, 21), k=np.arange(65, 258), save=False, legend=True, folder=None):
    df_city = df_all[df_all["city"] == city].copy()
    data_full = df_city["tappmax"].values
    days = 35 * 92
    # split periods
    vX_p1 = data_full[:days]
    vX_p2 = data_full[-days:]
    def compute_deltak(data, d, k):
        tht = Theta_ck(data=data, d=d, k=k)
        return np.column_stack((1 - tht[:, 0], tht[:, :-1] - tht[:, 1:]))
    deltak_p1 = compute_deltak(vX_p1, d, k)
    deltak_p2 = compute_deltak(vX_p2, d, k)
    # -------- PLOT 2×1 --------
    fig, axes = plt.subplots(2, 1, figsize=(9, 12), dpi=200, sharex=True, sharey=True)
    palette = sns.color_palette("Dark2", 4)
    # ============================
    # TOP PANEL: 1940–1974
    # ============================
    ax = axes[0]
    for i in range(4):
        if i ==0:
            lw = 5
        else:
            lw = 2.2
        ax.plot(k / days, deltak_p1[:, i],
                color=palette[i], linewidth=lw,
                label=fr"$s={i+1}$")
    ax.plot(k / days, 1 / np.sqrt(k),
            color="black", linestyle="--", linewidth=3,
            label=r"$1/\sqrt{k}$")

    ax.set_title("1940–1974", fontsize=30)
    ax.set_ylabel(r"$\hat\delta(s)$", fontsize=30)
    ax.set_ylim([-0.01, 0.7])
    ax.tick_params(axis="both", labelsize=30)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(frameon=False, fontsize=30, ncol=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ============================
    # BOTTOM PANEL: 1991–2025
    # ============================
    ax = axes[1]
    for i in range(4):
        if i ==0:
            lw = 5
        else:
            lw = 2.2
        ax.plot(k / days, deltak_p2[:, i],
                color=palette[i], linewidth=lw,
                label=fr"$s={i+1}$")

    ax.plot(k / days, 1 / np.sqrt(k),
            color="black", linestyle="--", linewidth=3)

    ax.set_title("1991–2025", fontsize=30)
    ax.set_xlabel(r"$k/n$", fontsize=30)
    ax.set_ylabel(r"$\hat\delta(s)$", fontsize=30)
    ax.set_ylim([-0.01, 0.7])
    ax.tick_params(axis="both", labelsize=30)
    ax.grid(True, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------- City title at top --------
    fig.suptitle(city, fontsize=32)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    # ---- force city name centered ----
    fig.subplots_adjust(top=0.90)
    fig.suptitle(city, fontsize=32, x=0.55)
    if save:
        if folder is None:
            folder = "Results/Figures/Figure3"

        os.makedirs(folder, exist_ok=True)

        plt.savefig(
            os.path.join(folder, f"Figure3_SelectdL_{city}.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()
### Figure 4 ###
def plot_period_parameter(df, para, frac, p_S, ylabel, estimate, ci_col,legend=True, ax=None):
    if ax is None:
        ax = plt.gca()
    # ---- filter by parameters ----
    df = df[(df['frac'] == frac) & (df['p_S'] == p_S)]
    cities = df['city'].unique()
    periods = np.sort(df['period'].unique())  # should be [1,2]

    # ---- styling ----
    fmts = ["o", "D"]
    palette = sns.color_palette("Dark2", n_colors=4)

    if para == 'theta':
        colors = [palette[0], palette[2]]
    else:
        colors = [palette[1], palette[3]]

    width = 0.18
    offsets = (np.arange(len(periods)) - (len(periods)-1)/2) * width
    period_labels = ['1940–1974', '1991–2025']

    x = np.arange(len(cities))

    # ===== MAIN LOOP =====
    for j, period in enumerate(periods):

        dfp = df[df['period'] == period]

        y = []
        err_low = []
        err_high = []
        x_plot = []

        for i, city in enumerate(cities):

            # possible skip rule (optional)
            # if para == "S" and (city == "Barcelona" ):
            #     continue

            row = dfp[dfp["city"] == city].iloc[0]

            v = row[estimate]
            low, high = row[ci_col]

            y.append(v)
            err_low.append(v - low)
            err_high.append(high - v)
            x_plot.append(i)

        x_plot = np.array(x_plot) + offsets[j]
        yerr = np.vstack([err_low, err_high])

        # plot
        ax.errorbar(
            x_plot, y, yerr=yerr,
            fmt=fmts[j], markersize=8,
            elinewidth=2.4, capsize=0,
            color=colors[j], label=period_labels[j]
        )

    # ----- X axis -----
    if para == "S":
        ax.set_xticks(x)
        ax.set_xticklabels(cities, fontsize=18)
    else:
        ax.set_xticks([])

    # ----- Y axis -----
    ax.tick_params(axis='y', labelsize=22)
    ax.set_ylabel(ylabel, fontsize=22)

    # styling
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if legend:
        ax.legend(
            fontsize=20,
            frameon=False,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5)
        )
def add_separator(ax, y_pos=0.0, x_pad=0, color="0.85", lw=1.4):
    line = Line2D(
        [x_pad, 1 - x_pad],
        [y_pos, y_pos],
        transform=ax.transAxes,
        color=color,
        linewidth=lw,
        solid_capstyle="butt",
        zorder=0,
        clip_on=False
    )
    ax.add_line(line)
def fPlotFig4(df_thetaci, df_Sci, save = False,folder=None):
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    # ===== two panels =====
    plot_period_parameter(
        df_thetaci, para='theta',
        frac =0.05, p_S= 97,
        ylabel=r'$\hat{\theta}$',
        estimate="theta", ci_col="ci_theta",
        ax=axs[0]
    )
    axs[0].set_yticks([0, 0.5, 1])

    plot_period_parameter(
        df_Sci, para='S',
        frac = 0.05, p_S=97,
        ylabel=r'$\hat S(\mathcal{T}^{\mathrm{fs}}_{97})$',
        estimate="S", ci_col="ci_S",
        legend=True,
        ax=axs[1]
    )
    axs[1].set_yticks([0, 0.5, 1])
    # # ===== separators =====
    add_separator(axs[0], color="0.85", lw=1.4)
    # ===== tuning =====
    axs[1].set_ylim(-0.01, 0.06)
    axs[1].set_yticks([0, 0.02, 0.04])
    plt.subplots_adjust(hspace=0.25)
    if save:

        if folder is None:
            folder = os.path.join(
                "Results",
                "Figures",
                "Figure4"
            )

        os.makedirs(folder, exist_ok=True)

        path = os.path.join(
            folder,
            "Figure4_Theta_S.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Figure saved to {path}")

    plt.show()
    plt.close()

### Figure S.3 ###
def plot_dun(df, City,save=False, folder=None):
    MAX_LAG = 30
    BLOCK_SIZE = 5
    # ── sub-period data frames ────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df_p1 = df[(df['year'] >= 1940) & (df['year'] <= 1974)]
    df_p2 = df[(df['year'] >= 1991) & (df['year'] <= 2025)]
    PERIODS = {'1940–1974': df_p1, '1991–2025': df_p2}
    CITY_ORDER    =City

    KN_VALUES     = [0.03, 0.05, 0.07]
    # ── visual constants ──────────────────────────────────────────────────
    palette = sns.color_palette("Dark2", n_colors=3)
    KN_COLORS = {kn: palette[i] for i, kn in enumerate(KN_VALUES)}
    KN_LS     = {0.03: '-',       0.05: '--',       0.07: ':'}
    KN_LABEL  = {0.03: r'$k/n=3\%$', 0.05: r'$k/n=5\%$',
                 0.07: r'$k/n=7\%$'}
    # -------------------------------------------------------------------
    # Compute results
    # -------------------------------------------------------------------
    results = {}
    for city in CITY_ORDER:
        results[city] = {}
        for pname, pdf in PERIODS.items():
            results[city][pname] = {}
            sub = pdf.loc[pdf["city"] == city, "tappmax"].dropna().reset_index(drop=True)

            for kn in KN_VALUES:
                lags, delta, u_n = dun_delta(
                    sub,
                    kn,
                    max_lag=MAX_LAG,
                    block_size=BLOCK_SIZE
                )

                results[city][pname][kn] = {
                    "lags": lags,
                    "delta": delta,
                    "u_n": u_n
                }

    period_names = list(PERIODS.keys())
    assert len(period_names) == 2, "This layout is designed for exactly 2 periods."

    p1, p2 = period_names[0], period_names[1]
    # -------------------------------------------------------------------
    # Layout: 3 city columns, each city has 2 stacked panels
    # -------------------------------------------------------------------
    ncols_city = 3
    n_city = len(CITY_ORDER)
    nrows_city = math.ceil(n_city / ncols_city)

    # common y-limit
    global_ymax = 0
    for city in CITY_ORDER:
        for pname in period_names:
            for kn in KN_VALUES:
                global_ymax = max(global_ymax, np.max(results[city][pname][kn]["delta"]))

    global_ymax = max(0.055, global_ymax * 1.05)

    fig = plt.figure(figsize=(18, 5.2 * nrows_city), dpi=200)
    outer = fig.add_gridspec(
        nrows=nrows_city,
        ncols=ncols_city,
        wspace=0.20,
        hspace=0.28
    )

    for idx, city in enumerate(CITY_ORDER):
        r = idx // ncols_city
        c = idx % ncols_city

        inner = outer[r, c].subgridspec(2, 1, hspace=0.50)

        ax_top = fig.add_subplot(inner[0, 0])
        ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top, sharey=ax_top)

        # ----------------------------
        # TOP PANEL: p1
        # ----------------------------
        for i, kn in enumerate(KN_VALUES):
            rr = results[city][p1][kn]
            lw = 4.0 if i == 0 else 2.0
            ax_top.plot(
                rr["lags"],
                rr["delta"],
                color=KN_COLORS[kn],
                lw=lw,
                ls=KN_LS[kn],
                label=KN_LABEL[kn]
            )

        # ----------------------------
        # BOTTOM PANEL: p2
        # ----------------------------
        for i, kn in enumerate(KN_VALUES):
            rr = results[city][p2][kn]
            lw = 4.0 if i == 0 else 2.0
            ax_bot.plot(
                rr["lags"],
                rr["delta"],
                color=KN_COLORS[kn],
                lw=lw,
                ls=KN_LS[kn]
            )
        # ----------------------------
        # Common styling
        # ----------------------------
        for ax in (ax_top, ax_bot):
            ax.set_xlim(0.5, MAX_LAG + 0.5)
            ax.set_ylim(0, global_ymax)
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

            # only requested y-ticks
            ax.yaxis.set_major_locator(mticker.FixedLocator([0.01, 0.03, 0.05]))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

            ax.tick_params(axis="both", labelsize=16)
            ax.grid(True, alpha=0.25)

            for spine in ax.spines.values():
                spine.set_visible(False)

        # hide x tick labels on top panel
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # year labels only
        ax_top.set_title(p1, fontsize=18, pad=6)
        ax_bot.set_title(p2, fontsize=18, pad=6)

        # city title once per block, above top panel
        ax_top.text(
            0.5, 1.18, city,
            transform=ax_top.transAxes,
            ha="center", va="bottom",
            fontsize=22, fontweight="normal"
        )

        # y-label only on first column
        if c == 0:
            ax_top.set_ylabel(r"$\hat\tau_l(g)$", fontsize=18)
            ax_bot.set_ylabel(r"$\hat\tau_l(g)$", fontsize=18)

        # x-label only on bottom row
        if r == nrows_city - 1:
            ax_bot.set_xlabel(r" $g$ ", fontsize=18)
    # -------------------------------------------------------------------
    # Hide unused cells
    # -------------------------------------------------------------------
    n_total_cells = nrows_city * ncols_city
    for idx in range(n_city, n_total_cells):
        r = idx // ncols_city
        c = idx % ncols_city
        ax_dummy = fig.add_subplot(outer[r, c])
        ax_dummy.axis("off")

    # -------------------------------------------------------------------
    # Global legend
    # -------------------------------------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=KN_COLORS[kn],
            lw=4.0 if i == 0 else 2.0,
            ls=KN_LS[kn],
            label=KN_LABEL[kn]
        )
        for i, kn in enumerate(KN_VALUES)
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=18,
        frameon=False,
    )
    if save:
        if folder is None:
            folder = os.path.join("Results", "Figures", "FigureS3")

        os.makedirs(folder, exist_ok=True)

        path = os.path.join(folder, "FigureS3_Dun.png")

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Figure saved to {path}")

    plt.show()
    plt.close()



### Figure S.5 ###
def plot_theta_city(df_thetaci, city, legend =True, save =False, folder=None):
    dfc = df_thetaci[df_thetaci["city"] == city].copy()
    dfc = dfc.sort_values(["frac", "period"])
    # period 1
    df1 = dfc[dfc["period"] == 1]
    x1 = df1["frac"].values
    theta1 = df1["theta"].values
    ci1 = np.vstack(df1["ci_theta"].values)
    ci1_low, ci1_high = ci1[:,0], ci1[:,1]
    # period 2
    df2 = dfc[dfc["period"] == 2]
    x2 = df2["frac"].values
    theta2 = df2["theta"].values
    ci2 = np.vstack(df2["ci_theta"].values)
    ci2_low, ci2_high = ci2[:,0], ci2[:,1]
    palette = sns.color_palette("Dark2", n_colors=3)
    col1 = palette[0]   # Period 1
    col2 = palette[2]   # Period 2
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    # period 1
    ax.plot(x1, theta1, color=col1, linewidth=2.5,
            label=r"1940–1974")
    ax.fill_between(x1, ci1_low, ci1_high,
                    color=col1, alpha=0.25)

    # period 2
    ax.plot(x2, theta2, color=col2, linewidth=2.5,
            label=r"1991–2025")
    ax.fill_between(x2, ci2_low, ci2_high,
                    color=col2, alpha=0.25)

    # labels & styling
    ax.set_title(fr"{city}", fontsize=24)
    ax.set_xlabel(r"$k/n$", fontsize=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$\hat{\theta}$", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.3)

    if legend:
        ax.legend(fontsize=20, frameon=False)
    # remove black frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    if save:

        if folder is None:
            folder = os.path.join(
                "Results",
                "Figures",
                "FigureS5"
            )

        os.makedirs(folder, exist_ok=True)

        png_path = os.path.join(
            folder,
            f"FigureS5_ThetaCI_{city}.png"
        )

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Figure saved to {png_path}")
###
### ======================== Tables ========================
###
### Table 3 ###
def fTable3(ndays,City,df_all, export=False,folder=None):
    r'''
    Get Table 3of \hat\theta.
    '''
    vfrac = np.array([0.03, 0.05, 0.07])
    vk = (ndays*vfrac).astype(int)
    nk = len(vk)
    results = []
    for city in City:
        vtht_p1 = np.zeros(nk)
        vtht_p2 = np.zeros(nk)
        vd_p1 = np.zeros(nk)
        vd_p2 = np.zeros(nk)
        vT_p1, vT_p2 = np.zeros(nk), np.zeros(nk)
        for i, k in enumerate(vk):
            df_city = df_all[df_all["city"] == city].copy()
            data_full = df_city['tappmax'].values
            vX_p1 = data_full[:ndays]
            vX_p2 = data_full[-1 * ndays:]
            tht_p1, d_p1 = estimate_theta(vX_p1, k)
            vT_p1[i] = np.sort(vX_p1)[-k]
            vtht_p1[i], vd_p1[i] = tht_p1[0], d_p1[0]
            tht_p2, d_p2 = estimate_theta(vX_p2, k)
            vtht_p2[i], vd_p2[i] = tht_p2[0], d_p2[0]
            vT_p2[i] = np.sort(vX_p2)[-k]
            results.append({
                "city": city,
                "frac": vfrac[i],
                'k': k,
                'T_p1': vT_p1[i],
                "theta_p1": vtht_p1[i],
                "d_p1": vd_p1[i],
                'T_p2': vT_p2[i],
                "theta_p2": vtht_p2[i],
                "d_p2": vd_p2[i],
            })
    if export:
        if folder is None:
            folder = os.path.join("Results", "Tables")

        os.makedirs(folder, exist_ok=True)

        csv_path = os.path.join(folder, "Table3.csv")

        pd.DataFrame(results).to_csv(
            csv_path,
            index=False
        )

        print(f"Table saved to {csv_path}")
    return pd.DataFrame(results)
### Table S.1 ###
def fTableS1(boot_res,City, df_all, export = False, folder=None):
    cols_no_boot = [c for c in boot_res.columns if "boot" not in c]
    df_clean = boot_res.loc[boot_res['frac'] == 0.05, cols_no_boot]
    # empirical joint exceedance estimator
    df_clean["Sc_p1"] = np.nan
    df_clean["Sc_p2"] = np.nan
    for city in City:
        for p_S in np.unique(df_clean.loc[df_clean["city"] == city, "p_S"]):
            mask = (df_clean["city"] == city) & (df_clean["p_S"] == p_S)
            sub_df = df_all[df_all["city"] == city].copy()
            data = sub_df["tappmax"].values
            # percentile threshold
            T = np.percentile(data, p_S)
            p1 = data[:ndays]
            p2 = data[-ndays:]
            # compute exceedance probabilities
            Sc_p1 = p_joint_exceedance(p1, T)
            Sc_p2 = p_joint_exceedance(p2, T)
            df_clean.loc[mask, "Sc_p1"] = Sc_p1
            df_clean.loc[mask, "Sc_p2"] = Sc_p2
    if export:
        if folder is None:
            folder = os.path.join("Results", "Tables")

        os.makedirs(folder, exist_ok=True)

        output_path = os.path.join(folder, "TableS1.csv")

        df_clean.to_csv(
            output_path,
            index=False
        )

        print(f"Table saved to {output_path}")
    return df_clean


###
### ======================== Main Functions ========================
###
City = ["London", "Paris", "Munich", "Budapest", "Milan", "Rome", "Barcelona", "Valencia", "Athens"]
# City information (lat, lon)
city_info = {
    "Athens": (37.98, 23.72),
    "Barcelona": (41.39, 2.17),
    "Budapest": (47.50, 19.04),
    "London": (51.51, -0.13),
    "Milan": (45.46, 9.19),
    "Munich": (48.14, 11.58),
    "Paris": (48.86, 2.35),
    "Rome": (41.90, 12.50),
    "Valencia": (39.47, -0.38)
    }
City_main = ["London", "Milan", "Athens"]
City_supp = ["Paris", "Munich", "Budapest", "Rome", "Barcelona", "Valencia"]
ndays = 35 * 92   # 35 summers x 92 days; module-level global, used by fBootstrap & fTableS1

# ---- Paths (self-contained; no shared config module) ----
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")
FIG_ROOT = os.path.join(ROOT, "output", "figures")
TABLE_DIR = os.path.join(ROOT, "output", "tables")
FIG2_DIR = os.path.join(FIG_ROOT, "Figure2")
FIG3_DIR = os.path.join(FIG_ROOT, "Figure3")
FIG4_DIR = os.path.join(FIG_ROOT, "Figure4")
FIGS2_DIR = os.path.join(FIG_ROOT, "FigureS2")
FIGS3_DIR = os.path.join(FIG_ROOT, "FigureS3")
FIGS4_DIR = os.path.join(FIG_ROOT, "FigureS4")
FIGS5_DIR = os.path.join(FIG_ROOT, "FigureS5")
EMP_BOOTSTRAP = os.path.join(DATA_DIR, "emp_bootstrap.parquet")

# Set True to recompute the empirical bootstrap (~20 minutes); False reads the
# shipped precomputed result data/processed/emp_bootstrap.parquet.
Run_Bootstrap = False

def main():
    for d in [TABLE_DIR, FIG2_DIR, FIG3_DIR, FIG4_DIR,FIGS2_DIR, FIGS3_DIR,FIGS4_DIR, FIGS5_DIR]:
        os.makedirs(d, exist_ok=True)
    df_all = pd.read_csv(os.path.join(DATA_DIR, "Tapp_9cities.csv"), parse_dates=["date"])

    ### Check for stationarity
    # df_adf = fAdf_test(df_all)
    # print(df_adf.head)

    ### Figure 2, left panel: a map of the city locations
    fPlotStation_info(city_info, save=True, folder=FIG2_DIR)

    ### Figure 2 (right panel) and Figure 3, for the main cities
    for city in City_main:
        fPlotTS(df_all, city, legend=True, save=True, folder=FIG2_DIR)
        fSelectdL(df_all, city, legend=True, save=True, folder=FIG3_DIR)

    ### Table 3: theta_hat
    fTable3(ndays, City, df_all, export=True, folder=TABLE_DIR)

    ### Empirical bootstrap: recompute (~20 min) or read the saved result.
    if Run_Bootstrap:
        boot_res = fBootstrap(City, df_all)
        boot_res.to_parquet(EMP_BOOTSTRAP)
    else:
        boot_res = pd.read_parquet(EMP_BOOTSTRAP)
        
    ### Figure S.2 & Figure S.4
    for city in City_supp:
        fPlotTS(df_all, city, legend=True, save=True, folder=FIGS2_DIR)
        fSelectdL(df_all, city, legend=True, save=True, folder=FIGS4_DIR)
    ### Figure S.3: the empirical condition D(u_n) check
    plot_dun(df_all, City, save=True, folder=FIGS3_DIR)
    

    ### Figure S.5: the confidence bands for theta for each city
    df_thetaci = fProcessTheta(boot_res)
    df_thetaci = df_thetaci[(df_thetaci['p_S'] == 97) & (df_thetaci['alpha'] == 0.1)]
    for city in City:
        plot_theta_city(df_thetaci, city, save=True, folder=FIGS5_DIR)

    ### Figure 4
    df_Sci = fProcessS(boot_res)
    df_Sci = df_Sci[(df_Sci['p_S'] == 97) & (df_Sci['alpha'] == 0.1)]
    fPlotFig4(df_thetaci, df_Sci, save=True, folder=FIG4_DIR)

    ### Table S.1
    fTableS1(boot_res, City, df_all, export=True, folder=TABLE_DIR)


if __name__ == "__main__":
    main()