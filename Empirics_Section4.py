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
# Juan-Juan Cai, Yicong Lin, Julia Schaumburg and Chenhui Wang (2026).Estimation and inference for the persistence of extreme high temperatures.

# Purpose: applying our methods to the empirical case in Section 4, daily maximum apparent temperature in Europe.
######################################################


### Imports
import numpy as np
import xarray as xr
import pandas as pd
import os
from scipy.stats import genpareto
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from statsmodels.tsa.stattools import adfuller
from matplotlib.lines import Line2D

### ======= Download Data Using the API and Cleaning (if not needed, just ignore this part),=======
### =======     the output file is named 9a1c0d2016f3a9a4e17cf03ed0e94f37.grib              =======
import cdsapi
dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "2m_dewpoint_temperature",
        "2m_temperature"
    ],
    "year": [
        "1940", "1941", "1942",
        "1943", "1944", "1945",
        "1946", "1947", "1948",
        "1949", "1950", "1951",
        "1952", "1953", "1954",
        "1955", "1956", "1957",
        "1958", "1959", "1960",
        "1961", "1962", "1963",
        "1964", "1965", "1966",
        "1967", "1968", "1969",
        "1970", "1971", "1972",
        "1973", "1974", "1975",
        "1976", "1977", "1978",
        "1979", "1980", "1981",
        "1982", "1983", "1984",
        "1985", "1986", "1987",
        "1988", "1989", "1990",
        "1991", "1992", "1993",
        "1994", "1995", "1996",
        "1997", "1998", "1999",
        "2000", "2001", "2002",
        "2003", "2004", "2005",
        "2006", "2007", "2008",
        "2009", "2010", "2011",
        "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018", "2019", "2020",
        "2021", "2022", "2023",
        "2024", "2025"
    ],
    "month": ["06", "07", "08"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "11:00", "12:00", "13:00",
        "14:00", "15:00", "16:00",
        "17:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [52.5, -2, 37, 25]
}
client = cdsapi.Client()
client.retrieve(dataset, request).download()

def fCleanData(filename, city_info):
    '''
    Load GRIB file and do initial processing, get the time series of daily maximum apperant temperature for the selected cities.
    '''
    ds = xr.open_dataset( filename, engine="cfgrib", backend_kwargs={ "indexpath": "", "filter_by_keys": {"typeOfLevel": "surface"} })
    tair = ds['t2m'] - 273.15 # 2m air temperature
    tdew = ds['d2m'] - 273.15 # 2m dewpoint temperature
    # Hourly apparent temperature
    tapp = -2.653 + 0.994 * tair + 0.0153 * (tdew ** 2)
    # Daily Tappmax
    Tappmax = tapp.groupby(tapp["time"].dt.date).max("time")
    dates = pd.to_datetime(Tappmax['date'].values)
    Tappmax = Tappmax.assign_coords(date=("date", dates))
    df = Tappmax.to_dataframe(name="tappmax").reset_index()
    Tappmax_all = df[["date", "latitude", "longitude", "tappmax"]] # a large box containing grid data for all cities
    # Auxiliary functions
    def get_nearest_grid(df, lat, lon):
        """Return nearest ERA5 grid coordinate."""
        lat_unique = df["latitude"].unique()
        lon_unique = df["longitude"].unique()
        nearest_lat = lat_unique[(abs(lat_unique - lat)).argmin()]
        nearest_lon = lon_unique[(abs(lon_unique - lon)).argmin()]
        return float(nearest_lat), float(nearest_lon)
    def get_city_ts(df, city_name, lat, lon):
        """Return clean time series for one city."""
        g_lat, g_lon = get_nearest_grid(df, lat, lon)
        ts = df[(df["latitude"] == g_lat) &
                (df["longitude"] == g_lon)][["date", "tappmax"]].copy()
        ts = ts.sort_values("date")
        ts["city"] = city_name
        return ts[["date", "city", "tappmax"]]
    # Get time series for all 9 cities
    all_city_ts = [get_city_ts(Tappmax_all, city, lat, lon) for city, (lat, lon) in city_info.items()]
    df_9cities = pd.concat(all_city_ts, ignore_index=True)
    return df_9cities

filename = '9a1c0d2016f3a9a4e17cf03ed0e94f37.grib'
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
df_9cities = fCleanData(filename,city_info) # takes about 3 minutes, recommand open the dataset directly
df_9cities.to_parquet("Tapp_9cities.parquet", index=False)
### ==========================================================================================================

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
def fPlotStation_info(cities):
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
    plt.show()
def fPlotTS(df, city,legend = False, save=False):
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
        save_folder = "overview_timeseries_2T"
        os.makedirs(save_folder, exist_ok=True)
        path = os.path.join(save_folder, f"ts_{city}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {path}")
    plt.show()

### Figure 3 ###
def fSelectdL(df_all, city, d=np.arange(2, 21), k=np.arange(65, 258), save=False, legend=True):
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
        folder = "Tapp/Delta_two_periods"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/{city}_Delta_two_periods.png",
                    dpi=300, bbox_inches="tight")

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
def fPlotFig4(df_thetaci, df_Sci, save = False):
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
        plt.savefig(f"{folder}/CI_5percent97_tht10_S10.png", dpi=300, bbox_inches="tight")
    plt.show()

### Figure S.4 ###
def plot_theta_city(df_thetaci, city, legend =True, save =False):
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
        outdir = os.path.join("Tapp", "thetaplot_ci")
        os.makedirs(outdir, exist_ok=True)
        png_path = os.path.join(outdir, f"theta_ci_{city}.png")
        # pdf_path = os.path.join(outdir, f"theta_ci_{city}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        # fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.show()

###
### ======================== Tables ========================
###
### Table 3 ###
def fTable3(ndays,City,df_all, export=False):
    '''
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
        excel_path = os.path.join(folder, "Table3_theta.xlsx")
        df_tab3.to_excel(excel_path, index=False)
    return pd.DataFrame(results)
### Table S.1 ###
def fTableS1(boot_res,City, df_all, export = False):
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
        output_path = os.path.join(folder, "Count_S.xlsx")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_clean.to_excel(writer, index=False)
    return df_clean


###
### ======================== Main Functions ========================
###
### Read the cleaned dataset ###
df_all = pd.read_parquet("Tapp_9cities.parquet")
ndays = 35 * 92
folder = "Tapp"
os.makedirs(folder, exist_ok=True)

### Make plot: A map of locations (Figure 2, left panel)
fPlotStation_info(city_info)

### Make plot: Time series for each city, for example London (Figure 2, right panel)
fPlotTS(df_all, 'London', legend=True, save=False)

### Check for stationarity
df_adf = fAdf_test(df_all)
print(df_adf.head)

### Make plot: Figure for selection of d_L  (Figure 3)
fSelectdL(df_all, 'Paris', legend=True, save=False)

### Table 3: theta_hat
df_tab3 = fTable3(ndays,City,df_all)

### Implement bootstrap, takes 20 minutes. You can also turn into next block of code to read the result directly.
boot_res =  fBootstrap(City, df_all)
boot_res.to_parquet("emp_bootstrap.parquet")

### Read the bootstrap result directly.
boot_res = pd.read_parquet("emp_bootstrap.parquet")

### Make plot S.4: the confidence bands for theta for each city
df_thetaci = fProcessTheta(boot_res)
df_thetaci = df_thetaci[(df_thetaci['p_S']==97) & (df_thetaci['alpha']==0.1)] # p_S does not matter here, choose random one, 90% level of confidence interval, aloha =0.1
plot_theta_city(df_thetaci, 'London', save = False)

### Make Figure 4
df_Sci =  fProcessS( boot_res)
df_Sci = df_Sci[(df_Sci['p_S']==97) & (df_Sci['alpha']==0.1)]
fPlotFig4(df_thetaci, df_Sci, save = False)

### Obtain table S.1
df_TabS1 = fTableS1(boot_res,City, df_all)
