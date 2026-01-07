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

# Purpose: Simulation studies in Section 3.
######################################################


### Imports
import os
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import seaborn as sns

###
### ================ DGP: Models ================
###
### model 1: max-autoregressive with theta=tht, d=2
def MAAR(n, z):
    xi = -1 / np.log(np.random.uniform(size=n))
    result = np.zeros(n)
    result[0] = xi[0] / (1 - z)
    for i in range(1, n):
        result[i] = max(z * result[i - 1], xi[i])
    return result

### model 2: moving-max with theta=1/z, d=2
def MM(n, z):
    X = np.zeros(n)
    U = np.random.uniform(size=n + z - 1)
    Xi = -1 / (z * np.log(U))
    for i in range(n):
        X[i] = np.max(Xi[i:i + z])
    return X

### model 3 #AR(1) with cauchy margin, X_i=a*X_{i-1}+x_i, where x_i cauchy distributed;
### For z<0, theta=1-z^2 and d=3; For z >0, theta=1-z, d=2
def ARC(n, z):
    X = np.zeros(n)
    X[0] = cauchy.rvs(loc=0, scale=1)
    for i in range(1, n):
        scale = 1 - np.abs(z)
        xi = cauchy.rvs(loc=0, scale=scale)
        X[i] = z * X[i - 1] + xi
    return X

### model 4:  Arch model  X_i=(2*10^(-5)+zX_{i-1}^2)^(1/2)e_i, where e_i iid N(0, 1).
### z=(0.1,0.5,0.7,0.99); theta=(0.999,0.835,0.721,0.571).
def arch(n, z, burnin=1000):
    iidn = np.random.normal(size=n + burnin)
    data = np.zeros(n + burnin)
    data[0] = iidn[0]
    for i in range(1, n + burnin):
        data[i] = np.sqrt(2 * 10 ** -5 + z * data[i - 1] ** 2) * iidn[i]
    return data[burnin:]

### model 5: AR(1) model with Gaussian innovation
def ar1(n, z):
    burnin = 100
    w = np.random.normal(size=n + burnin)
    result = np.zeros(n + burnin)
    for i in range(1, len(result)):
        result[i] = z * result[i - 1] + w[i]
    return result[burnin:]

### model 6: Heavy tailed data based on ARC
def ARC_GPD(n, z, gamma=-0.3, x_F=50, mu=20):
    # Step 1: generate ARC(n, z)
    Y = ARC(n, z)
    # Step 2: transform ARC to Uniform(0,1)
    U = 0.5 + np.arctan(Y) / np.pi
    # Step 3: GP tail transformation (heavy-tailed)
    X = x_F + (mu - x_F) * (1 - U)**(-gamma)
    return X

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

def theta_sliding(data, b):
    '''
    Pseudo MLE by Berghaus & Bucher (2018) — Sliding block
    Parameters
    ----------
    data : array-like, shape (n,)
        Univariate time series (or sample) used to estimate the extremal index.
    b : array-like
        Block-length parameter(s).

    Returns
    -------
    result : ndarray, shape (len(b),)
        Sliding-blocks extremal index estimates for each entry in `b`.
    '''
    data = np.asarray(data)
    b = np.asarray(b, dtype=float)
    n = data.size
    pseudo = rankdata(data, method="average") / (n + 1.0)
    result = np.empty(b.size, dtype=float)
    for j, l0 in enumerate(b):
        l0 = float(l0)  # original (possibly non-integer) l
        L = int(l0)  # block length used for indexing
        nb1 = int(n - l0 + 1)

        if L < 1 or nb1 <= 0:
            result[j] = np.nan
            continue
        # each row = one sliding block, length L
        slide_pseudo = np.array([pseudo[i:(i + L)] for i in range(nb1)])
        max_slide = slide_pseudo.max(axis=1)
        zni = (1.0 - max_slide) * l0
        tn_hat = zni.mean()
        result[j] = 1.0 / tn_hat
    return result

def theta_int(data, k):
    '''
   Estimate the extremal index using the Ferro & Segers (2003) interval estimator. 

    Parameters
    ----------
    data : array-like, shape (n,)
        Univariate time series (or sample).

    k : array-like of int
        Indices selecting the threshold level via order statistics.
    Returns
    -------
    result : ndarray, shape (len(k),)
        Extremal index estimates for each `ki`. 
    '''
    Xs = np.sort(data)[::-1]
    result = np.zeros(len(k))
    for i, ki in enumerate(k):
        S = np.where(data > Xs[ki])[0]
        if len(S) < 2:
            result[i] = np.nan
            continue
        T = np.diff(S)
        if np.max(T) <= 2:
            tht = 2 * (np.sum(T - 1)) ** 2 / ((len(T)) * np.sum(T ** 2))
        else:
            tht = 2 * (np.sum(T - 1)) ** 2 / ((len(T)) * np.sum((T - 1) * (T - 2)))
        result[i] = min(1, tht)
    return result

models = {
    1: MAAR,
    2: MM,
    3: ARC,
    4: arch,
    5: ar1,
    6: ARC_GPD
}

def GetVerify(n=5000, k=(50, 100), dist=1, m=100, z=0.5, du=10, d0=1, c=1):
    '''
    The verification of Condition $D^{(d)}(u_n)$ used in the paper.

    For each replication, the function simulates a time series from `models[dist]`, computes the sequence of extremal-index estimates `Theta_ck` over a grid of
    parameters `d`, and checks whether the maximum discrete increment stays below a threshold of order 1/sqrt(k).

    Parameters
    ----------
    n : int, default=5000
        Sample size per Monte Carlo replication.

    k : sequence of int, default=(50, 100)
        Threshold indices used in `Theta_ck` (upper-tail order statistic indices).
    dist : int, default=1
        Key selecting the data-generating process from the global dictionary `models`.
    m : int, default=100
        Number of Monte Carlo replications.
    z : float, default=0.5
        Model-specific parameter passed to `models[dist](n, z)` .
    du : int, default=10
        Maximum run parameter value (inclusive) for the grid `d`.

    d0 : int, default=1
        Starting value logic for the increment construction.
    c : float, default=1
        Scaling constant in the acceptance threshold.

    Returns
    -------
    out : dict
        Dictionary with one key:
        - "testds": ndarray of shape (m, len(k))
            Binary acceptance indicators.
    '''
    d = np.arange(max(2, d0), du + 1)
    Acpt1 = np.zeros((m, len(k)))
    for i in range(m):
        data = models[dist](n, z)
        for j, kk in enumerate(k):
            thrd = 1 / np.sqrt(kk)
            thtck = Theta_ck(data=data, d=d, k=[kk])
            if d0 == 1:
                deltak = np.column_stack((1 - thtck[:, 0], thtck[:, :-1] - thtck[:, 1:]))
            else:
                deltak = thtck[:, :-1] - thtck[:, 1:]
            if np.max(deltak) < thrd / c:
                Acpt1[i, j] = 1
    return {"testds": Acpt1}
### estimation and obtain MSE
def main_tht(n=5000, k=None, b=None, dist=1, m=1000, z=0.5, du=10, theta=1, plot=False):
    """
    Performs estimation and MSE comparison
    """
    if k is None:
        k = np.arange(50, 101)
    if b is None:
        b = n / k

    d = np.arange(2, du + 1)
    nd = len(d)

    de = np.full((m, len(k)), du)
    tht_ck = np.zeros((m, len(k)))
    tht_int = np.zeros((m, len(k)))
    tht_sl = np.zeros((m, len(b)))

    for i in range(m):
        data = models[dist](n, z)

        tht_int[i, :] = theta_int(data=data, k=k)
        tht_sl[i, :] = theta_sliding(data=data, b=b)
        for j, kk in enumerate(k):
            thrd = 1 / np.sqrt(kk)
            thtck = Theta_ck(data=data, d=d, k=kk)
            part1 = 1 - thtck[:, 0]
            part2 = thtck[:, :-1] - thtck[:, 1:]
            deltak = np.concatenate((part1, part2.ravel(order='F')))
            for l in range(nd - 1):
                if np.max(deltak[l:nd]) < thrd:
                    de[i, j] = l + 1
                    break
            if de[i, j] == 1:
                tht_ck[i, j] = 1
            elif de[i, j] >= 2:
                tht_ck[i, j] = Theta_ck(data=data, d=de[i, j].item(), k=kk)
    mse_ck = np.mean((tht_ck - theta) ** 2, axis=0)
    mse_int = np.mean((tht_int - theta) ** 2, axis=0)
    mse_sl_b = np.mean((tht_sl - theta) ** 2, axis=0)

    mse_k = np.column_stack((mse_ck, mse_int))
    mse_b = np.column_stack((mse_sl_b,))

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(b, mse_ck, label=r'$\hat{\theta}_n(\hat{d^*})$')
        plt.plot(b, mse_int, label=r'$\hat{\theta}_n^{int}$')
        plt.plot(b, mse_sl_b, label=r'$\hat{\theta}_n^{B,sl}$')
        plt.xlabel("b")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()

    return {
        "ck": tht_ck,
        "int": tht_int,
        "sliding": tht_sl,
        "mse_k": mse_k,
        "mse_b": mse_b,
        "dstr": de
    }

###
### ================ Plots ================
###
def fplotMSE(k, M, title, legend = False, save = False):
    palette = sns.color_palette("Dark2", n_colors=3)
    legends = [
        r"$\hat{\theta}_n(\hat{d}_L)$",
        r"$\hat{\theta}_n^{\mathrm{int}}$",
        r"$\hat{\theta}_n^{\mathrm{B,sl}}$"
    ]
    plt.figure(figsize=(7, 5))
    # --- Plot each estimator ---
    plt.plot(k, M[:, 0], color=palette[0],linewidth=3, label=legends[0])
    plt.plot(k, M[:, 1], color=palette[1], linewidth=3,linestyle ='dashed', label=legends[1])
    plt.plot(k, M[:, 2], color=palette[2], linewidth=3,linestyle = 'dotted' ,label=legends[2])
    # --- Labels and Title ---
    plt.xlabel(r"$k$", fontsize=20)
    plt.ylabel( "MSE", fontsize=20)
    plt.title(f"{title}", fontsize=26)
    plt.locator_params(axis="y", nbins=5)
    # --- Ticks ---
    plt.xticks(fontsize=20)
    if title == 'IID' or title == 'AR-N' or title == 'ARCH':
        plt.ylim(0,0.1)
    else:
        plt.ylim(0,0.02)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    if legend == True:
        plt.legend(fontsize=26, loc="upper left", frameon=False)
    plt.tight_layout()
    if save:
        os.makedirs("MSE_plots", exist_ok=True)
        plt.savefig(f"MSE_plots/{title}.png", dpi=300)
    plt.show()

###
### ================ Main Functions ================
###
def main():
    # Verification
    np.random.seed(12345)
    ### iid 
    for d0 in [1,2,3]:
        result_ARN = GetVerify(dist=5, z=0, m=1000, du=10, d0=d0)
        print(np.sum(result_ARN["testds"], axis=0))
    print('======IID======')
    ### AR-N
    for d0 in [1,2,3]:
        result_ARN = GetVerify(dist=5, z=0.5, m=1000, du=10, d0=d0)
        print(np.sum(result_ARN["testds"], axis=0))
    print('======ARN======')
    ### Moving maximum
    for d0 in [1,2,3]:
        result_MM = GetVerify(dist=2, z=3, m=1000, du=10, d0=d0)
        print(np.sum(result_MM["testds"], axis=0))
    print('======MM======')
    ### MAX-AR
    for d0 in [1,2,3]:
        result_maar = GetVerify(dist=1, m=1000, z=0.5, du=10, d0=d0)
        print(np.sum(result_maar["testds"], axis=0))
    print('======MAX-AR======')
    ### AR-C
    for d0 in [1,2,3]:
        result_ARC = GetVerify(dist=3, z=-0.5, m=1000, du=10, d0=d0)
        print(np.sum(result_ARC["testds"], axis=0))
        
    print('======AR-C======')
    ### transformed ARC
    for d0 in [1,2,3]:
        result_Transformed_ARC = GetVerify(dist=6, z=-0.5, m=1000, du=10, d0=d0)
        print(np.sum(result_Transformed_ARC["testds"], axis=0))
    print('======Transformed-AR-C======')
    ### ARCH
    for d0 in [1,2,3]:
        result_ARCH = GetVerify(dist=4, z=0.7, m=1000, du=10, d0=d0)
        print(np.sum(result_ARCH["testds"], axis=0))
    print('======ARCH======')   
    ### ARCH (n = 50000, k/n = 0.05, 0.1)
    np.random.seed(12345)
    for d0 in np.arange(1,10 ):
        n = 50000
        k = (0.05*n, 0.10*n)
        result_ARCH = GetVerify(n=n,k=k,dist=4, z=0.7, m=1000, du=10, d0=d0)
        print(np.sum(result_ARCH["testds"], axis=0))
    print('======ARCH(large n, large n/k)======')  
    ## Get results for MSE  and make plot: 
    n = 5000
    k = np.arange(30, 301, 4)
    b = n / k
    result_IID = main_tht(n=n, k=k, b=n / k, dist=5, z=0, theta=1)
    print('IID complete')
    result_ARN = main_tht(n=n, k=k, b=n / k, dist=5, z=0.5, theta=1)
    print('ARN complete')
    result_MM = main_tht(n=n, k=k, b=b, dist=2, z=3, theta=1 / 3)
    print('MM complete')
    result_maar = main_tht(n=n, k=k, b=b, dist=1, z=0.5, theta=0.5)
    print('MAAR complete')
    result_ARC = main_tht(n=n, k=k, b=b, dist=3, z=-0.5, theta=0.75)
    print('ARC complete')
    result_ARCH = main_tht(n=n, k=k, b=b, dist=4, z=0.7, theta=0.721)
    print('ARCH complete')
    result_Transformed_ARC = main_tht(n=n, k=k, b=b, dist=6, z=-0.5, theta=0.75)
    print('Transformed_ARC complete')
    
    M1 = np.column_stack((result_IID["mse_k"][:, :2], result_IID["mse_b"][:, 0]))
    M2 = np.column_stack((result_ARN["mse_k"][:, :2], result_ARN["mse_b"][:, 0]))
    M3 = np.column_stack((result_MM["mse_k"][:, :2], result_MM["mse_b"][:, 0]))
    M4 = np.column_stack((result_maar["mse_k"][:, :2], result_maar["mse_b"][:, 0]))
    M5 = np.column_stack((result_ARC["mse_k"][:, :2], result_ARC["mse_b"][:, 0]))
    M6 = np.column_stack((result_ARCH["mse_k"][:, :2], result_ARCH["mse_b"][:, 0]))
    M7 = np.column_stack((result_Transformed_ARC["mse_k"][:, :2], result_Transformed_ARC["mse_b"][:, 0]))
    
    fplotMSE(k, M1,'IID',legend = True)  
    fplotMSE(k, M2,'AR-N')
    fplotMSE(k, M3,'Moving Maxima')
    fplotMSE(k, M4,'Max−AR')
    fplotMSE(k, M5,'AR−C')
    fplotMSE(k, M6, 'ARCH')
    fplotMSE(k, M7,'Transformed AR-C')
main()
