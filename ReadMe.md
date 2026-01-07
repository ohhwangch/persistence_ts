# Codes and datasets to replicate the results presented in the paper: Estimation and inference for the persistence of extremely high temperatures
<!-- badges: start -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges: end -->

Authors: Juan-Juan Cai (j.cai@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute) , Yicong Lin (yc.lin@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute), Julia Schaumburg (j.schaumburg@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute) and Chenhui Wang (c.h.wang@vu.nl, Vrije Universiteit Amsterdam)

## Python packages required: 

numpy, pandas, joblib, datetime, os, matplotlib, seaborn, scipy, xarray, cartopy, statsmodels.


## Files contained:
- Simulation_MainSection3.py: Python script used to reproduce the Monte Carlo study in the Section 3 of the paper. 
- Empirics_Section4.py: Python script used to reproduce the empirical analysis presented in Section 4 of the paper. 
The script includes an API for downloading the complete dataset (see lines 33–149). However, downloading the full 
dataset is not recommended, as it is time-consuming and results in a very large file that requires substantial storage space. For convenience, we recommend using the pre-processed and cleaned subset "Tapp_9cities.parquet", which is provided with this repository.

- Simulation_SupplementS3.py: Python script used to reproduce the Monte Carlo study in the Section S3 of the online supplement. 
    

This version: January 7, 2026


Copyright: Juan-Juan Cai, Yicong Lin, Julia Schaumburg and Chenhui Wang


For any questions or feedback, please feel free to contact: c.h.wang@vu.nl
