

# Estimation and inference for the persistence of extremely high temperatures: README for the Replication Package
<!-- badges: start -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges: end -->

Authors: Juan-Juan Cai (j.cai@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute) , Yicong Lin (yc.lin@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute), Julia Schaumburg (j.schaumburg@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute) and Chenhui Wang (c.h.wang@vu.nl, Vrije Universiteit Amsterdam)

## Overview
This replication package contains Python code and data for reproducing the main simulation results, the empirical application, and the online supplement for the paper. The package is organized as follows:


The main manuscript results are reproduced by the following scripts:

- `code/1_simulation.py`, which produces Table 2 and Figure 1;
- `code/2_empirics.py`, which produces Figure 2, Figure 3, Table 3 and Figure 4;

The online supplement results are reproduced by the following scripts:
- `code/3_simulation_supp.py`, which produces Figure S.1
- `code/2_empirics.py`, which produces Figures S.2-5, and Table S.1

The Python packages required to run the code are listed in `requirements.txt`, supporting Python functions are stored in `code/block_bootstrap.py`.

## Data Availability and Provenance Statements

This replication package contains both original simulation code written for this paper and raw data downloaded from Climate Data Store. Simulation results  are based on simulated data generated within the submitted Python scripts. The raw data in Section 4 are obtained from Climate Data Store: 
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview. It requires a Copernicus CDS account. Code for data cleaning and analysis is provided in `code/00_build_dataset.py`  as part of the replication package.

The empirical application uses the cleaned dataset `data/processed/Tapp_9cities.csv`.

### Statement about Rights
- [x] We certify that the authors of the manuscript have legitimate access to and permission to use the data used in this manuscript. 

#### Summary of Data Availability
- [x] All data **are** publicly available.

### Details on each Data Source

| Data.Name  | Data.Files | Location | Provided | Citation |
| -- | -- | -- | -- | -- | 
| ERA5 Reanalysis Single Levels (raw data) | Raw ERA5 GRIB files downloaded via CDS API | Copernicus Climate Data Store | No       | Hersbach et al. (2020)                     |
| Cleaned analysis dataset                 | `Tapp_9cities.csv`                     | `data/processed`                       | Yes      | Derived from ERA5 |


## Computational requirements


### Software Requirements
- Python 3.11.5
  - numpy, scipy, pandas, matplotlib, seaborn, statsmodels, joblib, cartopy
  - `requirements.txt` lists these dependencies.

### Controlled Randomness
Random seed is set at line 354 of `1_simulation.py`, line 266 of `2_empirics.py`
and line 484 of `3_simulation_supp.py`.


### Memory, Runtime, Storage Requirements

#### Summary time to reproduce

Approximate time needed to reproduce the analyses on a standard (CURRENT YEAR) desktop machine:

- [ ] 10-60 minutes
- [ ] 1-2 hours
- [ ] 2-8 hours
- [ ] 8-24 hours
- [ ] 1-3 days
- [x] 3-14 days
- [ ] more than 14 days

#### Summary of required storage space

Approximate storage space needed:

- [ ] < 25 MBytes
- [ ] 25 MB - 250 MB
- [x] 250 MB - 2 GB
- [ ] 2 GB - 25 GB
- [ ] 25 GB - 250 GB

#### Computational Details
Routine data processing, analysis were conducted on a MacBook Air (Apple M2, 8 GB RAM).

The most demanding simulations were executed on the Dutch national supercomputing infrastructure, Snellius, using a single Genoa compute node with 192 CPU cores. 

## Description of programs/code

```
replication-package/
|-- README.md
|-- LICENSE.txt                GPL-3.0 license.
|-- requirements.txt           Pinned Python dependencies.
|-- code/
|   |-- 00_build_dataset.py    OPTIONAL: download raw ERA5 + clean it (not part of the analysis).
|   |-- 1_simulation.py        Produce all simulation results in Section 3.
|   |-- 2_empirics.py          Produce Figures 2-4, S.2-5; Tables 3, S.1.
|   |-- 3_simulation_supp.py   Produce Figure S.1.
|   `-- block_bootstrap.py     Shared block-bootstrap resampling primitives.
|-- data/
|   |-- raw/                   (empty) raw ERA5 GRIB goes here if rebuilding from source.
|   `-- processed/
|       |-- Tapp_9cities.csv
|       |   Cleaned dataset used for the empirical analysis.
|       |-- emp_bootstrap.parquet
|       |   Precomputed bootstrap results for the empirical analysis.
|       `-- ARC_S_n15000_B199_MC1000_a0.05.csv
|           Precomputed bootstrap results for the supplementary simulation analysis.
`-- output/
    |-- figures/               Generated figures (created by the run).
    `-- tables/                Generated tables (created by the run).
```


## Instructions to Replicators

- Open Python in the project root directory.
- Run **pip install -r requirements.txt** as the first step. 
- Run `code/1_simulation.py` to generate results in Section 3 in the main manuscript (Around 40 minutes).
- **Optional**: Run `code/00_build_dataset.py` to download the raw data (1.8GB, need to register first).
- Run `code/2_empirics.py` to generate results in Section 4 and in the online supplement using `data/processed/Tapp_9cities.csv` referenced above. 
  - Run_Bootstrap = **FALSE**, the precomputed results `data/processed/emp_bootstrap.parquet` are used. Set Run_Bootstrap=TRUE to rerun, which takes about **40 minutes**.
- Run `code/3_bootstrap.py` for Figure S.1.
  - RERUN = **False**, use precomputed result `data/processed/ARC_S_n15000_B199_MC1000_a0.05.csv` indefault. Set RERUN = True to rerun, which takes around **8 hours** on Snellius using a single Genoa compute node with 192 CPU cores.

## List of tables and programs
The provided code reproduces:
- [x] All tables and figures in the paper

| Figure/Table  | Program                   | Output file           | Note                       |
| -------------- | ------------------------- | --------------------- | -------------------------- |
| Table 2        | code/1_simulation.py    | output/tables/Table2.csv      | Around 10 minutes |
| Figure 1        | code/1_simulation.py    | output/figures/Figure1      | Around 40 minutes | 
| Figure 2        | code/2_empirics.py    | output/figures/Figure2     | 
| Figure 3        | code/2_empirics.py    | output/figures/Figure3     | 
| Table 3        | code/2_empirics.py    | output/tables/Table3.csv     |   
| Figure 4        | code/2_empirics.py    | output/figures/Figure4     | Bootstrap needed (40 minutes when Run_Bootstrap=TRUE)   |                 
| Figure S.1       | code/3_simulation_supp.py      | output/figures/FigureS1 | Long running time when RERUN = TRUE|
| Figure S.2       | code/2_empirics.py | output/figures/FigureS2 |  |
| Figure S.3       | code/2_empirics.py | output/figures/FigureS3 |  |
| Table S.1       | code/2_empirics.py | output/tables/TableS1.csv |  |
| Figure S.4       | code/2_empirics.py | output/figures/FigureS4 |  |
| Figure S.5       | code/2_empirics.py | output/figures/FigureS5 |  |


## References

Copernicus Climate Change Service (C3S). 2017. ERA5: Fifth generation of ECMWF atmospheric reanalysis of the global climate. Climate Data Store (CDS), European Centre for Medium-Range Weather Forecasts (ECMWF). Available at: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

Hersbach H, Bell B, Berrisford P, et al. The ERA5 global reanalysis. Q J R Meteorol Soc. 2020;146:1999–2049. https://doi.org/10.1002/qj.3803

---

## Acknowledgements

Some content on this page was copied or adapted from the template at https://social-science-data-editors.github.io/template_README/.
