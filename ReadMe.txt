====================================================================================================================================

This is the ReadMe file for the codes and datasets to replicate the results presented in the paper:

'Estimation and inference for the persistence of extreme high temperatures'

Authors: Juan-Juan Cai, Yicong Lin, Julia Schaumburg and Chenhui Wang
=====================================================================================================================================



The code requires several Python packages: numpy, pandas, joblib, datetime, os, matplotlib, seaborn, 
					   scipy, xarray, cartopy, statsmodels.

The library contains the following files:

- Simulation_MainSection3.py: Python script used to reproduce the Monte Carlo study in the Section 3 of the paper. 


- Empirics_Section4.py: Python script used to reproduce the empirical analysis presented in Section 4 of the paper. The script 
			includes an API for downloading the complete dataset (see lines 33–149). However, downloading the full 
			dataset is not recommended, as it is time-consuming and results in a very large file that requires 
			substantial storage space. For convenience, we recommend using the pre-processed and cleaned subset 
			"Tapp_9cities.parquet", which is provided with this repository.


- Simulation_SupplementS3.py: Python script used to reproduce the Monte Carlo study in the Section S3 of the online supplement. 
	


=====================================================================================================================================

This version: January 7, 2026


Copyright: Juan-Juan Cai, Yicong Lin, Julia Schaumburg and Chenhui Wang

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied 
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

======================================================================================================================================
















