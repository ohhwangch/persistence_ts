"""
00_build_dataset.py
===================
Regenerate data/processed/Tapp_9cities.csv  from the raw ERA5 reanalysis. 
This script is NOT part of the analysis: the three analysis scripts never call it, 
and it requires (a) a Copernicus CDS account/API key and (b) the cfgrib/eccodes stack. 
Files
-----
  data/raw/ERA5_hourly_t2m_d2m_JJA_1940-2025.grib   raw download
  data/processed/Tapp_9cities.csv                   cleaned output, 
Usage
-----
  python code/00_build_dataset.py
  Downloads the raw ERA5 GRIB into data/raw/ (needs a ~/.cdsapirc CDS key), then
  cleans it into data/processed/Tapp_9cities.csv.
ERA5 data citation
------------------
  Hersbach, H. et al. (2023): ERA5 hourly data on single levels from 1940 to
  present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
  DOI: 10.24381/cds.adbb2d47
"""
# ----------------------------------------------------------------------
# relative to this file, so the script works from any working directory.
#
#   <package root>/
#   ├── code/00_build_dataset.py   <- this file
#   └── data/
#       ├── raw/        <- raw ERA5 GRIB is downloaded here
#       └── processed/  <- cleaned Tapp_9cities.csv are written here
# ----------------------------------------------------------------------
import os
import pandas as pd
import ssl
import certifi
try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None  # required to clean the GRIB; install with: pip install xarray cfgrib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# Raw download target (descriptive name: variables, season, period).
RAW_GRIB = os.path.join(RAW_DIR, "ERA5_hourly_t2m_d2m_JJA_1940-2025.grib")
# Cleaned outputs.
PROCESSED_CSV = os.path.join(PROCESSED_DIR, "Tapp_9cities.csv")

# City information (lat, lon) 
City = ["London", "Paris", "Munich", "Budapest", "Milan", "Rome", "Barcelona", "Valencia", "Athens"]
city_info = {
    "Athens": (37.98, 23.72),
    "Barcelona": (41.39, 2.17),
    "Budapest": (47.50, 19.04),
    "London": (51.51, -0.13),
    "Milan": (45.46, 9.19),
    "Munich": (48.14, 11.58),
    "Paris": (48.86, 2.35),
    "Rome": (41.90, 12.50),
    "Valencia": (39.47, -0.38),
}
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)


def download_era5():
    """Download the raw ERA5 GRIB via the Copernicus CDS API (verbatim request)."""
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
    os.makedirs(RAW_DIR, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(RAW_GRIB)
    print(f"downloaded raw ERA5 GRIB -> {RAW_GRIB}")


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


def main(grib_path=RAW_GRIB):
    if xr is None:
        raise ModuleNotFoundError(
            "xarray (and cfgrib/eccodes) are required to build the dataset.\n"
            "Install with:  pip install xarray cfgrib  (plus the ecCodes system library)"
        )
    if not os.path.exists(grib_path):
        raise FileNotFoundError(
            f"GRIB file not found: {grib_path}\n"
            "Download it first with:  python code/00_build_dataset.py --download"
        )
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_9cities = fCleanData(grib_path, city_info) # takes about 3 minutes
    df_9cities.to_csv(PROCESSED_CSV, index=False)
    print(f"wrote {PROCESSED_CSV}")


if __name__ == "__main__":
    download_era5()
    main()