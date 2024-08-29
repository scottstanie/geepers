"""gps.py
Utilities for integrating GPS with InSAR maps

Links:

1. list of LLH for all gps stations: ftp://gneiss.nbmg.unr.edu/rapids/llh
Note: ^^ This file is stored in the `STATION_LLH_FILE`

2. Clickable names to explore: http://geodesy.unr.edu/NGLStationPages/GlobalStationList
3. Map of stations: http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html

"""

from __future__ import annotations

import datetime
import difflib  # For station name misspelling checks
import logging
import os
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests

from .utils import get_cache_dir

PathOrStr = Path | str

logger = logging.getLogger(__name__)

# URL for ascii file of 24-hour final GPS solutions in east-north-vertical (NA12)
GPS_BASE_URL = "http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{station}.tenv3"
GPS_FILE_TEMPLATE = GPS_BASE_URL.split("/")[-1]
# NOTE: if i also want IGS14 and plate, I need to divide directories and do more
# The main web page per station
GPS_PLATE_BASE_URL = (
    "http://geodesy.unr.edu/gps_timeseries/tenv3/plates/{plate}/{station}.{plate}.tenv3"
)
GPS_PLATE_FILE_TEMPLATE = GPS_BASE_URL.split("/")[-1].replace(".{plate}", "")
# We'll use this for now to scrape the plate information with a regex :(
GPS_STATION_URL = "http://geodesy.unr.edu/NGLStationPages/stations/{station}.sta"

GPS_XYZ_BASE_URL = "http://geodesy.unr.edu/gps_timeseries/txyz/IGS14/{station}.txyz2"
GPS_XYZ_FILE = GPS_XYZ_BASE_URL.split("/")[-1]

GPS_DIR = get_cache_dir(force_posix=True)
GPS_DIR.mkdir(exist_ok=True, parents=True)

# These lists get update occasionally... to keep fresh, download one for current day
# old ones will be removed upon new download
STATION_LLH_URL = "http://geodesy.unr.edu/NGLStationPages/llh.out"
STATION_LLH_FILE = str(GPS_DIR / "station_llh_all_{today}.csv")

STATION_XYZ_URL = "http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
STATION_XYZ_FILE = str(GPS_DIR / "station_xyz_all_{today}.csv")


def load_station_enu(
    station_name,
    start_date=None,
    end_date=None,
    download_if_missing=True,
    zero_by="mean",
    to_cm=True,
):
    """Loads one gps station's ENU data since start_date until end_date as a dataframe

    Args:
        station_name (str): 4 Letter name of GPS station
            See http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html for map
        start_date (datetime or str): Optional. cutoff for beginning of GPS data
        end_date (datetime or str): Optional. cut off for end of GPS data
        download_if_missing (bool): default True
    """
    if end_date is None:
        end_date = pd.to_datetime(pd.Timestamp.today())
    else:
        end_date = pd.to_datetime(end_date)

    if zero_by not in ("start", "mean"):
        raise ValueError("'zero_by' must be either 'start' or 'mean'")
    station_name = station_name.upper()
    gps_data_file = os.path.join(
        GPS_DIR, GPS_FILE_TEMPLATE.format(station=station_name)
    )
    if not os.path.exists(gps_data_file):
        if download_if_missing:
            logger.info(f"Downloading {station_name} to {gps_data_file}")
            download_station_data(station_name, coords="enu")
        else:
            raise ValueError(
                "{gps_data_file} does not exist, download_if_missing = False"
            )

    df = pd.read_csv(gps_data_file, header=0, sep=r"\s+", engine="c")
    clean_df = _clean_gps_df(df, start_date, end_date)

    # Check that we have up to date data
    if (clean_df["date"].max() < end_date) and download_if_missing:
        download_station_data(station_name)
        df = pd.read_csv(gps_data_file, header=0, sep=r"\s+", engine="c")
        clean_df = _clean_gps_df(df, start_date, end_date)
        # os.remove(gps_data_file)
        # logger.info(f"force removed {gps_data_file}")

    # multiplier = 100  to_cm:

    if to_cm:
        # logger.info("Converting %s GPS to cm" % station_name)
        clean_df[["east", "north", "up"]] = 100 * clean_df[["east", "north", "up"]]

    if zero_by.lower() == "mean":
        mean_val = clean_df[["east", "north", "up"]].mean()
        # enu_zeroed = clean_df[["east", "north", "up"]] - mean_val
        clean_df[["east", "north", "up"]] -= mean_val
    elif zero_by.lower() == "start":
        start_val = clean_df[["east", "north", "up"]].iloc[:10].mean()
        # enu_zeroed = clean_df[["east", "north", "up"]] - start_val
        clean_df[["east", "north", "up"]] -= start_val
    # Finally, make the 'date' column a DateIndex
    return clean_df.set_index("date")


@cache
def read_station_llas(filename=None, to_geodataframe=False):
    """Read in the name, lat, lon, alt list of gps stations

    Assumes file is a space-separated with "name,lat,lon,alt" as columns
    """
    today = datetime.date.today().strftime("%Y%m%d")
    filename = filename or STATION_LLH_FILE.format(today=today)

    lla_path = os.path.join(GPS_DIR, filename)
    _remove_old_lists(lla_path)
    logger.debug("Searching %s for gps data" % filename)

    try:
        df = pd.read_csv(lla_path, sep=r"\s+", engine="c", header=None)
    except FileNotFoundError:
        logger.info("Downloading from %s to %s", STATION_LLH_URL, lla_path)
        download_station_locations(lla_path, STATION_LLH_URL)
        df = pd.read_csv(lla_path, sep=r"\s+", engine="c", header=None)

    df.columns = ["name", "lat", "lon", "alt"]
    # Make sure the longitude is wrapped between -180 and 180
    # It comes in the range (-360, 0)
    df.loc[:, "lon"] = df.lon - (np.round(df.lon / (360)) * 360)
    if to_geodataframe:
        import geopandas as gpd

        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    else:
        return df


def _clean_gps_df(df, start_date=None, end_date=None, coords="enu"):
    """Reorganize the Nevada GPS data format"""
    df["date"] = pd.to_datetime(df["YYMMMDD"], format="%y%b%d")

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    if coords == "enu":
        df_out = df[["date", "__east(m)", "_north(m)", "____up(m)"]]
    elif coords == "xyz":
        df_out = df[["date", "x", "y", "z"]]
    else:
        raise ValueError(f"{coords} must be 'enu' or 'xyz'")

    df_out = df_out.rename(
        mapper=lambda s: s.replace("_", "").replace("(m)", ""), axis="columns"
    )
    df_out.reset_index(inplace=True, drop=True)
    return df_out


def load_station_xyz(
    station_name, start_date=None, end_date=None, download_if_missing=True
):
    station_name = station_name.upper()
    gps_data_file = os.path.join(GPS_DIR, GPS_XYZ_FILE.format(station=station_name))
    if not os.path.exists(gps_data_file):
        if download_if_missing:
            logger.info(f"Downloading {station_name} to {gps_data_file}")
            download_station_data(station_name, coords="xyz")
        else:
            raise ValueError(
                "{gps_data_file} does not exist, download_if_missing = False"
            )

    if end_date is None:
        end_date = pd.to_datetime(pd.Timestamp.today()) - pd.Timedelta(days=1)
    else:
        end_date = pd.to_datetime(end_date)

    # http://geodesy.unr.edu/gps_timeseries/README_txyz2.txt
    columns = [
        "station",
        "YYMMMDD",
        "decimal_year",
        "x",
        "y",
        "z",
        "sigma_x",
        "sigma_y",
        "sigma_z",
        "sigma_xy",
        "sigma_xz",
        "sigma_yz",
        "antenna height",
    ]
    df = pd.read_csv(gps_data_file, header=None, sep=" ", engine="c", names=columns)

    clean_df = _clean_gps_df(df, start_date, end_date, coords="xyz")

    # Check that we have up to date data
    if (clean_df["date"].max() < end_date) and download_if_missing:
        download_station_data(station_name)
        df = pd.read_csv(gps_data_file, header=None, sep=" ", engine="c", names=columns)
        clean_df = _clean_gps_df(df, start_date, end_date, coords="xyz")
        # os.remove(gps_data_file)
        # logger.info(f"force removed {gps_data_file}")
    return clean_df.set_index("date")


def _remove_old_lists(lla_path):
    today = datetime.date.today().strftime("%Y%m%d")
    gps_dir = Path(lla_path).parent
    station_list_files = sorted(gps_dir.glob("station_*"))
    files_to_delete = [f for f in station_list_files if today not in str(f)]
    for f in files_to_delete:
        logger.info("Removing old station list file: %s", f)
        f.unlink()


@cache
def read_station_xyzs(filename=None):
    """Read in the name, X, Y, Z position of gps stations."""
    today = datetime.date.today().strftime("%Y%m%d")
    filename = filename or STATION_XYZ_FILE.format(today=today)
    _remove_old_lists(filename)
    logger.debug("Searching %s for gps data" % filename)
    try:
        df = pd.read_csv(
            filename,
            sep=r"\s+",
            engine="c",
            warn_bad_lines=True,
            error_bad_lines=False,
        )
    except FileNotFoundError:
        logger.warning("%s not found; downloading from %s", filename, STATION_XYZ_URL)
        download_station_locations(filename, STATION_XYZ_URL)
        df = pd.read_csv(
            filename,
            sep=r"\s+",
            engine="c",
            warn_bad_lines=True,
            error_bad_lines=False,
        )
    orig_cols = "Sta Lat(deg) Long(deg) Hgt(m) X(m) Y(m) Z(m) Dtbeg Dtend Dtmod NumSol StaOrigName"
    new_cols = "name lat lon alt X Y Z dtbeg dtend dtmod numsol origname"
    mapping = dict(zip(orig_cols.split(), new_cols.split()))
    return df.rename(columns=mapping)


def download_station_locations(filename: PathOrStr, url: str):
    """Download either station LLH file or XYZ file from Nevada website
    url = [STATION_XYZ_URL or STATION_LLH_URL]
    """
    resp = requests.get(url)
    resp.raise_for_status()

    with open(filename, "w") as f:
        f.write(resp.text)


def download_station_data(
    station_name: str, coords: Literal["enu", "xyz"] = "enu"
) -> None:
    station_name = station_name.upper()
    if coords == "enu":
        url = GPS_BASE_URL.format(station=station_name)
    elif coords == "xyz":
        url = GPS_XYZ_BASE_URL.format(station=station_name)
        filename = os.path.join(GPS_DIR, GPS_XYZ_FILE.format(station=station_name))
    response = requests.get(url)
    response.raise_for_status()

    logger.info(f"Saving {url} to {filename}")

    with open(filename, "w") as f:
        f.write(response.text)


def station_lonlat(station_name) -> tuple[float, float]:
    """Return the (lon, lat) in degrees of `station_name`."""
    df = read_station_llas()
    station_name = station_name.upper()
    if station_name not in df["name"].values:
        closest_names = difflib.get_close_matches(station_name, df["name"], n=5)
        raise ValueError(
            f"No station named {station_name} found. Closest matches: {closest_names}"
        )
    name, lat, lon, alt = df[df["name"] == station_name].iloc[0]
    return lon, lat


def station_xyz(station_name):
    """Return the (X, Y, Z) in meters of `station_name`"""
    df = read_station_xyzs()
    station_name = station_name.upper()
    if station_name not in df["name"].values:
        closest_names = difflib.get_close_matches(station_name, df["name"], n=5)
        raise ValueError(
            "No station named %s found. Closest: %s" % (station_name, closest_names)
        )
    X, Y, Z = df.loc[df["name"] == station_name, ["X", "Y", "Z"]].iloc[0]
    return X, Y, Z
