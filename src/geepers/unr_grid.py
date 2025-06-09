#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = ["geopandas", "pyarrow", "tyro", "shapely", "requests"]
# ///
"""
A module to download and convert UNR gridded GPS time-series data into GeoParquet files.

Usage Example (CLI with tyro):
    python -m geepers.unr_grid --output-directory /path/to/output \
        --geoparquet-path /path/to/output/all_gps_data.parquet
"""

import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import requests
import tyro
from shapely.geometry import Point
from tqdm import tqdm

LOOKUP_FILE_URL = "https://geodesy.unr.edu/grid_timeseries/grid_latlon_lookup.txt"
GRID_DATA_BASE_URL = (
    "https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/IGS14/"
)


def download_lookup_file(output_dir: Path) -> Path:
    """
    Download the UNR grid latitude/longitude lookup table.

    Parameters
    ----------
    output_dir : Path
        Directory where the lookup file will be saved.

    Returns
    -------
    Path
        Full path to the downloaded lookup file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / "grid_latlon_lookup.txt"

    response = requests.get(LOOKUP_FILE_URL, stream=True)
    response.raise_for_status()

    with local_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path


def list_remote_data_files() -> List[str]:
    """
    Retrieve available .tenv8 filenames from the UNR grid data directory.

    Returns
    -------
    List[str]
        Filenames matching the pattern '######_IGS14.tenv8'.
    """
    response = requests.get(GRID_DATA_BASE_URL)
    response.raise_for_status()

    import re

    pattern = re.compile(r"(\d{6}_IGS14\.tenv8)")
    files = set(pattern.findall(response.text))
    return sorted(files)


def download_data_files(
    output_dir: Path,
    file_list: Optional[List[str]] = None,
    max_workers: int = 8,
) -> None:
    """
    Download .tenv8 data files in parallel, showing progress.

    Parameters
    ----------
    output_dir : Path
        Directory to store downloaded data files.
    file_list : Optional[List[str]]
        Specific filenames to download. If None, files are listed remotely.
    max_workers : int
        Number of threads to use for downloading in parallel.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if file_list is None:
        file_list = list_remote_data_files()

    def _download(fname: str) -> None:
        url = f"{GRID_DATA_BASE_URL}{fname}"
        dest = output_dir / fname
        if not dest.exists():
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download, fn): fn for fn in file_list}
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading data files"
        ):
            pass


def parse_lookup_file(file_path: Path) -> pd.DataFrame:
    """
    Parse lookup table to DataFrame of grid point coordinates.

    Parameters
    ----------
    file_path : Path
        Path to the lookup text file.

    Returns
    -------
    pd.DataFrame
        Columns: ['grid_point', 'longitude', 'latitude'].
    """
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        names=["grid_point", "longitude", "latitude"],
        dtype={"grid_point": str, "longitude": float, "latitude": float},
    )
    return df


def parse_data_file(file_path: Path) -> pd.DataFrame:
    """
    Parse a .tenv8 time-series data file into a DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the .tenv8 file.

    Returns
    -------
    pd.DataFrame
        Columns: [decimal_year, east, north, up, sigma_east, sigma_north, sigma_up, rapid_flag].
    """
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
        names=[
            "decimal_year",
            "east",
            "north",
            "up",
            "sigma_east",
            "sigma_north",
            "sigma_up",
            "rapid_flag",
        ],
    )
    return df


def decimal_year_to_datetime(decimal_year: float) -> datetime.datetime:
    """
    Convert a decimal year to a datetime object (approximate).

    Parameters
    ----------
    decimal_year : float
        Year expressed as a decimal (e.g., 2014.5).

    Returns
    -------
    datetime.datetime
        Corresponding calendar datetime (approximate to nearest day).
    """
    year = int(decimal_year)
    fraction = decimal_year - year
    day_of_year = int(round(fraction * 365.25))
    return datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year)


def convert_to_geoparquet(
    lookup_path: Path,
    data_dir: Path,
    geoparquet_path: Path,
) -> None:
    """
    Merge lookup coordinates with time-series and write GeoParquet.

    Parameters
    ----------
    lookup_path : Path
        Path to the grid lookup file.
    data_dir : Path
        Directory containing .tenv8 files.
    geoparquet_path : Path
        Output path for GeoParquet file.
    """
    lookup_df = parse_lookup_file(lookup_path)
    lookup_df["grid_point"] = lookup_df["grid_point"].str.zfill(6)

    records: List[pd.DataFrame] = []
    for data_file in sorted(data_dir.glob("*_IGS14.tenv8")):
        grid_id = data_file.stem.split("_")[0]
        df = parse_data_file(data_file)
        df["grid_point"] = grid_id
        merged = pd.merge(df, lookup_df, on="grid_point", how="left")
        merged["datetime"] = merged["decimal_year"].apply(decimal_year_to_datetime)
        records.append(merged)

    if not records:
        raise RuntimeError(f"No .tenv8 files found in {data_dir}")

    df_all = pd.concat(records, ignore_index=True)
    geometry = [Point(xy) for xy in zip(df_all.longitude, df_all.latitude)]
    gdf = gpd.GeoDataFrame(df_all, geometry=geometry, crs="EPSG:4326")
    gdf.drop(["latitude", "longitude", "decimal_year", "rapid_flag"], axis=1)
    gdf.to_parquet(geoparquet_path, index=False)


def download_and_convert_cli(
    data_dir: Path,
    geoparquet_path: Path | None = None,
    max_workers: int = 8,
) -> None:
    """
    CLI entry point to download and convert UNR grid data.

    Parameters
    ----------
    data_dir : Path
        Directory for intermediate downloads.
    geoparquet_path : Path, optional
        Final data_dir GeoParquet file location.
    max_workers : int
        Number of parallel download threads.
    """
    print("Downloading lookup file...")
    lookup_path = download_lookup_file(data_dir)
    print("Downloading data files...")
    download_data_files(data_dir, max_workers=max_workers)
    print("Converting to GeoParquet...")
    if geoparquet_path is None:
        today_str = datetime.datetime.today().strftime("%Y%m%d")
        geoparquet_path = data_dir / f"unr_gridded_{today_str}.parquet"

    convert_to_geoparquet(
        lookup_path=lookup_path,
        data_dir=data_dir,
        geoparquet_path=geoparquet_path,
    )
    print("Done.")


if __name__ == "__main__":
    tyro.cli(download_and_convert_cli)
