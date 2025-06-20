"""UNR gridded GPS data handling and downloading functionality."""

import datetime
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

from geepers.utils import get_cache_dir

LOOKUP_FILE_URL = "https://geodesy.unr.edu/grid_timeseries/grid_latlon_lookup.txt"
# Examples:
# https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/IGS14/000003_IGS14.tenv8
# https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/PA/000004_PA.tenv8
FILENAME_TEMPLATE = "{grid_point}_{reference}.tenv8"
GRID_DATA_TEMPLATE_URL = "https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/{reference}/{filename}"


class Reference(Enum):
    """Reference frames available for grid data."""

    IGS14 = "IGS14"
    NA = "NA"
    PA = "PA"


def load_grid_geometry() -> gpd.GeoDataFrame:
    """Read the grid points as a GeoSeries."""
    df = _read_grid_file()
    geometry = [
        Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude, strict=True)
    ]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


@cache
def _read_grid_file() -> pd.DataFrame:
    """Download the UNR grid latitude/longitude lookup table."""
    df = pd.read_csv(
        LOOKUP_FILE_URL,
        sep=r"\s+",
        names=["grid_point", "longitude", "latitude"],
    )

    # Validate the data
    if len(df) == 0:
        msg = "Grid lookup table is empty"
        raise ValueError(msg)

    # Validate coordinate ranges
    if not (df["longitude"].between(-180, 180).all()):
        msg = "Invalid longitude values in grid lookup table"
        raise ValueError(msg)

    if not (df["latitude"].between(-90, 90).all()):
        msg = "Invalid latitude values in grid lookup table"
        raise ValueError(msg)

    return df.set_index("grid_point")


def download_grid_timeseries(
    grid_idx: int,
    reference: Reference = Reference.IGS14,
) -> Path:
    """Download a single .tenv8 data file for a grid point.

    Parameters
    ----------
    grid_idx : int
        Grid point index (0-24800).
    reference : Reference
        Reference frame to use.

    Returns
    -------
    Path
        Path to the downloaded file.

    """
    if not (0 <= grid_idx <= 24800):
        msg = f"Grid index {grid_idx} must be between 0 and 24800"
        raise ValueError(msg)

    cache_dir = _get_cache_dir()

    filename = f"grid_{grid_idx:05d}_{reference.value}.tenv8"
    dest = cache_dir / filename

    if dest.exists():
        return dest

    # Format grid point for URL (6 digits with leading zeros)
    grid_point_str = f"{grid_idx:06d}"
    url_filename = FILENAME_TEMPLATE.format(
        grid_point=grid_point_str, reference=reference.value
    )
    url = GRID_DATA_TEMPLATE_URL.format(
        reference=reference.value, filename=url_filename
    )

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Validate that we actually got data
        if dest.stat().st_size == 0:
            dest.unlink()  # Remove empty file
            msg = f"Downloaded file {dest} is empty"
            raise ValueError(msg)

        return dest
    except requests.RequestException:
        # Clean up partial download if it exists
        if dest.exists():
            dest.unlink()
        raise


def download_grid_timeseries_bulk(
    grid_points: Sequence[str | int],
    output_dir: Path,
    file_list: list[str] | None = None,
    max_workers: int = 8,
    reference: Reference = Reference.IGS14,
) -> None:
    """Download .tenv8 data files in parallel, showing progress.

    Parameters
    ----------
    grid_points : Sequence[str | int]
        Grid point indices to download.
    output_dir : Path
        Directory to store downloaded data files.
    file_list : Optional[List[str]]
        Specific filenames to download. If None, files are listed remotely.
    max_workers : int
        Number of threads to use for downloading in parallel.
    reference : Reference
        Reference frame to use.

    """
    grid_points = [
        f"{grid_point:06d}" if isinstance(grid_point, int) else grid_point
        for grid_point in grid_points
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    # if file_list is None:
    # file_list = list_remote_data_files()

    def _download(grid_point: str) -> None:
        filename = FILENAME_TEMPLATE.format(
            grid_point=grid_point, reference=reference.value
        )
        dest = output_dir / filename
        url = (
            f"{GRID_DATA_TEMPLATE_URL.format(reference=reference.value, filename=filename)}"
        )
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


# Example data:
#  2012.0000       0.000       0.000       0.000  1.913  1.436  7.198 0
#  2012.0027      -1.302      -5.852       4.553  1.532  1.585  6.356 0
#  2012.0055       1.378      -2.093       0.792  1.956  1.590  6.394 0


def parse_data_file(uri: Path | str) -> pd.DataFrame:
    """Parse a .tenv8 time-series data file into a DataFrame.

    Parameters
    ----------
    uri : Path | str
        Path or URL to the .tenv8 file.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with columns
        ['datetime', 'de', 'dn', 'du', 'se', 'sn', 'su', 'rho'].

    """
    try:
        df = _read_tsv(uri)
        if len(df) == 0:
            return pd.DataFrame(
                columns=["datetime", "de", "dn", "du", "se", "sn", "su", "rho"]
            )

        df["datetime"] = df["decimal_year"].apply(decimal_year_to_datetime)
        df = df.drop(columns=["decimal_year"])

        # Rename columns to match expected format
        column_mapping = {
            "east": "de",
            "north": "dn",
            "up": "du",
            "sigma_east": "se",
            "sigma_north": "sn",
            "sigma_up": "su",
            "rapid_flag": "rho",
        }
        df = df.rename(columns=column_mapping)

        # Reorder columns to match expected format
        df = df[["datetime", "de", "dn", "du", "se", "sn", "su", "rho"]]
        return df
    except Exception:
        # Return empty DataFrame if parsing fails
        return pd.DataFrame(
            columns=["datetime", "de", "dn", "du", "se", "sn", "su", "rho"]
        )


@cache
def _read_tsv(uri: Path | str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            uri,
            sep=r"\s+",
            header=None,
            comment="#",
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
            on_bad_lines="warn",  # Use warn instead of skip to get better error handling
        )

        # Convert to numeric types and drop rows with NaN in decimal_year
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["decimal_year"])

        return df
    except Exception:
        # Return empty DataFrame if reading fails completely
        return pd.DataFrame(
            columns=[
                "decimal_year",
                "east",
                "north",
                "up",
                "sigma_east",
                "sigma_north",
                "sigma_up",
                "rapid_flag",
            ]
        )


def decimal_year_to_datetime(
    decimal_year: float | np.ndarray,
) -> datetime.datetime | list:
    """Convert a decimal year to a datetime object (approximate).

    Parameters
    ----------
    decimal_year : float or np.ndarray
        Year expressed as a decimal (e.g., 2014.5).

    Returns
    -------
    datetime.datetime or list
        Corresponding calendar datetime (approximate to nearest day).

    """
    if isinstance(decimal_year, np.ndarray):
        return [_convert_single_decimal_year(dy) for dy in decimal_year]
    else:
        return _convert_single_decimal_year(decimal_year)


def _convert_single_decimal_year(decimal_year: float) -> datetime.datetime:
    """Convert a single decimal year to datetime."""
    year = int(decimal_year)
    fraction = decimal_year - year

    # Handle leap years more accurately
    start_of_year = datetime.datetime(year, 1, 1)
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        days_in_year = 366
    else:
        days_in_year = 365

    day_of_year = round(fraction * days_in_year)
    return start_of_year + datetime.timedelta(days=day_of_year)


def get_grid_within_image(bounds_or_reader) -> gpd.GeoDataFrame:
    """Find grid points within a given geocoded image or bounds.

    Parameters
    ----------
    bounds_or_reader : XarrayReader or tuple
        Reader object containing the geocoded DataArray, or tuple of bounds (minx, miny, maxx, maxy).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing information about the grid points within the image.

    Notes
    -----
    This function assumes the image is in a geographic coordinate system (lat/lon).

    """
    from shapely import box

    if isinstance(bounds_or_reader, tuple):
        bounds = bounds_or_reader
    else:
        # XarrayReader case
        import rasterio.warp

        reader = bounds_or_reader
        if reader.crs != "EPSG:4326":
            bounds = rasterio.warp.transform_bounds(
                reader.crs, "EPSG:4326", *reader.da.rio.bounds()
            )
        else:
            bounds = reader.da.rio.bounds()

    # Handle invalid bounds
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        return gpd.GeoDataFrame(columns=["longitude", "latitude", "geometry"])

    bounds_poly = box(*bounds)

    # Get all grid points as GeoDataFrame
    df = _read_grid_file()
    geometry = [
        Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude, strict=True)
    ]
    gdf_all = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Find points within bounds
    mask = gdf_all.geometry.within(bounds_poly)
    gdf_within = gdf_all[mask].copy()

    return gdf_within


def load_grid_timeseries(
    grid_idx: int,
    start_date: str | datetime.date | None = None,
    end_date: str | datetime.date | None = None,
    download_if_missing: bool = True,
    zero_by: str = "mean",
    reference: Reference = Reference.IGS14,
) -> pd.DataFrame:
    """Load UNR grid time series data.

    Parameters
    ----------
    grid_idx : int
        The grid point index (0-24800).
    start_date : str or datetime.date, optional
        The start date for the data. If None, use all available data.
    end_date : str or datetime.date, optional
        The end date for the data. If None, use all available data.
    download_if_missing : bool, optional
        Whether to download the data if it's not found locally. Default is True.
    zero_by : str, optional
        How to zero the data. Either "mean" or "start". Default is "mean".
    reference : Reference, optional
        Reference frame to use. Default is IGS14.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the grid time series data.

    Raises
    ------
    ValueError
        If the grid data file is not found and download_if_missing is False.

    """
    cache_dir = _get_cache_dir()
    grid_data_file = cache_dir / f"grid_{grid_idx:05d}_{reference.value}.tenv8"

    if not grid_data_file.exists():
        if download_if_missing:
            download_grid_timeseries(grid_idx, reference)
        else:
            msg = f"{grid_data_file} does not exist, download_if_missing = False"
            raise ValueError(msg)

    df = parse_data_file(grid_data_file)
    df = _clean_grid_df(df, start_date, end_date)

    if zero_by.lower() == "mean":
        mean_val = df[["de", "dn", "du"]].mean()
        df[["de", "dn", "du"]] -= mean_val
    elif zero_by.lower() == "start":
        start_val = df[["de", "dn", "du"]].iloc[:10].mean()
        df[["de", "dn", "du"]] -= start_val
    else:
        msg = "zero_by must be either 'mean' or 'start'"
        raise ValueError(msg)

    return df.set_index("datetime")


def _clean_grid_df(
    df: pd.DataFrame,
    start_date: str | datetime.date | None = None,
    end_date: str | datetime.date | None = None,
) -> pd.DataFrame:
    """Clean and preprocess the grid DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw grid DataFrame.
    start_date : str or datetime.date, optional
        The start date for the data. If None, use all available data.
    end_date : str or datetime.date, optional
        The end date for the data. If None, use all available data.

    Returns
    -------
    pd.DataFrame
        The cleaned grid DataFrame.

    """
    if len(df) == 0:
        return df

    # Convert datetime column to pandas datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    if start_date:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def remove_outliers(
    df: pd.DataFrame, method: str = "iqr", threshold: float = 5.0
) -> pd.DataFrame:
    """Remove outliers from grid time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Grid data DataFrame.
    method : str
        Method for outlier detection. Either "iqr" or "std".
    threshold : float
        Threshold for outlier detection.
        Default is 5.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed.

    """
    if len(df) == 0:
        return df

    displacement_cols = ["de", "dn", "du"]
    outlier_mask = pd.Series(False, index=df.index)

    for col in displacement_cols:
        if col not in df.columns:
            continue

        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == "std":
            mean_val = df[col].mean()
            std_val = df[col].std()
            outlier_mask |= abs(df[col] - mean_val) > threshold * std_val
        else:
            msg = "method must be either 'iqr' or 'std'"
            raise ValueError(msg)

    initial_len = len(df)
    df_clean = df[~outlier_mask].copy()

    outliers = df[outlier_mask].copy()
    if len(outliers) > 0:
        import warnings

        msg = (
            f"Removed {initial_len - len(df_clean)} outlier points using"
            f" {method} method"
        )
        msg += f"\nOutliers: {outliers}"

        warnings.warn(msg, stacklevel=2)

    return df_clean


def _get_cache_dir() -> Path:
    """Get the cache directory for UNR grid data.

    Returns
    -------
    Path
        Cache directory path.

    """
    d = get_cache_dir() / "unr_grid"
    d.mkdir(parents=True, exist_ok=True)
    return d
