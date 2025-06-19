"""UNR gridded GPS data handling and downloading functionality."""

import datetime
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

from geepers.io import XarrayReader

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


def load_grid_geometry() -> gpd.GeoSeries:
    """Read the grid points as a GeoSeries."""
    df = _read_grid_file()
    return gpd.GeoSeries.from_xy(df.longitude, df.latitude, crs="EPSG:4326")


@cache
def _read_grid_file() -> pd.DataFrame:
    """Download the UNR grid latitude/longitude lookup table."""
    df = pd.read_csv(
        LOOKUP_FILE_URL,
        delim_whitespace=True,
        names=["grid_point", "longitude", "latitude"],
    )
    return df.set_index("grid_point")


def download_grid_timeseries(
    grid_points: Sequence[str | int],
    output_dir: Path,
    file_list: list[str] | None = None,
    max_workers: int = 8,
    reference: Reference = Reference.IGS14,
) -> None:
    """Download .tenv8 data files in parallel, showing progress.

    Parameters
    ----------
    output_dir : Path
        Directory to store downloaded data files.
    file_list : Optional[List[str]]
        Specific filenames to download. If None, files are listed remotely.
    max_workers : int
        Number of threads to use for downloading in parallel.

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
        url = f"{GRID_DATA_TEMPLATE_URL.format(reference=reference, filename=filename)}"
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
        Pandas datafram with columns
        [decimal_year, east, north, up, sigma_east, sigma_north, sigma_up, rapid_flag].

    """
    # Convert decimal year to datetime
    df = _read_tsv(uri)
    df["time"] = df["decimal_year"].apply(decimal_year_to_datetime)
    return df.drop(columns=["decimal_year"]).set_index("time")


@cache
def _read_tsv(uri: Path | str) -> pd.DataFrame:
    return pd.read_csv(
        uri,
        sep=r"\s+",
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


def decimal_year_to_datetime(decimal_year: float) -> datetime.datetime:
    """Convert a decimal year to a datetime object (approximate).

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
    day_of_year = round(fraction * 365.25)
    return datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year)


def get_grid_within_image(reader: XarrayReader) -> gpd.GeoDataFrame:
    """Find grid points within a given geocoded image.

    Parameters
    ----------
    reader : XarrayReader
        Reader object containing the geocoded DataArray.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing information about the GPS stations within the image.

    Notes
    -----
    This function assumes the image is in a geographic coordinate system (lat/lon).

    """
    import rasterio.warp
    from shapely import box

    if reader.crs != "EPSG:4326":
        bounds = rasterio.warp.transform_bounds(
            reader.crs, "EPSG:4326", *reader.da.rio.bounds()
        )
    else:
        bounds = reader.da.rio.bounds()
    bounds_poly = box(*bounds)

    # Get all GPS stations
    gdf_all = _read_grid_file()

    gdf_within = gdf_all.clip(bounds_poly)

    # Reset index for cleaner output
    gdf_within.reset_index(drop=True, inplace=True)
    return gdf_within
