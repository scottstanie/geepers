"""Compare InSAR time-series displacements with collocated GPS observations.

Examples
--------
    python gps_insar_comparison.py --help    # show usage
    python gps_insar_comparison.py \
        --timeseries-files data/ts/*.tif \
        --los-enu-file data/los_enu.tif \
        --reference-station P123 \
        --output-dir results/

"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tyro
from numpy.typing import ArrayLike
from opera_utils import get_dates
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

import geepers.gps
import geepers.rates
from geepers._types import PathOrStr
from geepers.constants import SENTINEL_1_WAVELENGTH
from geepers.io import RasterReader, RasterStackReader, get_raster_units

__all__ = [
    "_convert_to_meters",
    "compare_relative_gps_insar",
    "create_tidy_df",
    "main",
    "process_insar_data",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_tidy_df(station_to_merged_df: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-station dataframes into a tidy (long-form) dataframe.

    Parameters
    ----------
    station_to_merged_df
        Mapping from station name to a *wide* dataframe that contains one column
        per variable (e.g. ``los_gps``, ``los_insar``).

    Returns
    -------
    pandas.DataFrame
        Long-form dataframe with columns ``station``, ``date``, ``measurement``
        and ``value`` suitable for plotting with *seaborn* or *altair*.

    """
    dfs: list[pd.DataFrame] = []
    for station, df in station_to_merged_df.items():
        df_reset = df.reset_index()
        df_melted = pd.melt(
            df_reset, id_vars=["date"], var_name="measurement", value_name="value"
        )
        df_melted["station"] = station
        dfs.append(df_melted)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df[["station", "date", "measurement", "value"]]


def compare_relative_gps_insar(
    station_to_merged_df: Mapping[str, pd.DataFrame],
    *,
    reference_station: str,
) -> pd.DataFrame:
    """Compute relative displacement between all stations and a reference.

    The function subtracts the *GPS* and *InSAR* line-of-sight (LOS)
    displacements of *reference_station* from every other station, yielding
    time-series of relative motion.

    Parameters
    ----------
    station_to_merged_df
        Mapping from station name to merged GPS/InSAR dataframe produced by the
        main workflow.
    reference_station
        Name of the station to treat as the zero reference.

    Returns
    -------
    pandas.DataFrame
        Tidy dataframe with the relative series and their differences.

    """
    if reference_station not in station_to_merged_df:
        msg = f"Reference station '{reference_station}' not found."
        raise ValueError(msg)

    ref_df = station_to_merged_df[reference_station]
    results: list[pd.DataFrame] = []

    for station, df in station_to_merged_df.items():
        common_index = df.index.intersection(ref_df.index)
        if common_index.empty:
            logger.warning(
                "No common epochs between %s and %s", station, reference_station
            )
            continue

        station_df = df.loc[common_index]
        ref_df_aligned = ref_df.loc[common_index]

        relative_gps = station_df["los_gps"] - ref_df_aligned["los_gps"]
        relative_insar = station_df["los_insar"] - ref_df_aligned["los_insar"]
        difference = relative_insar - relative_gps

        results.append(
            pd.DataFrame(
                {
                    "station": station,
                    "date": common_index,
                    "relative_gps": relative_gps,
                    "relative_insar": relative_insar,
                    "difference": difference,
                }
            )
        )

    return pd.concat(results, ignore_index=True)


def process_insar_data(
    *,
    reader: RasterStackReader,
    df_gps_stations: pd.DataFrame,
    file_date_fmt: str = "%Y%m%d",
    reader_temporal_coherence: RasterReader | None = None,
    reader_similarity: RasterReader | None = None,
) -> dict[str, pd.DataFrame]:
    """Sample InSAR rasters at all station locations in one pass.

    Parameters
    ----------
    reader : RasterStackReader
        *RasterStackReader* opened on the displacement stack.
        Its ``file_list`` is assumed to be sorted chronologically.
    df_gps_stations
        DataFrame indexed by station name with at least ``lon`` and ``lat``
        columns (decimal degrees).
    file_date_fmt
        ``strftime``-compatible pattern used to parse dates from filenames.
    reader_temporal_coherence, reader_similarity
        Optional single-band rasters to sample alongside displacement.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A mapping from station name to a dataframe that contains *los_insar*
        (in **metres**), *temporal_coherence* and *similarity* columns indexed
        by acquisition date.

    """
    lons = df_gps_stations.lon.to_numpy()
    lats = df_gps_stations.lat.to_numpy()

    # Displacement is returned as (n_dates, n_stations)
    los_insar_rad = reader.read_lon_lat(lons, lats, masked=True).squeeze()

    temp_coh = (
        reader_temporal_coherence.read_lon_lat(lons, lats, masked=True).squeeze()
        if reader_temporal_coherence is not None
        else None
    )
    similarity = (
        reader_similarity.read_lon_lat(lons, lats, masked=True).squeeze()
        if reader_similarity is not None
        else None
    )

    los_insar_m = _convert_to_meters(reader.file_list[0], los_insar_rad)

    # Parse secondary dates from filenames
    sec_dates = pd.to_datetime(
        [get_dates(f, fmt=file_date_fmt)[1] for f in reader.file_list]
    )

    station_to_insar: dict[str, pd.DataFrame] = {}
    for i, station in enumerate(df_gps_stations.index):
        data = {
            "los_insar": los_insar_m[:, i],
        }
        if similarity is not None:
            data["similarity"] = similarity[i]
        if temp_coh is not None:
            data["temporal_coherence"] = temp_coh[i]

        station_to_insar[station] = pd.DataFrame(index=sec_dates, data=data)

    return station_to_insar


def _convert_to_meters(
    filename: PathOrStr,
    arr: ArrayLike,
    *,
    wavelength: float = SENTINEL_1_WAVELENGTH,
) -> np.ndarray:
    """Convert displacement raster to metres if stored in radians.

    Parameters
    ----------
    filename
        Path to a raster whose *units* metadata will be inspected.
    arr
        Array of displacements to convert.
    wavelength
        Radar wavelength (metres) used for phase → displacement conversion.

    Returns
    -------
    numpy.ndarray
        Displacements in metres.

    """
    phase2disp = float(wavelength) / (4.0 * np.pi)
    input_units = get_raster_units(filename)

    if input_units == "radians":
        return np.asarray(arr) * phase2disp
    if input_units == "meters":
        return np.asarray(arr)

    logger.debug("Unknown units '%s' for %s - assuming meters", input_units, filename)
    return np.asarray(arr)


def main(
    *,
    timeseries_files: Sequence[Path],
    los_enu_file: Path,
    output_dir: Path = Path("GPS"),
    file_date_fmt: str = "%Y%m%d",
    reference_station: str | None = None,
    temporal_coherence_file: Path | None = None,
    similarity_file: Path | None = None,
) -> None:
    """Process InSAR time-series and compare them to GPS displacements.

    Parameters
    ----------
    timeseries_files
        list of wrapped-phase (or displacement) rasters, *one per acquisition*.
        File names **must** encode the reference and secondary dates using
        *file_date_fmt*.
    los_enu_file
        Three-band GeoTIFF with the line-of-sight unit vector expressed in the
        local East-North-Up coordinate frame.
    output_dir
        Directory where CSV outputs will be written.  Will be created if it does
        not exist.
    file_date_fmt
        ``strftime`` pattern describing how dates are embedded in
        *timeseries_files*.
    reference_station
        Optional GPS station name - if provided, relative displacements are
        computed with respect to this station.
    temporal_coherence_file, similarity_file
        Optional rasters providing per-pixel temporal coherence and (Parizzi)
        phase similarity which will be sampled at station locations.

    Notes
    -----
    The script writes three CSV files into *output_dir*:

    ``combined_data.csv``
        Tidy table stacking raw GPS and InSAR series for each station.
    ``station_summary.csv``
        Per-station linear rates (mm/yr) computed from the combined table.
    ``relative_comparison.csv``
        Relative GPS/InSAR displacements if *reference_station* was specified.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("%d time-series rasters → %s", len(timeseries_files), output_dir)

    reader = RasterStackReader.from_file_list(timeseries_files)
    reader_temporal_coherence = (
        RasterReader.from_file(temporal_coherence_file)
        if temporal_coherence_file is not None
        else None
    )
    reader_similarity = (
        RasterReader.from_file(similarity_file) if similarity_file is not None else None
    )

    file_dates = [get_dates(f, fmt=file_date_fmt) for f in timeseries_files]
    ref_sec_dates = np.asarray(file_dates)
    if np.unique(ref_sec_dates[:, 0]).size > 1:
        msg = (
            "Multiple reference dates detected in the stack - "
            "processing currently assumes a single reference epoch."
        )
        raise ValueError(msg)

    start_date, end_date = np.min(ref_sec_dates), np.max(ref_sec_dates)
    logger.info("Reference→secondary range: %s → %s", start_date, end_date)

    df_gps_stations = geepers.gps.get_stations_within_image(timeseries_files[0])
    df_gps_stations.set_index("name", inplace=True)

    def _load_or_none(name: str) -> pd.DataFrame | None:
        try:
            return geepers.gps.load_station_enu(
                station_name=name, start_date=start_date, end_date=end_date
            )
        except requests.HTTPError:
            return None

    df_gps_list = thread_map(
        _load_or_none,
        df_gps_stations.index,
        max_workers=5,
        desc="Downloading GPS station data",
    )

    los_reader = RasterReader.from_file(los_enu_file)
    station_to_los_gps: dict[str, pd.DataFrame] = {}

    for station_row, df in tqdm(
        zip(df_gps_stations.itertuples(), df_gps_list, strict=False),
        total=len(df_gps_stations),
        desc="Projecting GPS→LOS",
    ):
        if df is None:
            warnings.warn(
                f"Failed to download {station_row.Index}; skipping.", stacklevel=2
            )
            continue

        enu_vec = los_reader.read_lon_lat(
            station_row.lon, station_row.lat, masked=True
        ).ravel()
        if np.allclose(enu_vec, 0):
            warnings.warn(
                f"{station_row.Index} lies outside LOS raster; skipping.", stacklevel=2
            )
            continue

        e, n, u = enu_vec
        df["los_gps"] = df.east * e + df.north * n + df.up * u
        df["los_gps"] -= df["los_gps"].iloc[:10].mean()  # remove arbitrary offset
        station_to_los_gps[station_row.Index] = df[["los_gps"]]

    # Sample InSAR rasters at station locations
    station_to_insar = process_insar_data(
        reader=reader,
        df_gps_stations=df_gps_stations,
        file_date_fmt=file_date_fmt,
        reader_temporal_coherence=reader_temporal_coherence,
        reader_similarity=reader_similarity,
    )

    # Merge GPS and InSAR tables per station
    station_to_merged: dict[str, pd.DataFrame] = {}
    for name in tqdm(station_to_los_gps, desc="Merging GPS↔InSAR"):
        station_to_merged[name] = pd.merge(
            station_to_los_gps[name],
            station_to_insar[name],
            how="left",
            left_index=True,
            right_index=True,
        )

    # Save results
    combined_df = create_tidy_df(station_to_merged)
    combined_df.to_csv(output_dir / "combined_data.csv", index=False)

    df_rates = geepers.rates.calculate_rates(df=combined_df, to_mm=True)
    df_rates.to_csv(output_dir / "station_summary.csv")

    if reference_station:
        rel_df = compare_relative_gps_insar(
            station_to_merged, reference_station=reference_station
        )
        rel_df.to_csv(output_dir / "relative_comparison.csv", index=False)

    logger.info("Finished - results written to %s", output_dir)


def cli():
    tyro.cli(main)


cli.__doc__ = main.__doc__


if __name__ == "__main__":
    cli()
