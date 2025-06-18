"""Compare InSAR time-series displacements with collocated GPS observations."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import requests
import tyro
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from tqdm.dask import TqdmCallback

import geepers.gps
import geepers.rates
from geepers.constants import SENTINEL_1_WAVELENGTH
from geepers.io import XarrayReader

__all__ = [
    "compare_relative_gps_insar",
    "create_tidy_df",
    "main",
    "process_insar_data",
]

logger = logging.getLogger("geepers")
logger.setLevel(logging.INFO)

PHASE_TO_METERS = float(SENTINEL_1_WAVELENGTH) / (4.0 * np.pi)


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
    reader: XarrayReader,
    df_gps_stations: pd.DataFrame,
    # reader_temporal_coherence: DatasetReader | None = None,
    # reader_similarity: DatasetReader | None = None,
) -> dict[str, pd.DataFrame]:
    """Sample InSAR rasters at all station locations in one pass.

    Parameters
    ----------
    reader : RasterStackReader
        *RasterStackReader* opened on the displacement stack.
    df_gps_stations
        DataFrame indexed by station name with at least ``lon`` and ``lat``
        columns (decimal degrees).
    reader_temporal_coherence, reader_similarity
        Optional single-band rasters to sample alongside displacement to
        compute temporal coherence and phase similarity.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A mapping from station name to a dataframe that contains *los_insar*
        (in **metres**), *temporal_coherence* and *similarity* columns indexed
        by acquisition date.

    """
    lons = df_gps_stations.lon.to_numpy()
    lats = df_gps_stations.lat.to_numpy()

    # los_insar gets stacks as (n_stations, len(time) ), each row one station
    from dask import compute

    with TqdmCallback(desc="Sampling InSAR data locations"):
        r = compute(reader.read_lon_lat(lons, lats))
    los_insar = np.stack(r).squeeze()

    # # TODO: make a list of variables to sample, doesn't have to be just one single
    # # band reader like I made here:
    # temp_coh = (
    #     np.stack(reader_temporal_coherence.read_lon_lat(lons, lats))
    #     if reader_temporal_coherence is not None
    #     else None
    # )
    # similarity = (
    #     np.stack(reader_similarity.read_lon_lat(lons, lats))
    #     if reader_similarity is not None
    #     else None
    # )

    if reader.units not in ("meters", "m"):
        logger.warning("Converting InSAR displacement to meters.")
        los_insar *= PHASE_TO_METERS

    station_to_insar: dict[str, pd.DataFrame] = {}
    for i, station in tqdm(
        enumerate(df_gps_stations.index), total=len(df_gps_stations)
    ):
        data = {
            "los_insar": los_insar[i],
        }
        # if similarity is not None:
        #     data["similarity"] = similarity[i]
        # if temp_coh is not None:
        #     data["temporal_coherence"] = temp_coh[i]

        station_to_insar[station] = pd.DataFrame(index=reader.da.time, data=data)

    return station_to_insar


def main(
    *,
    los_enu_file: Annotated[Path | str, tyro.conf.arg(aliases=["--los"])],
    timeseries_files: Sequence[Path | str] | None = None,
    timeseries_stack: Path | str | None = None,
    output_dir: Annotated[Path, tyro.conf.arg(aliases=["-o"])] = Path("./GPS"),
    file_date_fmt: str = "%Y%m%d",
    stack_data_var: str | None = "displacement",
    reference_station: Annotated[str | None, tyro.conf.arg(aliases=["--ref"])] = None,
    # temporal_coherence_file: Path | str | None = None,
    # similarity_file: Path | None = None,
    compute_rates: Annotated[bool, tyro.conf.arg(aliases=["--rates"])] = False,
) -> None:
    """Process InSAR time-series and compare them to GPS displacements.

    Parameters
    ----------
    los_enu_file
        Three-band GeoTIFF with the line-of-sight unit vector expressed in the
        local East-North-Up coordinate frame.
        LOS convention is that the unit vectors point from the ground toward
        the satellite (i.e. the "up" component is positive).
    timeseries_files
        List of wrapped-phase (or displacement) rasters, one per acquisition.
        File names must encode the reference and secondary dates using
        `file_date_fmt`.
    timeseries_stack
        Path to an xarray stack of wrapped-phase (or displacement) rasters.
        Alternative to `timeseries_files`
    output_dir
        Directory where CSV outputs will be written.  Will be created if it does
        not exist.
    file_date_fmt
        ``strftime`` pattern describing how dates are embedded in
        `timeseries_files`.
    stack_data_var
        Name of the variable in the timeseries stack to use for InSAR data.
        If `None`, the `timeseries_stack` must have only one data variable.
    reference_station
        Optional GPS station name - if provided, relative displacements are
        computed with respect to this station.
    temporal_coherence_file, similarity_file
        Optional rasters providing per-pixel temporal coherence and (Parizzi)
        phase similarity which will be sampled at station locations.
    compute_rates : bool
        If `True`, computes a separate summary DataFrame of the average rate
        for each GPS station/InSAR location.
        Default is False.

    Notes
    -----
    The script writes three CSV files into *output_dir*:

    ``combined_data.csv``
        Tidy table stacking raw GPS and InSAR series for each station.
    ``relative_comparison.csv``
        Relative GPS/InSAR displacements if *reference_station* was specified,
        if `reference_station` is not `None`.
    ``station_summary.csv``
        Per-station linear rates (mm/yr) computed from the combined table,
        if `compute_rates` is `True`.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if timeseries_files is None:
        if timeseries_stack is None:
            msg = "Must provide either timeseries_files or timeseries_stack"
            raise ValueError(msg)

        insar_reader = XarrayReader.from_file(timeseries_stack, data_var=stack_data_var)
    else:
        insar_reader = XarrayReader.from_file_list(timeseries_files, file_date_fmt)

    logger.info("Created %s", insar_reader)
    # # TODO: support temporal coherence and similarity dsets from xarray stack
    # reader_temporal_coherence = (
    #     RasterReader.from_file(temporal_coherence_file)
    #     if temporal_coherence_file is not None
    #     else None
    # )
    # reader_similarity = (
    #     RasterReader.from_file(similarity_file)
    #     if similarity_file is not None else None
    # )

    df_gps_stations = geepers.gps.get_stations_within_image(
        insar_reader, mask_invalid=False
    )
    df_gps_stations.set_index("name", inplace=True)

    start_date = insar_reader.da.time[0].to_pandas()
    end_date = insar_reader.da.time[-1].to_pandas()

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
        max_workers=10,
        desc="Downloading GPS station data",
    )

    los_reader = XarrayReader.from_file(los_enu_file, units="unitless")
    station_to_los_gps: dict[str, pd.DataFrame] = {}

    for station_row, df in tqdm(
        zip(df_gps_stations.itertuples(), df_gps_list, strict=False),
        total=len(df_gps_stations),
        desc="Projecting GPS -> LOS",
    ):
        if df is None:
            warnings.warn(
                f"Failed to download {station_row.Index}; skipping.", stacklevel=2
            )
            continue

        enu_vec = np.nan_to_num(
            los_reader.read_lon_lat(station_row.lon, station_row.lat)
        )
        assert enu_vec.shape == (3,)
        if np.allclose(enu_vec, 0):
            logger.info(f"{station_row.Index} lies has nodata in LOS raster; skipping.")
            continue

        e, n, u = enu_vec
        df["los_gps"] = df.east * e + df.north * n + df.up * u
        df["los_gps"] -= df["los_gps"].iloc[:10].mean()  # remove arbitrary offset
        station_to_los_gps[station_row.Index] = df[["los_gps"]]

    # Sample InSAR rasters at station locations
    logger.info("Sampling InSAR rasters at station locations")
    station_to_insar = process_insar_data(
        reader=insar_reader,
        df_gps_stations=df_gps_stations,
        # reader_temporal_coherence=reader_temporal_coherence,
        # reader_similarity=reader_similarity,
    )

    # Merge GPS and InSAR tables per station
    logger.info("Merging GPS and InSAR tables per station")
    station_to_merged: dict[str, pd.DataFrame] = {}
    for name in tqdm(station_to_los_gps, desc="Merging GPS and InSAR"):
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

    if reference_station:
        logger.info("Comparing GPS and InSAR relative to %s", reference_station)
        rel_df = compare_relative_gps_insar(
            station_to_merged, reference_station=reference_station
        )
        rel_df.to_csv(output_dir / "relative_comparison.csv", index=False)

    if compute_rates:
        df_rates = geepers.rates.calculate_rates(df=combined_df, to_mm=True)
        df_rates.to_csv(output_dir / "station_summary.csv")

    logger.info("Finished - results written to %s", output_dir)
