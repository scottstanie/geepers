import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import typer
from numpy.typing import ArrayLike
from opera_utils import get_dates
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from typing_extensions import Annotated

import geepers.gps
import geepers.io
import geepers.rates
from geepers._types import PathOrStr
from geepers.constants import SENTINEL_1_WAVELENGTH

logger = logging.getLogger(__name__)


main = typer.Typer()


def create_tidy_df(station_to_merged_df):
    dfs = []
    for station, df in station_to_merged_df.items():
        df_reset = df.reset_index()
        df_melted = pd.melt(
            df_reset, id_vars=["date"], var_name="measurement", value_name="value"
        )
        df_melted["station"] = station
        dfs.append(df_melted)

    combined_df = pd.concat(dfs, ignore_index=True)
    column_order = ["station", "date", "measurement", "value"]
    return combined_df[column_order]


def compare_relative_gps_insar(
    station_to_merged_df: dict[str, pd.DataFrame], reference_station: str
) -> pd.DataFrame:
    if reference_station not in station_to_merged_df:
        raise ValueError(
            f"Reference station '{reference_station}' not found in the data."
        )

    ref_df = station_to_merged_df[reference_station]
    results = []

    for station, df in station_to_merged_df.items():
        common_index = df.index.intersection(ref_df.index)
        station_df = df.loc[common_index]
        ref_df_aligned = ref_df.loc[common_index]

        relative_gps = station_df["los_gps"] - ref_df_aligned["los_gps"]
        relative_insar = station_df["los_insar"] - ref_df_aligned["los_insar"]
        difference = relative_insar - relative_gps

        station_result = pd.DataFrame(
            {
                "station": station,
                "date": common_index,
                "relative_gps": relative_gps,
                "relative_insar": relative_insar,
                "difference": difference,
            }
        )
        results.append(station_result)

    return pd.concat(results, ignore_index=True)


@main.command()
def run(
    timeseries_files: Annotated[
        list[Path], typer.Argument(help="Path to InSAR time series files")
    ],
    los_enu_file: Annotated[Path, typer.Option(help="Path to LOS ENU file")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save CSVs")] = Path(
        "GPS"
    ),
    file_date_fmt: Annotated[
        str, typer.Option(help="Date format in filenames")
    ] = "%Y%m%d",
    reference_station: Annotated[
        str, typer.Option(help="Reference GPS station name")
    ] = None,
    temporal_coherence_file: Annotated[
        Path, typer.Option(help="Path to temporal coherence")
    ] = None,
    similarity_file: Annotated[
        Path, typer.Option(help="Path to phase similarity ")
    ] = None,
):
    """
    Process InSAR time series and compare with GPS data.
    """

    # Rest of your main function logic goes here...
    typer.echo(f"Processing {len(timeseries_files)} time series files")
    typer.echo(f"Using LOS ENU file: {los_enu_file}")
    typer.echo(f"Output directory: {output_dir}")
    typer.echo(f"File date format: {file_date_fmt}")
    if reference_station:
        typer.echo(f"Reference station: {reference_station}")

    # Initialize RasterStackReader
    reader = geepers.io.RasterStackReader.from_file_list(timeseries_files)
    if temporal_coherence_file:
        reader_temporal_coherence = geepers.io.RasterReader.from_file(
            temporal_coherence_file
        )
    else:
        reader_temporal_coherence = None
    if similarity_file:
        reader_similarity = geepers.io.RasterReader.from_file(similarity_file)
    else:
        reader_temporal_coherence = None

    output_dir.mkdir(exist_ok=True)

    # Parse dates from filenames
    file_dates = [get_dates(f, fmt=file_date_fmt) for f in timeseries_files]
    ref_sec_dates = np.array(file_dates)
    start_date, end_date = np.min(ref_sec_dates), np.max(ref_sec_dates)
    sec_date_series = pd.to_datetime(ref_sec_dates[:, 1])
    if np.unique(ref_sec_dates[:, 0]).size > 1:
        raise ValueError(
            f"Parsed more than one reference date in file list: {file_dates}"
        )

    # Get GPS stations within the image
    df_gps_stations = geepers.gps.get_stations_within_image(timeseries_files[0])
    df_gps_stations.set_index("name", inplace=True)
    num_stations = len(df_gps_stations)

    # Download GPS data
    def _load_or_none(name: str) -> pd.DataFrame | None:
        try:
            return geepers.gps.load_station_enu(
                station_name=name, start_date=start_date, end_date=end_date
            )
        except requests.HTTPError:
            return None

    max_workers = 5
    df_gps_list = thread_map(
        _load_or_none,
        df_gps_stations.index,
        max_workers=max_workers,
        desc="Downloading GPS Station data",
    )

    # Load LOS ENU data
    los_reader = geepers.io.RasterReader.from_file(los_enu_file)
    # Process GPS data for each station
    starting_points = 10
    station_to_los_gps_data: dict[str, pd.DataFrame] = {}
    for station_row, df in tqdm(
        zip(df_gps_stations.itertuples(), df_gps_list),
        total=num_stations,
        desc="Projecting GPS to LOS",
    ):
        if df is None:
            typer.echo(f"Failed to download {station_row.Index}. Skipping")
            continue
        enu_vec = los_reader.read_lon_lat(
            station_row.lon, station_row.lat, masked=True
        ).ravel()
        if (enu_vec == 0).all():
            warnings.warn(f"{station_row.Index} does not have LOS data. Skipping")
            continue

        e, n, u = enu_vec
        df["los_gps"] = df.east * e + df.north * n + df.up * u
        df["los_gps"] -= df["los_gps"][:starting_points].mean()
        station_to_los_gps_data[station_row.Index] = df[["los_gps"]]

    # Process InSAR data for each station
    # TODO: this is slow for large number of dates
    # Should probably pass all lons/lats, so that readers can each fetch all,
    # rather than each reader gets 1 and iterate through readers
    station_to_insar_data = process_insar_data(
        reader=reader,
        df_gps_stations=df_gps_stations,
        file_date_fmt=file_date_fmt,
        reader_temporal_coherence=reader_temporal_coherence,
        reader_similarity=reader_similarity,
    )

    # Merge GPS and InSAR data
    station_to_merged_df: dict[str, pd.DataFrame] = {}
    for name in tqdm(station_to_los_gps_data, desc="Merging GPS and InSAR"):
        df_merged = pd.merge(
            left=station_to_los_gps_data[name],
            right=station_to_insar_data[name],
            how="left",
            left_index=True,
            right_index=True,
        )
        station_to_merged_df[name] = df_merged

    # Create tidy DataFrame
    combined_df = create_tidy_df(station_to_merged_df)
    # Save results
    combined_df.to_csv(output_dir / "combined_data.csv", index=False)

    # Get the rates and summary stats
    df_rates = geepers.rates.calculate_rates(df=combined_df, to_mm=True)
    df_rates.to_csv(output_dir / "station_summary.csv", index=True)

    # TODO: this should probably be done above?
    # This relative part is pretty suspect:

    # Compare relative GPS and InSAR if reference station is provided
    if reference_station:
        compare_results = compare_relative_gps_insar(
            station_to_merged_df, reference_station=reference_station
        )
        print("Relative comparison results:")
        print(compare_results)
        compare_results.to_csv(output_dir / "relative_comparison.csv", index=False)

    typer.echo(f"Results saved in {output_dir}")


# Process InSAR data for each station more efficiently by reading all stations at once
def process_insar_data(
    reader: geepers.io.RasterStackReader,
    df_gps_stations: pd.DataFrame,
    file_date_fmt: str = "%Y%m%d",
    reader_temporal_coherence: geepers.io.RasterReader | None = None,
    reader_similarity: geepers.io.RasterReader | None = None,
) -> dict[str, pd.DataFrame]:
    """Process InSAR data for all stations more efficiently.

    Parameters
    ----------
    reader : RasterStackReader
        Reader containing the InSAR time series data
    df_gps_stations : pd.DataFrame
        DataFrame containing station information with 'lon' and 'lat' columns
        and station names as index

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping station names to DataFrames containing InSAR data
    """
    # Get all station coordinates at once
    lons = df_gps_stations.lon.values
    lats = df_gps_stations.lat.values

    # Read all stations at once for each time step
    # This returns array of shape (n_dates, n_stations)
    los_insar_rad = reader.read_lon_lat(lons, lats, masked=True).squeeze()

    temp_coh = (
        reader_temporal_coherence.read_lon_lat(lons, lats, masked=True).squeeze()
        if reader_temporal_coherence
        else None
    )
    similarity = (
        reader_similarity.read_lon_lat(lons, lats, masked=True).squeeze()
        if reader_similarity
        else None
    )

    # Convert to meters
    los_insar = convert_to_meters(
        reader.file_list[0], los_insar_rad, wavelength=SENTINEL_1_WAVELENGTH
    )

    # Create date series for the time steps
    sec_date_series = pd.to_datetime(
        [get_dates(f, fmt=file_date_fmt)[1] for f in reader.file_list]
    )

    # Create dictionary of DataFrames for each station
    station_to_insar_data = {}
    for i, station in enumerate(df_gps_stations.index):
        station_to_insar_data[station] = pd.DataFrame(
            index=sec_date_series,
            data={
                "los_insar": los_insar[:, i],
                "similarity": similarity[i],
                "temporal_coherence": temp_coh[i],
            },
        )

    return station_to_insar_data


def convert_to_meters(
    filename: PathOrStr, arr: ArrayLike, wavelength: float = SENTINEL_1_WAVELENGTH
):
    phase2disp = float(wavelength) / (4.0 * np.pi)
    input_units = geepers.io.get_raster_units(filename)
    if not input_units or input_units not in ("meters", "radians"):
        logger.debug(f"Unknown units for {filename}: assuming radians")
        return arr * phase2disp
    elif input_units == "radians":
        return arr * phase2disp
    else:
        return arr


if __name__ == "__main__":
    main()
