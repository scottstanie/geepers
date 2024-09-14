import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from numpy.typing import ArrayLike
from opera_utils import get_dates
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

import geepers.gps
import geepers.io
from geepers._types import PathOrStr
from geepers.constants import SENTINEL_1_WAVELENGTH

logger = logging.getLogger(__name__)


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


def _get_cli_args():
    parser = argparse.ArgumentParser(
        description="Process InSAR time series and compare with GPS data."
    )
    parser.add_argument(
        "timeseries_files", nargs="+", help="Path to InSAR time series files"
    )
    parser.add_argument("--los-enu-file", help="Path to LOS ENU file", required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Directory to save CSVs", default="GPS"
    )
    parser.add_argument(
        "--file_date_fmt",
        default="%Y%m%d",
        help="Date format in filenames (default: %(default)s)",
    )
    parser.add_argument("--reference_station", help="Reference GPS station name")
    return parser.parse_args()


main = typer.Typer()
from typing_extensions import Annotated


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
    max_workers = 5
    df_gps_list = thread_map(
        lambda name: geepers.gps.load_station_enu(
            station_name=name, start_date=start_date, end_date=end_date
        ),
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
        enu_vec = los_reader.read_lon_lat(
            station_row.lon, station_row.lat, masked=True
        ).ravel()
        if (enu_vec == 0).all():
            warnings.warn(f"{station_row.Index} does not have LOS data")
        e, n, u = enu_vec
        df["los_gps"] = df.east * e + df.north * n + df.up * u
        df["los_gps"] -= df["los_gps"][:starting_points].mean()
        station_to_los_gps_data[station_row.Index] = df[["los_gps"]]

    # Process InSAR data for each station
    station_to_insar_data: dict[str, pd.DataFrame] = {}
    for station_row in tqdm(
        df_gps_stations.itertuples(),
        total=num_stations,
        desc="Loading InSAR at station locations",
    ):
        los_insar_rad = (
            reader.read_lon_lat(station_row.lon, station_row.lat, masked=True)
            .ravel()
            .squeeze()
        )
        los_insar = convert_to_meters(
            reader.file_list[0], los_insar_rad, wavelength=SENTINEL_1_WAVELENGTH
        )

        station_to_insar_data[station_row.Index] = pd.DataFrame(
            index=sec_date_series, data={"los_insar": los_insar}
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

    # Compare relative GPS and InSAR if reference station is provided
    if reference_station:
        compare_results = compare_relative_gps_insar(
            station_to_merged_df, reference_station=reference_station
        )
        print("Relative comparison results:")
        print(compare_results)
        compare_results.to_csv(output_dir / "relative_comparison.csv", index=False)

    typer.echo(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main()
