import zipfile
from pathlib import Path

import pandas as pd

import geepers.gps as gps
from geepers.core import main


def test_main(tmp_path, monkeypatch):
    data_dir = Path(__file__).parent / "data/hawaii"
    unr_data_zipped = Path(__file__).parent / "data/unr.zip"
    # unzip, and set to GPS dir:
    with zipfile.ZipFile(unr_data_zipped, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    monkeypatch.setattr(gps, "GPS_DIR", tmp_path)

    main(
        los_enu_file=data_dir / "hawaii_los_enu.tif",
        timeseries_files=sorted(data_dir.glob("displacement_*.tif")),
        output_dir=tmp_path / "GPS",
        compute_rates=True,
    )
    assert (tmp_path / "GPS").exists()

    df = pd.read_csv(tmp_path / "GPS" / "combined_data.csv")
    expected_stations = [
        "HLNA",
        "MANE",
        "KOSM",
        "AHUP",
        "OUTL",
        "CNPK",
        "CRIM",
    ]
    assert set(df.station) == set(expected_stations)
    expected_entry = {
        "station": "HLNA",
        "date": "2016-07-23",
        "measurement": "los_gps",
        "value": 0.0127341801257807,
    }
    pd.testing.assert_series_equal(
        df[df.station == "HLNA"].iloc[0], pd.Series(expected_entry, name=0)
    )
