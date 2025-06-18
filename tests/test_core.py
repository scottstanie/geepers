from pathlib import Path

import pandas as pd

from geepers.core import main


def test_main(tmp_path):
    data_dir = Path(__file__).parent / "data"
    main(
        los_enu_file=data_dir / "hawaii_los_enu.tif",
        timeseries_files=sorted(data_dir.glob("displacement_*.tif")),
        output_dir=tmp_path / "GPS",
        compute_rates=False,
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
