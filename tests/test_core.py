from pathlib import Path

import pandas as pd

from geepers.core import main


def test_main(tmp_path):
    data_dir = Path(__file__).parent / "data"
    main(
        los_enu_file=data_dir / "hawaii_los_enu.tif",
        timeseries_files=sorted(data_dir.glob("displacement_*.tif")),
        output_dir=tmp_path / "GPS",
    )
    assert (tmp_path / "GPS").exists()

    pd.read_csv(tmp_path / "GPS" / "relative_gps_insar.csv").head()
