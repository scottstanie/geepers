"""Tests for the CLI module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from geepers.cli import (
    _convert_to_meters,
    compare_relative_gps_insar,
    create_tidy_df,
    main,
    process_insar_data,
)


class TestCreateTidyDf:
    """Tests for the create_tidy_df function."""

    def test_single_station(self):
        """Test creating tidy dataframe from single station."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"los_gps": [1.0, 2.0, 3.0], "los_insar": [1.1, 2.1, 3.1]}, index=dates
        )
        station_dict = {"STAT1": df}

        result = create_tidy_df(station_dict)

        expected_rows = 6  # 2 measurements × 3 dates × 1 station
        assert len(result) == expected_rows
        assert list(result.columns) == ["station", "date", "measurement", "value"]
        assert result["station"].unique().tolist() == ["STAT1"]
        assert sorted(result["measurement"].unique()) == ["los_gps", "los_insar"]

    def test_multiple_stations(self):
        """Test creating tidy dataframe from multiple stations."""
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        df1 = pd.DataFrame(
            {"los_gps": [1.0, 2.0], "los_insar": [1.1, 2.1]}, index=dates
        )
        df2 = pd.DataFrame(
            {"los_gps": [3.0, 4.0], "los_insar": [3.1, 4.1]}, index=dates
        )
        station_dict = {"STAT1": df1, "STAT2": df2}

        result = create_tidy_df(station_dict)

        expected_rows = 8  # 2 measurements × 2 dates × 2 stations
        assert len(result) == expected_rows
        assert sorted(result["station"].unique()) == ["STAT1", "STAT2"]


class TestCompareRelativeGpsInsar:
    """Tests for the compare_relative_gps_insar function."""

    def test_valid_reference_station(self):
        """Test relative comparison with valid reference station."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        ref_df = pd.DataFrame(
            {"los_gps": [0.0, 1.0, 2.0], "los_insar": [0.1, 1.1, 2.1]}, index=dates
        )
        test_df = pd.DataFrame(
            {"los_gps": [1.0, 2.0, 3.0], "los_insar": [1.2, 2.2, 3.2]}, index=dates
        )
        station_dict = {"REF": ref_df, "TEST": test_df}

        result = compare_relative_gps_insar(station_dict, reference_station="REF")

        assert len(result) == 6  # 2 stations × 3 dates
        assert "relative_gps" in result.columns
        assert "relative_insar" in result.columns
        assert "difference" in result.columns

        # Check that reference station has zero relative displacement
        ref_rows = result[result["station"] == "REF"]
        assert np.allclose(ref_rows["relative_gps"], 0.0)
        assert np.allclose(ref_rows["relative_insar"], 0.0)

    def test_missing_reference_station(self):
        """Test error when reference station is missing."""
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        df = pd.DataFrame({"los_gps": [1.0, 2.0], "los_insar": [1.1, 2.1]}, index=dates)
        station_dict = {"STAT1": df}

        with pytest.raises(ValueError, match="Reference station 'MISSING' not found"):
            compare_relative_gps_insar(station_dict, reference_station="MISSING")

    def test_no_common_epochs(self):
        """Test handling when stations have no common epochs."""
        dates1 = pd.date_range("2020-01-01", periods=2, freq="D")
        dates2 = pd.date_range("2020-01-05", periods=2, freq="D")
        df1 = pd.DataFrame(
            {"los_gps": [1.0, 2.0], "los_insar": [1.1, 2.1]}, index=dates1
        )
        df2 = pd.DataFrame(
            {"los_gps": [3.0, 4.0], "los_insar": [3.1, 4.1]}, index=dates2
        )
        station_dict = {"REF": df1, "TEST": df2}

        with patch("geepers.cli.logger") as mock_logger:
            result = compare_relative_gps_insar(station_dict, reference_station="REF")
            mock_logger.warning.assert_called()

        # Should only have reference station data
        assert len(result[result["station"] == "TEST"]) == 0


class TestConvertToMeters:
    """Tests for the _convert_to_meters function."""

    def test_radians_conversion(self):
        """Test conversion from radians to meters."""
        with patch("geepers.cli.get_raster_units", return_value="radians"):
            arr = np.array([np.pi, 2 * np.pi])
            result = _convert_to_meters("dummy.tif", arr)

            expected = arr * (0.05546576 / (4 * np.pi))  # Sentinel-1 wavelength
            np.testing.assert_array_almost_equal(result, expected)

    def test_meters_passthrough(self):
        """Test that meters are passed through unchanged."""
        with patch("geepers.cli.get_raster_units", return_value="meters"):
            arr = np.array([1.0, 2.0, 3.0])
            result = _convert_to_meters("dummy.tif", arr)
            np.testing.assert_array_equal(result, arr)

    def test_unknown_units_default_to_meters(self):
        """Test that unknown units default to meters with debug log."""
        with patch("geepers.cli.get_raster_units", return_value="unknown"):
            with patch("geepers.cli.logger") as mock_logger:
                arr = np.array([1.0, 2.0, 3.0])
                result = _convert_to_meters("dummy.tif", arr)
                np.testing.assert_array_equal(result, arr)
                mock_logger.debug.assert_called()


class TestProcessInsarData:
    """Tests for the process_insar_data function."""

    @pytest.fixture
    def sample_gps_stations(self):
        """Create sample GPS stations dataframe."""
        return pd.DataFrame(
            {"lon": [-118.0, -119.0], "lat": [34.0, 35.0]}, index=["STAT1", "STAT2"]
        )

    @pytest.fixture
    def mock_raster_reader(self):
        """Create mock raster stack reader."""
        mock_reader = Mock()
        mock_reader.file_list = [
            "20200101_20200113.tif",
            "20200101_20200125.tif",
        ]
        # Mock displacement data: (n_dates, n_stations)
        mock_reader.read_lon_lat.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        return mock_reader

    def test_basic_processing(self, sample_gps_stations, mock_raster_reader):
        """Test basic InSAR data processing."""
        with (
            patch(
                "geepers.cli._convert_to_meters",
                return_value=np.array([[0.1, 0.2], [0.3, 0.4]]),
            ),
            patch("geepers.cli.get_dates") as mock_get_dates,
        ):
            mock_get_dates.side_effect = [
                ("20200101", "20200113"),
                ("20200101", "20200125"),
            ]

            result = process_insar_data(
                reader=mock_raster_reader,
                df_gps_stations=sample_gps_stations,
            )

            assert len(result) == 2  # Two stations
            assert "STAT1" in result
            assert "STAT2" in result

            for station_data in result.values():
                assert "los_insar" in station_data.columns
                assert len(station_data) == 2  # Two dates

    def test_with_temporal_coherence(self, sample_gps_stations, mock_raster_reader):
        """Test processing with temporal coherence data."""
        mock_temp_coh_reader = Mock()
        mock_temp_coh_reader.read_lon_lat.return_value = np.array([0.8, 0.9])

        with (
            patch(
                "geepers.cli._convert_to_meters",
                return_value=np.array([[0.1, 0.2], [0.3, 0.4]]),
            ),
            patch("geepers.cli.get_dates") as mock_get_dates,
        ):
            mock_get_dates.side_effect = [
                ("20200101", "20200113"),
                ("20200101", "20200125"),
            ]

            result = process_insar_data(
                reader=mock_raster_reader,
                df_gps_stations=sample_gps_stations,
                reader_temporal_coherence=mock_temp_coh_reader,
            )

            for station_data in result.values():
                assert "temporal_coherence" in station_data.columns


class TestMainFunction:
    """Tests for the main function."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_files(self, temp_output_dir):
        """Create sample input files."""
        ts_files = []
        for _i, date in enumerate(["20200113", "20200125"]):
            f = temp_output_dir / f"20200101_{date}.tif"
            f.touch()
            ts_files.append(f)

        los_file = temp_output_dir / "los_enu.tif"
        los_file.touch()

        return ts_files, los_file

    def test_main_integration_mock(self, sample_files, temp_output_dir):
        """Test main function with mocked dependencies."""
        ts_files, los_file = sample_files

        # Mock all the heavy dependencies
        with patch("geepers.cli.RasterStackReader"):
            with patch("geepers.cli.RasterReader") as mock_reader:
                with patch(
                    "geepers.cli.geepers.gps.get_stations_within_image"
                ) as mock_get_stations:
                    with patch("geepers.cli.thread_map") as mock_thread_map:
                        with patch("geepers.cli.get_dates") as mock_get_dates:
                            with patch(
                                "geepers.cli.process_insar_data"
                            ) as mock_process_insar:
                                with patch(
                                    "geepers.cli.geepers.rates.calculate_rates"
                                ) as mock_calc_rates:
                                    # Set up mocks
                                    mock_get_stations.return_value = pd.DataFrame(
                                        {"lon": [-118.0], "lat": [34.0]},
                                        index=["STAT1"],
                                    )
                                    mock_thread_map.return_value = [
                                        pd.DataFrame(
                                            {"los_gps": [1.0, 2.0]},
                                            index=pd.date_range(
                                                "2020-01-01", periods=2
                                            ),
                                        )
                                    ]
                                    mock_get_dates.side_effect = [
                                        ("20200101", "20200113"),
                                        ("20200101", "20200125"),
                                    ]
                                    mock_process_insar.return_value = {
                                        "STAT1": pd.DataFrame(
                                            {"los_insar": [1.1, 2.1]},
                                            index=pd.date_range(
                                                "2020-01-13", periods=2
                                            ),
                                        )
                                    }
                                    mock_calc_rates.return_value = pd.DataFrame(
                                        {"rate": [1.0]}, index=["STAT1"]
                                    )

                                    # Mock raster reader
                                    mock_los_reader = Mock()
                                    mock_los_reader.read_lon_lat.return_value = (
                                        np.array([0.1, 0.2, 0.3])
                                    )
                                    mock_reader.from_file.return_value = mock_los_reader

                                    # Run main function
                                    main(
                                        timeseries_files=ts_files,
                                        los_enu_file=los_file,
                                        output_dir=temp_output_dir,
                                    )

                                    # Check that output files were created
                                    assert (
                                        temp_output_dir / "combined_data.csv"
                                    ).exists()
                                    assert (
                                        temp_output_dir / "station_summary.csv"
                                    ).exists()

    def test_main_with_reference_station(self, sample_files, temp_output_dir):
        """Test main function with reference station."""
        ts_files, los_file = sample_files

        with patch("geepers.cli.RasterStackReader"):
            with patch("geepers.cli.RasterReader"):
                with patch(
                    "geepers.cli.geepers.gps.get_stations_within_image"
                ) as mock_get_stations:
                    with patch("geepers.cli.thread_map"):
                        with patch("geepers.cli.get_dates") as mock_get_dates:
                            with patch("geepers.cli.process_insar_data"):
                                with patch("geepers.cli.geepers.rates.calculate_rates"):
                                    with patch(
                                        "geepers.cli.compare_relative_gps_insar"
                                    ) as mock_compare:
                                        mock_get_stations.return_value = pd.DataFrame(
                                            {"lon": [-118.0], "lat": [34.0]},
                                            index=["STAT1"],
                                        )
                                        mock_get_dates.side_effect = [
                                            ("20200101", "20200113"),
                                            ("20200101", "20200125"),
                                        ]
                                        mock_compare.return_value = pd.DataFrame(
                                            {
                                                "station": ["STAT1"],
                                                "relative_gps": [0.0],
                                                "relative_insar": [0.0],
                                                "difference": [0.0],
                                            }
                                        )

                                        main(
                                            timeseries_files=ts_files,
                                            los_enu_file=los_file,
                                            output_dir=temp_output_dir,
                                            reference_station="STAT1",
                                        )

                                        # Check that relative comparison file was created
                                        assert (
                                            temp_output_dir / "relative_comparison.csv"
                                        ).exists()
                                        mock_compare.assert_called_once()

    def test_main_multiple_reference_dates_error(self, sample_files, temp_output_dir):
        """Test main function raises error with multiple reference dates."""
        ts_files, los_file = sample_files

        with patch("geepers.cli.RasterStackReader"):
            with patch("geepers.cli.get_dates") as mock_get_dates:
                # Mock different reference dates
                mock_get_dates.side_effect = [
                    ("20200101", "20200113"),
                    ("20200201", "20200125"),  # Different reference date
                ]

                with pytest.raises(
                    ValueError, match="Multiple reference dates detected"
                ):
                    main(
                        timeseries_files=ts_files,
                        los_enu_file=los_file,
                        output_dir=temp_output_dir,
                    )
