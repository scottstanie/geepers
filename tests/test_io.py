"""Tests for the IO module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401 - needed for rio accessor
import xarray as xr

from geepers.io import XarrayReader


class TestXarrayReader:
    """Tests for XarrayReader class."""

    @pytest.fixture
    def dataarray_2d(self):
        data = np.random.rand(10, 10)
        da = xr.DataArray(
            data,
            coords={
                "y": np.linspace(0, 10, 10),
                "x": np.linspace(0, 10, 10),
            },
            dims=["y", "x"],
            attrs={"units": "meters"},
        )
        # Set CRS
        da.rio.write_crs("EPSG:4326", inplace=True)
        return da

    @pytest.fixture
    def dataarray_3d(self):
        data = np.random.rand(3, 10, 10)
        times = [
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-02"),
            np.datetime64("2020-01-03"),
        ]
        da = xr.DataArray(
            data,
            coords={
                "time": times,
                "y": np.linspace(0, 10, 10),
                "x": np.linspace(0, 10, 10),
            },
            dims=["time", "y", "x"],
            attrs={"units": "meters"},
        )
        # Set CRS
        da.rio.write_crs("EPSG:4326", inplace=True)
        return da

    def test_reader_2d_creation(self, dataarray_2d):
        """Test creating a 2D XarrayReader."""
        reader = XarrayReader(dataarray_2d)
        assert reader.ndim == 2
        assert reader.shape == (10, 10)
        assert reader.crs == "EPSG:4326"

    def test_reader_3d_creation(self, dataarray_3d):
        """Test creating a 3D XarrayReader with time dimension."""
        reader = XarrayReader(dataarray_3d)
        assert reader.ndim == 3
        assert reader.shape == (3, 10, 10)
        assert reader.crs == "EPSG:4326"
        assert "time" in reader.da.coords

    def test_reader_coordinate_normalization(self):
        """Test that lon/lat coordinates are normalized to x/y."""
        data = np.random.rand(10, 10)
        da = xr.DataArray(
            data,
            coords={
                "lat": np.linspace(0, 10, 10),
                "lon": np.linspace(0, 10, 10),
            },
            dims=["lat", "lon"],
            attrs={"units": "meters"},
        )

        reader = XarrayReader(da)
        assert "x" in reader.da.coords
        assert "y" in reader.da.coords
        assert "lat" not in reader.da.coords
        assert "lon" not in reader.da.coords
        assert reader.crs == "EPSG:4326"

    def test_reader_missing_coordinates_fails(self):
        """Test that DataArray without x/y coordinates fails."""
        data = np.random.rand(10, 10)
        da = xr.DataArray(
            data,
            coords={
                "a": np.linspace(0, 10, 10),
                "b": np.linspace(0, 10, 10),
            },
            dims=["a", "b"],
            attrs={"units": "meters"},
        )
        da.rio.write_crs("EPSG:4326", inplace=True)

        with pytest.raises(
            ValueError, match="DataArray must have 'x' and 'y' coordinates"
        ):
            XarrayReader(da)

    def test_reader_missing_crs_fails(self):
        """Test that DataArray without CRS fails."""
        # Create a DataArray without CRS
        data = np.random.rand(10, 10)
        da = xr.DataArray(
            data,
            coords={
                "y": np.linspace(0, 10, 10),
                "x": np.linspace(0, 10, 10),
            },
            dims=["y", "x"],
            attrs={"units": "meters"},
        )
        # Don't set CRS

        with pytest.raises(ValueError, match="CRS is not set"):
            XarrayReader(da)

    def test_reader_missing_units_fails(self, dataarray_2d):
        """Test that DataArray without units fails."""
        # Remove units
        del dataarray_2d.attrs["units"]

        with pytest.raises(ValueError, match="Units are not set"):
            XarrayReader(dataarray_2d)

    def test_from_file_list(self, tmp_path, dataarray_3d):
        """Test basic from_file_list functionality."""

        files = [tmp_path / "20200101_20200102.tif", tmp_path / "20200102_20200103.tif"]
        dataarray_3d[0].rio.to_raster(files[0])
        dataarray_3d[1].rio.to_raster(files[1])
        reader = XarrayReader.from_file_list(files)

        assert reader.ndim == 3
        assert "time" in reader.da.coords

    def test_from_zarr(self, tmp_path, dataarray_3d):
        """Test XarrayReader with a zarr stack that already has time dimension."""
        zarr_path = tmp_path / "test.zarr"
        dataarray_3d.to_zarr(zarr_path, consolidated=False)
        reader = XarrayReader.from_file(zarr_path)

        assert reader.ndim == 3
        assert reader.shape == (3, 10, 10)
        assert len(reader.da.coords["time"]) == 3
        assert reader.crs == "EPSG:4326"
        assert reader.da.attrs["units"] == "meters"

        # Test that time coordinates are preserved
        assert all(reader.da.coords["time"] == dataarray_3d.coords["time"])

    def test_from_range_file_list_basic(self, tmp_path, dataarray_2d):
        """Test basic from_range_file_list functionality."""
        # Create quality raster files with range-based naming
        quality_files = [
            tmp_path / "similarity_20160705_20170302.tif",
            tmp_path / "similarity_20170314_20171121.tif",
            tmp_path / "similarity_20171203_20180613.tif",
        ]

        # Create test rasters with different values for each file
        test_data = dataarray_2d.copy()
        for i, qf in enumerate(quality_files):
            test_data.values.fill(i + 1.0)  # Fill with 1.0, 2.0, 3.0 respectively
            test_data.rio.to_raster(qf)

        # Define target times that should map to these files
        target_times = [
            "2016-07-05",  # Should map to first file (20160705_20170302)
            "2016-10-01",
            "2017-03-14",  # Should map to second file (20170314_20171121)
            "2017-11-01",
            "2018-01-01",  # Should map to third file (20171203_20180613)
            "2018-02-01",
        ]

        reader = XarrayReader.from_range_file_list(
            quality_files, target_times, units="similarity"
        )

        assert reader.ndim == 3
        assert reader.shape == (6, 10, 10)
        assert "time" in reader.da.coords
        assert len(reader.da.coords["time"]) == 6
        assert reader.da.attrs["units"] == "similarity"

        # Verify that each time slice has the expected value
        assert np.all(reader.da[0].values == 1.0)  # First target time -> first file
        assert np.all(reader.da[1].values == 1.0)  # Second target time -> first file
        assert np.all(reader.da[2].values == 2.0)  # Third target time -> second file
        assert np.all(reader.da[3].values == 2.0)  # Fourth target time -> second file
        assert np.all(reader.da[4].values == 3.0)  # Fifth target time -> third file
        assert np.all(reader.da[5].values == 3.0)  # Sixth target time -> third file

    def test_from_range_file_list_no_matching_epochs(self, tmp_path, dataarray_2d):
        """Test error when no files cover any requested epochs."""
        quality_files = [
            tmp_path / "similarity_20160101_20161231.tif",
            tmp_path / "similarity_20170101_20171231.tif",
        ]

        # Create test rasters
        test_data = dataarray_2d.copy()
        for qf in quality_files:
            test_data.rio.to_raster(qf)

        # Target times that don't overlap with any file ranges
        target_times = [
            "2015-06-15",  # Before any file ranges
            "2018-06-15",  # After any file ranges
        ]

        with pytest.raises(
            ValueError, match="None of the files cover any requested epoch"
        ):
            XarrayReader.from_range_file_list(
                quality_files, target_times, units="coherence"
            )

    def test_from_range_file_list_with_timeseries_reader(self, tmp_path, dataarray_3d):
        """Test using target_times from an existing timeseries reader."""
        # Create a timeseries reader first
        timeseries_files = [
            tmp_path / "20160927_20161009.tif",
            tmp_path / "20161009_20161021.tif",
            tmp_path / "20161021_20161102.tif",
        ]

        for i, tf in enumerate(timeseries_files):
            dataarray_3d[i].rio.to_raster(tf)

        timeseries_reader = XarrayReader.from_file_list(
            timeseries_files, file_date_fmt="%Y%m%d", file_date_idx=1
        )

        # Create quality raster that covers the timeseries period
        quality_files = [tmp_path / "similarity_20160901_20161130.tif"]
        quality_data = dataarray_3d[0].copy()
        quality_data.values.fill(0.8)
        quality_data.rio.to_raster(quality_files[0])

        # Use timeseries times as target times
        quality_reader = XarrayReader.from_range_file_list(
            quality_files, timeseries_reader.da.coords["time"], units="coherence"
        )

        assert quality_reader.shape == timeseries_reader.shape
        assert np.all(quality_reader.da.values == 0.8)

        # Time coordinates should match
        assert all(
            quality_reader.da.coords["time"] == timeseries_reader.da.coords["time"]
        )

    def test_read_window_oob_nan(self, dataarray_2d):
        reader = XarrayReader(dataarray_2d)
        out = reader.read_window([-999], [-999], buffer_pixels=1)[0]
        # Window is 3x3, all NaN
        assert out.shape == (3, 3)
        assert np.isnan(out.values).all()

    def test_read_window_mixed_in_and_oob(self, dataarray_2d):
        reader = XarrayReader(dataarray_2d)
        lons = [5, -999]
        lats = [5, -999]
        wins = reader.read_window(lons, lats, buffer_pixels=0)
        assert len(wins) == 2
        assert not np.isnan(wins[0]).all()  # valid point
        assert np.isnan(wins[1]).all()  # OOB -> NaNs


class TestXarrayRealData:
    def test_read_lon_lat(self):
        test_los_enu_file = Path(__file__).parent / "data/hawaii/hawaii_los_enu.tif"
        expected_los_enu = [-0.674805, -0.12548828, 0.72753906]
        reader = XarrayReader.from_file(test_los_enu_file, units="unitless")
        los_enu = reader.read_lon_lat(-155, 20)[0].values.squeeze()
        assert np.allclose(los_enu, expected_los_enu)
