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


class TestXarrayRealData:
    def test_read_lon_lat(self):
        test_los_enu_file = Path(__file__).parent / "data/hawaii/hawaii_los_enu.tif"
        expected_los_enu = [-0.6738281, -0.12548828, 0.72753906]
        reader = XarrayReader.from_file(test_los_enu_file, units="unitless")
        sample_point = [-155, 20]
        los_enu = reader.read_lon_lat(*sample_point).values
        assert np.allclose(los_enu, expected_los_enu)
