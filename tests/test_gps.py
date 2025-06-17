"""Tests for the GPS module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
import requests

import geepers.gps as gps


class TestDownloadStationData:
    """Tests for download_station_data function."""

    @pytest.mark.vcr
    def test_download_enu_data(self, tmp_path, monkeypatch):
        """Test downloading ENU data for a station."""
        with monkeypatch.context() as m:
            m.setattr(gps, "GPS_DIR", tmp_path)
            # This will make a real HTTP request (recorded by VCR)
            gps.download_station_data("P123", coords="enu")

            expected_file = Path(tmp_path) / "P123.tenv3"
            assert expected_file.exists()

            # Check file has content
            content = expected_file.read_text()
            assert len(content) > 0
            assert "YYMMMDD" in content  # Header should be present

    @pytest.mark.vcr
    def test_download_xyz_data(self, tmp_path, monkeypatch):
        """Test downloading XYZ data for a station."""
        with monkeypatch.context() as m:
            m.setattr(gps, "GPS_DIR", tmp_path)
            gps.download_station_data("P123", coords="xyz")

            expected_file = Path(tmp_path) / "P123.txyz2"
            assert expected_file.exists()

    def test_download_invalid_coords(self):
        """Test error for invalid coordinate system."""
        with pytest.raises(ValueError, match="coords must be either 'enu' or 'xyz'"):
            gps.download_station_data("P123", coords="invalid")

    @pytest.mark.vcr
    def test_download_nonexistent_station(self, tmp_path, monkeypatch):
        """Test error when downloading nonexistent station."""
        with monkeypatch.context() as m:
            m.setattr(gps, "GPS_DIR", tmp_path)
            with pytest.raises(requests.HTTPError):
                gps.download_station_data("NONEXISTENT", coords="enu")


@pytest.fixture
def station_df() -> pd.DataFrame:
    """A minimal station list used throughout the tests (CRIM & OUTL)."""
    return pd.DataFrame(
        {
            "name": ["CRIM", "OUTL"],
            "lat": [19.395, 19.387],
            "lon": [-155.274, -155.281],
            "alt": [1147.608, 1103.498],
        }
    )


@pytest.fixture
def station_gdf(station_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """GeoDataFrame version of :func:`station_df`."""
    return gpd.GeoDataFrame(
        station_df,
        geometry=gpd.points_from_xy(station_df.lon, station_df.lat),
        crs="EPSG:4326",
    )


class TestGetStationsWithinImage:
    @pytest.fixture
    def mock_src(
        self,
        bounds=(-156.5, 18.0, -154.0, 20.0),
        crs="EPSG:4326",
        index_side_effect=None,
        read_side_effect=None,
    ):
        m = Mock()
        m.bounds = bounds
        m.crs = crs
        if index_side_effect is not None:
            m.index.side_effect = index_side_effect
        if read_side_effect is not None:
            m.read.side_effect = read_side_effect
        return m

    def test_stations_within_bounds(self, station_gdf, mock_src):
        """Both stations fall inside the mocked raster bounds."""
        with (
            patch("geepers.gps.read_station_llas", return_value=station_gdf),
            patch("rasterio.open") as mock_rio,
        ):
            mock_rio.return_value.__enter__.return_value = mock_src

            result = gps.get_stations_within_image("dummy.tif", mask_invalid=False)

            assert len(result) == 2
            assert set(result.name) == {"CRIM", "OUTL"}

    def test_exclude_stations(self, station_gdf, mock_src):
        """Explicitly exclude OUTL from the returned GeoDataFrame."""
        with (
            patch("geepers.gps.read_station_llas", return_value=station_gdf),
            patch("rasterio.open") as mock_rio,
        ):
            mock_rio.return_value.__enter__.return_value = mock_src

            result = gps.get_stations_within_image(
                "dummy.tif", mask_invalid=False, exclude_stations=["OUTL"]
            )

            assert len(result) == 1
            assert result.name.iloc[0] == "CRIM"
