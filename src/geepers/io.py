from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.windows
import xarray as xr
from affine import cached_property
from opera_utils import get_dates
from pyproj import Transformer
from rasterio.vrt import WarpedVRT

from ._types import PathOrStr

__all__ = ["RasterReader", "XarrayReader"]


if TYPE_CHECKING:
    from dolphin._types import Index


@dataclass
class RasterReader:
    """A single raster band of a GDAL-compatible dataset.

    See Also
    --------
    BinaryReader
    HDF5

    Notes
    -----
    The file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.

    """

    filename: PathOrStr
    """PathOrStr : The file path."""

    band: int
    """int : Band index (1-based)."""

    driver: str
    """str : Raster format driver name."""

    crs: rio.crs.CRS
    """rio.crs.CRS : The dataset's coordinate reference system."""

    transform: rio.transform.Affine
    """
    rasterio.transform.Affine : The dataset's georeferencing transformation matrix.

    This transform maps pixel row/column coordinates to coordinates in the dataset's
    coordinate reference system.
    """

    shape: tuple[int, int]
    dtype: np.dtype

    nodata: float | None = None
    """Optional[float] : Value to use for nodata pixels."""

    masked: bool = True
    """bool : If True, reads in data as a MaskedArray, masking nodata values."""

    @classmethod
    def from_file(
        cls,
        filename: PathOrStr,
        band: int = 1,
        nodata: float | None = None,
        **options,
    ) -> RasterReader:
        with rio.open(filename, "r", **options) as src:
            shape = (src.height, src.width)
            dtype = np.dtype(src.dtypes[band - 1])
            driver = src.driver
            crs = src.crs
            nodata = nodata or src.nodatavals[band - 1]
            transform = src.transform

            return cls(
                filename=filename,
                band=band,
                driver=driver,
                crs=crs,
                transform=transform,
                shape=shape,
                dtype=dtype,
                nodata=nodata,
            )

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """Int : Number of array dimensions."""
        return 2

    @cached_property
    def units(self) -> str:
        with rio.open(self.filename) as src:
            return src.units[self.band - 1]

    def __array__(self) -> np.ndarray:
        return self[:, :]

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        if key is ... or key == ():
            key = (slice(None), slice(None))

        if not isinstance(key, tuple):
            msg = "Index must be a tuple of slices or integers."
            raise TypeError(msg)

        r_slice, c_slice = _ensure_slices(*key[-2:])
        window = rasterio.windows.Window.from_slices(
            r_slice,
            c_slice,
            height=self.shape[0],
            width=self.shape[1],
        )

        with rio.open(self.filename) as src:
            out = src.read(self.band, window=window, masked=True)
        # Note that Rasterio doesn't use the `step` of a slice, so we need to
        # manually slice the output array.
        r_step, c_step = r_slice.step or 1, c_slice.step or 1
        o = out[::r_step, ::c_step]
        return np.ma.squeeze(o)

    def read_lon_lat(self, lons, lats) -> np.ndarray:
        """Get pixel values from a raster file for given longitudes and latitudes.

        Parameters
        ----------
        raster_path : str
            Path to the raster file.
        lons : float
            Longitude of the points.
        lats : float
            Latitude of the points.
        masked : bool, optional
            If True, reads in data as a MaskedArray, masking nodata values.
            Default is False.

        Returns
        -------
        np.ndarray
            pixel_values at the nearest point

        """
        with ExitStack() as stack:
            src = stack.enter_context(rio.open(self.filename))

            lon_list = [lons] if np.isscalar(lons) else lons
            lat_list = [lats] if np.isscalar(lats) else lats

            with WarpedVRT(src, crs="EPSG:4326") as vrt:
                return np.array(
                    list(
                        vrt.sample(
                            xy=zip(lon_list, lat_list, strict=False), masked=self.masked
                        )
                    )
                )

    def read_window(
        self,
        lon: float,
        lat: float,
        buffer_pixels: int = 0,
        op=round,
    ):
        """Get a window of pixel values for a given longitude and latitude.

        Parameters
        ----------
        lon : float
            Longitude of the central point.
        lat : float
            Latitude of the central point.
        buffer_pixels : int
            Number of pixels to buffer around the central point.
        op : callable, optional
            Operation to use when calculating the central pixel. Default is round.
            Options are "round", "math.floor", "math.ceil"
        masked : bool, optional
            If True, reads in data as a MaskedArray, masking nodata values.
            Default is False.

        Returns
        -------
        np.ndarray
            Window of pixel values around the specified point.

        """
        with ExitStack() as stack:
            src = stack.enter_context(rio.open(self.filename))

            # Transform the lon/lat to the raster's CRS
            x, y = rio.warp.transform("EPSG:4326", src.crs, [lon], [lat])

            # Get the row and column of the central pixel
            row, col = src.index(x[0], y[0], op=op)

            # Calculate the window boundaries
            window = rasterio.windows.Window(
                col - buffer_pixels,
                row - buffer_pixels,
                2 * buffer_pixels + 1,
                2 * buffer_pixels + 1,
            )

            return src.read(self.band, window=window, masked=self.masked)


@dataclass
class XarrayReader:
    """A wrapper for an `xarray.DataArray` with georeferencing/windowed reading."""

    da: xr.DataArray

    def __post_init__(self):
        # Check that the DataArray has the required coordinates
        if self.ndim == 3 and "time" not in self.da.coords:
            msg = "DataArray must have 'time' coordinate."
            raise ValueError(msg)

        # Normalize so we have 'x' and 'y' coordinates
        if "lon" in self.da.coords:
            self.da = self.da.rename({"lon": "x"})
            if self.crs is not None and self.crs != "EPSG:4326":
                msg = "CRS is not EPSG:4326, but 'lon' coordinate is present."
                raise ValueError(msg)
            self.da.rio.write_crs("EPSG:4326", inplace=True)
        if "lat" in self.da.coords:
            self.da = self.da.rename({"lat": "y"})

        if "x" not in self.da.coords or "y" not in self.da.coords:
            msg = "DataArray must have 'x' and 'y' coordinates."
            raise ValueError(msg)

    @classmethod
    def from_stack_file(
        cls,
        filename: Path | str,
        data_var: str,
        engine: str | None = None,
        nodata: float | None = None,
        crs: rio.crs.CRS | None = None,
        units: str | None = None,
    ) -> Self:
        da = xr.open_dataset(filename, engine=engine)[data_var]
        # Apply overrides, if given
        if crs is not None:
            da.rio.write_crs(crs, inplace=True)
        if nodata is not None:
            da.rio.write_nodata(nodata, inplace=True)
        if units is not None:
            da.attrs["units"] = units
        return cls(da)

    @classmethod
    def from_geotiff_files(
        cls,
        file_list: Sequence[Path | str],
        file_date_fmt: str = "%Y%m%d",
        file_date_idx: int = 0,
    ) -> Self:
        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            """Preprocess individual dataset when loading with open_mfdataset."""
            fname = ds.encoding["source"]
            date = get_dates(fname, fmt=file_date_fmt)[file_date_idx]
            if len(ds.band) == 1:
                ds = ds.sel(band=ds.band[0]).drop_vars("band")
            return ds.expand_dims(time=[pd.to_datetime(date)])

        ds = xr.open_mfdataset(
            sorted(file_list), engine="rasterio", preprocess=preprocess
        )
        return cls(ds.band_data)

    @property
    def ndim(self):
        return self.da.ndim

    @property
    def shape(self):
        return self.da.shape

    @property
    def dtype(self):
        return self.da.dtype

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        return self.da[key].values

    @property
    def crs(self):
        return self.da.rio.crs()

    def read_lon_lat(self, lons: Sequence[float], lats: Sequence[float]) -> np.ndarray:
        if self.crs != "EPSG:4326":
            x, y = self._transformer_from_lonlat.transform(lons, lats)
        else:
            x, y = np.asarray(lons), np.asarray(lats)
        return self.da.sel(x=x, y=y, method="nearest").values

    def read_window(self, lon: float, lat: float, buffer_pixels: int = 0, op=round):
        if self.crs != "EPSG:4326":
            x, y = self._transformer_from_lonlat.transform(lon, lat)
        else:
            x, y = lon, lat
        x_int, y_int = op(x), op(y)
        x_slice = slice(x_int - buffer_pixels, x_int + buffer_pixels + 1)
        y_slice = slice(y_int - buffer_pixels, y_int + buffer_pixels + 1)
        return self.da.sel(x=x_slice, y=y_slice, method="nearest").values

    @cached_property
    def _transformer_from_lonlat(self):
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    @property
    def units(self) -> str:
        return self.da.units


def _ensure_slices(rows: Index, cols: Index) -> tuple[slice, slice]:
    def _parse(key: Index):
        if isinstance(key, int):
            return slice(key, key + 1)
        elif key is ...:
            return slice(None)
        else:
            return key

    return _parse(rows), _parse(cols)
