from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.windows
import xarray as xr
from affine import cached_property
from numpy.typing import ArrayLike
from opera_utils import get_dates
from pyproj import Transformer
from rasterio.vrt import WarpedVRT
from tqdm.contrib.concurrent import thread_map

from ._types import PathOrStr

__all__ = ["DatasetReader", "RasterReader", "RasterStackReader", "StackReader"]


if TYPE_CHECKING:
    from dolphin._types import Index


@runtime_checkable
class DatasetReader(Protocol):
    """An array-like interface for reading input datasets.

    `DatasetReader` defines the abstract interface that types must conform to in order
    to be read by functions which iterate in blocks over the input data.
    Such objects must export NumPy-like `dtype`, `shape`, and `ndim` attributes,
    and must support NumPy-style slice-based indexing.

    Note that this protocol allows objects to be passed to `dask.array.from_array`
    which needs `.shape`, `.ndim`, `.dtype` and support numpy-style slicing.
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __getitem__(self, key: tuple[Index, ...], /) -> ArrayLike:
        """Read a block of data."""
        ...

    def read_window(
        self,
        lon: float,
        lat: float,
        buffer_pixels: int = 0,
        op=round,
    ) -> np.ndarray:
        """Read in pixel values in a spatial window centered at `lon`, `lat`."""
        ...

    def read_lon_lat(self, lons, lats) -> np.ndarray:
        """Read in raster values located at `lons`, `lats`."""
        ...


@runtime_checkable
class StackReader(DatasetReader, Protocol):
    """An array-like interface for reading a 3D stack of input datasets.

    `StackReader` defines the abstract interface that types must conform to in order
    to be valid inputs to be read in functions like [dolphin.ps.create_ps][].
    It is a specialization of [DatasetReader][] that requires a 3D shape.
    """

    ndim: int = 3
    """int : Number of array dimensions."""

    shape: tuple[int, int, int]
    """tuple of int : Tuple of array dimensions."""

    def __len__(self) -> int:
        """Int : Number of images in the stack."""
        return self.shape[0]

    def read_lon_lat(self, lons, lats) -> np.ndarray:
        """Read in raster values located at `lons`, `lats`."""
        ...

    def read_window(
        self,
        lon: float,
        lat: float,
        buffer_pixels: int = 0,
        op=round,
    ) -> np.ndarray:
        """Read in pixel values in a spatial window centered at `lon`, `lat`."""
        ...


@dataclass
class RasterReader(DatasetReader):
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


def _read_3d(
    key: tuple[Index, ...], readers: Sequence[DatasetReader], num_threads: int = 1
):
    bands, r_slice, c_slice = _unpack_3d_slices(key)

    if isinstance(bands, slice):
        # convert the bands to -1-indexed list
        total_num_bands = len(readers)
        band_idxs = list(range(*bands.indices(total_num_bands)))
    elif isinstance(bands, int):
        band_idxs = [bands]
    else:
        msg = "Band index must be an integer or slice."
        raise TypeError(msg)

    # Get only the bands we need
    if num_threads == 1:
        out = np.stack([readers[i][r_slice, c_slice] for i in band_idxs], axis=0)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(lambda i: readers[i][r_slice, c_slice], band_idxs)
        out = np.stack(list(results), axis=0)

    # TODO: Do i want a "keep_dims" option to not collapse singleton dimensions?
    return np.squeeze(out)


@dataclass
class XarrayStackReader(StackReader):
    """A stack of datasets for any GDAL-readable rasters."""

    da: xr.DataArray

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

        ds = xr.open_mfdataset(sorted(file_list), preprocess=preprocess)
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

    def read_lon_lat(self, lons, lats):
        if self.crs != "EPSG:4326":
            with WarpedVRT(self.da, crs="EPSG:4326") as vrt:
                return vrt.sel(x=lons, y=lats, method="nearest").values
        return self.da.sel(x=lons, y=lats, method="nearest").values

    @cached_property
    def _transformer_from_lonlat(self):
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def _get_point_values(self, lon: float, lat: float) -> np.ndarray:
        """Get point values for a dataset at lon/lat."""
        x, y = self._transformer_from_lonlat.transform(lon, lat)
        point_data = self.da.sel(x=x, y=y, method="nearest")
        return np.atleast_1d(point_data.values)


@dataclass
class RasterStackReader(StackReader):
    """A stack of datasets for any GDAL-readable rasters."""

    file_list: Sequence[PathOrStr]
    readers: Sequence[DatasetReader]
    num_threads: int = 1
    nodata: float | None = None
    masked: bool = True

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        return _read_3d(key, self.readers, num_threads=self.num_threads)

    @property
    def shape_2d(self):
        return self.readers[0].shape

    @property
    def shape(self):
        return (len(self.file_list), *self.shape_2d)

    @property
    def dtype(self):
        return self.readers[0].dtype

    @classmethod
    def from_file_list(
        cls,
        file_list: Sequence[PathOrStr],
        bands: int | Sequence[int] = 1,
        keep_open: bool = False,
        num_threads: int = 1,
        nodata: float | None = None,
        masked: bool = True,
    ) -> RasterStackReader:
        """Create a RasterStackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[PathOrStr]
            List of paths to the files to read.
        bands : int | Sequence[int]
            Band to read from each file.
            If a single int, will be used for all files.
            Default = 1.
        keep_open : bool, optional (default False)
            If True, keep the rasterio file handles open for faster reading.
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata : float, optional
            Manually set value to use for nodata pixels, by default None
        masked : bool, optional
            If True, reads in data as a MaskedArray, masking nodata values.
            Default is True.

        Returns
        -------
        RasterStackReader
            The RasterStackReader object.

        """
        if isinstance(bands, int):
            bands = [bands] * len(file_list)

        readers = [
            RasterReader.from_file(f, band=b, keep_open=keep_open, masked=masked)
            for (f, b) in zip(file_list, bands, strict=False)
        ]
        # Check if nodata values were found in the files
        nds = {r.nodata for r in readers}
        if len(nds) == 1:
            nodata = nds.pop()
        return cls(file_list, readers, num_threads=num_threads, nodata=nodata)

    def read_lon_lat(self, lons, lats) -> np.ndarray:
        """Read in raster values located at `lons`, `lats`."""

        def _read_single(reader):
            return reader.read_lon_lat(lons, lats, masked=reader.masked)

        results = thread_map(
            _read_single,
            self.readers,
            max_workers=self.num_threads,
            desc="Reading points from time series",
        )
        return np.array(list(results))

    def read_window(
        self,
        lon: float,
        lat: float,
        buffer_pixels: int = 0,
        op=round,
    ):
        """Get a window of pixel values for a given longitude and latitude."""
        return np.array(
            [
                reader.read_window(lon, lat, buffer_pixels=buffer_pixels, op=op)
                for reader in self.readers
            ]
        )


def _ensure_slices(rows: Index, cols: Index) -> tuple[slice, slice]:
    def _parse(key: Index):
        if isinstance(key, int):
            return slice(key, key + 1)
        elif key is ...:
            return slice(None)
        else:
            return key

    return _parse(rows), _parse(cols)


def _unpack_3d_slices(key: tuple[Index, ...]) -> tuple[Index, slice, slice]:
    # Check that it's a tuple of slices
    if not isinstance(key, tuple):
        msg = "Index must be a tuple of slices."
        raise TypeError(msg)
    if len(key) not in (1, 3):
        msg = "Index must be a tuple of 1 or 3 slices."
        raise TypeError(msg)
    # If only the band is passed (e.g. stack[0]), convert to (0, :, :)
    if len(key) == 1:
        key = (key[0], slice(None), slice(None))
    # unpack the slices
    bands, rows, cols = key
    # convert the rows/cols to slices
    r_slice, c_slice = _ensure_slices(rows, cols)
    return bands, r_slice, c_slice


def get_raster_units(filename: PathOrStr, band: int = 1) -> str | None:
    with rio.open(filename) as src:
        return src.units[band - 1]
