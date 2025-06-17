import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._types import DateOrDatetime


def datetime_to_float(dates: Sequence[DateOrDatetime]) -> np.ndarray:
    """Convert a sequence of datetime objects to a float representation.

    Output units are in days since the first item in `dates`.

    Parameters
    ----------
    dates : Sequence[DateOrDatetime]
        List of datetime objects to convert to floats

    Returns
    -------
    date_arr : np.array 1D
        The float representation of the datetime objects

    """
    sec_per_day = 60 * 60 * 24
    date_arr = np.asarray(dates).astype("datetime64[s]")
    # Reference the 0 to the first date
    date_arr = date_arr - date_arr[0]
    return date_arr.astype(float) / sec_per_day


def get_cache_dir(force_posix=False, app_name="geepers") -> Path:
    """Return the cache folder for the application.

    The directory is used to store gps timeseries data.
    The default behavior is to return whatever is most appropriate for the OS.


    the following folders could be returned:
    Mac OS X:
      ``~/Library/Application Support/geepers``
    Mac OS X (POSIX):
      ``~/.geepers``
    Unix:
      ``~/.cache/geepers``
    Unix (POSIX):
      ``~/.geepers``

    Parameters
    ----------
    force_posix : bool
        If this is set to `True` then on any POSIX system the folder will be stored
        in the home folder with a leading dot instead of the XDG config home or darwin's
        application support folder.
    app_name : str
        Name of the application (for naming the subfolder)

    Returns
    -------
    Path
        Folder to store cached data.

    Starting source:
    https://github.com/pallets/click/blob/ca5e1c3d75e95cbc70fa6ed51ef263592e9ac0d0/src/click/utils.py#L32

    """
    if force_posix:
        return Path(f"~/.{app_name}").expanduser()
    if sys.platform == "darwin":
        return Path("~/Library/Application Support").expanduser() / app_name
    else:
        base_path = Path(
            os.environ.get("XDG_CONFIG_HOME", Path("~/.cache").expanduser())
        )
        return base_path / app_name
