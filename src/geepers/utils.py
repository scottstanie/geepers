import os
import sys
from pathlib import Path


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


def station_distance(station_name1: str, station_name2: str) -> float:
    """Find geodetic distance (in meters) between two gps stations.

    Uses the WGS84 ellipsoid for calculation.

    Parameters
    ----------
    station_name1 :str)
        name of first GPS station
    station_name2 :str)
        name of second GPS station

    Returns:
        float: distance (in meters)
    """
    from .download import station_lonlat
    from pyproj import Geod

    lon1, lat1 = station_lonlat(station_name1)
    lon2, lat2 = station_lonlat(station_name2)

    g = Geod(ellps="WGS84")

    return g.line_length([lon1, lon2], [lat1, lat2], radians=False)
