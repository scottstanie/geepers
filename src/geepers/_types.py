import datetime
from pathlib import Path
from typing import NamedTuple

PathOrStr = str | Path
DateOrDatetime = datetime.datetime | datetime.date


class ReferencePoint(NamedTuple):
    row: int
    col: int


class Bbox(NamedTuple):
    """Bounding box named tuple, defining extent in cartesian coordinates.

    Usage:

        Bbox(left, bottom, right, top)

    Attributes
    ----------
    left : float
        Left coordinate (xmin)
    bottom : float
        Bottom coordinate (ymin)
    right : float
        Right coordinate (xmax)
    top : float
        Top coordinate (ymax)

    """

    left: float
    bottom: float
    right: float
    top: float
