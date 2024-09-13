import datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypeVar

# Some classes are declared as generic in stubs, but not at runtime.
# In Python 3.9 and earlier, os.PathLike is not subscriptable, results in runtime error
if TYPE_CHECKING:
    from builtins import ellipsis

    Index = ellipsis | slice | int
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike

PathOrStr = str | Path | PathLikeStr
DateOrDatetime = datetime.datetime | datetime.date


PathLikeT = TypeVar("PathLikeT", bound=PathLikeStr)


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
