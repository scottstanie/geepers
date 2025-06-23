from datetime import date, datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from builtins import ellipsis

    Index = ellipsis | slice | int

PathLikeStr = PathLike[str]
PathLikeT = TypeVar("PathLikeT", bound=PathLikeStr)

PathOrStr: TypeAlias = str | Path | PathLikeStr
DateOrDatetime: TypeAlias = datetime | date
DatetimeLike: TypeAlias = pd.Timestamp | np.datetime64 | datetime | str
