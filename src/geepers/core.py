"""Compare InSAR time-series displacements with collocated GPS observations."""

from __future__ import annotations

import logging

# Re-export main components for backward compatibility
from geepers.analysis import compare_relative_gps_insar, create_tidy_df
from geepers.processing import process_insar_data
from geepers.quality import select_gps_reference
from geepers.workflows import main

__all__ = [
    "compare_relative_gps_insar",
    "create_tidy_df",
    "main",
    "process_insar_data",
    "select_gps_reference",
]

logger = logging.getLogger("geepers")
logger.setLevel(logging.INFO)
