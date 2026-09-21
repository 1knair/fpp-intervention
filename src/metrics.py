"""Collect a few basic values that confirm the processing pipeline works.

The project is not interpreting driving performance yet. This module only keeps
row counts, elapsed driving time, and simple velocity values across file chunks.
"""

from dataclasses import dataclass
import math

import pandas as pd

from src.preprocessing import validate_required_columns


REQUIRED_SUMMARY_COLUMNS = ("Sim Time", "Velocity")


def _finite_numeric(series: pd.Series) -> pd.Series:
    """Convert usable values to numbers and remove missing or infinite values."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.loc[
        numeric.notna() & numeric.ne(math.inf) & numeric.ne(-math.inf)
    ]


@dataclass
class MetricAccumulator:
    """Keep the small running totals needed while reading a drive in chunks."""

    raw_rows: int = 0
    driving_rows: int = 0
    _time_min: float = math.inf
    _time_max: float = -math.inf
    _velocity_sum: float = 0.0
    _velocity_count: int = 0
    _velocity_max: float = -math.inf

    def update(self, frame: pd.DataFrame, raw_row_count: int) -> None:
        """Add one chunk of filtered driving rows to the running summary."""
        validate_required_columns(frame, REQUIRED_SUMMARY_COLUMNS)
        self.raw_rows += raw_row_count
        self.driving_rows += len(frame)

        # Only valid numeric values contribute to the time and velocity results.
        sim_time = _finite_numeric(frame["Sim Time"])
        if not sim_time.empty:
            self._time_min = min(self._time_min, float(sim_time.min()))
            self._time_max = max(self._time_max, float(sim_time.max()))

        velocity = _finite_numeric(frame["Velocity"])
        if not velocity.empty:
            self._velocity_sum += float(velocity.sum())
            self._velocity_count += len(velocity)
            self._velocity_max = max(self._velocity_max, float(velocity.max()))

    def finalize(self) -> dict[str, int | float]:
        """Return the five basic exploratory values for one drive."""
        duration = (
            self._time_max - self._time_min
            if math.isfinite(self._time_min) and math.isfinite(self._time_max)
            else math.nan
        )
        mean_velocity = (
            self._velocity_sum / self._velocity_count
            if self._velocity_count
            else math.nan
        )
        max_velocity = (
            self._velocity_max if math.isfinite(self._velocity_max) else math.nan
        )

        return {
            "raw_rows": self.raw_rows,
            "driving_rows": self.driving_rows,
            "duration": duration,
            "mean_velocity": mean_velocity,
            "max_velocity": max_velocity,
        }
