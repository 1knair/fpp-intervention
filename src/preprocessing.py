"""Prepare raw telemetry rows before a basic drive summary is created.

The helpers in this file validate input columns and select active driving rows
without changing the original pandas DataFrame.
"""

from collections.abc import Iterable

import pandas as pd


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str = "telemetry data",
) -> None:
    """Raise a readable error listing required columns that are absent."""
    # Reporting every missing column at once makes malformed input easier to fix.
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def filter_driving_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing only rows where ``Gear == 3``.

    The input frame and all of its raw columns remain unchanged. Numeric
    coercion is used only for the temporary filter mask.
    """
    validate_required_columns(frame, ("Gear",))
    # Invalid or blank gear values become NaN and therefore do not pass the filter.
    gear = pd.to_numeric(frame["Gear"], errors="coerce")
    # Return a copy so later processing cannot accidentally edit the raw chunk.
    return frame.loc[gear.eq(3)].copy()
