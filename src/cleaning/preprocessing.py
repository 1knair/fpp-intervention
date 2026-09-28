"""Prepare raw telemetry rows before a basic drive summary is created.

The helpers in this file validate input columns and select active driving rows
without changing the original pandas DataFrame.
"""

from collections.abc import Iterable

import pandas as pd


ANALYSIS_COLUMNS = (
    "Sim Time",
    "Velocity",
    "Acceleration",
    "Brake Pedal Force",
    "Headway Time",
    "Headway Distance",
    "Lane Offset",
    "Reaction Time",
    "Collision",
    "Events",
    "Speed Limit",
)

# Headway Distance 10000.0 is the repeated no-lead sentinel, so both headway
# fields are missing on those rows. Exact Headway Time 10000.0 is also treated
# as a sentinel. Other large values remain because no cutoff is established.
HEADWAY_SENTINEL = 10000.0


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


def replace_headway_sentinel(series: pd.Series) -> pd.Series:
    """Replace only the documented repeated 10000.0 headway sentinel with NaN."""
    return series.mask(series.eq(HEADWAY_SENTINEL))


def clean_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric, finite analysis columns without modifying ``frame``.

    Invalid numeric text and positive or negative infinity become missing. A
    no-lead Headway Distance clears both headway fields on that row. Brake force
    is never clipped; a boolean column flags values above 170 N. Collision and
    Reaction Time remain row-level signals and are not aggregated.
    """
    validate_required_columns(frame, ANALYSIS_COLUMNS)
    cleaned = frame.loc[:, ANALYSIS_COLUMNS].copy()

    for column in ANALYSIS_COLUMNS:
        numeric = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = numeric.where(
            numeric.notna() & numeric.ne(float("inf")) & numeric.ne(float("-inf"))
        )

    no_lead = cleaned["Headway Distance"].eq(HEADWAY_SENTINEL)
    cleaned.loc[no_lead, ["Headway Distance", "Headway Time"]] = float("nan")
    cleaned["Headway Time"] = replace_headway_sentinel(cleaned["Headway Time"])

    cleaned["brake_force_over_170"] = cleaned["Brake Pedal Force"].gt(170)
    return cleaned
