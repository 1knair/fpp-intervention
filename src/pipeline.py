"""Connect file discovery, preprocessing, and basic drive summaries.

The pipeline reads one drive at a time and returns one summary row per valid
file. Raw participant files are only read; they are never modified here.
"""

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data_loader import (
    DriveFile,
    TelemetryFileError,
    discover_drive_files,
    iter_plt_chunks,
    load_headers,
)
from src.metrics import MetricAccumulator, REQUIRED_SUMMARY_COLUMNS
from src.preprocessing import filter_driving_rows, validate_required_columns


LOGGER = logging.getLogger(__name__)

# Keeping a fixed column order makes printed tables and saved CSV files consistent.
OUTPUT_COLUMNS = [
    "participant_id",
    "visit",
    "session_date",
    "drive",
    "raw_rows",
    "driving_rows",
    "duration",
    "mean_velocity",
    "max_velocity",
]


def _process_drive_file(
    drive_file: DriveFile,
    column_names: Sequence[str],
    *,
    chunksize: int = 100_000,
) -> dict[str, object]:
    """Stream one drive file into a single compact summary row."""
    accumulator = MetricAccumulator()
    # Filtering and updating each chunk avoids loading an entire drive into memory.
    for chunk in iter_plt_chunks(
        drive_file.path,
        column_names,
        chunksize=chunksize,
    ):
        driving_rows = filter_driving_rows(chunk)
        accumulator.update(driving_rows, raw_row_count=len(chunk))

    # File-path metadata identifies the drive; finalize() adds its measurements.
    row: dict[str, object] = {
        "participant_id": drive_file.participant_id,
        "visit": drive_file.visit,
        "session_date": drive_file.session_date,
        "drive": drive_file.drive,
    }
    row.update(accumulator.finalize())
    return row


def process_dataset(
    data_root: str | Path,
    participant_ids: Sequence[str | int] | None = None,
    *,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Process discovered files while skipping and warning on bad files."""
    headers = load_headers(data_root)
    # Validate headers once before opening large telemetry files. An empty frame
    # is enough because this check only needs the column names.
    header_frame = pd.DataFrame(columns=headers)
    validate_required_columns(
        header_frame,
        ("Gear", *REQUIRED_SUMMARY_COLUMNS),
        context="headers.csv",
    )

    # participant_ids is optional; None tells discovery to include everyone.
    drive_files = discover_drive_files(data_root, participant_ids)

    rows: list[dict[str, object]] = []
    for drive_file in drive_files:
        try:
            row = _process_drive_file(
                drive_file,
                headers,
                chunksize=chunksize,
            )
        # One unreadable file should not stop the remaining participant drives.
        except (
            TelemetryFileError,
            pd.errors.ParserError,
            OSError,
            UnicodeError,
        ) as error:
            LOGGER.warning(
                "Skipping malformed telemetry file %s: %s", drive_file.path, error
            )
            continue

        if row["driving_rows"] == 0:
            LOGGER.warning("No Gear == 3 rows found in %s", drive_file.path)
        rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
