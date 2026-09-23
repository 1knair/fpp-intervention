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
from src.preprocessing import (
    ANALYSIS_COLUMNS,
    HEADWAY_SENTINEL,
    clean_analysis_columns,
    filter_driving_rows,
    validate_required_columns,
)


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

CLEANED_OUTPUT_COLUMNS = [
    "participant_id",
    "visit",
    "session_date",
    "drive",
    *ANALYSIS_COLUMNS,
    "brake_force_over_170",
]

CLEANING_REPORT_COLUMNS = [
    "participant_id",
    "visit",
    "session_date",
    "drive",
    "driving_rows",
    "headway_time_sentinel_rows",
    "headway_distance_sentinel_rows",
    "brake_force_over_170_rows",
    "positive_reaction_time_rows",
    "consecutive_repeated_reaction_time_rows",
    "output_file",
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


def _clean_output_name(drive_file: DriveFile) -> str:
    """Build a stable file name from path-derived drive metadata."""
    participant = drive_file.participant_id or "unknown"
    visit = drive_file.visit or "unknown"
    date = drive_file.session_date or "unknown-date"
    drive = drive_file.drive if drive_file.drive is not None else "unknown"
    return f"participant_{participant}_{visit}_{date}_drive_{drive}_cleaned.csv"


def _export_cleaned_drive(
    drive_file: DriveFile,
    column_names: Sequence[str],
    output_dir: Path,
    *,
    chunksize: int,
) -> dict[str, object]:
    """Write one cleaned drive CSV and return its cleaning diagnostics."""
    output_path = output_dir / _clean_output_name(drive_file)
    first_chunk = True
    previous_reaction_time: float | None = None
    report: dict[str, object] = {
        "participant_id": drive_file.participant_id,
        "visit": drive_file.visit,
        "session_date": drive_file.session_date,
        "drive": drive_file.drive,
        "driving_rows": 0,
        "headway_time_sentinel_rows": 0,
        "headway_distance_sentinel_rows": 0,
        "brake_force_over_170_rows": 0,
        "positive_reaction_time_rows": 0,
        "consecutive_repeated_reaction_time_rows": 0,
        "output_file": str(output_path),
    }

    for chunk in iter_plt_chunks(
        drive_file.path,
        column_names,
        chunksize=chunksize,
    ):
        driving_rows = filter_driving_rows(chunk)

        for column, report_column in (
            ("Headway Time", "headway_time_sentinel_rows"),
            ("Headway Distance", "headway_distance_sentinel_rows"),
        ):
            numeric = pd.to_numeric(driving_rows[column], errors="coerce")
            report[report_column] += int(numeric.eq(HEADWAY_SENTINEL).sum())

        cleaned = clean_analysis_columns(driving_rows)
        reaction_time = cleaned["Reaction Time"]
        positive = reaction_time.gt(0)
        previous = reaction_time.shift(1)
        if previous_reaction_time is not None and not reaction_time.empty:
            previous.iloc[0] = previous_reaction_time

        report["driving_rows"] += len(cleaned)
        report["brake_force_over_170_rows"] += int(
            cleaned["brake_force_over_170"].sum()
        )
        report["positive_reaction_time_rows"] += int(positive.sum())
        report["consecutive_repeated_reaction_time_rows"] += int(
            (positive & reaction_time.eq(previous)).sum()
        )
        if not reaction_time.empty:
            previous_reaction_time = reaction_time.iloc[-1]

        metadata = {
            "participant_id": drive_file.participant_id,
            "visit": drive_file.visit,
            "session_date": drive_file.session_date,
            "drive": drive_file.drive,
        }
        cleaned = cleaned.assign(**metadata).loc[:, CLEANED_OUTPUT_COLUMNS]
        cleaned.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

    if first_chunk:
        pd.DataFrame(columns=CLEANED_OUTPUT_COLUMNS).to_csv(output_path, index=False)
    return report


def export_cleaned_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    participant_ids: Sequence[str | int] | None = None,
    *,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Export one cleaned CSV per drive and return a per-drive cleaning report.

    Raw telemetry is read only. The separate output files retain identifiers on
    every row and prevent a multi-participant export from becoming one huge CSV.
    """
    headers = load_headers(data_root)
    validate_required_columns(
        pd.DataFrame(columns=headers),
        ("Gear", *ANALYSIS_COLUMNS),
        context="headers.csv",
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    for drive_file in discover_drive_files(data_root, participant_ids):
        try:
            reports.append(
                _export_cleaned_drive(
                    drive_file,
                    headers,
                    output_path,
                    chunksize=chunksize,
                )
            )
        except (
            TelemetryFileError,
            pd.errors.ParserError,
            OSError,
            UnicodeError,
        ) as error:
            LOGGER.warning(
                "Skipping malformed telemetry file %s: %s", drive_file.path, error
            )

    report_frame = pd.DataFrame(reports, columns=CLEANING_REPORT_COLUMNS)
    report_frame.to_csv(output_path / "cleaning_report.csv", index=False)
    return report_frame
