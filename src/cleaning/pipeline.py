"""Connect file discovery, preprocessing, and basic drive summaries.

The pipeline reads one drive at a time and returns one summary row per valid
file. Raw participant files are only read; they are never modified here.
"""

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.cleaning.data_loader import (
    DriveFile,
    TelemetryFileError,
    discover_drive_files,
    iter_plt_chunks,
    load_header_mapping,
    load_headers,
)
from src.cleaning.metrics import MetricAccumulator, REQUIRED_SUMMARY_COLUMNS
from src.cleaning.preprocessing import (
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

CLEANED_METADATA_COLUMNS = [
    "participant_id",
    "visit",
    "session_date",
    "drive",
]
CLEANED_DERIVED_COLUMNS = ["brake_force_over_170"]

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


def _cleaned_output_columns(column_names: Sequence[str]) -> list[str]:
    """Return metadata, header-defined telemetry, and derived output columns."""
    return [*CLEANED_METADATA_COLUMNS, *column_names, *CLEANED_DERIVED_COLUMNS]


def _validate_cleaned_output_columns(
    frame: pd.DataFrame,
    column_names: Sequence[str],
) -> None:
    """Ensure exported telemetry exactly matches ``headers.csv`` and its order."""
    expected = _cleaned_output_columns(column_names)
    actual = frame.columns.tolist()
    if actual != expected:
        raise ValueError(
            "Cleaned export columns do not match metadata plus headers.csv: "
            f"expected {expected}, got {actual}"
        )


def _export_cleaned_drive(
    drive_file: DriveFile,
    column_names: Sequence[str],
    output_dir: Path,
    *,
    chunksize: int,
) -> dict[str, object]:
    """Write one cleaned drive CSV and return its cleaning diagnostics."""
    output_path = output_dir / _clean_output_name(drive_file)
    output_columns = _cleaned_output_columns(column_names)
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

        cleaned = clean_analysis_columns(driving_rows, column_names)
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

        metadata = pd.DataFrame(
            {
                "participant_id": drive_file.participant_id,
                "visit": drive_file.visit,
                "session_date": drive_file.session_date,
                "drive": drive_file.drive,
            },
            index=cleaned.index,
        )
        cleaned = pd.concat([metadata, cleaned], axis="columns")
        _validate_cleaned_output_columns(cleaned, column_names)
        cleaned.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

    if first_chunk:
        empty = pd.DataFrame(columns=output_columns)
        _validate_cleaned_output_columns(empty, column_names)
        empty.to_csv(output_path, index=False)
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
    load_header_mapping(data_root).to_csv(
        output_path / "header_mapping.csv", index=False
    )

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
