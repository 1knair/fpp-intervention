"""Find participant data and read simulator files in manageable chunks.

This module handles file-system work only. It does not calculate any driving
metrics or change the raw study files.
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
from typing import Iterator, Sequence

import pandas as pd


# These names let the project find a local dataset without storing a
# computer-specific path in the source code.
DATA_ROOT_ENV_VAR = "FPP_DATA_ROOT"
PROJECT_FOLDER_NAMES = (
    "Fermented Papaya and Driving Study (Yuki 2024) - FPP Participant Data",
    "FPP Participant Data",
)
# Session folders begin with a visit label such as T1 and may include a date.
SESSION_PATTERN = re.compile(
    r"^(T[123])(?:_(\d{8}))?(?:_.*)?$",
    re.IGNORECASE,
)
# Drive numbers are read from file names such as FPP_Sub_002_Drive_1.plt.
DRIVE_PATTERN = re.compile(r"(?:^|_)Drive_(\d+)(?:_|\.|$)", re.IGNORECASE)


class TelemetryFileError(ValueError):
    """Raised when a telemetry file cannot be safely read."""


@dataclass(frozen=True)
class DriveFile:
    """Path-derived identifiers for one simulator telemetry file."""

    path: Path
    participant_id: str | None
    visit: str | None
    session_date: str | None
    drive: int | None


def _candidate_data_roots() -> list[Path]:
    """Build conservative OneDrive fallback paths without a username."""
    # Windows may provide one or more of these variables depending on the
    # installed OneDrive account type.
    one_drive_roots: list[Path] = []
    for variable in ("OneDriveCommercial", "OneDrive", "OneDriveConsumer"):
        value = os.getenv(variable)
        if value:
            one_drive_roots.append(Path(value).expanduser())

    home = Path.home()
    one_drive_roots.extend(
        (
            home / "OneDrive - University of Florida",
            home / "OneDrive",
        )
    )

    candidates: list[Path] = []
    for one_drive_root in one_drive_roots:
        for folder_name in PROJECT_FOLDER_NAMES:
            candidate = one_drive_root / folder_name
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def get_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the dataset root from an argument, environment, or OneDrive.

    Resolution order is: an explicit ``data_root`` argument,
    ``FPP_DATA_ROOT``, and a short list of local OneDrive candidates.
    """
    configured_root = data_root or os.getenv(DATA_ROOT_ENV_VAR)
    if configured_root:
        root = Path(configured_root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        return root.resolve()

    candidates = _candidate_data_roots()
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    checked = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate the FPP dataset. Set {DATA_ROOT_ENV_VAR} to its "
        f"root folder. OneDrive locations checked:\n  - {checked}"
    )


def get_participants(data_root: str | Path) -> list[Path]:
    """Return all available numeric participant directories."""
    root = Path(data_root)
    # Converting the name to an integer gives a natural order: 2 comes before 10.
    return sorted(
        (
            folder
            for folder in root.iterdir()
            if folder.is_dir() and folder.name.isdigit()
        ),
        key=lambda folder: int(folder.name),
    )


def get_sessions(participant_folder: str | Path) -> list[Path]:
    """Return session folders whose names begin with T1, T2, or T3."""
    participant_path = Path(participant_folder)
    sessions = [
        folder
        for folder in participant_path.iterdir()
        if folder.is_dir() and SESSION_PATTERN.match(folder.name)
    ]

    def sort_key(folder: Path) -> tuple[int, str]:
        match = SESSION_PATTERN.match(folder.name)
        assert match is not None
        return int(match.group(1)[1]), folder.name.lower()

    return sorted(sessions, key=sort_key)


def _is_groundsim(path: Path, session_folder: Path) -> bool:
    relative_parts = path.relative_to(session_folder).parts
    return any("groundsim" in part.lower() for part in relative_parts)


def _drive_number(path: Path) -> int | None:
    match = DRIVE_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def get_plt_files(
    session_folder: str | Path,
) -> list[Path]:
    """Discover drive files while excluding GroundSim data."""
    session_path = Path(session_folder)
    # rglob searches the session and any subfolders because exports are not
    # always stored at exactly the same directory depth.
    files = [
        path
        for path in session_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".plt"
        and not _is_groundsim(path, session_path)
    ]
    return sorted(
        files,
        key=lambda path: (
            _drive_number(path) is None,
            _drive_number(path) if _drive_number(path) is not None else 0,
            str(path).lower(),
        ),
    )


def extract_drive_metadata(file_path: str | Path) -> DriveFile:
    """Extract participant, visit, date, and drive identifiers from a path."""
    path = Path(file_path)
    session_folder: Path | None = None
    session_match: re.Match[str] | None = None
    # Walk upward from the file until a T1/T2/T3 session folder is found.
    for parent in path.parents:
        match = SESSION_PATTERN.match(parent.name)
        if match:
            session_folder = parent
            session_match = match
            break

    participant_id: str | None = None
    if session_folder is not None and session_folder.parent.name.isdigit():
        participant_id = session_folder.parent.name

    session_date: str | None = None
    visit: str | None = None
    if session_match is not None:
        visit = session_match.group(1).upper()
        raw_date = session_match.group(2)
        if raw_date:
            try:
                session_date = datetime.strptime(raw_date, "%Y%m%d").date().isoformat()
            except ValueError:
                session_date = raw_date

    return DriveFile(
        path=path,
        participant_id=participant_id,
        visit=visit,
        session_date=session_date,
        drive=_drive_number(path),
    )


def discover_drive_files(
    data_root: str | Path,
    participant_ids: Sequence[str | int] | None = None,
) -> list[DriveFile]:
    """Discover non-practice drive files and their path metadata."""
    # Participant IDs are compared as integers so "2" and "002" both select
    # the same numeric participant folder.
    selected_ids = (
        {int(participant_id) for participant_id in participant_ids}
        if participant_ids is not None
        else None
    )
    records: list[DriveFile] = []
    for participant in get_participants(data_root):
        if selected_ids is not None and int(participant.name) not in selected_ids:
            continue
        for session in get_sessions(participant):
            for path in get_plt_files(session):
                record = extract_drive_metadata(path)
                # Drive 0 is practice data and is not part of this first version.
                if record.drive == 0:
                    continue
                records.append(record)
    return records


def load_headers(data_root: str | Path) -> list[str]:
    """Load and validate telemetry column names from ``headers.csv``."""
    headers_file = Path(data_root) / "headers.csv"
    if not headers_file.is_file():
        raise FileNotFoundError(f"Could not find headers.csv at: {headers_file}")

    headers = [
        str(column).lstrip("\ufeff").strip()
        for column in pd.read_csv(headers_file, nrows=0).columns
    ]
    if not headers or any(not column for column in headers):
        raise ValueError(
            f"headers.csv contains empty or missing column names: {headers_file}"
        )
    duplicates = sorted({column for column in headers if headers.count(column) > 1})
    if duplicates:
        raise ValueError(f"headers.csv contains duplicate columns: {duplicates}")
    return headers


def _first_data_field_count(file_path: Path) -> int:
    """Count values in the first non-empty row of a telemetry file."""
    with file_path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                return len(stripped.split())
    raise TelemetryFileError(f"Telemetry file is empty: {file_path}")


def validate_header_count(file_path: str | Path, column_names: Sequence[str]) -> None:
    """Check the first data row against the number of supplied headers."""
    path = Path(file_path)
    actual_count = _first_data_field_count(path)
    expected_count = len(column_names)
    if actual_count != expected_count:
        raise TelemetryFileError(
            f"Header count mismatch for {path}: headers.csv has {expected_count} "
            f"columns but the first data row has {actual_count} fields"
        )


def iter_plt_chunks(
    file_path: str | Path,
    column_names: Sequence[str],
    *,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Yield bounded chunks from one telemetry file after header validation."""
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")
    path = Path(file_path)
    validate_header_count(path, column_names)
    # PLT exports are whitespace-separated and have no header row of their own.
    # Returning an iterator keeps large participant files out of memory at once.
    yield from pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=list(column_names),
        chunksize=chunksize,
        on_bad_lines="error",
    )
