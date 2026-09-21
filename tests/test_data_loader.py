"""Tests use only temporary, synthetic directory structures and values."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    TelemetryFileError,
    discover_drive_files,
    get_data_root,
    get_participants,
    get_sessions,
    load_headers,
    validate_header_count,
)


def test_participant_and_drive_discovery(tmp_path: Path) -> None:
    (tmp_path / "headers.csv").write_text("A,B\n", encoding="utf-8")
    t1 = tmp_path / "002" / "T1_20260918"
    t2 = tmp_path / "002" / "T2_20260919"
    t3 = tmp_path / "010" / "T3_20260920"
    t1.mkdir(parents=True)
    t2.mkdir()
    t3.mkdir(parents=True)
    (tmp_path / "notes").mkdir()

    (t1 / "FPP_Sub_002_Drive_0.plt").touch()
    (t1 / "FPP_Sub_002_Drive_1.plt").touch()
    (t2 / "FPP_Sub_002_Drive_2.plt").touch()
    (t3 / "FPP_Sub_010_Drive_3.plt").touch()
    groundsim = t1 / "GroundSim"
    groundsim.mkdir()
    (groundsim / "FPP_Sub_002_Drive_4.plt").touch()

    assert [path.name for path in get_participants(tmp_path)] == ["002", "010"]
    assert [path.name for path in get_sessions(tmp_path / "002")] == [
        "T1_20260918",
        "T2_20260919",
    ]

    records = discover_drive_files(tmp_path)
    assert [record.visit for record in records] == ["T1", "T2", "T3"]
    assert [record.drive for record in records] == [1, 2, 3]
    assert records[0].participant_id == "002"
    assert records[0].session_date == "2026-09-18"

    selected = discover_drive_files(tmp_path, participant_ids=["010"])
    assert [record.participant_id for record in selected] == ["010"]


def test_data_root_uses_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPP_DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path.resolve()


def test_loads_57_headers_and_validates_telemetry(tmp_path: Path) -> None:
    expected_headers = [
        "Sim Time",
        "Gear",
        "Velocity",
        *(f"Column {number}" for number in range(4, 58)),
    ]
    pd.DataFrame(columns=expected_headers).to_csv(
        tmp_path / "headers.csv", index=False
    )
    assert len(expected_headers) == 57
    assert load_headers(tmp_path) == expected_headers

    telemetry_file = tmp_path / "synthetic.plt"
    telemetry_file.write_text(
        " ".join(str(number) for number in range(57)), encoding="utf-8"
    )

    validate_header_count(telemetry_file, expected_headers)
    with pytest.raises(TelemetryFileError, match="has 56 columns"):
        validate_header_count(telemetry_file, expected_headers[:-1])
