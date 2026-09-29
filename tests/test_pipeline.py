from pathlib import Path

import pandas as pd
import pytest

from src.cleaning.preprocessing import ANALYSIS_COLUMNS
from src.cleaning.pipeline import export_cleaned_dataset, process_dataset


def test_pipeline_streams_one_synthetic_drive(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Sim Time": [0.0, 1.0, 2.0],
            "Gear": [1, 3, 3],
            "Velocity": [0.0, 10.0, 20.0],
        }
    )
    pd.DataFrame(columns=frame.columns).to_csv(tmp_path / "headers.csv", index=False)
    session = tmp_path / "002" / "T1_20260918"
    session.mkdir(parents=True)
    frame.to_csv(
        session / "FPP_Sub_002_Drive_1.plt",
        sep=" ",
        header=False,
        index=False,
    )

    result = process_dataset(tmp_path, participant_ids=["002"], chunksize=2)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["participant_id"] == "002"
    assert row["drive"] == 1
    assert row["raw_rows"] == 3
    assert row["driving_rows"] == 2
    assert row["duration"] == pytest.approx(1.0)
    assert row["mean_velocity"] == pytest.approx(15.0)
    assert row["max_velocity"] == pytest.approx(20.0)
    assert result.columns.tolist() == [
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


def test_cleaned_export_keeps_all_headers_in_exact_order(tmp_path: Path) -> None:
    raw_headers = [
        "Unused Before",
        "DriveID",
        "Gear",
        *ANALYSIS_COLUMNS,
        "Unused After",
        "DriveID",
    ]
    exported_headers = [*raw_headers[:-1], "DriveID.1"]
    rows = [
        [
            "bad",
            101,
            3,
            0,
            10,
            1,
            200,
            10000,
            10000,
            0.1,
            0.5,
            0,
            0,
            25,
            1,
            101,
        ],
        [
            2,
            101,
            3,
            1,
            20,
            2,
            100,
            2,
            30,
            0.2,
            0.5,
            0,
            0,
            25,
            float("inf"),
            101,
        ],
        [3, 101, 1, 2, 30, 3, 200, 3, 40, 0.3, 1.0, 0, 0, 25, 3, 101],
    ]
    (tmp_path / "headers.csv").write_text(
        ",".join(raw_headers) + "\n", encoding="utf-8"
    )
    session = tmp_path / "002" / "T1_20260918"
    session.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(
        session / "FPP_Sub_002_Drive_1.plt",
        sep=" ",
        header=False,
        index=False,
    )
    output_dir = tmp_path / "cleaned"

    report = export_cleaned_dataset(tmp_path, output_dir, chunksize=1)

    cleaned = pd.read_csv(
        output_dir / "participant_002_T1_2026-09-18_drive_1_cleaned.csv"
    )
    assert cleaned.columns.tolist() == [
        "participant_id",
        "visit",
        "session_date",
        "drive",
        *exported_headers,
        "brake_force_over_170",
    ]
    assert len(cleaned) == 2
    assert cleaned["Gear"].tolist() == [3, 3]
    assert cleaned["Unused Before"].isna().tolist() == [True, False]
    assert cleaned["Unused After"].isna().tolist() == [False, True]
    assert cleaned["Headway Time"].isna().tolist() == [True, False]
    assert cleaned["Headway Distance"].isna().tolist() == [True, False]
    assert cleaned["Reaction Time"].tolist() == [0.5, 0.5]
    assert cleaned["brake_force_over_170"].tolist() == [True, False]
    assert report.loc[0, "positive_reaction_time_rows"] == 2
    assert report.loc[0, "consecutive_repeated_reaction_time_rows"] == 1

    mapping = pd.read_csv(output_dir / "header_mapping.csv")
    assert mapping.columns.tolist() == [
        "telemetry_column_position",
        "raw_header_text",
        "exported_pandas_column_name",
    ]
    assert mapping["telemetry_column_position"].tolist() == list(
        range(1, len(raw_headers) + 1)
    )
    assert mapping["raw_header_text"].tolist() == raw_headers
    assert mapping["exported_pandas_column_name"].tolist() == exported_headers
