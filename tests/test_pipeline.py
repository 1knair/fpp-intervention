from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import process_dataset


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
