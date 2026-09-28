from src.representation.features import calc_drive
from src.representation.pipeline import build_feature_table
import pandas as pd
import pytest

def sample_recording():
    return pd.DataFrame({
        "Velocity": [27.5, 15.0, 45.0],
        "Lane Offset": [-2.5, 0.0, 1.0],
        "Acceleration": [-5.8, 1.5, 2.5],
    })

def test_calc_drive():
    result = calc_drive(sample_recording())

    assert result == pytest.approx({
        "velocity_mean": 29.166667,
        "velocity_sd": 15.069284,
        "lane_offset_mean": -0.5,
        "lane_offset_sd": 1.802776,
        "lane_offset_abs_mean": 1.166667,
        "acc_mean": -0.6,
        "acc_sd": 4.531004,
    })

# easy
def test_missing_values():
    frame = sample_recording()
    frame.loc[2, "Velocity"] = float("nan")
    frame["Acceleration"] = float("nan")
    result = calc_drive(frame)

    # expect to ignore missing telemetry
    assert result["velocity_mean"] == pytest.approx(21.25)
    assert result["velocity_sd"] == pytest.approx(8.838835)
    # expect nan from pandas
    assert pd.isna(result["acc_mean"])
    assert pd.isna(result["acc_sd"])

def test_build_feature_table(tmp_path):
    # sample recordings -> CSVs -> pipeline -> feature table; check metadata, feats & saved output in temp path
    in_dir = tmp_path / "cleaned"
    in_dir.mkdir()
    out_path = tmp_path / "features" / "drive_features.csv"
    for visit in ("T1", "T2"):
        sample_recording().to_csv(
            in_dir / "participant_002_{}_2025-04-21_drive_2_cleaned.csv".format(visit),
            index=False,
        )
    result = build_feature_table(in_dir, out_path, "drive")
    res_frame = pd.read_csv(out_path, dtype={"participant_id": str})

    assert len(result) == 2
    assert res_frame["participant_id"].tolist() == ["002", "002"]
    assert res_frame["visit"].tolist() == ["T1", "T2"]
    assert res_frame["drive_number"].tolist() == [2, 2]
    assert res_frame["recording_id"].tolist() == [
        "participant_002_T1_2025-04-21_drive_2_cleaned",
        "participant_002_T2_2025-04-21_drive_2_cleaned",
    ]
    assert res_frame["velocity_mean"].tolist() == pytest.approx([29.166667, 29.166667])
    pd.testing.assert_frame_equal(res_frame, result)
