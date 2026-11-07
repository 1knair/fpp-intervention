import numpy as np
import pandas as pd
import pytest

from src.representation.drive_features import (
    FEATURE_COLS, build_drive_features, calc_drive_features, school_zone_windows,
)


def sample_drive(n=160, step=0.05):
    return pd.DataFrame({
        "Sim Time": np.arange(n) * step,
        "Velocity": [20 * 0.44704] * n,
        "Speed Limit": [25] * n,
        "Acceleration": [0.] * n,
        "Brake Pedal Force": [0.] * n,
        "Speed Check": [0] * n,
        "Signs": [0] * n,
        "Events": [0] * n,
    })


def test_speed_check_counts_windows_instead_of_samples():
    dr = sample_drive()
    dr.loc[10:39, "Speed Check"] = 1
    dr.loc[10:39, "Velocity"] = 30 * 0.44704
    dr.loc[60:69, "Speed Check"] = 1
    features = calc_drive_features(dr)
    assert features["speed_check_opportunities"] == 2
    assert features["speed_check_overspeed_count"] == 1


def test_sustained_episode_counts_and_exposure():
    dr = sample_drive()
    dr.loc[10:39, "Velocity"] = 40 * 0.44704  # 1.5-second speeding episode
    dr.loc[60:64, "Velocity"] = 40 * 0.44704  # short spike, not an episode
    dr.loc[80:91, "Acceleration"] = -5       # 0.6-second hard-braking episode
    dr.loc[110:112, "Acceleration"] = -5     # short spike
    features = calc_drive_features(dr)
    assert features["speeding_episode_count"] == 1
    assert features["hard_brake_episode_count"] == 1
    assert features["speeding_episodes_per_km"] == pytest.approx(1 / features["speed_exposure_km"])
    assert features["hard_brake_episodes_per_km"] == pytest.approx(1 / features["brake_exposure_km"])


def test_no_events_are_invented_at_gaps_or_recording_edges():
    dr = sample_drive()
    dr.loc[:30, "Velocity"] = 40 * 0.44704   # starts active: onset unknown
    dr.loc[50:100, "Velocity"] = 40 * 0.44704
    dr.loc[75:, "Sim Time"] += 5             # gap inside observed speeding
    dr.loc[50:100, "Speed Check"] = 1
    features = calc_drive_features(dr)
    assert features["speeding_episode_count"] == 0
    assert features["speed_check_opportunities"] == 0
    assert pd.isna(features["speed_check_overspeed_count"])


@pytest.mark.parametrize("event_code", [3, 4])
def test_school_zone_requires_marker_evidence_and_correct_units(event_code):
    dr = sample_drive()
    dr.loc[10:39, "Speed Limit"] = 40
    dr.loc[10:39, "Velocity"] = 36 * 0.44704
    dr.loc[40:79, "Speed Limit"] = 20
    dr.loc[40:79, "Velocity"] = 23 * 0.44704
    dr.loc[80:109, "Speed Limit"] = 40
    dr.loc[80:109, "Velocity"] = 32 * 0.44704
    assert pd.isna(calc_drive_features(dr)["school_zone_entry_excess_mph"])
    dr.loc[39:42, "Signs"] = 20
    dr.loc[39:80, "Events"] = event_code
    dt = dr["Sim Time"].shift(-1) - dr["Sim Time"]
    assert school_zone_windows(dr, dt) == [(40, 80)]
    features = calc_drive_features(dr)
    assert features["school_zone_opportunities"] == 1
    assert features["school_zone_entry_excess_mph"] == pytest.approx(3)
    assert features["pre_school_zone_40mph_shortfall"] == pytest.approx(4)
    assert features["school_zone_resume_35mph_shortfall"] == pytest.approx(3)


@pytest.mark.parametrize("step", [0.025, 0.05])
def test_episode_count_is_independent_of_sampling_rate(step):
    dr = sample_drive(n=round(8 / step), step=step)
    dr.loc[dr["Sim Time"].ge(1) & dr["Sim Time"].lt(3), "Velocity"] = 40 * 0.44704
    assert calc_drive_features(dr)["speeding_episode_count"] == 1


def test_absent_context_stays_missing_and_observed_no_event_is_zero():
    features = calc_drive_features(sample_drive())
    assert pd.isna(features["school_zone_entry_excess_mph"])
    assert pd.isna(features["pre_school_zone_40mph_shortfall"])
    assert features["speeding_episodes_per_km"] == 0
    assert features["hard_brake_episodes_per_km"] == 0
    dr = sample_drive()
    dr["Velocity"] = 0
    assert pd.isna(calc_drive_features(dr)["speeding_episodes_per_km"])


def test_whole_drive_csv_keeps_recording_identity(tmp_path):
    sample_drive().to_csv(tmp_path / "participant_002_T1_2025-04-21_drive_2_cleaned.csv", index=False)
    sample_drive().to_csv(tmp_path / "participant_002_T1_2025-04-21_drive_21_cleaned.csv", index=False)
    result = build_drive_features(tmp_path, tmp_path / "features.csv")
    assert len(result) == 2
    assert result.recording_id.is_unique
    assert result.participant_id.tolist() == ["002", "002"]
    assert set(result.drive_number) == {2, 21}
    assert set(FEATURE_COLS).issubset(result.columns)
