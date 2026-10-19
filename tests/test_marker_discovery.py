"""Marker discovery tests use synthetic telemetry only."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.analysis.marker_discovery import (
    COMPARISON_COLUMNS,
    COMPARISON_DETAIL_COLUMNS,
    EVENT_CONTEXT_COLUMNS,
    EVIDENCE_COLUMNS,
    RECORDING_CHAIN_COLUMNS,
    SELECTION_COLUMNS,
    INTERVAL_COLUMNS,
    RT_EPISODE_COLUMNS,
    SUMMARY_COLUMNS,
    TRANSITION_COLUMNS,
    analyze_file,
    analyze_recording,
    build_event_context,
    compare_recordings,
    discover_cleaned_files,
    main,
    report_recording_chains,
    select_preferred_recordings,
)


EXPECTED_COMPARISON_COLUMNS = [
    "recording_id", "participant_id", "visit", "drive", "scenario", "selection_reason", "row_count",
    "rt_episode_count",
    "collision1_event_time", "collision1_rt_time", "collision1_rt_value_change", "collision1_rt_status",
    "collision1_peak_brake", "collision1_peak_deceleration",
    "collision2_candidate_rt_time", "collision2_candidate_rt_value_change",
    "collision2_candidate_peak_brake", "collision2_candidate_peak_deceleration", "collision2_candidate_status",
    "events_values_in_order", "events_1_start_time", "events_3_start_time", "events_4_start_time", "events_5_start_time",
    "intersection_count", "sign_values_in_order", "speed_limit_values_in_order", "missing_columns", "notes",
]
BATCH_OUTPUT_NAMES = [
    "scenario_comparison", "recording_selection", "recording_chain_qc", "scenario_comparison_details",
]


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    """Keep relative CLI output paths inside pytest's temporary directory."""
    monkeypatch.chdir(tmp_path)


def telemetry(**columns) -> pd.DataFrame:
    length = len(next(iter(columns.values()))) if columns else 0
    return pd.DataFrame({"Sim Time": [row * 0.5 for row in range(length)], **columns})


def metric(result, column, name):
    rows = result.summary.loc[
        result.summary.column.eq(column) & result.summary.metric.eq(name), "value"
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def intervals(result, column, kind="value"):
    return result.intervals.loc[
        result.intervals.column.eq(column) & result.intervals.interval_type.eq(kind)
    ].reset_index(drop=True)


def changes(result, column):
    return result.transitions.loc[
        result.transitions.column.eq(column) & result.transitions.record_type.eq("transition")
    ].reset_index(drop=True)


def comparison_drive() -> pd.DataFrame:
    """Two RT bursts, persistent positivity, and independent marker patterns."""
    return pd.DataFrame({
        "Sim Time": [0, 1, 1.2, 1.5, 2, 3, 5, 6, 10, 10.2, 10.5, 11, 13, 14, 16],
        "Reaction Time": [0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        "Brake Pedal Force": [999, 0, 0, 152, 60, 0, 999, 0, 0, 0, 73, 20, 0, 999, 0],
        "Acceleration": [-99, 0, 0, -8.46, -1, 0, -99, 0, 0, 0, -5.03, -1, 0, -99, 0],
        "Events": [0, 1, 1, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0],
        "Intersections": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Signs": [0, 35, 0, -3, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Speed Limit": [35, 35, 10, 10, 0, 20, 20, 20, 35, 35, 35, 35, 35, 35, 35],
    })


def write_comparison_drive(tmp_path: Path, frame: pd.DataFrame, participant="002", drive=1, visit="T2") -> Path:
    path = tmp_path / f"participant_{participant}_{visit}_2025-04-22_drive_{drive}_cleaned.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def route_context_drive(updates) -> pd.DataFrame:
    """Synthetic route containing delayed event braking and later RT braking."""
    times = sorted({300, 309.4, 309.5, 314.7, 316, 480, 482.6, 483.2, 484, 484.2, 485, 614.7, 615,
                    *(time for time, _ in updates)})
    updates = dict(updates)
    current = 0
    reaction = []
    for time in times:
        current = updates.get(time, current)
        reaction.append(current)
    return pd.DataFrame({
        "Sim Time": times, "Reaction Time": reaction,
        "Events": [1 if time == 309.4 else 3 if time == 614.7 else 0 for time in times],
        "Brake Pedal Force": [127 if time == 314.7 else 51 if time == 484.2 else 0 for time in times],
        "Acceleration": [-7.89 if time == 314.7 else -3.86 if time == 484.2 else 0 for time in times],
    })


def event_context_drive() -> pd.DataFrame:
    return pd.DataFrame({
        "Sim Time": list(range(10)),
        "Events": [0, 0, 0, 0, 3, 3, 0, 4, 0, 0],
        "Signs": [0, 35, 35, 0, 0, -3, -3, 0, 9, 0],
        "Speed Limit": [35, 35, 10, 10, 10, 10, 20, 20, 25, 25],
        "Intersections": [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
        "Reaction Time": [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1],
    })


def scenario1_evidence_drive():
    frame = pd.DataFrame({
        "Sim Time": list(range(27)), "Events": [0] * 27, "Signs": [0] * 27,
        "Intersections": [0] * 27, "Speed Limit": [35] * 27,
        "Reaction Time": [0.0] * 27, "Brake Pedal Force": [0] * 27, "Acceleration": [0] * 27,
    })
    frame.loc[[1, 3, 16, 19, 21], "Signs"] = [35, -3, 20, -6, -7]
    frame.loc[3:4, "Speed Limit"] = 10
    frame.loc[15:, "Speed Limit"] = 40
    frame.loc[16:17, "Speed Limit"] = 20
    frame.loc[[6, 15, 16, 17, 20, 23, 24], "Events"] = [1, 3, 3, 3, 4, 5, 5]
    frame.loc[[22, 25], "Intersections"] = 1
    for row, value in ((6, 0.2), (7, 0.5), (10, 1.0), (11, 1.5)):
        frame.loc[row:, "Reaction Time"] = value
    frame.loc[[8, 12], "Brake Pedal Force"] = [180, 50]
    frame.loc[[8, 12], "Acceleration"] = [-5, -3]
    return frame


@pytest.mark.parametrize("at_eof", [False, True], ids=["bounded", "eof"])
def test_interval_bounds_use_positions_and_exclusive_ends(at_eof):
    frame = telemetry(Events=[0, 2, 2] if at_eof else [0, 2, 2, 0])
    frame["Sim Time"] = [10, 11, 13] if at_eof else [10, 11, 13, 19]
    frame.index = range(20, 20 + len(frame))
    result = analyze_recording(frame)
    run = intervals(result, "Events").iloc[0]
    assert (run.interval_start_row, run.interval_end_row) == (1, 3)
    assert frame.iloc[run.interval_start_row:run.interval_end_row].Events.tolist() == [2, 2]
    assert (run.interval_start_time, run.last_active_time, run.duration_s) == (11, 13, 2)
    assert run.last_active_row == 2
    assert bool(run.right_censored) == at_eof
    if at_eof:
        assert pd.isna(run.interval_end_time)
    else:
        assert run.interval_end_time == 19
    assert changes(result, "Events").iloc[0].interval_end_row == 3


def test_transitions_ignore_repeated_constants_and_preserve_negative_signs():
    result = analyze_recording(telemetry(Signs=[0, 35, 35, 0, -3, 0]))
    log = changes(result, "Signs")
    assert log.row.tolist() == [1, 3, 4, 5]
    assert log.previous_value.tolist() == [0, 35, 0, -3]
    assert log.new_value.tolist() == [35, 0, -3, 0]
    assert log.loc[0, "occurrence"] == 1
    assert pd.isna(log.loc[1, "occurrence"])
    assert log.loc[2, "occurrence"] == 2
    assert json.loads(metric(result, "Signs", "ordered_non_default_values")) == [35, -3]


def test_intersection_order_and_activity_across_nonzero_value_changes():
    result = analyze_recording(telemetry(Intersections=[0, 1, 2, 0, 1, 1, 0]))
    active = intervals(result, "Intersections", "active")
    assert active.occurrence.tolist() == [1, 2]
    assert active.interval_start_row.tolist() == [1, 4]
    assert active.interval_end_row.tolist() == [3, 6]
    assert active.interval_start_time.tolist() == [0.5, 2.0]
    assert active.interval_end_time.tolist() == [1.5, 3.0]
    assert pd.isna(active.loc[0, "value"])  # Active run contains two codes.
    assert len(intervals(result, "Intersections", "value")) == 3


def test_event_pulses_require_observed_defaults_on_both_sides():
    result = analyze_recording(telemetry(Events=[7, 0, 1, 0, 2, 2, 0, 3, 4, 0, 5]))
    runs = intervals(result, "Events")
    assert runs.value.tolist() == [7, 1, 2, 3, 4, 5]
    assert runs.occurrence.tolist() == [1, 2, 3, 4, 5, 6]
    assert runs.is_pulse.tolist() == [False, True, False, False, False, False]
    assert runs.left_censored.tolist() == [True, False, False, False, False, False]
    assert runs.right_censored.tolist() == [False, False, False, False, False, True]
    pulse = runs.iloc[1]
    assert (pulse.interval_start_row, pulse.interval_end_row) == (2, 3)
    assert pulse.duration_s == 0.0
    assert metric(result, "Events", "pulse_count") == 1


def test_missing_discrete_samples_break_intervals_without_fabricating_transitions():
    result = analyze_recording(telemetry(Signs=[0, 35, None, 35, 0]))
    runs = intervals(result, "Signs")
    assert runs.interval_start_row.tolist() == [1, 3]
    assert runs.interval_end_row.tolist() == [2, 4]
    assert not runs.is_pulse.any()
    assert changes(result, "Signs").row.tolist() == [1, 4]
    assert result.transitions.loc[result.transitions.row.eq(3), "record_type"].item() == "initial_state"


def test_speed_limits_keep_ordered_changes_including_zero():
    result = analyze_recording(telemetry(**{"Speed Limit": [35, 35, 10, 10, 0, 25]}))
    log = changes(result, "Speed Limit")
    assert log.previous_value.tolist() == [35, 10, 0]
    assert log.new_value.tolist() == [10, 0, 25]


def test_task_driver_response_and_user_fields_preserve_text_and_negative_codes():
    result = analyze_recording(telemetry(**{
        "Task Marker": [0, 8, 0], "Driver Response 6": [0, -2, 0],
        "User5": ["0", "cue", "0"], "User17": [0, 11, 0], "Collision": [0, 1, 0],
    }))
    for column, expected in [("Task Marker", 8), ("Driver Response 6", -2),
                             ("User5", "cue"), ("User17", 11), ("Collision", 1)]:
        assert json.loads(metric(result, column, "unique_non_default_values")) == [expected]
        run = intervals(result, column).iloc[0]
        assert run.value == expected
        assert run.is_pulse


def test_combined_log_is_chronological_with_stable_ties_and_missing_times_last():
    frame = telemetry(**{
        "Signs": [0, 35, 0, -3], "Events": [0, 1, 0, 1],
        "Speed Limit": [35, 10, 10, 25], "Brake Pedal Force": [0, 0, 240, 0],
    })
    frame["Sim Time"] = [8, 2, 5, None]
    result = analyze_recording(frame)
    log = result.transitions
    assert log.sim_time.dropna().is_monotonic_increasing
    assert log.loc[log.sim_time.eq(2), "column"].tolist() == ["Signs", "Speed Limit", "Events"]
    assert log.iloc[-3:].sim_time.isna().all()
    peak = log.loc[log.record_type.eq("braking_peak")].iloc[0]
    assert peak.sim_time == 5
    assert peak.new_value == 240
    assert "Nonmonotonic" in metric(result, "Sim Time", "advisory")


@pytest.mark.parametrize("times", [[0, 2, 1, 4], [0, None, 2, 4]])
def test_invalid_active_timestamps_do_not_invent_durations(times):
    frame = telemetry(Events=[0, 1, 1, 0])
    frame["Sim Time"] = times
    result = analyze_recording(frame)
    assert pd.isna(intervals(result, "Events").iloc[0].duration_s)


def test_rt_positive_intervals_are_descriptive_and_raw_changes_are_preserved():
    result = analyze_recording(telemetry(**{
        "Reaction Time": [0, 0.4, 0.4, 0.6, 0.6, 0, 0.7, 0.7, 0],
    }))
    assert metric(result, "Reaction Time", "candidate_event_count") == 1
    assert metric(result, "Reaction Time", "update_episode_count") == 1
    active = intervals(result, "Reaction Time", "active")
    assert active.interval_start_row.tolist() == [1, 6]
    assert active.interval_end_row.tolist() == [5, 8]
    assert len(intervals(result, "Reaction Time")) == 3
    log = changes(result, "Reaction Time")
    assert log.row.tolist() == [1, 3, 5, 6, 8]
    assert log.loc[1, "previous_value"] == 0.4
    assert log.loc[1, "new_value"] == 0.6
    assert result.rt_update_episodes.iloc[0].transition_count == len(log)


def test_reaction_time_missing_sample_preserves_intervals_and_breaks_episode_continuity():
    result = analyze_recording(telemetry(**{"Reaction Time": [0, 0.5, 0.5, None, 0.5, 0]}))
    assert metric(result, "Reaction Time", "candidate_event_count") == 2
    active = intervals(result, "Reaction Time", "active")
    assert active.interval_start_row.tolist() == [1, 4]
    assert active.interval_end_row.tolist() == [3, 5]
    assert active.right_censored.tolist() == [True, False]
    assert active.left_censored.tolist() == [False, True]
    assert not active.is_pulse.any()
    resumed = result.transitions.loc[result.transitions.row.eq(4)].iloc[0]
    assert resumed.record_type == "initial_state"
    assert pd.isna(resumed.previous_value)
    # After the gap, row 4 still carries the prior value; row 5 is the next update.
    assert result.rt_update_episodes.start_row.tolist() == [1, 5]


def test_rt_rises_then_persists_then_rises_again_produces_two_update_episodes():
    frame = telemetry(**{"Reaction Time": [0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.4, 0.5, 0.5]})
    frame["Sim Time"] = [310, 311, 311.2, 311.4, 312.5, 461, 462, 462.2, 464]
    original = frame.copy(deep=True)
    result = analyze_recording(frame)
    pd.testing.assert_frame_equal(frame, original)
    episodes = result.rt_update_episodes
    assert len(intervals(result, "Reaction Time", "active")) == 1
    assert metric(result, "Reaction Time", "update_episode_count") == 2
    assert metric(result, "Reaction Time", "candidate_event_count") == 2
    assert episodes.columns.tolist() == RT_EPISODE_COLUMNS
    assert episodes.occurrence.tolist() == [1, 2]
    assert episodes.start_row.tolist() == [1, 6]
    assert episodes.end_row.tolist() == [4, 8]
    assert episodes.start_time.tolist() == [311, 462]
    assert episodes.end_time.tolist() == [312.5, 464]
    assert episodes.last_transition_row.tolist() == [3, 7]
    assert episodes.last_transition_time.tolist() == [311.4, 462.2]
    assert episodes.starting_rt_value.tolist() == [0, 0.3]
    assert episodes.ending_rt_value.tolist() == [0.3, 0.5]
    assert episodes.rt_value_change.tolist() == pytest.approx([0.3, 0.2])
    assert episodes.transition_count.tolist() == [3, 2]
    assert episodes.duration_s.tolist() == pytest.approx([0.4, 0.2])
    assert episodes.rt_transition_gap_s.tolist() == [1, 1]
    assert episodes.right_censored.tolist() == [False, False]
    assert not ((result.summary.column == "Reaction Time") & (result.summary.metric == "advisory")).any()
    assert changes(result, "Reaction Time").row.tolist() == [1, 2, 3, 6, 7]


def test_rt_isolated_decrease_and_reset_are_updates_even_without_positive_boundaries():
    frame = telemetry(**{"Reaction Time": [0.8, 0.8, 0.6, 0.6, 0, 0]})
    frame["Sim Time"] = [0, 1, 2, 5, 6, 8]
    episodes = analyze_recording(frame).rt_update_episodes
    assert episodes.starting_rt_value.tolist() == [0.8, 0.6]
    assert episodes.ending_rt_value.tolist() == [0.6, 0]
    assert episodes.rt_value_change.tolist() == pytest.approx([-0.2, -0.6])
    assert episodes.transition_count.tolist() == [1, 1]


@pytest.mark.parametrize("boundary", ["missing_rt", "missing_time", "time_reversal"])
def test_rt_episodes_do_not_cluster_across_unknown_or_reversed_time(boundary):
    frame = telemetry(**{"Reaction Time": [0, 0.1, 0.1, 0.2, 0.3, 0.3]})
    frame["Sim Time"] = [0, 0.1, 0.2, 0.3, 0.4, 2]
    if boundary == "missing_rt":
        frame.loc[2, "Reaction Time"] = float("nan")
    elif boundary == "missing_time":
        frame.loc[2, "Sim Time"] = float("nan")
    else:
        frame["Sim Time"] = [0, 1, 0.1, 0.2, 0.3, 2]
    episodes = analyze_recording(frame).rt_update_episodes
    assert len(episodes) == 2
    assert episodes.loc[0, "start_row"] == 1
    assert episodes.loc[0, "right_censored"]


def test_rt_final_update_is_reported_with_exclusive_eof_end_and_censor_flag():
    result = analyze_recording(telemetry(**{"Reaction Time": [0.1, 0.1, 0.2]}))
    episode = result.rt_update_episodes.iloc[0]
    assert episode.start_row == 2
    assert episode.end_row == 3
    assert episode.last_transition_time == 1.0
    assert pd.isna(episode.end_time)
    assert episode.duration_s == 0
    assert episode.right_censored


@pytest.mark.parametrize("values", [[0] * 4, [0.4] * 4], ids=["zero", "persistent-positive"])
def test_constant_rt_values_do_not_create_update_episodes(values):
    result = analyze_recording(telemetry(**{"Reaction Time": values}))
    assert result.rt_update_episodes.empty
    assert metric(result, "Reaction Time", "candidate_event_count") == 0
    assert "observed 0" in metric(result, "Reaction Time", "advisory")


@pytest.mark.parametrize("gap,counts", [(1, [2, 1]), (2, [3]), (0.25, [1, 1, 1])])
def test_rt_gap_uses_seconds_and_does_not_change_raw_transitions(gap, counts):
    frame = telemetry(**{"Reaction Time": [0, 0.1, 0.2, 0.3, 0.3]})
    frame["Sim Time"] = [0, 10, 11, 12.001, 14]
    result = analyze_recording(frame, rt_transition_gap_s=gap)
    assert result.rt_update_episodes.transition_count.tolist() == counts
    pd.testing.assert_frame_equal(result.transitions, analyze_recording(frame).transitions)


@pytest.mark.parametrize("option,value", [
    ("rt_transition_gap_s", 0), ("brake_threshold", -1),
    ("post_rt_window_s", float("nan")), ("events_near_window_s", float("inf")),
    ("post_event_window_s", 0),
])
def test_invalid_analysis_settings_are_rejected(option, value):
    with pytest.raises(ValueError, match="finite and greater than zero"):
        analyze_recording(telemetry(Events=[0]), **{option: value})


def test_missing_time_preserves_raw_changes_without_inventing_episodes():
    frame = pd.DataFrame({"Events": [0, 1, 0], "Reaction Time": [0, 0.1, 0.2]})
    result = analyze_recording(frame)
    assert metric(result, "Signs", "available") is False
    assert metric(result, "Sim Time", "available") is False
    assert result.transitions.sim_time.isna().all()
    assert intervals(result, "Events").duration_s.isna().all()
    assert result.rt_update_episodes.empty
    assert changes(result, "Reaction Time").row.tolist() == [1, 2]
    assert metric(result, "Reaction Time", "untimed_transition_count") == 2


@pytest.mark.parametrize("threshold", [None, 180, 300])
def test_braking_preserves_raw_peaks_with_optional_threshold(threshold):
    frame = telemetry(**{"Brake Pedal Force": [0, 180, 245.5, 245.5, 20]})
    original = frame.copy(deep=True)
    result = analyze_recording(frame, brake_threshold=threshold)
    pd.testing.assert_frame_equal(frame, original)
    assert metric(result, "Brake Pedal Force", "maximum") == 245.5
    assert metric(result, "Brake Pedal Force", "peak_row") == 2
    runs = intervals(result, "Brake Pedal Force", "strong_braking")
    if threshold == 180:
        assert runs.interval_start_row.tolist() == [1]
        assert runs.interval_end_row.tolist() == [4]
        assert runs.peak_value.tolist() == [245.5]
    else:
        assert runs.empty


def test_empty_recording_has_fixed_schemas_and_no_events():
    result = analyze_recording(pd.DataFrame(columns=["Signs", "Reaction Time", "Brake Pedal Force"]))
    assert result.transitions.empty
    assert result.intervals.empty
    assert result.transitions.columns.tolist() == TRANSITION_COLUMNS
    assert result.intervals.columns.tolist() == INTERVAL_COLUMNS
    assert result.summary.columns.tolist() == SUMMARY_COLUMNS
    assert result.rt_update_episodes.columns.tolist() == RT_EPISODE_COLUMNS
    assert metric(result, "Reaction Time", "candidate_event_count") == 0
    assert pd.isna(metric(result, "Brake Pedal Force", "maximum"))


def test_explicit_braking_threshold_generates_intervals_and_preserves_peaks():
    frame = telemetry(**{"Brake Pedal Force": [0, 170, 220, None, 190, 100, 0]})
    result = analyze_recording(frame, brake_threshold=170)
    runs = intervals(result, "Brake Pedal Force", "strong_braking")
    assert runs.interval_start_row.tolist() == [1, 4]
    assert runs.interval_end_row.tolist() == [3, 5]
    assert runs.peak_value.tolist() == [220, 190]
    assert runs.peak_row.tolist() == [2, 4]
    assert runs.brake_threshold.tolist() == [170, 170]
    assert runs.right_censored.tolist() == [True, False]
    assert runs.left_censored.tolist() == [False, True]
    assert runs.is_pulse.isna().all()
    assert metric(result, "Brake Pedal Force", "brake_threshold") == 170
    assert result.transitions.new_value.tolist() == [220, 190]


def test_metadata_loader_keeps_leading_zeros_and_uses_filename_fallback(tmp_path: Path):
    path = tmp_path / "participant_002_T2_2025-04-22_drive_1_cleaned.csv"
    frame = telemetry(Events=[0, 1, 0])
    frame["participant_id"] = "002"
    frame.to_csv(path, index=False)
    before = path.read_bytes()
    result = analyze_file(path)
    assert result.transitions.participant_id.unique().tolist() == ["002"]
    assert result.transitions.visit.unique().tolist() == ["T2"]
    assert result.transitions.drive.unique().tolist() == [1]
    assert result.transitions.recording_id.unique().tolist() == [path.stem]
    assert path.read_bytes() == before


def test_conflicting_metadata_is_rejected():
    frame = telemetry(Events=[0, 1])
    frame["participant_id"] = ["002", "003"]
    with pytest.raises(ValueError, match="Conflicting participant_id"):
        analyze_recording(frame)
    frame["participant_id"] = "003"
    with pytest.raises(ValueError, match="metadata and filename"):
        analyze_recording(frame, "participant_002_T2_2025-04-22_drive_1_cleaned")


def test_module_entrypoint_with_synthetic_csv(tmp_path: Path):
    path = tmp_path / "synthetic.csv"
    telemetry(Events=[0, 1, 0]).to_csv(path, index=False)
    output = tmp_path / "reports"
    completed = subprocess.run(
        [sys.executable, "-m", "src.analysis.marker_discovery", "--file", str(path), "--output-dir", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == ["Created:", *[
        str(output / f"synthetic_{name}.csv") for name in ("transitions", "intervals", "summary", "rt_update_episodes")
    ]]


def test_batch_comparison_has_one_row_per_recording_and_expected_patterns(tmp_path: Path):
    first = comparison_drive()
    second = comparison_drive()
    second["Sim Time"] += 100
    second.loc[second.Events.eq(1) | second.Events.eq(3), "Events"] = 0
    second.loc[7, "Events"] = 1
    second.loc[10:12, "Brake Pedal Force"] = 0
    second.loc[10:12, "Acceleration"] = 1
    paths = [write_comparison_drive(tmp_path, first), write_comparison_drive(tmp_path, second, "003")]
    originals = [path.read_bytes() for path in paths]
    result = compare_recordings(paths[::-1])
    assert result.columns.tolist() == COMPARISON_DETAIL_COLUMNS
    assert len(result) == 2
    assert result.participant_id.tolist() == ["002", "003"]
    assert result.visit.tolist() == ["T2", "T2"]
    assert result.drive.tolist() == [1, 1]
    assert result.rt_episode_count.tolist() == [2, 2]
    row = result.iloc[0]
    assert row.rt1_start_time == 1
    # The follow-up window anchors at the last update, not the exclusive boundary.
    assert row.rt1_end_time == 1.5
    assert row.rt1_last_transition_time == 1.2
    assert row.rt1_value_change == pytest.approx(0.2)
    assert row.rt2_start_time == 10
    assert row.rt2_value_change == pytest.approx(0.2)
    assert row.rt1_peak_brake_after == 152
    assert row.rt1_peak_deceleration_after == -8.46
    assert row.rt2_peak_brake_after == 73
    assert row.rt2_peak_deceleration_after == -5.03
    assert result.events_1_near_rt1.tolist() == [True, False]
    assert result.braking_after_rt1.tolist() == [True, True]
    assert result.braking_after_rt2.tolist() == [True, False]
    assert result.events_3_present.tolist() == [True, False]
    assert row.events_1_start_time == 1
    assert row.events_3_start_time == 3
    assert row.events_4_start_time == 5
    assert row.events_5_start_time == 6
    assert json.loads(row.events_values_in_order) == [1, 3, 4, 5]
    assert json.loads(row.sign_values_in_order) == [35, -3, 35]
    assert json.loads(row.speed_limit_values_in_order) == [35, 10, 0, 20, 35]
    assert row.intersection_count == 2
    ordered = json.loads(row.intersection_intervals_in_order)
    assert [item["occurrence"] for item in ordered] == [1, 2]
    assert [item["interval_start_time"] for item in ordered] == [1.5, 14]
    assert ordered[0]["interval_end_row"] == 5
    assert [path.read_bytes() for path in paths] == originals


def test_batch_post_rt_window_is_configurable_and_excludes_update_sample(tmp_path: Path):
    frame = comparison_drive()
    frame.loc[2, "Brake Pedal Force"] = 1000  # Final RT update frame is excluded.
    frame.loc[5, "Brake Pedal Force"] = 250  # Outside the short window.
    frame.loc[5, "Acceleration"] = -12
    path = write_comparison_drive(tmp_path, frame)
    short = compare_recordings([path], post_rt_window_s=0.5).iloc[0]
    long = compare_recordings([path], post_rt_window_s=2).iloc[0]
    assert short.rt1_peak_brake_after == 152
    assert short.rt1_peak_deceleration_after == -8.46
    assert short.rt1_after_sample_count == 1
    assert long.rt1_peak_brake_after == 250  # Values >170 remain raw.
    assert long.rt1_peak_deceleration_after == -12
    assert long.post_rt_window_s == 2
    assert long.rt1_after_window_complete


def test_batch_event_proximity_uses_any_overlapping_interval_and_configured_tolerance(tmp_path: Path):
    frame = comparison_drive()
    frame["Events"] = 0
    frame.loc[0, "Events"] = 1
    frame.loc[3:4, "Events"] = 1
    path = write_comparison_drive(tmp_path, frame)
    assert compare_recordings([path], events_near_window_s=0.5).iloc[0].events_1_near_rt1
    assert not compare_recordings([path], events_near_window_s=0.1).iloc[0].events_1_near_rt1
    # The first occurrence is still reported even when a later occurrence is nearer.
    assert compare_recordings([path]).iloc[0].events_1_start_time == 0


def test_batch_missing_signals_or_episodes_are_na_but_observed_absence_is_zero(tmp_path: Path):
    minimal = write_comparison_drive(tmp_path, telemetry(Events=[0, 0]), "002")
    constant = write_comparison_drive(tmp_path, telemetry(**{"Reaction Time": [0.2, 0.2]}), "003")
    table = compare_recordings([minimal, constant])
    assert pd.isna(table.iloc[0].rt_episode_count)
    assert not table.iloc[0].events_1_present
    assert pd.isna(table.iloc[0].events_1_start_time)
    assert pd.isna(table.iloc[0].events_1_near_rt1)
    assert table.iloc[1].rt_episode_count == 0
    assert table.braking_after_rt1.isna().all()
    assert table.braking_after_rt2.isna().all()
    assert table.intersection_count.isna().all()
    assert "Reaction Time" in json.loads(table.iloc[0].missing_columns)


def test_batch_untimed_rt_count_is_unknown_and_time_reversal_limits_window(tmp_path: Path):
    frame = comparison_drive()
    frame.loc[1, "Sim Time"] = float("nan")
    row = compare_recordings([write_comparison_drive(tmp_path, frame)]).iloc[0]
    assert pd.isna(row.rt_episode_count)
    assert "lack valid timestamps" in row.notes
    frame = comparison_drive()
    frame.loc[3, "Sim Time"] = 0  # Immediately after RT1: no safe post-window samples.
    row = compare_recordings([write_comparison_drive(tmp_path, frame)]).iloc[0]
    assert pd.isna(row.rt1_peak_brake_after)
    assert pd.isna(row.braking_after_rt1)
    assert not row.rt1_after_window_complete


def test_batch_directory_and_glob_discovery_are_sorted_recursive_and_cleaned_only(tmp_path: Path):
    first = write_comparison_drive(tmp_path / "cleaned_a", comparison_drive())
    second = write_comparison_drive(tmp_path / "cleaned_b", comparison_drive(), "003")
    (tmp_path / "cleaned_a" / "header_mapping.csv").write_text("A,B\n", encoding="utf-8")
    expected = [first.resolve(), second.resolve()]
    assert discover_cleaned_files(directory=tmp_path) == expected
    assert discover_cleaned_files(pattern=str(tmp_path / "cleaned_*" / "*drive_1_cleaned.csv")) == expected
    assert discover_cleaned_files(pattern=str(tmp_path / "cleaned_*" / "*drive_1_cleaned.csv").replace("/", "\\")) == expected


@pytest.mark.parametrize("brake,acceleration,expected", [
    (0, 0, False), (None, -0.001, True), (0.001, None, True), (None, 0, None),
])
def test_batch_braking_flags_use_signs_without_force_cutoff_or_false_missingness(
    tmp_path: Path, brake, acceleration, expected,
):
    frame = comparison_drive()
    for column, value in [("Brake Pedal Force", brake), ("Acceleration", acceleration)]:
        if value is None:
            frame = frame.drop(columns=column)
        else:
            frame[column] = value
    row = compare_recordings([write_comparison_drive(tmp_path, frame)]).iloc[0]
    if expected is None:
        assert pd.isna(row.braking_after_rt1)
    else:
        assert bool(row.braking_after_rt1) == expected
    if brake is not None:
        assert row.rt1_peak_brake_after == brake
    if acceleration is not None:
        assert row.rt1_peak_deceleration_after == acceleration


def test_single_file_cli_is_quiet_and_lists_exports(tmp_path, capsys):
    path = write_comparison_drive(tmp_path, comparison_drive())
    assert main(["--file", str(path)]) == 0
    assert list(tmp_path.iterdir()) == [path]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    write_comparison_drive(tmp_path, comparison_drive(), drive=11)
    output = tmp_path / "reports"
    assert main(["--file", str(path), "--output-dir", str(output), "--brake-threshold", "180"]) == 0
    schemas = {"transitions": TRANSITION_COLUMNS, "intervals": INTERVAL_COLUMNS,
               "summary": SUMMARY_COLUMNS, "rt_update_episodes": RT_EPISODE_COLUMNS}
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == ["Created:", *[str(output / f"{path.stem}_{name}.csv") for name in schemas]]
    assert {file.name for file in output.iterdir()} == {f"{path.stem}_{name}.csv" for name in schemas}
    for name, schema in schemas.items():
        table = pd.read_csv(output / f"{path.stem}_{name}.csv", dtype={"participant_id": str})
        assert table.columns.tolist() == schema
        assert table.participant_id.unique().tolist() == ["002"]
        assert table.drive.unique().tolist() == [1]


@pytest.mark.parametrize("case", ["missing", "malformed", "no-matches", "conflicting-modes", "single-scenario"])
def test_cli_rejects_invalid_inputs(tmp_path, capsys, case):
    path = tmp_path / "input.csv"
    path.write_text('Sim Time,Events\n0,1\n1,"unterminated', encoding="utf-8")
    args = {
        "missing": ["--file", str(tmp_path / "missing.csv")],
        "malformed": ["--file", str(path)],
        "no-matches": ["--glob", str(tmp_path / "absent*.csv")],
        "conflicting-modes": ["--file", str(path), "--directory", str(tmp_path)],
        "single-scenario": ["--file", str(path), "--scenario", "1"],
    }[case]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["--files", "--glob", "--directory"])
def test_batch_cli_selects_restart_and_exports_schemas(tmp_path, capsys, mode):
    inputs = tmp_path / "inputs"
    base = write_comparison_drive(inputs, comparison_drive(), drive=1)
    write_comparison_drive(inputs, comparison_drive(), drive=11)
    selection = {
        "--files": str(base), "--glob": str(inputs / "*drive_1_cleaned.csv"), "--directory": str(inputs),
    }[mode]
    assert main([mode, selection, "--scenario", "1"]) == 0
    assert not (tmp_path / "reports").exists()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    output = tmp_path / "reports"
    assert main([mode, selection, "--scenario", "1", "--output-dir", str(output),
                 "--post-rt-window", "2", "--events-near-window", "0.5"]) == 0
    schemas = {"scenario_comparison": COMPARISON_COLUMNS, "recording_selection": SELECTION_COLUMNS,
               "recording_chain_qc": RECORDING_CHAIN_COLUMNS, "scenario_comparison_details": COMPARISON_DETAIL_COLUMNS}
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == ["Created:", *[str(output / f"{name}.csv") for name in BATCH_OUTPUT_NAMES]]
    assert {file.name for file in output.iterdir()} == {f"{name}.csv" for name in schemas}
    for name, schema in schemas.items():
        assert pd.read_csv(output / f"{name}.csv").columns.tolist() == schema
    comparison = pd.read_csv(output / "scenario_comparison.csv", dtype={"participant_id": str})
    assert comparison.columns.tolist() == EXPECTED_COMPARISON_COLUMNS
    assert comparison.participant_id.tolist() == ["002"]
    assert comparison.drive.tolist() == [11]
    assert comparison.scenario.tolist() == [1]
    details = pd.read_csv(output / "scenario_comparison_details.csv", dtype={"participant_id": str})
    pd.testing.assert_frame_equal(comparison, details.loc[:, EXPECTED_COMPARISON_COLUMNS])
    assert details.post_rt_window_s.tolist() == [2]
    assert details.events_near_window_s.tolist() == [0.5]


@pytest.mark.parametrize("drives,selected,scenario", [
    ([1], 1, 1), ([1, 11], 11, 1), ([11], 11, 1),
    ([2], 2, 2), ([2, 21], 21, 2), ([21], 21, 2), ([3], 3, 3),
])
def test_preferred_recordings_replace_original_without_stitching(tmp_path, drives, selected, scenario):
    paths = [write_comparison_drive(
        tmp_path, comparison_drive() if drive in (11, 21) else comparison_drive().iloc[:5], drive=drive,
    ) for drive in drives]
    originals = [path.read_bytes() for path in paths]
    inventory = compare_recordings(paths, preferred_only=False)
    preferred, audit = select_preferred_recordings(inventory)
    assert preferred.drive.tolist() == [selected]
    assert preferred.scenario.tolist() == [scenario]
    assert preferred.row_count.tolist() == [15 if selected in (11, 21) else 5]
    assert audit.selected_drive.tolist() == [selected]
    assert audit.columns.tolist() == SELECTION_COLUMNS
    if len(drives) == 2:
        assert audit.superseded_drive.tolist() == [drives[0]]
        chain = report_recording_chains(inventory)
        assert chain.base_drive.tolist() == [drives[0]]
        assert chain.related_drive.tolist() == [selected]
        assert chain.base_row_count.tolist() == [5]
    else:
        assert audit.superseded_drive.isna().all()
    assert [path.read_bytes() for path in paths] == originals


def test_restart_selection_and_chain_qc_stay_within_participant_and_visit(tmp_path):
    groups = [("002", "T1", 1), ("002", "T2", 11), ("003", "T1", 11),
              ("004", "T1", 2), ("005", "T1", 21)]
    paths = [write_comparison_drive(tmp_path, comparison_drive(), participant, drive, visit)
             for participant, visit, drive in groups]
    inventory = compare_recordings(paths, preferred_only=False)
    assert report_recording_chains(inventory).empty
    preferred, _ = select_preferred_recordings(inventory)
    assert len(preferred) == len(groups)
    assert preferred.loc[preferred.participant_id.eq("002") & preferred.visit.eq("T1"), "drive"].item() == 1


def test_unmapped_drive_numbers_are_retained_without_restart_pairs(tmp_path):
    drives = [1, 11, 2, 21, 3, 12, 22, 31, 111]
    paths = [write_comparison_drive(tmp_path, telemetry(Events=[0, 1, 0]), drive=drive) for drive in drives]
    inventory = compare_recordings(paths, preferred_only=False)
    chains = report_recording_chains(inventory)
    assert list(zip(chains.base_drive, chains.related_drive)) == [(1, 11), (2, 21)]
    preferred, _ = select_preferred_recordings(inventory)
    assert sorted(preferred.drive.tolist()) == [3, 11, 12, 21, 22, 31, 111]
    unmapped = preferred.loc[preferred.drive.isin([12, 22, 31, 111])]
    assert unmapped.scenario.isna().all()
    assert unmapped.selection_reason.eq("unmapped_drive_retained").all()


def test_anonymous_recordings_are_not_superseded_or_paired(tmp_path):
    paths = []
    for drive in (1, 11):
        path = tmp_path / f"anonymous_{drive}_cleaned.csv"
        pd.DataFrame({"drive": [drive], "Events": [0]}).to_csv(path, index=False)
        paths.append(path)
    inventory = compare_recordings(paths, preferred_only=False)
    assert report_recording_chains(inventory).empty
    preferred, _ = select_preferred_recordings(inventory)
    assert sorted(preferred.drive.tolist()) == [1, 11]
    assert preferred.selection_reason.eq("insufficient_metadata").all()


def test_duplicate_recordings_remain_traceable_in_selection_and_chain_qc(tmp_path):
    paths = [write_comparison_drive(tmp_path / copy, comparison_drive(), drive=drive)
             for copy in ("a", "b") for drive in (1, 11)]
    inventory = compare_recordings(paths, preferred_only=False)
    preferred, audit = select_preferred_recordings(inventory)
    assert preferred.drive.tolist() == [11, 11]
    assert preferred.source_file.nunique() == 2
    assert audit.selected_source_file.nunique() == 2
    assert audit.superseded_source_file.nunique() == 2
    assert audit.notes.str.contains("Multiple recordings").all()
    chains = report_recording_chains(inventory)
    assert len(chains) == 4
    assert chains.notes.str.contains("Ambiguous participant/visit group").all()


def test_recording_chain_qc_missing_markers_and_endpoint_times_are_na(tmp_path: Path):
    frame = pd.DataFrame({"Sim Time": [None, 5, 3, None]})
    paths = [write_comparison_drive(tmp_path, frame, drive=drive) for drive in (1, 11)]
    comparison = compare_recordings(paths, preferred_only=False)
    row = report_recording_chains(comparison).iloc[0]
    assert row.base_row_count == 4
    # Missing endpoints remain unknown despite valid interior timestamps.
    assert pd.isna(row.base_start_time)
    assert pd.isna(row.base_end_time)
    assert pd.isna(row.related_start_time)
    assert pd.isna(row.base_rt_episode_count)
    assert pd.isna(row.related_event_values)
    assert pd.isna(row.base_intersection_count)
    assert pd.isna(row.related_sign_values)
    assert "Nonmonotonic" in row.notes


def test_empty_scenario_selection_writes_header_only_outputs(tmp_path: Path, capsys):
    base = write_comparison_drive(tmp_path, comparison_drive(), drive=1)
    output = tmp_path / "empty"
    assert main(["--files", str(base), "--scenario", "3", "--output-dir", str(output)]) == 0
    assert pd.read_csv(output / "scenario_comparison.csv").empty
    assert pd.read_csv(output / "recording_selection.csv").empty
    assert pd.read_csv(output / "scenario_comparison.csv").columns.tolist() == EXPECTED_COMPARISON_COLUMNS
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == ["Created:", *[str(output / f"{name}.csv") for name in BATCH_OUTPUT_NAMES]]


@pytest.mark.parametrize("first_updates,first_status", [
    ([(310, 0.2), (310.5, 0.5)], "associated"),
    ([], "rt_missing"),
])
def test_route_context_normal_and_participant_023_style_regression(tmp_path: Path, first_updates, first_status):
    start_value = first_updates[-1][1] if first_updates else 0
    frame = route_context_drive(first_updates + [
        (482.6, start_value + 0.5), (483.2, start_value + 1), (484, start_value + 1.467),
    ])
    path = write_comparison_drive(tmp_path, frame, "023")
    before = path.read_bytes()
    row = compare_recordings([path]).iloc[0]
    assert row.collision1_event_time == 309.4
    assert row.collision1_rt_status == first_status
    assert row.collision1_peak_brake == 127
    assert row.collision1_peak_deceleration == -7.89
    if first_updates:
        assert row.collision1_rt_time == 310
        assert row.collision1_rt_value_change == 0.5
        assert row.rt_episode_count == 2
    else:
        assert pd.isna(row.collision1_rt_time)
        assert pd.isna(row.collision1_rt_value_change)
        assert row.rt_episode_count == 1
        # The sole raw RT episode is not automatically called collision 1.
        assert row.rt1_start_time == 482.6
        assert pd.isna(row.rt2_start_time)
    assert row.collision2_candidate_rt_time == 482.6
    assert row.collision2_candidate_rt_value_change == pytest.approx(1.467)
    assert row.collision2_candidate_peak_brake == 51
    assert row.collision2_candidate_peak_deceleration == -3.86
    assert row.collision2_candidate_status == "candidate"
    assert row.collision2_candidate_count == 1
    assert path.read_bytes() == before


def test_route_context_configuration_does_not_change_raw_detection():
    frame = route_context_drive([(310, 0.2), (310.5, 0.5), (482.6, 1), (483.2, 1.5), (484, 1.967)])
    recording = "participant_023_T2_2025-04-22_drive_1_cleaned"
    normal = analyze_recording(frame, recording)
    short = analyze_recording(frame, recording, events_near_window_s=0.1, post_event_window_s=3)
    for attribute in ("transitions", "intervals", "rt_update_episodes"):
        pd.testing.assert_frame_equal(getattr(normal, attribute), getattr(short, attribute))
    assert metric(normal, "route_context", "collision1_rt_status") == "associated"
    assert metric(short, "route_context", "collision1_rt_status") == "rt_missing"
    assert metric(normal, "route_context", "collision1_peak_brake") == 127
    assert metric(short, "route_context", "collision1_peak_brake") == 0


def test_route_context_requires_event_boundaries_and_excludes_episodes_after_event3(tmp_path: Path):
    frame = route_context_drive([(310, 0.2), (616, 0.7), (616.5, 1.2)])
    path = write_comparison_drive(tmp_path, frame)
    row = compare_recordings([path]).iloc[0]
    assert row.collision1_rt_status == "associated"
    assert row.collision2_candidate_status == "no_candidate"
    frame.loc[frame.Events.eq(3), "Events"] = 0
    row = compare_recordings([write_comparison_drive(tmp_path, frame)]).iloc[0]
    assert row.collision1_rt_status == "associated"
    assert row.collision2_candidate_status == "boundary_missing"


def test_route_context_repeated_event1_markers_are_ambiguous(tmp_path: Path):
    frame = route_context_drive([(310, 0.2), (482.6, 0.7)])
    frame.loc[frame["Sim Time"].eq(316), "Events"] = 1
    row = compare_recordings([write_comparison_drive(tmp_path, frame)]).iloc[0]
    assert row.collision1_rt_status == "ambiguous_event_marker"
    assert row.collision2_candidate_status == "ambiguous_event_marker"
    assert pd.isna(row.collision2_candidate_rt_time)


def test_event_context_neighbors_use_time_order_not_row_distance():
    frame = pd.DataFrame({
        "Sim Time": [0, 9, 2, 5, 3, 8, 10], "Events": [0, 0, 0, 3, 0, 0, 0],
        "Signs": [0, 35, 0, 0, -3, 0, 9], "Speed Limit": [35, 10, 20, 20, 30, 40, 50],
    })
    context = build_event_context(frame, analyze_recording(frame))
    row = context.iloc[0]
    assert row.previous_sign_value == -3
    assert row.previous_sign_time == 3
    assert row.next_sign_value == 35
    assert row.next_sign_time == 9
    assert row.previous_speed_limit == 30
    assert row.previous_speed_change_time == 3
    assert row.next_speed_limit == 40
    assert row.next_speed_change_time == 8
    assert row.speed_limit_at_event_start == 20


def test_event_context_reports_every_nonzero_value_interval_and_original_occurrence():
    frame = telemetry(Events=[0, 1, 1, 3, 0, 3, 0, -2, 0])
    result = analyze_recording(frame, "participant_023_T2_2025-06-12_drive_11_cleaned")
    context = build_event_context(frame, result)
    assert context.columns.tolist() == EVENT_CONTEXT_COLUMNS
    assert context.event_value.tolist() == [1, 3, 3, -2]
    assert context.event_occurrence.tolist() == [1, 2, 3, 4]
    assert context.participant_id.unique().tolist() == ["023"]
    assert context.drive.unique().tolist() == [11]
    assert context.iloc[0].event_start_time == 0.5
    assert context.iloc[0].event_end_time == 1.5
    assert context.iloc[0].event_duration_s == 0.5


@pytest.mark.parametrize("updates,first_status,second_status,count", [
    ([], "rt_missing", "no_candidate", 0),
    ([(310, 0.2), (310.5, 0.5)], "associated", "no_candidate", 0),
    ([(400, 0.2), (482.6, 0.7)], "rt_missing", "ambiguous", 2),
    ([(308, 0.1), (310, 0.3), (482.6, 0.8)], "ambiguous", "candidate", 1),
    ([(302, 0.1), (310, 0.3), (482.6, 0.8)], "associated", "candidate", 1),
])
def test_collision_association_uses_route_evidence_instead_of_rt_ordinal(
    tmp_path, updates, first_status, second_status, count,
):
    path = write_comparison_drive(tmp_path, route_context_drive(updates))
    row = compare_recordings([path]).iloc[0]
    assert row.collision1_rt_status == first_status
    assert row.collision2_candidate_status == second_status
    assert row.collision2_candidate_count == count
    if first_status == "associated":
        assert row.collision1_rt_time == 310
    else:
        assert pd.isna(row.collision1_rt_time)
    if second_status == "candidate":
        assert row.collision2_candidate_rt_time == 482.6
    else:
        assert pd.isna(row.collision2_candidate_rt_time)


@pytest.mark.parametrize("drive", [2, 3])
def test_collision_route_context_is_specific_to_scenario1(tmp_path, drive):
    path = write_comparison_drive(tmp_path, route_context_drive([(310, 0.2)]), drive=drive)
    row = compare_recordings([path]).iloc[0]
    assert row.collision1_rt_status == "not_applicable"
    assert row.collision2_candidate_status == "not_applicable"


def test_event_context_reports_surrounding_markers_without_mutating_input():
    frame = event_context_drive()
    before = frame.copy(deep=True)
    result = analyze_recording(frame)
    tables = {name: getattr(result, name).copy(deep=True) for name in
              ("transitions", "intervals", "summary", "rt_update_episodes")}
    context = build_event_context(frame, result)
    expected = {
        "previous_sign_value": [35, -3], "previous_sign_time": [1, 5],
        "next_sign_value": [-3, 9], "next_sign_time": [5, 8],
        "speed_limit_at_event_start": [10, 20], "previous_speed_limit": [10, 20],
        "previous_speed_change_time": [2, 6], "next_speed_limit": [20, 25],
        "next_speed_change_time": [6, 8], "previous_intersection_start_time": [2, 7],
        "nearest_rt_episode_time": [3, 8], "nearest_rt_episode_delta_s": [-1, 1],
    }
    for column, values in expected.items():
        assert context[column].tolist() == values
    assert context.intersection_active_at_event_start.tolist() == [True, True]
    assert context.iloc[0].next_intersection_start_time == 7
    assert pd.isna(context.iloc[1].next_intersection_start_time)
    pd.testing.assert_frame_equal(frame, before)
    for name, original in tables.items():
        pd.testing.assert_frame_equal(getattr(result, name), original)


def test_event_context_nearest_rt_tie_prefers_earlier_episode():
    frame = event_context_drive()
    frame["Reaction Time"] = [0, 0, 0, 0.5, 0.5, 1, 1, 1, 1, 1]
    row = build_event_context(frame, analyze_recording(frame)).iloc[0]
    assert row.nearest_rt_episode_time == 3
    assert row.nearest_rt_episode_delta_s == -1


@pytest.mark.parametrize("case", ["no-events", "no-time", "no-neighbors"])
def test_event_context_missing_evidence_keeps_schema_and_unknowns(case):
    frames = {
        "no-events": pd.DataFrame({"Sim Time": [0, 1]}),
        "no-time": pd.DataFrame({"Events": [0, 3, 0]}),
        "no-neighbors": pd.DataFrame({"Sim Time": [0, 1, 2], "Events": [1, 0, 5],
                                       "Speed Limit": [35] * 3, "Signs": [0] * 3, "Intersections": [0] * 3}),
    }
    frame = frames[case]
    context = build_event_context(frame, analyze_recording(frame))
    assert context.columns.tolist() == EVENT_CONTEXT_COLUMNS
    if case == "no-events":
        assert context.empty
        return
    for column in ("previous_sign_value", "next_sign_value", "previous_speed_change_time",
                   "next_speed_change_time", "nearest_rt_episode_time"):
        assert context[column].isna().all()
    if case == "no-time":
        assert context.event_start_time.isna().all()
        assert context.intersection_active_at_event_start.isna().all()
    else:
        assert context.speed_limit_at_event_start.tolist() == [35, 35]
        assert context.intersection_active_at_event_start.tolist() == [False, False]
        assert pd.isna(context.iloc[-1].event_end_time)


@pytest.mark.parametrize("mode,selected", [("--file", 1), ("--files", 11)])
def test_event_context_cli_preserves_reports_and_uses_selected_recording(tmp_path, capsys, mode, selected):
    base = write_comparison_drive(tmp_path, event_context_drive(), drive=1)
    write_comparison_drive(tmp_path, event_context_drive(), drive=11)
    inputs = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert main([mode, str(base), "--event-context"]) == 0
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == inputs
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    output = tmp_path / "reports"
    assert main([mode, str(base), "--output-dir", str(output)]) == 0
    capsys.readouterr()
    existing = {path: path.read_bytes() for path in output.iterdir()}
    assert main([mode, str(base), "--event-context", "--output-dir", str(output)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == ["Created:", str(output / "event_context.csv")]
    assert {path: path.read_bytes() for path in existing} == existing
    assert {path.name for path in output.iterdir()} == {path.name for path in existing} | {"event_context.csv"}
    context = pd.read_csv(output / "event_context.csv")
    assert context.columns.tolist() == EVENT_CONTEXT_COLUMNS
    assert context.drive.unique().tolist() == [selected]

def test_scenario1_evidence_counts_marker_support(tmp_path, capsys):
    frame = scenario1_evidence_drive()
    first = write_comparison_drive(tmp_path, frame, "002")
    second_frame = frame.copy()
    second_frame.loc[16, "Signs"] = 0
    second = write_comparison_drive(tmp_path, second_frame, "003")
    assert main(["--files", str(first), str(second), "--scenario1-evidence"]) == 0
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    output = tmp_path / "evidence"
    assert main(["--files", str(first), str(second), "--scenario1-evidence", "--output-dir", str(output)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == ["Created:", str(output / "scenario1_marker_evidence.csv")]
    assert [path.name for path in output.iterdir()] == ["scenario1_marker_evidence.csv"]
    evidence = pd.read_csv(output / "scenario1_marker_evidence.csv").set_index("maneuver")
    assert evidence.reset_index().columns.tolist() == EVIDENCE_COLUMNS
    assert evidence.recordings_evaluated.eq(2).all()
    assert evidence.loc["Collision Avoidance 1", "recordings_matching"] == 2
    assert evidence.loc["Lane Closure", "recordings_matching"] == 2
    assert evidence.loc["Drive-through/KFC region candidate", "recordings_matching"] == 2
    assert evidence.loc["Roundabout", "recordings_matching"] == 2
    assert evidence.loc["School Zone", "recordings_matching"] == 1
    assert evidence.loc["Collision Avoidance 2", "recordings_matching"] == 2
    assert evidence.loc["Collision Avoidance 2", "status"] == "candidate"
    assert evidence.loc["Lane Closure", "status"] == "strong_candidate"
    for maneuver in ("Enter KFC candidate", "Exit KFC candidate"):
        assert pd.isna(evidence.loc[maneuver, "recordings_matching"])
        assert evidence.loc[maneuver, "status"] == "unresolved_candidate"
