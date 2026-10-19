"""Exploratory marker extraction, scenario comparison, and evidence reports.

Intervals use zero-based [start_row, end_row) bounds. End time is the timestamp
at the exclusive boundary (NA at EOF); duration uses first/last active samples,
not a guessed sampling period. Missing samples break continuity. Pulses are
single non-default samples bounded by observed defaults; negative codes survive.

RT update episodes cluster actual value changes, including zero resets, with a
configurable stable gap. Initial positive states alone are not updates. Episode
end rows are exclusive; last_transition_time anchors post-RT windows. Missing
RT/time or reversed clocks prevent clustering across those boundaries.

Braking summaries preserve raw force, including values above 170 N. Strong-
braking intervals require an explicit threshold. Post-marker windows exclude
its timestamp and include the configured end; braking flags mean force > 0 OR
acceleration < 0. Missing evidence is NA, not false absence.

Per participant/visit, Drive11 supersedes Drive1 and Drive21 supersedes Drive2;
Drive3 remains Scenario 3. Original drive numbers/files survive. Same-directory
restart siblings are checked for base-only batch inputs. Duplicate exports stay
traceable; no files are stitched and no scenario completeness is inferred.

Scenario 1 route context anchors Collision Avoidance 1 on Events 1; nearby RT is
optional. Collision Avoidance 2 is only a unique unassociated RT candidate wholly
between Events 1 and 3. RT ordinal never determines collision identity. Event
context uses chronological (time, row) neighbors: nonzero sign arrivals, actual
speed changes, active intersections, and nearest RT starts (signed RT-event delta).

--event-context and --scenario1-evidence write only their respective reports.
The main comparison is compact; scenario_comparison_details.csv preserves the
full comparison. Successful exports print only created file paths. Evidence
conclusions describe the reviewed candidate patterns, not production segmentation
or scoring-item mappings.
All time windows are exploratory/configurable. See README.md and --help.
"""

import argparse
from dataclasses import dataclass
import glob as glob_module
import json
import math
from pathlib import Path
import re
from typing import Sequence

import pandas as pd


DEFAULT_RT_GAP_S = 1.0
DEFAULT_POST_RT_WINDOW_S = 3.0
DEFAULT_EVENTS_NEAR_WINDOW_S = 3.0
DEFAULT_POST_EVENT_WINDOW_S = 10.0
EVENT_CODES = (1, 3, 4, 5)
RESTART_DRIVES = {1: 11, 2: 21}
SINGLE_OUTPUTS = ("transitions", "intervals", "summary", "rt_update_episodes")


MARKER_COLUMNS = (
    "Signs", "Speed Limit", "Events", "Task Marker", "Intersections",
    "Collision", "Reaction Time",
    *(f"Driver Response {number}" for number in range(1, 7)),
    "User5", "User17",
)
BRAKE_COLUMN = "Brake Pedal Force"
METADATA_COLUMNS = ["recording_id", "participant_id", "visit", "drive"]
BOUNDS_COLUMNS = [
    "interval_start_row", "interval_end_row", "interval_start_time",
    "interval_end_time", "last_active_row", "last_active_time", "duration_s",
]
TRANSITION_COLUMNS = [
    *METADATA_COLUMNS, "record_type", "row", "sim_time", "column",
    "previous_value", "new_value", "occurrence", *BOUNDS_COLUMNS,
]
INTERVAL_COLUMNS = [
    *METADATA_COLUMNS, "column", "interval_type", "occurrence", "value",
    *BOUNDS_COLUMNS, "sample_count", "is_pulse", "left_censored",
    "right_censored", "peak_value", "peak_row", "peak_time",
    "brake_threshold",
]
SUMMARY_COLUMNS = [*METADATA_COLUMNS, "column", "metric", "value", "notes"]
RT_EPISODE_COLUMNS = [
    *METADATA_COLUMNS, "occurrence", "start_row", "end_row", "start_time", "end_time",
    "starting_rt_value", "ending_rt_value", "rt_value_change", "transition_count",
    "last_transition_row", "last_transition_time", "duration_s",
    "rt_transition_gap_s", "right_censored",
]
ROUTE_CONTEXT_COLUMNS = [
    "collision1_event_time", "collision1_rt_time", "collision1_rt_value_change", "collision1_rt_status",
    "collision1_peak_brake", "collision1_peak_deceleration",
    "collision2_candidate_rt_time", "collision2_candidate_rt_value_change",
    "collision2_candidate_peak_brake", "collision2_candidate_peak_deceleration", "collision2_candidate_status",
    "collision1_rt_candidate_count", "collision2_candidate_count", "post_event_window_s", "route_context_notes",
]
COMPARISON_COLUMNS = [
    *METADATA_COLUMNS, "scenario", "selection_reason", "row_count", "rt_episode_count",
    "collision1_event_time", "collision1_rt_time", "collision1_rt_value_change", "collision1_rt_status",
    "collision1_peak_brake", "collision1_peak_deceleration",
    "collision2_candidate_rt_time", "collision2_candidate_rt_value_change",
    "collision2_candidate_peak_brake", "collision2_candidate_peak_deceleration", "collision2_candidate_status",
    "events_values_in_order", *(f"events_{code}_start_time" for code in EVENT_CODES),
    "intersection_count", "sign_values_in_order", "speed_limit_values_in_order",
    "missing_columns", "notes",
]
COMPARISON_DETAIL_COLUMNS = [
    *METADATA_COLUMNS, "rt_episode_count",
    *(f"rt{number}_{field}" for number in (1, 2) for field in (
        "start_time", "end_time", "value_change", "peak_brake_after", "peak_deceleration_after",
    )),
    "events_values_in_order",
    *(f"events_{code}_start_time" for code in EVENT_CODES),
    "intersection_count", "sign_values_in_order", "speed_limit_values_in_order",
    "events_1_near_rt1", "braking_after_rt1", "braking_after_rt2",
    *(f"events_{code}_present" for code in EVENT_CODES),
    "intersection_intervals_in_order",
    *(f"rt{number}_{field}" for number in (1, 2) for field in (
        "last_transition_time", "after_sample_count", "after_window_complete",
    )),
    "rt_transition_gap_s", "post_rt_window_s", "events_near_window_s",
    "row_count", "recording_start_time", "recording_end_time", "scenario", "selection_reason",
    "missing_columns", "notes", "source_file",
    *ROUTE_CONTEXT_COLUMNS,
]
SELECTION_COLUMNS = [
    "participant_id", "visit", "scenario", "selected_drive", "selected_recording_id",
    "superseded_drive", "superseded_recording_id", "selection_reason",
    "selected_source_file", "superseded_source_file", "notes",
]
SCENARIO_DRIVES = {3: 3, **{drive: base for base, restart in RESTART_DRIVES.items() for drive in (base, restart)}}
EVENT_CONTEXT_COLUMNS = [
    *METADATA_COLUMNS, "event_value", "event_occurrence", "event_start_time", "event_end_time", "event_duration_s",
    "speed_limit_at_event_start", "previous_speed_limit", "previous_speed_change_time",
    "next_speed_limit", "next_speed_change_time", "previous_sign_value", "previous_sign_time",
    "next_sign_value", "next_sign_time", "intersection_active_at_event_start",
    "previous_intersection_start_time", "next_intersection_start_time",
    "nearest_rt_episode_time", "nearest_rt_episode_delta_s",
]
EVIDENCE_COLUMNS = [
    "maneuver", "candidate_marker", "supporting_pattern", "recordings_evaluated",
    "recordings_matching", "confidence", "status", "notes",
]
RECORDING_CHAIN_COLUMNS = [
    "participant_id", "visit", "base_drive", "related_drive",
    "base_row_count", "related_row_count", "base_start_time", "base_end_time",
    "related_start_time", "related_end_time", "base_rt_episode_count", "related_rt_episode_count",
    "base_event_values", "related_event_values", "base_intersection_count", "related_intersection_count",
    "base_sign_values", "related_sign_values", "relationship_status", "notes",
    "base_recording_id", "related_recording_id", "base_source_file", "related_source_file",
]
_CLEANED_NAME = re.compile(
    r"^participant_(?P<participant_id>[^_]+)_(?P<visit>[^_]+)_"
    r".+_drive_(?P<drive>\d+|unknown)_cleaned$"
)


@dataclass
class MarkerDiscoveryResult:
    """Fixed-schema tables for one recording, including empty inputs."""

    transitions: pd.DataFrame
    intervals: pd.DataFrame
    summary: pd.DataFrame
    rt_update_episodes: pd.DataFrame


def _metadata(frame: pd.DataFrame, recording_id: str) -> dict:
    match = _CLEANED_NAME.fullmatch(recording_id)
    fallback = match.groupdict() if match else {}
    metadata = {"recording_id": recording_id}
    for column in METADATA_COLUMNS[1:]:
        values = frame[column].dropna().unique() if column in frame else []
        if len(values) > 1:
            raise ValueError(f"Conflicting {column} values: expected one recording")
        value = values[0] if len(values) else fallback.get(column, pd.NA)
        if column == "drive" and not pd.isna(value) and str(value) != "unknown":
            number = float(value)
            if not math.isfinite(number) or number < 0 or not number.is_integer():
                raise ValueError("drive must be a nonnegative integer")
            value = int(number)
        expected = fallback.get(column)
        if expected and expected != "unknown" and not pd.isna(value):
            agrees = value == int(expected) if column == "drive" else str(value) == expected
            if not agrees:
                raise ValueError(f"Conflicting {column} in metadata and filename")
        metadata[column] = pd.NA if str(value) == "unknown" else value
    return metadata


def _numeric(series: pd.Series) -> pd.Series:
    """A temporary finite numeric view for comparisons, never an input edit."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.notna() & ~numeric.isin([float("inf"), -float("inf")]))


def _runs(mask: pd.Series, values: pd.Series | None = None) -> list[tuple[int, int]]:
    """Find positional half-open activity runs, optionally splitting by value."""
    starts = mask & ~mask.shift(1, fill_value=False)
    ends = mask & ~mask.shift(-1, fill_value=False)
    if values is not None:
        starts |= mask & ~values.eq(values.shift()).fillna(False)
        ends |= mask & ~values.eq(values.shift(-1)).fillna(False)
    return list(zip(mask.index[starts].tolist(), (mask.index[ends] + 1).tolist()))


def _bounds(times: pd.Series, start: int, end: int) -> dict:
    active_times = times.iloc[start:end]
    valid = active_times.notna().all() and active_times.is_monotonic_increasing
    return {
        "interval_start_row": start,
        "interval_end_row": end,
        "interval_start_time": times.iloc[start],
        "interval_end_time": times.iloc[end] if end < len(times) else pd.NA,
        "last_active_row": end - 1,
        "last_active_time": times.iloc[end - 1],
        "duration_s": active_times.iloc[-1] - active_times.iloc[0] if valid else pd.NA,
    }


def _json_values(values) -> str:
    """Encode NumPy scalars and text marker codes as a JSON array."""
    return json.dumps([value.item() if hasattr(value, "item") else value for value in values])


def _rt_episodes(
    values: pd.Series, times: pd.Series, changed: pd.Series, metadata: dict, gap_s: float,
) -> list[dict]:
    """Cluster RT changes; carried-forward values do not add updates.

    Split when the transition gap exceeds gap_s or continuity breaks. A single
    transition counts, regardless of its amplitude or direction.
    """
    continuous = values.notna() & times.notna()
    blocks = (
        ~continuous | ~continuous.shift(1, fill_value=False)
        | times.lt(times.shift()).fillna(False)
    ).cumsum()
    updates = pd.DataFrame({"time": times, "block": blocks}).loc[changed & continuous]
    if updates.empty:
        return []
    groups = (
        updates.block.ne(updates.block.shift()) | updates.time.diff().gt(gap_s)
    ).cumsum()
    block_end_times = times[continuous].groupby(blocks[continuous]).last()
    episodes = []
    for occurrence, (_, group) in enumerate(updates.groupby(groups, sort=False), 1):
        start, last = int(group.index[0]), int(group.index[-1])
        end = last + 1
        episodes.append({
            **metadata, "occurrence": occurrence, "start_row": start, "end_row": end,
            "start_time": times.iloc[start],
            "end_time": times.iloc[end] if end < len(times) else pd.NA,
            "starting_rt_value": values.iloc[start - 1], "ending_rt_value": values.iloc[last],
            "rt_value_change": values.iloc[last] - values.iloc[start - 1],
            "transition_count": len(group), "last_transition_row": last,
            "last_transition_time": times.iloc[last],
            "duration_s": times.iloc[last] - times.iloc[start],
            "rt_transition_gap_s": gap_s,
            "right_censored": block_end_times.loc[group.block.iloc[-1]] - times.iloc[last] <= gap_s,
        })
    return episodes


def _table(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    integer_columns = [
        "row", "occurrence", "interval_start_row", "interval_end_row",
        "last_active_row", "sample_count", "peak_row", "start_row", "end_row",
        "last_transition_row", "transition_count",
    ]
    for column in integer_columns:
        if column in frame:
            frame[column] = frame[column].astype("Int64")
    return frame


def analyze_recording(
    frame: pd.DataFrame,
    recording_id: str = "recording",
    *,
    brake_threshold: float | None = None,
    rt_transition_gap_s: float = DEFAULT_RT_GAP_S,
    post_rt_window_s: float = DEFAULT_POST_RT_WINDOW_S,
    events_near_window_s: float = DEFAULT_EVENTS_NEAR_WINDOW_S,
    post_event_window_s: float = DEFAULT_POST_EVENT_WINDOW_S,
) -> MarkerDiscoveryResult:
    """Return marker tables from one cleaned DataFrame, without mutating it.

    Occurrences follow row order within each column/interval type. Transition
    bounds describe the destination value interval; defaults have NA bounds.
    Braking peaks describe threshold intervals or the global maximum (earliest tie).
    """
    settings = dict(rt_transition_gap_s=rt_transition_gap_s, post_rt_window_s=post_rt_window_s,
                    events_near_window_s=events_near_window_s, post_event_window_s=post_event_window_s)
    if brake_threshold is not None:
        settings["brake_threshold"] = brake_threshold
    _validate_positive_values(**settings)
    frame = frame.reset_index(drop=True)
    metadata = _metadata(frame, recording_id)
    transitions: list[dict] = []
    intervals: list[dict] = []
    summaries: list[dict] = []
    rt_episodes: list[dict] = []

    def summarize(column: str, metric: str, value, notes: str = "") -> None:
        summaries.append({**metadata, "column": column, "metric": metric,
                          "value": value, "notes": notes})

    summarize("recording", "row_count", len(frame))
    summarize("recording", "boundary_convention", "[start, end)",
              "Zero-based cleaned rows; duration is last active time minus start time; "
              "NA if active timestamps are invalid or decrease.")
    for column in METADATA_COLUMNS[1:]:
        if pd.isna(metadata[column]):
            summarize(column, "advisory", "Metadata unavailable")
    times = _numeric(frame["Sim Time"]) if "Sim Time" in frame else pd.Series(
        float("nan"), index=frame.index,
    )
    summarize("Sim Time", "available", "Sim Time" in frame)
    summarize("Sim Time", "missing_or_invalid_samples", int(times.isna().sum()))
    if not times.dropna().is_monotonic_increasing:
        summarize("Sim Time", "advisory", "Nonmonotonic simulator time",
                  "Log sorted by time; intervals detected in recording row order.")
    summarize("Reaction Time", "rt_transition_gap_s", rt_transition_gap_s,
              "Exploratory setting, not a scientific cutoff. A gap strictly greater than "
              "this many seconds between RT changes starts a new update episode.")

    for column in MARKER_COLUMNS:
        summarize(column, "available", column in frame)
        if column not in frame:
            continue
        values = frame[column]
        numeric = _numeric(values)
        observed = values.notna() & ~values.astype("string").str.strip().eq("").fillna(False)
        if column == "Reaction Time":
            observed &= numeric.notna()
            active = observed & numeric.gt(0).fillna(False)
        else:
            active = observed & ~numeric.eq(0).fillna(False)
        defaults = observed & ~active
        summarize(column, "missing_or_invalid_samples", int((~observed).sum()))
        summarize(column, "unique_non_default_values", _json_values(pd.unique(values[active])))

        value_runs = _runs(active, values)
        active_runs = _runs(active)
        value_starts: dict[int, dict] = {}
        pulse_count = 0
        for kind, runs in (("value", value_runs), ("active", active_runs)):
            for occurrence, (start, end) in enumerate(runs, 1):
                left_censored = start == 0 or not observed.iloc[start - 1]
                right_censored = end == len(frame) or not observed.iloc[end]
                pulse = (
                    end - start == 1 and not left_censored and not right_censored
                    and bool(defaults.iloc[start - 1]) and bool(defaults.iloc[end])
                )
                interval_values = values.iloc[start:end]
                value = interval_values.iloc[0] if interval_values.nunique() == 1 else pd.NA
                record = {
                    **metadata, "column": column, "interval_type": kind,
                    "occurrence": occurrence, "value": value,
                    **_bounds(times, start, end), "sample_count": end - start,
                    "is_pulse": pulse, "left_censored": left_censored,
                    "right_censored": right_censored,
                }
                intervals.append(record)
                if kind == "value":
                    value_starts[start] = record
                    pulse_count += int(pulse)

        # Missing samples reset observation; they do not prove a transition.
        initial = observed & ~observed.shift(1, fill_value=False)
        changed = observed & observed.shift(1, fill_value=False) & ~values.eq(values.shift()).fillna(False)
        for row in frame.index[initial | changed]:
            interval = value_starts.get(row, {})
            transitions.append({
                **metadata, "record_type": "initial_state" if initial.iloc[row] else "transition",
                "row": row, "sim_time": times.iloc[row], "column": column,
                "previous_value": pd.NA if initial.iloc[row] else values.iloc[row - 1],
                "new_value": values.iloc[row], "occurrence": interval.get("occurrence", pd.NA),
                **{key: interval.get(key, pd.NA) for key in BOUNDS_COLUMNS},
            })
        summarize(column, "transition_count", int(changed.sum()))
        summarize(column, "value_interval_count", len(value_runs))
        summarize(column, "active_interval_count", len(active_runs))
        summarize(column, "pulse_count", pulse_count)
        summarize(column, "ordered_non_default_values",
                  _json_values([values.iloc[start] for start, _ in value_runs]))
        if column == "Reaction Time":
            rt_episodes = _rt_episodes(numeric, times, changed, metadata, rt_transition_gap_s)
            summarize(column, "update_episode_count", len(rt_episodes),
                      "Time-clustered RT transitions; positive intervals are descriptive only.")
            summarize(column, "candidate_event_count", len(rt_episodes),
                      "Alias of update_episode_count; no longer counts contiguous positive activity.")
            untimed_count = int((changed & times.isna()).sum())
            summarize(column, "untimed_transition_count", untimed_count)
            if untimed_count:
                summarize(column, "advisory", f"{untimed_count} RT transitions lack valid timestamps",
                          "Preserved in raw transitions; excluded from time-based episodes.")
            if len(rt_episodes) != 2:
                summarize(column, "advisory", f"Expected 2 RT update episodes; observed {len(rt_episodes)}",
                          "Episodes are exploratory; no event count is forced or maneuver assigned.")

    summarize(BRAKE_COLUMN, "available", BRAKE_COLUMN in frame)
    summarize(BRAKE_COLUMN, "brake_threshold", brake_threshold if brake_threshold is not None else pd.NA,
              "Strong braking uses force >= supplied threshold (N)." if brake_threshold is not None
              else "No threshold supplied; no strong-braking classification.")
    if BRAKE_COLUMN in frame:
        raw_brake = frame[BRAKE_COLUMN]
        brake = _numeric(raw_brake)
        summarize(BRAKE_COLUMN, "missing_or_invalid_samples", int(brake.isna().sum()))
        peak_row = int(brake.idxmax()) if brake.notna().any() else None
        summarize(BRAKE_COLUMN, "maximum", raw_brake.iloc[peak_row] if peak_row is not None else pd.NA,
                  "Raw force (N), preserved without clipping; earliest sample wins ties.")
        summarize(BRAKE_COLUMN, "peak_row", peak_row if peak_row is not None else pd.NA)
        summarize(BRAKE_COLUMN, "peak_time", times.iloc[peak_row] if peak_row is not None else pd.NA)
        braking_runs = _runs(brake.ge(brake_threshold).fillna(False)) if brake_threshold is not None else []
        summarize(BRAKE_COLUMN, "strong_braking_interval_count", len(braking_runs))
        peaks = []
        for occurrence, (start, end) in enumerate(braking_runs, 1):
            row = int(brake.iloc[start:end].idxmax())
            record = {
                **metadata, "column": BRAKE_COLUMN, "interval_type": "strong_braking",
                "occurrence": occurrence, "value": pd.NA, **_bounds(times, start, end),
                "sample_count": end - start, "is_pulse": pd.NA,
                "left_censored": start == 0 or pd.isna(brake.iloc[start - 1]),
                "right_censored": end == len(frame) or pd.isna(brake.iloc[end]),
                "peak_value": raw_brake.iloc[row], "peak_row": row,
                "peak_time": times.iloc[row], "brake_threshold": brake_threshold,
            }
            intervals.append(record)
            peaks.append((row, record))
        if brake_threshold is None and peak_row is not None:
            peaks.append((peak_row, {}))
        for row, interval in peaks:
            transitions.append({
                **metadata, "record_type": "braking_peak", "row": row,
                "sim_time": times.iloc[row], "column": BRAKE_COLUMN,
                "previous_value": pd.NA, "new_value": raw_brake.iloc[row],
                "occurrence": interval.get("occurrence", pd.NA),
                **{key: interval.get(key, pd.NA) for key in BOUNDS_COLUMNS},
            })

    transition_frame = _table(transitions, TRANSITION_COLUMNS)
    interval_frame = _table(intervals, INTERVAL_COLUMNS)
    order = {column: number for number, column in enumerate((*MARKER_COLUMNS, BRAKE_COLUMN))}
    for table, time_key, row_key in (
        (transition_frame, "sim_time", "row"),
        (interval_frame, "interval_start_time", "interval_start_row"),
    ):
        table["_column_order"] = table["column"].map(order)
        table.sort_values([time_key, row_key, "_column_order"], kind="stable", na_position="last", inplace=True)
        table.drop(columns="_column_order", inplace=True)
        table.reset_index(drop=True, inplace=True)
    result = MarkerDiscoveryResult(
        transition_frame, interval_frame, _table(summaries, SUMMARY_COLUMNS),
        _table(rt_episodes, RT_EPISODE_COLUMNS),
    )
    associations = associate_route_context(frame, result, post_rt_window_s=post_rt_window_s,
                                           events_near_window_s=events_near_window_s,
                                           post_event_window_s=post_event_window_s)
    for name, value in associations.items():
        summarize("route_context", name, value, "Exploratory Scenario 1 association; not a segmentation rule.")
    summarize("route_context", "events_near_window_s", events_near_window_s)
    summarize("route_context", "post_rt_window_s", post_rt_window_s)
    result.summary = _table(summaries, SUMMARY_COLUMNS)
    return result


def analyze_file(
    file: str | Path, *, brake_threshold: float | None = None, rt_transition_gap_s: float = DEFAULT_RT_GAP_S,
    post_rt_window_s: float = DEFAULT_POST_RT_WINDOW_S, events_near_window_s: float = DEFAULT_EVENTS_NEAR_WINDOW_S, post_event_window_s: float = DEFAULT_POST_EVENT_WINDOW_S,
) -> MarkerDiscoveryResult:
    """Read an existing cleaned CSV, preserving participant IDs as strings."""
    _, result = _read_recording(file, brake_threshold=brake_threshold,
                                rt_transition_gap_s=rt_transition_gap_s, post_rt_window_s=post_rt_window_s,
                                events_near_window_s=events_near_window_s, post_event_window_s=post_event_window_s)
    return result


def _read_recording(file: str | Path, **options) -> tuple[pd.DataFrame, MarkerDiscoveryResult]:
    path = Path(file)
    frame = pd.read_csv(path, dtype={"participant_id": str})
    return frame, analyze_recording(frame, path.stem, **options)


def _validate_positive_values(**settings) -> None:
    for name, value in settings.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")


def discover_cleaned_files(
    *, directory: str | Path | None = None, pattern: str | None = None,
) -> list[Path]:
    """Resolve directory/glob input deterministically; retain distinct file paths."""
    if (directory is None) == (pattern is None):
        raise ValueError("Supply exactly one directory or glob pattern")
    if directory is not None:
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise ValueError(f"Cleaned recording directory does not exist: {root}")
        candidates = root.rglob("*_cleaned.csv")
    else:
        # Accept the documented Windows separator even on other platforms.
        normalized = str(Path(pattern).expanduser()).replace("\\", "/")
        candidates = (Path(path) for path in glob_module.iglob(normalized, recursive=True))
    paths = sorted(
        {path.resolve() for path in candidates if path.is_file() and path.name.endswith("_cleaned.csv")},
        key=lambda path: path.as_posix().casefold(),
    )
    if not paths:
        raise ValueError("No cleaned recordings matched (expected *_cleaned.csv)")
    return paths


def _post_rt_measurements(
    frame: pd.DataFrame, times: pd.Series, episode: pd.Series, window_s: float,
) -> dict:
    return _post_marker_measurements(frame, times, int(episode.end_row), episode.last_transition_time, window_s)


def _post_marker_measurements(
    frame: pd.DataFrame, times: pd.Series, start_row: int, anchor: float, window_s: float,
) -> dict:
    """Preserve raw force/acceleration extrema strictly after a marker timestamp."""
    tail = times.iloc[start_row:]
    breaks = tail.isna() | tail.lt(times.shift().loc[tail.index]).fillna(False)
    if breaks.any():
        tail = tail.loc[tail.index < breaks.index[breaks][0]]
    selected = tail.index[tail.gt(anchor) & tail.le(anchor + window_s)]
    complete = bool(not tail.empty and tail.iloc[-1] >= anchor + window_s)
    peaks = {}
    for column, name, operation in (
        (BRAKE_COLUMN, "peak_brake_after", "max"),
        ("Acceleration", "peak_deceleration_after", "min"),
    ):
        values = _numeric(frame.loc[selected, column]) if column in frame else pd.Series(dtype=float)
        peaks[name] = getattr(values, operation)() if values.notna().any() else pd.NA
    brake, acceleration = peaks["peak_brake_after"], peaks["peak_deceleration_after"]
    braking = pd.NA
    if (not pd.isna(brake) and brake > 0) or (not pd.isna(acceleration) and acceleration < 0):
        braking = True
    elif not pd.isna(brake) and not pd.isna(acceleration):
        braking = False
    return {**peaks, "braking": braking, "after_sample_count": len(selected),
            "after_window_complete": complete}


def associate_route_context(
    frame: pd.DataFrame, result: MarkerDiscoveryResult, *, events_near_window_s: float = DEFAULT_EVENTS_NEAR_WINDOW_S,
    post_rt_window_s: float = DEFAULT_POST_RT_WINDOW_S, post_event_window_s: float = DEFAULT_POST_EVENT_WINDOW_S,
) -> dict:
    """Anchor Collision Avoidance 1 to Events 1, with optional nearby RT.

    Collision Avoidance 2 remains a candidate before the first following Events 3;
    RT ordinal never identifies collision number. Proximity uses first/last update
    times, not the exclusive episode end_time.
    """
    values = dict.fromkeys(ROUTE_CONTEXT_COLUMNS, pd.NA)
    values.update(post_event_window_s=post_event_window_s, route_context_notes="Exploratory route context; collision 2 is never confirmed.")

    def unavailable(status: str) -> dict:
        values.update(collision1_rt_status=status, collision2_candidate_status=status)
        return values

    drive = result.summary.iloc[0].drive
    if pd.isna(drive):
        return unavailable("scenario_unknown")
    if drive not in (1, 11):
        return unavailable("not_applicable")
    events = result.intervals.loc[
        result.intervals.column.eq("Events") & result.intervals.interval_type.eq("value")
    ].sort_values("interval_start_row")
    event_codes = _numeric(events.value)
    first_markers = events.loc[event_codes.eq(1).fillna(False)]
    if first_markers.empty:
        return unavailable("event_missing" if "Events" in frame else "events_unavailable")
    if len(first_markers) > 1:
        return unavailable("ambiguous_event_marker")
    marker = first_markers.iloc[0]
    values["collision1_event_time"] = marker.interval_start_time
    times = _numeric(frame["Sim Time"]) if "Sim Time" in frame else pd.Series(float("nan"), index=frame.index)
    if times.isna().any() or not times.is_monotonic_increasing:
        return unavailable("time_unavailable")
    peaks = _post_marker_measurements(frame, times, int(marker.interval_start_row) + 1,
                                      marker.interval_start_time, post_event_window_s)
    values.update(collision1_peak_brake=peaks["peak_brake_after"],
                  collision1_peak_deceleration=peaks["peak_deceleration_after"])
    episodes = result.rt_update_episodes
    near = episodes.start_time.le(marker.interval_start_time + events_near_window_s) & (
        episodes.last_transition_time.ge(marker.interval_start_time - events_near_window_s)
    )
    matches = episodes.loc[near]
    values["collision1_rt_candidate_count"] = len(matches)
    if matches.empty:
        values["collision1_rt_status"] = "rt_missing"
    elif len(matches) > 1:
        values["collision1_rt_status"] = "ambiguous"
    else:
        episode = matches.iloc[0]
        values.update(collision1_rt_status="associated", collision1_rt_time=episode.start_time,
                      collision1_rt_value_change=episode.rt_value_change)
    boundaries = events.loc[
        event_codes.eq(3).fillna(False) & events.interval_start_row.gt(marker.interval_start_row)
        & events.interval_start_time.gt(marker.interval_start_time)
    ]
    if boundaries.empty:
        values["collision2_candidate_status"] = "boundary_missing"
        return values
    boundary = boundaries.iloc[0]
    eligible = episodes.loc[
        ~near & episodes.start_time.gt(marker.interval_start_time)
        & episodes.last_transition_time.lt(boundary.interval_start_time)
        & episodes.start_row.gt(marker.interval_start_row)
        & episodes.last_transition_row.lt(boundary.interval_start_row)
    ]
    values["collision2_candidate_count"] = len(eligible)
    if eligible.empty:
        values["collision2_candidate_status"] = "no_candidate"
    elif len(eligible) > 1:
        values["collision2_candidate_status"] = "ambiguous"
    else:
        episode = eligible.iloc[0]
        peaks = _post_rt_measurements(frame, times, episode, post_rt_window_s)
        values.update(collision2_candidate_status="candidate", collision2_candidate_rt_time=episode.start_time,
                      collision2_candidate_rt_value_change=episode.rt_value_change,
                      collision2_candidate_peak_brake=peaks["peak_brake_after"],
                      collision2_candidate_peak_deceleration=peaks["peak_deceleration_after"])
    return values


def _comparison_row(
    frame: pd.DataFrame, result: MarkerDiscoveryResult, path: Path,
    rt_transition_gap_s: float, post_rt_window_s: float, events_near_window_s: float,
) -> dict:
    row = {column: pd.NA for column in COMPARISON_DETAIL_COLUMNS}
    row.update({key: result.summary.iloc[0][key] for key in METADATA_COLUMNS})
    row.update(source_file=str(path.resolve()), rt_transition_gap_s=rt_transition_gap_s,
               post_rt_window_s=post_rt_window_s, events_near_window_s=events_near_window_s)
    needed = ("Sim Time", "Reaction Time", "Events", "Intersections", "Signs",
              "Speed Limit", BRAKE_COLUMN, "Acceleration")
    row["missing_columns"] = _json_values([column for column in needed if column not in frame])
    times = _numeric(frame["Sim Time"]) if "Sim Time" in frame else pd.Series(float("nan"), index=frame.index)
    row.update(row_count=len(frame), recording_start_time=times.iloc[0] if len(times) else pd.NA,
               recording_end_time=times.iloc[-1] if len(times) else pd.NA)
    rt = _numeric(frame["Reaction Time"]) if "Reaction Time" in frame else pd.Series(dtype=float)
    episodes = result.rt_update_episodes.sort_values("occurrence")
    untimed = result.summary.loc[
        result.summary.column.eq("Reaction Time") & result.summary.metric.eq("untimed_transition_count"), "value"
    ]
    if rt.notna().any() and times.notna().any() and not (len(untimed) and untimed.iloc[0]):
        row["rt_episode_count"] = len(episodes)
    notes = [f"{item.column}: {item.value}" for item in result.summary.loc[
        result.summary.metric.eq("advisory")
    ].itertuples()]
    for number, (_, episode) in enumerate(episodes.iloc[:2].iterrows(), 1):
        prefix = f"rt{number}_"
        row.update({prefix + "start_time": episode.start_time, prefix + "end_time": episode.end_time,
                    prefix + "last_transition_time": episode.last_transition_time,
                    prefix + "value_change": episode.rt_value_change})
        measurements = _post_rt_measurements(frame, times, episode, post_rt_window_s)
        row[f"braking_after_rt{number}"] = measurements.pop("braking")
        row.update({prefix + key: value for key, value in measurements.items()})
    if len(episodes) > 2:
        notes.append("Only the first two RT episodes have dedicated columns; rt_episode_count includes all episodes.")

    for column, target in (("Events", "events_values_in_order"), ("Signs", "sign_values_in_order")):
        if column in frame and frame[column].notna().any():
            runs = result.intervals.loc[
                result.intervals.column.eq(column) & result.intervals.interval_type.eq("value")
            ].sort_values("interval_start_row")
            row[target] = _json_values(runs.value.tolist())
    if "Speed Limit" in frame and frame["Speed Limit"].notna().any():
        states = result.transitions.loc[result.transitions.column.eq("Speed Limit")].sort_values("row")
        row["speed_limit_values_in_order"] = _json_values(states.new_value.tolist())
    events = result.intervals.loc[
        result.intervals.column.eq("Events") & result.intervals.interval_type.eq("value")
    ].sort_values("interval_start_row")
    event_codes = _numeric(events.value)
    events_known = "Events" in frame and _numeric(frame["Events"]).notna().any()
    for code in EVENT_CODES:
        matches = events.loc[event_codes.eq(code).fillna(False)]
        row[f"events_{code}_present"] = bool(len(matches)) if events_known else pd.NA
        if not matches.empty:
            row[f"events_{code}_start_time"] = matches.iloc[0].interval_start_time
    if len(episodes) and events_known:
        first = episodes.iloc[0]
        matches = events.loc[event_codes.eq(1).fillna(False)]
        near = matches.interval_start_time.le(first.last_transition_time + events_near_window_s) & (
            matches.last_active_time.ge(first.start_time - events_near_window_s)
        )
        if near.any():
            row["events_1_near_rt1"] = True
        elif matches.interval_start_time.notna().all() and matches.last_active_time.notna().all():
            row["events_1_near_rt1"] = False
    if "Intersections" in frame and _numeric(frame["Intersections"]).notna().any():
        intersections = result.intervals.loc[
            result.intervals.column.eq("Intersections") & result.intervals.interval_type.eq("active")
        ].sort_values("interval_start_row")
        row["intersection_count"] = len(intersections)
        # Use JSON null for unavailable times.
        row["intersection_intervals_in_order"] = intersections.loc[:, [
            "occurrence", "interval_start_row", "interval_end_row",
            "interval_start_time", "last_active_time",
        ]].to_json(orient="records")
    row["notes"] = " | ".join(notes)
    associations = result.summary.loc[result.summary.column.eq("route_context")]
    row.update(associations.set_index("metric").value.reindex(ROUTE_CONTEXT_COLUMNS).to_dict())
    return row


def compare_recordings(
    files: Sequence[str | Path], *, rt_transition_gap_s: float = DEFAULT_RT_GAP_S,
    post_rt_window_s: float = DEFAULT_POST_RT_WINDOW_S, events_near_window_s: float = DEFAULT_EVENTS_NEAR_WINDOW_S,
    post_event_window_s: float = DEFAULT_POST_EVENT_WINDOW_S,
    brake_threshold: float | None = None,
    preferred_only: bool = True,
    scenario: int | None = None,
) -> pd.DataFrame:
    """Compare preferred recordings; preserve original drives and source files.

    Incomplete RT counts are NA. All episodes count; only the first two have
    dedicated window columns. Duplicate exports remain traceable. Selection
    includes same-directory restarts; preferred_only=False returns the input
    inventory without sibling discovery. Scenario filtering follows selection.
    The returned table retains the full detail schema; compacting is export-only.
    """
    if scenario is not None and scenario not in (1, 2, 3):
        raise ValueError("scenario must be 1, 2, or 3")
    if scenario is not None and not preferred_only:
        raise ValueError("scenario filtering requires preferred recording selection")
    options = dict(brake_threshold=brake_threshold, rt_transition_gap_s=rt_transition_gap_s,
                   post_rt_window_s=post_rt_window_s, events_near_window_s=events_near_window_s,
                   post_event_window_s=post_event_window_s)
    paths = include_restart_recordings(files) if preferred_only else files
    table, _ = _collect_recordings(paths, options)
    if preferred_only:
        table, _ = select_preferred_recordings(table)
        if scenario is not None:
            table = table.loc[table.scenario.eq(scenario).fillna(False)].reset_index(drop=True)
    return table


def _collect_recordings(files: Sequence[str | Path], options: dict, report: str | None = None):
    """Analyze each distinct input once, retaining only requested report details."""
    paths = sorted({Path(file).resolve() for file in files}, key=lambda path: path.as_posix().casefold())
    if not paths:
        raise ValueError("No cleaned recordings supplied for comparison")
    rows, details = [], {}
    for path in paths:
        try:
            frame, result = _read_recording(path, **options)
            rows.append(_comparison_row(frame, result, path,
                options.get("rt_transition_gap_s", DEFAULT_RT_GAP_S),
                options.get("post_rt_window_s", DEFAULT_POST_RT_WINDOW_S),
                options.get("events_near_window_s", DEFAULT_EVENTS_NEAR_WINDOW_S)))
            drive = result.summary.iloc[0].drive
            if report == "context" or (report == "evidence" and not pd.isna(drive) and drive in (1, 11)):
                context = build_event_context(frame, result)
                details[str(path)] = context if report == "context" else _scenario1_evidence_support(
                    result, context, options.get("events_near_window_s", DEFAULT_EVENTS_NEAR_WINDOW_S))
        except (ValueError, OSError) as error:
            raise ValueError(f"Could not compare {path}: {error}") from error
    table = pd.DataFrame(rows, columns=COMPARISON_DETAIL_COLUMNS)
    duplicates = table.recording_id.duplicated(keep=False)
    table.loc[duplicates, "notes"] += " | Duplicate recording_id across source files; rows retained."
    for column in ("rt_episode_count", "intersection_count", "rt1_after_sample_count", "rt2_after_sample_count", "row_count"):
        table[column] = table[column].astype("Int64")
    for column in ("events_1_near_rt1", "braking_after_rt1", "braking_after_rt2",
                   *(f"events_{code}_present" for code in EVENT_CODES),
                   "rt1_after_window_complete", "rt2_after_window_complete"):
        table[column] = table[column].astype("boolean")
    return table, details


def include_restart_recordings(files: Sequence[str | Path]) -> list[Path]:
    """Find same-directory restarts omitted by a base-only batch glob.

    Match participant/visit from standard cleaned filenames.
    """
    paths = {Path(file).resolve() for file in files}
    checked = set()
    for path in list(paths):
        match = _CLEANED_NAME.fullmatch(path.stem)
        if match is None or match["drive"] not in ("1", "2"):
            continue
        restart = str(RESTART_DRIVES[int(match["drive"])])
        key = (path.parent, match["participant_id"], match["visit"], restart)
        if key in checked:
            continue
        checked.add(key)
        for sibling in path.parent.glob("*_cleaned.csv"):
            candidate = _CLEANED_NAME.fullmatch(sibling.stem)
            if (sibling.is_file() and candidate is not None
                    and candidate["participant_id"] == match["participant_id"]
                    and candidate["visit"] == match["visit"] and candidate["drive"] == restart):
                paths.add(sibling.resolve())
    return sorted(paths, key=lambda path: path.as_posix().casefold())


def select_preferred_recordings(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer Drive11 over Drive1 and Drive21 over Drive2 per participant/visit.

    Retain and flag duplicate preferred exports. Missing metadata or unmapped
    drive numbers do not supersede other recordings.
    """
    table = comparison.copy()
    table["scenario"] = _numeric(table.drive).map(SCENARIO_DRIVES).astype("Int64")
    table["selection_reason"] = "normal_recording"
    table.loc[table.scenario.isna(), "selection_reason"] = "unmapped_drive_retained"
    known = table.participant_id.notna() & table.visit.notna()
    table.loc[~known, "selection_reason"] = "insufficient_metadata"
    excluded = set()
    audit = []

    def record(selected: pd.Series, superseded: pd.Series | None = None, notes: str = "") -> None:
        audit.append({
            "participant_id": selected.participant_id, "visit": selected.visit,
            "scenario": selected.scenario, "selected_drive": selected.drive,
            "selected_recording_id": selected.recording_id,
            "superseded_drive": superseded.drive if superseded is not None else pd.NA,
            "superseded_recording_id": superseded.recording_id if superseded is not None else pd.NA,
            "selection_reason": selected.selection_reason, "selected_source_file": selected.source_file,
            "superseded_source_file": superseded.source_file if superseded is not None else pd.NA,
            "notes": notes,
        })

    eligible = known & table.scenario.notna()
    for (_, _, scenario), group in table.loc[eligible].groupby(["participant_id", "visit", "scenario"], sort=True):
        restart = RESTART_DRIVES.get(scenario)
        restarts = group.loc[group.drive.eq(restart)] if restart is not None else group.iloc[:0]
        originals = group.loc[group.drive.eq(scenario)]
        winners = restarts if not restarts.empty else originals
        losers = originals if not restarts.empty else group.iloc[:0]
        excluded.update(losers.index)
        reason = "restart_supersedes_original" if not restarts.empty else "normal_recording"
        table.loc[winners.index, "selection_reason"] = reason
        notes = "Selection uses the specified drive replacement rule; scenario completeness is not inferred."
        if not restarts.empty and originals.empty:
            notes += " Restart selected without an original recording in the input inventory."
        if len(winners) > 1 or len(losers) > 1:
            notes += " Multiple recordings for a drive; preferred-drive copies retained and all superseded associations reported."
            table.loc[winners.index, "notes"] += " | " + notes
        for index in winners.index:
            if losers.empty:
                record(table.loc[index], notes=notes)
            else:
                for _, original in losers.iterrows():
                    record(table.loc[index], original, notes)
    for _, other in table.loc[~eligible].iterrows():
        record(other, notes="No replacement applied: unmapped drive number or missing participant/visit.")
    report = pd.DataFrame(audit, columns=SELECTION_COLUMNS)
    for column in ("scenario", "selected_drive", "superseded_drive"):
        report[column] = report[column].astype("Int64")
    selected = table.drop(index=list(excluded)).reset_index(drop=True)
    return selected, report


def report_recording_chains(comparison: pd.DataFrame) -> pd.DataFrame:
    """Document Drive11/21 superseding Drive1/2 before selection.

    Duplicate exports remain traceable individual pairs.
    """
    rows = []
    known = comparison.dropna(subset=["participant_id", "visit"])
    for (participant, visit), group in known.groupby(["participant_id", "visit"], sort=True):
        drives = _numeric(group.drive)
        for base_drive, related_drive in RESTART_DRIVES.items():
            bases = group.loc[drives.eq(base_drive).fillna(False)]
            related = group.loc[drives.eq(related_drive).fillna(False)]
            if bases.empty or related.empty:
                continue
            notes = (
                "Restart replaces original for analysis under the specified drive selection rule. "
                "Original cleaned files are preserved; marker/RT counts are not combined. "
                "No clock alignment or complete scenario inferred."
            )
            if len(bases) > 1 or len(related) > 1:
                notes += (
                    f" Ambiguous participant/visit group: {len(bases)} base and {len(related)} related recordings; "
                    "all candidate combinations reported. Review source dates and duplicate exports."
                )
            for _, base in bases.iterrows():
                for _, other in related.iterrows():
                    row = {"participant_id": participant, "visit": visit,
                           "base_drive": base_drive, "related_drive": related_drive,
                           "relationship_status": "restart_supersedes_original", "notes": notes}
                    for prefix, source in (("base", base), ("related", other)):
                        for target, origin in (
                            ("row_count", "row_count"), ("start_time", "recording_start_time"),
                            ("end_time", "recording_end_time"), ("rt_episode_count", "rt_episode_count"),
                            ("event_values", "events_values_in_order"), ("intersection_count", "intersection_count"),
                            ("sign_values", "sign_values_in_order"), ("recording_id", "recording_id"),
                            ("source_file", "source_file"),
                        ):
                            row[f"{prefix}_{target}"] = source[origin]
                        if source["notes"]:
                            row["notes"] += f" {prefix} advisories: {source['notes']}"
                    rows.append(row)
    report = pd.DataFrame(rows, columns=RECORDING_CHAIN_COLUMNS)
    for column in (
        "base_drive", "related_drive", "base_row_count", "related_row_count",
        "base_rt_episode_count", "related_rt_episode_count", "base_intersection_count", "related_intersection_count",
    ):
        report[column] = report[column].astype("Int64")
    return report


def build_event_context(frame: pd.DataFrame, result: MarkerDiscoveryResult) -> pd.DataFrame:
    """Describe nonzero Events intervals using chronological (time, row) markers.

    Initial states are not confirmed transitions; untimed markers cannot be
    chronological neighbors. Missing context is NA.
    """
    frame = frame.reset_index(drop=True)
    metadata = {key: result.summary.iloc[0][key] for key in METADATA_COLUMNS}
    events = result.intervals.loc[
        result.intervals.column.eq("Events") & result.intervals.interval_type.eq("value")
    ].sort_values(["interval_start_time", "interval_start_row"], kind="stable", na_position="last")
    actual_changes = result.transitions.loc[
        result.transitions.record_type.eq("transition") & result.transitions.sim_time.notna()
    ].sort_values(["sim_time", "row"], kind="stable")
    speed_changes = actual_changes.loc[actual_changes.column.eq("Speed Limit")]
    sign_changes = actual_changes.loc[
        actual_changes.column.eq("Signs") & ~_numeric(actual_changes.new_value).eq(0).fillna(False)
    ]
    intersections = result.intervals.loc[
        result.intervals.column.eq("Intersections") & result.intervals.interval_type.eq("active")
        & result.intervals.interval_start_time.notna()
    ].sort_values(["interval_start_time", "interval_start_row"], kind="stable")
    rt = result.rt_update_episodes.loc[result.rt_update_episodes.start_time.notna()].sort_values(
        ["start_time", "start_row"], kind="stable",
    )

    def neighbors(table: pd.DataFrame, time_key: str, row_key: str, event_time, event_row):
        if pd.isna(event_time):
            return None, None
        before = table[time_key].lt(event_time) | (table[time_key].eq(event_time) & table[row_key].le(event_row))
        previous, following = table.loc[before], table.loc[~before]
        return (previous.iloc[-1] if not previous.empty else None,
                following.iloc[0] if not following.empty else None)

    rows = []
    for event in events.itertuples(index=False):
        start = int(event.interval_start_row)
        row = {key: pd.NA for key in EVENT_CONTEXT_COLUMNS}
        row.update(metadata)
        row.update(event_value=event.value, event_occurrence=event.occurrence,
                   event_start_time=event.interval_start_time, event_end_time=event.interval_end_time,
                   event_duration_s=event.duration_s)
        if "Speed Limit" in frame:
            row["speed_limit_at_event_start"] = frame.iloc[start]["Speed Limit"]
        if "Intersections" in frame:
            active = _numeric(frame["Intersections"].iloc[start:start + 1]).iloc[0]
            if not pd.isna(active):
                row["intersection_active_at_event_start"] = bool(active != 0)
        for changes, value_name, time_name in (
            (speed_changes, "speed_limit", "speed_change_time"),
            (sign_changes, "sign_value", "sign_time"),
        ):
            previous, following = neighbors(changes, "sim_time", "row", event.interval_start_time, start)
            for prefix, marker in (("previous", previous), ("next", following)):
                if marker is not None:
                    row[f"{prefix}_{value_name}"] = marker.new_value
                    row[f"{prefix}_{time_name}"] = marker.sim_time
        previous, following = neighbors(
            intersections, "interval_start_time", "interval_start_row", event.interval_start_time, start,
        )
        for prefix, marker in (("previous", previous), ("next", following)):
            if marker is not None:
                row[f"{prefix}_intersection_start_time"] = marker.interval_start_time
        if not pd.isna(event.interval_start_time) and not rt.empty:
            deltas = rt.start_time - event.interval_start_time
            # Earlier RT starts win equal-distance ties.
            nearest = rt.loc[deltas.abs().idxmin()]
            row.update(nearest_rt_episode_time=nearest.start_time,
                       nearest_rt_episode_delta_s=nearest.start_time - event.interval_start_time)
        rows.append(row)
    table = pd.DataFrame(rows, columns=EVENT_CONTEXT_COLUMNS)
    table["event_occurrence"] = table.event_occurrence.astype("Int64")
    table["intersection_active_at_event_start"] = table.intersection_active_at_event_start.astype("boolean")
    return table


def _scenario1_evidence_support(result: MarkerDiscoveryResult, context: pd.DataFrame, window_s: float) -> dict:
    """Evaluate the documented Scenario 1 marker-pattern checks."""
    def runs(column, kind="value"):
        return result.intervals.loc[
            result.intervals.column.eq(column) & result.intervals.interval_type.eq(kind)
        ].sort_values("interval_start_row")

    signs, events, intersections = runs("Signs"), runs("Events"), runs("Intersections", "active")
    speed = result.transitions.loc[
        result.transitions.column.eq("Speed Limit") & result.transitions.record_type.eq("transition")
    ]
    route = result.summary.loc[result.summary.column.eq("route_context")].set_index("metric").value
    codes = _numeric(events.value)
    by_code = {code: events.loc[codes.eq(code)] for code in EVENT_CODES}
    support = dict(recording_id=result.summary.iloc[0].recording_id,
                   roundabout=False, collision1=len(by_code[1]) == 1,
                   collision1_rt=route.get("collision1_rt_status") == "associated",
                   collision2=route.get("collision2_candidate_status") == "candidate",
                   collision2_status=route.get("collision2_candidate_status"),
                   school=False, lane=False, region=False)
    for marker in signs.loc[_numeric(signs.value).eq(-3)].itertuples():
        earlier = signs.loc[signs.interval_start_row.lt(marker.interval_start_row)]
        following = speed.loc[speed.new_value.eq(10) & speed.row.ge(marker.interval_start_row)
                              & speed.sim_time.between(marker.interval_start_time, marker.interval_start_time + window_s)]
        if len(earlier) and earlier.iloc[-1].value == 35 and len(following):
            support["roundabout"] = True
    if len(by_code[3]) == 1:
        event = by_code[3].iloc[0]
        onset = speed.sim_time.between(event.interval_start_time, event.interval_start_time + window_s)
        sign20 = signs.loc[_numeric(signs.value).eq(20) & signs.interval_start_time.between(
            event.interval_start_time, event.interval_start_time + window_s)]
        ending = (speed.sim_time - event.interval_end_time).abs().le(window_s) if not pd.isna(event.interval_end_time) else False
        support["school"] = bool(len(sign20)
            and (onset & speed.previous_value.eq(40) & speed.new_value.eq(20)).any()
            and (ending & speed.previous_value.eq(20) & speed.new_value.eq(40)).any())
    if len(by_code[4]) == 1:
        event = by_code[4].iloc[0]
        ctx = context.loc[_numeric(context.event_value).eq(4)].iloc[0]
        if all(not pd.isna(value) for value in (
            ctx.previous_sign_value, ctx.next_sign_value, ctx.next_sign_time, event.interval_end_time,
        )):
            support["lane"] = bool(ctx.previous_sign_value == -6 and ctx.next_sign_value == -7
                                   and event.interval_end_time <= ctx.next_sign_time)
    if len(by_code[5]) == 1 and len(intersections) >= 2:
        event = by_code[5].iloc[0]
        previous = intersections.loc[intersections.last_active_row.lt(event.interval_start_row)]
        following = intersections.loc[intersections.interval_start_row.ge(event.interval_end_row)]
        support["region"] = bool(len(previous) and len(following)
            and previous.iloc[-1].occurrence == len(intersections) - 1
            and following.iloc[0].occurrence == len(intersections))
    for number, prefix in ((1, "collision1"), (2, "collision2_candidate")):
        brake, acceleration = route.get(prefix + "_peak_brake"), route.get(prefix + "_peak_deceleration")
        support[f"collision{number}_braking"] = bool(
            (number == 1 or support["collision2"])
            and ((not pd.isna(brake) and brake > 0) or (not pd.isna(acceleration) and acceleration < 0)))
    return support


def summarize_scenario1_evidence(records: Sequence[dict], **settings) -> pd.DataFrame:
    """Summarize reviewed candidate checks; confidence is not a scoring rule.

    KFC entry/exit matches remain NA even when the region pattern is supported.
    """
    n = len(records)

    def count(key):
        return sum(bool(record[key]) for record in records)

    window = settings.get("events_near_window_s", DEFAULT_EVENTS_NEAR_WINDOW_S)
    definitions = [
        ("Roundabout", "Signs 35 -> -3 plus Speed Limit -> 10", "roundabout", "high", "high_confidence",
         f"Consecutive nonzero signs 35 then -3, followed by speed limit -> 10 within {window:g} s."),
        ("Collision Avoidance 1", "Events == 1", "collision1", "high", "high_confidence",
         f"One Events 1 interval; nearby RT in {count('collision1_rt')}/{n}, post-event braking in {count('collision1_braking')}/{n}. RT is optional."),
        ("Collision Avoidance 2", "Unassociated RT update between Events 1 and Events 3", "collision2", "moderate", "candidate",
         f"One eligible RT update under existing association logic; braking in {count('collision2_braking')}/{count('collision2')} unique candidates."),
        ("School Zone", "Events == 3 with Sign 20 and Speed Limit 40 -> 20 -> 40", "school", "high", "high_confidence",
         f"Sign 20 and 40 -> 20 within {window:g} s after onset; 20 -> 40 within +/-{window:g} s of exclusive end."),
        ("Lane Closure", "Events == 4 between Sign -6 and Sign -7", "lane", "moderate_to_high", "strong_candidate",
         "One Events 4 interval bracketed by nonzero signs -6 and -7, ending before sign -7."),
        ("Enter KFC candidate", "Unverified entry boundary before Events == 5", None, "insufficient", "unresolved_candidate",
         f"Broad region context supported in {count('region')}/{n}; no entry-specific boundary established."),
        ("Drive-through/KFC region candidate", "Events == 5 between the final two intersection intervals", "region", "moderate", "candidate",
         "One Events 5 interval wholly after the penultimate active intersection and before the final active intersection."),
        ("Exit KFC candidate", "Unverified exit boundary after Events == 5", None, "insufficient", "unresolved_candidate",
         f"Broad region context supported in {count('region')}/{n}; no exit-specific boundary established."),
    ]
    rows = []
    for maneuver, marker, key, confidence, status, pattern in definitions:
        notes = "Exploratory evidence only; no segmentation or scoring-item mapping. "
        if key is None:
            notes += "recordings_matching is NA because an item-specific match has not been established. "
        if key == "collision2":
            notes += "RT ordinal never identifies collision identity; ambiguous recordings are nonmatches for the unique-candidate check. "
        notes += "KFC region support does not establish entry/exit scoring-item boundaries. " if key == "region" else ""
        notes += json.dumps({
            "cohort": [r["recording_id"] for r in records],
            "matching_recordings": [r["recording_id"] for r in records if r[key]] if key else None,
            "nonmatching_recordings": [r["recording_id"] for r in records if not r[key]] if key else None,
            "settings": settings,
        })
        rows.append(dict(maneuver=maneuver, candidate_marker=marker, supporting_pattern=pattern,
                         recordings_evaluated=n, recordings_matching=count(key) if key else pd.NA,
                         confidence=confidence, status=status, notes=notes))
    table = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)
    for column in ("recordings_evaluated", "recordings_matching"):
        table[column] = table[column].astype("Int64")
    return table


def output_event_context(table: pd.DataFrame, output_dir: Path | None) -> list[Path]:
    """Write event context and return only paths successfully created."""
    if output_dir is None:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "event_context.csv"
    table.to_csv(path, index=False, na_rep="NA")
    return [path]


def write_outputs(result: MarkerDiscoveryResult, output_dir: str | Path, recording_id: str) -> list[Path]:
    """Export the four single-recording tables and return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    for name in SINGLE_OUTPUTS:
        path = directory / f"{recording_id}_{name}.csv"
        getattr(result, name).to_csv(path, index=False, na_rep="NA")
        created.append(path)
    return created


def print_created(paths: Sequence[Path]) -> None:
    """Report written files without printing analysis or routine diagnostics."""
    if paths:
        print("Created:")
        for path in paths:
            print(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--file", type=Path, help="One cleaned recording CSV")
    inputs.add_argument("--files", nargs="+", type=Path, metavar="PATH",
                        help="Batch: input recordings; prefer same-participant/visit restart recordings")
    inputs.add_argument("--glob", dest="glob_pattern", help="Batch: quoted cleaned-CSV glob; supports **")
    inputs.add_argument("--directory", type=Path, help="Batch: recursively find *_cleaned.csv in this directory")
    parser.add_argument("--output-dir", type=Path, help="Write CSV tables here; omit to validate without writing or success output")
    parser.add_argument("--brake-threshold", type=float, default=None,
                        help="Optional positive force in N; classify intervals at or above this value")
    for flag, default, description in (
        ("--rt-transition-gap", DEFAULT_RT_GAP_S, "Stable gap separating RT update episodes"),
        ("--post-rt-window", DEFAULT_POST_RT_WINDOW_S, "Follow-up window after the final RT update"),
        ("--events-near-window", DEFAULT_EVENTS_NEAR_WINDOW_S, "Events 1/RT proximity and evidence-check tolerance"),
        ("--post-event-window", DEFAULT_POST_EVENT_WINDOW_S, "Braking follow-up window after Events 1"),
    ):
        parser.add_argument(flag, type=float, default=default, metavar="SECONDS",
                            help=f"{description} (default: {default:g} s; exploratory, not a scientific cutoff)")
    parser.add_argument("--scenario", type=int, choices=(1, 2, 3),
                        help="Batch: filter after preferred selection (1: Drive11/1; 2: Drive21/2; 3: Drive3)")
    reports = parser.add_mutually_exclusive_group()
    reports.add_argument("--event-context", action="store_true",
                         help="Only write event_context.csv; existing report files are left untouched")
    reports.add_argument("--scenario1-evidence", action="store_true",
                         help="Reproduce the existing Scenario 1 candidate-evidence summary only")
    args = parser.parse_args(argv)
    options = dict(brake_threshold=args.brake_threshold, rt_transition_gap_s=args.rt_transition_gap,
                   post_rt_window_s=args.post_rt_window, events_near_window_s=args.events_near_window,
                   post_event_window_s=args.post_event_window)
    created = []
    try:
        if args.file is not None:
            if args.scenario is not None or args.scenario1_evidence:
                parser.error("--scenario and --scenario1-evidence require batch input (--files, --glob, or --directory)")
            frame, result = _read_recording(args.file, **options)
            if args.event_context:
                created = output_event_context(build_event_context(frame, result), args.output_dir)
            elif args.output_dir is not None:
                created = write_outputs(result, args.output_dir, args.file.stem)
            print_created(created)
            return 0
        if args.scenario1_evidence and args.scenario not in (None, 1):
            parser.error("--scenario1-evidence applies only to Scenario 1")
        files = args.files if args.files is not None else discover_cleaned_files(
            directory=args.directory, pattern=args.glob_pattern)
        report_mode = "context" if args.event_context else "evidence" if args.scenario1_evidence else None
        inventory, details = _collect_recordings(include_restart_recordings(files), options, report=report_mode)
        comparison, selection = select_preferred_recordings(inventory)
        scenario = 1 if args.scenario1_evidence else args.scenario
        if scenario is not None:
            comparison = comparison.loc[comparison.scenario.eq(scenario).fillna(False)].reset_index(drop=True)
            selection = selection.loc[selection.scenario.eq(scenario).fillna(False)].reset_index(drop=True)
        if report_mode is not None:
            selected_details = [details[file] for file in comparison.source_file]
            if args.event_context:
                context = pd.concat(selected_details, ignore_index=True) if selected_details else pd.DataFrame(columns=EVENT_CONTEXT_COLUMNS)
                created = output_event_context(context, args.output_dir)
            else:
                evidence = summarize_scenario1_evidence(selected_details, **options)
                if args.output_dir is not None:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    path = args.output_dir / "scenario1_marker_evidence.csv"
                    evidence.to_csv(path, index=False, na_rep="NA")
                    created.append(path)
            print_created(created)
            return 0
        chains = report_recording_chains(inventory)
        if scenario is not None:
            chains = chains.loc[chains.base_drive.eq(scenario)].reset_index(drop=True)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for name, table in (
                ("scenario_comparison", comparison.loc[:, COMPARISON_COLUMNS]),
                ("recording_selection", selection), ("recording_chain_qc", chains),
                ("scenario_comparison_details", comparison),
            ):
                path = args.output_dir / f"{name}.csv"
                table.to_csv(path, index=False, na_rep="NA")
                created.append(path)
        print_created(created)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
