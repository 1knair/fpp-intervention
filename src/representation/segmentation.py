import pandas as pd

def intervals(mask):
    mask = mask.fillna(False).reset_index(drop=True)
    starts = mask.index[mask & ~mask.shift(fill_value=False)]
    ends = mask.index[mask & ~mask.shift(-1, fill_value=False)] + 1
    return list(zip(starts, ends))

def roundabout(frame):
    # 35 sign -> -3 /// 10mph zone -> limit leaves 10; approach included
    if not {"Signs", "Speed Limit"}.issubset(frame.columns):
        return None, "missing_headers"
    candidates = []
    for start, end in intervals(frame["Speed Limit"].eq(10)):
        if end == len(frame):
            continue
        previous = frame.iloc[:start + 1]
        signs = previous.index[previous["Signs"].eq(-3)]
        if len(signs) == 0:
            continue
        marker = signs[-1]
        if "Sim Time" in frame and frame.loc[start, "Sim Time"] - frame.loc[marker, "Sim Time"] > 2:
            continue
        sign35 = [a for a, b in intervals(previous["Signs"].eq(35)) if b <= marker]
        if sign35:
            candidates.append((sign35[-1], end))
    if len(candidates) != 1:
        return None, "weird_bounds"
    return candidates[0], "candidate"

def collision_avoidance(frame):
    # first car pulling out; includes the marker's "extra time" recovery period
    if "Events" not in frame:
        return None, "missing_headers"
    candidates = intervals(frame["Events"].eq(1))
    if len(candidates) != 1:
        return None, "weird_bounds"
    start, end = candidates[0]
    if start == 0 or end == len(frame):
        return None, "truncated_marker"
    return (start, end), "candidate"

def not_implemented(frame):
    # TODO define after video / telemetry checks
    return None, "rule_not_implemented"

RULES = {
    "Roundabout": roundabout,
    "Collision Avoidance": collision_avoidance,
    "Collisions Avoid 2": not_implemented,
    "Speed Hump": not_implemented,
    "Speed Hump 2": not_implemented,
    "Unprotected Left 1 PED X": not_implemented,
    "Unprotected Left 2 Cyclist": not_implemented,
    "Unprotected Left 1": not_implemented,
    "Unprotected Left 2 PED X": not_implemented,
    "Unprotected Left 1 Cyclist": not_implemented,
    "Unprotected left 2": not_implemented,
    "Straight Intersection with Stop": not_implemented,
    "Straight Intersection with Stop with Cyclists": not_implemented,
    "PED X": not_implemented,
    "Four Lane": not_implemented,
    "Lane Closure": not_implemented,
    "School Zone": not_implemented,
    "Enter KFC": not_implemented,
    "Enter and Exit Drivethru": not_implemented,
    "Exit KFC": not_implemented,
}

def segment_recording(telemetry_frame, selected, drive_metadata):
    # selected mapping -> actual bounds for this recording
    frame = telemetry_frame.reset_index(drop=True)
    rows = []
    for mapping in selected.to_dict("records"):
        rule = RULES.get(mapping["segment_name"], not_implemented)
        bounds, status = rule(frame)
        row = dict(mapping)
        row.update(drive_metadata)
        row.update({"start_row": pd.NA, "end_row": pd.NA,
                    "start_time_s": pd.NA, "end_time_s": pd.NA,
                    "segmentation_status": status})
        if bounds is not None:
            start, end = bounds
            row.update({"start_row": int(start), "end_row": int(end)})
            if "Sim Time" in frame:
                row["start_time_s"] = frame.loc[start, "Sim Time"]
                row["end_time_s"] = frame.loc[end, "Sim Time"]
        rows.append(row)
    return pd.DataFrame(rows)
