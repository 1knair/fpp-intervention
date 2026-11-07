# Names: seg = segment; rec = recording; dr = drive; telem = telemetry
# cl = clinician; cols = columns; meta = mdata; row = one table row
import pandas as pd

def intervals(mask):
    """Mask -> start/end row pairs; end row excluded"""
    mask = mask.fillna(False).reset_index(drop=True)
    starts = mask.index[mask & ~mask.shift(fill_value=False)]
    ends = mask.index[mask & ~mask.shift(-1, fill_value=False)] + 1
    return list(zip(starts, ends))

def roundabout(telem):
    # 35 sign -> -3 /// 10mph zone -> limit leaves 10; approach included
    if not {"Signs", "Speed Limit"}.issubset(telem.columns): return None, "missing_headers"
    candidates = []
    for start, end in intervals(telem["Speed Limit"].eq(10)):
        if end == len(telem): continue
        approach_telem = telem.iloc[:start + 1]
        signs = approach_telem.index[approach_telem["Signs"].eq(-3)]
        if len(signs) == 0: continue
        marker = signs[-1]
        if "Sim Time" in telem and telem.loc[start, "Sim Time"] - telem.loc[marker, "Sim Time"] > 2: continue
        sign35 = [a for a, b in intervals(approach_telem["Signs"].eq(35)) if b <= marker]
        if sign35: candidates.append((sign35[-1], end))
    if len(candidates) != 1: return None, "weird_bounds"
    return candidates[0], "candidate"

def collision_avoidance(telem):
    # 1st car pulling out; includes the marker's "extra time" recovery period
    if "Events" not in telem: return None, "missing_headers"
    candidates = intervals(telem["Events"].eq(1))
    if len(candidates) != 1: return None, "weird_bounds"
    start, end = candidates[0]
    if start == 0 or end == len(telem): return None, "truncated_marker"
    # event codes change meaning across scenarios; scenario 3 uses code 1
    # for lane closure, independently identified by its -6/-7 sign brackets
    if "Signs" in telem:
        before = telem["Signs"].iloc[:start].dropna()
        after = telem["Signs"].iloc[end:].dropna()
        before = before.loc[before.ne(0)]
        after = after.loc[after.ne(0)]
        if len(before) and len(after) and before.iloc[-1] == -6 and after.iloc[0] == -7: return None, "event1_is_lane_closure"
    return (start, end), "candidate"

def not_implemented(telem):
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

def segment_recording(dr_telem, selected_segs, dr_meta):
    # selected mapping -> actual bounds for this recording
    """Drive telem + rules + mdata -> seg index DF; bounds use cleaned rows"""
    telem = dr_telem.reset_index(drop=True)
    rows = []
    for seg_mapping in selected_segs.to_dict("records"):
        rule = RULES.get(seg_mapping["segment_name"], not_implemented)
        bounds, status = rule(telem)
        row = dict(seg_mapping)
        row.update(dr_meta)
        row.update({"start_row": pd.NA, "end_row": pd.NA, "start_time_s": pd.NA, "end_time_s": pd.NA, "segmentation_status": status})
        if bounds is not None:
            start, end = bounds
            row.update({"start_row": int(start), "end_row": int(end)})
            if "Sim Time" in telem:
                row["start_time_s"] = telem.loc[start, "Sim Time"]
                row["end_time_s"] = telem.loc[end, "Sim Time"]
        rows.append(row)
    return pd.DataFrame(rows)
