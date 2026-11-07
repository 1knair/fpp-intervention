import numpy as np
import pandas as pd

from src.representation.segmentation import intervals

# dr = drive; curve = 1 moving brake press

BRAKE_COLS = ["Sim Time", "Velocity", "Acceleration", "Brake Pedal Force"]
BRAKE_FEATURE_COLS = [
    "brake_episodes_per_km", "brake_rise_time_median_s",
    "brake_jerk_rms_p90_mps3", "brake_peak_decel_p90_mps2",
]
CURVE_COLS = [
    "start_row", "end_row", "start_time_s", "end_time_s", "duration_s",
    "start_speed_mps", "speed_loss_mps", "peak_force_n", "rise_time_s",
    "jerk_rms_mps3", "peak_decel_mps2", "end_reason",
]
DEBUG = True

PRESS_N, RELEASE_N = 5, 2
MIN_BRAKE_S, MERGE_RELEASE_S, MAX_INTERVAL_S = 0.5, 0.2, 0.1


# raw telem -> smoothed pedal force, acc and jerk
def braking_signals(dr_telem):
    """Drive telem -> force/accel/jerk DF; smooth within clock blocks only"""
    telem = dr_telem[BRAKE_COLS].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    times = telem["Sim Time"]
    steps = times.diff()
    known = telem.notna().all(axis=1)
    blocks = (~steps.gt(0) | steps.gt(MAX_INTERVAL_S) | ~known | ~known.shift(fill_value=False)).cumsum()
    signals = pd.DataFrame(index=telem.index, columns=["force_n", "acc_mps2", "jerk_mps3"], dtype=float)

    if DEBUG: print("brake signals", len(telem), "rows /", blocks.loc[known].nunique(), "clock blocks")

    for _, block in telem.loc[known].groupby(blocks.loc[known], sort=False):
        if len(block) < 3: continue
        timed = block.set_index(pd.to_timedelta(block["Sim Time"], unit="s"))
        force = timed["Brake Pedal Force"].rolling("100ms", center=True, min_periods=1).mean().to_numpy()
        acc = timed["Acceleration"].rolling("200ms", center=True, min_periods=3).mean()
        jerk = acc.diff().to_numpy() / block["Sim Time"].diff().to_numpy()
        # print("acc diffs / dt", acc.diff().head(), block["Sim Time"].diff().head())

        signals.loc[block.index, "force_n"] = force
        signals.loc[block.index, "acc_mps2"] = acc.to_numpy()
        signals.loc[block.index, "jerk_mps3"] = jerk
    return signals


# a force ramp + times + level -> interpolated 1st upward crossing time
def crossing_time(force, times, level):
    index = int(np.flatnonzero(force >= level)[0])
    if index == 0: return float(times[0])
    fraction = (level - force[index - 1]) / (force[index] - force[index - 1])
    # print("crossing", level, "force", force[index-1:index+1], "fraction", fraction)
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


# one cleaned drive -> episode table (exact CLEANED row bounds + curve features)
def find_braking_curves(dr_telem):
    """Drive telem -> brake episodes DF; merge brief releases, skip gaps/cut off presses"""
    telem = dr_telem[BRAKE_COLS].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    times, force = telem["Sim Time"], telem["Brake Pedal Force"]
    dt = times.shift(-1) - times
    known = telem.notna().all(axis=1)
    valid = dt.gt(0) & dt.le(MAX_INTERVAL_S) & known & known.shift(-1, fill_value=False)

    # DETECT: low release threshold avoids splitting a press at small force changes
    presses = []
    for start, end in intervals(force.gt(RELEASE_N)):
        if presses:
            previous_start, previous_end = presses[-1]
            gap_s = times.iloc[start] - times.iloc[previous_end]
            if gap_s <= MERGE_RELEASE_S + 1e-9 and valid.iloc[previous_end:start].all():
                presses[-1] = (previous_start, end)
                continue
        presses.append((start, end))

    if DEBUG: print("brake candidates", len(presses), "after merging short releases")


    # MEASURE: reject unknown edges/gaps; stop the slowing curve when car stops
    signals = braking_signals(telem)

    rows = []
    for start, release in presses:
        if start == 0 or release == len(telem) or not valid.iloc[start - 1:release].all(): continue
        if force.iloc[start:release].max() < PRESS_N or telem["Velocity"].iloc[start] <= 1: continue
        stop_rows = telem.index[start:release][telem["Velocity"].iloc[start:release].le(1)]
        end = int(stop_rows[0]) if len(stop_rows) else release
        if times.iloc[end] - times.iloc[start] < MIN_BRAKE_S - 1e-9: continue

        curve = signals.iloc[start - 1:end + 1]
        peak_row = curve["force_n"].idxmax()
        ramp = signals.loc[start - 1:peak_row, "force_n"].to_numpy()
        ramp_times = times.loc[start - 1:peak_row].to_numpy()
        amplitude = ramp.max() - ramp[0]

        rise_time = np.nan
        if amplitude > 0:
            rise_time = (crossing_time(ramp, ramp_times, ramp[0] + 0.9 * amplitude)
                         - crossing_time(ramp, ramp_times, ramp[0] + 0.1 * amplitude))

        weights = dt.iloc[start:end]
        squared_jerk = signals["jerk_mps3"].iloc[start:end].pow(2)
        measured = squared_jerk.notna()
        jerk_rms = np.sqrt((squared_jerk * weights).sum() / weights.loc[measured].sum()) if measured.any() else np.nan

        if DEBUG and len(rows) < 3:
            print("brake rows", start, end, "time", times.iloc[start], times.iloc[end])
            print("force amp", amplitude, "rise 10-90% s", rise_time, "jerk rms", jerk_rms)

        # compare weighted jerk by hand on one curve
        """
        check_df = pd.DataFrame({"dt": weights, "jerk_sq": squared_jerk})
        check_df["weighted"] = check_df.dt * check_df.jerk_sq
        print(check_df.head(12))
        jerk_sum = check_df.weighted.sum()
        known_seconds = check_df.loc[measured, "dt"].sum()
        print("sum jerk^2*dt", jerk_sum, "/ known s", known_seconds)
        if known_seconds > 0:
            check_rms = np.sqrt(jerk_sum / known_seconds)
            assert np.isclose(check_rms, jerk_rms)
        # print(ramp_times, ramp)  # check 10/90 crossing if rise looks off
        """

        rows.append({
            "start_row": start, "end_row": end + 1,
            "start_time_s": times.iloc[start], "end_time_s": times.iloc[end],
            "duration_s": times.iloc[end] - times.iloc[start],
            "start_speed_mps": telem["Velocity"].iloc[start],
            "speed_loss_mps": telem["Velocity"].iloc[start] - telem["Velocity"].iloc[end],
            "peak_force_n": curve["force_n"].max(), "rise_time_s": rise_time,
            "jerk_rms_mps3": jerk_rms,
            "peak_decel_mps2": max(0, -curve["acc_mps2"].min()),
            "end_reason": "stopped" if len(stop_rows) else "released",
        })
    if DEBUG: print("kept brake curves", len(rows), "/", len(presses))
    return pd.DataFrame(rows, columns=CURVE_COLS)


# episode table + source telem -> four drive-level inputs and measured exposure
def summarize_braking(curves, dr_telem):
    """Brake episodes + drive telem -> feature dict; rates use moving distance"""
    telem = dr_telem[BRAKE_COLS].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    dt = telem["Sim Time"].shift(-1) - telem["Sim Time"]
    measured = (dt.gt(0) & dt.le(MAX_INTERVAL_S) & telem["Velocity"].gt(1)
                & telem.notna().all(axis=1))
    distance_km = (telem["Velocity"] * dt).loc[measured].sum() / 1000

    if DEBUG: print("brake rate", len(curves), "curves / km", distance_km)
    # print(curves[["duration_s", "rise_time_s", "jerk_rms_mps3", "peak_decel_mps2"]].describe())

    return {
        "brake_episodes_per_km": len(curves) / distance_km if distance_km else np.nan,
        "brake_rise_time_median_s": curves["rise_time_s"].median(),
        "brake_jerk_rms_p90_mps3": curves["jerk_rms_mps3"].quantile(0.9),
        "brake_peak_decel_p90_mps2": curves["peak_decel_mps2"].quantile(0.9),
        "brake_curve_count": len(curves), "brake_curve_exposure_km": distance_km,
    }
