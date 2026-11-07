"""Timing and flow measurements from an already located lane-closure window.

Landmarks are inputs, not inferred from clinician grades. Corridor thresholds are
centering proxies, not vehicle-body clearance or accepted-gap measurements.
"""
import numpy as np
from src.representation.segmentation import intervals
from src.representation.event_features import lane_closure_windows


DEBUG = True


# native clock + requested interval -> sample durations, rejecting unknown gaps
def window_weights(times, start_s, end_s):
    """Times + bounds -> sample durations; clip edge samples, reject gaps"""
    times = np.asarray(times, dtype=float)
    if not times[0] <= start_s < end_s <= times[-1]: raise ValueError('window outside recorded time')
    step = np.diff(times)
    weights = np.maximum(0, np.minimum(times[1:], end_s)-np.maximum(times[:-1], start_s))
    used = weights > 0
    if not np.isfinite(step[used]).all() or not ((step[used] > 0) & (step[used] <= .1)).all(): raise ValueError('clock gap / bad timestep in window')
    if not np.isclose(weights.sum(), end_s-start_s, atol=1e-5): raise ValueError('window missing time samples')
    return np.r_[weights, 0.]


# telem + sign/barrel/end/completion times + target lane -> timing/flow features
def calc_lane_closure_features(dr_telem, sign_s, barrel_s, end_s, target_lane,
                               completion_s=None, centering_m=.9, return_hold_s=1):
    """Telem + sign/barrel/end times -> timing/flow dict; positive lead = merged before barrel"""
    t = dr_telem['Sim Time'].to_numpy(float)
    xy = dr_telem[['X Position', 'Y Position']].to_numpy(float)
    sign_xy = np.array([np.interp(sign_s, t, xy[:, j]) for j in [0, 1]])
    end_xy = np.array([np.interp(end_s, t, xy[:, j]) for j in [0, 1]])
    axis = end_xy-sign_xy
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length < 30: raise ValueError('cant get local road direction')
    progress = (xy-sign_xy)@(axis/length)

    flow_w = window_weights(t, barrel_s, end_s)
    v = dr_telem['Velocity'].to_numpy(float)
    lane = dr_telem['Lane Number'].to_numpy(float)
    offset = dr_telem['Lane Offset'].to_numpy(float)
    used = flow_w > 0
    if not np.isfinite(np.column_stack([v, lane, offset])[used]).all(): raise ValueError('missing flow measurements')

    # BASELINE: crossing of 50m before the warning sign, within the prior 20s
    before = (t[:-1] >= sign_s-20) & (t[1:] <= sign_s)
    crossings = np.flatnonzero(before & (progress[:-1] <= -50) & (progress[1:] > -50))
    baseline_s = np.nan
    baseline_mph = np.nan
    if len(crossings) == 1:
        i = crossings[0]
        baseline_s = float(np.interp(-50, progress[i:i+2], t[i:i+2]))
        try:
            baseline_w = window_weights(t, baseline_s, sign_s)
            if np.isfinite(v[baseline_w > 0]).all(): baseline_mph = np.dot(np.nan_to_num(v), baseline_w)/baseline_w.sum()/.44704
        except ValueError:
            pass  # unknown baseline stays blank, not 0 speed reduction

    flow_mph = np.dot(v[used], flow_w[used])/flow_w.sum()/.44704
    centered = (np.abs(lane) == target_lane) & (np.abs(offset) <= centering_m)
    offcenter_fraction = np.dot((~centered)[used], flow_w[used])/flow_w.sum()

    # TIMING: positive lead means completion before reaching 1st barrel
    completed = completion_s is not None and np.isfinite(completion_s)
    lead_s = barrel_s-completion_s if completed else np.nan
    lead_m = (np.interp(barrel_s, t, progress)-np.interp(completion_s, t, progress)) if completed else np.nan
    returned = np.nan
    if completed and completion_s+return_hold_s < end_s:
        after_s = max(barrel_s, completion_s+return_hold_s)
        return_w = window_weights(t, after_s, end_s)
        # abs lane number so road direction doesnt flip lane id
        wrong_lane = (np.abs(lane) != target_lane) & (return_w > 0)
        starts = np.flatnonzero(wrong_lane & ~np.r_[False, wrong_lane[:-1]])
        ends = np.flatnonzero(wrong_lane & ~np.r_[wrong_lane[1:], False])+1
        returned = int(any(return_w[a:b].sum() >= return_hold_s-1e-8 for a, b in zip(starts, ends)))


    result = dict(completion_lead_s=lead_s, completion_lead_m=lead_m,
        no_completion=int(not completed), baseline_start_s=baseline_s,
        baseline_speed_mph=baseline_mph, zone_speed_mph=flow_mph,
        speed_reduction_mph=baseline_mph-flow_mph,
        outside_corridor_fraction=offcenter_fraction, return_in_window=returned,
        near_stop_s=float(flow_w[(v < 1) & used].sum()), flow_duration_s=end_s-barrel_s)
    if 'Collision' in dr_telem:
        contact = dr_telem['Collision'].to_numpy(float)
        result['contact_in_window'] = int(np.any(contact[used] == 1)) if np.isfinite(contact[used]).all() else np.nan
    return result



LANE_COLS = ["Sim Time", "Signs", "Events", "Velocity", "Lane Number", "Lane Offset", "X Position", "Y Position"]


def locate_lane_closure(telem_df):
    time_srs=telem_df['Sim Time'];dt_srs=time_srs.shift(-1)-time_srs
    windows=lane_closure_windows(telem_df,dt_srs)
    if len(windows)!=1:return None,'no_unique_closure'
    a,b=windows[0]
    signs=[(s,e)for s,e in intervals(telem_df.Signs.eq(-6))if e<=a]
    if not signs:return None,'missing_sign'
    s,e=signs[-1];sign_time=(time_srs.iloc[s]+time_srs.iloc[e-1])/2
    xy=telem_df[['X Position','Y Position']].to_numpy()
    sign_xy=np.array([np.interp(sign_time,time_srs,xy[:,j])for j in [0,1]])
    axis=xy[b-1]-sign_xy;length=np.linalg.norm(axis)
    if not np.isfinite(length) or length<30:return None,'bad_road_axis'
    axis=axis/length;progress=(xy-sign_xy)@axis
    begin=int(time_srs.searchsorted(sign_time-20))
    if begin==0 or b>=len(telem_df) or not dt_srs.iloc[begin:b].between(1e-8,.1).all():return None,'window_clock_gap_or_truncation'
    if telem_df.iloc[begin:b][LANE_COLS].isna().any().any():return None,'missing_window_values'
    # skip if backtracking too far / road layout doesnt match
    if np.min(np.diff(progress[s:b])) < -.5:return None,'backtracking_review'
    return dict(a=a,b=b,s=s,e=e,begin=begin,sign_time=sign_time,sign_lo=time_srs.iloc[s],sign_hi=time_srs.iloc[e-1],
                sign_xy=sign_xy,axis=axis,progress=progress,event_distance=progress[a]),'located'


def crossing_time(t,progress,start,end,distance):
    candidates=np.flatnonzero((progress[start:end-1]<=distance)&(progress[start+1:end]>distance))+start
    if len(candidates)!=1:return np.nan
    i=int(candidates[0]);return float(np.interp(distance,progress[i:i+2],t.iloc[i:i+2]))


def settled_lane_change(telem_df,context_dict,barrel_time,target_lane,tolerance):
    """Telem + target lane -> merge timing dict; need 1s centered, flag already-centered start"""
    begin,end=context_dict['begin'],context_dict['b'];time_srs=telem_df['Sim Time'];dt_srs=time_srs.shift(-1)-time_srs
    mask=telem_df['Lane Number'].abs().eq(target_lane)&telem_df['Lane Offset'].abs().le(tolerance)
    runs=intervals(mask.iloc[begin:end].reset_index(drop=True))
    found=[]
    for s,e in runs:
        s+=begin;e+=begin
        if dt_srs.iloc[s:e].sum()>=1 and time_srs.iloc[e]>barrel_time:found.append((s,e))
    if not found:return dict(lead_s=np.nan,settled_by_barrel=0,no_completion=1,left_censored=0,completion_s=np.nan)
    s,e=found[0]
    return dict(lead_s=barrel_time-time_srs.iloc[s],settled_by_barrel=int(time_srs.iloc[s]<=barrel_time),no_completion=0,
                left_censored=int(s==begin),completion_s=time_srs.iloc[s])


# ALIGN: sign and fixed barrel distance -> merge timing and flow

def calc_aligned_lane_features(telem_df, scenario_id, anchors_df):
    """Telem + anchors -> feature dict + status; check road match before using barrel location"""
    context_dict, status = locate_lane_closure(telem_df)
    if context_dict is None: return {}, status
    anchor_srs = anchors_df.loc[scenario_id]

    if abs(context_dict["event_distance"] - anchor_srs.event_distance) > 10: return {}, "geometry_mismatch"
    barrel_s = crossing_time(telem_df["Sim Time"], context_dict["progress"], context_dict["s"], context_dict["b"], anchor_srs.distance_after_sign_m)
    if not np.isfinite(barrel_s): return {}, "no_barrel_crossing"
    target_lane = {1: 1, 2: 2, 3: 2}[scenario_id]

    completion_dict = settled_lane_change(telem_df, context_dict, barrel_s, target_lane, .9)
    completion_s = np.nan
    if not completion_dict["left_censored"]: completion_s = completion_dict["completion_s"]
    feature_dict = calc_lane_closure_features(telem_df, context_dict["sign_time"], barrel_s, float(telem_df["Sim Time"].iloc[context_dict["b"]]), target_lane, completion_s)
    if DEBUG:
        print("lane", scenario_id, "sign/barrel s", context_dict["sign_time"], barrel_s)
        print("completion", completion_dict)
        print("lane values", feature_dict)

    # check speed avg uses time weights, not number of rows
    """
    print("target lane", target_lane, "completion s", completion_s)
    print("anchor distance / observed", anchor_srs.event_distance, context_dict["event_distance"])
    # TODO NEED TO plot lane offset with barrel time if completion looks late
    """
    return feature_dict, "aligned"
