"""Find event windows and measure their telem features
1 source clock at a time, restart handling stays in extract_features
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.representation.braking_features import braking_signals, find_braking_curves
from src.representation.segmentation import intervals, roundabout, collision_avoidance
from src.representation.drive_features import complete_windows, school_zone_windows


CONTEXT_COLS = [
    'Sim Time', 'Velocity', 'Acceleration', 'Brake Pedal Force', 'Speed Limit',
    'Signs', 'Events', 'Z Position', 'X Position', 'Y Position', 'Lane Number',
]
DEBUG = True

HUMP_PROMINENCE_M = .09
APPROACH_M = 100


# HUMPS: vertical position + valid clock blocks -> candidate crest rows
def find_hump_candidates(dr_telem):
    """Drive telem -> hump candidates DF; Z peaks only, need location check"""
    telem = dr_telem.reset_index(drop=True)
    columns = ['peak_row', 'time_s', 'prominence_m', 'speed_mph', 'x_m', 'y_m', 'z_m']
    times, height = telem['Sim Time'], telem['Z Position']
    valid = times.notna() & height.notna()
    steps = times.diff()
    blocks = (~steps.gt(0) | steps.gt(.1) | ~valid | ~valid.shift(fill_value=False)).cumsum()
    rows = []
    for _, part in telem.loc[valid].groupby(blocks.loc[valid], sort=False):
        if len(part) < 3: continue
        sampling_step = part['Sim Time'].diff().median()
        peaks, properties = find_peaks(part['Z Position'].to_numpy(), prominence=HUMP_PROMINENCE_M, distance=max(1, round(1 / sampling_step)))
        for position, prominence in zip(peaks, properties['prominences']):
            row = part.iloc[position]
            if row['Speed Limit'] not in (15, 25) or pd.isna(row['Velocity']) or row['Velocity'] <= 1: continue
            rows.append({'peak_row': int(part.index[position]), 'time_s': row['Sim Time'],
                'prominence_m': prominence, 'speed_mph': row['Velocity'] / .44704,
                'x_m': row['X Position'], 'y_m': row['Y Position'], 'z_m': row['Z Position']})
    if DEBUG: print("hump candidates", len(rows), "/", len(telem), "rows")
    # print(pd.DataFrame(rows)[["time_s", "prominence_m", "speed_mph"]] if rows else "no crests")
    return pd.DataFrame(rows, columns=columns)


# LANE CLOSURE: Jacob's -6/-7 signs -> candidate windows
def lane_closure_windows(telem, dt):
    found = []
    signs = telem['Signs']
    for code in telem['Events'].dropna().unique():
        if code == 0: continue
        for start, end in complete_windows(telem['Events'].eq(code), dt):
            before = signs.iloc[:start].loc[lambda x: x.notna() & x.ne(0)]
            after = signs.iloc[end:].loc[lambda x: x.notna() & x.ne(0)]
            if len(before) and len(after) and before.iloc[-1] == -6 and after.iloc[0] == -7: found.append((start, end))
    return found


# FEATURES: one drive + braking curves -> speed hump and lane features
def calc_event_features(dr_telem, brake_curves):
    """Telem + brake curves -> feature dict/hump rows/lane windows; missing events stay NaN"""
    telem = dr_telem[CONTEXT_COLS].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    times, velocity = telem['Sim Time'], telem['Velocity']
    mph = velocity / .44704
    dt = times.shift(-1) - times
    good = dt.gt(0) & dt.le(.1)
    # skip clock gaps when adding distance
    distance = (velocity.clip(lower=0) * dt).where(good, 0).fillna(0).cumsum().shift(fill_value=0)
    timed_speed = pd.Series(mph.to_numpy(), index=pd.to_timedelta(times, unit='s'))

    # needs a full 1s inside the window
    smooth_speed = timed_speed.rolling('1s', min_periods=3).mean().to_numpy()
    signals = braking_signals(telem)
    result = {}

    def speed_section(prefix, start, end, threshold):
        section = mph.iloc[start:end]
        observed = dt.iloc[start:end]
        fully_supported = times.iloc[start:end].ge(times.iloc[start] + 1).to_numpy()
        smoothed = smooth_speed[start:end][fully_supported]
        result[prefix + '_max1s_mph'] = float(np.nanmax(smoothed)) if len(smoothed) else np.nan
        result[prefix + '_p90_mph'] = section.quantile(.9)
        result[prefix + '_above_threshold_s'] = observed.loc[section.ge(threshold)].sum()
        maximum = result[prefix + '_max1s_mph']
        result[prefix + '_shortfall_mph'] = max(0, threshold - maximum) if pd.notna(maximum) else np.nan

        if DEBUG: print(prefix, "rows", start, end, "max1s", maximum, "threshold", threshold, "shortfall", result[prefix + '_shortfall_mph'])
        # print(section.head(), smoothed[:10])  # raw vs 1s avg


    def motion_section(prefix, start, end):
        if not good.iloc[start:end].all(): return
        moving = velocity.iloc[start:end].gt(1)
        weights = dt.iloc[start:end].loc[moving]
        jerk = signals['jerk_mps3'].iloc[start:end].loc[moving]
        known = jerk.notna()

        result[prefix + '_jerk_rms'] = np.sqrt((jerk.pow(2) * weights).sum() / weights.loc[known].sum()) if known.any() else np.nan
        lowest_acc = signals['acc_mps2'].iloc[start:end].min()
        result[prefix + '_peak_decel'] = max(0, -lowest_acc) if pd.notna(lowest_acc) else np.nan
        result[prefix + '_peak_force'] = signals['force_n'].iloc[start:end].max()
        result[prefix + '_min_mph'] = mph.iloc[start:end].min()
        result[prefix + '_stop_s'] = dt.iloc[start:end].loc[~moving].sum()
        overlapping = brake_curves.loc[brake_curves.start_row.lt(end) & brake_curves.end_row.gt(start)]
        result[prefix + '_brake_count'] = len(overlapping)

        if DEBUG: print(prefix, "moving s", weights.sum(), "jerk rms", result[prefix + '_jerk_rms'], "brakes", len(overlapping))


    # SCHOOL: sign markers -> speed features before inside and after the zone
    for prefix in ['pre_zone', 'post_zone']:
        for suffix in ['max1s_mph', 'p90_mph', 'above_threshold_s', 'shortfall_mph']: result[prefix + '_' + suffix] = np.nan
    result['zone_entry_mph'] = np.nan
    zones = school_zone_windows(telem, dt)
    result['school_context_count'] = len(zones)
    if len(zones) == 1:
        start, end = zones[0]
        result['zone_entry_mph'] = mph.iloc[start]
        for a, b in complete_windows(telem['Speed Limit'].eq(40), dt):
            if b == start: speed_section('pre_zone', a, b, 40)
            if a == end: speed_section('post_zone', a, b, 35)

    # HUMPS: two full crests in encounter order
    humps = find_hump_candidates(telem)
    result['hump_candidate_count'] = len(humps)
    for number in [1, 2]:
        for suffix in ['crossing_mph', 'approach5m_mph', 'brake_onset_distance_m',
                       'approach_jerk_rms', 'approach_peak_decel', 'approach_peak_force',
                       'approach_min_mph', 'approach_stop_s', 'approach_brake_count']:
            result[f'hump{number}_{suffix}'] = np.nan
    result['humps_over15_count'] = result['humps_excess15_mean'] = np.nan
    if len(humps) == 2:
        result['humps_over15_count'] = int(humps.speed_mph.gt(15).sum())
        result['humps_excess15_mean'] = (humps.speed_mph - 15).clip(lower=0).mean()
        for number, row in enumerate(humps.itertuples(index=False), 1):
            peak = row.peak_row
            result[f'hump{number}_crossing_mph'] = row.speed_mph
            approach_start = int(distance.searchsorted(max(0, distance.iloc[peak] - APPROACH_M)))
            if not good.iloc[approach_start:peak].all(): continue
            five_before = int(distance.searchsorted(max(0, distance.iloc[peak] - 5)))
            result[f'hump{number}_approach5m_mph'] = mph.iloc[five_before]
            motion_section(f'hump{number}_approach', approach_start, peak)
            brakes = brake_curves.loc[brake_curves.start_row.between(approach_start, peak)]
            if len(brakes): result[f'hump{number}_brake_onset_distance_m'] = distance.iloc[peak] - distance.iloc[int(brakes.iloc[0].start_row)]

    # LANE CLOSURE: early lane choice and stops near cones
    lane_windows = lane_closure_windows(telem, dt)
    result['lane_context_count'] = len(lane_windows)
    for suffix in ['jerk_rms', 'peak_decel', 'peak_force', 'min_mph', 'stop_s', 'brake_count']:
        result['lane_' + suffix] = np.nan
        result['lane_approach_' + suffix] = np.nan
    result['lane_change_count'] = result['lane_last_change_lead_s'] = np.nan
    if len(lane_windows) == 1:
        start, end = lane_windows[0]
        motion_section('lane', start, end)
        approach = int(distance.searchsorted(max(0, distance.iloc[start] - APPROACH_M)))
        if good.iloc[approach:end].all():
            motion_section('lane_approach', approach, end)
            lanes = telem['Lane Number']
            change = lanes.ne(lanes.shift()) & (lanes * lanes.shift()).gt(0)
            changes = []
            for row in change.index[change & change.index.to_series().between(approach + 1, end - 1)]:
                future = times.le(times.iloc[row] + .5) & times.ge(times.iloc[row])
                if future.sum() and times.iloc[end - 1] >= times.iloc[row] + .5 and lanes.loc[future].eq(lanes.iloc[row]).all(): changes.append(row)
            result['lane_change_count'] = len(changes)
            if changes: result['lane_last_change_lead_s'] = times.iloc[start] - times.iloc[changes[-1]]
    if DEBUG: print("event counts / school", len(zones), "humps", len(humps), "lane", len(lane_windows))
    return result, humps, lane_windows


# HUMP: one matched crest -> crossing and approach features
def calc_hump_features(telem_df, curves_df, peak):
    t, v = telem_df['Sim Time'], telem_df.Velocity
    dt_srs = t.shift(-1)-t
    good = dt_srs.gt(0)&dt_srs.le(.1)
    distance = (v.clip(lower=0)*dt_srs).where(good,0).fillna(0).cumsum().shift(fill_value=0)
    a = int(distance.searchsorted(max(0, distance.iloc[peak]-100)))
    five = int(distance.searchsorted(max(0, distance.iloc[peak]-5)))
    feature_dict = dict(crossing_mph=float(v.iloc[peak]/.44704))

    if DEBUG: print("hump peak", peak, "approach rows", a, five, "crossing mph", feature_dict["crossing_mph"])
    if not good.iloc[a:peak].all(): return feature_dict
    signals_df = braking_signals(telem_df)
    moving = v.iloc[a:peak].gt(1)
    weights = dt_srs.iloc[a:peak].loc[moving]
    jerk = signals_df.jerk_mps3.iloc[a:peak].loc[moving]
    known = jerk.notna()

    braking = curves_df.loc[curves_df.start_row.lt(peak)&curves_df.end_row.gt(a)]
    onset = curves_df.loc[curves_df.start_row.between(a,peak)]
    lowest = signals_df.acc_mps2.iloc[a:peak].min()
    feature_dict.update(approach5m_mph=float(v.iloc[five]/.44704),
        approach_jerk_rms=np.sqrt((jerk.pow(2)*weights).sum()/weights.loc[known].sum()) if known.any() else np.nan,
        approach_peak_decel=max(0,-lowest) if pd.notna(lowest) else np.nan,
        approach_peak_force=signals_df.force_n.iloc[a:peak].max(),
        approach_min_mph=(v.iloc[a:peak]/.44704).min(),
        approach_stop_s=dt_srs.iloc[a:peak].loc[~moving].sum(),
        approach_brake_count=len(braking),
        brake_onset_distance_m=distance.iloc[peak]-distance.iloc[int(onset.iloc[0].start_row)] if len(onset) else np.nan)

    if DEBUG: print("hump approach", feature_dict)
    return feature_dict


KFC_COLS = ["Sim Time", "Heading", "Velocity", "Acceleration", "Brake Pedal Force", "Events", "Intersections", "X Position", "Y Position", "Z Position"]

# TURN: event marker and heading -> KFC entry turn rows
def locate_kfc_turn(telem_df):
    """Drive telem -> turn bounds dict + status; use marker/heading after nearby intersection"""
    time_srs=telem_df['Sim Time']; dt_srs=time_srs.shift(-1)-time_srs
    markers=complete_windows(telem_df.Events.eq(5),dt_srs)
    if len(markers)!=1:return None,'no_unique_complete_region'
    marker,region_end=markers[0]
    intersections=complete_windows(telem_df.Intersections.ne(0)&telem_df.Intersections.notna(),dt_srs)
    previous=[(a,b)for a,b in intersections if b<=marker]
    following=[(a,b)for a,b in intersections if a>=region_end]
    if not previous or not following or previous[-1]!=intersections[-2] or following[0]!=intersections[-1]:
        return None,'unresolved_intersection_context'
    marker_time=time_srs.iloc[marker]
    rows=telem_df.index[time_srs.between(marker_time-8,marker_time+8)&telem_df.index.to_series().ge(previous[-1][1])]
    if len(rows)<3:return None,'short_search'
    first,last=int(rows[0]),int(rows[-1])
    if not dt_srs.iloc[first:last].gt(0).all() or not dt_srs.iloc[first:last].le(.1).all():return None,'search_clock_gap'
    if telem_df.Heading.iloc[first:last+1].isna().any():return None,'missing_heading'
    heading_srs=pd.Series(np.rad2deg(np.unwrap(np.deg2rad(telem_df.Heading.iloc[first:last+1]))),index=rows)
    baseline_rows=rows[time_srs.loc[rows].between(marker_time-5,marker_time-2)]
    if len(baseline_rows)<3 or time_srs.loc[baseline_rows].iloc[-1]-time_srs.loc[baseline_rows].iloc[0]<1: return None,'short_heading_baseline'
    baseline=heading_srs.loc[baseline_rows].median()
    final=heading_srs.iloc[-60:].median()-baseline
    if not 60<=abs(final)<=120:return None,'not_single_quarter_turn'
    angle_srs=(heading_srs-baseline)*np.sign(final)
    ends=angle_srs.index[angle_srs.ge(80)]
    if not len(ends):return None,'missing_turn_end'
    turn_end=int(ends[0])
    # Do not mistake the preceding intersection's remaining heading change for entry onset
    starts=angle_srs.index[angle_srs.ge(10)&angle_srs.shift().lt(10)&(angle_srs.index<turn_end)]
    if not len(starts):return None,'missing_turn_start'
    turn_start=int(starts[-1])
    if not time_srs.iloc[turn_start]-1<=marker_time<=time_srs.iloc[turn_end]+1:return None,'turn_not_at_region_entry'
    if not .5<=time_srs.iloc[turn_end]-time_srs.iloc[turn_start]<=15:return None,'turn_duration_review'
    start=int(time_srs.searchsorted(time_srs.iloc[turn_start]-1)); end=turn_end+1
    if start==0 or end>=len(telem_df):return None,'truncated_window'
    turn_df=telem_df.iloc[start:end]
    if turn_df[KFC_COLS[:5]].isna().any().any() or not dt_srs.iloc[start:end].between(1e-8,.1).all():return None,'invalid_feature_window'
    return dict(start_row=start,end_row=end,turn_start_row=turn_start,turn_end_row=turn_end,
                start_s=time_srs.iloc[start],end_s=time_srs.iloc[turn_end],marker_s=marker_time,
                turn_start_s=time_srs.iloc[turn_start],heading_change_deg=final,
                marker_x=telem_df['X Position'].iloc[marker],marker_y=telem_df['Y Position'].iloc[marker]),'located_candidate'


# KFC: located entry turn -> acceleration and jerk

def calc_kfc_turn_features(telem_df):
    """Drive telem -> entry accel/jerk dict; no located turn -> NaNs"""
    feature_dict = {"turn_peak_accel_mps2": np.nan, "turn_jerk_rms_mps3": np.nan}
    bounds_dict, status = locate_kfc_turn(telem_df)
    if DEBUG: print("kfc turn", status, bounds_dict)
    if bounds_dict is None: return feature_dict

    start_row = bounds_dict["start_row"]
    end_row = bounds_dict["end_row"]
    signals_df = braking_signals(telem_df)
    dt_srs = (telem_df["Sim Time"].shift(-1) - telem_df["Sim Time"]).iloc[start_row:end_row]
    acc_srs = signals_df.acc_mps2.iloc[start_row:end_row]
    jerk_srs = signals_df.jerk_mps3.iloc[start_row:end_row]
    if acc_srs.notna().all() and jerk_srs.notna().all():
        feature_dict["turn_peak_accel_mps2"] = max(0., float(acc_srs.max()))
        feature_dict["turn_jerk_rms_mps3"] = np.sqrt((jerk_srs.pow(2) * dt_srs).sum() / dt_srs.sum())
    return feature_dict


# EVENT WINDOWS: shared detectors and measurements for turns stops and other events

# WINDOWS: keep full runs within one source clock

def complete_runs(telem, mask, min_s=0):
    t=telem['Sim Time'].to_numpy(float);dt=np.diff(t)
    result=[]
    for a,b in intervals(pd.Series(np.asarray(mask))):
        if a==0 or b>=len(t):continue
        if not ((dt[a-1:b]>0)&(dt[a-1:b]<=.1)).all():continue
        if t[b]-t[a]>=min_s:result.append((a,b))
    return result


# TURNS: intersection markers + heading -> turn rows

def junctions(telem):
    """Telem -> intersections DF; heading before/after tells left vs right vs straight"""
    rows=[];t=telem['Sim Time'].to_numpy(float)
    for a,b in complete_runs(telem,telem.Intersections.fillna(0).ne(0)):
        prior=telem.Heading.loc[telem['Sim Time'].between(t[a]-2,t[a]-1)]
        after=telem.Heading.loc[telem['Sim Time'].between(t[b]+1,t[b]+2)]
        if len(prior)<3 or len(after)<3:continue
        # use the nearby heading so 359 -> 0 does not look like a huge turn
        h0=float(prior.iloc[0]);p=h0+(prior-h0+180)%360-180;q=h0+(after-h0+180)%360-180
        angle=float((q.median()-p.median()+180)%360-180)
        rows.append(dict(start_row=a,end_row=b,start_s=t[a],end_s=t[b],turn_deg=angle,
            maneuver='left'if angle<=-45 else 'right'if angle>=45 else 'straight'if abs(angle)<=30 else 'ambiguous'))
    return pd.DataFrame(rows,columns=['start_row','end_row','start_s','end_s','turn_deg','maneuver'])


# BEHAVIOR: stops acceleration contacts and lane changes -> event rows

def detect(telem):
    """Telem -> stop/accel/contact/turn/lane change rows; one source clock only"""
    t=telem['Sim Time'].to_numpy(float);rows=[]
    masks=[('stop',telem.Velocity.abs().lt(.3),1.),
           ('acceleration',telem.Acceleration.gt(1),.5),
           ('contact',telem.Collision.fillna(0).gt(0),0.)]
    for name,mask,minimum in masks:
        for a,b in complete_runs(telem,mask,minimum):rows.append(dict(kind=name,start_row=a,end_row=b,start_s=t[a],end_s=t[b],value=t[b]-t[a]))

    js=junctions(telem)
    for r in js.itertuples(index=False):rows.append(dict(kind='junction_'+r.maneuver,start_row=r.start_row,end_row=r.end_row,start_s=r.start_s,end_s=r.end_s,value=r.turn_deg))

    lane=telem['Lane Number'].abs();dt=np.diff(t)
    changes=np.flatnonzero(lane.ne(lane.shift()).to_numpy()&(lane>0).to_numpy()&(lane.shift()>0).to_numpy())
    for a in changes:
        b=int(np.searchsorted(t,t[a]+1))
        if a==0 or b>=len(t)or not((dt[a-1:b]>0)&(dt[a-1:b]<=.1)).all():continue
        if lane.iloc[a:b+1].eq(lane.iloc[a]).all()and telem.Velocity.iloc[a:b].gt(1).all():
            rows.append(dict(kind='lane_change',start_row=int(a),end_row=b,start_s=t[a],end_s=t[b],value=float(lane.iloc[a]-lane.iloc[a-1])))
    return pd.DataFrame(rows,columns=['kind','start_row','end_row','start_s','end_s','value'])


# MEASURE: one event window -> feature dict and coverage status

def summarize(telem):
    """Event telem -> feature dict + status; skip short windows/gaps/missing motion"""
    if len(telem)<5:return {},'short_window'
    t=telem['Sim Time'].to_numpy(float);dt=np.diff(t)
    if not((dt>0)&(dt<=.1)).all():return {},'clock_gap'

    required=['Velocity','Heading','Acceleration','Brake Pedal Force']
    if telem[required].isna().any().any():return {},'missing_motion'

    v=telem.Velocity.to_numpy(float);heading=np.unwrap(np.deg2rad(telem.Heading.to_numpy(float)))
    yaw=np.diff(heading)/dt;signals=braking_signals(telem.reset_index(drop=True))
    stopped=abs(v[:-1])<.3; moving=abs(v[:-1])>1;turning=abs(yaw)>np.deg2rad(5)

    acceleration=telem.Acceleration.to_numpy(float)
    pedal=telem['Accelerator Pedal Position'].to_numpy(float)

    xy=telem[['X Position','Y Position']].to_numpy(float)
    distance=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    path=float(np.nansum(distance))

    # check heading wrap / yaw before turn features
    """
    motion_check_df = pd.DataFrame({"dt": dt, "speed": v[:-1], "yaw": yaw})
    motion_check_df["lateral_acc"] = motion_check_df.speed * motion_check_df.yaw
    print(motion_check_df.head(10))
    print("moving / turning rows", moving.sum(), turning.sum())
    print("path", path, "straight line", np.linalg.norm(xy[-1]-xy[0]))
    # TODO NEED TO look at yaw spikes near junction edges
    """

    result=dict(path_m=path,path_efficiency=float(np.linalg.norm(xy[-1]-xy[0])/path)if path>1 else np.nan,
        abs_turn_deg=float(np.rad2deg(np.abs(np.diff(heading))[moving].sum())),
        duration_s=float(t[-1]-t[0]),min_speed_mph=float(np.min(v)/.44704),
        speed_p90_mph=float(np.quantile(v,.9)/.44704),stop_s=float(dt[stopped].sum()),
        brake_on_fraction=float(np.average(telem['Brake Pedal Force'].to_numpy()[:-1]>5,weights=dt)),
        brake_jerk_rms=float(np.sqrt(np.nanmean(signals.jerk_mps3.to_numpy()**2))),
        peak_decel=float(max(0,-np.nanmin(signals.acc_mps2))),
        peak_accel=float(max(0,np.nanmax(signals.acc_mps2))),
        turning_accel_p90=float(np.quantile(acceleration[:-1][turning],.9))if turning.any()else 0.,
        turning_throttle_p90=float(np.quantile(pedal[:-1][turning],.9))if turning.any()else 0.,
        lateral_acc_p95=float(np.quantile(abs(v[:-1]*yaw),.95)),
        yaw_rate_p95_deg_s=float(np.rad2deg(np.quantile(abs(yaw),.95))),
        heading_change_deg=float(np.rad2deg(heading[-1]-heading[0])),
        lane_offset_p95=float(telem['Lane Offset'].abs().quantile(.95)),
        heading_error_p95=float(telem['Heading Error'].abs().quantile(.95)),
        contact=float(telem.Collision.gt(0).any()))

    if DEBUG:
        print("turn yaw p95", result['yaw_rate_p95_deg_s'], "lateral acc p95", result['lateral_acc_p95'])
        print("turn accel p90", result['turning_accel_p90'], "brake jerk rms", result['brake_jerk_rms'])


    curves=find_braking_curves(telem.reset_index(drop=True))
    result['brake_rise_time_median_s']=float(curves.rise_time_s.median())if len(curves)else np.nan
    result['brake_duration_p90_s']=float(curves.duration_s.quantile(.9))if len(curves)else np.nan
    result['brake_speed_loss_p90_mph']=float(curves.speed_loss_mps.quantile(.9)/.44704)if len(curves)else np.nan

    if {'CabIO-SigOut1','CabIO-SigOut2'}.issubset(telem):
        s1=telem['CabIO-SigOut1'].to_numpy()>0;s2=telem['CabIO-SigOut2'].to_numpy()>0
        result['signal1_active_fraction']=float(np.average(s1[:-1],weights=dt))
        result['signal2_active_fraction']=float(np.average(s2[:-1],weights=dt))
        turns=np.flatnonzero(turning&moving);signal=np.flatnonzero(s1|s2)
        result['signal_to_turn_lead_s']=float(t[turns[0]]-t[signal[0]])if len(turns)and len(signal)else np.nan
        result['signal_left_censored']=float(bool(s1[0]or s2[0]))

    events=detect(telem)
    for kind in ['stop','acceleration','contact','junction_left','junction_right','junction_straight','lane_change']:
        result[kind+'_count']=int(events.kind.eq(kind).sum())
        result[kind+'_per_km']=float(result[kind+'_count']/(path/1000))if path>10 else np.nan

    steering=telem['Steering Wheel Position'].to_numpy(float);delta=np.diff(steering)/dt
    result['steering_rate_abs_p95_native']=float(np.nanquantile(abs(delta),.95))
    sign=np.sign(delta);good=np.isfinite(delta)&(abs(delta)>.01)
    known=sign[good];result['steering_reversal_per_s']=float(np.sum(known[1:]!=known[:-1])/(t[-1]-t[0]))

    vertical_velocity_arr = telem['Vertical Velocity'].to_numpy(float)
    pitch_arr = telem['Pitch'].to_numpy(float)
    roll_arr = telem['Roll'].to_numpy(float)
    result['vertical_velocity_abs_p95'] = float(np.nanquantile(abs(vertical_velocity_arr), .95))
    result['pitch_span_native'] = float(np.nanquantile(pitch_arr, .95) - np.nanquantile(pitch_arr, .05))
    result['roll_span_native'] = float(np.nanquantile(roll_arr, .95) - np.nanquantile(roll_arr, .05))

    head=telem['Headway Time'].to_numpy(float);distance=telem['Headway Distance'].to_numpy(float)
    valid=moving&np.isfinite(head[:-1])&np.isfinite(distance[:-1])&(head[:-1]>.05)&(head[:-1]<=10)&(distance[:-1]>0)&(distance[:-1]<=200)
    result['lead_headway_observed_fraction']=float(dt[valid].sum()/dt[moving].sum())if moving.any()else np.nan
    result['lead_headway_p10_s']=float(np.quantile(head[:-1][valid],.1))if dt[valid].sum()>=1 else np.nan
    result['lead_headway_under1s_fraction']=float(dt[valid&(head[:-1]<1)].sum()/dt[valid].sum())if dt[valid].sum()>=1 else np.nan
    return result,'complete'


# SPEED LIMITS: limit change -> response timing

def speed_transitions(telem):
    """Telem -> speed limit response rows; speed response only, no signal phase/gaze"""
    t=telem['Sim Time'].to_numpy(float);limit=telem['Speed Limit'].to_numpy(float);mph=telem.Velocity.to_numpy(float)/.44704
    rows=[]
    for a in np.flatnonzero(np.r_[False,np.diff(limit)!=0]):
        if not(np.isfinite(limit[a-1])and np.isfinite(limit[a]))or min(limit[a-1],limit[a])<=0:continue
        b=min(len(t)-1,int(np.searchsorted(t,t[a]+10)))
        dt=np.diff(t[a:b+1])
        if not len(dt)or not((dt>0)&(dt<=.1)).all():continue
        down=limit[a]<limit[a-1];response=np.flatnonzero(mph[a:b+1]<=limit[a]if down else mph[a:b+1]>=limit[a])
        delay=float(t[a+response[0]]-t[a])if len(response)else np.nan
        rows.append(dict(start_row=int(a),start_s=t[a],old_limit=limit[a-1],new_limit=limit[a],direction='down'if down else'up',entry_error_mph=max(0,mph[a]-limit[a])if down else max(0,limit[a]-mph[a]),response_delay_s=delay,reached=int(bool(len(response)))))
    return pd.DataFrame(rows)




# EVENTS: source local rules -> feature values for each matched window

def calc_event_window_features(telem_df, scenario_id):
    """Telem + scenario -> feature dict + window mdata; route order gives guessed locations"""
    time_srs = telem_df["Sim Time"]
    dt_srs = time_srs.shift(-1) - time_srs
    part_df = telem_df
    feature_dict = {}
    seen = set()
    window_dicts = []
    event_windows=[]


    # MARKERS: locate roundabout collision lane and school windows
    roundabout_bounds,ra_status=roundabout(part_df);collision1_bounds,c1_status=collision_avoidance(part_df)
    window_dicts.append(dict(region='c1_marker',status=c1_status))
    if roundabout_bounds:event_windows.append(('roundabout',*roundabout_bounds,'sign_supported'))
    if collision1_bounds:event_windows.append(('collision1',*collision1_bounds,'marker_supported_scenarios1_2'))
    lane_bounds=lane_closure_windows(part_df,dt_srs)
    if len(lane_bounds)==1:event_windows.append(('lane',*lane_bounds[0],'sign_bracketed'))
    school_bounds=school_zone_windows(part_df,dt_srs)
    if len(school_bounds)==1:event_windows.append(('school',*school_bounds[0],'sign_limit_marker_supported'))
    # KFC: full path plus 10s either side of entry and exit
    kfc_bounds=intervals(part_df.Events.eq(5))
    if len(kfc_bounds)==1:
        a,b=kfc_bounds[0]
        if a>0 and b<len(part_df):
            event_windows.append(('kfc_path',a,b,'event5_supported'))
            for name,edge in [('kfc_entry',a),('kfc_exit',b)]:
                x=int(time_srs.searchsorted(time_srs.iloc[edge]-10));y=int(time_srs.searchsorted(time_srs.iloc[edge]+10))
                if x>0 and y<len(part_df):event_windows.append((name,x,y,'marker_edge_window'))
    # HUMPS: 5s either side of each crest
    humps=find_hump_candidates(part_df)
    if len(humps)==2:
        for i,peak in enumerate(humps.peak_row):
            x=int(time_srs.searchsorted(time_srs.iloc[peak]-5));y=int(time_srs.searchsorted(time_srs.iloc[peak]+5))
            if x>0 and y<len(part_df):event_windows.append((f'hump{i+1}',x,y,'two_crest_candidate'))
    # print("test marker windows", event_windows)
    junctions_df=junctions(part_df)


    # ROUTE: candidate encounters from turn order
    # route order is a location guess here, not a rating
    if roundabout_bounds and len(school_bounds)==1 and len(junctions_df):
        road_df=junctions_df.loc[junctions_df.start_s.gt(time_srs.iloc[roundabout_bounds[1]])&junctions_df.end_s.lt(time_srs.iloc[school_bounds[0][0]])]
        left_turns_df=road_df.loc[road_df.maneuver.eq('left')]
        wanted=2
        if len(left_turns_df)>=wanted:
            for i,r in enumerate(left_turns_df.iloc[:2].itertuples(index=False),1):
                x=int(time_srs.searchsorted(r.start_s-10));y=int(time_srs.searchsorted(r.end_s+3))
                if x>0 and y<len(part_df):event_windows.append((f'left{i}',x,y,'candidate_route_order'))
            following_junctions_df=road_df.loc[road_df.start_s.gt(left_turns_df.iloc[1].end_s)]
            stop_srs=following_junctions_df.iloc[0]if len(following_junctions_df)else None
            if scenario_id==3:
                straight_junctions_df=following_junctions_df
                if len(straight_junctions_df)>=2:
                    junction_srs=straight_junctions_df.iloc[1];x=int(time_srs.searchsorted(junction_srs.start_s-10));y=int(time_srs.searchsorted(junction_srs.end_s+3))
                    if x>0 and y<len(part_df):event_windows.append(('pedestrian',x,y,'candidate_route_order'))
            if stop_srs is not None:
                x=int(time_srs.searchsorted(stop_srs.start_s-10));y=int(time_srs.searchsorted(stop_srs.end_s+3))
                if x>0 and y<len(part_df):event_windows.append(('stop_junction',x,y,'candidate_route_order'))
                if scenario_id in [1,2]:
                    a=left_turns_df.iloc[1].end_row if scenario_id==1 else left_turns_df.iloc[0].end_row
                    b=int(stop_srs.start_row)if scenario_id==1 else int(left_turns_df.iloc[1].start_row)
                    event_windows.append(('collision2_route_proxy',int(a),int(b),'between_encounters_not_hazard_onset'))
                else:
                    event_windows.append(('collision1_route_proxy',int(left_turns_df.iloc[1].end_row),int(stop_srs.start_row),'between_encounters_not_hazard_onset'))
        # scenario 3 event 3 can overlap other markers, exclude school and lane 1st
        # event 3 meaning still unclear - video check
        if scenario_id==3:
            occupied=lane_bounds+school_bounds
            for a,b in intervals(part_df.Events.eq(3)):
                if a>0 and b<len(part_df)and not any(a<y and b>x for x,y in occupied):event_windows.append(('collision2_marker_candidate',a,b,'unvalidated_event3_context'))
    if DEBUG: print("event windows", scenario_id, event_windows)

    # row bounds can look fine while clock has gaps
    """
    windows_df = pd.DataFrame(event_windows, columns=["event", "start_row", "end_row", "rule"])
    print(windows_df)
    for window_row in windows_df.itertuples(index=False):
        window_telem_df = part_df.iloc[window_row.start_row:window_row.end_row]
        time_diffs_srs = window_telem_df["Sim Time"].diff().iloc[1:]
        print(window_row.event, "rows", len(window_telem_df), "max dt", time_diffs_srs.max())
        assert len(window_telem_df) == window_row.end_row - window_row.start_row
        # TODO NEED TO check source gaps before changing rule
    # overlaps: which ones actually matter?
    # kfc entry / full path overlap is expected
    """

    # MEASURE: located rows -> features, keep the 1st full window per event
    for name,a,b,status in event_windows:
        if name in seen:continue  # same event can show up twice in the candidate list
        summary_dict,quality=summarize(part_df.iloc[a:b].reset_index(drop=True))
        window_dicts.append(dict(region=name,status=status,quality=quality,start_s=float(time_srs.iloc[a]),end_s=float(time_srs.iloc[b-1])))
        if DEBUG:
            print("event", name, "rows", a, b, quality, "features", len(summary_dict))
            print("path m", summary_dict.get("path_m"), "efficiency", summary_dict.get("path_efficiency"), "jerk", summary_dict.get("brake_jerk_rms"))

        # check this window against the video
        """
        if name == "kfc_entry":
            event_telem_df = part_df.iloc[a:b]
            event_telem_df.plot(x="Sim Time", y=["Velocity", "Brake Pedal Force"])
            #split axes, speed vs brake force
            # TODO NEED TO get video offset
        """
        if summary_dict:
            feature_dict.update({name+'__'+k:v for k,v in summary_dict.items()});seen.add(name)
    return feature_dict, window_dicts
