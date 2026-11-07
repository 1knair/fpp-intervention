from pathlib import Path
import pandas as pd
import numpy as np

from src.representation.segmentation import intervals
from src.representation.braking_features import BRAKE_FEATURE_COLS, find_braking_curves, summarize_braking


TELEM_COLS = ["Sim Time", "Velocity", "Acceleration", "Brake Pedal Force", "Speed Limit", "Speed Check", "Signs", "Events"]
FEATURE_COLS = [
    "speed_check_overspeed_count",
    "school_zone_entry_excess_mph",
    "pre_school_zone_40mph_shortfall",
    "school_zone_resume_35mph_shortfall",
    "speeding_episodes_per_km",
    *BRAKE_FEATURE_COLS,
]

# fixed feat defs, not picked using val scores
DEBUG = True

MAX_INTERVAL_S = 0.1
CHECK_TOLERANCE_MPH = 2
SPEEDING_EXCESS_MPH = 5
SPEEDING_MIN_S = 1
HARD_BRAKE_MPS2 = 0.4 * 9.80665
HARD_BRAKE_MIN_S = 0.5


# flag + forward time intervals -> full event windows (end row excluded)
def complete_windows(flag, dt, min_seconds=0):
    """Mask + timesteps -> full row windows; skip cut off edges/gaps, end row excluded"""
    windows = []
    valid_dt = dt.gt(0) & dt.le(MAX_INTERVAL_S)
    for start, end in intervals(flag):
        # need both event edges, skip cut off windows/clock gaps
        if start == 0 or end == len(flag) or not valid_dt.iloc[start - 1:end].all(): continue
        if dt.iloc[start:end].sum() >= min_seconds - 1e-9: windows.append((start, end))
    return windows


# telem + time intervals -> school zones supported by Jacob's marker pattern
def school_zone_windows(telem, dt):
    # Jacob's pattern: sign 20 + 40 -> 20 -> 40 + matching event edges (within 3 s) TODO check
    # event code changes by scenario: 3 on drives 1/2, 4 on drive 3
    """Telem + timesteps -> school windows; sign/limit/event markers need to agree"""
    times, limits = telem["Sim Time"], telem["Speed Limit"]
    markers = [window for code in telem["Events"].dropna().unique() if code != 0
               for window in complete_windows(telem["Events"].eq(code), dt)]
    sign_times = times.loc[telem["Signs"].eq(20)]
    zones = []
    for start, end in complete_windows(limits.eq(20), dt):
        if limits.iloc[start - 1] != 40 or limits.iloc[end] != 40: continue
        sign_matches = (sign_times - times.iloc[start]).abs().le(3).any()
        event_matches = sum(
            abs(times.iloc[a] - times.iloc[start]) <= 3
            and abs(times.iloc[b] - times.iloc[end]) <= 3
            for a, b in markers
        )
        if sign_matches and event_matches == 1: zones.append((start, end))
    return zones


# whole-drive telemetry -> context/braking features + exposure/count mdata
# FEATURE_COLS go into X, other cols show what we could measure
def calc_drive_features(dr_telem, brake_curves=None):
    # loader passes curves so we dont detect them twice
    """Drive telem + brake curves -> feature dict; missing window stays NaN, not 0"""
    if brake_curves is None: brake_curves = find_braking_curves(dr_telem)
    telem = dr_telem[TELEM_COLS].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    speed_mph = telem["Velocity"] / 0.44704
    limits = telem["Speed Limit"]
    dt = telem["Sim Time"].shift(-1) - telem["Sim Time"]
    valid_dt = dt.gt(0) & dt.le(MAX_INTERVAL_S)
    moving = telem["Velocity"].gt(1)
    excess_mph = speed_mph - limits

    # SPEED-CHECK EVENTS: one marker window counts once, not once per sample
    checks = [
        (start, end) for start, end in complete_windows(telem["Speed Check"].eq(1), dt)
        if (limits.gt(0) & speed_mph.notna()).iloc[start:end].all()
    ]
    overspeed_checks = sum(excess_mph.iloc[start:end].median() > CHECK_TOLERANCE_MPH for start, end in checks)

    # SCHOOL-ZONE CONTEXT: reuse marker pattern, not every 20mph road
    # 40mph section before zone stays a guess for Four Lane location
    zones = school_zone_windows(telem, dt)
    forty_windows = complete_windows(limits.eq(40), dt)
    entry_excess, before_zone, after_zone = [], [], []
    for zone_start, zone_end in zones:
        if pd.notna(speed_mph.iloc[zone_start]): entry_excess.append(max(0, speed_mph.iloc[zone_start] - 20))
        for start, end in forty_windows:
            section_speed = speed_mph.iloc[start:end].loc[moving.iloc[start:end]]
            if section_speed.empty: continue
            if end == zone_start: before_zone.append(max(0, 40 - section_speed.quantile(0.9)))
            if start == zone_end: after_zone.append(max(0, 35 - section_speed.quantile(0.9)))

    # SUSTAINED EPISODES: tell repeated events apart from time / sample counts
    speed_exposure = valid_dt & moving & limits.gt(0)
    brake_exposure = valid_dt & moving & telem["Acceleration"].notna()
    speed_km = (telem["Velocity"] * dt).loc[speed_exposure].sum() / 1000
    brake_km = (telem["Velocity"] * dt).loc[brake_exposure].sum() / 1000
    speeding_mask = moving & limits.gt(0) & excess_mph.gt(SPEEDING_EXCESS_MPH)
    hard_braking_mask = moving & telem["Acceleration"].le(-HARD_BRAKE_MPS2)

    speeding = complete_windows(speeding_mask, dt, SPEEDING_MIN_S)
    hard_braking = complete_windows(hard_braking_mask, dt, HARD_BRAKE_MIN_S)

    marker_codes = set()
    for start, end in zones:
        zone_events_srs = telem["Events"].iloc[start:end]
        for code in zone_events_srs.dropna().unique():
            if code != 0: marker_codes.add(code)
    marker_codes_text = "|".join(str(code) for code in sorted(marker_codes))


    if DEBUG:
        print("speed checks", len(checks), "overspeed", overspeed_checks, "school windows", len(zones))
        print("episode counts speed/brake", len(speeding), len(hard_braking), "km", speed_km, brake_km)
    # print("school entry / before / after", entry_excess, before_zone, after_zone)

    # OUTPUT: window not found -> blank / window found but no event -> 0
    return {
        **summarize_braking(brake_curves, dr_telem),
        "speed_check_overspeed_count": int(overspeed_checks) if checks else float("nan"),
        "school_zone_entry_excess_mph": pd.Series(entry_excess, dtype=float).mean(),
        "pre_school_zone_40mph_shortfall": pd.Series(before_zone, dtype=float).mean(),
        "school_zone_resume_35mph_shortfall": pd.Series(after_zone, dtype=float).mean(),
        "speeding_episodes_per_km": len(speeding) / speed_km if speed_km else float("nan"),
        "hard_brake_episodes_per_km": len(hard_braking) / brake_km if brake_km else float("nan"),
        "speed_check_opportunities": len(checks),
        "school_zone_opportunities": len(entry_excess),
        "school_zone_marker_codes": marker_codes_text,
        "pre_school_zone_opportunities": len(before_zone),
        "post_school_zone_opportunities": len(after_zone),
        "speeding_episode_count": len(speeding),
        "hard_brake_episode_count": len(hard_braking),
        "speed_exposure_km": speed_km,
        "brake_exposure_km": brake_km,
    }


# cleaned drive folder + output CSV -> feature table, one row per recording
def build_drive_features(cleaned_dir, out_path):
    dr_files = sorted(Path(cleaned_dir).glob("*_cleaned.csv"))
    if not dr_files: raise FileNotFoundError(f"MISSING: cleaned drives in {cleaned_dir}")
    rows = []
    all_curves = []
    for dr_file in dr_files:
        _, participant_id, visit, visit_date, _, drive_number, _ = dr_file.stem.split("_")
        dr_telem = pd.read_csv(dr_file, usecols=TELEM_COLS)
        if dr_telem.empty: raise ValueError(f"empty cleaned drive: {dr_file}")

        row = {
            "recording_id": dr_file.stem,
            "participant_id": participant_id,
            "visit": visit,
            "visit_date": visit_date,
            "drive_number": int(drive_number),
            "row_count": len(dr_telem),
        }

        # drive selection info, not model features
        dt = dr_telem["Sim Time"].shift(-1) - dr_telem["Sim Time"]
        moving = dr_telem["Velocity"].gt(1.0) & dt.gt(0) & dt.le(0.1)
        row["moving_seconds"] = dt.loc[moving].sum()
        brake_curves = find_braking_curves(dr_telem)
        row.update(calc_drive_features(dr_telem, brake_curves))
        all_curves.append(brake_curves.assign(recording_id=dr_file.stem, participant_id=participant_id, visit=visit))
        rows.append(row)

    dr_features = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dr_features.to_csv(out_path, index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(out_path.parent / "braking_episodes.csv", index=False)
    return dr_features




BASIC_TELEM_COLS = ["Sim Time", "Velocity", "Lane Offset", "Acceleration", "Speed Limit"]

# BASIC: cleaned drive -> whole drive summaries
def calc_basic_features(dr_telem_df):
    telem_df = dr_telem_df[BASIC_TELEM_COLS].replace([float("inf"), -float("inf")], float("nan"))
    # speed is m/s; posted limit is mph; compare only after conversion
    speed_mph = telem_df["Velocity"] / 0.44704
    dt_srs = telem_df["Sim Time"].shift(-1) - telem_df["Sim Time"]
    #removed rows leave gaps, dont count as driving time
    valid_mask = (
        dt_srs.gt(0) & dt_srs.le(0.1)
        & telem_df["Velocity"].gt(0)
        & telem_df["Speed Limit"].gt(0)
    )
    observed_seconds = dt_srs.loc[valid_mask].sum()
    speeding_seconds = dt_srs.loc[valid_mask & speed_mph.gt(telem_df["Speed Limit"])].sum()
    return {
        "velocity_sd": telem_df["Velocity"].std(ddof=1),
        "lane_offset_abs_mean": telem_df["Lane Offset"].abs().mean(),
        "acc_sd": telem_df["Acceleration"].std(ddof=1),
        "speeding_fraction": speeding_seconds / observed_seconds if observed_seconds else float("nan"),
    }

# JOIN: original and restart -> whole visit features without reset diffs
def calc_joined_drive_features(part_dfs, brake_dfs):
    """Fragment DFs -> feature dict + check count; use both attempts, skip reset gaps/count checks once"""
    motion_dfs, pooled_curve_dfs, part_feature_dicts = [], [], []
    clock, offset = 0., 0

    for telem_df, curves_df in zip(part_dfs, brake_dfs):
        part_feature_dicts.append(calc_drive_features(telem_df,curves_df))
        motion_df = telem_df.copy()
        motion_df['Sim Time'] = telem_df['Sim Time']-telem_df['Sim Time'].iloc[0]+clock

        adjusted_curves_df = curves_df.copy()
        adjusted_curves_df[['start_row','end_row']] += offset
        pooled_curve_dfs.append(adjusted_curves_df)
        motion_dfs.append(motion_df)
        offset += len(motion_df)
        clock = float(motion_df['Sim Time'].iloc[-1])+1.

    whole_df = pd.concat(motion_dfs,ignore_index=True)
    pooled_curves_df = pd.concat(pooled_curve_dfs,ignore_index=True)

    feature_dict = calc_basic_features(whole_df)
    feature_dict.update(calc_drive_features(whole_df,pooled_curves_df))

    # TOTALS: counts / distance across both fragments
    speed_km = 0
    brake_km = 0
    speeding_count = 0
    hard_brake_count = 0


    for part_dict in part_feature_dicts:
        speed_km += part_dict['speed_exposure_km']
        brake_km += part_dict['brake_exposure_km']
        speeding_count += part_dict['speeding_episode_count']
        hard_brake_count += part_dict['hard_brake_episode_count']


    feature_dict['speeding_episodes_per_km'] = speeding_count / speed_km if speed_km else np.nan
    feature_dict['hard_brake_episodes_per_km'] = hard_brake_count / brake_km if brake_km else np.nan


    # restart repeats: keep 1st full check with same
    # position / heading / limit, keep separate brake actions though
    checkpoints = []
    for source, telem_df in enumerate(part_dfs):
        dt = telem_df['Sim Time'].shift(-1)-telem_df['Sim Time']
        for a,b in complete_windows(telem_df['Speed Check'].eq(1),dt):
            seg_df = telem_df.iloc[a:b]
            if not (seg_df['Speed Limit'].gt(0)&seg_df.Velocity.notna()).all(): continue
            x,y = seg_df[['X Position','Y Position']].median()
            heading = float(seg_df.Heading.iloc[0])
            limit = float(seg_df['Speed Limit'].median())

            duplicate = False
            for checkpoint_dict in checkpoints:
                distance_m = np.hypot(x - checkpoint_dict['x'], y - checkpoint_dict['y'])
                heading_diff = abs((heading - checkpoint_dict['heading'] + 180) % 360 - 180)
                same_limit = limit == checkpoint_dict['limit']
                if distance_m <= 10 and heading_diff <= 15 and same_limit:
                    duplicate = True
                    break
            if duplicate: continue

            excess_mph_srs = seg_df.Velocity / .44704 - seg_df['Speed Limit']
            error = float(excess_mph_srs.median() > 2)
            checkpoint_dict = dict(x=x, y=y, heading=heading, limit=limit, source=source, error=error)
            checkpoints.append(checkpoint_dict)
    feature_dict['speed_check_overspeed_count'] = sum(c['error'] for c in checkpoints) if checkpoints else np.nan
    return feature_dict, len(checkpoints)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract interpretable whole-drive features")
    parser.add_argument("--cleaned-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build_drive_features(args.cleaned_dir, args.output)


    print(f"saved features for {len(result)} drives to {args.output}")
