"""Cleaned drives -> feat table + approved ratings"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

from src.representation.braking_features import find_braking_curves
from src.representation.drive_features import calc_basic_features, calc_drive_features, calc_joined_drive_features
from src.representation.event_features import calc_event_features, find_hump_candidates, calc_hump_features, calc_kfc_turn_features, calc_event_window_features
from src.representation.laneclosure_features import calc_aligned_lane_features

ROOT = Path(__file__).resolve().parents[2]
DEBUG = True


# VISIT: cleaned telemetry -> feature dict and event lookup rows

def calc_visit_features(dr_telem_df, scenario_id, anchors_df):
    """Visit telem + anchors -> feat dict + event mdata; keep 1st full event across restart fragments"""
    part_dfs = []
    if "source_fragment" in dr_telem_df:
        observed_df = dr_telem_df.loc[~dr_telem_df.is_reset_boundary.fillna(False)]
        for fragment, part_df in observed_df.groupby("source_fragment", sort=False): part_dfs.append(part_df.reset_index(drop=True))
    else:
        part_dfs.append(dr_telem_df.reset_index(drop=True))

    # reset rows removed; check fragments still add up
    """
    part_rows = 0
    for fragment, part_df in enumerate(part_dfs):
        dt_srs = part_df["Sim Time"].diff().iloc[1:]
        bad_time_df = part_df.loc[dt_srs.index[dt_srs.le(0) | dt_srs.gt(.1)]]
        print("part", fragment, "rows", len(part_df), "time gaps", len(bad_time_df))
        print(bad_time_df[["Sim Time", "Velocity"]].head())
        part_rows += len(part_df)
    expected_rows = len(observed_df) if "source_fragment" in dr_telem_df else len(dr_telem_df)
    assert part_rows == expected_rows
    """
    if DEBUG: print("visit fragments", len(part_dfs), "rows", [len(part_df) for part_df in part_dfs])

    curve_dfs = []
    drive_dicts = []

    for part_df in part_dfs:
        curves_df = find_braking_curves(part_df)
        curve_dfs.append(curves_df)
        drive_dicts.append(calc_drive_features(part_df, curves_df))


    # whole visit features keep actual motion from both attempts
    if len(part_dfs) == 1:
        feature_dict = calc_basic_features(part_dfs[0])
        feature_dict.update(drive_dicts[0])
    else:
        print("  using original + restart, matching event windows once", flush=True)
        feature_dict, unique_checks = calc_joined_drive_features(part_dfs, curve_dfs)

    # CONTEXTS: each source keeps its own clock and full event windows
    context_dicts = []
    for part_df, curves_df in zip(part_dfs, curve_dfs):
        context_dict, _, _ = calc_event_features(part_df, curves_df)
        context_dicts.append(context_dict)

    if len(part_dfs) == 1:
        feature_dict.update(context_dicts[0])
        lane_dict, lane_status = calc_aligned_lane_features(part_dfs[0], scenario_id, anchors_df)
        feature_dict.update(lane_dict)
        feature_dict.update(calc_kfc_turn_features(part_dfs[0]))
    else:
        # 1st full school section wins; partial restart cannot erase it
        context_blocks = [
            ["zone_entry_mph", "school_zone_entry_excess_mph"],
            ["pre_zone_max1s_mph", "pre_zone_p90_mph", "pre_zone_above_threshold_s", "pre_zone_shortfall_mph", "pre_school_zone_40mph_shortfall"],
            ["post_zone_max1s_mph", "post_zone_p90_mph", "post_zone_above_threshold_s", "post_zone_shortfall_mph", "school_zone_resume_35mph_shortfall"],
        ]
        for block_cols in context_blocks:
            for col in block_cols: feature_dict[col] = np.nan
            for drive_dict, context_dict in zip(drive_dicts, context_dicts):
                section_dict = drive_dict.copy()
                section_dict.update(context_dict)
                has_section = any(pd.notna(section_dict.get(col, np.nan)) for col in block_cols)
                if not has_section: continue
                for col in block_cols: feature_dict[col] = section_dict.get(col, np.nan)
                break

        # HUMP: keep hump identity by loc EVEN IF restart only sees 1
        anchor_srs = anchors_df.loc[scenario_id]
        hump_xy_arr = np.array([ [anchor_srs.hump1_x_m, anchor_srs.hump1_y_m], [anchor_srs.hump2_x_m, anchor_srs.hump2_y_m], ])
        seen_humps = set()
        feature_dict["hump1_crossing_mph"] = np.nan
        feature_dict["hump2_crossing_mph"] = np.nan
        for part_df, curves_df in zip(part_dfs, curve_dfs):
            peaks_df = find_hump_candidates(part_df)
            for peak_row in peaks_df.itertuples(index=False):
                diffs_arr = np.linalg.norm(hump_xy_arr - np.array([peak_row.x_m, peak_row.y_m]), axis=1)
                hump_number = int(diffs_arr.argmin()) + 1
                if diffs_arr[hump_number - 1] > 20 or hump_number in seen_humps: continue
                seen_humps.add(hump_number)
                hump_dict = calc_hump_features(part_df, curves_df, int(peak_row.peak_row))
                for col, value in hump_dict.items(): feature_dict[f"hump{hump_number}_{col}"] = value

        # LANE: use 1st full sign bracket; dont join its timing over a reset
        for part_df, context_dict in zip(part_dfs, context_dicts):
            if context_dict.get("lane_context_count") != 1: continue
            for col, value in context_dict.items():
                if col.startswith("lane_"): feature_dict[col] = value
            lane_dict, lane_status = calc_aligned_lane_features(part_df, scenario_id, anchors_df)
            feature_dict.update(lane_dict)
            break

        # KFC: full entry is in the restart for  joined visits
        for part_df in part_dfs:
            kfc_dict = calc_kfc_turn_features(part_df)
            if pd.notna(kfc_dict["turn_peak_accel_mps2"]):
                feature_dict.update(kfc_dict)
                break

    # EVENTS: 1st full obs per named event NOT one per fragment
    event_rows = []
    seen_regions = set()
    for fragment, part_df in enumerate(part_dfs):
        event_dict, coverage_dicts = calc_event_window_features(part_df, scenario_id)
        use_regions = set()
        for coverage_dict in coverage_dicts:
            region = coverage_dict["region"]
            if coverage_dict.get("quality") == "complete" and region not in seen_regions: use_regions.add(region)
            coverage_dict["fragment"] = fragment
            coverage_dict["used"] = region in use_regions
            event_rows.append(coverage_dict)
        for col, value in event_dict.items():
            if col.split("__")[0] in use_regions: feature_dict[col] = value
        seen_regions.update(use_regions)

    feature_dict["hump1_excess15_mph"] = np.maximum(0, feature_dict.get("hump1_crossing_mph", np.nan) - 15)
    feature_dict["pre_zone_shortfall40_mph"] = np.maximum(0, 40 - feature_dict.get("pre_zone_max1s_mph", np.nan))
    feature_dict["post_zone_shortfall35_mph"] = np.maximum(0, 35 - feature_dict.get("post_zone_max1s_mph", np.nan))
    if DEBUG: print("visit features", len(feature_dict), "event rows", len(event_rows))
    return feature_dict, event_rows


# BUILD: approved drive sources + matched labels -> model_data.csv

def extract_features():
    """Approved drive DFs + labels -> model DF/CSV; keep splits, rebuild features only"""
    settings_df = pd.read_csv(ROOT / "config/settings.csv", dtype=str, keep_default_na=False)
    settings_dict = settings_df.set_index("setting")["value"].to_dict()
    model_path = ROOT / settings_dict["model_data"]
    info_path = ROOT / settings_dict["model_data_info"]
    input_dir = model_path.parent

    # LOAD: approved labels define the visit order and participant assignments
    labels_df = pd.read_csv(input_dir / "model_labels.csv")
    sources_df = pd.read_csv(ROOT / "outputs/data/joined_visits/visit_sources.csv")
    drive_map_df = pd.read_csv(input_dir / "drive_mapping.csv")
    anchors_df = pd.read_csv(input_dir / "event_anchors.csv").set_index("scenario_id")
    feature_sets_df = pd.read_csv(ROOT / "config/feature_sets.csv", dtype=str, keep_default_na=False)
    scores_df = pd.read_csv(ROOT / "config/scores.csv")
    score_cols = scores_df["score"].tolist()
    meta_cols = ["recording_id", "participant_id", "visit", "split", "fold"]
    all_rating_cols = []
    for col in labels_df.columns:
        if col not in meta_cols and col not in score_cols: all_rating_cols.append(col)

    #no folder glob here, groundsim & retired drives never hit our roster
    visit_index_df = labels_df[meta_cols].merge(sources_df[["recording_id", "output_path"]], on="recording_id", how="left", validate="one_to_one",)
    visit_index_df = visit_index_df.merge(
        drive_map_df.loc[drive_map_df.label_status.eq("matched"), ["recording_id", "scenario_id"]],
        on="recording_id", how="left", validate="one_to_one",
    )
    assert visit_index_df.output_path.notna().all(), "MISSING: cleaned file for approved visit"
    feature_cols = []
    for feature_row in feature_sets_df.itertuples(index=False):
        for col in feature_row.features.splitlines():
            if col not in feature_cols: feature_cols.append(col)
    feature_cols = sorted(feature_cols)
    print("building inputs", len(visit_index_df), "visits", len(feature_cols), "features", flush=True)
    # print("test source rows", visit_index_df[["recording_id", "output_path"]].head())

    # EXTRACT: each selected visit -> feature dict
    feature_rows = []
    event_rows = []
    source_dicts = []
    known_feature_cols = set()
    for number, dr_row in enumerate(visit_index_df.itertuples(index=False), 1):
        print(f"features {number}/{len(visit_index_df)} {dr_row.recording_id}", flush=True)

        source_path = ROOT / dr_row.output_path
        dr_telem_df = pd.read_csv(source_path, low_memory=False)
        # print("test loaded telem", dr_telem_df.shape, source_path.name)

        feature_dict, visit_events = calc_visit_features(dr_telem_df, int(dr_row.scenario_id), anchors_df)
        # check one visits detected events
        """
        if dr_row.participant_id == 39:
            visit_events_df = pd.DataFrame(visit_events)
            print(dr_row.recording_id, visit_events_df)
            #id str or int? add visit filter too
            # TODO NEED TO show fragment before comparing restart times
        """
        known_feature_cols.update(feature_dict)
        row_dict = {"recording_id": dr_row.recording_id}
        for col in feature_cols: row_dict[col] = feature_dict.get(col, np.nan)
        feature_rows.append(row_dict)

        for event_dict in visit_events:
            event_dict["recording_id"] = dr_row.recording_id
            event_rows.append(event_dict)
        source_dicts.append({"recording_id": dr_row.recording_id, "path": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()})

    # source check while changing rules
    """
    try:
        print("last source", source_path, dr_telem_df.shape)
        print("missing here", [col for col in feature_cols if pd.isna(feature_dict.get(col))])
    except KeyError as err:
        print("test column name off", err)
        raise
    """
    unknown_cols = set(feature_cols) - known_feature_cols
    assert not unknown_cols, f"MISSING: no extractor for features {unknown_cols}"
    features_df = pd.DataFrame(feature_rows)


    print("extracted", features_df.shape, "joining ratings next", flush=True)
    # print("test feature gaps", features_df.isna().sum().sort_values().tail())

    # JOIN: feat vals + the same approved labels -> analysis input
    model_df = labels_df.merge(features_df, on="recording_id", validate="one_to_one")
    model_df = model_df[meta_cols + feature_cols + all_rating_cols + score_cols]
    # print("test joined ids", model_df[["recording_id", "participant_id", "split"]].head())
    model_df.to_csv(model_path, index=False)
    pd.DataFrame(event_rows).to_csv(input_dir / "feature_windows.csv", index=False)
    info_dict = {
        "rows": len(model_df),
        "training_validation_rows": int(model_df.split.ne("test").sum()),
        "test_rows": int(model_df.split.eq("test").sum()),
        "features": feature_cols,
        "ratings": all_rating_cols,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "feature_builder": "src/representation/extract_features.py",
        "labels_file": str((input_dir / "model_labels.csv").relative_to(ROOT)),
        "source_manifest": "outputs/data/joined_visits/visit_sources.csv",
        "event_anchors": str((input_dir / "event_anchors.csv").relative_to(ROOT)),
        "source_files": source_dicts,
        "note": "Features rebuilt from approved cleaned sources with matched labels and existing splits",
    }
    info_path.write_text(json.dumps(info_dict, indent=2))


    print("saved inputs", model_path, model_df.shape, flush=True)
    return model_df
