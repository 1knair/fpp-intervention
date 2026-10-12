from pathlib import Path
import pandas as pd
from src.representation.segmentation import segment_recording # JUST RULES FOR GETTING START_ROW END_ROW

def calc_stats(frame):
    return {
        "velocity_mean": frame["Velocity"].mean(),
        "velocity_sd": frame["Velocity"].std(ddof=1),

        "lane_offset_mean": frame["Lane Offset"].mean(),
        "lane_offset_sd": frame["Lane Offset"].std(ddof=1),
        "lane_offset_abs_mean": frame["Lane Offset"].abs().mean(),

        "acc_mean": frame["Acceleration"].mean(),
        "acc_sd": frame["Acceleration"].std(ddof=1),
    }


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT/"outputs"/"cleaned"
DATA_DIR = ROOT/"data"
OUT_DIR = ROOT / "outputs"


def build_feature_table(in_dir: Path, out_path: Path, representation: str) -> pd.DataFrame:
    files = sorted(in_dir.glob("*_cleaned.csv")) # all cleaned recordings
    if not files: raise FileNotFoundError(f"NO RECORDINGS \n {in_dir}")
    seg_map_frame = pd.read_csv(DATA_DIR / "mappings"/"segment_item_mapping.csv")

    rows = []
    segment_indexes = []
    for file in files:
        telemetry_frame = pd.read_csv(file, dtype={"participant_id": str})
        if telemetry_frame.empty: raise ValueError(f"Cleaned recording is empty: {file}")

        # filename -> drive metadata; participant_002_T1_2025-04-21_drive_2_cleaned.csv
        _, participant_id, visit, visit_date, _, scenario_id, _ = file.stem.split("_")
        scenario_id = int(scenario_id)
        drive_metadata = {
            "recording_id": file.stem,
            "participant_id": participant_id,
            "visit": visit,
            "visit_date": visit_date,
            "drive_number": scenario_id,
        }

        # in_frame -> drive feats; drive feats + drive metadata -> row (1/drive)
        if representation == "drive_stats":
            row = calc_stats(telemetry_frame)
            row.update(drive_metadata)
            rows.append(row)

        # get each segment's telemetry based on segment mapping and drive scenario
        else:
            selected = seg_map_frame.loc[
                (seg_map_frame["scenario_id"] == scenario_id) & (seg_map_frame["visit"] == visit)
                ]

            # apply rule
            selected = segment_recording(telemetry_frame, selected, drive_metadata)
            segment_indexes.append(selected)
            selected = selected.loc[selected["segmentation_status"] == "candidate"] if not selected.empty else selected

            # for each row (segment mapping) in the selected scenario's form
                # copy the in_frame's data using the provided start/end
            for seg_metadata in selected.itertuples(index=False):
                start_row, end_row = int(seg_metadata.start_row), int(seg_metadata.end_row)
                segment_frame = telemetry_frame.iloc[start_row:end_row]

                # since training example is now per segment we need the name/type (which model?) and id (which y?)
                segment_metadata = {
                    "segment_id": seg_metadata.segment_id,
                    "segment_name": seg_metadata.segment_name,
                }

                if representation == "seg_stats":
                    row = calc_stats(segment_frame)
                    row.update(drive_metadata)
                    row.update(segment_metadata)
                    rows.append(row)

                elif representation == "seg_telemetry": # TODO helper/optimize
                    for row in segment_frame.to_dict("records"):
                        row.update(drive_metadata)
                        row.update(segment_metadata)
                        rows.append(row)

    out_frame = pd.DataFrame(rows)
    output_file = out_path / "features" / f"{representation}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(output_file, index=False)

    '''
    if segment_indexes:
        pd.concat(segment_indexes, ignore_index=True).to_csv(output_file.parent / "segment_index.csv", index=False)
    '''
    return out_frame


if __name__ == "__main__":
    build_feature_table(IN_DIR, OUT_DIR, "drive_stats")
