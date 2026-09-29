from pathlib import Path
import pandas as pd
from src.representation.features import calc_drive

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT/"outputs"/"cleaned"
OUT_PATH = ROOT/"outputs"/"features"/"drive_features.csv"

def build_feature_table(in_dir: Path, out_path: Path, representation: str) -> pd.DataFrame:
    files = sorted(in_dir.glob("*_cleaned.csv")) # for now, all cleaning outputs
    if not files: raise FileNotFoundError(f"NO RECORDINGS \n {in_dir}")
    
    rows = []
    for file in files:
        # file -> in_frame
        in_frame = pd.read_csv(file, dtype={"participant_id": str})
        if in_frame.empty: raise ValueError(f"Cleaned recording is empty: {file}")

        # filename -> drive metadata
        # participant_002_T1_2025-04-21_drive_2_cleaned.csv
        _, participant_id, visit, visit_date, _, drive_num, _ = file.stem.split("_")
        metadata = {
            "recording_id": file.stem,
            "participant_id": participant_id,
            "visit": visit,
            "visit_date": visit_date,
            "drive_number": int(drive_num),
        }

        # in_frame -> drive feats; drive feats + drive metadata -> row (1/drive) 
        if representation == "drive":
            row = calc_drive(in_frame)
            row.update(metadata)
            rows.append(row)

        # drive metadata + segment metadata + segment features -> row (1/segment/drive)
        elif representation == "segment":
            raise NotImplementedError("TODO")

    features = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    return features

if __name__ == "__main__":
    build_feature_table(IN_DIR, OUT_PATH, "drive")
    
