import pandas as pd

def match_labels(segment_index, clinician_frame, id_map):
    # id map columns: participant_id, record_id;
    mapping = id_map.copy() #TODO need Jason reply
    mapping["participant_id"] = mapping["participant_id"].astype(str).str.lstrip("0").replace("", "0")
    mapping["record_id"] = mapping["record_id"].astype(str)
    clinician = clinician_frame.copy()
    clinician["record_id"] = clinician["record_id"].astype(str)
    rows = []

    for example in segment_index.to_dict("records"):
        participant = str(example["participant_id"]).lstrip("0") or "0"
        mapped = mapping.loc[mapping["participant_id"] == participant]
        record_id = None
        record = None
        status = "unmapped_participant"
        if len(mapped) > 1:
            status = "ambiguous_id_map"
        elif len(mapped) == 1:
            record_id = mapped.iloc[0]["record_id"]
            if (mapping["record_id"] == record_id).sum() != 1:
                status = "ambiguous_id_map"
            else:
                event = f"visit_{int(example['visit'][1:])}_arm_1"
                matches = clinician.loc[(clinician["record_id"] == record_id)
                                       & (clinician["redcap_event_name"] == event)]
                status = "missing_assessment" if matches.empty else "ambiguous_assessment"
                if len(matches) == 1:
                    record = matches.iloc[0]
                    status = "matched"
                    complete_field = f"v{example['visit'][1:]}_driving_error_sheet_complete"
                    if complete_field not in record or str(record[complete_field]) not in {"2", "2.0"}:
                        status = "incomplete_assessment"
                    selector = example.get("scenario_selector_field")
                    if pd.notna(selector) and selector:
                        value = record.get(selector)
                        expected = example.get("scenario_selector_value")
                        if pd.isna(value) or pd.isna(expected) or str(value).removesuffix(".0") != str(expected).removesuffix(".0"):
                            status = "scenario_mismatch"

        # 1 row / item; occurrence, reason-for-no, all performance items
        for field in example["item_set_fields"].split("|"):
            row = {key: example[key] for key in
                   ["recording_id", "participant_id", "visit", "scenario_id", "segment_id", "segment_name"]}
            row.update({"record_id": record_id, "item_field": field,
                        "response": pd.NA, "label_status": status})
            if status == "matched":
                if field not in record:
                    row["label_status"] = "missing_item_field"
                elif pd.isna(record[field]) or record[field] == "":
                    row["label_status"] = "blank_response"
                else:
                    row["response"] = record[field]
            rows.append(row)
    return pd.DataFrame(rows)
