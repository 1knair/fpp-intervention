"""Compare recorded ratings and scores across T1, T2 and T3"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from scipy import stats
from statsmodels.stats.anova import AnovaRM

ROOT = Path(__file__).resolve().parents[2]


# COMPARE VISITS: ratings_df -> statistics dict + visits_df

def compare_visits(ratings_df, column, permutations, seed):
    # ARRANGE: one row per person, one column per visit
    """Ratings DF + col -> stats dict/visits DF; same people with T1 T2 T3 values"""
    visits_df = ratings_df.pivot(index="participant_id", columns="visit", values=column)
    visits_df = visits_df[["T1", "T2", "T3"]].dropna()  # same people across all three visits
    # print("test visits", column, visits_df.shape)
    ratings_arr = visits_df.to_numpy()

    ratings_long_df = visits_df.reset_index().melt(id_vars="participant_id", var_name="visit", value_name="score")

    # FIT: ratings -> ANOVA stats
    anova_model = AnovaRM(ratings_long_df, "score", "participant_id", within=["visit"])
    anova_result = anova_model.fit()
    anova_df = anova_result.anova_table
    anova_srs = anova_df.loc["visit"]

    # CORRECT: adjust ANOVA for unequal variances of visit differences
    n_people, n_visits = ratings_arr.shape
    centering_arr = np.eye(n_visits) - np.ones((n_visits, n_visits)) / n_visits
    covariance_arr = centering_arr @ np.cov(ratings_arr, rowvar=False) @ centering_arr
    epsilon = np.trace(covariance_arr) ** 2 / ((n_visits - 1) * np.trace(covariance_arr @ covariance_arr))
    epsilon = float(np.clip(epsilon, 1 / (n_visits - 1), 1))
    df1, df2 = float(anova_srs["Num DF"]), float(anova_srs["Den DF"])
    f = float(anova_srs["F Value"])

    # CHECK: compare ranks after shuffling visit order within each person
    ranks_arr = stats.rankdata(ratings_arr, axis=1)
    observed = np.sum((ranks_arr.sum(axis=0) - n_people * (n_visits + 1) / 2) ** 2)
    rng = np.random.default_rng(seed)
    order_arr = rng.random((permutations, n_people, n_visits)).argsort(axis=2)
    permuted_arr = np.take_along_axis(np.broadcast_to(ranks_arr, order_arr.shape), order_arr, axis=2)
    stats_arr = ((permuted_arr.sum(axis=1) - n_people * (n_visits + 1) / 2) ** 2).sum(axis=1)
    p_rank = (1 + np.count_nonzero(stats_arr >= observed - 1e-10)) / (permutations + 1)
    friedman_result = stats.friedmanchisquare(*ratings_arr.T)

    stats_dict = {
        "score": column,
        "participants": n_people,
        "F": f,
        "df1": df1,
        "df2": df2,
        "p_anova": float(anova_srs["Pr > F"]),
        "epsilon_GG": epsilon,
        "p_anova_GG": float(stats.f.sf(f, df1 * epsilon, df2 * epsilon)),
        "friedman_statistic": float(friedman_result.statistic),
        "p_friedman_permutation": float(p_rank),
    }
    return stats_dict, visits_df


# MATCH: assessment question -> shared safety-rating name

def safety_slot(seg_name, question_text):
    if "pedestrian" in question_text: return "pedestrian"
    if "cyclists" in question_text: return "cyclist"
    slot_dict = {
        "roundabout": "roundabout",
        "collision avoidance": "collision1",
        "collisions avoid 2": "collision2",
        "enter kfc": "kfc_entry",
        "enter and exit drivethru": "kfc_path",
        "exit kfc": "kfc_exit",
    }
    if seg_name in slot_dict: return slot_dict[seg_name]
    if "unprotected left 1" in seg_name: return "left1"
    if "unprotected left 2" in seg_name: return "left2"
    if "straight intersection" in seg_name: return "straight_stop"
    if seg_name == "lane closure":
        if "before" in question_text: return "lane_change"
        return "lane_flow"
    raise ValueError(f"rating not mapped: {seg_name} / {question_text}")


# NAME: scenario-specific question -> shared question name

def get_rating_name(field_row):
    seg_name = field_row.segment.lower()
    question_text = field_row.question_key.split(" | ")[1]
    if field_row.kind == "safety_severity": return "safety_" + safety_slot(seg_name, question_text)
    if "unprotected left 1" in seg_name: seg_name = "left1"
    elif "unprotected left 2" in seg_name: seg_name = "left2"
    elif "straight intersection" in seg_name: seg_name = "stop intersection"
    return seg_name + " | " + question_text


# LOAD RATINGS: drive mapping + assessment -> scores_df + label_audit_df

def load_scores(assessment_file):
    """Assessment CSV -> ratings/scores DF + field mdata; keep existing/custom scores separate"""
    dr_map_df = pd.read_csv(ROOT / "outputs/data/model_inputs/drive_mapping.csv")
    dr_map_df = dr_map_df.loc[dr_map_df.label_status.eq("matched")]
    field_map_df = pd.read_csv(ROOT / "outputs/data/model_inputs/question_labels.csv")
    field_map_df = field_map_df.drop_duplicates(["visit", "scenario_id", "field"])
    cl_df = pd.read_csv(assessment_file).set_index(["record_id", "redcap_event_name"], verify_integrity=True)

    # MATCH: assessment fields for each drive
    score_dicts = []
    audit_dicts = []
    for dr_row in dr_map_df.itertuples(index=False):
        cl_srs = cl_df.loc[(dr_row.record_id, f"visit_{dr_row.visit[1:]}_arm_1")]  # T2 -> visit_2_arm_1
        cl_fields_df = field_map_df.loc[field_map_df.visit.eq(dr_row.visit) & field_map_df.scenario_id.eq(dr_row.scenario_id)]
        assert len(cl_fields_df) == 35
        score_dict = {
            "recording_id": dr_row.recording_id,
            "participant_id": dr_row.participant_id,
            "visit": dr_row.visit,
            "scenario_id": dr_row.scenario_id,
        }
        for field_row in cl_fields_df.itertuples(index=False):
            rating_name = get_rating_name(field_row)
            assert rating_name not in score_dict
            raw_rating = pd.to_numeric(cl_srs[field_row.field], errors="coerce")
            # CONVERT: recorded answer -> error or safety grade
            if pd.isna(raw_rating):
                # missing answer
                rating = np.nan
            elif field_row.kind == "safety_severity":
                # keep the 0 to 2 grade
                assert raw_rating in [0, 1, 2]
                rating = raw_rating
            elif field_row.kind == "no_or_cued_error":
                # no or needed a cue -> one error
                assert raw_rating in [0, 1, 2]
                rating = float(raw_rating in [0, 2])
            else:
                # yes -> no error
                assert raw_rating in [0, 1]
                rating = 1 - raw_rating
            score_dict[rating_name] = rating

            audit_dict = {
                "participant_id": dr_row.participant_id,
                "visit": dr_row.visit,
                "scenario_id": dr_row.scenario_id,
                "target": rating_name,
                "field": field_row.field,
                "kind": field_row.kind,
                "raw": raw_rating,
                "score": rating,
            }
            audit_dicts.append(audit_dict)

        for domain in ["attn", "wm", "rt", "ef"]:
            score_dict[f"official_{domain}_errors"] = pd.to_numeric(cl_srs[f"v{dr_row.visit[1:]}_{domain}_err_total"], errors="coerce")
        score_dict["official_total_errors"] = pd.to_numeric(cl_srs[dr_row.score_field], errors="coerce")
        score_dicts.append(score_dict)

    # BUILD: rating dicts -> visit rows + custom scores
    scores_df = pd.DataFrame(score_dicts)
    speed3_cols = ["speed hump | slows to less than or equal to 15mph to go over speed hump",
              "four lane | speeds up to at least 40 mph",
              "school zone | resume previous speed at end school zone"]
    speed5_cols = speed3_cols + ["speed hump 2 | slows to less than or equal to 15mph to go over speed hump",
                      "school zone | adjusting speed down to less than or equal to 20 mph at school speed limit sign"]
    scores_df["speed3_errors"] = scores_df[speed3_cols].sum(axis=1, min_count=3)
    scores_df["speed5_errors"] = scores_df[speed5_cols].sum(axis=1, min_count=5)
    safety_cols = []
    for col in scores_df.columns:
        if col.startswith("safety_"): safety_cols.append(col)
    assert len(safety_cols) == 13
    scores_df["safety13_grade_sum"] = scores_df[safety_cols].sum(axis=1, min_count=13)
    # missing answers stay missing, not zero
    scores_df["attention_without_speed5"] = scores_df.official_attn_errors - scores_df.speed5_errors
    category_cols = ["official_attn_errors", "official_wm_errors", "official_rt_errors", "official_ef_errors"]
    scores_df["performance_subscores_sum"] = scores_df[category_cols].sum(axis=1, min_count=4)
    label_audit_df = pd.DataFrame(audit_dicts)
    return scores_df, label_audit_df


# RUN ANOVA: scores_df -> visit means and p-values

def run_anova(assessment_file, permutations, seed, results_dir, model_data_file):
    scores_df, label_audit_df = load_scores(assessment_file)
    meta_cols = ["recording_id", "participant_id", "visit", "scenario_id"]
    target_cols = []
    for col in scores_df.columns:
        if col not in meta_cols: target_cols.append(col)
    visit_count_srs = scores_df.groupby("participant_id").visit.nunique()
    complete_ids = visit_count_srs.loc[visit_count_srs.eq(3)].index.tolist()
    complete_df = scores_df.loc[scores_df.participant_id.isin(complete_ids)].copy()

    # COHORTS: full ratings / full telemetry visits
    model_data_df = pd.read_csv(model_data_file)
    telem_visit_count_srs = model_data_df.groupby("participant_id").visit.nunique()
    complete_telem_ids = telem_visit_count_srs.loc[telem_visit_count_srs.eq(3)].index.tolist()
    result_dicts = []
    complete_telem_df = complete_df.loc[complete_df.participant_id.isin(complete_telem_ids)]
    cohorts_dict = {
        "all_complete_clinician": complete_df,
        "same27_with_telemetry": complete_telem_df,
    }

    # TEST: ratings/scores across visits
    for cohort, cohort_df in cohorts_dict.items():
        for target in target_cols:
            visits_df = cohort_df.pivot(index="participant_id", columns="visit", values=target)
            visits_df = visits_df[["T1", "T2", "T3"]].dropna()
            status = "tested"
            if len(visits_df) < 6: status = "too_few_complete_participants"
            elif np.all(visits_df.nunique(axis=1).eq(1)): status = "no_within_participant_change"
            result_dict = {
                "cohort": cohort,
                "target": target,
                "participants": len(visits_df),
                "T1_mean": visits_df.T1.mean(),
                "T2_mean": visits_df.T2.mean(),
                "T3_mean": visits_df.T3.mean(),
                "status": status,
            }

            # print("test status", cohort, target, status)
            # SKIP: no within-person change or too few full participants
            if status != "tested":
                if len(visits_df) >= 6: p_value = 1.0  # no change across visits
                else:
                    p_value = np.nan  # not enough full participants
                result_dict["p_anova_GG"] = p_value
                result_dict["p_friedman_permutation"] = p_value
                result_dicts.append(result_dict)
                continue

            # COMPARE: full visit rows -> ANOVA and rank-check results
            ratings_long_df = visits_df.reset_index().melt(id_vars="participant_id", var_name="visit", value_name="score",)
            test_dict, _ = compare_visits(
                ratings_df=ratings_long_df,
                column="score",
                permutations=permutations,
                seed=seed,
            )
            test_dict.pop("score")
            result_dict.update(test_dict)
            result_dicts.append(result_dict)
        print(f"done {cohort}: {len(cohort_df) // 3} participants, {len(target_cols)} outcomes", flush=True)


    # ADJUST: multiple ratings/scores -> corrected p values
    results_df = pd.DataFrame(result_dicts)
    for cohort in cohorts_dict:
        for p_col in ["p_anova_GG", "p_friedman_permutation"]:
            p_mask = results_df.cohort.eq(cohort) & results_df[p_col].notna()
            p_srs = results_df.loc[p_mask, p_col]
            fdr_arr = multipletests(p_srs, method="fdr_bh")[1]
            holm_arr = multipletests(p_srs, method="holm")[1]
            results_df.loc[p_mask, p_col + "_FDR"] = fdr_arr
            results_df.loc[p_mask, p_col + "_Holm"] = holm_arr

    # SAVE: visit means / stats / field lookup
    run_dir = results_dir / ("all_clinician_visit_changes__" + datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S_%f"))
    run_dir.mkdir()
    results_df.to_csv(run_dir / "metrics.csv", index=False)
    scores_df.to_csv(run_dir / "clinician_scores.csv", index=False)
    label_audit_df.to_csv(run_dir / "question_fields.csv", index=False)
    settings_dict = {
        "primary_cohort": "all approved matched clinician records with all three visits; raw telemetry not required",
        "sensitivity_cohort": "same27 complete participants used in prior speed comparison",
        "outcomes": target_cols,
        "tests": f"Repeated-measures ANOVA with Greenhouse-Geisser correction; rank permutation check, {permutations} permutations seed{seed}",
        "multiplicity": "FDR BH and Holm across all screened outcomes, separately per cohort and test method; constants included with p=1",
        "interpretation": "Exploratory within-participant visit differences; no treatment-group adjustment; visit/scenario confounding possible; source labels unchanged",
    }
    (run_dir / "settings.json").write_text(json.dumps(settings_dict, indent=2))
    display_cols = [
        "target", "participants", "T1_mean", "T2_mean", "T3_mean",
        "p_anova_GG", "p_anova_GG_FDR", "p_friedman_permutation", "p_friedman_permutation_FDR",
    ]
    display_df = results_df.loc[results_df.cohort.eq("all_complete_clinician")]
    display_df = display_df.sort_values("p_anova_GG")[display_cols].head(12)

    print(display_df.round(5).to_string(index=False))
    print(run_dir)
    return run_dir
