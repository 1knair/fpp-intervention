"""Prepped feature table -> val and test predictions -> CSV results"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# EXPERIMENTS: settings dict -> full feature sets or subsets (TODO flag)
def build_experiments(experiments_dict):
    """Experiment dicts -> run dicts; optinal subsets includes full feature set"""
    runs_dict = {}
    for name, exp_dict in experiments_dict.items():
        if not exp_dict["enabled"]: continue
        feature_combos_list = [exp_dict["features"]]

        # SUBSETS: optionally try smaller combinations of the same inputs
        if exp_dict["test_feature_subsets"]:
            feature_combos_list = []
            for size in range(1, len(exp_dict["features"]) + 1):
                for subset in combinations(exp_dict["features"], size): feature_combos_list.append(list(subset))

        for number, feature_cols in enumerate(feature_combos_list, 1):
            run_dict = exp_dict.copy()
            run_dict["experiment"] = name
            run_dict["features"] = feature_cols
            # label generated subsets vs original custom feature_set
            if feature_cols == exp_dict["features"]: run_dict["feature_subset"] = "full set"
            else:
                run_dict["feature_subset"] = f"subset {number:02d}"
            runs_dict[f"{name}__{number:03d}"] = run_dict
    return runs_dict


# OUTPUTS: data_df + experiment settings -> dict of Y Series
def make_y(data_df, exp_dict, all_rating_cols, score_limits_dict):
    # SELECT: individual ratings and their custom sum
    # todo decide: add individual/sum flag later
    """Data + experiment -> rating / score srs dict; missing custom rating -> missing custom sum"""
    y_dict = {}
    # redcap ratings alr have all of them, choose all or just our custom target set's ratings
    if exp_dict["test_all_ratings"]: selected_rating_cols = all_rating_cols
    else:
        selected_rating_cols = exp_dict["ratings"]
    for column in selected_rating_cols: y_dict[column] = data_df[column]

    # SUM: custom target-set ratings -> custom score
    custom_rating_cols = exp_dict["ratings"]
    custom_ratings_df = data_df[custom_rating_cols]
    custom_sum_srs = custom_ratings_df.sum(axis=1, min_count=len(custom_rating_cols))  # any missing rating -> missing sum
    y_dict["sum"] = custom_sum_srs

    # SCORES: always include the existing scores
    # todo decide: add existing-scores flag later or keep automatic
    for column in score_limits_dict: y_dict[column] = data_df[column]
    return y_dict


# MODEL: settings -> fresh imputer, scaler and predictor
def make_model(exp_dict):
    """Settings -> fresh imputer/scaler/reg pipeline; nothing fitted yet"""
    if exp_dict["model"] == "linear": predictor = LinearRegression()
    elif exp_dict["model"] == "ridge": predictor = Ridge(alpha=exp_dict["alpha"])
    elif exp_dict["model"] == "neural_net":
        predictor = MLPRegressor(
            hidden_layer_sizes=tuple(exp_dict["hidden_layers"]),
            alpha=exp_dict["alpha"],
            solver="lbfgs",
            max_iter=exp_dict["max_iter"],
            random_state=exp_dict["seed"],
        )
    else:
        raise ValueError("pick linear, ridge or neural_net")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    # PIPELINE: fill missing X values -> scale X -> chosen reg
    # imputer and scaler fit on training rows only
    model = make_pipeline(imputer, scaler, predictor)
    return model


# TRAIN + PREDICT: train_df + eval_df -> predictions_df
def predict_split(train_df, eval_df, exp_dict, all_rating_cols, score_limits_dict, run_id):
    """Train/eval DFs -> predictions DF; separate reg per Y, preproc fits on train only"""
    assert set(train_df.participant_id).isdisjoint(eval_df.participant_id)

    train_y_dict = make_y(train_df, exp_dict, all_rating_cols, score_limits_dict)
    eval_y_dict = make_y(eval_df, exp_dict, all_rating_cols, score_limits_dict)
    prediction_dfs = []
    # todo maybe combine into 1 multi-output model
    """
    train_y_df = pd.DataFrame(train_y_dict)
    eval_y_df = pd.DataFrame(eval_y_dict)
    train_x_df = train_df[exp_dict["features"]]
    eval_x_df = eval_df[exp_dict["features"]]

    multi_model = make_model(exp_dict)
    # todo missing ratings - each Y currently has different valid rows
    # multi_model.fit(train_x_df, train_y_df)
    # multi_pred_arr = multi_model.predict(eval_x_df)

    # todo array columns -> train_y_df.columns -> prediction rows
    # todo limits differ for ratings vs scores
    """
    for target_col, train_y_srs in train_y_dict.items():
        eval_y_srs = eval_y_dict[target_col]
        train_mask = train_y_srs.notna()
        eval_mask = eval_y_srs.notna()

        # TRAIN: fit preprocessing and reg using training rows only
        model = make_model(exp_dict)
        train_x_df = train_df.loc[train_mask, exp_dict["features"]]
        eval_x_df = eval_df.loc[eval_mask, exp_dict["features"]]
        # print("test x y", target_col, train_x_df.shape, train_y_srs.loc[train_mask].shape)
        model.fit(train_x_df, train_y_srs.loc[train_mask])
        # todo maybe save reg models/weights

        # PREDICT: fitted model -> val/test predictions
        raw_pred_arr = model.predict(eval_x_df)

        # LIMIT: keep predictions inside the rating or score range
        # todo maybe define rating maxes in csv
        lower = 0
        if target_col == "sum":
            # custom sum
            upper = 0
            for col in exp_dict["ratings"]:
                if col.startswith("safety_"): upper += 2
                else:
                    upper += 1

        elif target_col in score_limits_dict:
            # existing score
            lower, upper = score_limits_dict[target_col]

        else:
            # individual rating
            if target_col.startswith("safety_"): upper = 2
            else:
                upper = 1

        # COLLECT: actual answers, predictions and the training-median guess
        pred_df = eval_df.loc[eval_mask, ["recording_id", "participant_id", "visit", "fold"]].copy()
        pred_df["target"] = target_col
        pred_df["actual"] = eval_y_srs.loc[eval_mask]
        pred_df["predicted"] = np.clip(raw_pred_arr, lower, upper)
        pred_df["baseline"] = train_y_srs.loc[train_mask].median()
        prediction_dfs.append(pred_df)
    predictions_df = pd.concat(prediction_dfs, ignore_index=True)

    # SUM: add individual predictions for the custom score
    if len(exp_dict["ratings"]) > 1:
        individual_df = predictions_df.loc[predictions_df.target.isin(exp_dict["ratings"])]
        visit_cols = ["recording_id", "participant_id", "visit", "fold"]
        totals_df = individual_df.groupby(visit_cols, dropna=False).agg(# keep test rows, whose fold is blank
            count=("target", "size"),
            actual=("actual", "sum"),
            predicted=("predicted", "sum"),
        )
        totals_df = totals_df.reset_index()
        totals_df = totals_df.loc[totals_df["count"].eq(len(exp_dict["ratings"]))].drop(columns="count")  # require every rating
        totals_df["target"] = "sum_of_predictions"
        train_sum_srs = train_df[exp_dict["ratings"]].sum(axis=1, min_count=len(exp_dict["ratings"]))
        totals_df["baseline"] = train_sum_srs.median()
        predictions_df = pd.concat([predictions_df, totals_df], ignore_index=True)

    # LABEL: keep the experiment and set names beside its predictions
    predictions_df["run_id"] = run_id
    predictions_df["experiment"] = exp_dict["experiment"]
    predictions_df["feature_set"] = exp_dict["feature_set"]
    predictions_df["target_set"] = ""
    custom_mask = predictions_df["target"].isin(["sum", "sum_of_predictions"])
    predictions_df.loc[custom_mask, "target_set"] = exp_dict["target_set"]
    return predictions_df


# RESULTS: predictions_df -> score/rating metrics CSVs
def save_results(predictions_df, runs_dict, score_limits_dict, out_dir):
    score_dicts = []
    rating_dicts = []
    for (run_id, target_col), pred_df in predictions_df.groupby(["run_id", "target"], sort=False):
        exp_dict = runs_dict[run_id]
        is_custom = target_col in ["sum", "sum_of_predictions"]
        is_existing = target_col in score_limits_dict
        if is_custom:
            score_name = exp_dict["score_name"]
            if target_col == "sum": method = "direct"
            else:
                method = "added"
            target_cols = exp_dict["ratings"]
        else:
            score_name = target_col
            method = "direct"
            target_cols = [target_col]

        # MEASURE: model MAE vs training median
        mae = mean_absolute_error(pred_df.actual, pred_df.predicted)
        baseline_mae = mean_absolute_error(pred_df.actual, pred_df.baseline)

        # COMPARE: MAE diff before rounding
        if mae < baseline_mae - 1e-10: result = "lower error"
        elif mae > baseline_mae + 1e-10: result = "higher error"
        else:
            result = "same"
        if pred_df.actual.nunique() == 1: result = "no variation"
        if is_custom: score_type = "scores (custom)"
        elif is_existing: score_type = "scores (existing)"
        else:
            score_type = "ratings"
        target_set = ""
        if is_custom: target_set = exp_dict["target_set"]

        # COLLECT: one metrics row per feature combo / Y
        metric_dict = {
            "experiment": exp_dict["experiment"],
            "feature_set": exp_dict["feature_set"],
            "target_set": target_set,
            "features": " | ".join(exp_dict["features"]),
            "target_variables": " | ".join(target_cols),
            "score": score_name,
            "method": method,
            "model": exp_dict["model"],
            "mae": mae,
            "baseline_mae": baseline_mae,
            "result": result,
            "visits": len(pred_df),
            "people": pred_df.participant_id.nunique(),
            "type": score_type,
            "target": target_col,
            "feature_subset": exp_dict["feature_subset"],
            "rmse": mean_squared_error(pred_df.actual, pred_df.predicted) ** .5,
            "baseline_rmse": mean_squared_error(pred_df.actual, pred_df.baseline) ** .5,
            "r2": r2_score(pred_df.actual, pred_df.predicted),
        }
        if is_custom or is_existing: score_dicts.append(metric_dict)
        else:
            rating_dicts.append(metric_dict)

    # SAVE: metric dicts -> DFs -> CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    score_metrics_df = pd.DataFrame(score_dicts)
    score_metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    rating_metrics_df = pd.DataFrame(rating_dicts)
    rating_metrics_df.to_csv(out_dir / "rating_metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "experiments.json").write_text(json.dumps(runs_dict, indent=2))
    print(score_metrics_df[["experiment", "score", "method", "mae", "baseline_mae"]].round(3).to_string(index=False))


# RUN: prepared data_df -> validation results + test results
def run_regression(experiments_dict, score_limits_dict, run_dir, data_path, data_info_path):
    """Prepared data + exp settings -> val/test files; test participants stay out of training"""
    runs_dict = build_experiments(experiments_dict)
    # print("test combos", list(runs_dict))
    # LOAD: prepared input table -> training/validation and test rows
    data_df = pd.read_csv(data_path)
    metadata_dict = json.loads(data_info_path.read_text())
    all_rating_cols = metadata_dict["ratings"]
    trainval_df = data_df.loc[data_df.split.ne("test")]
    test_df = data_df.loc[data_df.split.eq("test")]
    assert set(trainval_df.participant_id).isdisjoint(test_df.participant_id)

    run_dir.mkdir(parents=True)
    (run_dir / "experiments.json").write_text(json.dumps(runs_dict, indent=2))


    # VALIDATE: fold participants stay out of training
    print("\nREGRESSION - VALIDATION", flush=True)
    validation_dfs = []
    folds = sorted(trainval_df.fold.unique())
    for run_id, exp_dict in runs_dict.items():
        for fold in folds:
            train_df = trainval_df.loc[trainval_df.fold.ne(fold)]
            val_df = trainval_df.loc[trainval_df.fold.eq(fold)]
            pred_df = predict_split(
                train_df=train_df,
                eval_df=val_df,
                exp_dict=exp_dict,
                all_rating_cols=all_rating_cols,
                score_limits_dict=score_limits_dict,
                run_id=run_id,
            )
            validation_dfs.append(pred_df)
    validation_df = pd.concat(validation_dfs, ignore_index=True)

    save_results(validation_df, runs_dict, score_limits_dict, run_dir / "validation")

    # TEST: train on all training/validation people and predict test people
    print("\nREGRESSION - TEST", flush=True)
    test_prediction_dfs = []
    for run_id, exp_dict in runs_dict.items():
        pred_df = predict_split(
            train_df=trainval_df,
            eval_df=test_df,
            exp_dict=exp_dict,
            all_rating_cols=all_rating_cols,
            score_limits_dict=score_limits_dict,
            run_id=run_id,
        )
        test_prediction_dfs.append(pred_df)
    test_predictions_df = pd.concat(test_prediction_dfs, ignore_index=True)


    save_results(test_predictions_df, runs_dict, score_limits_dict, run_dir / "test")
    print("\nsaved:", run_dir)
