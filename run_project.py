"""Read config/*.csv and do the selected analysis"""
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from shutil import copy2
import pandas as pd

from src.representation.extract_features import extract_features
from src.modeling.anova import run_anova
from src.modeling.regression import run_regression

ROOT = Path(__file__).resolve().parent


# LOAD EXPERIMENTS: CSV -> dict of experiment settings
# _row = 1 CSV row, _cols = column names

def load_experiments(csv_path):
    experiments_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    feature_sets_df = pd.read_csv(csv_path.parent / "feature_sets.csv", dtype=str, keep_default_na=False)
    target_sets_df = pd.read_csv(csv_path.parent / "target_sets.csv", dtype=str, keep_default_na=False)

    # JOIN: each experiment -> its actual feature and rating lists
    experiments_df = experiments_df.merge(feature_sets_df[["feature_set", "features"]], on="feature_set", how="left", validate="many_to_one")
    experiments_df = experiments_df.merge(target_sets_df, on="target_set", how="left", validate="many_to_one")
    if experiments_df["features"].isna().any(): raise ValueError("CHECK CSV: feature set name not found")
    if experiments_df["ratings"].isna().any(): raise ValueError("CHECK CSV: target set name not found")

    bool_dict = {"true": True, "false": False}

    experiments_dict = {}

    # PARSE: one joined row -> one experiment dict
    for exp_row in experiments_df.itertuples(index=False):
        # PARSE: feature-set cell -> list of feature col names
        # todo data val: unknown col names, spaces
        feature_cols = []
        for col in exp_row.features.splitlines(): feature_cols.append(col)

        # PARSE: rating(target)-set cell -> list of rating col names
        rating_cols = []
        for col in exp_row.ratings.splitlines(): rating_cols.append(col)

        exp_dict = {
            "enabled": bool_dict[exp_row.enabled.lower()],
            "feature_set": exp_row.feature_set,
            "target_set": exp_row.target_set,
            "features": feature_cols,
            "ratings": rating_cols,
            "score_name": exp_row.score_name,
            "model": exp_row.model,
            "test_all_ratings": bool_dict[exp_row.test_all_ratings.lower()],
            "test_feature_subsets": bool_dict[exp_row.test_feature_subsets.lower()],
            "notes": exp_row.notes,
        }
        if exp_row.alpha: exp_dict["alpha"] = float(exp_row.alpha)
        if exp_row.hidden_layers:
            layer_sizes = []
            for size in exp_row.hidden_layers.splitlines(): layer_sizes.append(int(size))
            exp_dict["hidden_layers"] = layer_sizes
        if exp_row.max_iter: exp_dict["max_iter"] = int(exp_row.max_iter)
        if exp_row.seed: exp_dict["seed"] = int(exp_row.seed)

        experiments_dict[exp_row.experiment] = exp_dict

    # OUTPUT: experiment name -> settings dict
    """
    {
        "general_safety_ridge": {
            "feature_set": "general_driving",
            "target_set": "safety13",
            "features": ["velocity_sd", "lane_offset_abs_mean", ...],
            "ratings": ["safety_roundabout", "safety_collision1", ...],
            "score_name": "Safety-grade sum (13 ratings)",
            "model": "ridge",
            "alpha": 10.0,
            "enabled": True,
            ...
        },
        "speed_raw_linear": {...},
    }
    """
    return experiments_dict


# RUN: cleaned drives -> inputs, or prepared inputs -> analysis results

if __name__ == "__main__":
    # CMD: build inputs or run one analysis
    parser = argparse.ArgumentParser(description="Build driving features or run the analysis")
    parser.add_argument("command", nargs="?", choices=["inputs", "regression", "anova"])
    args = parser.parse_args()
    # print("test command", args.command)

    # LOAD: CSV settings -> settings_dict
    config_dir = ROOT / "config"
    settings_df = pd.read_csv(config_dir / "settings.csv", dtype=str, keep_default_na=False)
    settings_srs = settings_df.set_index("setting")["value"]
    settings_dict = settings_srs.to_dict()
    results_dir = ROOT / settings_dict["results_dir"]

    # SELECT: cmd wins, else use  mode in settings
    command = args.command
    if command is None: command = settings_dict["mode"]

    if command == "inputs": extract_features()
    elif command == "anova":
        run_dir = run_anova(
            assessment_file=ROOT / settings_dict["assessment_file"],
            permutations=int(settings_dict["anova_permutations"]),
            seed=int(settings_dict["anova_seed"]),
            results_dir=results_dir,
            model_data_file=ROOT / settings_dict["model_data"],
        )

    elif command == "regression":
        experiments_dict = load_experiments(config_dir / "experiments.csv")
        scores_df = pd.read_csv(config_dir / "scores.csv")
        score_limits_dict = {}
        for score_row in scores_df.itertuples(index=False): score_limits_dict[score_row.score] = [score_row.min, score_row.max]


        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S_%f")
        run_dir = results_dir / f"{settings_dict['run_name']}__{stamp}"

        run_regression(
            experiments_dict=experiments_dict,
            score_limits_dict=score_limits_dict,
            run_dir=run_dir,
            data_path=ROOT / settings_dict["model_data"],
            data_info_path=ROOT / settings_dict["model_data_info"],
        )
    else:
        raise ValueError('pick inputs, regression or anova')

    if command != "inputs":
        # SAVE: keep the settings used for this run
        saved_config_dir = run_dir / "config"
        saved_config_dir.mkdir(exist_ok=True)
        for csv_path in config_dir.glob("*.csv"): copy2(csv_path, saved_config_dir / csv_path.name)
