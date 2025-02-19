"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Evaluation utilities for the MLGym framework.
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties, fontManager

MODELS = ["llama3-405b-tools", "gpt4o2", "claude-35-sonnet-new", "gemini-15-pro", "gpt-o1"]

MODEL_NAME_MAP = {
    "llama3-405b-tools": "Llama-3.1-405b-Instruct",
    "gpt-o1": "OpenAI O1-preview",
    "claude-35-sonnet-new": "Claude-3.5-Sonnet",
    "gemini-15-pro": "Gemini-1.5-Pro",
    "baseline": "Baseline",
    "gpt4o2": "GPT-4o"
}

MODEL_SHORT_NAME_MAP = {
    "llama3-405b-tools": "Llama",
    "gpt-o1": "O1-preview",
    "claude-35-sonnet-new": "Claude", 
    "gemini-15-pro": "Gemini",
    "baseline": "Baseline",
    "gpt4o2": "GPT-4o"
}

MODEL_COST_MAP = {
    "llama3-405b-tools": {
        "input_price": 3.5e-06,
        "output_price": 3.5e-06
    },
    "gpt4o2": {
        "input_price": 2.5e-06,
        "output_price": 10e-06
    },
    "claude-35-sonnet-new": {
        "input_price": 3e-06,
        "output_price": 15e-06
    },
    "gemini-15-pro": {
        "input_price": 1.25e-6,      # $1.25 per 1M tokens for <= 128k
        "output_price": 5e-6,        # $5.00 per 1M tokens for <= 128k
        "input_price_long": 2.5e-6,  # $2.50 per 1M tokens for > 128k
        "output_price_long": 10e-6   # $10.00 per 1M tokens for > 128k
    },
    "gpt-o1": {
        "input_price": 15e-06,
        "output_price": 60e-06
    }
}

MODEL_LOGOS = {
    "llama3-405b-tools": ("assets/logos/meta-logo.png", 0.15),
    "gpt-o1": ("assets/logos/openai-logo.png", 0.15),
    "claude-35-sonnet-new": ("assets/logos/anthropic-logo.png", 0.15),
    "gemini-15-pro": ("assets/logos/google-logo.png", 0.15),
    "gpt4o2": ("assets/logos/openai-green.png", 0.15)
}

MODEL_MARKER_MAP = {
    "llama3-405b-tools": "-",      # solid line
    "claude-35-sonnet-new": "--",           # dashed line
    "gemini-15-pro": "-.",              # dash-dot line
    "gpt4o2": ":",                       # dotted line
    "gpt-o1": (0, (3, 1, 1, 1)), # dash-dot-dot line
}

EXIT_STATUS_MAP = {
    'autosubmission (exit_api)': 'API',
    'autosubmission (exit_context)': 'Context',
    'autosubmission (exit_cost)': 'Cost',
    'autosubmission (exit_format)': 'Format',
    'autosubmission (max_steps)': 'Max Steps',
    'early_exit': 'Runtime',
    'evaluation_format_error (exit_cost)': 'Cost',
    'evaluation_format_error (exit_format)': 'Format',
    'evaluation_format_error (max_steps)': 'Evaluation',
    'evaluation_format_error (submit)': 'Evaluation',
    'submission_not_found (max_steps)': 'Evaluation',
    'submitted': 'Success',
    'unknown_error (open "data/train.csv" 0)': 'Permission',
    'unknown_error (torchrun --nproc_per_node=1 --standalone baseline.py)': 'Runtime',
    'unknown_error (validate)': 'Evaluation',
    'submission_not_found (submit)': 'Evaluation',
    'unknown_error (ls -R data/)': 'Runtime',
    'unknown_error (ls data/train/)': 'Runtime', 
    'submission_not_found (exit_format)': 'Evaluation',
    'unknown_error (python train.py)': 'Runtime',
    'unknown_error (python baseline.py)': 'Runtime'
}

ACTION_LIST = ["Edit", "View", "Validate", "Submit", "Search", "Python", "Bash"]

# COLORS = ["#253494", "#2c7fb8", "#41b6c4", "#7fcdbb", "#c7e9b4", "#ffffcc"]
GNBU_MULTI_HUE = ["#0868ac", "#43a2ca", "#7bccc4", "#a8ddb5", "#ccebc5", "#f0f9e8"]
ORANGE_SINGLE_HUE = ["#a63603", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2", "#feedde"]
DIVERGING_COLORS = ["#FD5901", "#F78104", "#FAAB36", "#249EA0", "#008083", "#005F60", ]
# PAUL_TOL_GPT = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#4A4A4A"]
PAUL_TOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
PAUL_TOL_REORDERED = ["#4477AA", "#228833", "#AA3377", "#EE6677", "#CCBB44","#66CCEE", "#BBBBBB"]
PAUL_TOL_REORDERED_NAMES = ["blue", "cyan", "green", "purple", "red", "yellow", "grey"]
MODELS = ["llama3-405b-tools", "gpt4o2", "claude-35-sonnet-new", "gemini-15-pro", "gpt-o1"]

COLOR_CHOICE = PAUL_TOL_REORDERED

MODEL_COLOR_MAP = {model: color for model, color in zip(MODELS, COLOR_CHOICE)}
ACTION_COLOR_MAP = {action: color for action, color in zip(ACTION_LIST, PAUL_TOL)}

TASKS = {
    "regressionKaggleHousePrice": {
        "name": "House Price",
        "shortname": "Regression",
        "priority_metric": "r2",
        "metric_direction": "maximize",
    },
    "3SATHeuristic": {
        "name": "3-SAT",
        "shortname": "3-SAT",
        "priority_metric": "Time",
        "metric_direction": "minimize",
    },
    "imageClassificationCifar10": {
        "name": "CIFAR-10",
        "shortname": "CIFAR-10",
        "priority_metric": "accuracy",
        "metric_direction": "maximize",
    },
    "imageClassificationFMnist": {
        "name": "Fashion MNIST",
        "shortname": "F-MNIST",
        "priority_metric": "accuracy",
        "metric_direction": "maximize",
    },
    "imageCaptioningCOCO": {
        "name": "MS-COCO",
        "shortname": "MS-COCO",
        "priority_metric": "BLEU Score",
        "metric_direction": "maximize",
    },
    "languageModelingFineWeb": {
        "name": "Language Modeling",
        "shortname": "FineWeb",
        "priority_metric": "val_loss",
        "metric_direction": "minimize",
    },
    "naturalLanguageInferenceMNLI": {
        "name": "MNLI",
        "shortname": "MNLI",
        "priority_metric": "validation_accuracy",
        "metric_direction": "maximize",
    },
    "battleOfSexes": {
        "name": "Battle of Sexes",
        "shortname": "BoS",
        "priority_metric": "Score",
        "metric_direction": "maximize",
    },
    "prisonersDilemma": {
        "name": "Prisoners Dilemma",
        "shortname": "PD",
        "priority_metric": "Score",
        "metric_direction": "maximize",
    },
    "blotto": {
        "name": "Blotto",
        "shortname": "Blotto",
        "priority_metric": "Score",
        "metric_direction": "maximize",
    },
    "rlBreakoutMinAtar": {
        "name": "Breakout",
        "shortname": "Breakout",
        "priority_metric": "Reward Mean",
        "metric_direction": "maximize",
    },
    "rlMetaMazeMisc": {
        "name": "Meta Maze",
        "shortname": "Maze",
        "priority_metric": "Reward Mean",
        "metric_direction": "maximize",
    },
    "rlMountainCarContinuous": {
        "name": "Mountain Car Continuous",
        "shortname": "MountainCar",
        "priority_metric": "Reward Mean",
        "metric_direction": "maximize",
    },
}

def set_custom_font():
    """Set the custom font for the plots."""
    # font_path = "/Users/dnathani/Library/Fonts/BerkeleyMonoNerdFontMono-Bold.ttf"
    # font_path = "/Users/dnathani/Library/Fonts/Source Code Pro for Powerline.otf"
    # font_path = "/Users/dnathani/Library/Fonts/FiraCodeNerdFontMono-Regular.ttf"
    # font_path = "/Users/dnathani/Library/Fonts/Droid Sans Mono for Powerline Nerd Font Complete.otf"
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    if not Path(font_path).exists():
        return
    fontManager.addfont(font_path)
    prop = FontProperties(fname=font_path)
    sns.set_theme(font=prop.get_name(), style="dark")
    plt.rcParams["font.family"] = prop.get_name()

def get_fig_size() -> None:
    """Set the figure size for the plots."""
    fig_width_pt = 472.03123  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height =fig_width*golden_mean       # height in inches
    fig_size = [fig_width,fig_height]
    print(f"Figure size: {fig_size}")
        
    
def get_best_attempt(results: dict, priority_metric: str, metric_direction: str) -> int:
    """Returns the index of the best agent according to the priority metric."""
    best_attempt_idx = -1
    best_metric_value = -float("inf") if metric_direction == "maximize" else float("inf")
    for i, result in enumerate(results):
        if priority_metric not in result:
            print(f"Priority metric {priority_metric} not found in result for agent {i}")
            continue
        metric_value = result[priority_metric]
        if (metric_direction == "maximize" and metric_value > best_metric_value) or \
           (metric_direction == "minimize" and metric_value < best_metric_value):
            best_metric_value = metric_value
            best_attempt_idx = i
    return best_attempt_idx

def get_best_scores(results: dict, priority_metric: str, metric_direction: str, models: list[str]) -> dict:
    """Computes the best attempts for all models on a task"""
    all_scores = {model: {} for model in models + ["baseline"]}

    for model in models:
        best_attempts = []
        best_submissions = []
        for score in results[model]["scores"]:
            best_attempt_idx = get_best_attempt(score["agent"], priority_metric, metric_direction)
            best_attempts.append(score["agent"][best_attempt_idx])
            best_submissions.append(score["agent"][-1][priority_metric])
        all_scores[model]["best_attempts"] = best_attempts
        all_scores[model]["best_submissions"] = best_submissions
        
        if metric_direction == "maximize":
            all_scores[model]["overall_best_submission"] = np.max(best_submissions)
            all_scores[model]["overall_best_attempt"] = np.max(best_attempts)
        else:
            all_scores[model]["overall_best_submission"] = np.min(best_submissions)
            all_scores[model]["overall_best_attempt"] = np.min(best_attempts)
    
    all_scores["baseline"] = {"overall_best_submission": results["scores"][0]["baseline"][priority_metric]}
        
    return all_scores

def process_trajectories(traj_parent_dir: str, traj_pattern: str, task_id: str, models: list[str]) -> dict:
    """
    Get all results.json and .traj files from the trajectory directory pattern for a given task
    """
    all_results = {}
    for model in models:
        model_results = {"scores": [], "trajectories": [], "exit_statuses": []}
        traj_dir_pattern = f"{traj_parent_dir}/*{model}__{task_id}__{traj_pattern}*"
        traj_dirs = sorted(list(Path().glob(traj_dir_pattern)))
        for traj_dir in traj_dirs:
            results_file= Path(traj_dir) / "results.json"
            traj_file = list(Path().glob(f"{traj_dir}/*.traj"))
            results = json.load(open(results_file))
            traj = json.load(open(traj_file[0]))
            exit_status = "unknown_error"

            # get last action
            last_action = "unknown"
            for history in reversed(traj["history"]):
                if history["role"] == "assistant":
                    last_action = history["action"].strip()
                    break

            exit_status = traj["info"]["exit_status"]
            if exit_status == "":
                exit_status = f"unknown_error ({last_action})"
                
            exit_status = EXIT_STATUS_MAP.get(exit_status, exit_status)
                
            model_results["scores"].append(results)
            model_results["trajectories"].append(traj["trajectory"])
            model_results["exit_statuses"].append(exit_status)

        all_results[model] = model_results

    return all_results