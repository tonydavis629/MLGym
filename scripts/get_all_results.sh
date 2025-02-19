# Copyright (c) Meta Platforms, Inc. and affiliates.

# List of models to test
MODELS=(
    "llama3-405b-tools"
    "gpt4o2"
    "gpt-o1"
    "claude-35-sonnet-new"
    "gemini-15-pro"
)

# SAT
python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern 3SATHeuristic__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "Time" \
    --metric_direction minimize > results/3SATHeuristic.res

# RL
python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern rlMountainCarContinuous__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "Reward Mean" \
    --metric_direction maximize > results/rlMountainCarContinuous.res

python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern rlMetaMaze__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "Reward Mean" \
    --metric_direction maximize > results/rlMetaMaze.res

python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern rlBreakoutMinAtar__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "Reward Mean" \
    --metric_direction maximize > results/rlBreakoutMinAtar.res

# Game Theory
python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern blotto__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "Score" \
    --metric_direction maximize > results/blotto.res

python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern prisonersDilemma__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "Score" \
    --metric_direction maximize > results/prisonersDilemma.res

python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern battleOfSexes__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "Score" \
    --metric_direction maximize > results/battleOfSexes.res

# Data Science
python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern regressionKaggleHousePrice__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "r2" \
    --metric_direction maximize > results/regressionKaggleHousePrice.res

# NLP
python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern languageModelingFineWeb__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "val_loss" \
    --metric_direction minimize > results/languageModelingFineWeb.res

# python scripts/process_results.py \
#     --traj_parent_dir trajectories/mlgym_bench_v0/ \
#     --traj_pattern naturalLanguageInferenceMNLI__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
#     --models "${MODELS[@]}" \
#     --priority_metric "validation_loss" \
#     --metric_direction minimize > results/naturalLanguageInferenceMNLI.res

# Vision
python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern imageClassificationFMnist__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "accuracy" \
    --metric_direction maximize > results/imageClassificationFMnist.res

python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern imageClassificationCifar10__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "accuracy" \
    --metric_direction maximize > results/imageClassificationCifar10.res

python scripts/process_results.py \
    --traj_parent_dir trajectories/mlgym_bench_v0/ \
    --traj_pattern imageCaptioningCOCO__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents \
    --models "${MODELS[@]}" \
    --priority_metric "val_loss" \
    --metric_direction minimize > results/imageCaptioningCOCO.res
