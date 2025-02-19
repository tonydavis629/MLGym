"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import random


def heuristic_fn(clauses, assignments, conflict_count, assignment_history, level):
    # Randomly selects a variable from the unassigned ones.
    
    # DO NOT CHANGE THIS SEED. IF YOU CHANGE THIS, YOUR RESULTS WILL NOT BE REPRODUCIBLE AND YOU WILL GET A ZERO.
    random.seed(10)

    unassigned_vars = list(set([
        abs(lit) for clause in clauses for lit in clause if abs(lit) not in assignments
    ]))
    if not unassigned_vars:
        return None, True
    variable = random.choice(unassigned_vars)

    # Always assigns True to the given variable.
    value = True
    return variable, value
