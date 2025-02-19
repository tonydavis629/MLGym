"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from typing import Callable, Tuple

import numpy


class DPLL:
    """DPLL solver for 3-SAT"""

    def __init__(self, clauses, heuristic_function):
        # Initialize the DPLL solver with a heuristic function for variable selection
        self.assignments = (
            {}
        )  # Current variable assignments, keys are positive literals
        self.heuristic_function = heuristic_function  # Heuristic function for variable selection and assignment
        self.conflict_count = {}  # Track conflict frequency for each variable
        self.assignment_history = []  # Track the order of variable assignments
        self.resolved_clauses = (
            []
        )  # resolved clauses (have at least one positve literal)
        self.unresolved_clauses = (
            clauses  # clauses which are unresolved (no positive literal in the clause)
        )
        self.level = (
            []
        )  # maintains the level of assignment, for unit propagation, the level doesn't change
        # level only changes for decision variables
        self.num_decisions = (
            0  # number of decision variable assignments made by calling a heuristic fn
        )

    @staticmethod
    def is_satisfied_clause(clause, assignments) -> bool:
        """
        :param clause
        :param assignments
        :return: True, if the clause is satisfied: has at least one literal that is positively assigned
        """
        return any(
            (
                lit > 0 and lit in assignments and assignments[lit] is True
            )  # Positive literal is True
            or (
                lit < 0
                and abs(lit) in assignments
                and assignments[abs(lit)] is False
                # Negative literal is True
            )
            for lit in clause  # Iterate over literals in the clause
        )

    @staticmethod
    def is_conflicted_clause(clause, assignments) -> bool:
        """
        :param clause
        :param assignments
        :return: True, if the clause has conflicts: has all literals as False
        """
        if all(abs(lit) in assignments for lit in clause):
            if all(
                (lit > 0 and assignments[lit] is False)  # Positive literal is False
                or (
                    lit < 0 and assignments[abs(lit)] is True
                )  # Negative literal is False
                for lit in clause  # Iterate over literals in the clause
            ):
                return True
        return False

    @staticmethod
    def is_unit_clause(clause, assignments) -> bool:
        """
        :param clause
        :param assignments:
        :return: True, if the clause is unit: has only one unassigned literal
        """
        return sum(1 for lit in clause if abs(lit) not in assignments) == 1 and any(
            abs(lit) in assignments for lit in clause
        )

    def is_satisfied(self):
        """
        Check if all clauses are satisfied.
        A clause is satisfied if at least one of its literals is True.
        """
        if len(self.unresolved_clauses) == 0:
            return True
        return all(
            DPLL.is_satisfied_clause(clause, self.assignments)
            for clause in self.unresolved_clauses
        )

    def has_conflict(self):
        """
        Return True if there is a conflict in the current assignments.
        A conflict occurs when all literals in a clause are False under the current assignments.
        """
        conflict = False
        for clause in self.unresolved_clauses:
            if DPLL.is_conflicted_clause(clause, self.assignments):
                conflict = True
                for lit in clause:
                    var = abs(lit)
                    if var in self.conflict_count:
                        self.conflict_count[var] += 1
                    else:
                        self.conflict_count[var] = 1
        return conflict

    def unit_propagate(self):
        """
        Perform unit propagation to simplify the formula.
        A unit clause is a clause with only one unassigned literal,
        which must be assigned in a way that satisfies the clause.
        """
        assignments = deepcopy(self.assignments)
        assignment_history = copy(self.assignment_history)
        # find the unsatisfied clauses (where not even a single literal is True)
        unresolved_clauses = copy(self.unresolved_clauses)
        resolved_clauses = copy(self.resolved_clauses)
        # find the unit clauses amongst unsatisfied clauses
        unit_clauses = [
            clause
            for clause in unresolved_clauses
            if DPLL.is_unit_clause(clause, assignments)
        ]
        while len(unit_clauses) > 0:
            for unit_clause in unit_clauses:
                # Find the unassigned literal in the unit clause
                unassigned = [lit for lit in unit_clause if abs(lit) not in assignments]
                if len(unassigned) == 0:
                    continue
                unit_literal = unassigned[0]
                # Assign the value that satisfies the unit clause
                var = abs(unit_literal)
                # for a negative literal, assign the opposite value to the positive literal
                assignments[var] = True if unit_literal > 0 else False
                assignment_history.append(var)  # Track the assignment order
                ind = 0
                while ind < len(unresolved_clauses):
                    unresolved = unresolved_clauses[ind]
                    # if the unresolved clause has a positive assignment
                    if unit_literal in unresolved:
                        resolved_clauses.append(unresolved_clauses.pop(ind))
                    else:
                        ind += 1
                # If all clauses are satisfied, return True
                if len(unresolved_clauses) == 0 or all(
                    DPLL.is_satisfied_clause(clause, assignments)
                    for clause in unresolved_clauses
                ):
                    # TODO: uncomment if solution is needed
                    # self.level[-1].extend([k for k in assignments.keys() if k not in self.assignments.keys()])
                    # self.assignments = assignments
                    # self.assignment_history = assignment_history
                    # self.resolved_clauses = resolved_clauses
                    # self.unresolved_clauses = unresolved_clauses
                    return True
                # If a conflict arises, return False
                if any(
                    DPLL.is_conflicted_clause(clause, assignments)
                    for clause in unresolved_clauses
                ):
                    if var in self.conflict_count:
                        self.conflict_count[var] += 1
                    else:
                        self.conflict_count[var] = 1
                    return False

            unit_clauses = [
                clause
                for clause in unresolved_clauses
                if DPLL.is_unit_clause(clause, assignments)
            ]

        if len(assignments) > 0:
            self.level[-1].extend(
                [k for k in assignments.keys() if k not in self.assignments.keys()]
            )
        self.assignments = assignments
        self.assignment_history = assignment_history
        self.resolved_clauses = resolved_clauses
        self.unresolved_clauses = unresolved_clauses
        return None  # No definitive result from unit propagation

    def assign(self, literal, value):
        """
        Assign a value to a variable.
        Keys are always positive literals
        """
        var = abs(literal)
        # for a negative literal, assign the opposite value to the positive literal
        self.assignments[var] = value if literal > 0 else not value
        self.assignment_history.append(var)  # Track the assignment order
        self.level.append([var])
        # check if some clause in unresolved_clauses has now been resolved,
        # and move it to self.resolved_clauses
        ind = 0
        while ind < len(self.unresolved_clauses):
            clause = self.unresolved_clauses[ind]
            # if the unresolved clause has a positive assignment
            if (
                literal in clause
                and value is True
                or -literal in clause
                and value is False
            ):
                self.resolved_clauses.append(self.unresolved_clauses.pop(ind))
            else:
                ind += 1

    def backtrack(self):
        """Backtrack on variable assignments."""
        # if self.assignments:
        # remove the whole last level
        # this includes the decision variable as well as other variables assigned through unit propagation
        level = self.level.pop()
        if not level:
            return
        for var in level:
            del self.assignments[var]
            # remove clauses from resolved_clauses and
            # put them into unresolved clauses
            ind = 0
            while ind < len(self.resolved_clauses):
                clause = self.resolved_clauses[ind]
                if not DPLL.is_satisfied_clause(clause, self.assignments):
                    self.unresolved_clauses.append(self.resolved_clauses.pop(ind))
                else:
                    ind += 1

    def solve(self):
        """Solve the CNF formula using DPLL."""
        if self.is_satisfied():
            return True
        if self.has_conflict():
            return False

        # Unit propagate before branching
        pred_is_sat = self.unit_propagate()
        if pred_is_sat is not None:
            return pred_is_sat

        # # Use the provided heuristic function to select next variable to assign and its value
        variable, value = self.heuristic_function(
            self.unresolved_clauses,
            self.assignments,
            self.conflict_count,
            self.assignment_history,
            self.level,
        )
        self.num_decisions += 1
        if variable is None:
            return False

        for val in [value, not value]:
            self.assign(variable, val)
            if self.solve():  # Recursively solve with the new assignment
                return True
            # Backtrack if the current assignment did not lead to a solution
            self.backtrack()
        return False

    @staticmethod
    def num_new_clauses(args):
        """
        calculates the number of new clauses generated after assigning a variable.
        useful for simulating a variable assignment for lookahead search.
        Here, clauses are unresolved clauses wherein no literal is True.
        # returns: a score based on number of new clauses generated
        # if satisfied, return a very high score
        # if conflict arises, return a very low score
        """
        variable, value, clauses, assignments = args
        # all size 3 at the beginning of the simulation
        # TODO: tailored for 3-SAT, make it more general
        size_3_clauses = [
            clause
            for clause in clauses
            if all(abs(lit) not in assignments for lit in clause)
        ]

        # assign the variable
        assignments[variable] = value
        ind = 0
        while ind < len(clauses):
            clause = clauses[ind]
            # if the unresolved clause has a positive assignment
            if (
                variable in clause
                and value is True
                or -variable in clause
                and value is False
            ):
                clauses.pop(ind)
            else:
                ind += 1
        # If all clauses are satisfied, return True
        if len(clauses) == 0 or all(
            DPLL.is_satisfied_clause(clause, assignments) for clause in clauses
        ):
            return 1e4
        # If a conflict arises, return False
        if any(DPLL.is_conflicted_clause(clause, assignments) for clause in clauses):
            return -1e4

        # simulate unit propagation with new assignment
        unit_clauses = [
            clause for clause in clauses if DPLL.is_unit_clause(clause, assignments)
        ]

        while unit_clauses:
            for clause in unit_clauses:
                # Find the unassigned literal in the unit clause
                unassigned = [lit for lit in clause if abs(lit) not in assignments]
                if len(unassigned) == 0:
                    continue
                unit_literal = unassigned[0]
                # Assign the value that satisfies the unit clause
                assignments[abs(unit_literal)] = True if unit_literal > 0 else False
                ind = 0
                while ind < len(clauses):
                    clause = clauses[ind]
                    # if the unresolved clause has a positive assignment
                    if unit_literal in clause:
                        clauses.pop(ind)
                    else:
                        ind += 1
                # If all clauses are satisfied, return True
                if len(clauses) == 0 or all(
                    DPLL.is_satisfied_clause(clause, assignments) for clause in clauses
                ):
                    return 1e4
                # If a conflict arises, return False
                if any(
                    DPLL.is_conflicted_clause(clause, assignments) for clause in clauses
                ):
                    return -1e4
            # Update the list of unit clauses after assignments
            unit_clauses = [
                clause for clause in clauses if DPLL.is_unit_clause(clause, assignments)
            ]

        # Return the number of new clauses generated after the unit propagation
        new_clauses = (
            []
        )  # can only be size 2 formed from size 3 clauses in the beginning
        for clause in clauses:
            if (
                clause in size_3_clauses
                and sum(1 for lit in clause if abs(lit) not in assignments.keys()) == 2
            ):
                new_clauses.append(clause)
        return len(new_clauses)

