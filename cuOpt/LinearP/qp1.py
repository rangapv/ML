#!/usr/bin/env python3
#author:rangapv@yahoo.com
#15-02-26

from cuopt.linear_programming.problem import (
    MINIMIZE,
    Problem,
)


def main():
    # Create a new optimization problem
    prob = Problem("Simple QP")

    # Add variables with non-negative bounds
    x = prob.addVariable(lb=0)
    y = prob.addVariable(lb=0)

    # Add constraint: x + y >= 1
    prob.addConstraint(x + y >= 1)
    prob.addConstraint(0.75 * x + y <= 1)

    # Set quadratic objective: minimize x^2 + y^2
    # Using Variable * Variable to create quadratic terms
    quad_obj = x * x + y * y
    prob.setObjective(quad_obj, sense=MINIMIZE)

    # Solve the problem
    prob.solve()

    # Print results
    print(f"Optimal solution found in {prob.SolveTime:.2f} seconds")
    print(f"x = {x.Value}")
    print(f"y = {y.Value}")
    print(f"Objective value = {prob.ObjValue}")


if __name__ == "__main__":
    main()
