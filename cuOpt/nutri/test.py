#!/usr/bin/env python3

import numpy as np
import pandas as pd
from cuopt.linear_programming.problem import Problem, VType, sense, LinearExpression
from cuopt.linear_programming.solver_settings import SolverSettings
import time


categories = {
    "calories": {
        "min": 1800,
        "max": 2200
    },
    "protein": {
        "min": 91,
        "max": float('inf')
    },
    "fat": {
        "min": 0,
        "max": 65
    },
    "sodium": {
        "min": 0,
        "max": 1779
    }
}

food_costs = {
    "hamburger": 2.49,
    "chicken": 2.89,
    "hot dog": 1.50,
    "fries": 1.89,
    "macaroni": 2.09,
    "pizza": 1.99,
    "salad": 2.49,
    "milk": 0.89,
    "ice cream": 1.59
}

# Nutrition values for each food (per serving)
nutrition_data = {
    "hamburger": [410, 24, 26, 730],
    "chicken": [420, 32, 10, 1190],
    "hot dog": [560, 20, 32, 1800],
    "fries": [380, 4, 19, 270],
    "macaroni": [320, 12, 10, 930],
    "pizza": [320, 15, 12, 820],
    "salad": [320, 31, 12, 1230],
    "milk": [100, 8, 2.5, 125],
    "ice cream": [330, 8, 10, 180]
}


nutrition_df = pd.DataFrame(nutrition_data, index=categories.keys()).T
nutrition_df.columns = [f"{cat} (per serving)" for cat in categories.keys()]
print("Nutritional Values per Serving:")
print(nutrition_df)


problem = Problem("diet_optimization")

# Add decision variables for each food (amount to buy)
buy_vars = {}
for food_name in food_costs:
    var = problem.addVariable(name=f"{food_name}", vtype=VType.CONTINUOUS, lb=0.0, ub=float('inf'))
    buy_vars[food_name] = var

print(f"Created {len(buy_vars)} decision variables for foods")
print(f"Variables: {[var.getVariableName() for var in buy_vars.values()]}")

objective_expr = LinearExpression([], [], 0.0)

for var in buy_vars.values():
    if food_costs[var.getVariableName()] != 0:  # Only include non-zero coefficients
        objective_expr += var * food_costs[var.getVariableName()]

# Set objective function: minimize total cost
problem.setObjective(objective_expr, sense.MINIMIZE)

constraint_names = []

for i, category in enumerate(categories):
    # Calculate total nutrition from all foods for this category
    nutrition_expr = LinearExpression([], [], 0.0)

    for food_name in food_costs:
        nutrition_value = nutrition_data[food_name][i]
        if nutrition_value != 0:  # Only include non-zero coefficients
            nutrition_expr += buy_vars[food_name] * nutrition_value

    # Add constraint: min_nutrition[i] <= nutrition_expr <= max_nutrition[i]
    min_val = categories[category]["min"]
    max_val = categories[category]["max"]

    if max_val == float('inf'):
        # Only lower bound constraint
        constraint = problem.addConstraint(nutrition_expr >= min_val, name=f"min_{category}")
        constraint_names.append(f"min_{category}")
    else:
        # Range constraint (both lower and upper bounds)
        constraint = problem.addConstraint(nutrition_expr >= min_val, name=f"min_{category}")
        constraint_names.append(f"min_{category}")
        constraint = problem.addConstraint(nutrition_expr <= max_val, name=f"max_{category}")
        constraint_names.append(f"max_{category}")

print(f"Added {len(constraint_names)} nutrition constraints")
print(f"Constraints: {constraint_names}")

settings = SolverSettings()
settings.set_parameter("time_limit", 60.0)  # 60 second time limit
settings.set_parameter("log_to_console", True)  # Enable solver logging
settings.set_parameter("method", 0)  # Use default method

print("Solver configured with 60-second time limit")

print("Solving diet optimization problem...")
print(f"Problem type: {'MIP' if problem.IsMIP else 'LP'}")

start_time = time.time()
problem.solve(settings)
solve_time = time.time() - start_time

print(f"\nSolve completed in {solve_time:.3f} seconds")
print(f"Solver status: {problem.Status.name}")
print(f"Objective value: ${problem.ObjValue:.2f}")


def print_solution():
    """Print the optimal solution in a readable format"""
    if problem.Status.name == "Optimal":
        print(f"\nOptimal Solution Found!")
        print(f"Total Cost: ${problem.ObjValue:.2f}")
        print("\nFood Purchases:")

        total_cost = 0
        for var in buy_vars.values():
            amount = var.getValue()
            if amount > 0.0001:  # Only show foods with significant amounts
                food_cost = amount * food_costs[var.getVariableName()]
                total_cost += food_cost
                print(f"  {var.getVariableName()}: {amount:.3f} servings (${food_cost:.2f})")

        print(f"\nTotal Cost: ${total_cost:.2f}")

        # Check nutritional intake
        print("\nNutritional Intake:")
        for i, category in enumerate(categories):
            total_nutrition = 0
            for var in buy_vars.values():
                amount = var.getValue()
                nutrition_value = nutrition_data[var.getVariableName()][i]
                total_nutrition += amount * nutrition_value

            min_req = categories[category]["min"]
            max_req = categories[category]["max"]

            # Check constraints with tolerance for floating point precision
            tolerance = 1e-6
            min_satisfied = total_nutrition >= (min_req - tolerance)
            max_satisfied = (max_req == float('inf')) or (total_nutrition <= (max_req + tolerance))
            status = "✓" if (min_satisfied and max_satisfied) else "✗"

            if max_req == float('inf'):
                print(f"  {category}: {total_nutrition:.1f} (min: {min_req}) {status}")
            else:
                print(f"  {category}: {total_nutrition:.1f} (min: {min_req}, max: {max_req}) {status}")
    else:
        print(f"No optimal solution found. Status: {problem.Status.name}")

print_solution()


