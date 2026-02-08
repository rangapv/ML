#!/usr/bin/env python3
#author:rangapv@yahoo.com
#04-02-26

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# cuOpt imports
from cuopt.linear_programming.problem import Problem, VType, sense, LinearExpression
from cuopt.linear_programming.solver_settings import SolverSettings, PDLPSolverMode
from cuopt.linear_programming.solver.solver_parameters import *

# Set random seed for reproducibility
np.random.seed(42)


# Configure solver settings for larger problem
solver_settings = SolverSettings()
solver_settings.set_parameter("time_limit", 300.0)  # 5 minute time limit for larger problem
solver_settings.set_parameter("log_to_console", True)  # Enable solver logging
solver_settings.set_parameter("method", 0)  # Use default method


# Load S&P 500 data from GitHub
data_url = 'https://raw.githubusercontent.com/NVIDIA/cuopt-examples/refs/heads/main/portfolio_optimization/data/sp500.csv'
df = pd.read_csv(data_url, index_col='Date', parse_dates=True)

print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Number of assets: {len(df.columns)}")
print(f"\nFirst few columns: {list(df.columns[:10])}")

# Display basic statistics
df.head()



# Use all S&P 500 assets with complete data
# Remove any assets with missing data
price_data = df.dropna(axis=1, how='any')  # Drop columns with any NaN values
selected_assets = price_data.columns

print(f"Total assets in dataset: {len(df.columns)}")
print(f"Assets with complete data: {len(selected_assets)}")
print(f"Price data shape: {price_data.shape}")
print(f"Selected assets (first 10): {list(selected_assets[:10])}")

# Calculate log returns
returns = np.log(price_data / price_data.shift(1)).dropna()

print(f"Returns data shape: {returns.shape}")
print(f"Returns date range: {returns.index.min()} to {returns.index.max()}")

# Display return statistics
print("\nReturn Statistics (first 5 assets):")
print(returns.iloc[:, :5].describe())




mu = returns.mean().values  # Expected returns
Sigma = returns.cov().values  # Covariance matrix
n_assets = len(selected_assets)

# Annualize returns (assuming 252 trading days)
mu_annual = mu * 252
Sigma_annual = Sigma * 252

print(f"\nAnnualized expected returns (top 5):")
for i in range(5):
    print(f"{selected_assets[i]}: {mu_annual[i]:.4f}")


# Historical simulation scenarios
historical_scenarios = returns.values
n_scenarios_hist = historical_scenarios.shape[0]

print(f"Historical scenarios: {n_scenarios_hist}")
print(f"Number of assets: {len(selected_assets)}")

# For computational efficiency with many assets, use fewer Monte Carlo scenarios
# Adjust based on problem size
n_scenarios_mc = min(2000, n_scenarios_hist)  # Use at most 2000 MC scenarios
mc_scenarios = np.random.multivariate_normal(mu, Sigma, n_scenarios_mc)

print(f"Monte Carlo scenarios: {n_scenarios_mc}")

# Combine scenarios
all_scenarios = np.vstack([historical_scenarios, mc_scenarios])
n_scenarios_total = all_scenarios.shape[0]
scenario_probs = np.ones(n_scenarios_total) / n_scenarios_total

print(f"Total scenarios: {n_scenarios_total}")
print(f"Scenario matrix shape: {all_scenarios.shape}")
print(f"Problem size: {len(selected_assets)} assets × {n_scenarios_total} scenarios = {len(selected_assets) * n_scenarios_total} scenario-asset combinations")



##
def solve_cvar_portfolio(scenarios, scenario_probs, mu, alpha=0.95, lambda_risk=1.0,
                        w_min=None, w_max=None, solver_settings=None):
    """
    Solve CVaR portfolio optimization using cuOpt linear programming.

    Parameters:
    - scenarios: numpy array of return scenarios (n_scenarios x n_assets)
    - scenario_probs: probability weights for scenarios
    - mu: expected returns vector
    - alpha: confidence level for CVaR (default 0.95)
    - lambda_risk: risk aversion parameter (default 1.0)
    - w_min, w_max: bounds on portfolio weights
    - solver_settings: cuOpt solver settings

    Returns:
    - optimal_weights: optimal portfolio weights
    - cvar_value: CVaR value at optimum
    - expected_return: expected portfolio return
    """

    n_scenarios, n_assets = scenarios.shape

    if w_min is None:
        w_min = np.zeros(n_assets)
    if w_max is None:
        w_max = np.ones(n_assets)

    # Create the linear programming problem
    problem = Problem("cvar_portfolio_optimization")

    # Decision variables
    # Portfolio weights
    w = {}
    for i in range(n_assets):
        w[i] = problem.addVariable(name=f"w_{i}", vtype=VType.CONTINUOUS,
                                  lb=w_min[i], ub=w_max[i])

    # CVaR auxiliary variables
    t = problem.addVariable(name="t", vtype=VType.CONTINUOUS,
                           lb=-float('inf'), ub=float('inf'))  # VaR variable
    u = {}
    for s in range(n_scenarios):
        u[s] = problem.addVariable(name=f"u_{s}", vtype=VType.CONTINUOUS,
                                  lb=0.0, ub=float('inf'))  # CVaR auxiliary

    # Objective: maximize expected return - lambda * CVaR
    # CVaR = t + (1/(1-alpha)) * sum(p_s * u_s)
    objective_expr = LinearExpression([], [], 0.0)

    # Add expected return terms
    for i in range(n_assets):
        if mu[i] != 0:
            objective_expr += w[i] * mu[i]

    # Subtract CVaR terms to penalize higher risk (lower CVaR increases objective value)
    if lambda_risk != 0:
        objective_expr -= t * lambda_risk
        cvar_coeff = lambda_risk / (1.0 - alpha)
        for s in range(n_scenarios):
            if scenario_probs[s] != 0:
                objective_expr -= u[s] * (cvar_coeff * scenario_probs[s])

    problem.setObjective(objective_expr, sense.MAXIMIZE)

    # Constraints
    # Budget constraint: sum of weights = 1
    budget_expr = LinearExpression([], [], 0.0)
    for i in range(n_assets):
        budget_expr += w[i]
    problem.addConstraint(budget_expr == 1.0, name="budget")

    # CVaR constraints: u_s >= -R_s^T * w - t for all scenarios s
    for s in range(n_scenarios):
        cvar_constraint_expr = LinearExpression([], [], 0.0)
        cvar_constraint_expr += u[s]  # u_s
        cvar_constraint_expr += t     # + t
        #cvar_constraint_expr += (1 - alpha)     # + t

        # Add portfolio return terms: + R_s^T * w
        for i in range(n_assets):
            if scenarios[s, i] != 0:
                cvar_constraint_expr += w[i] * scenarios[s, i]

        problem.addConstraint(cvar_constraint_expr >= 0.0, name=f"cvar_{s}")

    # Solve the optimization problem
    if solver_settings is not None:
        problem.solve(solver_settings)
    else:
        problem.solve()

    if problem.Status.name == "Optimal":
        # Extract optimal solution
        optimal_weights = np.array([w[i].getValue() for i in range(n_assets)])
        t_value = t.getValue()
        print(f"t_value is {t_value}")
        u_values = np.array([u[s].getValue() for s in range(n_scenarios)])

        # Calculate CVaR and expected return
        cvar_value1 = t_value + (1.0 - alpha) * np.sum(scenario_probs * u_values)
        cvar_value = t_value + (1.0 / (1.0 - alpha)) * np.sum(scenario_probs * u_values)
        expected_return = np.dot(mu, optimal_weights)

        return optimal_weights, cvar_value, expected_return, cvar_value1, problem
    else:
        raise RuntimeError(f"Optimization failed with status: {problem.Status.name}")



##

# Set optimization parameters
alpha = 0.95  # 95% confidence level
lambda_risk = 2.0  # Risk aversion parameter

# Portfolio weight bounds for DIVERSIFIED portfolio
w_min = np.zeros(n_assets)  # No short selling
w_max = np.ones(n_assets) # Maximum can be 100% in any single asset

print(f"Diversification constraints:")
print(f"- Maximum weight per asset: {w_max[0]:.1%}")
print(f"- This forces allocation across at least {1/w_max[0]:.0f} assets")

# Alternative diversification strategies (uncomment to try):

# Strategy 1: Even more diversified (max 10% per asset)
# w_max = np.ones(n_assets) * 0.10

# Strategy 2: Minimum holdings requirement (forces broader diversification)
# min_holdings = 30  # Require at least 30 assets
# w_min = np.zeros(n_assets)
# w_min[:min_holdings] = 0.005  # Minimum 0.5% in top assets

# Strategy 3: Lower risk aversion (allows more return-seeking behavior)
# lambda_risk = 0.5  # Less conservative approach

print(f"- Confidence level (alpha): {alpha}")
print(f"- Risk aversion (lambda): {lambda_risk}")
print(f"- Number of scenarios: {n_scenarios_total}")
print(f"- Number of assets: {n_assets}")

# Solve the optimization problem
try:
    optimal_weights, cvar_value, expected_return, cvar_value1, solve_result = solve_cvar_portfolio(
        scenarios=all_scenarios,
        scenario_probs=scenario_probs,
        mu=mu_annual,  # Use annualized returns
        alpha=alpha,
        lambda_risk=lambda_risk,
        w_min=w_min,
        w_max=w_max,
        solver_settings=solver_settings
    )
    
    print(f"\nOptimization successfuli!")
    print(f"Status: {solve_result.Status.name}")
    print(f"Objective value: {solve_result.ObjValue:.6f}")
    print(f"Expected annual return: {expected_return:.4f} ({expected_return*100:.2f}%)")
    print(f"CVaR (95%): {cvar_value:.4f}")
    print(f"VaR (95%): {cvar_value1:.4f}")
    
except Exception as e:
    print(f"Optimization failed: {e}")


# Create portfolio results DataFrame
portfolio_df = pd.DataFrame({
    'Asset': selected_assets,
    'Weight': optimal_weights,
    'Expected_Return': mu_annual
})

# Sort by weight (descending)
portfolio_df = portfolio_df.sort_values('Weight', ascending=False)

# Display portfolio composition (top holdings only)
significant_holdings = portfolio_df[portfolio_df['Weight'] > 0.001]  # Only assets with weight > 0.1%
top_holdings = significant_holdings.head(20)  # Show top 20 holdings

print("Optimal Portfolio Composition (Top 20 Holdings):")
print("=" * 70)
for _, row in top_holdings.iterrows():
    print(f"{row['Asset']:>6}: {row['Weight']:>8.4f} ({row['Weight']*100:>6.2f}%) | Expected Return: {row['Expected_Return']:>8.4f}")


# Visualize portfolio composition
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Portfolio weights bar chart (top 20 holdings)
top_20_holdings = significant_holdings.head(20)
bars = ax1.bar(range(len(top_20_holdings)), top_20_holdings['Weight'])
ax1.set_xlabel('Assets (Top 20 Holdings)')
ax1.set_ylabel('Portfolio Weight')
ax1.set_title(f'Optimal Portfolio Weights - Top 20 Holdings\n({len(selected_assets)} total assets, {len(significant_holdings)} with positive weights)')
ax1.set_xticks(range(len(top_20_holdings)))
ax1.set_xticklabels(top_20_holdings['Asset'], rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Add value labels on bars for top holdings
for i, bar in enumerate(bars):
    height = bar.get_height()
    if height > 0.01:  # Only label if weight > 1%
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Portfolio weights pie chart (top 10 holdings)
top_10_holdings = significant_holdings.head(10)
other_weight = significant_holdings.iloc[10:]['Weight'].sum() if len(significant_holdings) > 10 else 0

if other_weight > 0:
    pie_data = list(top_10_holdings['Weight']) + [other_weight]
    pie_labels = list(top_10_holdings['Asset']) + [f'Others ({len(significant_holdings)-10} assets)']
else:
    pie_data = top_10_holdings['Weight']
    pie_labels = top_10_holdings['Asset']

wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                  startangle=90, textprops={'fontsize': 9})
ax2.set_title('Portfolio Allocation - Top 10 Holdings + Others')

# Improve pie chart readability
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()

# Additional statistics
print(f"\nConcentration Analysis:")
print(f"Herfindahl-Hirschman Index (HHI): {np.sum(optimal_weights**2):.6f}")
print(f"Effective number of assets: {1/np.sum(optimal_weights**2):.2f}")
print(f"Diversification ratio: {len(significant_holdings)}/{len(selected_assets)} = {len(significant_holdings)/len(selected_assets):.2%}")



# Final summary statistics
print("CVaR Portfolio Optimization Summary")
print("=" * 50)
print(f"Dataset: S&P 500 stocks ({n_assets} assets)")
print(f"Optimization method: CVaR with cuOpt GPU acceleration")
print(f"Confidence level: {alpha*100}%")
print(f"Risk aversion parameter: {lambda_risk}")
print(f"Number of scenarios: {n_scenarios_total:,}")

if 'optimal_weights' in locals():
    portfolio_std = np.std(all_scenarios @ optimal_weights) * np.sqrt(252)
    print(f"\nOptimal Portfolio Performance:")
    print(f"- Expected annual return: {expected_return:.2%}")
    print(f"- Annual volatility: {portfolio_std:.2%}")
    print(f"- Sharpe ratio: {expected_return/portfolio_std:.3f}")
    print(f"- CVaR (95%): {cvar_value:.2%}")
    print(f"- Number of assets with positive weights: {np.sum(optimal_weights > 0.001)}")

    # Top 5 holdings
    top_5 = portfolio_df.head(5)
    print(f"\nTop 5 Holdings:")
    for _, row in top_5.iterrows():
        if row['Weight'] > 0.001:
            print(f"- {row['Asset']}: {row['Weight']:.2%}")

    print(f"\nComputational Performance:")
    print(f"- Solver status: {solve_result.Status.name}")
    print(f"- Objective value: {solve_result.ObjValue:.6f}")
else:
    print("\nOptimization was not successful - please check the previous cells.")



