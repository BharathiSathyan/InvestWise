
import numpy as np

from scipy.optimize import minimize


RISK_PROFILES = {
    "Conservative": 10,
    "Moderate": 5,
    "Aggressive": 2
}


def equal_weight_portfolio(n):

    weights = np.ones(n) / n

    return weights


def calculate_portfolio_return(weights, mu):

    return float(weights @ mu)


def calculate_portfolio_risk(weights, covariance):

    variance = weights @ covariance @ weights

    return float(np.sqrt(max(variance, 0)))


def calculate_sharpe(weights, mu, covariance, rf=0.06):

    portfolio_return = calculate_portfolio_return(
        weights, mu
    )

    portfolio_risk = calculate_portfolio_risk(
        weights, covariance
    )

    if portfolio_risk == 0:
        return 0

    return (portfolio_return - rf) / portfolio_risk


def optimize_portfolio(
    expected_returns,
    covariance,
    current_weights,
    risk_aversion,
    max_weight,
    transaction_cost_rate=0
):

    mu = np.asarray(expected_returns, dtype=float)

    Sigma = np.asarray(covariance, dtype=float)

    current = np.asarray(current_weights, dtype=float)

    n = len(mu)

    # Check input dimensions
    if Sigma.shape != (n, n):
        raise ValueError("Invalid covariance matrix shape")

    if len(current) != n:
        raise ValueError("Current weights have incorrect length")

    if not np.all(np.isfinite(mu)):
        raise ValueError("Expected returns contain invalid values")

    if not np.all(np.isfinite(Sigma)):
        raise ValueError("Covariance contains invalid values")

    if max_weight <= 0 or n * max_weight < 1:
        raise ValueError("Maximum weight makes allocation infeasible")

    # Initial equal-weight portfolio
    initial = np.ones(n) / n

    # Objective function
    def objective(weights):

        portfolio_return = weights @ mu

        portfolio_variance = weights @ Sigma @ weights

        transaction_penalty = (
            transaction_cost_rate
            * np.sum(np.abs(weights - current))
        )

        utility = (
            portfolio_return
            - (risk_aversion / 2) * portfolio_variance
            - transaction_penalty
        )

        # Minimize negative utility
        return -utility

    # Budget constraint
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        }
    ]

    # Individual stock weight limits
    bounds = [(0, max_weight)] * n

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10
        }
    )

    if not result.success:
        raise RuntimeError(
            "Optimization failed: " + result.message
        )

    weights = result.x

    # Remove tiny numerical errors
    weights[np.abs(weights) < 1e-10] = 0

    portfolio_return = calculate_portfolio_return(
        weights, mu
    )

    portfolio_risk = calculate_portfolio_risk(
        weights, Sigma
    )

    sharpe = calculate_sharpe(
        weights, mu, Sigma
    )

    return {
        "weights": weights,
        "return": portfolio_return,
        "risk": portfolio_risk,
        "sharpe": sharpe,
        "success": result.success,
        "message": result.message
    }


def generate_portfolios(
    expected_returns,
    covariance,
    current_weights,
    risk_aversion,
    max_weight,
    transaction_cost_rate
):

    n = len(expected_returns)

    # 1. Equal-weight
    equal_weights = equal_weight_portfolio(n)

    # 2. Traditional Markowitz
    markowitz = optimize_portfolio(
        expected_returns,
        covariance,
        current_weights,
        risk_aversion,
        max_weight,
        transaction_cost_rate=0
    )

    # 3. Transaction-cost-aware
    cost_aware = optimize_portfolio(
        expected_returns,
        covariance,
        current_weights,
        risk_aversion,
        max_weight,
        transaction_cost_rate
    )

    return {
        "equal_weight": equal_weights,
        "markowitz": markowitz,
        "cost_aware": cost_aware
    }
