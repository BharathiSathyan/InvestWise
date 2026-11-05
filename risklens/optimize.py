
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




def generate_investment_plans(
    expected_returns,
    covariance,
    current_weights,
    risk_aversion,
    max_weight,
    transaction_cost_rate=0.001,
    cash_reserve_weight=0.0,
    horizon_years=1.0,
):
    """Generate comparable total-NAV portfolio plans.

    All weights are fractions of total NAV. Cash is assumed to earn zero and
    have zero volatility. The same cash reserve target applies to all plans.

    Fresh-capital-first retains every existing stock position and invests only
    the additional amount needed to reach the equity budget.
    Full rebalance may sell/buy to reach the target equity budget.
    Cost-aware uses the same constraints but includes one-time proportional
    trading costs, amortized over the selected horizon in the annual utility.
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    current = np.asarray(current_weights, dtype=float)
    n = len(mu)

    if n == 0 or sigma.shape != (n, n) or current.shape != (n,):
        raise ValueError("Expected returns, covariance and current weights have incompatible dimensions.")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(current))):
        raise ValueError("Optimizer inputs contain non-finite values.")
    if np.any(current < -1e-9) or current.sum() > 1 + 1e-6:
        raise ValueError("Current stock weights must be non-negative and cannot exceed 100% of total portfolio value.")
    if not 0 < max_weight <= 1:
        raise ValueError("Maximum stock weight must be between 0% and 100%.")
    if not 0 <= cash_reserve_weight < 1:
        raise ValueError("Cash reserve must be between 0% and 100%.")
    if horizon_years <= 0:
        raise ValueError("Investment horizon must be positive.")
    if transaction_cost_rate < 0:
        raise ValueError("Transaction cost rate cannot be negative.")

    current = np.clip(current, 0.0, 1.0)
    equity_budget = 1.0 - float(cash_reserve_weight)
    tolerance = 1e-8

    if current.sum() > equity_budget + tolerance:
        raise ValueError(
            "The requested cash reserve is larger than the cash currently available. "
            "Reduce the reserve or choose a rebalancing strategy that sells existing holdings."
        )
    if n * max_weight < equity_budget - tolerance:
        raise ValueError(
            f"The {max_weight:.0%} stock cap is infeasible for a {equity_budget:.0%} equity target "
            f"across {n} stocks. Increase the cap to at least {equity_budget / n:.1%}."
        )

    def utility(weights, include_cost):
        ret = float(weights @ mu)
        variance = float(weights @ sigma @ weights)
        # A one-time trading cost is amortized over the selected holding horizon.
        annualized_cost_penalty = (
            transaction_cost_rate
            * float(np.abs(weights - current).sum())
            / float(horizon_years)
        ) if include_cost else 0.0
        return ret - (risk_aversion / 2.0) * variance - annualized_cost_penalty

    def feasible_start(lower, upper):
        start = np.asarray(lower, dtype=float).copy()
        remaining = equity_budget - float(start.sum())
        room = np.maximum(np.asarray(upper, dtype=float) - start, 0.0)
        if room.sum() + tolerance < remaining:
            raise ValueError("The selected stock caps cannot accommodate the required equity allocation.")
        while remaining > tolerance:
            room_sum = float(room.sum())
            if room_sum <= tolerance:
                break
            add = np.minimum(room, remaining * room / room_sum)
            start += add
            remaining -= float(add.sum())
            room = np.maximum(np.asarray(upper, dtype=float) - start, 0.0)
        return start

    
    def solve(label, lower, upper, include_cost):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        if np.any(lower > upper + tolerance):
            raise ValueError(
                f"{label}: current holdings conflict with the selected stock cap."
            )

        initial_w = feasible_start(lower, upper)

        budget_constraint = {
            "type": "eq",
            "fun": lambda x: float(np.sum(x[:n]) - equity_budget),
        }

        if include_cost and transaction_cost_rate > 0:
            # Auxiliary variables represent absolute weight changes.
            initial_u = np.abs(initial_w - current)
            initial = np.concatenate([initial_w, initial_u])

            def objective(x):
                w = x[:n]
                u = x[n:]

                ret = float(w @ mu)
                variance = float(w @ sigma @ w)
                penalty = (
                    transaction_cost_rate
                    * float(np.sum(u))
                    / float(horizon_years)
                )

                return -(
                    ret
                    - (risk_aversion / 2.0) * variance
                    - penalty
                )

            constraints = [
                budget_constraint,
                {
                    "type": "ineq",
                    "fun": lambda x: x[n:] - (x[:n] - current),
                },
                {
                    "type": "ineq",
                    "fun": lambda x: x[n:] - (current - x[:n]),
                },
            ]

            bounds = (
                list(zip(lower, upper))
                + [(0.0, None)] * n
            )

            result = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 3000, "ftol": 1e-12},
            )

            weights = result.x[:n]

        else:
            def objective(w):
                return -utility(w, False)

            result = minimize(
                objective,
                initial_w,
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints=[budget_constraint],
                options={"maxiter": 3000, "ftol": 1e-12},
            )

            weights = result.x

        if not result.success:
            raise RuntimeError(
                f"{label} optimization failed: {result.message}"
            )

        weights = np.clip(weights, lower, upper)

        drift = equity_budget - float(weights.sum())

        if abs(drift) > 1e-7:
            adjustable = (
                upper - weights if drift > 0 else weights - lower
            )

            if adjustable.sum() + tolerance < abs(drift):
                raise RuntimeError(
                    f"{label}: unable to correct numerical budget drift."
                )

            weights += (
                np.sign(drift)
                * np.minimum(
                    adjustable,
                    abs(drift) * adjustable / adjustable.sum(),
                )
            )

        gross_traded_weight = float(np.abs(weights - current).sum())
        cash_weight = max(0.0, 1.0 - float(weights.sum()))

        return {
            "weights": weights,
            "cash_weight": cash_weight,
            "return": float(weights @ mu),
            "risk": float(
                np.sqrt(max(float(weights @ sigma @ weights), 0.0))
            ),
            "utility": float(utility(weights, include_cost)),
            "gross_traded_weight": gross_traded_weight,
            "estimated_transaction_cost_rate": (
                transaction_cost_rate * gross_traded_weight
            ),
            "success": bool(result.success),
            "message": str(result.message),
        }


    # Existing holdings are hard lower bounds: no stock can be sold.
    fresh_upper = np.maximum(max_weight, current)
    fresh = solve(
        "Fresh-capital-first",
        current,
        fresh_upper,
        include_cost=False,  # no sales; buying fresh cash is the planned contribution
    )
    full = solve(
        "Full rebalance",
        np.zeros(n),
        np.full(n, max_weight),
        include_cost=False,
    )
    cost_aware = solve(
        "Transaction-cost-aware",
        np.zeros(n),
        np.full(n, max_weight),
        include_cost=True,
    )

    return {
        "fresh_capital_first": fresh,
        "full_rebalance": full,
        "cost_aware": cost_aware,
    }
