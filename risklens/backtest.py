
import numpy as np
import pandas as pd

from risklens.config import (
    TICKERS,
    BENCHMARK,
    TRADING_DAYS
)

from risklens.optimize import (
    optimize_portfolio,
    RISK_PROFILES
)

from risklens.costs import (
    calculate_transaction_cost
)


def calculate_performance(nav, daily_returns, rf=0.06):

    # Total return
    total_return = nav.iloc[-1] / nav.iloc[0] - 1

    # Number of daily observations
    days = len(daily_returns)

    # Annualized return
    annual_return = (
        (nav.iloc[-1] / nav.iloc[0])
        ** (TRADING_DAYS / days)
    ) - 1

    # Annualized volatility
    annual_volatility = (
        daily_returns.std() * np.sqrt(TRADING_DAYS)
    )

    # Sharpe ratio
    daily_rf = rf / TRADING_DAYS

    sharpe = (
        (daily_returns.mean() - daily_rf)
        / daily_returns.std()
    ) * np.sqrt(TRADING_DAYS)

    # Drawdown
    running_peak = nav.cummax()

    drawdown = nav / running_peak - 1

    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
    }


def run_backtest(
    prices,
    initial_capital=500000,
    risk_profile="Moderate",
    max_weight=0.30,
    transaction_cost_rate=0.001,
    estimation_window=252
):

    # Calculate stock and benchmark daily returns
    returns = prices.pct_change().dropna()

    # Initial portfolio: equal weight
    n = len(TICKERS)

    equal_weights = np.ones(n) / n

    # Three strategies
    strategies = [
        "equal_weight",
        "markowitz",
        "cost_aware"
    ]

    nav = {
        name: initial_capital
        for name in strategies
    }

    weights = {
        name: equal_weights.copy()
        for name in strategies
    }

    cumulative_cost = {
        name: 0.0
        for name in strategies
    }

    total_turnover = {
        name: 0.0
        for name in strategies
    }

    nav_history = {
        name: []
        for name in strategies
    }

    cost_history = {
        name: []
        for name in strategies
    }

    turnover_history = {
        name: []
        for name in strategies
    }

    # Benchmark NAV
    benchmark_nav = initial_capital

    benchmark_history = []

    # Risk preference
    gamma = RISK_PROFILES[risk_profile]

    # Out-of-sample dates
    test_returns = returns.loc["2024-01-01":"2025-12-31"]

    # Find first trading day of each quarter
    quarter_groups = test_returns.groupby(
        test_returns.index.to_period("Q")
    )

    rebalance_dates = [
        group.index[0]
        for _, group in quarter_groups
    ]

    for rebalance_date in rebalance_dates:

        # Historical observations strictly before rebalance
        historical = returns.loc[
            returns.index < rebalance_date
        ].tail(estimation_window)

        if len(historical) < estimation_window:
            continue

        stock_returns = historical[TICKERS]

        # Annualized expected return
        mu = stock_returns.mean() * TRADING_DAYS

        # Annualized sample covariance
        covariance = (
            stock_returns.cov() * TRADING_DAYS
        )

        # Returns available during this quarter
        next_rebalance_dates = [
            d for d in rebalance_dates
            if d > rebalance_date
        ]

        if next_rebalance_dates:

            next_date = next_rebalance_dates[0]

            period_returns = test_returns.loc[
                (test_returns.index >= rebalance_date)
                & (test_returns.index < next_date)
            ]

        else:

            period_returns = test_returns.loc[
                test_returns.index >= rebalance_date
            ]

        # Rebalance all three strategies
        for strategy in strategies:

            current = weights[strategy]

            if strategy == "equal_weight":

                target = equal_weights.copy()

            elif strategy == "markowitz":

                result = optimize_portfolio(
                    mu.values,
                    covariance.values,
                    current,
                    gamma,
                    max_weight,
                    transaction_cost_rate=0
                )

                target = result["weights"]

            else:

                result = optimize_portfolio(
                    mu.values,
                    covariance.values,
                    current,
                    gamma,
                    max_weight,
                    transaction_cost_rate
                )

                target = result["weights"]

            # Calculate actual rebalance cost
            costs = calculate_transaction_cost(
                current,
                target,
                nav[strategy],
                transaction_cost_rate
            )

            cost = costs["transaction_cost"]

            turnover = costs["turnover"]

            # Deduct transaction cost ONCE
            nav[strategy] -= cost

            cumulative_cost[strategy] += cost

            total_turnover[strategy] += turnover

            cost_history[strategy].append(cost)

            turnover_history[strategy].append(turnover)

            # Update target weights
            weights[strategy] = target.copy()

        # Apply daily returns after rebalancing
        for date, daily_return in period_returns.iterrows():

            for strategy in strategies:

                current_weights = weights[strategy]

                asset_returns = daily_return[
                    TICKERS
                ].values

                portfolio_return = (
                    current_weights @ asset_returns
                )

                # Update NAV
                nav[strategy] *= (
                    1 + portfolio_return
                )

                # Allow weights to drift with returns
                new_weights = (
                    current_weights
                    * (1 + asset_returns)
                )

                weights[strategy] = (
                    new_weights / new_weights.sum()
                )

                nav_history[strategy].append(
                    (date, nav[strategy])
                )

            # Benchmark performance
            benchmark_nav *= (
                1 + daily_return[BENCHMARK]
            )

            benchmark_history.append(
                (date, benchmark_nav)
            )

    # Construct NAV DataFrame
    nav_df = pd.DataFrame({
        name: pd.Series(
            dict(nav_history[name])
        )
        for name in strategies
    })

    benchmark_series = pd.Series(
        dict(benchmark_history),
        name="NIFTY50"
    )

    nav_df = nav_df.join(benchmark_series)

    # Calculate daily portfolio returns
    daily_nav_returns = nav_df.pct_change().dropna()

    # Calculate performance
    performance = {}

    for strategy in strategies:

        strategy_nav = nav_df[strategy].dropna()

        strategy_returns = (
            strategy_nav.pct_change().dropna()
        )

        performance[strategy] = (
            calculate_performance(
                strategy_nav,
                strategy_returns
            )
        )

        performance[strategy]["transaction_cost"] = (
            cumulative_cost[strategy]
        )

        performance[strategy]["turnover"] = (
            total_turnover[strategy]
        )

    # Benchmark performance
    benchmark_returns = (
        benchmark_series.pct_change().dropna()
    )

    performance["NIFTY50"] = calculate_performance(
        benchmark_series.dropna(),
        benchmark_returns
    )

    return {
        "nav": nav_df,
        "performance": performance
    }
