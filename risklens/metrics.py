
import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf

from risklens.config import TICKERS, BENCHMARK, TRADING_DAYS


RF = 0.06


def calculate_daily_returns(prices):

    # Simple daily percentage returns
    returns = prices.pct_change().dropna()

    return returns


def calculate_expected_returns(returns):

    # Separate stock returns from market returns
    stocks = returns[TICKERS]

    # Annualized arithmetic mean
    mu = stocks.mean() * TRADING_DAYS

    return mu


def calculate_sample_covariance(returns):

    stocks = returns[TICKERS]

    # Annualized sample covariance
    covariance = stocks.cov() * TRADING_DAYS

    return covariance


def calculate_shrinkage_covariance(returns):

    stocks = returns[TICKERS]

    # Ledoit-Wolf covariance estimator
    model = LedoitWolf()

    model.fit(stocks)

    # Annualize daily covariance
    covariance = model.covariance_ * TRADING_DAYS

    covariance = pd.DataFrame(
        covariance,
        index=TICKERS,
        columns=TICKERS
    )

    return covariance


def calculate_volatility(covariance):

    # Standard deviation from covariance diagonal
    volatility = np.sqrt(np.diag(covariance))

    result = pd.Series(
        volatility,
        index=TICKERS
    )

    return result


def calculate_correlation(returns):

    stocks = returns[TICKERS]

    correlation = stocks.corr()

    return correlation


def calculate_beta(returns):

    market = returns[BENCHMARK]

    market_variance = market.var()

    beta = {}

    for stock in TICKERS:

        stock_returns = returns[stock]

        covariance = stock_returns.cov(market)

        beta[stock] = covariance / market_variance

    return pd.Series(beta)


def calculate_stock_sharpe(mu, volatility):

    sharpe = (mu - RF) / volatility

    return sharpe


def calculate_metrics(prices):

    # 1. Calculate daily returns
    returns = calculate_daily_returns(prices)

    # 2. Expected returns
    mu = calculate_expected_returns(returns)

    # 3. Sample covariance
    sample_cov = calculate_sample_covariance(returns)

    # 4. Shrinkage covariance
    shrink_cov = calculate_shrinkage_covariance(returns)

    # 5. Individual volatility
    volatility = calculate_volatility(sample_cov)

    # 6. Correlation
    correlation = calculate_correlation(returns)

    # 7. Beta
    beta = calculate_beta(returns)

    # 8. Individual stock Sharpe
    sharpe = calculate_stock_sharpe(mu, volatility)

    results = {
        "returns": returns,
        "expected_returns": mu,
        "sample_covariance": sample_cov,
        "shrinkage_covariance": shrink_cov,
        "volatility": volatility,
        "correlation": correlation,
        "beta": beta,
        "stock_sharpe": sharpe
    }

    return results
