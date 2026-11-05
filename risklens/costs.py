
import numpy as np
import pandas as pd


def calculate_turnover(current_weights, new_weights):

    difference = new_weights - current_weights

    turnover = 0.5 * np.sum(np.abs(difference))

    return turnover


def calculate_transaction_cost(
    current_weights,
    new_weights,
    portfolio_value,
    cost_rate
):

    difference = new_weights - current_weights

    # Total absolute change
    total_traded_weight = np.sum(np.abs(difference))

    # Gross traded amount
    traded_amount = portfolio_value * total_traded_weight

    # Estimated transaction cost
    cost = traded_amount * cost_rate

    # One-way turnover
    turnover = 0.5 * total_traded_weight

    return {
        "turnover": turnover,
        "traded_amount": traded_amount,
        "transaction_cost": cost
    }


def calculate_buy_sell(
    current_weights,
    new_weights,
    portfolio_value
):

    difference = new_weights - current_weights

    buy_amount = portfolio_value * np.maximum(
        difference, 0
    )

    sell_amount = portfolio_value * np.maximum(
        -difference, 0
    )

    result = pd.DataFrame({
        "Current Weight": current_weights,
        "New Weight": new_weights,
        "Buy Amount": buy_amount,
        "Sell Amount": sell_amount
    })

    return result
