import streamlit as st
import pandas as pd

from storage import (
    load_profile,
    save_profile,
    load_holdings,
    save_holdings,
    holdings_to_dataframe
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="InvestWise",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# LOAD SAVED INFORMATION
# --------------------------------------------------

if "profile" not in st.session_state:
    st.session_state.profile = load_profile()

if "holdings" not in st.session_state:
    st.session_state.holdings = load_holdings()


profile = st.session_state.profile


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📈 InvestWise")

st.subheader(
    "Personalized Portfolio Optimization & Investment Decision Support"
)

st.markdown(
    "Make informed investment decisions using mathematical "
    "modelling, portfolio optimization, risk analysis "
    "and transaction-cost-aware rebalancing."
)

st.divider()


# --------------------------------------------------
# INVESTOR PROFILE
# --------------------------------------------------

st.header("1. Investor Profile")

with st.form("investor_profile_form"):

    col1, col2 = st.columns(2)

    with col1:

        capital = st.number_input(
            "Investment Capital (₹)",
            min_value=1000.0,
            value=float(profile["capital"]),
            step=5000.0
        )

        risk_profile = st.selectbox(
            "Risk Preference",
            ["Conservative", "Moderate", "Aggressive"],
            index=[
                "Conservative",
                "Moderate",
                "Aggressive"
            ].index(profile["risk_profile"])
        )

        max_weight_percent =  st.slider(
            "Maximum Allocation per Stock (%)",
            min_value=10,
            max_value=100,
            value=int(float(profile["max_weight"]) * 100),
            step=5
        )

    max_weight = max_weight_percent / 100

    with col2:

        horizons = [
            "Short-term (up to 1 year)",
            "Medium-term (1–3 years)",
            "Long-term (above 3 years)"
        ]

        investment_horizon = st.selectbox(
            "Investment Horizon",
            horizons,
            index=horizons.index(profile["investment_horizon"])
        )

        target_return = st.number_input(
            "Target Annual Return (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(profile["target_return"]),
            step=1.0
        )

    profile_submitted = st.form_submit_button(
        "Save Investor Profile",
        type="primary"
    )

    if profile_submitted:

        updated_profile = {
            "profile_id": "default_user",
            "capital": capital,
            "risk_profile": risk_profile,
            "investment_horizon": investment_horizon,
            "target_return": target_return,
            "max_weight": max_weight
        }

        save_profile(updated_profile)

        st.session_state.profile = updated_profile

        st.success("Investor profile saved successfully!")


st.divider()


# --------------------------------------------------
# EXISTING PORTFOLIO
# --------------------------------------------------

st.header("2. Existing Portfolio")

st.write(
    "If you already own stocks, enter your holdings below. "
    "If you are a new investor, leave the table empty."
)

st.caption(
    "Enter the NSE ticker in the same format used by our market-data engine."
)

current_holdings_df = holdings_to_dataframe(
    st.session_state.holdings
)

edited_holdings = st.data_editor(
    current_holdings_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn(
            "Stock Ticker",
            help="Example: TCS.NS",
            required=True
        ),
        "Quantity": st.column_config.NumberColumn(
            "Quantity",
            min_value=0,
            step=1
        ),
        "Average Purchase Price": st.column_config.NumberColumn(
            "Average Purchase Price (₹)",
            min_value=0.0,
            format="₹%.2f"
        )
    },
    key="holdings_editor"
)


if st.button("Save Existing Holdings"):

    cleaned_holdings = []

    for _, row in edited_holdings.iterrows():

        ticker = str(row["Ticker"]).strip().upper()

        if not ticker or ticker == "NAN":
            continue

        quantity = row["Quantity"]
        purchase_price = row["Average Purchase Price"]

        if pd.isna(quantity) or quantity <= 0:
            st.error(f"Invalid quantity for {ticker}.")
            st.stop()

        if pd.isna(purchase_price):
            purchase_price = None

        elif purchase_price < 0:
            st.error(f"Invalid purchase price for {ticker}.")
            st.stop()

        cleaned_holdings.append({
            "Ticker": ticker,
            "Quantity": int(quantity),
            "Average Purchase Price": (
                float(purchase_price)
                if purchase_price is not None
                else None
            )
        })

    save_holdings(cleaned_holdings)

    st.session_state.holdings = cleaned_holdings

    st.success("Existing holdings saved successfully!")


st.divider()


# --------------------------------------------------
# PORTFOLIO SUMMARY
# --------------------------------------------------

st.header("3. Portfolio Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Investment Capital",
        f"₹{st.session_state.profile['capital']:,.2f}"
    )

with col2:
    st.metric(
        "Risk Preference",
        st.session_state.profile["risk_profile"]
    )

with col3:
    st.metric(
        "Existing Stocks",
        len(st.session_state.holdings)
    )

if not st.session_state.holdings:
    st.info(
        "You are starting with fresh investment capital. "
        "No existing holdings have been added."
    )
else:
    st.dataframe(
        pd.DataFrame(st.session_state.holdings),
        use_container_width=True
    )


st.divider()


# --------------------------------------------------
# OPTIMIZATION PLACEHOLDER
# --------------------------------------------------

st.header("4. Portfolio Optimization")

st.info(
    "The investor profile and existing holdings are saved. "
    "The next stage will connect the mathematical optimization engine."
)