import streamlit as st

st.set_page_config(
    page_title="InvestWise",
    page_icon="📈",
    layout="wide"
)

st.title("📈 InvestWise")
st.subheader("Personalized Portfolio Optimization & Investment Decision Support")

st.markdown("""
Welcome to InvestWise!

Make informed investment decisions using mathematical modelling,
portfolio optimization, risk analysis and transaction-cost-aware rebalancing.
""")

st.divider()

st.header("Investor Profile")

col1, col2 = st.columns(2)

with col1:
    capital = st.number_input(
        "Investment Capital (₹)",
        min_value=1000,
        value=100000,
        step=5000
    )

    risk_profile = st.selectbox(
        "Risk Preference",
        ["Conservative", "Moderate", "Aggressive"]
    )

with col2:
    investment_horizon = st.selectbox(
        "Investment Horizon",
        ["Short-term (up to 1 year)",
         "Medium-term (1–3 years)",
         "Long-term (above 3 years)"]
    )

    target_return = st.number_input(
        "Target Annual Return (%) — Optional",
        min_value=0.0,
        max_value=100.0,
        value=12.0
    )

st.divider()

st.header("Your Portfolio")

st.info("Your existing holdings and current market values will appear here.")

st.header("Portfolio Optimization")

if st.button("Optimize My Portfolio", type="primary"):
    st.success("Investor profile captured! Optimization engine integration is coming next.")