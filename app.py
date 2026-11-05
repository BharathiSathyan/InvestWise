import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from storage import load_profile, save_profile, load_holdings, save_holdings
from risklens.config import TICKERS, BENCHMARK, TRADING_DAYS
from risklens.data import get_aligned_prices, refresh_market_data
from risklens.metrics import calculate_metrics
from risklens.optimize import generate_portfolios, RISK_PROFILES
from risklens.costs import calculate_transaction_cost, calculate_buy_sell
from risklens.backtest import run_backtest


# ============================================================
# PAGE SETUP AND COMMON HELPERS
# ============================================================
st.set_page_config(
    page_title="InvestWise | Portfolio Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px; border-radius: 10px;}
      div[data-testid="stTabs"] button {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_PROFILE = {
    "profile_id": "default_user",
    "capital": 100000.0,
    "risk_profile": "Moderate",
    "investment_horizon": "Long-term (above 3 years)",
    "target_return": 12.0,
    "max_weight": 0.30,
}

if "profile" not in st.session_state:
    st.session_state.profile = {**DEFAULT_PROFILE, **load_profile()}
if "holdings" not in st.session_state:
    st.session_state.holdings = load_holdings()


def money(value):
    return f"₹{value:,.2f}"


def load_prices_cached():
    return get_aligned_prices()


def build_holdings_valuation(holdings, latest_prices):
    rows = []
    values = pd.Series(0.0, index=TICKERS, dtype=float)
    total_known_cost = 0.0
    total_known_market = 0.0

    for holding in holdings:
        ticker = holding.get("Ticker")
        qty = int(holding.get("Quantity", 0))
        purchase = holding.get("Average Purchase Price")
        if ticker not in TICKERS or ticker not in latest_prices.index or qty <= 0:
            continue

        current = float(latest_prices[ticker])
        market_value = qty * current
        invested = qty * float(purchase) if purchase is not None and pd.notna(purchase) else np.nan
        pnl = market_value - invested if pd.notna(invested) else np.nan
        pnl_pct = (pnl / invested * 100) if pd.notna(invested) and invested > 0 else np.nan
        values.loc[ticker] += market_value

        if pd.notna(invested):
            total_known_cost += invested
            total_known_market += market_value

        rows.append({
            "Stock": ticker,
            "Quantity": qty,
            "Average Purchase Price (₹)": purchase,
            "Latest Price (₹)": current,
            "Invested Value (₹)": invested,
            "Current Market Value (₹)": market_value,
            "Unrealized P&L (₹)": pnl,
            "P&L (%)": pnl_pct,
        })

    return pd.DataFrame(rows), values, total_known_cost, total_known_market


def portfolio_risk_metrics(stock_weights, metrics):
    mu = metrics["expected_returns"].reindex(TICKERS).fillna(0.0)
    cov = metrics["sample_covariance"].reindex(index=TICKERS, columns=TICKERS)
    beta = metrics["beta"].reindex(TICKERS).fillna(0.0)
    w = stock_weights.reindex(TICKERS).fillna(0.0).to_numpy(dtype=float)
    expected_return = float(w @ mu.to_numpy(dtype=float))
    variance = float(w @ cov.to_numpy(dtype=float) @ w)
    volatility = float(np.sqrt(max(variance, 0.0)))
    portfolio_beta = float(w @ beta.to_numpy(dtype=float))
    sharpe = (expected_return - 0.06) / volatility if volatility > 0 else np.nan
    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "beta": portfolio_beta,
        "sharpe": sharpe,
    }


def make_portfolio_history(prices, holdings, fresh_capital):
    """Historical value of today's share quantities plus constant uninvested cash.

    This is a hypothetical fixed-quantity reconstruction, not a broker account
    statement or a record of when the user actually bought each share.
    """
    if not holdings:
        return None
    quantity_by_ticker = pd.Series(0.0, index=TICKERS)
    for holding in holdings:
        ticker = holding.get("Ticker")
        if ticker in TICKERS:
            quantity_by_ticker.loc[ticker] += float(holding.get("Quantity", 0))
    selected = prices[TICKERS]
    historical_market_value = selected.mul(quantity_by_ticker, axis=1).sum(axis=1)
    return historical_market_value + float(fresh_capital)


# ============================================================
# SIDEBAR: MARKET DATA AND GLOBAL CONTEXT
# ============================================================
st.sidebar.title("📈 InvestWise")
st.sidebar.caption("Portfolio intelligence for Indian equities")

if st.sidebar.button("↻ Refresh market data", width="stretch"):
    with st.spinner("Fetching latest available market data..."):
        try:
            _, refresh_status = refresh_market_data(force_refresh=True)
            if refresh_status.get("updated"):
                st.sidebar.success(refresh_status.get("message", "Refresh completed."))
            else:
                st.sidebar.warning(refresh_status.get("message", "Using existing local data."))
            st.sidebar.caption(f"Latest date: {refresh_status.get('last_date', 'Unknown')}")
        except Exception as exc:
            st.sidebar.error(f"Refresh failed: {exc}")

try:
    prices = load_prices_cached()
    prices = prices.sort_index().dropna(subset=TICKERS + [BENCHMARK])
    latest_prices = prices.iloc[-1]
    latest_date = prices.index[-1]
    data_error = None
except Exception as exc:
    prices = None
    latest_prices = None
    latest_date = None
    data_error = str(exc)

profile = st.session_state.profile
holdings = st.session_state.holdings
fresh_capital = float(profile.get("capital", 100000.0))

st.title("InvestWise")
st.caption("Personalized portfolio optimization, risk analytics and transaction-cost-aware investment decisions")
if data_error:
    st.error(f"Market data could not be loaded: {data_error}")
else:
    st.sidebar.success("Market data loaded from local cache")
    st.sidebar.caption(f"Latest available observation: {latest_date:%d %b %Y}")
    st.sidebar.caption(f"{len(prices):,} aligned trading observations · {len(TICKERS)} equities + NIFTY 50")

# Three simple investor-facing sections; advanced tools are tucked into expanders.
tabs = st.tabs([
    "🏠 Home",
    "💼 My Investments",
    "🧭 Investment Planner",
])


# ============================================================
# PREPARE SHARED CURRENT PORTFOLIO VALUES
# ============================================================
if prices is not None:
    holdings_table, holding_values, known_cost, known_market = build_holdings_valuation(holdings, latest_prices)
else:
    holdings_table = pd.DataFrame()
    holding_values = pd.Series(0.0, index=TICKERS)
    known_cost = 0.0
    known_market = 0.0

existing_market_value = float(holding_values.sum())
total_portfolio_value = existing_market_value + fresh_capital
current_weights = holding_values / total_portfolio_value if total_portfolio_value > 0 else holding_values.copy()
current_cash_weight = fresh_capital / total_portfolio_value if total_portfolio_value > 0 else 1.0

# ============================================================
# HOME
# ============================================================
with tabs[0]:
    st.subheader("Portfolio at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fresh capital", money(fresh_capital))
    c2.metric("Existing holdings", money(existing_market_value))
    c3.metric("Total portfolio value", money(total_portfolio_value))
    pnl_known = known_market - known_cost
    c4.metric("Known unrealized P&L", money(pnl_known), delta=f"{(pnl_known / known_cost * 100):.2f}%" if known_cost > 0 else None)
    st.caption("Unrealized P&L includes only holdings with an entered average purchase price. Values use the latest available cached market observation, not a streaming quote.")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("#### Portfolio composition")
        composition = pd.DataFrame({"Asset": list(TICKERS) + ["Fresh capital (cash)"], "Value": list(holding_values.values) + [fresh_capital]})
        composition = composition[composition["Value"] > 0]
        if not composition.empty:
            fig = px.pie(composition, names="Asset", values="Value", hole=0.55)
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), legend_title="")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No portfolio values to display yet.")
    with right:
        st.markdown("#### Investor profile")
        st.write(f"**Risk preference:** {profile.get('risk_profile', 'Moderate')}")
        st.write(f"**Investment horizon:** {profile.get('investment_horizon', 'Not set')}")
        st.write(f"**Target annual return:** {float(profile.get('target_return', 0)):.1f}%")
        st.write(f"**Maximum stock allocation:** {float(profile.get('max_weight', .30)):.0%}")
        st.write(f"**Existing stock positions:** {len(holdings)}")
        st.write(f"**Market data date:** {latest_date:%d %B %Y}" if latest_date is not None else "**Market data date:** unavailable")

    st.markdown("#### Latest holdings")
    if not holdings_table.empty:
        st.dataframe(holdings_table, width="stretch", hide_index=True)
    else:
        st.info("You have no existing stock holdings. Your fresh capital is ready for the optimization stage.")


    # ============================================================

    with st.expander("Advanced analytics", expanded=False):
        st.subheader("Portfolio Analytics")
        st.caption("Historical and statistical analysis based on the cached price history.")
        if prices is None:
            st.warning("Load market data to view analytics.")
        else:
            with st.spinner("Calculating return, risk and correlation statistics..."):
                metrics = calculate_metrics(prices)

            mu = metrics["expected_returns"].reindex(TICKERS)
            vol = metrics["volatility"].reindex(TICKERS)
            beta = metrics["beta"].reindex(TICKERS)
            stock_sharpe = metrics["stock_sharpe"].reindex(TICKERS)

            st.markdown("#### Stock universe: risk versus return")
            risk_return = pd.DataFrame({
                "Stock": TICKERS,
                "Expected annual return (%)": mu.values * 100,
                "Annual volatility (%)": vol.values * 100,
                "Beta": beta.values,
                "Sharpe ratio": stock_sharpe.values,
            })
            fig_scatter = px.scatter(risk_return, x="Annual volatility (%)", y="Expected annual return (%)", text="Stock", size="Beta", color="Sharpe ratio", hover_data=["Beta"])
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_scatter, width="stretch")

            st.markdown("#### Historical normalized price performance")
            selected_assets = st.multiselect("Choose stocks / benchmark", TICKERS + [BENCHMARK], default=["TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", BENCHMARK])
            if selected_assets:
                normalized = prices[selected_assets].div(prices[selected_assets].iloc[0]).mul(100)
                fig_norm = px.line(normalized, x=normalized.index, y=selected_assets, labels={"value": "Growth of ₹100 (index)", "Date": "Date", "variable": "Asset"})
                st.plotly_chart(fig_norm, width="stretch")

            st.markdown("#### Return correlation heatmap")
            corr = metrics["correlation"].reindex(index=TICKERS, columns=TICKERS)
            fig_corr = px.imshow(corr, x=TICKERS, y=TICKERS, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto")
            fig_corr.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_corr, width="stretch")

            st.markdown("#### Individual stock metrics")
            st.dataframe(risk_return.sort_values("Sharpe ratio", ascending=False), width="stretch", hide_index=True)

            if existing_market_value > 0:
                st.markdown("#### Current holdings: historical fixed-quantity illustration")
                history = make_portfolio_history(prices, holdings, fresh_capital)
                if history is not None:
                    hist_df = pd.DataFrame({"Portfolio value (₹)": history})
                    fig_hist = px.line(hist_df, x=hist_df.index, y="Portfolio value (₹)")
                    st.plotly_chart(fig_hist, width="stretch")
                    daily_hist_returns = history.pct_change().dropna()
                    if not daily_hist_returns.empty:
                        rolling_vol = daily_hist_returns.rolling(63).std() * np.sqrt(TRADING_DAYS) * 100
                        fig_vol = px.line(rolling_vol, x=rolling_vol.index, y=rolling_vol.values, labels={"x": "Date", "y": "Rolling 63-day annualized volatility (%)"}, title="Rolling 63-trading-day volatility")
                        st.plotly_chart(fig_vol, width="stretch")

                        historical_drawdown = history / history.cummax() - 1
                        fig_portfolio_dd = px.area(
                            historical_drawdown,
                            x=historical_drawdown.index,
                            y=historical_drawdown.values,
                            labels={"x": "Date", "y": "Drawdown"},
                            title="Historical drawdown of the illustrative holdings basket",
                        )
                        fig_portfolio_dd.update_yaxes(tickformat=".0%")
                        st.plotly_chart(fig_portfolio_dd, width="stretch")
                    st.caption("This chart reconstructs the historical value of today's share quantities plus constant fresh cash. It is a hypothetical fixed-quantity illustration, not the user's actual historical account performance.")

            if existing_market_value > 0:
                current_risk = portfolio_risk_metrics(current_weights, metrics)
                st.markdown("#### Existing portfolio risk snapshot (cash assumed risk-free)")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Estimated annual return", f"{current_risk['expected_return']:.2%}")
                r2.metric("Estimated annual volatility", f"{current_risk['volatility']:.2%}")
                r3.metric("Portfolio beta", f"{current_risk['beta']:.2f}")
                r4.metric("Sharpe ratio", f"{current_risk['sharpe']:.2f}" if pd.notna(current_risk["sharpe"]) else "N/A")
                st.caption("These are model estimates using the full selected historical sample, 6% risk-free rate and zero-return/zero-risk cash simplification. They are not forecasts.")


        # ============================================================

# ============================================================
# MY INVESTMENTS
# ============================================================
with tabs[1]:
    st.subheader("My Portfolio")
    st.caption("Actual holdings are kept separate from recommended allocations.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Market value of holdings", money(existing_market_value))
    k2.metric("Fresh cash available", money(fresh_capital))
    k3.metric("Combined portfolio value", money(total_portfolio_value))

    if not holdings_table.empty:
        st.markdown("#### Holdings and valuation")
        st.dataframe(holdings_table, width="stretch", hide_index=True,
                     column_config={
                         "Average Purchase Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "Latest Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "Invested Value (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "Current Market Value (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "Unrealized P&L (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "P&L (%)": st.column_config.NumberColumn(format="%.2f%%"),
                     })

        alloc = pd.DataFrame({"Stock": TICKERS, "Market Value (₹)": holding_values.values})
        alloc = alloc[alloc["Market Value (₹)"] > 0].copy()
        alloc["Portfolio Weight (%)"] = alloc["Market Value (₹)"] / total_portfolio_value * 100
        st.markdown("#### Stock-wise allocation")
        st.dataframe(alloc.sort_values("Portfolio Weight (%)", ascending=False), width="stretch", hide_index=True)
        fig_alloc = px.bar(alloc.sort_values("Portfolio Weight (%)", ascending=True), x="Portfolio Weight (%)", y="Stock", orientation="h", text="Portfolio Weight (%)")
        fig_alloc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_alloc.update_layout(xaxis_title="Share of total portfolio value", yaxis_title="", margin=dict(l=10, r=30, t=20, b=10))
        st.plotly_chart(fig_alloc, width="stretch")

        if known_cost > 0:
            st.markdown("#### Unrealized profit / loss")
            pnl_view = holdings_table.dropna(subset=["Unrealized P&L (₹)"])[["Stock", "Unrealized P&L (₹)", "P&L (%)"]]
            if not pnl_view.empty:
                fig_pnl = px.bar(pnl_view, x="Stock", y="Unrealized P&L (₹)", color="Unrealized P&L (₹)", color_continuous_midpoint=0)
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_pnl, width="stretch")
    else:
        st.info("No existing holdings yet. Add holdings in Investor Settings. This is a valid new-investor portfolio.")

    st.markdown("#### Market data provenance")
    st.write(f"Latest available trading date in local cache: **{latest_date:%d %b %Y}**" if latest_date is not None else "Market data unavailable.")
    st.caption("Yahoo Finance historical adjusted-close data is used as a prototype data source. It may be delayed, revised or unavailable; it is not an executable broker quote.")


    # ============================================================

    with st.expander("Profile & settings", expanded=False):
        st.subheader("Investor Settings")
        st.markdown("#### Investment preferences")
        with st.form("settings_profile_form"):
            c1, c2 = st.columns(2)
            with c1:
                setting_capital = st.number_input("Additional fresh investment capital (₹)", min_value=1000.0, value=float(profile.get("capital", 100000)), step=5000.0)
                setting_risk = st.selectbox("Risk preference", list(RISK_PROFILES.keys()), index=list(RISK_PROFILES.keys()).index(profile.get("risk_profile", "Moderate")))
                setting_max = st.slider("Maximum weight per stock (%)", 10, 100, int(float(profile.get("max_weight", .30)) * 100), 5)
            with c2:
                horizon_options = ["Short-term (up to 1 year)", "Medium-term (1–3 years)", "Long-term (above 3 years)"]
                setting_horizon = st.selectbox("Investment horizon", horizon_options, index=horizon_options.index(profile.get("investment_horizon", horizon_options[2])))
                setting_target = st.number_input("Target annual return (%)", min_value=0.0, max_value=100.0, value=float(profile.get("target_return", 12.0)), step=1.0)
            if st.form_submit_button("Save investor settings", type="primary"):
                updated = {
                    "profile_id": profile.get("profile_id", "default_user"),
                    "capital": setting_capital,
                    "risk_profile": setting_risk,
                    "investment_horizon": setting_horizon,
                    "target_return": setting_target,
                    "max_weight": setting_max / 100,
                }
                save_profile(updated)
                st.session_state.profile = updated
                st.success("Investor settings saved. Rerun the app to refresh all dependent calculations.")

        st.divider()
        st.markdown("#### Manage existing holdings")
        holdings_editor_df = pd.DataFrame(st.session_state.holdings, columns=["Ticker", "Quantity", "Average Purchase Price"])
        if holdings_editor_df.empty:
            holdings_editor_df = pd.DataFrame(columns=["Ticker", "Quantity", "Average Purchase Price"])
        edited = st.data_editor(
            holdings_editor_df,
            num_rows="dynamic",
            width="stretch",
            key="settings_holdings_editor",
            column_config={
                "Ticker": st.column_config.SelectboxColumn("Stock ticker", options=TICKERS, required=True),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1),
                "Average Purchase Price": st.column_config.NumberColumn("Average purchase price (₹)", min_value=0.0, format="₹%.2f"),
            },
        )
        if st.button("Save holdings", key="save_settings_holdings"):
            clean = []
            errors = []
            for _, row in edited.iterrows():
                ticker = row.get("Ticker")
                if pd.isna(ticker) or not str(ticker).strip():
                    continue
                ticker = str(ticker).strip().upper()
                qty = row.get("Quantity")
                purchase = row.get("Average Purchase Price")
                if ticker not in TICKERS:
                    errors.append(f"Unsupported ticker: {ticker}")
                    continue
                if pd.isna(qty) or qty <= 0 or not float(qty).is_integer():
                    errors.append(f"Enter a positive whole-share quantity for {ticker}")
                    continue
                if pd.notna(purchase) and purchase < 0:
                    errors.append(f"Invalid purchase price for {ticker}")
                    continue
                clean.append({"Ticker": ticker, "Quantity": int(qty), "Average Purchase Price": float(purchase) if pd.notna(purchase) else None})
            if errors:
                for err in errors:
                    st.error(err)
            else:
                save_holdings(clean)
                st.session_state.holdings = clean
                st.success("Holdings saved. Rerun the app to refresh portfolio calculations.")

        st.divider()
        st.markdown("#### Data and model assumptions")
        st.write(f"Supported universe: {', '.join(TICKERS)}")
        st.write(f"Benchmark: {BENCHMARK}")
        st.write(f"Risk-free rate used by the supplied engine: 6% annually")
        st.write("Cash is treated as zero-return and zero-risk in the current portfolio snapshot. The optimizer itself allocates 100% across the supported equities.")
        st.write("Market prices are sourced from the local CSV cache populated by Yahoo Finance historical data downloads.")

        st.divider()

# ============================================================
# INVESTMENT PLANNER
# ============================================================
with tabs[2]:
    st.subheader("Optimize Portfolio")
    st.caption("Compare allocation strategies using the selected historical estimates and your current holdings.")
    if prices is None:
        st.warning("Market data is required before optimization.")
    else:
        metrics = calculate_metrics(prices)
        mu = metrics["expected_returns"].reindex(TICKERS)
        covariance_choice = st.radio("Covariance estimator", ["Sample covariance", "Ledoit–Wolf shrinkage covariance"], horizontal=True)
        covariance = metrics["sample_covariance"] if covariance_choice == "Sample covariance" else metrics["shrinkage_covariance"]
        covariance = covariance.reindex(index=TICKERS, columns=TICKERS)

        gamma = RISK_PROFILES.get(profile.get("risk_profile", "Moderate"), 5)
        max_weight = float(profile.get("max_weight", 0.30))
        cost_bps = st.slider("Estimated proportional transaction cost (basis points)", min_value=0, max_value=50, value=10, step=1)
        cost_rate = cost_bps / 10000

        st.info(f"Risk preference: **{profile.get('risk_profile')}** · Risk-aversion parameter γ = **{gamma}** · Maximum stock weight = **{max_weight:.0%}**")
        st.warning("The current optimizer requires fully invested equity weights summing to 100%. The recommendation therefore assumes all fresh capital plus existing holdings are deployed across the 10-stock universe; it does not retain a cash reserve.")

        if st.button("Run Portfolio Optimization", type="primary", width="stretch"):
            try:
                portfolios = generate_portfolios(
                    mu.to_numpy(dtype=float),
                    covariance.to_numpy(dtype=float),
                    current_weights.reindex(TICKERS).to_numpy(dtype=float),
                    gamma,
                    max_weight,
                    cost_rate,
                )
                st.session_state.optimization_result = {
                    "portfolios": portfolios,
                    "covariance_name": covariance_choice,
                    "cost_rate": cost_rate,
                    "total_value": total_portfolio_value,
                    "current_weights": current_weights.copy(),
                    "expected_returns": mu.copy(),
                    "covariance": covariance.copy(),
                    "latest_date": str(latest_date.date()),
                }
                st.success("Optimization completed. Review the strategy comparison below.")
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")

        result_state = st.session_state.get("optimization_result")
        if result_state:
            portfolios = result_state["portfolios"]
            strategies = {
                "Equal weight": portfolios["equal_weight"],
                "Markowitz": portfolios["markowitz"]["weights"],
                "Transaction-cost-aware": portfolios["cost_aware"]["weights"],
            }
            comparison_rows = []
            strategy_weights = {}
            for label, raw_weights in strategies.items():
                weights_series = pd.Series(np.asarray(raw_weights, dtype=float), index=TICKERS)
                strategy_weights[label] = weights_series
                stats = portfolio_risk_metrics(weights_series, {
                    "expected_returns": result_state["expected_returns"],
                    "sample_covariance": result_state["covariance"],
                    "beta": metrics["beta"],
                })
                comparison_rows.append({
                    "Strategy": label,
                    "Expected annual return": stats["expected_return"],
                    "Annual volatility": stats["volatility"],
                    "Beta": stats["beta"],
                    "Sharpe ratio (6% RF)": stats["sharpe"],
                })
            comparison = pd.DataFrame(comparison_rows)
            st.markdown("#### Strategy comparison")
            display_comparison = comparison.copy()
            for col in ["Expected annual return", "Annual volatility"]:
                display_comparison[col] = display_comparison[col].map(lambda x: f"{x:.2%}")
            display_comparison["Beta"] = display_comparison["Beta"].map(lambda x: f"{x:.2f}")
            display_comparison["Sharpe ratio (6% RF)"] = display_comparison["Sharpe ratio (6% RF)"].map(lambda x: f"{x:.2f}")
            st.dataframe(display_comparison, width="stretch", hide_index=True)

            weights_long = pd.concat([series.rename(label) for label, series in strategy_weights.items()], axis=1).reset_index(names="Stock")
            weights_long = weights_long.melt(id_vars="Stock", var_name="Strategy", value_name="Weight")
            fig_weights = px.bar(weights_long, x="Stock", y="Weight", color="Strategy", barmode="group", title="Recommended stock weights by strategy")
            fig_weights.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_weights, width="stretch")

            selected_strategy = st.selectbox("Inspect allocation and trades", list(strategy_weights.keys()), key="inspect_strategy")
            target_weights = strategy_weights[selected_strategy]
            current_w = result_state["current_weights"].reindex(TICKERS).fillna(0.0)
            nav = float(result_state["total_value"])
            trade_df = calculate_buy_sell(current_w.to_numpy(), target_weights.to_numpy(), nav)
            trade_df.index = TICKERS
            trade_df = trade_df.rename_axis("Stock").reset_index()
            trade_df["Target Weight"] = target_weights.values
            trade_df["Current Holding Value (₹)"] = current_w.values * nav
            trade_df["Target Holding Value (₹)"] = target_weights.values * nav
            trade_df["Buy Amount (₹)"] = trade_df["Buy Amount"]
            trade_df["Sell Amount (₹)"] = trade_df["Sell Amount"]
            trade_df["Estimated Shares to Buy"] = np.floor(trade_df["Buy Amount (₹)"].to_numpy() / latest_prices.reindex(TICKERS).to_numpy())
            trade_df["Estimated Shares to Sell"] = np.floor(trade_df["Sell Amount (₹)"].to_numpy() / latest_prices.reindex(TICKERS).to_numpy())

            cost_info = calculate_transaction_cost(current_w.to_numpy(), target_weights.to_numpy(), nav, result_state["cost_rate"])
            st.markdown(f"#### {selected_strategy}: allocation and trade plan")
            a1, a2, a3 = st.columns(3)
            a1.metric("Estimated gross traded amount", money(cost_info["traded_amount"]))
            a2.metric("Estimated transaction cost", money(cost_info["transaction_cost"]))
            a3.metric("One-way turnover", f"{cost_info['turnover']:.2%}")
            st.dataframe(trade_df[["Stock", "Current Holding Value (₹)", "Target Weight", "Target Holding Value (₹)", "Buy Amount (₹)", "Sell Amount (₹)", "Estimated Shares to Buy", "Estimated Shares to Sell"]], width="stretch", hide_index=True,
                         column_config={
                             "Current Holding Value (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                             "Target Weight": st.column_config.NumberColumn(format="%.2%"),
                             "Target Holding Value (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                             "Buy Amount (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                             "Sell Amount (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         })
            st.caption("Share quantities are approximate whole-share estimates. They exclude lot-size, price movement, taxes, brokerage slabs and minimum charges. This is a recommendation simulation, not an order.")
            st.download_button("Download selected allocation CSV", data=trade_df.to_csv(index=False).encode("utf-8"), file_name="investwise_recommended_allocation.csv", mime="text/csv")


    # ============================================================

    with st.expander("Backtesting & research", expanded=False):
        st.subheader("Backtesting & Research")
        st.caption("Historical walk-forward strategy comparison implemented by the quantitative engine.")
        if prices is None:
            st.warning("Market data is required for backtesting.")
        else:
            st.write("The supplied backtest evaluates quarterly rebalancing over its configured out-of-sample period (2024–2025). It starts with the selected initial capital and does not model your personal historical holdings.")
            bt_capital = st.number_input("Backtest initial capital (₹)", min_value=10000.0, value=fresh_capital, step=10000.0, key="bt_capital")
            bt_cost_bps = st.slider("Backtest transaction cost (basis points)", min_value=0, max_value=50, value=10, key="bt_cost")
            if st.button("Run historical backtest", type="primary"):
                try:
                    with st.spinner("Running quarterly walk-forward backtest..."):
                        backtest_result = run_backtest(
                            prices,
                            initial_capital=bt_capital,
                            risk_profile=profile.get("risk_profile", "Moderate"),
                            max_weight=float(profile.get("max_weight", 0.30)),
                            transaction_cost_rate=bt_cost_bps / 10000,
                        )
                    st.session_state.backtest_result = backtest_result
                    st.success("Backtest completed.")
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")

            bt = st.session_state.get("backtest_result")
            if bt:
                nav_df = bt["nav"].dropna(how="all")
                perf = bt["performance"]
                if not nav_df.empty:
                    fig_nav = px.line(nav_df, x=nav_df.index, y=nav_df.columns, title="Walk-forward portfolio NAV")
                    st.plotly_chart(fig_nav, width="stretch")
                    summary = pd.DataFrame(perf).T
                    st.markdown("#### Performance summary")
                    st.dataframe(summary.style.format({
                        "total_return": "{:.2%}",
                        "annual_return": "{:.2%}",
                        "annual_volatility": "{:.2%}",
                        "sharpe": "{:.2f}",
                        "max_drawdown": "{:.2%}",
                        "transaction_cost": "₹{:,.2f}",
                        "turnover": "{:.2f}",
                    }, na_rep="—"), width="stretch")
                    daily_returns = nav_df.pct_change().dropna()
                    if not daily_returns.empty:
                        drawdown = nav_df.div(nav_df.cummax()).sub(1)
                        fig_dd = px.line(drawdown, x=drawdown.index, y=drawdown.columns, title="Drawdown through the backtest")
                        fig_dd.update_yaxes(tickformat=".0%")
                        st.plotly_chart(fig_dd, width="stretch")
                        st.download_button("Download backtest NAV CSV", data=nav_df.to_csv().encode("utf-8"), file_name="investwise_backtest_nav.csv", mime="text/csv")


        # ============================================================

st.divider()
st.caption("InvestWise is an academic decision-support prototype. Historical estimates are not guarantees or personalized financial advice. No broker orders are placed.")
