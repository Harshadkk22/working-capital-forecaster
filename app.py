import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Optional ML import check
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="Executive Sales & Working Capital Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 96%;
    }
    h1, h2, h3, h4 {
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .stMarkdown p {
        color: #334155;
        font-size: 0.95rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #64748B;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.82rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .sub-positive { color: #059669; }
    .sub-negative { color: #DC2626; }
    .sub-neutral { color: #475569; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. CORE FINANCIAL & CAPACITY ENGINE
# ============================================================================
def calculate_single_scenario(
    n_reps: int,
    monthly_capacity: float,
    base_prod: float,
    target_prod: float,
    decay_rate: float,
    gross_margin_pct: float,
    cost_per_rep: float,
    fixed_overhead: float,
    add_overhead_per_rep: float,
    credit_sales_pct: float,
    dso: float,
    annual_financing_rate: float
) -> dict:
    """Computes financial, capacity, and working capital metrics for a given headcount."""
    if n_reps <= 0:
        return {
            "headcount": 0, "potential_sales": 0.0, "realized_sales": 0.0,
            "sales_per_rep": 0.0, "capacity_utilization": 0.0,
            "target_achievement_pct": 0.0, "target_status": "N/A",
            "payroll": 0.0, "var_overhead": 0.0, "total_opex": fixed_overhead,
            "ar_balance": 0.0, "monthly_financing_cost": 0.0, "net_contribution": -fixed_overhead
        }

    cap_headcount_threshold = max(1, int(monthly_capacity / base_prod)) if base_prod > 0 else 1
    excess_reps = max(0, n_reps - cap_headcount_threshold)

    productivity_per_rep = base_prod * ((1.0 - decay_rate) ** excess_reps)
    potential_sales = n_reps * productivity_per_rep
    realized_sales = min(monthly_capacity, potential_sales)
    sales_per_rep = realized_sales / n_reps

    capacity_utilization = (realized_sales / monthly_capacity * 100.0) if monthly_capacity > 0 else 0.0
    target_achievement_pct = (sales_per_rep / target_prod * 100.0) if target_prod > 0 else 0.0

    if target_achievement_pct >= 100.0:
        target_status = "Above Target"
    elif target_achievement_pct >= 90.0:
        target_status = "Near Target"
    elif target_achievement_pct >= 80.0:
        target_status = "Watch"
    else:
        target_status = "Underperforming"

    gross_profit = realized_sales * gross_margin_pct
    payroll = n_reps * cost_per_rep
    var_overhead = n_reps * add_overhead_per_rep
    total_opex = fixed_overhead + payroll + var_overhead

    credit_sales = realized_sales * credit_sales_pct
    ar_balance = credit_sales * (dso / 30.0)
    monthly_financing_cost = ar_balance * (annual_financing_rate / 12.0)

    net_contribution = gross_profit - total_opex - monthly_financing_cost

    return {
        "headcount": n_reps, 
        "potential_sales": potential_sales, 
        "realized_sales": realized_sales,
        "sales_per_rep": sales_per_rep, 
        "capacity_utilization": capacity_utilization,
        "target_achievement_pct": target_achievement_pct, 
        "target_status": target_status,
        "payroll": payroll, 
        "var_overhead": var_overhead, 
        "total_opex": total_opex,
        "ar_balance": ar_balance, 
        "monthly_financing_cost": monthly_financing_cost,
        "net_contribution": net_contribution
    }


# ============================================================================
# 3. SIDEBAR CONTROLS
# ============================================================================
st.sidebar.markdown("### Executive Control Panel")

proposed_n = st.sidebar.number_input("Target / Proposed Sales People", min_value=1, max_value=30, value=9, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Capacity & Sales Target Parameters**")
monthly_capacity = st.sidebar.number_input("Monthly Company Capacity (SAR)", min_value=100000, value=3000000, step=100000, format="%d")
target_prod = st.sidebar.number_input("Target Sales / Rep (SAR)", min_value=10000, value=600000, step=25000, format="%d")
base_prod = st.sidebar.number_input("Base Productivity / Rep (SAR)", min_value=10000, value=600000, step=25000, format="%d")
decay_rate = st.sidebar.slider("Productivity Decay Rate (%)", min_value=0.0, max_value=30.0, value=12.5, step=0.5) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("**Cost & Profitability Parameters**")
gross_margin_pct = st.sidebar.slider("Gross Margin (%)", min_value=1.0, max_value=50.0, value=15.0, step=0.5) / 100.0
cost_per_rep = st.sidebar.number_input("Payroll Cost / Rep / Mo (SAR)", min_value=5000, value=50000, step=5000, format="%d")
fixed_overhead = st.sidebar.number_input("Fixed Monthly Overhead (SAR)", min_value=0, value=100000, step=10000, format="%d")
add_overhead_per_rep = st.sidebar.number_input("Additional Overhead / Rep (SAR)", min_value=0, value=10000, step=1000, format="%d")

st.sidebar.markdown("---")
st.sidebar.markdown("**Working Capital Parameters**")
credit_sales_pct = st.sidebar.slider("Credit Sales Ratio (%)", min_value=0.0, max_value=100.0, value=85.0, step=5.0) / 100.0
dso = st.sidebar.number_input("Days Sales Outstanding (DSO)", min_value=0, max_value=180, value=60, step=5)
annual_financing_rate = st.sidebar.slider("Annual Financing Cost (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.5) / 100.0

params = {
    "monthly_capacity": float(monthly_capacity),
    "base_prod": float(base_prod),
    "target_prod": float(target_prod),
    "decay_rate": float(decay_rate),
    "gross_margin_pct": float(gross_margin_pct),
    "cost_per_rep": float(cost_per_rep),
    "fixed_overhead": float(fixed_overhead),
    "add_overhead_per_rep": float(add_overhead_per_rep),
    "credit_sales_pct": float(credit_sales_pct),
    "dso": float(dso),
    "annual_financing_rate": float(annual_financing_rate)
}


# ============================================================================
# 4. CALCULATIONS
# ============================================================================
prop_res = calculate_single_scenario(proposed_n, **params)

# Generate multi-headcount table for charts & sensitivity analysis (1 to 20 reps)
scenarios = [calculate_single_scenario(n, **params) for n in range(1, 21)]
df_scenarios = pd.DataFrame(scenarios)


# ============================================================================
# 5. HEADER
# ============================================================================
st.markdown("<h1>Executive Sales & Working Capital Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.05rem; color: #64748B; margin-top: -10px; margin-bottom: 20px;'>Headcount, Productivity, Profitability & AR Exposure Analysis</p>", unsafe_allow_html=True)


# ============================================================================
# 6. METRIC CARDS
# ============================================================================
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Monthly Realized Sales</div>
        <div class="metric-value">SAR {prop_res['realized_sales']/1e6:.2f}M</div>
        <div class="metric-sub sub-neutral">Cap: SAR {monthly_capacity/1e6:.2f}M</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sales / Rep</div>
        <div class="metric-value">SAR {prop_res['sales_per_rep']/1e3:.0f}K</div>
        <div class="metric-sub sub-neutral">Target: SAR {target_prod/1e3:.0f}K</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Target Achievement</div>
        <div class="metric-value">{prop_res['target_achievement_pct']:.1f}%</div>
        <div class="metric-sub {'sub-positive' if prop_res['target_achievement_pct']>=90 else 'sub-negative'}">{prop_res['target_status']}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Capacity Utilization</div>
        <div class="metric-value">{prop_res['capacity_utilization']:.1f}%</div>
        <div class="metric-sub {'sub-negative' if prop_res['capacity_utilization']>=98 else 'sub-positive'}">{'At Ceiling' if prop_res['capacity_utilization']>=99.5 else 'Optimal Range'}</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Net Contribution</div>
        <div class="metric-value">SAR {prop_res['net_contribution']/1e3:.0f}K</div>
        <div class="metric-sub {'sub-positive' if prop_res['net_contribution']>0 else 'sub-negative'}">{'Profitable' if prop_res['net_contribution']>0 else 'Loss-Making'}</div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">AR Cash Lock-Up</div>
        <div class="metric-value">SAR {prop_res['ar_balance']/1e6:.2f}M</div>
        <div class="metric-sub sub-neutral">{prop_res['ar_balance']/prop_res['realized_sales'] if prop_res['realized_sales']>0 else 0:.1f}x Monthly Sales</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 7. ANALYTICAL CHARTS
# ============================================================================
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    fig_sales = go.Figure()
    fig_sales.add_trace(go.Scatter(x=df_scenarios["headcount"], y=df_scenarios["realized_sales"]/1e6,
                                   mode="lines+markers", name="Realized Monthly Sales (M SAR)",
                                   line=dict(color="#0284C7", width=3)))
    fig_sales.add_trace(go.Scatter(x=df_scenarios["headcount"], y=df_scenarios["potential_sales"]/1e6,
                                   mode="lines", name="Unconstrained Potential (M SAR)",
                                   line=dict(color="#94A3B8", dash="dash")))
    fig_sales.add_hline(y=monthly_capacity/1e6, line_dash="dot", line_color="#DC2626", annotation_text="Monthly Capacity Ceiling")
    fig_sales.update_layout(
        title="Sales Curve & Capacity Ceiling",
        xaxis_title="Headcount (Salespeople)",
        yaxis_title="Monthly Sales (Million SAR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_sales, use_container_width=True)

with c2:
    fig_profit = go.Figure()
    fig_profit.add_trace(go.Scatter(x=df_scenarios["headcount"], y=df_scenarios["net_contribution"]/1e3,
                                    mode="lines+markers", name="Net Contribution (K SAR)",
                                    line=dict(color="#059669", width=3)))
    fig_profit.add_trace(go.Scatter(x=df_scenarios["headcount"], y=df_scenarios["ar_balance"]/1e6,
                                    mode="lines+markers", name="AR Cash Lock-Up (M SAR)",
                                    line=dict(color="#D97706", width=2, dash="dash"), yaxis="y2"))
    
    fig_profit.update_layout(
        title="Net Profitability vs. AR Cash Lock-Up",
        xaxis_title="Headcount (Salespeople)",
        yaxis=dict(title="Net Contribution (Thousand SAR)"),
        yaxis2=dict(title="AR Balance (Million SAR)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_profit, use_container_width=True)


# ============================================================================
# 8. SENSITIVITY HEATMAP & DETAILED SCENARIO TABLE
# ============================================================================
tab_table, tab_sens = st.tabs(["📋 Multi-Headcount Scenario Table", "🔥 2D Sensitivity Matrix"])

with tab_table:
    formatted_df = df_scenarios.copy()
    formatted_df["realized_sales"] = formatted_df["realized_sales"].map("SAR {:,.0f}".format)
    formatted_df["sales_per_rep"] = formatted_df["sales_per_rep"].map("SAR {:,.0f}".format)
    formatted_df["target_achievement_pct"] = formatted_df["target_achievement_pct"].map("{:.1f}%".format)
    formatted_df["capacity_utilization"] = formatted_df["capacity_utilization"].map("{:.1f}%".format)
    formatted_df["total_opex"] = formatted_df["total_opex"].map("SAR {:,.0f}".format)
    formatted_df["ar_balance"] = formatted_df["ar_balance"].map("SAR {:,.0f}".format)
    formatted_df["net_contribution"] = formatted_df["net_contribution"].map("SAR {:,.0f}".format)
    
    st.dataframe(
        formatted_df[[
            "headcount", "realized_sales", "sales_per_rep", "target_achievement_pct",
            "capacity_utilization", "total_opex", "ar_balance", "net_contribution"
        ]],
        use_container_width=True,
        hide_index=True
    )

with tab_sens:
    gm_range = np.linspace(max(0.05, gross_margin_pct - 0.10), min(0.50, gross_margin_pct + 0.10), 7)
    reps_range = list(range(1, 16))
    
    matrix_gm = []
    for gm in gm_range:
        row = []
        for r in reps_range:
            p = params.copy()
            p["gross_margin_pct"] = gm
            res = calculate_single_scenario(r, **p)
            row.append(res["net_contribution"] / 1e3)
        matrix_gm.append(row)
        
    fig_sens_gm = go.Figure(data=go.Heatmap(
        z=matrix_gm,
        x=[f"{r} Reps" for r in reps_range],
        y=[f"Margin {gm*100:.1f}%" for gm in gm_range],
        colorscale="RdYlGn",
        colorbar=dict(title="Net Profit (K SAR)")
    ))
    fig_sens_gm.update_layout(title="Net Contribution Sensitivity: Headcount vs. Gross Margin", template="plotly_white")
    st.plotly_chart(fig_sens_gm, use_container_width=True)


# ============================================================================
# 9. MACHINE LEARNING MODULE (OPTIONAL SIMULATION)
# ============================================================================
with st.expander("🤖 Machine Learning Model (Synthetic Sales Regressor)"):
    if not SKLEARN_AVAILABLE:
        st.error("`scikit-learn` is not installed in the environment. Run `pip install scikit-learn` to enable this section.")
    else:
        st.caption("Trains a Random Forest Regressor on synthetic operational data generated around your active parameters to predict non-linear sales curve outcomes.")
        
        c_ml1, c_ml2 = st.columns(2)
        with c_ml1:
            n_samples = st.slider("Synthetic Training Samples", 100, 2000, 500, step=100)
        with c_ml2:
            n_estimators = st.slider("Random Forest Trees", 10, 200, 100, step=10)

        if st.button("🚀 Train & Predict Sales Model", use_container_width=True):
            np.random.seed(42)
            synth_reps = np.random.randint(1, 20, size=n_samples)
            synth_cap = np.random.normal(monthly_capacity, monthly_capacity * 0.1, size=n_samples)
            synth_target = np.random.normal(target_prod, target_prod * 0.05, size=n_samples)
            
            y_clean = np.array([
                calculate_single_scenario(
                    r, c, base_prod, t, decay_rate, gross_margin_pct,
                    cost_per_rep, fixed_overhead, add_overhead_per_rep,
                    credit_sales_pct, dso, annual_financing_rate
                )["realized_sales"]
                for r, c, t in zip(synth_reps, synth_cap, synth_target)
            ])
            
            noise = np.random.normal(0, y_clean * 0.05, size=n_samples)
            y_noisy = (y_clean + noise).clip(0, None)
            
            X = pd.DataFrame({
                "salespeople": synth_reps,
                "company_capacity": synth_cap,
                "target_per_rep": synth_target
            })
            
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            rf.fit(X, y_noisy)
            
            r2_score = rf.score(X, y_noisy)
            
            # Predict for current scenario
            current_X = pd.DataFrame({
                "salespeople": [proposed_n],
                "company_capacity": [monthly_capacity],
                "target_per_rep": [target_prod]
            })
            ml_pred = rf.predict(current_X)[0]
            
            st.success(f"✅ Random Forest Regressor Trained! Model R² Score: **{r2_score:.4f}**")
            st.metric(
                label=f"ML Predicted Realized Sales ({proposed_n} Reps)",
                value=f"SAR {ml_pred/1e6:.2f}M",
                delta=f"{((ml_pred - prop_res['realized_sales']) / max(prop_res['realized_sales'], 1.0)) * 100:.1f}% vs Formula"
            )


# ============================================================================
# 10. EXPORT & FOOTER
# ============================================================================
st.markdown("---")
c_foot1, c_foot2 = st.columns([3, 1])

with c_foot1:
    st.caption("Executive Sales & Working Capital Dashboard • Decision Model for Headcount Expansion & AR Risk Management.")

with c_foot2:
    csv_data = df_scenarios.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Data (CSV)",
        data=csv_data,
        file_name="sales_working_capital_simulation.csv",
        mime="text/csv",
        use_container_width=True
    )