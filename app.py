"""
================================================================================
 LIVE SALES CAPACITY, PROFITABILITY & RECEIVABLES SIMULATION MODEL
================================================================================
Illustrative Simulation — Replace with Actual Company Data

A Streamlit decision-support tool that tests the hypothesis:
    "Increasing sales headcount does not automatically increase company sales."

Author note: All figures in this file are FAKE / ILLUSTRATIVE. Replace the
default assumptions in the sidebar (or the DEFAULTS dict below) with real
company data when available. The calculation engine is fully parametric and
does not hard-code any company-specific numbers inside the formulas.

Run locally with:
    pip install streamlit pandas numpy plotly scikit-learn
    streamlit run app.py
================================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------
# ML is optional and only used on synthetic data — see section 16 (ML tab)
# ------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================================
# 1. PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Sales Capacity & Profitability Decision Model",
    page_icon="📊",
    layout="wide",
)

EPS = 1e-9  # guards against division by zero everywhere in this file


# ============================================================================
# 2. DEFAULT ASSUMPTIONS (fake / illustrative — replace with real data)
# ============================================================================
DEFAULTS = dict(
    company_capacity=3_000_000,      # SAR / month, hard operating ceiling
    min_reps=1,
    max_reps=20,
    default_reps=5,
    target_per_rep=600_000,          # SAR / month
    dso=60,                          # days
    gross_margin=0.15,               # 15%
    cost_per_rep=50_000,             # SAR / month, fully loaded
    base_overhead=100_000,           # SAR / month
    overhead_per_extra_rep=10_000,   # SAR / month per rep beyond the first
    productivity_decline_pct=0.12,   # 12% compounding decline per rep over capacity
    min_target_achievement=0.80,     # 80%
    credit_sales_pct=0.80,           # 80% of sales sold on credit
    inventory_days=30,
    ap_days=30,
)

SCENARIO_PRESETS = {
    "Scenario A — Lean Team (5 reps)": 5,
    "Scenario B — Moderate Expansion (7 reps)": 7,
    "Scenario C — Aggressive Expansion (10 reps)": 10,
    "Scenario D — Overcapacity (15 reps)": 15,
}


# ============================================================================
# 3. CORE CALCULATION ENGINE
#    (pure functions — no Streamlit calls — so they can be reused / tested
#    and later swapped for real-data / ML-driven versions)
# ============================================================================

def capacity_headcount(company_capacity: float, target_per_rep: float) -> float:
    """Number of reps at which the company's hard sales ceiling is reached,
    assuming each rep produces exactly the target."""
    return company_capacity / max(target_per_rep, EPS)


def productivity_per_rep(n_reps: int, company_capacity: float, target_per_rep: float,
                         decline_pct: float) -> float:
    """
    Two-phase productivity model.

    Phase 1 (n_reps <= capacity headcount): productivity ~= target per rep.
    Phase 2 (n_reps > capacity headcount): productivity decays progressively,
             compounding by `decline_pct` for every rep beyond the capacity
             point (soft, non-linear decline rather than an abrupt collapse).
    """
    cap_reps = capacity_headcount(company_capacity, target_per_rep)
    if n_reps <= cap_reps:
        return target_per_rep
    excess = n_reps - cap_reps
    decay_factor = (1 - decline_pct) ** excess
    return target_per_rep * decay_factor


def run_scenario(n_reps: int, company_capacity: float, target_per_rep: float,
                  decline_pct: float, gross_margin: float, cost_per_rep: float,
                  base_overhead: float, overhead_per_extra_rep: float,
                  dso: float, credit_sales_pct: float,
                  inventory_days: float, ap_days: float,
                  min_target_achievement: float) -> dict:
    """Compute every metric for a single headcount scenario."""
    n_reps = max(n_reps, 0)

    prod_per_rep = productivity_per_rep(n_reps, company_capacity, target_per_rep, decline_pct)
    potential_sales = n_reps * target_per_rep      # if every rep hit target
    actual_sales = n_reps * prod_per_rep            # reflects productivity dilution

    sales_per_rep = actual_sales / max(n_reps, EPS) if n_reps > 0 else 0.0
    target_achievement = sales_per_rep / max(target_per_rep, EPS)
    capacity_utilization = actual_sales / max(company_capacity, EPS)

    gross_profit = actual_sales * gross_margin
    sales_payroll = n_reps * cost_per_rep
    mgmt_overhead = base_overhead + overhead_per_extra_rep * max(n_reps - 1, 0)
    total_cost = sales_payroll + mgmt_overhead
    net_contribution = gross_profit - total_cost

    credit_sales = actual_sales * credit_sales_pct
    outstanding_ar = credit_sales * dso / 30.0
    ar_to_sales = outstanding_ar / max(actual_sales, EPS)

    cogs = actual_sales * (1 - gross_margin)
    inventory_requirement = cogs * inventory_days / 30.0
    accounts_payable = cogs * ap_days / 30.0
    working_capital = outstanding_ar + inventory_requirement - accounts_payable

    comm_links = n_reps * (n_reps - 1) / 2

    # WIN / LOSS decision logic
    is_win = (
        net_contribution > 0
        and target_achievement >= min_target_achievement
        and capacity_utilization <= 1.15   # >15% over hard capacity = unsustainable dilution
    )
    if net_contribution <= 0:
        status = "🔴 LOSS"
    elif target_achievement < min_target_achievement:
        status = "🔴 LOSS"
    elif capacity_utilization > 1.0:
        status = "🟡 WARNING"
    else:
        status = "🟢 WIN"

    return dict(
        salespeople=n_reps,
        productivity_per_rep=prod_per_rep,
        potential_sales=potential_sales,
        actual_sales=actual_sales,
        sales_per_rep=sales_per_rep,
        target_achievement=target_achievement,
        capacity_utilization=capacity_utilization,
        gross_profit=gross_profit,
        sales_payroll=sales_payroll,
        mgmt_overhead=mgmt_overhead,
        total_cost=total_cost,
        net_contribution=net_contribution,
        outstanding_ar=outstanding_ar,
        ar_to_sales=ar_to_sales,
        working_capital=working_capital,
        comm_links=comm_links,
        is_win=is_win,
        status=status,
    )


def build_headcount_table(max_reps: int, **params) -> pd.DataFrame:
    """Run run_scenario() for every headcount from 1 to max_reps."""
    rows = [run_scenario(n, **params) for n in range(1, max_reps + 1)]
    return pd.DataFrame(rows)


def recommend_headcount(df: pd.DataFrame, min_target_achievement: float) -> int:
    """
    Optimal headcount = the scenario with the highest Net Contribution among
    those that still meet the minimum target-achievement threshold. If none
    meet the threshold, fall back to the global max Net Contribution.
    """
    eligible = df[df["target_achievement"] >= min_target_achievement]
    pool = eligible if not eligible.empty else df
    best_row = pool.loc[pool["net_contribution"].idxmax()]
    return int(best_row["salespeople"])


# ============================================================================
# 4. SIDEBAR — LIVE SCENARIO CONTROLS
# ============================================================================
st.sidebar.title("⚙️ Live Scenario Controls")
st.sidebar.caption("Illustrative Simulation — Replace with Actual Company Data")

if "n_reps" not in st.session_state:
    st.session_state.n_reps = DEFAULTS["default_reps"]

st.sidebar.subheader("Quick Scenario Presets")
preset_cols = st.sidebar.columns(2)
preset_names = list(SCENARIO_PRESETS.keys())
for i, name in enumerate(preset_names):
    col = preset_cols[i % 2]
    short_label = name.split("—")[0].strip()
    if col.button(short_label, use_container_width=True, key=f"preset_{i}"):
        st.session_state.n_reps = SCENARIO_PRESETS[name]

st.sidebar.subheader("Sales Team")
n_reps = st.sidebar.slider(
    "Number of Salespeople", DEFAULTS["min_reps"], DEFAULTS["max_reps"],
    key="n_reps",
)

st.sidebar.subheader("Capacity & Targets")
company_capacity = st.sidebar.number_input(
    "Monthly Company Sales Capacity (SAR)", min_value=100_000,
    value=DEFAULTS["company_capacity"], step=100_000, format="%d",
)
target_per_rep = st.sidebar.number_input(
    "Target per Salesperson (SAR/month)", min_value=10_000,
    value=DEFAULTS["target_per_rep"], step=10_000, format="%d",
)
productivity_decline_pct = st.sidebar.slider(
    "Productivity Decline After Capacity (%)", 0, 50,
    int(DEFAULTS["productivity_decline_pct"] * 100),
) / 100.0

st.sidebar.subheader("Costs")
cost_per_rep = st.sidebar.number_input(
    "Monthly Cost per Salesperson (SAR)", min_value=0,
    value=DEFAULTS["cost_per_rep"], step=5_000, format="%d",
)
base_overhead = st.sidebar.number_input(
    "Base Management Overhead (SAR/month)", min_value=0,
    value=DEFAULTS["base_overhead"], step=5_000, format="%d",
)
overhead_per_extra_rep = st.sidebar.number_input(
    "Additional Overhead per Extra Salesperson (SAR)", min_value=0,
    value=DEFAULTS["overhead_per_extra_rep"], step=1_000, format="%d",
)

st.sidebar.subheader("Profitability")
gross_margin = st.sidebar.slider(
    "Gross Margin (%)", 1, 90, int(DEFAULTS["gross_margin"] * 100),
) / 100.0
min_target_achievement = st.sidebar.slider(
    "Minimum Acceptable Target Achievement (%)", 30, 100,
    int(DEFAULTS["min_target_achievement"] * 100),
) / 100.0

st.sidebar.subheader("Receivables & Working Capital")
dso = st.sidebar.slider("Days Sales Outstanding — DSO", 15, 180, DEFAULTS["dso"])
credit_sales_pct = st.sidebar.slider(
    "Credit Sales (% of Total Sales)", 0, 100, int(DEFAULTS["credit_sales_pct"] * 100),
) / 100.0
inventory_days = st.sidebar.slider("Inventory Days", 0, 180, DEFAULTS["inventory_days"])
ap_days = st.sidebar.slider("Accounts Payable Days", 0, 180, DEFAULTS["ap_days"])

# bundle shared params for reuse
engine_params = dict(
    company_capacity=company_capacity,
    target_per_rep=target_per_rep,
    decline_pct=productivity_decline_pct,
    gross_margin=gross_margin,
    cost_per_rep=cost_per_rep,
    base_overhead=base_overhead,
    overhead_per_extra_rep=overhead_per_extra_rep,
    dso=dso,
    credit_sales_pct=credit_sales_pct,
    inventory_days=inventory_days,
    ap_days=ap_days,
    min_target_achievement=min_target_achievement,
)

# ============================================================================
# 5. RUN THE MODEL
# ============================================================================
current = run_scenario(n_reps, **engine_params)
table = build_headcount_table(DEFAULTS["max_reps"], **engine_params)
recommended_n = recommend_headcount(table, min_target_achievement)
cap_reps_point = capacity_headcount(company_capacity, target_per_rep)


# ============================================================================
# 6. HEADER + LIVE WIN/LOSS BANNER
# ============================================================================
st.title("📊 Sales Capacity, Headcount, Profitability & Receivables Model")
st.warning("⚠️ **Illustrative Simulation — Replace with Actual Company Data.** "
           "All figures below are fake/synthetic and for demonstration only.")

banner_color = {"🟢 WIN": "#0f5132", "🟡 WARNING": "#7a5b00", "🔴 LOSS": "#842029"}
banner_bg = {"🟢 WIN": "#d1e7dd", "🟡 WARNING": "#fff3cd", "🔴 LOSS": "#f8d7da"}
status_key = current["status"]

st.markdown(
    f"""
    <div style="background-color:{banner_bg[status_key]}; color:{banner_color[status_key]};
                padding:20px; border-radius:10px; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">LIVE BUSINESS DECISION: {status_key.split(' ')[0]}
        {"VALUE CREATING" if "WIN" in status_key else ("CAPACITY PRESSURE" if "WARNING" in status_key else "VALUE DESTRUCTIVE")}
        </h2>
        <p style="margin:4px 0 0 0; font-size:16px;">
        {n_reps} salespeople • Net Contribution SAR {current['net_contribution']:,.0f}/month •
        Target Achievement {current['target_achievement']*100:,.1f}% •
        Capacity Utilization {current['capacity_utilization']*100:,.1f}%
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(f"💡 **Recommended Headcount: {recommended_n} Salespeople** "
        f"(see Section: Optimal Headcount Engine below for the explanation)")


# ============================================================================
# 7. KPI CARDS
# ============================================================================
st.subheader("Executive KPI Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Monthly Sales (Actual)", f"SAR {current['actual_sales']:,.0f}")
k2.metric("Sales / Rep", f"SAR {current['sales_per_rep']:,.0f}")
k3.metric("Target Achievement", f"{current['target_achievement']*100:,.1f}%")
k4.metric("Capacity Utilization", f"{current['capacity_utilization']*100:,.1f}%")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Gross Profit", f"SAR {current['gross_profit']:,.0f}")
k6.metric("Sales Payroll", f"SAR {current['sales_payroll']:,.0f}")
k7.metric("Total Overhead", f"SAR {current['mgmt_overhead']:,.0f}")
k8.metric("Net Contribution", f"SAR {current['net_contribution']:,.0f}",
          delta=f"{'Positive' if current['net_contribution'] >= 0 else 'Negative'}")

k9, k10, k11, k12 = st.columns(4)
k9.metric("Outstanding AR", f"SAR {current['outstanding_ar']:,.0f}")
k10.metric("AR Multiplier (AR/Sales)", f"{current['ar_to_sales']:,.2f}x")
k11.metric("Working Capital Requirement", f"SAR {current['working_capital']:,.0f}")
k12.metric("Communication Links", f"{int(current['comm_links']):,}")


# ============================================================================
# 8. LIVE MANAGEMENT COMMENTARY
# ============================================================================
st.subheader("📝 Live Management Commentary")

comments = []
if current["capacity_utilization"] < 0.95:
    comments.append("🟢 The company has unused sales capacity. Additional sales headcount "
                     "may create incremental revenue, subject to profitability and "
                     "working-capital availability.")
elif 0.95 <= current["capacity_utilization"] <= 1.05:
    comments.append("🟢 The current headcount is aligned with the company's operating "
                     "capacity. Productivity and target achievement are strong.")
else:
    comments.append("🟠 The company is operating beyond its estimated productive headcount. "
                     "Additional salespeople are diluting average productivity.")

if current["net_contribution"] < 0:
    comments.append("🔴 Additional sales headcount is value destructive under the current "
                     "assumptions. Payroll and overhead are increasing faster than gross "
                     "profit generation.")

if dso >= 90 or current["ar_to_sales"] >= 2.5:
    comments.append("🔴 Receivables exposure is increasing and may create significant "
                     "working-capital pressure despite sales growth.")
elif dso >= 60:
    comments.append("🟠 Receivables are moderately elevated; monitor DSO and collections closely.")

for c in comments:
    st.markdown(f"- {c}")


# ============================================================================
# 9. CHARTS
# ============================================================================
st.subheader("📈 Charts")

tabs = st.tabs([
    "1. Sales vs Headcount", "2. Productivity vs Headcount",
    "3. Target Achievement", "4. Net Contribution",
    "5. AR vs Headcount", "6. Sales vs AR",
])

with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["salespeople"], y=table["actual_sales"],
                             mode="lines+markers", name="Actual Sales"))
    fig.add_hline(y=company_capacity, line_dash="dash", line_color="red",
                  annotation_text="Company Capacity")
    fig.add_vline(x=n_reps, line_dash="dot", line_color="gray",
                  annotation_text="Current Headcount")
    fig.update_layout(xaxis_title="Salespeople", yaxis_title="Monthly Sales (SAR)",
                      title="Sales vs Headcount — where sales peak and decline")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["salespeople"], y=table["sales_per_rep"],
                             mode="lines+markers", name="Sales / Rep"))
    fig.add_hline(y=target_per_rep, line_dash="dash", line_color="green",
                  annotation_text="Target per Rep")
    fig.add_vline(x=cap_reps_point, line_dash="dot", line_color="red",
                  annotation_text="Capacity Point")
    fig.update_layout(xaxis_title="Salespeople", yaxis_title="Average Sales / Rep (SAR)",
                      title="Productivity vs Headcount")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["salespeople"], y=table["target_achievement"] * 100,
                             mode="lines+markers", name="Target Achievement %"))
    fig.add_hline(y=min_target_achievement * 100, line_dash="dash", line_color="red",
                  annotation_text="Minimum Acceptable")
    fig.update_layout(xaxis_title="Salespeople", yaxis_title="Target Achievement (%)",
                      title="Target Achievement vs Headcount")
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in table["net_contribution"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["salespeople"], y=table["net_contribution"],
                         marker_color=colors, name="Net Contribution"))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(xaxis_title="Salespeople", yaxis_title="Net Contribution (SAR)",
                      title="Net Contribution vs Headcount — WIN → LOSS transition")
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["salespeople"], y=table["outstanding_ar"],
                             mode="lines+markers", name=f"Outstanding AR (DSO={dso}d)"))
    fig.update_layout(xaxis_title="Salespeople", yaxis_title="Outstanding AR (SAR)",
                      title="Accounts Receivable vs Headcount")
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["actual_sales"], y=table["outstanding_ar"],
                             mode="markers+lines", text=table["salespeople"],
                             hovertemplate="Sales: SAR %{x:,.0f}<br>AR: SAR %{y:,.0f}<br>Reps: %{text}"))
    fig.update_layout(xaxis_title="Monthly Sales (SAR)", yaxis_title="Outstanding AR (SAR)",
                      title="Monthly Sales vs Outstanding AR")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 10. OPTIMAL HEADCOUNT ENGINE
# ============================================================================
st.subheader("🎯 Optimal Headcount Engine")

rec_row = table[table["salespeople"] == recommended_n].iloc[0]
st.success(
    f"**Recommended Headcount: {recommended_n} Salespeople**\n\n"
    f"At {recommended_n} salespeople, the model estimates monthly sales of "
    f"SAR {rec_row['actual_sales']:,.0f} with productivity of "
    f"SAR {rec_row['productivity_per_rep']:,.0f} per rep "
    f"({rec_row['target_achievement']*100:,.0f}% of target), generating a net "
    f"contribution of SAR {rec_row['net_contribution']:,.0f}/month. "
    f"This headcount best balances sales capacity, productivity, payroll, "
    f"overhead and working-capital exposure. Additional headcount beyond this "
    f"point reduces average productivity and increases fixed costs faster than "
    f"gross profit."
)


# ============================================================================
# 11. HEADCOUNT SCENARIO TABLE (1–20)
# ============================================================================
st.subheader("📋 Headcount Scenario Table (1–20 Salespeople)")

display_table = table.copy()
display_table["Target Achievement %"] = (display_table["target_achievement"] * 100).round(1)
display_table["Capacity Utilization %"] = (display_table["capacity_utilization"] * 100).round(1)
display_table["AR / Sales"] = display_table["ar_to_sales"].round(2)
display_table["Communication Links"] = display_table["comm_links"].astype(int)

show_cols = {
    "salespeople": "Salespeople",
    "potential_sales": "Potential Sales",
    "actual_sales": "Actual Sales",
    "sales_per_rep": "Sales / Rep",
    "Target Achievement %": "Target Achievement %",
    "Capacity Utilization %": "Capacity Utilization %",
    "gross_profit": "Gross Profit",
    "sales_payroll": "Sales Payroll",
    "mgmt_overhead": "Management Overhead",
    "total_cost": "Total Cost",
    "net_contribution": "Net Contribution",
    "outstanding_ar": "Outstanding AR",
    "AR / Sales": "AR / Sales",
    "Communication Links": "Communication Links",
    "status": "WIN / LOSS",
}
display_df = display_table[list(show_cols.keys())].rename(columns=show_cols)

money_cols = ["Potential Sales", "Actual Sales", "Sales / Rep", "Gross Profit",
              "Sales Payroll", "Management Overhead", "Total Cost",
              "Net Contribution", "Outstanding AR"]


def highlight_status(row):
    if "WIN" in row["WIN / LOSS"]:
        color = "background-color: #d1e7dd"
    elif "WARNING" in row["WIN / LOSS"]:
        color = "background-color: #fff3cd"
    else:
        color = "background-color: #f8d7da"
    return [color] * len(row)


styled = (
    display_df.style
    .apply(highlight_status, axis=1)
    .format({c: "SAR {:,.0f}" for c in money_cols})
    .format({"Target Achievement %": "{:.1f}%", "Capacity Utilization %": "{:.1f}%",
              "AR / Sales": "{:.2f}x"})
)
st.dataframe(styled, use_container_width=True, height=460)


# ============================================================================
# 12. SCENARIO COMPARISON (A/B/C/D presets)
# ============================================================================
st.subheader("🔀 Scenario Comparison — A / B / C / D")

comparison_rows = []
for name, n in SCENARIO_PRESETS.items():
    r = run_scenario(n, **engine_params)
    r["scenario"] = name
    comparison_rows.append(r)
comparison_df = pd.DataFrame(comparison_rows)[
    ["scenario", "salespeople", "actual_sales", "sales_per_rep",
     "target_achievement", "net_contribution", "outstanding_ar", "status"]
]
comparison_df.columns = ["Scenario", "Salespeople", "Actual Sales", "Sales/Rep",
                          "Target Achievement", "Net Contribution", "Outstanding AR", "Status"]
comparison_df["Target Achievement"] = (comparison_df["Target Achievement"] * 100).round(1).astype(str) + "%"
for c in ["Actual Sales", "Sales/Rep", "Net Contribution", "Outstanding AR"]:
    comparison_df[c] = comparison_df[c].map(lambda v: f"SAR {v:,.0f}")
st.table(comparison_df)


# ============================================================================
# 13. EXECUTIVE SUMMARY (auto-generated narrative)
# ============================================================================
st.subheader("🧾 Executive Summary")

capacity_state = (
    "below capacity" if current["capacity_utilization"] < 0.95 else
    "at capacity" if current["capacity_utilization"] <= 1.05 else
    "above capacity"
)

summary_md = f"""
**Current Situation:** The model is simulating **{n_reps} salespeople**, generating
estimated monthly sales of **SAR {current['actual_sales']:,.0f}** against a company
capacity of **SAR {company_capacity:,.0f}**.

**Capacity Assessment:** The company is currently **{capacity_state}**
({current['capacity_utilization']*100:,.1f}% utilization). The estimated capacity
point is approximately **{cap_reps_point:,.1f} salespeople** given a target of
SAR {target_per_rep:,.0f}/rep.

**Productivity Assessment:** Average sales per rep are **SAR {current['sales_per_rep']:,.0f}**,
representing **{current['target_achievement']*100:,.1f}%** of the SAR {target_per_rep:,.0f}
target (minimum acceptable: {min_target_achievement*100:,.0f}%).

**Financial Assessment:** Gross profit is **SAR {current['gross_profit']:,.0f}**, sales
payroll is **SAR {current['sales_payroll']:,.0f}**, and management overhead is
**SAR {current['mgmt_overhead']:,.0f}**, resulting in a net contribution of
**SAR {current['net_contribution']:,.0f}/month**.

**Receivables Assessment:** At {dso} days DSO and {credit_sales_pct*100:,.0f}% credit
sales, outstanding AR is estimated at **SAR {current['outstanding_ar']:,.0f}**
({current['ar_to_sales']:.2f}x monthly sales), with an estimated working-capital
requirement of **SAR {current['working_capital']:,.0f}**.

**Management Recommendation:** {"Increasing headcount further is **not** recommended under current assumptions; the model suggests **" + str(recommended_n) + " salespeople** as the profitable optimum." if n_reps >= recommended_n else "There is room to grow headcount toward the model's recommended optimum of **" + str(recommended_n) + " salespeople** before productivity dilution and cost growth erode returns."}
"""
st.markdown(summary_md)


# ============================================================================
# 14. RECEIVABLES & WORKING CAPITAL DEEP-DIVE
# ============================================================================
with st.expander("💰 Receivables & Working Capital — Deep Dive"):
    st.markdown(f"""
    - **Credit Sales:** SAR {current['actual_sales']*credit_sales_pct:,.0f}
      ({credit_sales_pct*100:,.0f}% of total sales)
    - **DSO:** {dso} days → AR ≈ Credit Sales × DSO / 30
    - **Outstanding AR:** SAR {current['outstanding_ar']:,.0f}
    - **AR / Sales multiplier:** {current['ar_to_sales']:.2f}x
      (30d ≈ 1.0x, 60d ≈ 2.0x, 90d ≈ 3.0x — illustrative rule of thumb)
    - **Inventory Requirement** ({inventory_days} days of COGS): SAR {current['actual_sales']*(1-gross_margin)*inventory_days/30:,.0f}
    - **Accounts Payable** ({ap_days} days of COGS): SAR {current['actual_sales']*(1-gross_margin)*ap_days/30:,.0f}
    - **Working Capital Requirement** (AR + Inventory − AP): **SAR {current['working_capital']:,.0f}**
    """)

    dso_range = list(range(15, 181, 15))
    ar_by_dso = [current["actual_sales"] * credit_sales_pct * d / 30.0 for d in dso_range]
    fig_dso = go.Figure()
    fig_dso.add_trace(go.Scatter(x=dso_range, y=ar_by_dso, mode="lines+markers"))
    fig_dso.add_vline(x=dso, line_dash="dot", line_color="red", annotation_text="Current DSO")
    fig_dso.update_layout(xaxis_title="DSO (days)", yaxis_title="Outstanding AR (SAR)",
                           title="Sensitivity: AR vs DSO at Current Sales Level")
    st.plotly_chart(fig_dso, use_container_width=True)


# ============================================================================
# 15. 12-MONTH EXTENSION (optional forward simulation)
# ============================================================================
with st.expander("📅 12-Month Extension (optional forward simulation)"):
    st.caption("Illustrative — assumptions below are independent of the live controls above.")

    c1, c2, c3 = st.columns(3)
    with c1:
        start_reps = st.number_input("Starting Headcount", 1, 20, n_reps, key="m_start_reps")
        monthly_headcount_growth = st.number_input("Headcount added per month", 0.0, 5.0, 0.0, 0.5, key="m_hc_growth")
    with c2:
        monthly_sales_growth = st.slider("Extra Monthly Sales Growth (%)", -10, 10, 0, key="m_sales_growth") / 100.0
        collection_rate = st.slider("Monthly Collection Rate of Opening AR (%)", 0, 100, 70, key="m_collection") / 100.0
    with c3:
        m_dso = st.slider("DSO (months)", 15, 180, dso, key="m_dso")
        m_margin = st.slider("Gross Margin (months, %)", 1, 90, int(gross_margin*100), key="m_margin") / 100.0

    months = list(range(1, 13))
    sim_rows = []
    opening_ar = 0.0
    for m in months:
        m_reps = max(1, round(start_reps + monthly_headcount_growth * (m - 1)))
        base = run_scenario(m_reps, **{**engine_params, "dso": m_dso, "gross_margin": m_margin})
        sales = base["actual_sales"] * ((1 + monthly_sales_growth) ** (m - 1))
        gross_profit = sales * m_margin
        payroll = m_reps * cost_per_rep
        overhead = base_overhead + overhead_per_extra_rep * max(m_reps - 1, 0)
        net_contribution = gross_profit - payroll - overhead

        new_credit_sales = sales * credit_sales_pct
        collections = opening_ar * collection_rate
        closing_ar = opening_ar + new_credit_sales - collections
        cash_flow = collections - payroll - overhead  # simplified operating cash view

        sim_rows.append(dict(Month=m, Headcount=m_reps, Sales=sales, GrossProfit=gross_profit,
                             Payroll=payroll, Overhead=overhead, NetContribution=net_contribution,
                             OpeningAR=opening_ar, NewCreditSales=new_credit_sales,
                             Collections=collections, ClosingAR=closing_ar, CashFlow=cash_flow))
        opening_ar = closing_ar

    sim_df = pd.DataFrame(sim_rows)

    fig12 = make_subplots(specs=[[{"secondary_y": True}]])
    fig12.add_trace(go.Scatter(x=sim_df["Month"], y=sim_df["Sales"], name="Sales"), secondary_y=False)
    fig12.add_trace(go.Scatter(x=sim_df["Month"], y=sim_df["NetContribution"], name="Net Contribution"), secondary_y=False)
    fig12.add_trace(go.Scatter(x=sim_df["Month"], y=sim_df["ClosingAR"], name="Closing AR", line=dict(dash="dot")), secondary_y=True)
    fig12.update_layout(title="12-Month Simulation: Sales, Net Contribution & Closing AR")
    fig12.update_yaxes(title_text="SAR (Sales / Net Contribution)", secondary_y=False)
    fig12.update_yaxes(title_text="Closing AR (SAR)", secondary_y=True)
    st.plotly_chart(fig12, use_container_width=True)

    money_cols_12 = ["Sales", "GrossProfit", "Payroll", "Overhead", "NetContribution",
                      "OpeningAR", "NewCreditSales", "Collections", "ClosingAR", "CashFlow"]
    st.dataframe(sim_df.style.format({c: "SAR {:,.0f}" for c in money_cols_12}),
                 use_container_width=True)

    st.markdown(
        f"**Cumulative 12-month Net Contribution:** SAR {sim_df['NetContribution'].sum():,.0f}  \n"
        f"**Cumulative 12-month Cash Flow (simplified):** SAR {sim_df['CashFlow'].sum():,.0f}  \n"
        f"**Closing AR at Month 12:** SAR {sim_df['ClosingAR'].iloc[-1]:,.0f}"
    )


# ============================================================================
# 16. MACHINE LEARNING (synthetic-data demo only)
# ============================================================================
with st.expander("🤖 Machine Learning Module (synthetic data demo — not real predictions)"):
    st.markdown(
        "This module trains a Random Forest **only on synthetic observations generated "
        "by the transparent business-logic engine above**. It is included to show how the "
        "application can later be pointed at real historical company data to learn actual "
        "productivity decay, target-achievement probability, and collection rates."
    )

    if not SKLEARN_AVAILABLE:
        st.warning("`scikit-learn` is not installed. Run `pip install scikit-learn` to enable ML features.")
    else:
        np.random.seed(42)
        sample_size = 300

        # Generate synthetic historical features
        syn_reps = np.random.randint(1, 20, size=sample_size)
        syn_capacity = np.random.choice([2_500_000, 3_000_000, 3_500_000], size=sample_size)
        syn_target = np.random.choice([500_000, 600_000, 700_000], size=sample_size)

        syn_actual_sales = []
        for r, c, t in zip(syn_reps, syn_capacity, syn_target):
            res = run_scenario(
                n_reps=r, company_capacity=c, target_per_rep=t,
                decline_pct=productivity_decline_pct, gross_margin=gross_margin,
                cost_per_rep=cost_per_rep, base_overhead=base_overhead,
                overhead_per_extra_rep=overhead_per_extra_rep, dso=dso,
                credit_sales_pct=credit_sales_pct, inventory_days=inventory_days,
                ap_days=ap_days, min_target_achievement=min_target_achievement
            )
            # Add synthetic noise (+/- 5%)
            noise = np.random.normal(1.0, 0.05)
            syn_actual_sales.append(res["actual_sales"] * noise)

        X = pd.DataFrame({
            "salespeople": syn_reps,
            "company_capacity": syn_capacity,
            "target_per_rep": syn_target
        })
        y = np.array(syn_actual_sales)

        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X, y)

        st.success("Random Forest model trained successfully on synthetic data!")

        # Predict based on current sidebar inputs
        current_input = pd.DataFrame([{
            "salespeople": n_reps,
            "company_capacity": company_capacity,
            "target_per_rep": target_per_rep
        }])
        ml_pred = rf_model.predict(current_input)[0]

        ml_c1, ml_c2 = st.columns(2)
        ml_c1.metric("Formula-Based Actual Sales", f"SAR {current['actual_sales']:,.0f}")
        ml_c2.metric("ML Model Predicted Sales", f"SAR {ml_pred:,.0f}",
                      delta=f"{((ml_pred - current['actual_sales']) / max(current['actual_sales'], EPS)) * 100:.1f}% vs Formula)