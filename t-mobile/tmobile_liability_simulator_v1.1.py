# tmobile_liability_simulator_v2.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---- Page Config ----
st.set_page_config(
    page_title="T-Mobile Liability & Credit Forecasting Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- T-Mobile Magenta Theme CSS ----
primary_color = "#E20074"  # T-Mobile Magenta
st.markdown(
    f"""
    <style>
    /* Sidebar Background */
    .css-1d391kg .css-1v3fvcr {{
        background-color: {primary_color};
        color: white;
        padding: 20px;
    }}

    /* Main Panel Background */
    .css-1aumxhk {{
        background-color: #ffe6f0;
    }}

    /* Buttons */
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        font-weight: bold;
    }}

    /* Sliders */
    .stSlider>div>div>div>div>div {{
        background-color: {primary_color};
    }}

    /* Number input boxes */
    input[type=number] {{
        border: 2px solid {primary_color};
        border-radius: 6px;
        padding: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar Inputs ----
st.sidebar.markdown("## ⚙️ Simulation Parameters", unsafe_allow_html=True)

program_type = st.sidebar.selectbox(
    "Program Type", ["Jump Program Liability", "Apple Forever Valuation"]
)

region = st.sidebar.selectbox(
    "Region", ["North", "South", "East", "West"]
)

customer_segment = st.sidebar.selectbox(
    "Customer Segment", ["Young Adults", "Professionals", "Families", "Seniors"]
)

# Dollar amount inputs
device_cost = st.sidebar.number_input(
    "Average Device Cost ($)", min_value=200, max_value=1500, value=800, step=50
)
contract_amount = st.sidebar.number_input(
    "Average Service Contract ($)", min_value=20, max_value=200, value=75, step=5
)

# Slider inputs
replacement_cycle = st.sidebar.slider(
    "Replacement Cycle (Months)", 6, 36, 24, step=1
)
adoption_rate = st.sidebar.slider(
    "Program Adoption Rate (%)", 0, 100, 60, step=5
)
default_probability = st.sidebar.slider(
    "Default Probability (%)", 0, 20, 5, step=1
)

st.sidebar.markdown("---")
simulate_button = st.sidebar.button("Run Simulation")

# ---- Generate Synthetic Data Functions ----
def generate_device_liability_data(months=36):
    time = np.arange(months)
    base_liability = device_cost * adoption_rate / 100 * (1 + 0.05 * np.sin(time / 6))
    return pd.DataFrame({"Month": time + 1, "Liability ($M)": base_liability})

def generate_device_value_data(months=36):
    depreciation_rate = 0.05
    values = [device_cost]
    for i in range(1, months):
        values.append(values[-1] * (1 - depreciation_rate))
    return pd.DataFrame({"Month": np.arange(1, months + 1), "Device Value ($)": values})

def generate_credit_loss_data(customers=1000):
    np.random.seed(42)
    credit_scores = np.random.randint(500, 850, customers)
    defaults = np.random.binomial(1, default_probability/100, customers)
    df = pd.DataFrame({
        "Customer ID": range(1, customers+1),
        "Credit Score": credit_scores,
        "Defaulted": defaults,
        "Contract Amount ($)": contract_amount
    })
    return df

# ---- Main Panel ----
st.title("📱 T-Mobile Device Liability & Credit Forecasting Tool")
st.markdown("<hr style='border:2px solid #E20074'>", unsafe_allow_html=True)

if simulate_button:
    st.subheader(f"Simulation Results: {program_type}")
    
    # Device Liability
    liability_df = generate_device_liability_data()
    fig_liability = px.line(
        liability_df, x="Month", y="Liability ($M)",
        title="Projected Device Liabilities Over Time",
        markers=True, template="plotly_white"
    )
    st.plotly_chart(fig_liability, use_container_width=True)
    st.metric("Total Liability ($M)", f"{liability_df['Liability ($M)'].sum():,.2f}")
    
    # Device Value Depreciation
    device_value_df = generate_device_value_data()
    fig_value = px.line(
        device_value_df, x="Month", y="Device Value ($)",
        title="Device Depreciation Curve", markers=True, template="plotly_white"
    )
    st.plotly_chart(fig_value, use_container_width=True)
    
    # Credit Loss Forecast
    credit_df = generate_credit_loss_data()
    loss_rate = credit_df['Defaulted'].mean() * 100
    st.subheader("Credit Loss Forecasting")
    st.metric("Projected Credit Loss Rate (%)", f"{loss_rate:.2f}")
    fig_credit = px.histogram(
        credit_df, x="Credit Score", color="Defaulted",
        barmode="overlay", title="Customer Credit Score vs Default",
        template="plotly_white"
    )
    st.plotly_chart(fig_credit, use_container_width=True)
    
    # Adhoc Analytics Table
    st.subheader("Adhoc Analytics Preview")
    st.dataframe(credit_df.head(10))
    
    # Download Section
    st.subheader("Download Simulation Results")
    csv_liability = liability_df.to_csv(index=False).encode()
    st.download_button("Download Liability CSV", csv_liability, "liability.csv", "text/csv")
    
    csv_credit = credit_df.to_csv(index=False).encode()
    st.download_button("Download Credit CSV", csv_credit, "credit_loss.csv", "text/csv")

else:
    st.info("Configure simulation parameters on the left and click 'Run Simulation'.")

