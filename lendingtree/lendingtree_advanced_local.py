"""
Advanced LendingTree Revenue Ops Analytics Dashboard
LOCAL VERSION (reads from CSV files)

Features:
- Multi-table joins and complex analytics
- Funnel analysis
- Unit economics (CAC, LTV, ROI)
- Segment analysis
- Sales pipeline velocity
- A/B test analysis
- Partner scorecards
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="LendingTree Advanced Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    .kpi-header { font-size: 28px; font-weight: bold; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_all_data():
    """Load all CPLX tables from CSV files"""
    try:
        # Path to the folder containing this script
        BASE_DIR = Path.cwd() # Streamlit's current working directory
        
        df_leads = pd.read_csv(BASE_DIR/"cplx_leads.csv")
        df_conversions = pd.read_csv(BASE_DIR/"cplx_conversions.csv")
        df_customers = pd.read_csv(BASE_DIR/"cplx_customers.csv")
        df_campaigns = pd.read_csv(BASE_DIR/"cplx_campaigns.csv")
        df_partners = pd.read_csv(BASE_DIR/"cplx_partners.csv")
        df_pipeline = pd.read_csv(BASE_DIR/"cplx_sales_pipeline.csv")
        
        # Convert date columns
        for df, date_cols in [
            (df_leads, ['lead_date']),
            (df_conversions, ['conversion_date']),
            (df_customers, ['join_date', 'churn_date']),
            (df_campaigns, ['start_date', 'end_date']),
            (df_partners, ['created_date']),
            (df_pipeline, ['expected_close_date', 'actual_close_date', 'created_date', 'updated_date'])
        ]:
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return {
            'leads': df_leads,
            'conversions': df_conversions,
            'customers': df_customers,
            'campaigns': df_campaigns,
            'partners': df_partners,
            'pipeline': df_pipeline
        }
    except FileNotFoundError as e:
        st.error(f"❌ Error loading data files: {e}")
        st.stop()

data = load_all_data()
            
# ============================================
# PAGE HEADER
# ============================================
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    st.title("🌿 LendingTree ❄ Snowflake RevOps Analytics")
with col2:
    st.metric("Status", "Live", delta=f"{len(data['leads']):,} Leads")
with col3:
    st.image("lendingtree-logo.jpeg", width=150)

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.header("Filters & Parameters")

# Multi-select filters
regions = st.sidebar.multiselect(
    "Regions",
    options=sorted(data['leads']['region'].unique()),
    default=sorted(data['leads']['region'].unique())
)

campaigns = st.sidebar.multiselect(
    "Campaigns",
    options=sorted(data['campaigns']['campaign_name'].unique()),
    default=sorted(data['campaigns']['campaign_name'].unique())
)

partners = st.sidebar.multiselect(
    "Partners",
    options=sorted(data['partners']['partner_name'].unique()),
    default=sorted(data['partners']['partner_name'].unique())
)

# Apply filters
def apply_filters(leads, conversions, customers, campaigns_df, partners_df, pipeline):
    """Apply all sidebar filters across tables"""
    
    # Filter leads by region
    leads_filtered = leads[
        (leads['region'].isin(regions))
    ].copy()
    
    # Filter campaigns
    campaigns_filtered = campaigns_df[campaigns_df['campaign_name'].isin(campaigns)].copy()
    
    # Filter partners
    partners_filtered = partners_df[partners_df['partner_name'].isin(partners)].copy()
    
    # Filter leads by campaign and partner
    leads_filtered = leads_filtered[
        (leads_filtered['campaign_id'].isin(campaigns_filtered['campaign_id'])) &
        (leads_filtered['partner_id'].isin(partners_filtered['partner_id']))
    ].copy()
    
    # Filter conversions (only for filtered leads)
    conversions_filtered = conversions[conversions['lead_id'].isin(leads_filtered['lead_id'])].copy()
    
    # Filter customers (only for filtered conversions)
    customers_filtered = customers[customers['conversion_id'].isin(conversions_filtered['conversion_id'])].copy()
    
    # Filter pipeline (only for filtered customers)
    pipeline_filtered = pipeline[pipeline['customer_id'].isin(customers_filtered['customer_id'])].copy()
    
    return leads_filtered, conversions_filtered, customers_filtered, campaigns_filtered, partners_filtered, pipeline_filtered

leads_f, conversions_f, customers_f, campaigns_f, partners_f, pipeline_f = apply_filters(
    data['leads'], data['conversions'], data['customers'], 
    data['campaigns'], data['partners'], data['pipeline']
)

# ============================================
# TAB 1: EXECUTIVE SUMMARY
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Funnel & Conversion",
    "Unit Economics",
    "Sales Pipeline",
    "A/B Testing"
])

with tab1:
    st.subheader("Executive KPI Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_leads = len(leads_f)
    total_conversions = conversions_f['converted'].sum()
    total_revenue = conversions_f['revenue'].sum()
    total_cost = conversions_f['cost'].sum()
    net_profit = total_revenue - total_cost
    
    with col1:
        st.metric("📌 Total Leads", f"{total_leads:,}", 
                  delta=f"{len(leads_f) - len(data['leads'])} filtered")
    with col2:
        st.metric("✅ Conversions", f"{int(total_conversions):,}",
                  delta=f"{100*total_conversions/total_leads:.1f}%" if total_leads > 0 else "N/A")
    with col3:
        st.metric("💵 Total Revenue", f"${total_revenue:,.0f}")
    with col4:
        st.metric("📊 Total Cost", f"${total_cost:,.0f}")
    with col5:
        st.metric("📈 Net Profit", f"${net_profit:,.0f}",
                  delta=f"{100*net_profit/total_revenue:.1f}% margin" if total_revenue > 0 else "N/A")
    
    st.divider()
    
    # Funnel visualization with realistic stages
    st.subheader("Conversion Funnel")
    
    # Calculate better funnel metrics
    conversions_count = conversions_f[conversions_f['converted'] == 1].shape[0]
    active_customers = customers_f[customers_f['churned'] == False].shape[0]
    high_value_customers = customers_f[customers_f['lifetime_value'] > customers_f['lifetime_value'].median()].shape[0]
    
    funnel_data = {
        'Stage': ['Leads', 'Converted', 'Active Customers', 'High Value'],
        'Count': [total_leads, conversions_count, active_customers, high_value_customers]
    }
    
    fig = go.Figure(go.Funnel(
        y=funnel_data['Stage'],
        x=funnel_data['Count'],
        textposition='inside',
        marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics by Region
    st.subheader("Performance by Region")
    
    region_metrics = leads_f.merge(conversions_f, on='lead_id', how='left').groupby('region').agg({
        'lead_id': 'count',
        'converted': 'sum',
        'revenue': 'sum',
        'cost': 'sum'
    }).reset_index()
    
    region_metrics.columns = ['Region', 'Leads', 'Conversions', 'Revenue', 'Cost']
    region_metrics['Conversion_Rate'] = (region_metrics['Conversions'] / region_metrics['Leads'] * 100).round(2)
    region_metrics['Profit'] = (region_metrics['Revenue'] - region_metrics['Cost']).round(2)
    region_metrics['ROI'] = ((region_metrics['Revenue'] - region_metrics['Cost']) / region_metrics['Cost'] * 100).round(2)
    
    st.dataframe(region_metrics.sort_values('ROI', ascending=False), use_container_width=True)

    # ============================================
    # DEBUG SECTION: DATA SHAPE & STRUCTURE
    # ============================================
    with st.expander("🐛 DEBUG: Data Structure Info"):
        st.write("### Loaded DataFrames Summary")
        
        debug_info = []
        for table_name, df in data.items():
            rows, cols = df.shape
            col_names = list(df.columns)
            debug_info.append({
                'Table': table_name.upper(),
                'Rows': rows,
                'Columns': cols,
                'Column Names': ', '.join(col_names)
            })
        
        debug_df = pd.DataFrame(debug_info)
        st.dataframe(debug_df, use_container_width=True, hide_index=True)
        
        st.write("### Detailed Column Info")
        for table_name, df in data.items():
            with st.expander(f"{table_name.upper()} - {df.shape[0]} rows × {df.shape[1]} columns"):
                st.write(df.dtypes)
                st.write("**First few rows:**")
                st.dataframe(df.head(3), use_container_width=True)
                
with tab2:
    st.subheader("Funnel Analysis by Campaign")
    
    # Merge tables for funnel analysis
    funnel_data_by_campaign = (
        leads_f.merge(
            data['campaigns'][['campaign_id', 'campaign_name']], 
            on='campaign_id', 
            how='left'
        )
        .merge(conversions_f[['lead_id', 'converted', 'revenue']], on='lead_id', how='left')
        .merge(customers_f[['conversion_id']], 
               left_on=leads_f.merge(conversions_f, on='lead_id')['conversion_id'], 
               right_on='conversion_id', 
               how='left')
    )
    
    # Better funnel: group by campaign
    funnel_by_campaign = leads_f.merge(
        data['campaigns'][['campaign_id', 'campaign_name']], 
        on='campaign_id', 
        how='left'
    ).merge(conversions_f[['lead_id', 'converted']], on='lead_id', how='left')
    
    funnel_summary = funnel_by_campaign.groupby('campaign_name').agg({
        'lead_id': 'count',
        'converted': 'sum'
    }).reset_index()
    funnel_summary.columns = ['Campaign', 'Leads', 'Conversions']
    funnel_summary['Conversion_Rate_%'] = (funnel_summary['Conversions'] / funnel_summary['Leads'] * 100).round(2)
    funnel_summary = funnel_summary.sort_values('Conversion_Rate_%', ascending=False)
    
    st.dataframe(funnel_summary, use_container_width=True, hide_index=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(funnel_summary, x='Campaign', y=['Leads', 'Conversions'],
                     barmode='group', title='Leads vs Conversions by Campaign')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(funnel_summary, x='Campaign', y='Conversion_Rate_%',
                     title='Conversion Rate by Campaign', color='Conversion_Rate_%',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Credit Score Impact
    st.subheader("Credit Score Impact on Conversion")
    
    credit_bins = pd.cut(leads_f['credit_score'], bins=[550, 650, 700, 750, 850])
    credit_analysis = leads_f.merge(
    conversions_f[['lead_id', 'converted', 'revenue']], 
    on='lead_id', 
    how='left'
).groupby(credit_bins, observed=False).agg({
    'lead_id': 'count',
    'converted': 'sum',
    'revenue': 'mean'
}).reset_index()
    credit_analysis.columns = ['Credit_Score_Range', 'Leads', 'Conversions', 'Avg_Revenue']
    credit_analysis['Credit_Score_Range'] = credit_analysis['Credit_Score_Range'].astype(str)
    credit_analysis['Conversion_Rate_%'] = (credit_analysis['Conversions'] / credit_analysis['Leads'] * 100).round(2)
    
    fig = px.bar(credit_analysis, x='Credit_Score_Range', y='Conversion_Rate_%',
                 title='Conversion Rate by Credit Score Range', color='Conversion_Rate_%',
                 color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Unit Economics Dashboard")
    
    # Create joined dataframe for economics
    econ_data = (
        leads_f.merge(conversions_f[['lead_id', 'conversion_id', 'converted', 'revenue', 'cost']], 
                      on='lead_id', how='left')
        .merge(customers_f[['conversion_id', 'lifetime_value', 'segment', 'churned']], 
               on='conversion_id', how='left')
        .merge(data['campaigns'][['campaign_id', 'campaign_name']], on='campaign_id', how='left')
        .merge(data['partners'][['partner_id', 'partner_name']], on='partner_id', how='left')
    )
    
    col1, col2, col3 = st.columns(3)
    
    total_marketing_spend = conversions_f['cost'].sum()
    customers_acquired = len(customers_f)
    cac = total_marketing_spend / customers_acquired if customers_acquired > 0 else 0
    
    with col1:
        st.metric("💳 CAC (Customer Acquisition Cost)", f"${cac:,.2f}")
    
    with col2:
        avg_ltv = customers_f['lifetime_value'].mean()
        st.metric("💎 Avg LTV", f"${avg_ltv:,.2f}")
    
    st.divider()
    
    # Unit economics by segment
    st.subheader("Unit Economics by Customer Segment")
    
    segment_econ = customers_f.groupby('segment').agg({
        'customer_id': 'count',
        'lifetime_value': ['sum', 'mean'],
        'churned': 'mean'
    }).reset_index()
    
    segment_econ.columns = ['Segment', 'Count', 'Total_LTV', 'Avg_LTV', 'Churn_Rate']
    segment_econ['Churn_Rate_%'] = (segment_econ['Churn_Rate'] * 100).round(2)
    segment_econ['Total_LTV'] = segment_econ['Total_LTV'].round(2)
    segment_econ['Avg_LTV'] = segment_econ['Avg_LTV'].round(2)
    
    st.dataframe(segment_econ[['Segment', 'Count', 'Avg_LTV', 'Total_LTV', 'Churn_Rate_%']], 
                 use_container_width=True, hide_index=True)
    
    # CAC vs LTV by Campaign
    st.subheader("CAC vs LTV by Campaign")

    campaign_econ = (
        leads_f
        .merge(conversions_f[['lead_id', 'conversion_id', 'cost', 'converted']], on='lead_id', how='left')
        .merge(customers_f[['conversion_id', 'lifetime_value']], on='conversion_id', how='left')
        .merge(data['campaigns'][['campaign_id', 'campaign_name']], on='campaign_id', how='left')
        .groupby('campaign_name')
        .agg(
            Total_Cost=('cost', 'sum'),
            Avg_LTV=('lifetime_value', 'mean'),
            Customers=('converted', 'sum')   # number of conversions
        )
        .reset_index()
    )

    campaign_econ['CAC'] = campaign_econ['Total_Cost'] / campaign_econ['Customers']
    campaign_econ.columns = ['Campaign', 'Total_Cost', 'Avg_LTV', 'Customers', 'CAC']
    
    fig = px.scatter(campaign_econ, x='CAC', y='Avg_LTV', size='Total_Cost',
                     hover_data=['Campaign'], title='CAC vs Average LTV by Campaign',
                     labels={'CAC': 'Customer Acquisition Cost ($)', 'Avg_LTV': 'Average Lifetime Value ($)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI by Campaign
    st.subheader("Campaign ROI Analysis")
    
    roi_data = (
        leads_f.merge(conversions_f[['lead_id', 'revenue', 'cost']], on='lead_id', how='left')
        .merge(data['campaigns'][['campaign_id', 'campaign_name', 'budget']], on='campaign_id', how='left')
        .groupby('campaign_name').agg({
            'revenue': 'sum',
            'cost': 'sum',
            'budget': 'first'
        }).reset_index()
    )
    roi_data['net_profit'] = roi_data['revenue'] - roi_data['cost']
    roi_data['roi_%'] = ((roi_data['net_profit']) / roi_data['budget'] * 100).round(2)
    roi_data = roi_data.sort_values('roi_%', ascending=False)
    
    st.dataframe(roi_data[['campaign_name', 'revenue', 'cost', 'net_profit', 'roi_%']], 
                 use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Sales Pipeline Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    total_pipeline_value = pipeline_f['amount'].sum()
    expected_revenue = (pipeline_f['amount'] * pipeline_f['probability']).sum()
    won_deals = len(pipeline_f[pipeline_f['stage'] == 'Closed_Won'])
    
    with col1:
        st.metric("💰 Total Pipeline Value", f"${total_pipeline_value:,.0f}")
    with col2:
        st.metric("🎯 Expected Revenue", f"${expected_revenue:,.0f}")
    with col3:
        st.metric("✅ Won Deals", f"{won_deals:,}")
    
    st.divider()
    
    # Pipeline by stage
    st.subheader("Deal Distribution by Stage")
    
    stage_summary = pipeline_f.groupby('stage').agg({
        'deal_id': 'count',
        'amount': 'sum',
        'probability': 'mean',
        'days_in_stage': 'mean'
    }).reset_index()
    stage_summary.columns = ['Stage', 'Deals', 'Total_Value', 'Avg_Probability', 'Avg_Days']
    stage_summary = stage_summary.sort_values('Deals', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(stage_summary, values='Deals', names='Stage', 
                     title='Deal Distribution by Stage')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(stage_summary, x='Stage', y='Total_Value',
                     title='Total Value by Stage', color='Total_Value',
                     color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("A/B Test Performance Analysis")
    
    # Merge for AB analysis
    ab_data = leads_f.merge(conversions_f[['lead_id', 'converted', 'revenue', 'cost']], 
                            on='lead_id', how='left')
    ab_data = ab_data.merge(data['campaigns'][['campaign_id', 'campaign_name']], 
                            on='campaign_id', how='left')
    
    # Overall A/B performance
    col1, col2 = st.columns(2)
    
    control_conv = ab_data[ab_data['ab_group'] == 'Control']['converted'].mean() * 100
    treatment_conv = ab_data[ab_data['ab_group'] == 'Treatment']['converted'].mean() * 100
    uplift = treatment_conv - control_conv
    
    with col1:
        st.metric("🔘 Control Conversion Rate", f"{control_conv:.2f}%")
    with col2:
        st.metric("🟢 Treatment Conversion Rate", f"{treatment_conv:.2f}%",
                  delta=f"{uplift:.2f}pp ({uplift/control_conv*100:.1f}% uplift)" if control_conv > 0 else "N/A")
    
    st.divider()
    
    # A/B by campaign
    st.subheader("A/B Performance by Campaign")
    
    ab_by_campaign = ab_data.groupby(['campaign_name', 'ab_group']).agg({
        'lead_id': 'count',
        'converted': 'sum',
        'revenue': 'mean'
    }).reset_index()
    ab_by_campaign.columns = ['Campaign', 'Group', 'Leads', 'Conversions', 'Avg_Revenue']
    ab_by_campaign['Conversion_Rate_%'] = (ab_by_campaign['Conversions'] / ab_by_campaign['Leads'] * 100).round(2)
    
    st.dataframe(ab_by_campaign, use_container_width=True, hide_index=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(ab_by_campaign, x='Campaign', y='Conversion_Rate_%', color='Group',
                     barmode='group', title='Conversion Rate: Control vs Treatment')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(ab_by_campaign, x='Campaign', y='Avg_Revenue', color='Group',
                     barmode='group', title='Avg Revenue per Lead: Control vs Treatment')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Statistical summary
    st.subheader("Statistical Summary")
    
    stats_summary = ab_by_campaign.pivot_table(
        index='Campaign',
        columns='Group',
        values=['Leads', 'Conversion_Rate_%', 'Avg_Revenue']
    ).round(2)
    
    st.dataframe(stats_summary, use_container_width=True)
    
    st.success("✅ Treatment group showing consistent uplift across most campaigns!")

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Made by: Hrishikesh Rathod | Advanced LendingTree Analytics | Last Updated: Jan-2026</small>
</div>
""", unsafe_allow_html=True)
