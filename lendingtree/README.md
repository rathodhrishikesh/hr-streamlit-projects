# 🌿 LendingTree Revenue Ops Analytics

**Live Demo:** [https://hr-lendingtree-revops.streamlit.app/](https://hr-lendingtree-revops.streamlit.app/)

A Streamlit dashboard for advanced LendingTree Revenue Operations analytics.  

Features include:
- Multi-table joins and complex analytics
- Funnel analysis
- Unit economics (CAC, LTV, ROI)
- Segment analysis
- Sales pipeline velocity
- A/B test analysis
- Partner and campaign scorecards

## Project Screenshots

### LendingTree Database Diagram
![LendingTree Database Diagram](images/LendingTree%20Database%20Diagram.png)

### 🌿 LendingTree ❄ Snowflake - Dashboard 1
![🌿 LendingTree ❄ Snowflake 1](images/Snowflake%20LendingTree%20Streamlit%201.png)

### 🌿 LendingTree ❄ Snowflake - Dashboard 2
![🌿 LendingTree ❄ Snowflake 2](images/Snowflake%20LendingTree%20Streamlit%202.png)


## Data Description

The dashboard reads from 6 CSV files (**120,000 rows total**):

| File | Description | Rows |
|------|-------------|------|
| `cplx_partners.csv` | Partner metadata | 5 |
| `cplx_campaigns.csv` | Campaign details | 20 |
| `cplx_leads.csv` | Leads data | 120k |
| `cplx_conversions.csv` | Lead conversions | 120k |
| `cplx_customers.csv` | Converted customers | ~35k+ |
| `cplx_sales_pipeline.csv` | Sales pipeline deals | ~66k+ |
