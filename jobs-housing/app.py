import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import os

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="By Right County Dashboard",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Branding CSS
st.markdown(f"""
    <style>
    .main {{
        background-color: #FFFFFF;
    }}
    .stMetric {{
        background-color: #CED0CE;
        padding: 15px;
        border-radius: 5px;
    }}
    h1, h2, h3 {{
        color: #33683A !important;
    }}
    .stButton>button {{
        background-color: #33683A;
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# Logo and Title Header
col1, col2 = st.columns([1, 6])
with col1:
    if os.path.exists("ByRIGHT-small.png"):
        st.image("ByRIGHT-small.png", width=120)
with col2:
    st.title("By Right County Dashboard")

st.markdown("---")

# -----------------------
# DATA LOAD (GitHub Hosted)
# -----------------------
JH_URL = "https://github.com/kevinverhoff/by_right/raw/main/jobs-housing/county_jobs_housing.parquet"
LODES_URL = "https://github.com/kevinverhoff/by_right/raw/main/jobs-housing/lodes_commuting.parquet"

@st.cache_data
def load_geo_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    return requests.get(url).json()

@st.cache_data
def load_states_geojson():
    url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
    return requests.get(url).json()

@st.cache_data(ttl=3600)
def load_data():
    # Try GitHub first for Cloud compatibility, fallback to local
    try:
        df_jh = pd.read_parquet(JH_URL)
    except Exception:
        df_jh = pd.read_parquet("county_jobs_housing.parquet")
    
    try:
        df_lodes = pd.read_parquet(LODES_URL)
        df_lodes = df_lodes.rename(columns={
            "county_name": "county_name_lodes",
            "state_name": "state_name_lodes",
            "full_name": "full_name_lodes",
            "state": "state_fips_lodes"
        })
    except Exception:
        try:
            df_lodes = pd.read_parquet("lodes_commuting.parquet")
            df_lodes = df_lodes.rename(columns={
                "county_name": "county_name_lodes",
                "state_name": "state_name_lodes",
                "full_name": "full_name_lodes",
                "state": "state_fips_lodes"
            })
        except Exception:
            df_lodes = pd.DataFrame(columns=['fips', 'year', 'in_commuters', 'out_commuters', 'net_commute', 'lodes_total_jobs', 'internal_workers'])

    df = pd.merge(df_jh, df_lodes, on=["fips", "year"], how="outer")
    if "county_name_lodes" in df.columns:
        df["county_name"] = df["county_name"].fillna(df["county_name_lodes"])
        df["state_name"] = df["state_name"].fillna(df["state_name_lodes"])
        df["full_name"] = df["full_name"].fillna(df["full_name_lodes"])
        df["state"] = df["state"].fillna(df["state_fips_lodes"])
    df["commuter_ratio"] = df["in_commuters"] / df["out_commuters"].replace(0, np.nan)
    df["in_commuter_share"] = df["in_commuters"] / df["lodes_total_jobs"].replace(0, np.nan)
    df["total_residents_working"] = df["internal_workers"] + df["out_commuters"]
    df["resident_retention"] = df["internal_workers"] / df["total_residents_working"].replace(0, np.nan)
    return df

df = load_data()
counties_geojson = load_geo_data()
states_geojson = load_states_geojson()

if df.empty:
    st.error("No data found. Please run the scrapers first.")
    st.stop()

# -----------------------
# SIDEBAR / NAVIGATION
# -----------------------
st.sidebar.title("Navigation")
metric_category = st.sidebar.radio("Select Metric Category", ["Housing vs Jobs", "Commuter Flows", "Demographics"])

if metric_category == "Commuter Flows":
    view_mode = st.sidebar.selectbox("View Mode", ["Absolute (Net Flow)", "Commuter Ratio (In/Out)", "In-Commuter Job Share (% of Jobs)", "Resident Retention Share (% of Residents working in county)"])
elif metric_category == "Demographics":
    view_mode = st.sidebar.selectbox("View Mode", ["Average Age", "% Residents Under 18", "% Residents 18-22", "% Residents 23-34", "% Residents 35-49", "% Residents 50-64", "% Residents Over 65", "% Working Age (18-64)"])
else:
    view_mode = "Standard"

# Determine columns early
if metric_category == "Demographics":
    if "Average Age" in view_mode: main_metric_col = "avg_age"
    elif "Under 18" in view_mode: main_metric_col = "pct_under18"
    elif "18-22" in view_mode: main_metric_col = "pct_18_22"
    elif "23-34" in view_mode: main_metric_col = "pct_23_34"
    elif "35-49" in view_mode: main_metric_col = "pct_35_49"
    elif "50-64" in view_mode: main_metric_col = "pct_50_64"
    elif "Over 65" in view_mode: main_metric_col = "pct_over65"
    else: main_metric_col = "pct_working_age"
elif metric_category == "Housing vs Jobs":
    main_metric_col = "housing_per_job"
else:
    if "Ratio" in view_mode: main_metric_col = "commuter_ratio"
    elif "Share" in view_mode: main_metric_col = "in_commuter_share" if "In-Commuter" in view_mode else "resident_retention"
    else: main_metric_col = "net_commute"

# Highlight County Early
all_counties = sorted(df.dropna(subset=[main_metric_col])["full_name"].unique())
highlight_county = st.sidebar.selectbox("Highlight County", ["None"] + all_counties)

st.sidebar.markdown("---")
st.sidebar.title("Geography")

if main_metric_col in df.columns:
    available_years = sorted(df.dropna(subset=[main_metric_col])["year"].unique())
else:
    available_years = []

if main_metric_col in df.columns and "state_name" in df.columns:
    valid_states_df = df.dropna(subset=[main_metric_col, "state_name"])
    state_names = sorted(valid_states_df["state_name"].unique())
else:
    state_names = []
selected_state_names = st.sidebar.multiselect("States", state_names, default=state_names)

st.sidebar.markdown("---")
st.sidebar.title("Time")

selected_years = st.sidebar.multiselect("Years", available_years, default=[max(available_years)] if available_years else [])
selected_period = None
if metric_category == "Housing vs Jobs":
    if "period" in df.columns:
        period_df = df[df["year"].isin(selected_years)].dropna(subset=["housing_per_job"])
        available_periods = sorted(period_df["period"].unique())
        selected_period = st.sidebar.selectbox("Period", available_periods, index=len(available_periods)-1 if available_periods else 0)

st.sidebar.markdown("---")
# -----------------------
# CONFIG
# -----------------------
color_args = {}
labels_map = {"metric": ""}
if metric_category == "Housing vs Jobs":
    st.subheader("Jobs vs Housing Balance")
    metric_label, color_scale = "Housing Units per Job", "Viridis"
    hover_data_config = {"full_name": True, "metric": ":.3f", "jobs": ":,.0f", "housing_units": ":,.0f"}
    interpretation_text = "Values < 0.9 indicate potential housing pressure relative to local employment."
    labels_map = {"metric": metric_label}
elif metric_category == "Demographics":
    st.subheader(f"Demographics: {view_mode}")
    metric_label, color_scale = view_mode, "Magma"
    count_col_map = {"% Residents Under 18": "count_under18", "% Residents 18-22": "count_18_22", "% Residents 23-34": "count_23_34", "% Residents 35-49": "count_35_49", "% Residents 50-64": "count_50_64", "% Residents Over 65": "count_over65", "% Working Age (18-64)": "count_working_age"}
    hover_data_config = {"full_name": True, "state_name": True, "metric": ":.1%" if "Age" not in view_mode else ":.1f", "B01001_001E": ":,.0f"}
    labels_map = {"B01001_001E": "Total Population", "metric": metric_label}
    if view_mode in count_col_map:
        c_col = count_col_map[view_mode]
        if c_col in df.columns:
            hover_data_config[c_col] = ":,.0f"
            labels_map[c_col] = "Residents in range"
    interpretation_text = f"Visualizing {view_mode} across counties based on ACS 5-Year estimates."
else:
    if "Ratio" in view_mode:
        st.subheader("Commuter Ratio (Inflow / Outflow)")
        metric_label, color_scale = "In/Out Ratio", "RdBu"
        hover_data_config = {"full_name": True, "metric": ":.2f", "in_commuters": ":,.0f", "out_commuters": ":,.0f"}
        interpretation_text = "**Ratio > 1.0**: More people enter than leave.  \n**Ratio < 1.0**: More people leave than enter."
    elif "In-Commuter Job Share" in view_mode:
        st.subheader("In-Commuter Job Share")
        metric_label, color_scale = "% of Jobs held by In-Commuters", "YlGnBu"
        hover_data_config = {"full_name": True, "metric": ":.1%", "in_commuters": ":,.0f", "lodes_total_jobs": ":,.0f"}
        interpretation_text = "High % indicates the local workforce is primarily composed of people living outside the county."
    elif "Resident Retention" in view_mode:
        st.subheader("Resident Retention Share")
        metric_label, color_scale = "% of Residents working in County", "Purples"
        hover_data_config = {"full_name": True, "metric": ":.1%", "internal_workers": ":,.0f", "total_residents_working": ":,.0f"}
        interpretation_text = "High % means most employed residents stay in their home county for work."
    else:
        st.subheader("Total Net Commuter Flow")
        metric_label, color_scale = "Net Commuters (In - Out)", "RdBu"
        hover_data_config = {"full_name": True, "metric": ":,.0f", "in_commuters": ":,.0f", "out_commuters": ":,.0f"}
        interpretation_text = "Positive numbers mean the county is a net importer of labor."
    labels_map = {"metric": metric_label}

st.sidebar.title("How to interpret")
st.sidebar.markdown(interpretation_text)

# -----------------------
# FILTER & AGGREGATE
# -----------------------
if not available_years: st.error(f"No years found for {main_metric_col}"); st.stop()
mask = (df["state_name"].isin(selected_state_names)) & (df["year"].isin(selected_years))
if metric_category == "Housing vs Jobs" and selected_period: mask = mask & (df["period"] == selected_period)
filtered = df[mask].copy()
filtered["metric"] = filtered[main_metric_col]
if metric_category in ["Commuter Flows", "Demographics"]:
    group_cols = ["fips", "state", "state_name", "county_name", "full_name"]
    cols_to_agg = ["metric", "in_commuters", "out_commuters", "net_commute", "lodes_total_jobs", "internal_workers", "total_residents_working", "B01001_001E"] + [c for c in filtered.columns if c.startswith("pct_") or c.startswith("count_")]
    filtered = filtered.groupby(group_cols, as_index=False).agg({c: "mean" for c in cols_to_agg if c in filtered.columns})
filtered = filtered.dropna(subset=["metric"])

# -----------------------
# SUMMARY
# -----------------------
if not filtered.empty:
    if metric_category == "Housing vs Jobs":
        if "jobs" in filtered.columns and filtered["jobs"].sum() > 0: st.info(f"**Weighted regional ratio:** {filtered['housing_units'].sum() / filtered['jobs'].sum():.2f}")
    elif metric_category == "Demographics":
        if "B01001_001E" in filtered.columns and filtered["B01001_001E"].sum() > 0:
            pop_total = filtered["B01001_001E"].sum()
            avg = (filtered["metric"] * filtered["B01001_001E"]).sum() / pop_total
            label = "Average Age" if "Age" in view_mode else view_mode
            formatted_avg = f"{avg:.1f} years" if "Age" in view_mode else f"{avg:.1%}"
            st.info(f"**Regional Weighted {label}:** {formatted_avg}")
    else:
        if "In-Commuter Job Share" in view_mode: st.info(f"**Regional In-Commuter Share:** {filtered['in_commuters'].sum() / filtered['lodes_total_jobs'].sum():.1%}")
        elif "Resident Retention" in view_mode: st.info(f"**Regional Resident Retention:** {filtered['internal_workers'].sum() / filtered['total_residents_working'].sum():.1%}")
        else: st.info(f"**Avg Regional In:** {filtered['in_commuters'].mean():,.0f} | **Avg Regional Out:** {filtered['out_commuters'].mean():,.0f}")
else: st.warning("No data for these filters.")

# -----------------------
# MAP
# -----------------------
if not filtered.empty:
    st.subheader("County Map")
    if "Ratio" in view_mode:
        max_dev = max(abs(filtered["metric"].max() - 1), abs(filtered["metric"].min() - 1), 0.1)
        color_args = {"range_color": [1 - max_dev, 1 + max_dev]}
    elif "Absolute" in view_mode or (metric_category == "Commuter Flows" and main_metric_col == "net_commute"):
        limit = max(abs(filtered["metric"].min()), abs(filtered["metric"].max()), 1)
        color_args = {"range_color": [-limit, limit]}
    fig_map = px.choropleth(filtered, geojson=counties_geojson, locations="fips", color="metric", color_continuous_scale=color_scale, labels=labels_map, hover_data=hover_data_config, **color_args)
    fig_map.update_traces(marker_line_color="rgba(255,255,255,0.35)", marker_line_width=0.4)
    valid_states = df.dropna(subset=["state_name", "state"])
    if not valid_states.empty:
        state_lookup = valid_states.groupby("state_name")["state"].first()
        selected_state_fips = [state_lookup[s] for s in selected_state_names if s in state_lookup]
        fig_map.add_trace(go.Choropleth(geojson={"type": "FeatureCollection", "features": [f for f in states_geojson["features"] if f["id"] in selected_state_fips]}, locations=[f["id"] for f in filtered_states_geojson["features"] if f["id"] in selected_state_fips] if 'filtered_states_geojson' in locals() else [f["id"] for f in states_geojson["features"] if f["id"] in selected_state_fips], z=[1]*len(selected_state_fips), showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="white", marker_line_width=2, hoverinfo="skip"))
    
    # Restored Yellow Highlight
    if highlight_county != "None":
        sel_row_fips = df.loc[df["full_name"] == highlight_county, "fips"].iloc[0]
        fig_map.add_trace(go.Choropleth(
            geojson=counties_geojson,
            locations=[sel_row_fips],
            z=[1],
            showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_color="yellow", marker_line_width=3, hoverinfo="skip"
        ))

    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(height=800, margin=dict(l=0, r=0, t=30, b=0), hovermode="closest")
    st.plotly_chart(fig_map, width="stretch")

    # -----------------------
    # DISTRIBUTION
    # -----------------------
    st.subheader("Statistical Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=filtered["metric"], histnorm="probability density", nbinsx=60, opacity=0.25, name="Histogram", marker_color="gray"))
    vals = filtered["metric"].dropna().values
    if len(vals) > 5:
        try:
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 200)
            fig_dist.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode="lines", name="Density (KDE)", line=dict(width=3, color="#33683A")))
        except Exception: pass
    if highlight_county != "None":
        sel_row = filtered[filtered["full_name"] == highlight_county]
        if not sel_row.empty: fig_dist.add_vline(x=sel_row["metric"].mean(), line_width=3, line_dash="dash", line_color="red")
    fig_dist.update_layout(xaxis_title=metric_label, yaxis_title="Density", hovermode="x unified", template="plotly_white", height=450)
    st.plotly_chart(fig_dist, width="stretch")
    with st.expander("Debug Data View"): st.dataframe(filtered)
else: st.warning("No data available.")
