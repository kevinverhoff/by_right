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

# Mobile-Optimized Branding CSS
st.markdown(f"""
    <style>
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }}
    div[data-testid="stMetric"] {{
        background-color: #CED0CE;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #33683A;
    }}
    h1, h2, h3 {{
        color: #33683A !important;
    }}
    [data-testid="stHorizontalBlock"] {{
        align-items: center;
    }}
    @media (max-width: 640px) {{
        .main-title {{
            font-size: 1.5rem !important;
        }}
        .header-logo {{
            height: 2rem !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

import base64

# Header
LOGO_URL = "https://raw.githubusercontent.com/kevinverhoff/by_right/main/jobs-housing/ByRIGHT-small.png"
logo_path = "ByRIGHT-small.png"

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_logo_html(path, url):
    if os.path.exists(path):
        binary_data = get_base64_of_bin_file(path)
        return f'data:image/png;base64,{binary_data}'
    return url

display_logo = get_logo_html(logo_path, LOGO_URL)

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
        <img src="{display_logo}" class="header-logo" style="height: 3rem; width: auto; object-fit: contain;">
        <h1 class="main-title" style="margin: 0; line-height: 1;">By Right County Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------
# DATA LOAD
# -----------------------
JH_URL = "https://github.com/kevinverhoff/by_right/raw/main/jobs-housing/county_jobs_housing.parquet"
LODES_URL = "https://github.com/kevinverhoff/by_right/raw/main/jobs-housing/lodes_commuting.parquet"

FIPS_TO_ABBR = {
    "18": "IN",
    "17": "IL",
    "21": "KY",
    "26": "MI",
    "39": "OH"
}

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
    try:
        df_jh = pd.read_parquet(JH_URL)
    except Exception:
        df_jh = pd.read_parquet("county_jobs_housing.parquet")
    
    try:
        df_lodes = pd.read_parquet(LODES_URL)
        df_lodes = df_lodes.rename(columns={
            "county_name": "county_name_lodes", "state_name": "state_name_lodes",
            "full_name": "full_name_lodes", "state": "state_fips_lodes"
        })
    except Exception:
        try:
            df_lodes = pd.read_parquet("lodes_commuting.parquet")
            df_lodes = df_lodes.rename(columns={
                "county_name": "county_name_lodes", "state_name": "state_name_lodes",
                "full_name": "full_name_lodes", "state": "state_fips_lodes"
            })
        except Exception:
            df_lodes = pd.DataFrame(columns=['fips', 'year', 'in_commuters', 'out_commuters', 'net_commute', 'lodes_total_jobs', 'internal_workers', 'state_abbr'])

    df = pd.merge(df_jh, df_lodes, on=["fips", "year"], how="outer")
    
    # Ensure state_abbr is populated for all rows based on FIPS
    df["state_abbr"] = df["fips"].str[:2].map(FIPS_TO_ABBR)
    
    if "county_name_lodes" in df.columns:
        df["county_name"] = df["county_name"].fillna(df["county_name_lodes"])
        df["state_name"] = df["state_name"].fillna(df["state_name_lodes"])
        df["full_name"] = df["full_name"].fillna(df["full_name_lodes"])
        df["state"] = df["state"].fillna(df["state_fips_lodes"])
    
    # Core Metrics
    df["commuter_ratio"] = df["in_commuters"] / df["out_commuters"].replace(0, np.nan)
    df["in_commuter_share"] = df["in_commuters"] / df["lodes_total_jobs"].replace(0, np.nan)
    df["total_residents_working"] = df["internal_workers"] + df["out_commuters"]
    df["resident_retention"] = df["internal_workers"] / df["total_residents_working"].replace(0, np.nan)
    df["people_per_housing"] = df["B01001_001E"] / df["housing_units"].replace(0, np.nan)
    df["jobs_per_capita"] = df["jobs"] / df["B01001_001E"].replace(0, np.nan)
    df["jobs_per_working_age"] = df["jobs"] / df["count_working_age"].replace(0, np.nan)
    
    return df

df = load_data()
counties_geojson = load_geo_data()
states_geojson = load_states_geojson()

# -----------------------
# SIDEBAR / NAVIGATION
# -----------------------
st.sidebar.title("Navigation")
metric_category = st.sidebar.radio("Category", ["Housing vs Jobs", "Commuter Flows", "Demographics"])

if metric_category == "Housing vs Jobs":
    view_mode = st.sidebar.selectbox("View Mode", ["Housing Units per Job", "People per Housing Unit", "Jobs per Capita", "Jobs per Working Age Adult"])
elif metric_category == "Commuter Flows":
    view_mode = st.sidebar.selectbox("View Mode", ["Absolute (Net Flow)", "Commuter Ratio (In/Out)", "In-Commuter Job Share (% of Jobs)", "Resident Retention Share (% of Residents)"])
elif metric_category == "Demographics":
    view_mode = st.sidebar.selectbox("View Mode", ["Average Age", "% Residents Under 18", "% Residents 18-22", "% Residents 23-34", "% Residents 35-49", "% Residents 50-64", "% Residents Over 65", "% Working Age (18-64)"])

if metric_category == "Housing vs Jobs":
    main_metric_col = {"Housing Units per Job": "housing_per_job", "People per Housing Unit": "people_per_housing", "Jobs per Capita": "jobs_per_capita", "Jobs per Working Age Adult": "jobs_per_working_age"}[view_mode]
elif metric_category == "Demographics":
    main_metric_col = {"Average Age": "avg_age", "% Residents Under 18": "pct_under18", "% Residents 18-22": "pct_18_22", "% Residents 23-34": "pct_23_34", "% Residents 35-49": "pct_35_49", "% Residents 50-64": "pct_50_64", "% Residents Over 65": "pct_over65", "% Working Age (18-64)": "pct_working_age"}[view_mode]
else:
    main_metric_col = "commuter_ratio" if "Ratio" in view_mode else ("in_commuter_share" if "In-Commuter" in view_mode else ("resident_retention" if "Retention" in view_mode else "net_commute"))

highlight_county = st.sidebar.selectbox("Highlight County", ["None"] + sorted(df.dropna(subset=[main_metric_col])["full_name"].unique()))

st.sidebar.markdown("---")
st.sidebar.title("Geography")
state_names = sorted(df.dropna(subset=[main_metric_col, "state_name"])["state_name"].unique())
selected_state_names = st.sidebar.multiselect("States", state_names, default=[s for s in ["Indiana"] if s in state_names] if "Indiana" in state_names else state_names)

st.sidebar.markdown("---")
st.sidebar.title("Time")
available_years = sorted(df.dropna(subset=[main_metric_col])["year"].unique())
selected_years = st.sidebar.multiselect("Years", available_years, default=[max(available_years)] if available_years else [])

selected_period = None
if metric_category == "Housing vs Jobs":
    available_periods = sorted(df[df["year"].isin(selected_years)].dropna(subset=["housing_per_job"])["period"].unique())
    selected_period = st.sidebar.selectbox("Period", available_periods, index=len(available_periods)-1 if available_periods else 0)

# -----------------------
# FILTER & AGGREGATE
# -----------------------
mask = (df["state_name"].isin(selected_state_names)) & (df["year"].isin(selected_years))
if selected_period: mask = mask & (df["period"] == selected_period)
filtered = df[mask].copy()
filtered["metric"] = filtered[main_metric_col]

if metric_category in ["Commuter Flows", "Demographics"] or (metric_category == "Housing vs Jobs" and view_mode != "Housing Units per Job"):
    group_cols = ["fips", "state", "state_name", "county_name", "full_name", "state_abbr"]
    cols_to_agg = ["metric", "in_commuters", "out_commuters", "net_commute", "lodes_total_jobs", "internal_workers", "total_residents_working", "B01001_001E", "jobs", "housing_units", "count_working_age"] + [c for c in filtered.columns if c.startswith("pct_") or c.startswith("count_")]
    filtered = filtered.groupby(group_cols, as_index=False).agg({c: "mean" for c in cols_to_agg if c in filtered.columns})
filtered = filtered.dropna(subset=["metric"])

# -----------------------
# SUMMARY METRICS
# -----------------------
if not filtered.empty:
    m1, m2 = st.columns(2)
    if metric_category == "Housing vs Jobs":
        if view_mode == "Housing Units per Job":
            m1.metric("Reg. Housing/Job Ratio", f"{filtered['housing_units'].sum() / filtered['jobs'].sum():.2f}")
        elif view_mode == "People per Housing Unit":
            m1.metric("Reg. People/Housing", f"{filtered['B01001_001E'].sum() / filtered['housing_units'].sum():.2f}")
        elif view_mode == "Jobs per Capita":
            m1.metric("Reg. Jobs per Capita", f"{filtered['jobs'].sum() / filtered['B01001_001E'].sum():.2f}")
        else: # Jobs per working age
            m1.metric("Reg. Jobs / Working Age", f"{filtered['jobs'].sum() / filtered['count_working_age'].sum():.2f}")
    elif metric_category == "Demographics":
        pop_total = filtered["B01001_001E"].sum()
        avg = (filtered["metric"] * filtered["B01001_001E"]).sum() / pop_total
        m1.metric(f"Regional {view_mode}", f"{avg:.1f}" if "Age" in view_mode else f"{avg:.1%}")
    else:
        if "In-Commuter" in view_mode: m1.metric("Reg. In-Commuter Share", f"{filtered['in_commuters'].sum() / filtered['lodes_total_jobs'].sum():.1%}")
        elif "Retention" in view_mode: m1.metric("Reg. Resident Retention", f"{filtered['internal_workers'].sum() / filtered['total_residents_working'].sum():.1%}")
        else:
            m1.metric("Avg In-Commute", f"{filtered['in_commuters'].mean():,.0f}")
            m2.metric("Avg Out-Commute", f"{filtered['out_commuters'].mean():,.0f}")

st.markdown("---")

# -----------------------
# MAP
# -----------------------
if not filtered.empty:
    st.subheader(f"County Map: {view_mode}")
    color_args = {}
    if "Ratio" in view_mode:
        max_dev = max(abs(filtered["metric"].max() - 1), abs(filtered["metric"].min() - 1), 0.1)
        color_args = {"range_color": [1 - max_dev, 1 + max_dev]}
    elif "Absolute" in view_mode or (metric_category == "Commuter Flows" and main_metric_col == "net_commute"):
        limit = max(abs(filtered["metric"].min()), abs(filtered["metric"].max()), 1)
        color_args = {"range_color": [-limit, limit]}
    
    hover_config = {"full_name": True, "state_name": True, "metric": ":.1%" if "%" in view_mode else ":.2f"}
    fig_map = px.choropleth(filtered, geojson=counties_geojson, locations="fips", color="metric", 
                            color_continuous_scale="Viridis" if metric_category == "Housing vs Jobs" else ("Magma" if metric_category == "Demographics" else "RdBu"), 
                            labels={"metric": view_mode}, hover_data=hover_config, **color_args)
    fig_map.update_traces(marker_line_color="rgba(255,255,255,0.35)", marker_line_width=0.4)
    
    if "state_abbr" in filtered.columns:
        selected_state_abbrs = filtered["state_abbr"].unique().tolist()
        state_features = [f for f in states_geojson["features"] if f["id"] in selected_state_abbrs]
        if state_features:
            fig_map.add_trace(go.Choropleth(
                geojson={"type": "FeatureCollection", "features": state_features},
                locations=[f["id"] for f in state_features],
                z=[1] * len(state_features),
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_color="white",
                marker_line_width=3,
                hoverinfo="skip"
            ))
    
    if highlight_county != "None":
        sel_row_fips = df.loc[df["full_name"] == highlight_county, "fips"].iloc[0]
        fig_map.add_trace(go.Choropleth(geojson=counties_geojson, locations=[sel_row_fips], z=[1], showscale=False, 
                                       colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="yellow", marker_line_width=3, hoverinfo="skip"))

    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), hovermode="closest")
    st.plotly_chart(fig_map, width="stretch")

    # -----------------------
    # DISTRIBUTION
    # -----------------------
    st.subheader("Statistical Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=filtered["metric"], histnorm="probability density", nbinsx=40, opacity=0.25, marker_color="gray"))
    vals = filtered["metric"].dropna().values
    if len(vals) > 5:
        try:
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 100)
            fig_dist.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode="lines", line=dict(width=3, color="#33683A")))
        except Exception: pass
    if highlight_county != "None":
        sel_row = filtered[filtered["full_name"] == highlight_county]
        if not sel_row.empty: fig_dist.add_vline(x=sel_row["metric"].mean(), line_width=3, line_dash="dash", line_color="red")
    fig_dist.update_layout(xaxis_title=view_mode, yaxis_title="Density", height=350, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
    st.plotly_chart(fig_dist, width="stretch")
else:
    st.warning("No data available.")
