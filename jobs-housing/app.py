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
    page_icon="https://raw.githubusercontent.com/kevinverhoff/by_right/main/jobs-housing/ByRIGHT-small.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""<style>.block-container { padding-top: 2rem; } div[data-testid="stMetric"] { background-color: #CED0CE; padding: 15px; border-radius: 10px; border: 1px solid #33683A; } h1, h2, h3 { color: #33683A !important; }</style>""", unsafe_allow_html=True)

# Header
logo_url = "https://raw.githubusercontent.com/kevinverhoff/by_right/main/jobs-housing/ByRIGHT-small.png"
st.markdown(f'<div style="display: flex; align-items: center; gap: 15px;"><img src="{logo_url}" style="height: 3rem; width: auto;"><h1 style="margin: 0;">By Right County Dashboard</h1></div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# DATA LOAD
# -----------------------
@st.cache_data(ttl=3600)
def load_data():
    JH_URL = "https://github.com/kevinverhoff/by_right/raw/main/jobs-housing/county_jobs_housing.parquet"
    LODES_URL = "https://github.com/kevinverhoff/by_right/raw/main/jobs-housing/lodes_commuting.parquet"
    df = pd.merge(pd.read_parquet(JH_URL), pd.read_parquet(LODES_URL).rename(columns={"county_name": "county_name_lodes", "state_name": "state_name_lodes", "full_name": "full_name_lodes", "state": "state_fips_lodes"}), on=["fips", "year"], how="outer")
    df["state_abbr"] = df["fips"].str[:2].map({"18": "IN", "17": "IL", "21": "KY", "26": "MI", "39": "OH"})
    for c in ["county_name", "state_name", "full_name", "state"]: df[c] = df[c].fillna(df[c+"_lodes" if c != "state" else "state_fips_lodes"])
    df["commuter_ratio"] = df["in_commuters"] / df["out_commuters"].replace(0, np.nan)
    df["in_commuter_share"] = df["in_commuters"] / df["lodes_total_jobs"].replace(0, np.nan)
    df["resident_retention"] = df["internal_workers"] / (df["internal_workers"] + df["out_commuters"]).replace(0, np.nan)
    df["people_per_housing"] = df["B01001_001E"] / df["housing_units"].replace(0, np.nan)
    df["jobs_per_capita"] = df["jobs"] / df["B01001_001E"].replace(0, np.nan)
    df["jobs_per_working_age"] = df["jobs"] / df["count_working_age"].replace(0, np.nan)
    return df

df = load_data()
geo = requests.get("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json").json()
states_geo = requests.get("https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json").json()

# -----------------------
# SIDEBAR
# -----------------------
metric_category = st.sidebar.radio("Category", ["Housing vs Jobs", "Commuter Flows", "Demographics"])
view_mode = st.sidebar.selectbox("View Mode", {
    "Housing vs Jobs": ["Housing Units per Job", "People per Housing Unit", "Jobs per Capita", "Jobs per Working Age Adult"],
    "Commuter Flows": ["Absolute (Net Flow)", "Commuter Ratio (In/Out)", "In-Commuter Job Share (% of Jobs)", "Resident Retention Share (% of Residents)"],
    "Demographics": ["Average Age", "% Residents Under 18", "% Residents 18-22", "% Residents 23-34", "% Residents 35-49", "% Residents 50-64", "% Residents Over 65", "% Working Age (18-64)"]
}[metric_category])

main_metric_col = {
    "Housing Units per Job": "housing_per_job", "People per Housing Unit": "people_per_housing", "Jobs per Capita": "jobs_per_capita", "Jobs per Working Age Adult": "jobs_per_working_age",
    "Average Age": "avg_age", "% Residents Under 18": "pct_under18", "% Residents 18-22": "pct_18_22", "% Residents 23-34": "pct_23_34", "% Residents 35-49": "pct_35_49", "% Residents 50-64": "pct_50_64", "% Residents Over 65": "pct_over65", "% Working Age (18-64)": "pct_working_age",
    "Absolute (Net Flow)": "net_commute", "Commuter Ratio (In/Out)": "commuter_ratio", "In-Commuter Job Share (% of Jobs)": "in_commuter_share", "Resident Retention Share (% of Residents)": "resident_retention"
}[view_mode]

all_counties = sorted(df.dropna(subset=[main_metric_col])["full_name"].unique())
highlight_county = st.sidebar.selectbox("Highlight County", ["None"] + all_counties)
states = sorted(df.dropna(subset=[main_metric_col, "state_name"])["state_name"].unique())
selected_states = st.sidebar.multiselect("States", states, default=[s for s in ["Indiana"] if s in states] or states)
years = sorted(df.dropna(subset=[main_metric_col])["year"].unique())
selected_years = st.sidebar.multiselect("Years", years, default=[max(years)] if years else [])

# -----------------------
# FILTER
# -----------------------
filtered = df[(df["state_name"].isin(selected_states)) & (df["year"].isin(selected_years))].copy()
filtered["metric"] = filtered[main_metric_col]
if metric_category in ["Commuter Flows", "Demographics"]:
    filtered = filtered.groupby(["fips", "state", "state_name", "county_name", "full_name", "state_abbr"], as_index=False).mean(numeric_only=True)
filtered = filtered.dropna(subset=["metric"])

# -----------------------
# MAP
# -----------------------
if not filtered.empty:
    st.subheader(f"County Map: {view_mode}")
    color_args = {}
    if "Ratio" in view_mode:
        max_dev = max(abs(filtered["metric"].max() - 1), abs(filtered["metric"].min() - 1), 0.1)
        color_args = {"range_color": [1 - max_dev, 1 + max_dev]}
    elif "Absolute" in view_mode:
        limit = max(abs(filtered["metric"].min()), abs(filtered["metric"].max()), 1)
        color_args = {"range_color": [-limit, limit]}
    
    # Tooltip setup
    if metric_category == "Housing vs Jobs":
        hover_data = {
            "full_name": True,
            "housing_per_job": ":.3f", "people_per_housing": ":.2f", 
            "jobs_per_capita": ":.2f", "jobs_per_working_age": ":.2f"
        }
        hover_labels = {
            "full_name": "County", "housing_per_job": "Housing Units per Job",
            "people_per_housing": "People per Housing Unit", "jobs_per_capita": "Jobs per Capita",
            "jobs_per_working_age": "Jobs per Working Age Adult"
        }
    elif metric_category == "Demographics":
        hover_data = {
            "full_name": True, "B01001_001E": ":,.0f", "avg_age": ":.1f", 
            "pct_under18": ":.1%", "pct_over65": ":.1%", "pct_working_age": ":.1%"
        }
        hover_labels = {
            "full_name": "County", "B01001_001E": "Total Population", "avg_age": "Average Age",
            "pct_under18": "% Under 18", "pct_over65": "% Over 65", "pct_working_age": "% Working Age"
        }
    else: # Commuter Flows
        hover_data = {
            "full_name": True, "net_commute": ":,.0f", 
            "in_commuters": ":,.0f", "out_commuters": ":,.0f", "lodes_total_jobs": ":,.0f"
        }
        hover_labels = {
            "full_name": "County", "net_commute": "Net Commuters",
            "in_commuters": "In-Commuters", "out_commuters": "Out-Commuters", "lodes_total_jobs": "Total Jobs"
        }

    # Merge main labels with hover labels
    all_labels = {**hover_labels, "metric": view_mode}

    fig = px.choropleth(filtered, geojson=geo, locations="fips", color="metric", labels=all_labels, 
                        color_continuous_scale="Viridis" if metric_category=="Housing vs Jobs" else ("Magma" if metric_category=="Demographics" else "RdBu"), 
                        hover_data=hover_data, **color_args)


    fig.update_traces(marker_line_color="rgba(255,255,255,0.35)", marker_line_width=0.4)
    
    # State borders
    state_abbrs = filtered["state_abbr"].unique()
    state_features = [f for f in states_geo["features"] if f["id"] in state_abbrs]
    fig.add_trace(go.Choropleth(geojson={"type": "FeatureCollection", "features": state_features}, locations=[f["id"] for f in state_features], z=[1]*len(state_features), showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="white", marker_line_width=3, hoverinfo="skip"))
    
    # Highlights
    if highlight_county != "None":
        h_fips = filtered.loc[filtered["full_name"] == highlight_county, "fips"].iloc[0]
        h_val = filtered.loc[filtered["full_name"] == highlight_county, "metric"].iloc[0]
        filtered["diff"] = (filtered["metric"] - h_val).abs()
        peers = filtered.sort_values("diff").head(5)["fips"].tolist()[1:]
        
        fig.add_trace(go.Choropleth(geojson=geo, locations=[h_fips], z=[1], showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="yellow", marker_line_width=4, hoverinfo="skip"))
        fig.add_trace(go.Choropleth(geojson=geo, locations=peers, z=[1]*len(peers), showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="cyan", marker_line_width=3, hoverinfo="skip"))
        
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, width="stretch")
    
    # Legend
    st.markdown("""
        <div style="display: flex; gap: 20px; font-size: 0.9em; margin-bottom: 10px;">
            <div style="border: 2px solid yellow; padding: 2px 10px; border-radius: 4px;">Yellow = Highlight County</div>
            <div style="border: 2px solid cyan; padding: 2px 10px; border-radius: 4px;">Cyan = Similar Counties</div>
        </div>
    """, unsafe_allow_html=True)

    # Distribution
    st.subheader("Statistical Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=filtered["metric"], histnorm="probability density", marker_color="gray", opacity=0.3))
    try:
        kde = gaussian_kde(filtered["metric"].dropna())
        x = np.linspace(filtered["metric"].min(), filtered["metric"].max(), 100)
        fig_dist.add_trace(go.Scatter(x=x, y=kde(x), line=dict(width=3, color="#33683A")))
    except: pass
    if highlight_county != "None":
        h_val = filtered.loc[filtered["full_name"] == highlight_county, "metric"].mean()
        fig_dist.add_vline(x=h_val, line_color="yellow", line_width=3, line_dash="dash")
    fig_dist.update_layout(xaxis_title=view_mode, yaxis_title="Density", height=300, template="plotly_white")
    st.plotly_chart(fig_dist, width="stretch")
