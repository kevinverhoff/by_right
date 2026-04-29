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

st.sidebar.markdown("---")
st.sidebar.markdown(f'<a href="https://www.buymeacoffee.com/wZyLoMV" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: #ffdd00; color: black; text-align: center; border-radius: 5px; text-decoration: none; font-weight: bold; width: 100%;">☕ Buy me a coffee (I\'ll use it to keep this site going)</a>', unsafe_allow_html=True)

# -----------------------
# FILTER
# -----------------------
filtered = df[(df["state_name"].isin(selected_states)) & (df["year"].isin(selected_years))].copy()
filtered["metric"] = filtered[main_metric_col]
if metric_category in ["Commuter Flows", "Demographics"]:
    filtered = filtered.groupby(["fips", "state", "state_name", "county_name", "full_name", "state_abbr"], as_index=False).mean(numeric_only=True)
filtered = filtered.dropna(subset=["metric"])

# Add ranking
if not filtered.empty:
    filtered["rank"] = filtered["metric"].rank(ascending=False, method="min").astype(int)
    n_counties = len(filtered)
    def get_ordinal(n):
        if 11 <= (n % 100) <= 13: return f"{n}th"
        return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th') }"
    filtered["ranking_info"] = filtered["rank"].apply(lambda x: f"{get_ordinal(x)} highest (out of {n_counties} counties) in {view_mode}")

# -----------------------
# SUMMARY
# -----------------------
if not filtered.empty:
    m1, m2 = st.columns(2)
    if metric_category == "Housing vs Jobs":
        if view_mode == "Housing Units per Job":
            val = filtered['housing_units'].sum() / filtered['jobs'].sum()
            m1.metric("Avg. Housing Units per Job", f"{val:.2f}")
        elif view_mode == "People per Housing Unit":
            val = filtered['B01001_001E'].sum() / filtered['housing_units'].sum()
            m1.metric("Reg. People/Housing", f"{val:.2f}")
        elif view_mode == "Jobs per Capita":
            val = filtered['jobs'].sum() / filtered['B01001_001E'].sum()
            m1.metric("Reg. Jobs per Capita", f"{val:.2f}")
        else: # Jobs per Working Age Adult
            val = filtered['jobs'].sum() / filtered['count_working_age'].sum()
            m1.metric("Reg. Jobs/Work-Age", f"{val:.2f}")
    elif metric_category == "Demographics":
        pop_total = filtered["B01001_001E"].sum()
        avg = (filtered["metric"] * filtered["B01001_001E"]).sum() / pop_total
        m1.metric(f"Regional {view_mode}", f"{avg:.1f}" if "Age" in view_mode else f"{avg:.1%}")
    else:
        if "Ratio" in view_mode: m1.metric("Reg. Commuter Ratio", f"{filtered['in_commuters'].sum() / filtered['out_commuters'].sum():.2f}")
        elif "Share" in view_mode: m1.metric("Reg. Commuter Intensity", f"{filtered['in_commuters'].sum() / filtered['lodes_total_jobs'].sum():.1%}")
        else: m1.metric("Avg In-Commute", f"{filtered['in_commuters'].mean():,.0f}")
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
    elif "Absolute" in view_mode:
        limit = max(abs(filtered["metric"].min()), abs(filtered["metric"].max()), 1)
        color_args = {"range_color": [-limit, limit]}
    
    # Tooltip setup
    if metric_category == "Housing vs Jobs":
        hover_data = {
            "full_name": True,
            "housing_per_job": ":.3f", "people_per_housing": ":.2f", 
            "jobs_per_capita": ":.2f", "jobs_per_working_age": ":.2f",
            "ranking_info": True
        }
        hover_labels = {
            "full_name": "County", "housing_per_job": "Housing Units per Job",
            "people_per_housing": "People per Housing Unit", "jobs_per_capita": "Jobs per Capita",
            "jobs_per_working_age": "Jobs per Working Age Adult",
            "ranking_info": "Rank"
        }
    elif metric_category == "Demographics":
        hover_data = {
            "full_name": True, "B01001_001E": ":,.0f", "avg_age": ":.1f", 
            "pct_under18": ":.1%", "pct_over65": ":.1%", "pct_working_age": ":.1%",
            "ranking_info": True
        }
        hover_labels = {
            "full_name": "County", "B01001_001E": "Total Population", "avg_age": "Average Age",
            "pct_under18": "% Under 18", "pct_over65": "% Over 65", "pct_working_age": "% Working Age",
            "ranking_info": "Rank"
        }
    else: # Commuter Flows
        hover_data = {
            "full_name": True, "net_commute": ":,.0f", 
            "in_commuters": ":,.0f", "out_commuters": ":,.0f", "lodes_total_jobs": ":,.0f",
            "ranking_info": True
        }
        hover_labels = {
            "full_name": "County", "net_commute": "Net Commuters",
            "in_commuters": "In-Commuters", "out_commuters": "Out-Commuters", "lodes_total_jobs": "Total Jobs",
            "ranking_info": "Rank"
        }

    fig = px.choropleth(filtered, geojson=geo, locations="fips", color="metric", 
                        labels={**hover_labels, "metric": view_mode}, 
                        color_continuous_scale="Viridis" if metric_category=="Housing vs Jobs" else ("Magma" if metric_category=="Demographics" else "RdBu"), 
                        hover_data=hover_data)
    fig.update_traces(marker_line_color="rgba(255,255,255,0.35)", marker_line_width=0.4)
    state_abbrs = filtered["state_abbr"].unique()
    state_features = [f for f in states_geo["features"] if f["id"] in state_abbrs]
    fig.add_trace(go.Choropleth(geojson={"type": "FeatureCollection", "features": state_features}, locations=[f["id"] for f in state_features], z=[1]*len(state_features), showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="white", marker_line_width=3, hoverinfo="skip"))
    
    if highlight_county != "None":
        h_fips = filtered.loc[filtered["full_name"] == highlight_county, "fips"].iloc[0]
        fig.add_trace(go.Choropleth(geojson=geo, locations=[h_fips], z=[1], showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="yellow", marker_line_width=4, hoverinfo="skip"))
        
        filtered["diff"] = (filtered["metric"] - filtered.loc[filtered["full_name"] == highlight_county, "metric"].iloc[0]).abs()
        peers = filtered.sort_values("diff").head(5)["fips"].tolist()[1:]
        fig.add_trace(go.Choropleth(geojson=geo, locations=peers, z=[1]*len(peers), showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="cyan", marker_line_width=3, hoverinfo="skip"))
        
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, width="stretch")
    
    # -----------------------
    # RANKINGS / LEADERBOARD
    # -----------------------
    st.subheader(f"Rankings: {view_mode}")
    
    if highlight_county != "None":
        h_row = filtered[filtered["full_name"] == highlight_county]
        if not h_row.empty:
            rank_val = h_row["rank"].iloc[0]
            metric_val = h_row["metric"].iloc[0]
            val_str = f"{metric_val:.3f}" if metric_val < 10 else f"{metric_val:,.1f}"
            st.info(f"**{highlight_county}** is ranked **{get_ordinal(rank_val)}** highest with a value of **{val_str}**")

    r1, r2 = st.columns(2)
    top_10 = filtered.sort_values("metric", ascending=False).head(10)[["full_name", "metric", "rank"]]
    bot_10 = filtered.sort_values("metric", ascending=True).head(10)[["full_name", "metric", "rank"]]
    
    with r1:
        st.write("**Highest**")
        st.dataframe(top_10, column_config={"full_name": "County", "metric": view_mode, "rank": "Rank"}, hide_index=True)
    with r2:
        st.write("**Lowest**")
        st.dataframe(bot_10, column_config={"full_name": "County", "metric": view_mode, "rank": "Rank"}, hide_index=True)
    
    # -----------------------
    # DISTRIBUTION
    # -----------------------
    st.subheader("Statistical Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=filtered["metric"], histnorm="probability density", marker_color="gray", opacity=0.3))
    try:
        kde = gaussian_kde(filtered["metric"].dropna())
        x = np.linspace(filtered["metric"].min(), filtered["metric"].max(), 100)
        fig_dist.add_trace(go.Scatter(x=x, y=kde(x), line=dict(width=3, color="#33683A")))
    except: pass
    if highlight_county != "None":
        fig_dist.add_vline(x=filtered.loc[filtered["full_name"] == highlight_county, "metric"].mean(), line_color="yellow", line_width=3, line_dash="dash")
    fig_dist.update_layout(xaxis_title=view_mode, yaxis_title="Density", height=300, template="plotly_white")
    st.plotly_chart(fig_dist, width="stretch")

    # -----------------------
    # COMPARISON TABLE
    # -----------------------
    if highlight_county != "None":
        st.markdown("---")
        st.subheader("Highlighted County Profile & Comps")
        h_fips = filtered.loc[filtered["full_name"] == highlight_county, "fips"].iloc[0]
        h_val = filtered.loc[filtered["full_name"] == highlight_county, "metric"].iloc[0]
        filtered["diff"] = (filtered["metric"] - h_val).abs()
        peers = filtered.sort_values("diff").head(5)["fips"].tolist()
        comp = df[df["fips"].isin(peers)].sort_values("year", ascending=False).groupby("fips").first().loc[peers].copy()
        
        display = comp[["full_name", "housing_per_job", "people_per_housing", "jobs_per_working_age", "commuter_ratio", "in_commuter_share", "resident_retention", "avg_age", "pct_under18", "pct_18_22", "pct_23_34", "pct_over65"]].T
        display.columns = display.iloc[0]
        fmt = {
            "housing_per_job":      lambda x: f"{x:.2f}",
            "people_per_housing":   lambda x: f"{x:.2f}",
            "jobs_per_working_age": lambda x: f"{x:.2f}",
            "commuter_ratio":       lambda x: f"{x:.2f}",
            "in_commuter_share":    lambda x: f"{x*100:.1f}%",
            "resident_retention":   lambda x: f"{x*100:.1f}%",
            "avg_age":              lambda x: f"{x:.1f}",
            "pct_under18":          lambda x: f"{x*100:.1f}%",
            "pct_18_22":            lambda x: f"{x*100:.1f}%",
            "pct_23_34":            lambda x: f"{x*100:.1f}%",
            "pct_over65":           lambda x: f"{x*100:.1f}%",
        }

        for row, func in fmt.items():
            display.loc[row] = display.loc[row].apply(func)
        display = display.drop("full_name")
        display.index = ["Housing/Job", "People/House", "Jobs/Work-Age", "In/Out Ratio", "In-Commuter %", "Retention %", "Avg Age", "% < 18", "% 18-22", "% 23-34", "% > 65"]
        st.table(display)
