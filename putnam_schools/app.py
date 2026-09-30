"""FRL vs. proficiency explorer for Indiana schools, with Putnam County highlighted.

Run:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from putnam_data import (
    DEFAULT_COUNTY,
    SUBJECT_GRADES,
    build_measure,
    load_grade_detail,
    load_schools,
)

# --- Theme ------------------------------------------------------------------
# Two selected modes, not an automatic flip. The four district hues were picked
# by running the palette validator over every 4-subset of the categorical ramp;
# blue/yellow/magenta/green is one of only two subsets that clears the
# all-pairs CVD and normal-vision floors in BOTH modes.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e1",
        "other": "#9d9c96",
        "other_opacity": 0.35,
        "fit": "#2f2e2b",
        "band": "rgba(47,46,43,0.10)",
        "series": ["#2a78d6", "#eda100", "#e87ba4", "#008300"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "other": "#6f6e69",
        "other_opacity": 0.5,
        "fit": "#d8d7cf",
        "band": "rgba(216,215,207,0.12)",
        "series": ["#3987e5", "#c98500", "#d55181", "#008300"],
    },
}

# Four hues is the cap: these are the only 4-subsets of the categorical ramp
# that clear the all-pairs CVD and normal-vision floors in both modes. A
# selection wider than this folds to one highlight colour rather than inventing
# hues, and leans on labels, hover and the table for identity.
MAX_SERIES = 4
SYMBOL_ORDER = ["circle", "square", "diamond", "triangle-up"]

# Labelling every point stops being legible somewhere around a dozen. Past that
# only the biggest misses from the line get named; the rest live in hover and
# the table.
LABEL_ALL_UPTO = 14
LABEL_TOP_N = 6

# Trailing corporation boilerplate, dropped so legend entries stay readable.
CORP_SUFFIX = re.compile(
    r"\s+(Com(munity)?\s+)?(Cons\s+)?(Sch(ool)?s?\s+)?"
    r"(Corp(oration)?|Corp|District|Schools?|Sch\s+Corp)$", re.I
)

# Two Cloverdale schools plot close together, so labels keep the level tag.
LEVEL_TAGS = [
    (r"\s*Elementary School$", " ES"),
    (r"\s*Primary School$", " PS"),
    (r"\s*Intermediate School$", " IS"),
    (r"\s*Middle School$", " MS"),
    (r"\s*(Sr )?High School$", " HS"),
]


def grade_phrase(grades: list[int] | None, available: list[int]) -> str:
    """Plain-English grade span for the title: 'All Grades', 'Grade 5', 'Grades 3-5'.

    Only a subject tested across the full 3-8 range earns 'All Grades'. Science
    and Social Studies name their grades instead, since 'All Grades Social
    Studies' would read as 3-8 when only grade 5 is tested.
    """
    picked = sorted(grades) if grades is not None else sorted(available)
    if picked == sorted(available) and picked == list(range(3, 9)):
        return "All Grades"
    if len(picked) == 1:
        return f"Grade {picked[0]}"
    if picked == list(range(picked[0], picked[-1] + 1)):
        return f"Grades {picked[0]}-{picked[-1]}"
    return "Grades " + ", ".join(str(g) for g in picked)


def short_label(name: str) -> str:
    for pattern, tag in LEVEL_TAGS:
        shortened, hits = re.subn(pattern, tag, name)
        if hits:
            return shortened
    return name

def short_district(name: str) -> str:
    """Trim corporation boilerplate: 'Greencastle Community School Corp' -> 'Greencastle'."""
    trimmed = name
    for _ in range(3):                      # suffixes stack, e.g. '... Com Sch Corp'
        nxt = CORP_SUFFIX.sub("", trimmed).strip()
        if nxt == trimmed or not nxt:
            break
        trimmed = nxt
    return trimmed or name

st.set_page_config(page_title="FRL vs. Proficiency", layout="wide")


@st.cache_data(show_spinner="Reading the workbook…")
def get_data():
    return load_schools(), load_grade_detail()


schools, grade_detail = get_data()

# --- Controls ---------------------------------------------------------------
sb = st.sidebar
sb.header("Explore")

subject = sb.selectbox("Test", list(SUBJECT_GRADES), index=0)
available_grades = SUBJECT_GRADES[subject]

grade_mode = sb.radio(
    "Grades",
    ["School total (as published)", "Pick grades"],
    help=(
        "Picking grades rebuilds proficiency as summed proficient ÷ summed tested "
        "across the grades you choose. Grades the state suppressed (fewer than 10 "
        "students) drop out, so this can differ from the published school total."
    ),
)
if grade_mode == "Pick grades":
    grades = sb.multiselect(
        "Grade levels", available_grades, default=available_grades,
        format_func=lambda g: f"Grade {g}",
    )
    if not grades:
        st.warning("Pick at least one grade.")
        st.stop()
else:
    grades = None
    if len(available_grades) < 6:
        sb.caption(
            f"{subject} is only tested in "
            + " and ".join(f"grade {g}" for g in available_grades)
            + "."
        )

sb.divider()
sb.subheader("Highlight")

highlight_by = sb.radio("Highlight by", ["County", "Districts", "Nothing"])

counties = sorted(schools["county"].dropna().unique())
if highlight_by == "County":
    county = sb.selectbox(
        "County", counties,
        index=counties.index(DEFAULT_COUNTY) if DEFAULT_COUNTY in counties else 0,
    )
    picked = sorted(schools.loc[schools["county"] == county, "district"].unique())
    group_name = f"{county} County"
elif highlight_by == "Districts":
    all_districts = sorted(schools["district"].unique())
    default = sorted(schools.loc[schools["county"] == DEFAULT_COUNTY, "district"].unique())
    picked = sb.multiselect(
        "Districts", all_districts,
        default=[d for d in default if d in all_districts],
        format_func=short_district,
    )
    group_name = "Selected districts"
else:
    picked, group_name = [], None

sb.divider()
sb.subheader("Filters")

min_tested = sb.slider(
    "Minimum students tested", 0, 150, 10, step=5,
    help="Small schools swing wildly. 10 matches the state's own suppression floor.",
)
frl_lo, frl_hi = sb.slider("FRL % range", 0, 100, (0, 100))
drop_virtual = sb.checkbox(
    "Exclude virtual / distance-learning schools", value=True,
    help=(
        "Their FRL rate describes a statewide online roster, not a neighborhood. "
        "Cloverdale Distance Learning Academy, for one, reports 0.1% FRL on "
        "2,080 students — leaving it in drags the fit and plants a misleading "
        "dot at x≈0."
    ),
)

sb.divider()
app_theme = st.context.theme.type or "light"
mode = "dark" if sb.toggle(
    "Dark mode", value=app_theme == "dark",
    help="Follows your Streamlit theme by default.",
) else "light"
show_labels = sb.checkbox("Label highlighted points", value=True)
T = THEMES[mode]

# --- Shape the data ---------------------------------------------------------
data = build_measure(schools, grade_detail, subject, grades)
data = data.dropna(subset=["frl_pct", "proficiency_pct"])
data = data[data["tested"] >= max(min_tested, 1)]
data = data[data["frl_pct"].between(frl_lo, frl_hi)]
if drop_virtual:
    data = data[~data["is_virtual"]]

data["is_highlight"] = data["district"].isin(picked)

if len(data) < 3:
    st.error("Fewer than 3 schools survive these filters — loosen them.")
    st.stop()

x = data["frl_pct"].to_numpy(float)
y = data["proficiency_pct"].to_numpy(float)
fit = stats.linregress(x, y)

# 95% confidence band for the fitted mean.
n = len(x)
grid = np.linspace(x.min(), x.max(), 200)
resid = y - (fit.intercept + fit.slope * x)
s_err = np.sqrt(np.sum(resid**2) / (n - 2))
sxx = np.sum((x - x.mean()) ** 2)
half = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(1 / n + (grid - x.mean()) ** 2 / sxx)
fit_y = fit.intercept + fit.slope * grid

data["predicted"] = fit.intercept + fit.slope * data["frl_pct"]
data["residual"] = data["proficiency_pct"] - data["predicted"]

chart_title = (
    "2026 Indiana Schools - % Free and Reduced vs. "
    f"{grade_phrase(grades, available_grades)} {subject}"
)

# --- Chart ------------------------------------------------------------------
HOVER = (
    "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
    "FRL %{x:.1f}%<br>Proficient %{y:.1f}%<br>"
    "%{customdata[2]:,.0f} tested · %{customdata[3]:,.0f} enrolled"
    "<extra></extra>"
)


def custom(df: pd.DataFrame) -> np.ndarray:
    return df[["school", "district", "tested", "enrollment"]].to_numpy()


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=np.concatenate([grid, grid[::-1]]),
        y=np.concatenate([fit_y + half, (fit_y - half)[::-1]]),
        fill="toself", fillcolor=T["band"], line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    )
)

others = data[~data["is_highlight"]]
fig.add_trace(
    go.Scatter(
        x=others["frl_pct"], y=others["proficiency_pct"], mode="markers",
        name=(f"Other Indiana schools (n={len(others):,})" if picked
              else f"Indiana schools (n={len(others):,})"),
        marker=dict(size=7, color=T["other"], opacity=T["other_opacity"],
                    line=dict(width=0)),
        customdata=custom(others), hovertemplate=HOVER,
    )
)

fig.add_trace(
    go.Scatter(
        x=grid, y=fit_y, mode="lines",
        name=f"Best fit (R²={fit.rvalue**2:.2f})",
        line=dict(color=T["fit"], width=2), hoverinfo="skip",
    )
)

def label_text(frame: pd.DataFrame) -> pd.Series:
    """Names for a highlighted group — all of them, or just the big misses."""
    names = frame["school"].map(short_label)
    if len(frame) <= LABEL_ALL_UPTO:
        return names
    keep = frame["residual"].abs().nlargest(LABEL_TOP_N).index
    return names.where(frame.index.isin(keep), "")


def add_highlight(frame: pd.DataFrame, name: str, color: str, symbol: str) -> None:
    # Big groups get smaller marks so the backdrop stays visible under them.
    size = 13 if len(frame) <= 30 else 9
    fig.add_trace(
        go.Scatter(
            x=frame["frl_pct"], y=frame["proficiency_pct"],
            mode="markers+text" if show_labels else "markers",
            name=f"{name} ({len(frame)})",
            text=label_text(frame),
            textposition=np.where(frame["residual"] >= 0, "top center",
                                  "bottom center"),
            textfont=dict(size=10, color=T["muted"]),
            marker=dict(size=size, symbol=symbol, color=color,
                        line=dict(width=2, color=T["surface"])),
            customdata=custom(frame), hovertemplate=HOVER,
        )
    )


# Districts that survived the filters, in a fixed name order so a filter change
# can never repaint the survivors.
shown = [d for d in picked if (data["district"] == d).any()]

if not shown:
    pass
elif len(shown) <= MAX_SERIES:
    for i, district in enumerate(shown):
        add_highlight(data[data["district"] == district], short_district(district),
                      T["series"][i], SYMBOL_ORDER[i])
else:
    # Past four, hues stop being separable — one colour, identity from the table.
    add_highlight(data[data["is_highlight"]], group_name or "Highlighted",
                  T["series"][0], SYMBOL_ORDER[0])

fig.update_layout(
    height=620,
    paper_bgcolor=T["surface"], plot_bgcolor=T["surface"],
    font=dict(color=T["text"], size=13),
    title=dict(
        text=chart_title,
        font=dict(size=18, color=T["text"]),
    ),
    margin=dict(l=60, r=30, t=70, b=60),
    legend=dict(orientation="h", y=-0.16, x=0, font=dict(color=T["muted"])),
    hoverlabel=dict(bgcolor=T["surface"], font=dict(color=T["text"])),
)
axis = dict(gridcolor=T["grid"], zeroline=False, linecolor=T["grid"],
            tickfont=dict(color=T["muted"]),
            title_font=dict(color=T["muted"], size=12), ticksuffix="%")
fig.update_xaxes(title_text="Students on free & reduced lunch", **axis)
fig.update_yaxes(title_text=f"{subject} proficient", **axis)

st.plotly_chart(fig, width="stretch", theme=None)

# --- Readout ----------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Schools plotted", f"{n:,}")
c2.metric("Slope", f"{fit.slope:+.2f} pp",
          help="Change in proficiency for each 1-point rise in FRL.")
c3.metric("R²", f"{fit.rvalue**2:.2f}",
          help=f"Pearson r = {fit.rvalue:.2f}")
c4.metric("p-value", "<0.001" if fit.pvalue < 0.001 else f"{fit.pvalue:.3f}")

n_high = int(data["is_highlight"].sum())
if show_labels and n_high > LABEL_ALL_UPTO:
    st.caption(
        f"{group_name} has {n_high} schools on screen, too many to label. Only "
        f"the {LABEL_TOP_N} furthest from the line are named — hover any point, "
        "or open the table below."
    )

st.caption(
    "The line is fitted to every school shown, the highlighted ones included. "
    "It describes how strongly FRL tracks proficiency across Indiana — not what "
    "any one school caused."
)

# --- Tables -----------------------------------------------------------------
cols = ["school", "district", "frl_pct", "proficiency_pct", "predicted",
        "residual", "tested", "enrollment"]
names = {
    "school": "School", "district": "District", "frl_pct": "FRL %",
    "proficiency_pct": "Proficient %", "predicted": "Predicted %",
    "residual": "vs. prediction", "tested": "Tested", "enrollment": "Enrolled",
}
fmt = {"FRL %": "{:.1f}", "Proficient %": "{:.1f}", "Predicted %": "{:.1f}",
       "vs. prediction": "{:+.1f}", "Tested": "{:,.0f}", "Enrolled": "{:,.0f}"}

group_tab, all_tab = st.tabs([group_name or "Highlighted", "All schools shown"])

with group_tab:
    tbl = (data[data["is_highlight"]][cols]
           .sort_values("residual", ascending=False)
           .rename(columns=names))
    if not picked:
        st.info("Pick a county or some districts in the sidebar to highlight them.")
    elif tbl.empty:
        st.info(f"No school in {group_name} clears the current filters.")
    else:
        st.dataframe(tbl.style.format(fmt), width="stretch",
                     hide_index=True)
        st.caption(
            "**vs. prediction** is how far above or below the statewide line a "
            "school sits, in percentage points — the comparison that holds "
            "poverty roughly constant."
        )

with all_tab:
    st.dataframe(
        data[cols].sort_values("residual", ascending=False)
        .rename(columns=names).style.format(fmt),
        width="stretch", hide_index=True, height=420,
    )
    st.download_button(
        "Download this view (CSV)",
        data[cols].rename(columns=names).to_csv(index=False).encode(),
        file_name=(
            "frl_vs_"
            f"{subject.replace(' & ', '_').replace(' ', '_').lower()}"
            f"{'_' + group_name.replace(' ', '_').lower() if group_name else ''}.csv"
        ),
        mime="text/csv",
    )

st.caption(
    "Source: Indiana-2026-School-Data-Combined.xlsx — ILEARN Spring 2026 "
    "(grades 3–8) joined to 2025–26 enrollment and meal status. County comes "
    "from the NCES district directory (`corp_county.csv`) and is the "
    "corporation's administrative county, so a corporation straddling a county "
    "line is filed under one of them."
)
