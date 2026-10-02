"""Indiana township trustees, 2025 — an explorer for the SBOA / Gateway filings.

Run:  .venv/bin/streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import metrics as mx
import theme as th

st.set_page_config(page_title="Indiana Township Trustees 2025", layout="wide")


@st.cache_data
def load_data():
    return mx.load()


@st.cache_data
def load_geo():
    with open(mx.GEOJSON) as f:
        return json.load(f)


df = load_data()
df["trustee_label"] = df["trustee_name"].fillna("No trustee named")

# ---------------------------------------------------------------- filters
# One filter set, in the sidebar, scoping every tab.
# Follow whatever theme Streamlit is in, rather than carrying a second toggle
# that can disagree with the page chrome around the charts.
t = th.tokens(getattr(st.context.theme, "type", "light") == "dark")

with st.sidebar:
    st.markdown("### Filters")
    counties = sorted(df["county"].unique())
    picked = st.multiselect("County", counties, default=[],
                            help="Empty means all 92 counties.")

    pop_hi = int(np.nanmax(df["population_2025_est"]))
    pop_lo, pop_hi_sel = st.select_slider(
        "Population (2025 est.)",
        options=[0, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, pop_hi],
        value=(0, pop_hi),
        format_func=lambda v: f"{v:,}",
    )
    only_complete = st.checkbox("Only townships with a 2025 AFR and a named trustee",
                                value=False, key="only_complete")

    # Deliberately not a filter: highlighting keeps the rest of the state on
    # screen as context, which is the whole point of it.
    st.markdown("### Highlight")
    hl_county = st.selectbox(
        "County", ["None"] + counties, index=0, key="hl_county",
        help="Picks one county out of the full field without removing the rest. "
             "To drop the others instead, use the County filter above.",
    )
    highlight = None if hl_county == "None" else hl_county

d = df.copy()
if picked:
    d = d[d["county"].isin(picked)]
if (pop_lo, pop_hi_sel) != (0, pop_hi):
    d = d[d["population_2025_est"].between(pop_lo, pop_hi_sel)]
if only_complete:
    d = d[d["trustee_name"].notna() & (d["afr_year"] == 2025)]

st.markdown(th.css(t), unsafe_allow_html=True)
PLOTLY_CFG = {"displayModeBar": False, "scrollZoom": False}

hl_n = int((d["county"] == highlight).sum()) if highlight else 0
if highlight and hl_n == 0:
    st.warning(
        f"{highlight} County is highlighted but the filters have excluded all of its "
        "townships, so nothing will stand out. Widen the filters or clear the highlight."
    )


def fit_trend(xs, ys, log_x, log_y):
    """Least-squares fit in the space the axes are drawn in.

    With both axes logged this is a power law, which plots as a straight line
    and whose slope reads as an elasticity: a slope of -0.3 means a township
    ten times larger spends about half as much per resident.
    """
    # A zero or negative slips through as -inf/nan rather than raising; the
    # finite mask below drops it.
    with np.errstate(divide="ignore", invalid="ignore"):
        fx = np.log10(xs) if log_x else np.asarray(xs, dtype=float)
        fy = np.log10(ys) if log_y else np.asarray(ys, dtype=float)
    ok = np.isfinite(fx) & np.isfinite(fy)
    if ok.sum() < 3:
        return None
    fx, fy = fx[ok], fy[ok]
    slope, intercept = np.polyfit(fx, fy, 1)
    resid = fy - (slope * fx + intercept)
    ss_tot = float(((fy - fy.mean()) ** 2).sum())
    gx = np.linspace(fx.min(), fx.max(), 120)
    gy = slope * gx + intercept
    return dict(
        line_x=10 ** gx if log_x else gx,
        line_y=10 ** gy if log_y else gy,
        slope=float(slope),
        r2=(1 - float((resid ** 2).sum()) / ss_tot) if ss_tot > 0 else float("nan"),
        n=int(ok.sum()),
        residual=lambda xv, yv: (
            (np.log10(yv) if log_y else np.asarray(yv, dtype=float))
            - (slope * (np.log10(xv) if log_x else np.asarray(xv, dtype=float)) + intercept)
        ),
    )


def stat_tiles(frame):
    """A headline row — these are single numbers, so they are tiles, not charts."""
    med_comp = frame["trustee_2025_compensation"].median()
    cash = frame["ending_cash_and_investments"].sum()
    med_res = frame["reserve_years"].median()
    tiles = [
        ("Townships", f"{len(frame):,}", f"of {len(df):,} statewide"),
        ("Median trustee pay", mx.fmt(med_comp, "usd0"),
         f"{mx.fmt(frame['trustee_2025_compensation'].max(), 'usd0')} highest"),
        ("Cash and investments held", mx.fmt(cash, "usd0"), "end of filing year"),
        ("Median reserve", mx.fmt(med_res, "ratio"), "years of spending held"),
        ("Median assistance share", mx.fmt(frame["assistance_share"].median(), "pct"),
         "of total disbursements"),
    ]
    html = "".join(
        f'<div class="stat-tile"><span class="label">{lab}</span>'
        f'<div class="value">{val}</div><span class="sub">{sub}</span></div>'
        for lab, val, sub in tiles
    )
    st.markdown(f'<div class="stat-row">{html}</div>', unsafe_allow_html=True)


st.markdown("## Indiana township trustees, 2025")
stale = int((df["afr_year"] != 2025).sum())
unnamed = int(df["trustee_name"].isna().sum())
st.markdown(
    f'<p class="metric-note">{len(df):,} townships, as reported to the State Board of Accounts '
    f"(Form 100R compensation) and Gateway (annual financial reports). "
    f"{stale} have no 2025 AFR, so their financials are 2024 or missing; "
    f"{unnamed} name no trustee. Both are kept in view rather than dropped — "
    f'use the sidebar to exclude them.</p>',
    unsafe_allow_html=True,
)
stat_tiles(d)

tab_bubble, tab_map, tab_rank, tab_table = st.tabs(
    ["Bubble", "Map", "Rankings", "Table"]
)

# ---------------------------------------------------------------- bubble
with tab_bubble:
    y_opts = [m.label for m in mx.CATALOG if m.col != "population_2025_est"]
    size_opts = ["None (uniform)"] + [m.label for m in mx.CATALOG if m.positive]

    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.8])
    y_label = c1.selectbox("Y axis", y_opts, key="y_metric",
                           index=y_opts.index("Disbursements per resident"))
    size_label = c2.selectbox("Bubble size", size_opts, key="size_metric",
                              index=size_opts.index("Total disbursements"))
    color_label = c3.selectbox(
        "Color", ["Population tier", "None"] + [m.label for m in mx.CATALOG],
        key="color_metric", index=0,
    )
    ym = mx.BY_COL[mx.LABEL_TO_COL[y_label]]
    xm = mx.BY_COL["population_2025_est"]

    with c4:
        st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        log_x = st.checkbox("Log x", value=True, key="log_x")
        # A measure that goes negative has no log axis; say so rather than
        # quietly dropping every township below zero.
        log_y = st.checkbox(
            "Log y", value=True, key="log_y", disabled=not ym.log_ok,
            help=None if ym.log_ok else f"{ym.label} goes negative, so it has no log axis.",
        ) and ym.log_ok
        show_fit = st.checkbox("Trend", value=True, key="show_fit",
                               help="Least-squares fit across the townships plotted.")

    b = d[d["population_2025_est"].notna() & d[ym.col].notna()].copy()
    if log_y:
        b = b[b[ym.col] > 0]

    custom = ["trustee_label", "population_2025_est", ym.col]

    size_col = None
    if size_label != "None (uniform)":
        size_col = mx.LABEL_TO_COL[size_label]
        b = b[b[size_col].notna() & (b[size_col] >= 0)]
        if size_col not in custom:
            custom.append(size_col)

    color_col = None
    discrete = False
    if color_label == "Population tier":
        color_col, discrete = "pop_tier", True
    elif color_label != "None":
        color_col = mx.LABEL_TO_COL[color_label]
        b = b[b[color_col].notna()]
        if color_col not in custom:
            custom.append(color_col)

    if b.empty:
        st.info("No townships match these filters.")
    else:
        args = dict(
            x="population_2025_est",
            y=ym.col,
            hover_name="name",
            custom_data=custom + ["county"],
            log_x=log_x,
            log_y=log_y,
        )
        if size_col:
            args.update(size=size_col, size_max=40)
        if discrete:
            args.update(color=color_col,
                        category_orders={"pop_tier": list(d["pop_tier"].cat.categories)},
                        color_discrete_sequence=t["series"])
        elif color_col:
            args.update(color=color_col, color_continuous_scale=t["sequential"])

        fig = px.scatter(b, **args)

        # Hover: name, then the encoded measures in the order the controls list them.
        lines = [f"<b>%{{hovertext}}</b>", "%{customdata[0]}", "<br>"]
        for i, col in enumerate(custom[1:], start=1):
            m = mx.BY_COL[col]
            lines.append(f"{m.label}: <b>%{{customdata[{i}]:{mx.tickformat(m.fmt)}}}</b>")
        tmpl = "<br>".join(lines) + "<extra></extra>"

        ring = 2 if len(b) < 400 else 1
        fig.update_traces(
            marker=dict(
                opacity=0.78,
                line=dict(width=ring, color=t["surface"]),
                sizemin=5,
            ),
            hovertemplate=tmpl,
            selector=dict(type="scatter"),
        )
        if not size_col:
            fig.update_traces(marker_size=9)

        if highlight:
            # Per-point opacity and outline rather than a second colour: the
            # highlight has to sit on top of whatever the Color control is
            # already encoding, so it gets its own channels.
            for tr in fig.data:
                if tr.customdata is None:
                    continue
                is_hl = [str(row[-1]) == highlight for row in tr.customdata]
                tr.marker.opacity = [0.95 if h else 0.12 for h in is_hl]
                tr.marker.line.width = [2 if h else 0 for h in is_hl]
                tr.marker.line.color = [t["outline"] if h else t["surface"] for h in is_hl]
            # A legend entry for the outline, since the rings are an encoding too.
            fig.add_scatter(
                x=[None], y=[None], mode="markers", name=f"{highlight} County",
                marker=dict(size=11, color=t["context"],
                            line=dict(width=2, color=t["outline"])),
                hoverinfo="skip", showlegend=True,
            )

        # Name what the colours encode; when a county is outlined on top of
        # that, the title says so rather than leaving a stray legend entry.
        legend_title = color_label if discrete else ""
        if discrete and highlight:
            legend_title = f"{color_label}  ·  outlined: {highlight} Co."
        elif highlight:
            legend_title = ""

        trend = fit_trend(b["population_2025_est"].to_numpy(),
                          b[ym.col].to_numpy(), log_x, log_y) if show_fit else None
        if trend:
            fig.add_scatter(
                x=trend["line_x"], y=trend["line_y"], mode="lines",
                name="Trend", hoverinfo="skip", showlegend=False,
                line=dict(color=t["ink_secondary"], width=2, dash="solid"),
            )
            # Under the marks: it is a reference, not a series.
            fig.data = fig.data[-1:] + fig.data[:-1]

            # beta and R-squared beside the line, in the corner the line leaves
            # empty: a downward fit frees the top right, an upward one the top left.
            corner = dict(x=0.99, xanchor="right") if trend["slope"] < 0 \
                else dict(x=0.01, xanchor="left")
            fig.add_annotation(
                xref="paper", yref="paper", y=0.98, yanchor="top", **corner,
                align="left", showarrow=False,
                text=(f"<b>β</b> {trend['slope']:+.3f}<br>"
                      f"<b>R²</b> {trend['r2']:.3f}<br>"
                      f"<b>n</b> {trend['n']:,}"),
                font=dict(family=th.FONT, size=12, color=t["ink_secondary"]),
                bgcolor=t["surface"], bordercolor=t["grid"], borderwidth=1,
                borderpad=8,
            )

        fig.update_layout(
            **th.layout(t, legend_title=legend_title),
            height=620,
            # Nearest-point hover, so a 5px bubble in a dense field is still reachable.
            hovermode="closest",
            xaxis=th.axis(t, xm.label, log_x),
            yaxis=th.axis(t, ym.label, log_y),
            showlegend=discrete or bool(highlight),
        )
        fig.update_xaxes(tickformat=mx.tickformat(xm.fmt, log_x))
        fig.update_yaxes(tickformat=mx.tickformat(ym.fmt, log_y))
        if color_col and not discrete:
            cm = mx.BY_COL[color_col]
            fig.update_layout(coloraxis_colorbar=dict(
                title=dict(text=cm.label.replace(" ", "<br>", 1),
                           font=dict(color=t["ink_secondary"], size=12)),
                tickfont=dict(color=t["ink_muted"], size=11),
                thickness=12, len=0.6, outlinewidth=0,
                tickformat=mx.tickformat(cm.fmt),
            ))

        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)

        notes = [n for n in (ym.note, mx.BY_COL[size_col].note if size_col else "") if n]
        shown = len(b)
        caption = f"{shown:,} townships plotted."
        if shown < len(d):
            caption += f" {len(d) - shown:,} dropped for missing or non-positive values."

        if trend:
            if log_x and log_y:
                # On log-log axes beta is an elasticity, so it has a plain reading.
                caption += (
                    f" <b>β</b> is the log–log slope: a township ten times larger has "
                    f"{10 ** trend['slope']:.2f}× the {ym.label.lower()}."
                )
            else:
                caption += (f" <b>β</b> is {trend['slope']:+,.4g} "
                            f"{'per resident' if not log_x else 'per tenfold of population'}, "
                            "fitted on the axes as drawn.")
            # A slope is easy to over-read when it explains almost nothing.
            if trend["r2"] < 0.10:
                caption += (f" Treat that lightly: R² is {trend['r2']:.3f}, so population "
                            "accounts for almost none of the spread and the townships are "
                            "better read individually than off the line.")
            elif trend["r2"] < 0.30:
                caption += (f" R² is {trend['r2']:.2f}, so the line is a weak central "
                            "tendency, not a prediction.")

        if highlight and hl_n:
            h = b[b["county"] == highlight]
            if len(h):
                caption += (f" <b>{highlight} County</b>: {len(h)} of {hl_n} townships "
                            f"plotted, median {ym.label.lower()} "
                            f"{mx.fmt(h[ym.col].median(), ym.fmt)} against "
                            f"{mx.fmt(b[ym.col].median(), ym.fmt)} for the field.")
                if trend:
                    above = int((trend["residual"](h["population_2025_est"].to_numpy(),
                                                   h[ym.col].to_numpy()) > 0).sum())
                    caption += f" {above} of {len(h)} sit above the trend line."
        st.markdown(
            f'<p class="metric-note">{caption} ' + " ".join(notes) + "</p>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- map
with tab_map:
    map_opts = [m.label for m in mx.CATALOG]
    mc1, mc2 = st.columns([1, 2])
    map_label = mc1.selectbox("Shade by", map_opts,
                              index=map_opts.index("Disbursements per resident"),
                              key="map_metric")
    mm = mx.BY_COL[mx.LABEL_TO_COL[map_label]]
    scale_mode = mc2.radio(
        "Color scale",
        ["By rank", "By value", "By value, clipped"],
        horizontal=True, key="map_scale",
        help="These measures are heavily right-skewed, so a linear scale by value "
             "paints almost every township the same shade. Ranking spreads the ramp "
             "evenly across the townships in view; clipping cuts the tails at the "
             "2nd and 98th percentile.",
    )

    geo = load_geo()
    m = d[d[mm.col].notna()].copy()

    if m.empty:
        st.info("No townships match these filters.")
    else:
        vals = m[mm.col]
        by_rank = scale_mode == "By rank"

        if by_rank:
            # Percentile within the townships currently in view, so the ramp is
            # used evenly no matter how skewed the underlying measure is.
            m["_shade"] = vals.rank(pct=True)
            shade_col, rng, diverging = "_shade", [0.0, 1.0], False
            cb_title, cb_fmt = f"{mm.label}<br>(percentile)", ".0%"
        else:
            shade_col = mm.col
            rng = ([float(vals.quantile(0.02)), float(vals.quantile(0.98))]
                   if scale_mode == "By value, clipped"
                   else [float(vals.min()), float(vals.max())])
            if rng[0] == rng[1]:
                rng = [rng[0], rng[0] + 1]
            diverging = (vals.min() < 0 < vals.max())
            if diverging:
                lim = max(abs(rng[0]), abs(rng[1]))
                rng = [-lim, lim]
            cb_title, cb_fmt = mm.label, mx.tickformat(mm.fmt)

        scale = t["diverging"] if diverging else t["sequential"]

        # Population is always in the hover; don't list it twice when it is also
        # the shaded measure.
        mcustom = ["trustee_label", "population_2025_est"]
        if mm.col != "population_2025_est":
            mcustom.append(mm.col)
        # The hover always reports the real value, never the percentile.
        mrows = ["<b>%{hovertext}</b>", "%{customdata[0]}", "<br>",
                 "Population: <b>%{customdata[1]:,.0f}</b>"]
        if mm.col != "population_2025_est":
            mrows.append(f"{mm.label}: <b>%{{customdata[2]:{mx.tickformat(mm.fmt)}}}</b>")

        fig = px.choropleth(
            m,
            geojson=geo,
            locations="geo_key",
            featureidkey="properties.key",
            color=shade_col,
            color_continuous_scale=scale,
            range_color=rng,
            hover_name="name",
            custom_data=mcustom,
        )
        # Explicit mercator bounds for Indiana. fitbounds="locations" leaves the
        # state filling barely a tenth of the canvas; these bounds fill it.
        fig.update_geos(
            visible=False, bgcolor=t["surface"], projection_type="mercator",
            lataxis_range=[37.70, 41.85], lonaxis_range=[-88.15, -84.72],
            # Leave the bottom strip of the canvas for the horizontal scale.
            domain=dict(x=[0, 1], y=[0.10, 1.0]),
        )
        fig.update_traces(
            marker_line_color=t["surface"],
            marker_line_width=0.4,
            hovertemplate="<br>".join(mrows) + "<extra></extra>",
        )
        if highlight and hl_n:
            # A second trace over the same colour axis: identical shading, but
            # outlined, so the county reads as part of the state rather than
            # a separate chart.
            hm = m[m["county"] == highlight]
            fig.add_trace(go.Choropleth(
                geojson=geo,
                locations=hm["geo_key"],
                featureidkey="properties.key",
                z=hm[shade_col],
                coloraxis="coloraxis",
                marker_line_color=t["outline"],
                marker_line_width=1.4,
                name=f"{highlight} County",
                hoverinfo="skip",
                showlegend=False,
            ))

        map_layout = th.layout(t)
        map_layout["margin"] = dict(l=0, r=0, t=4, b=4)
        fig.update_layout(
            **map_layout,
            height=680,
            # Indiana is tall and narrow, so the scale goes underneath rather
            # than taking width the map needs.
            coloraxis_colorbar=dict(
                title=dict(text=cb_title, side="top",
                           font=dict(color=t["ink_secondary"], size=12)),
                tickfont=dict(color=t["ink_muted"], size=11),
                orientation="h", x=0.5, xanchor="center", y=0.012, yanchor="bottom",
                thickness=11, len=0.58, outlinewidth=0,
                tickformat=cb_fmt,
            ),
        )
        _, mmid, _ = st.columns([1, 2.6, 1])
        with mmid:
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        if by_rank:
            tail = (" Shaded by rank within the townships in view, so equal areas of "
                    "color hold equal numbers of townships; hover for the real value.")
        elif scale_mode == "By value, clipped":
            tail = (" Color scale clipped at the 2nd and 98th percentile "
                    f"({mx.fmt(rng[0], mm.fmt)} to {mx.fmt(rng[1], mm.fmt)}); "
                    "townships beyond that range take the end color.")
        else:
            tail = ""
        hl_note = ""
        if highlight and hl_n:
            hl_note = (f" <b>{highlight} County</b>'s {hl_n} townships are outlined, "
                       "keeping their shading so they read against the rest of the state.")
        st.markdown(
            f'<p class="metric-note">{len(m):,} townships shaded. {mm.note}{tail}{hl_note} '
            "Boundaries: Census 2023 cartographic county subdivisions.</p>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- rankings
with tab_rank:
    r1, r2, r3 = st.columns([1.4, 0.7, 0.7])
    rank_opts = [m.label for m in mx.CATALOG]
    rank_label = r1.selectbox("Rank by", rank_opts,
                              index=rank_opts.index("Trustee pay per resident"),
                              key="rank_metric")
    direction = r2.radio("Order", ["Highest", "Lowest"], horizontal=True, key="rank_order")
    topn = r3.slider("Rows", 10, 40, 20)

    rm = mx.BY_COL[mx.LABEL_TO_COL[rank_label]]
    r = d[d[rm.col].notna()].nlargest(topn, rm.col) if direction == "Highest" \
        else d[d[rm.col].notna()].nsmallest(topn, rm.col)
    r = r.sort_values(rm.col)

    # Only de-emphasise the field when there is actually something to pick out;
    # otherwise every bar goes grey and the chart loses its colour for nothing.
    hl_in_view = int((r["county"] == highlight).sum()) if highlight else 0
    use_hl = bool(highlight) and hl_in_view > 0

    if r.empty:
        st.info("No townships match these filters.")
    else:
        fig = go.Figure(go.Bar(
            x=r[rm.col],
            y=r["name"],
            orientation="h",
            # One hue for the field; the highlighted county keeps it while the
            # rest step back, so the bars stay ranked by length, not by colour.
            marker=dict(
                color=([t["series"][0] if c == highlight else t["context"]
                        for c in r["county"]] if use_hl else t["series"][0]),
                line=dict(width=0),
            ),
            # 4px rounded data-end, anchored to the baseline.
            marker_cornerradius=4,
            text=[mx.fmt(v, rm.fmt) for v in r[rm.col]],
            textposition="outside",
            textfont=dict(color=t["ink_secondary"], size=11, family=th.FONT),
            customdata=np.stack([r["trustee_label"], r["population_2025_est"]], axis=-1),
            hovertemplate=("<b>%{y}</b><br>%{customdata[0]}<br><br>"
                           f"{rm.label}: <b>%{{x:{mx.tickformat(rm.fmt)}}}</b><br>"
                           "Population: <b>%{customdata[1]:,.0f}</b><extra></extra>"),
        ))
        fig.update_layout(
            **th.layout(t),
            height=max(320, 26 * len(r) + 90),
            bargap=0.35,
            xaxis=th.axis(t, rm.label),
            yaxis=dict(title=None, tickfont=dict(color=t["ink_secondary"], size=11),
                       showgrid=False, linecolor=t["axis"], zeroline=False),
            showlegend=False,
        )
        fig.update_xaxes(tickformat=mx.tickformat(rm.fmt), showgrid=True,
                         range=[0, float(r[rm.col].max()) * 1.18] if rm.positive else None)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        rank_note = rm.note
        if use_hl:
            rank_note = (f"{highlight} County holds {hl_in_view} of these {len(r)} rows, "
                         f"in colour against the rest in grey. {rm.note}")
        elif highlight and hl_n:
            # Nothing of theirs made the cut, so say where they do land.
            full = (d[d[rm.col].notna()]
                    .sort_values(rm.col, ascending=(direction == "Lowest"))
                    .reset_index(drop=True))
            hits = full.index[full["county"] == highlight]
            if len(hits):
                i = int(hits.min())
                rank_note = (f"No {highlight} County township is in the {direction.lower()} "
                             f"{len(r)}. Its leader is {full.loc[i, 'name']} at "
                             f"{mx.fmt(full.loc[i, rm.col], rm.fmt)}, ranked "
                             f"{i + 1:,} of {len(full):,}. {rm.note}")
        st.markdown(f'<p class="metric-note">{rank_note}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- table
with tab_table:
    st.markdown(
        '<p class="metric-note">Every township in the current filter, with the derived '
        "ratios alongside the reported figures. Click a column header to sort; the AFR "
        "column links to the township's filing on Gateway.</p>",
        unsafe_allow_html=True,
    )
    cols = ["county", "township", "trustee_name", "trustee_2025_compensation",
            "population_2025_est", "total_receipts", "total_disbursements",
            "operating_costs", "township_assistance_paid", "capital_outlays",
            "ending_cash_and_investments", "disb_per_resident", "comp_per_resident",
            "assistance_per_resident", "reserve_years", "assistance_share",
            "net_surplus", "afr_year", "afr_full_report_link", "notes"]
    tbl = d[cols].copy()
    # Shares are stored as fractions; the table shows percentage points.
    tbl["assistance_share"] = tbl["assistance_share"] * 100
    show = tbl.rename(columns={
        c: mx.BY_COL[c].label for c in cols if c in mx.BY_COL
    }).rename(columns={
        "county": "County", "township": "Township", "trustee_name": "Trustee",
        "afr_year": "AFR year", "afr_full_report_link": "AFR", "notes": "Notes",
    })
    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
        height=620,
        column_config={
            "AFR": st.column_config.LinkColumn("AFR", display_text="Gateway"),
            "Trustee compensation": st.column_config.NumberColumn(format="$%d"),
            "Population (2025 est.)": st.column_config.NumberColumn(format="%d"),
            "Total receipts": st.column_config.NumberColumn(format="$%d"),
            "Total disbursements": st.column_config.NumberColumn(format="$%d"),
            "Operating costs": st.column_config.NumberColumn(format="$%d"),
            "Township assistance paid": st.column_config.NumberColumn(format="$%d"),
            "Capital outlays": st.column_config.NumberColumn(format="$%d"),
            "Ending cash and investments": st.column_config.NumberColumn(format="$%d"),
            "Disbursements per resident": st.column_config.NumberColumn(format="$%.2f"),
            "Trustee pay per resident": st.column_config.NumberColumn(format="$%.2f"),
            "Assistance paid per resident": st.column_config.NumberColumn(format="$%.2f"),
            "Reserve (years of spending held)": st.column_config.NumberColumn(format="%.2f"),
            "Assistance as share of spending": st.column_config.NumberColumn(
                "Assistance share of spending", format="%.1f%%",
                help="Township assistance paid as a percentage of total disbursements."),
            "Net surplus (receipts − disbursements)": st.column_config.NumberColumn(format="$%d"),
        },
    )
    st.download_button(
        "Download this slice as CSV",
        d[cols].to_csv(index=False).encode(),
        file_name="indiana_townships_filtered.csv",
        mime="text/csv",
    )
