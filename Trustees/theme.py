"""Design tokens for the charts.

Two selected modes, not an automatic flip: the dark steps are the same hues
re-stepped for the dark surface. The categorical slots are capped at three
because every chart here is an all-pairs form (bubble, choropleth), where a
fourth slot puts yellow beside orange and fails the separation floors.
"""

LIGHT = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    # Categorical slots 1-3. Validated all-pairs: worst CVD dE 9.2, normal 24.0.
    # Aqua sits at 2.74:1 on this surface, so the legend and the Table tab
    # carry identity alongside hue.
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    # Sequential blue, light -> dark. The lightest step may recede toward the
    # surface because it means "near zero".
    "sequential": [
        "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
        "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b",
    ],
    # Diverging blue <-> red with a neutral gray midpoint, for signed measures.
    "diverging": ["#104281", "#256abf", "#86b6ef", "#f0efec", "#ec8a89", "#d03b3b", "#8f2525"],
    "no_data": "#e1e0d9",
    "context": "#c3c2b7",
    "outline": "#0b0b0b",
}

DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "ink": "#ffffff",
    "ink_secondary": "#c3c2b7",
    "ink_muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70"],
    # Low -> high runs dark -> light so "near zero" recedes toward the surface.
    # Stopped at step 600 rather than 700: these measures are right-skewed, so
    # most of the map sits at the low end and the darkest steps would sink into
    # the surface.
    "sequential": [
        "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
        "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb",
    ],
    "diverging": ["#cde2fb", "#86b6ef", "#3987e5", "#383835", "#e66767", "#d03b3b", "#8f2525"],
    "no_data": "#2c2c2a",
    "context": "#52514e",
    "outline": "#ffffff",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def tokens(dark: bool) -> dict:
    return DARK if dark else LIGHT


def layout(t: dict, legend_title: str = "") -> dict:
    """Shared Plotly layout: recessive chrome, ink-token text, no chart title.

    legend_title names the dimension the colors encode, so a legend never leaves
    the reader guessing what separates the series.
    """
    return dict(
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font=dict(family=FONT, size=13, color=t["ink_secondary"]),
        margin=dict(l=8, r=8, t=8, b=8),
        hoverlabel=dict(
            bgcolor=t["surface"],
            bordercolor=t["axis"],
            font=dict(family=FONT, size=12, color=t["ink"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["ink_secondary"], size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title=dict(text=legend_title,
                       font=dict(color=t["ink_secondary"], size=12)),
        ),
    )


def axis(t: dict, title: str, log: bool = False) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=t["ink_secondary"], size=13)),
        type="log" if log else "linear",
        gridcolor=t["grid"],
        griddash="solid",
        gridwidth=1,
        zeroline=False,
        linecolor=t["axis"],
        linewidth=1,
        tickfont=dict(color=t["ink_muted"], size=11),
        showgrid=True,
    )


def css(t: dict) -> str:
    """Page chrome that Streamlit's own theme doesn't cover."""
    return f"""
    <style>
      .stApp, .stMain {{ background: {t["plane"]}; }}
      html, body, [class*="st-"] {{ font-family: {FONT}; }}
      .metric-note {{
        color: {t["ink_muted"]}; font-size: 0.8rem; line-height: 1.45;
        margin: 0.1rem 0 0.9rem 0;
      }}
      .stat-row {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem; }}
      .stat-tile {{
        flex: 1 1 160px; background: {t["surface"]}; border-radius: 10px;
        border: 1px solid {t["grid"]}; padding: 0.7rem 0.9rem;
      }}
      .stat-tile .label {{
        color: {t["ink_muted"]}; font-size: 0.72rem; letter-spacing: 0.04em;
        text-transform: uppercase; display: block; margin-bottom: 0.25rem;
      }}
      .stat-tile .value {{
        color: {t["ink"]}; font-size: 1.5rem; font-weight: 600; line-height: 1.1;
      }}
      .stat-tile .sub {{ color: {t["ink_secondary"]}; font-size: 0.78rem; }}
    </style>
    """
