# Indiana township trustees, 2025

A Streamlit explorer for the 1,003 Indiana township trustee filings in
`indiana_townships_2025.csv` — compensation from the 2025 Form 100R, financials
from the annual financial reports on Gateway.

## Run it

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Deploying

Entrypoint `Trustees/app.py`. Streamlit Community Cloud runs from the **repo
root**, not from this directory, so:

- Data paths are anchored with `Path(__file__).with_name(...)` in `metrics.py`,
  never resolved against the working directory. A bare `"indiana_townships_2025.csv"`
  works locally and fails on deploy.
- `.streamlit/config.toml` lives here, not at the repo root. Streamlit resolves a
  script-level config last so it overrides the project and global ones, which means
  this app's theme applies on Cloud without disturbing the sibling apps in this repo.
- `requirements.txt` lives here too; Cloud prefers the file next to the entrypoint
  over the repo root one.

`scripts/sweep_controls.py` runs from the repo root for this reason — a path
resolved against the working directory fails there rather than on deploy.

## What's in it

| Tab | What it shows |
|---|---|
| **Bubble** | Population on x; y, bubble size, and color each switch between any of the 20 measures. Log-log by default, with a least-squares trend line and its β / R² / n. |
| **Map** | All 1,003 townships shaded by any measure, by rank or by value. |
| **Rankings** | Top or bottom N townships on any measure. |
| **Table** | The full filtered slice, sortable, with links to each AFR and a CSV download. |

Sidebar filters (county, population band, completeness) scope all four tabs.

## Highlight vs. filter

Two different controls, deliberately kept apart:

- The **County filter** *subsets* — everything else leaves the view.
- **Highlight → County** *keeps the whole field* and picks one county out of it, so
  you can see where its townships sit relative to the other 990. On the bubble it
  outlines them and fades the rest; on the map it outlines them in place, keeping
  their shading; in Rankings it colors their bars, or, if none of them made the
  top N, tells you where their leader actually ranks.

Each chart's caption reports how the highlighted county compares — median against
the field, and how many of its townships sit above the trend line.

## Reading the trend line

The fit runs in whatever space the axes are drawn in. With both logged — the
default — it is a power law, so **β is an elasticity**: β = −0.5 means pay per
resident falls by half for every fourfold increase in population. The caption
translates β into a plain "ten times larger has N× the …" reading, and cautions
when R² is below 0.30 (weak) or 0.10 (essentially nothing), because a slope is
easy to over-read when it explains almost none of the spread.

Worth knowing before you go looking: most per-resident measures have a very low
R² against population. The one strong relationship is **trustee pay per
resident** (β −0.50, R² 0.64) — trustee pay scales as roughly the square root of
population, so small townships pay far more per resident.

## Derived measures

The CSV reports raw dollars, which mostly track population — a $2M township looks
alarming until you notice it has 40,000 residents. `metrics.py` adds the ratios
that make townships of different sizes comparable:

- per-resident disbursements, operating costs, assistance, trustee pay, and cash
- **reserve years** — ending cash and investments over one year of disbursements
- **assistance share** and **trustee pay share** of total disbursements
- net surplus and 2020→2025 population change

## Data caveats

- Compensation is the **highest-paid trustee** where a township reported more than
  one; the others are in the `notes` column.
- 12 townships have no 2025 AFR — their financials are 2024 or missing. 31 name no
  trustee. Both are kept in view by default; the sidebar can exclude them.
- Townships with a reported population of 0 or missing are dropped from
  per-resident measures rather than divided by zero.

## Files

| File | |
|---|---|
| `app.py` | The Streamlit app: filters, four tabs, hovers |
| `metrics.py` | Loads the CSV, derives the ratios, holds the measure catalog |
| `theme.py` | Chart color tokens and Plotly layout, light and dark |
| `.streamlit/config.toml` | Both Streamlit themes, matched to `theme.py` |
| `data/indiana_townships.geojson` | Township boundaries for the map (built once; see below) |

### Rebuilding the map boundaries

`data/indiana_townships.geojson` is derived from the Census 2023 cartographic
boundary file for Indiana county subdivisions (`cb_2023_18_cousub_500k`),
simplified and re-keyed to the CSV's county/township names. It is checked in, so
the app never needs the network. `scripts/build_geo.py` regenerates it.

### Smoke-testing the controls

`scripts/sweep_controls.py` runs the app headlessly through every value of every
selector (and the edge slices: one county, the smallest population band, signed
measures, empty results) and fails on any exception:

```sh
.venv/bin/python scripts/sweep_controls.py
```
