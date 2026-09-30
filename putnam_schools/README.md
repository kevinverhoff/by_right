# FRL vs. proficiency explorer

Scatter of free & reduced lunch rate (x) against ILEARN proficiency (y) for every
Indiana school, with any county or set of districts highlighted against the rest
of the state. Opens on Putnam County.

## Run it

```bash
cd ~/Desktop/github/by_right/putnam_schools
source .venv/bin/activate
streamlit run app.py
```

It opens at http://localhost:8501. `Ctrl+C` in the terminal stops it,
`deactivate` leaves the venv.

One-liner, no activation needed:

```bash
~/Desktop/github/by_right/putnam_schools/.venv/bin/streamlit run \
  ~/Desktop/github/by_right/putnam_schools/app.py
```

### Rebuilding the venv from scratch

```bash
cd ~/Desktop/github/by_right/putnam_schools
python3 -m venv .venv
.venv/bin/pip install pandas openpyxl numpy scipy plotly streamlit
```

## Controls

| Control | What it does |
|---|---|
| **Test** | ELA, Math, ELA & Math, Science, Social Studies |
| **Highlight by** | A county (all 92), a hand-picked set of districts, or nothing |
| **Grades** | Published school total, or pick grades and rebuild the rate from summed counts |
| **Minimum students tested** | Drops small-n schools that swing the scatter. Default 10 |
| **FRL % range** | Zoom the comparison set to a poverty band around Putnam |
| **Exclude virtual schools** | On by default — see caveats |
| **Dark mode / Label Putnam points** | Display only. Dark mode follows your Streamlit theme by default |

Hover any dot for the school, district, FRL, proficiency, tested and enrolled
counts. The first tab ranks the highlighted schools by how far they sit above or
below the statewide line; **All schools shown** does the same for everything on
screen and exports to CSV.

Highlighting adapts to how much you selected. Up to four districts each get
their own colour and marker shape — that is the cap, because four is the widest
set of hues that stays distinguishable for colour-blind readers in both light
and dark mode. A wider selection (Marion County is 82 districts) collapses to a
single highlight colour, and labels thin to the six schools furthest from the
line; the rest stay in hover and the table.

## What the numbers mean

- **Slope** — percentage points of proficiency lost per 1-point rise in FRL.
- **vs. prediction** — a school's proficiency minus what the statewide line
  predicts at its FRL rate. This is the comparison that roughly holds poverty
  constant, and it's the column worth reading.
- The line is fitted to **every school on screen, Putnam included**, so changing
  a filter re-fits it.

## Caveats

- **Grade totals vs. published totals disagree on purpose.** The state suppresses
  any cell with fewer than 10 students (`***`). Picking grades rebuilds the rate
  from the unsuppressed cells only, so it can differ from the published school
  total. The "School total (as published)" option is the state's own figure.
- **Virtual schools are excluded by default.** Cloverdale Distance Learning
  Academy reports 0.1% FRL on 2,080 students — a statewide online roster, not a
  Putnam building. Leaving it in drags the fit and puts a misleading Putnam dot
  at x≈0. Uncheck the filter to see it.
- **Subjects aren't tested in every grade.** Science is grades 4 and 6 only;
  Social Studies is grade 5 only, so its school total *is* the grade 5 figure.
- **County is the corporation's administrative county.** It comes from
  `corp_county.csv`, built from the NCES district directory (2021–2024, newest
  year winning) joined on the state-assigned corporation id. A corporation
  straddling a county line is filed under one of them, and four recent charters
  postdate the directory — they appear in the statewide backdrop but can't be
  picked by county.
- **This is a correlation across schools, not a causal claim** about any school.
- Enrollment is all grades; tested counts are grades 3–8 only. Don't read
  tested ÷ enrolled as a participation rate.

## Files

- `app.py` — the Streamlit UI and chart
- `putnam_data.py` — loads the workbook into a school frame and a tidy
  (school, subject, grade) frame, and attaches county
- `corp_county.csv` — corp ID → county crosswalk (425 corporations, 92 counties)
- `Indiana-2026-School-Data-Combined.xlsx` — source data (ILEARN Spring 2026
  joined to 2025–26 enrollment and meal status)
