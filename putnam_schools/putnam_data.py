"""Load and reshape the Indiana 2026 combined school file.

One tidy row per (school, subject, grade) with proficient/tested counts, plus a
school-level frame carrying enrollment and FRL. Everything else in the explorer
is built from these two.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

WORKBOOK = Path(__file__).with_name("Indiana-2026-School-Data-Combined.xlsx")
COUNTY_CROSSWALK = Path(__file__).with_name("corp_county.csv")

# Grades tested per subject, per the workbook's Notes sheet.
SUBJECT_GRADES: dict[str, list[int]] = {
    "ELA": [3, 4, 5, 6, 7, 8],
    "Math": [3, 4, 5, 6, 7, 8],
    "ELA & Math": [3, 4, 5, 6, 7, 8],
    "Science": [4, 6],
    "Social Studies": [5],
}

# The county the explorer opens on.
DEFAULT_COUNTY = "Putnam"

# Schools whose FRL figure describes a statewide virtual roster rather than a
# building, so plotting them against a neighbourhood FRL rate is meaningless.
VIRTUAL_PATTERN = re.compile(
    r"virtual|online|distance learning|digital academy|e-?learning", re.I
)


def _num(series: pd.Series) -> pd.Series:
    """Coerce a column to numeric, turning the '***' suppression marker into NaN."""
    return pd.to_numeric(series, errors="coerce")


def load_county_crosswalk() -> pd.Series:
    """Corp ID -> county name.

    Built from the NCES Common Core of Data district directory (2021-2024,
    newest year winning) via the Urban Institute education data API, joined on
    the state-assigned district id. It is the LEA's administrative county, so a
    corporation straddling a county line is filed under one of them.
    """
    xwalk = pd.read_csv(COUNTY_CROSSWALK)
    return pd.Series(
        xwalk["county"].values, index=xwalk["corp_id"].astype(int), name="county"
    )


def load_schools() -> pd.DataFrame:
    """One row per school: identity, enrollment, FRL, and published subject totals."""
    df = pd.read_excel(WORKBOOK, sheet_name="Combined Data", header=1)

    schools = pd.DataFrame(
        {
            "corp_id": df["Corp ID"].astype(int),
            "district": df["District"].astype(str).str.strip(),
            "school_id": df["School ID"].astype(str).str.strip(),
            "school": df["School Name"].astype(str).str.strip(),
            "data_available": df["Data Available"],
            "enrollment": _num(df["Total Enrollment"]),
            "frl_students": _num(df["FRL Students"]),
            "frl_pct": _num(df["FRL %"]) * 100,
        }
    )

    # Published school totals, kept alongside the by-grade build because
    # suppressed grade cells make the two legitimately disagree.
    for subject in SUBJECT_GRADES:
        schools[f"total__{subject}__proficient"] = _num(
            df[f"{subject} Students Proficient"]
        )
        schools[f"total__{subject}__tested"] = _num(df[f"{subject} Students Tested"])

    # A handful of charters postdate the crosswalk; they stay in the statewide
    # backdrop but can't be picked by county.
    schools["county"] = schools["corp_id"].map(load_county_crosswalk())
    schools["is_virtual"] = schools["school"].str.contains(VIRTUAL_PATTERN)
    return schools


def load_grade_detail() -> pd.DataFrame:
    """Tidy (school, subject, grade) counts from the Grade Level Detail sheet."""
    df = pd.read_excel(WORKBOOK, sheet_name="Grade Level Detail")
    key = pd.DataFrame(
        {
            "corp_id": df["Corp ID"].astype(int),
            "school_id": df["School ID"].astype(str).str.strip(),
        }
    )

    frames = []
    for subject, grades in SUBJECT_GRADES.items():
        for grade in grades:
            proficient = _num(df[f"{subject} Gr {grade} Students Proficient"])
            tested = _num(df[f"{subject} Gr {grade} Students Tested"])
            # A suppressed numerator makes the denominator unusable too.
            tested = tested.where(proficient.notna())
            frames.append(
                key.assign(subject=subject, grade=grade,
                           proficient=proficient, tested=tested)
            )

    tidy = pd.concat(frames, ignore_index=True)
    return tidy.dropna(subset=["tested"]).query("tested > 0").reset_index(drop=True)


def build_measure(
    schools: pd.DataFrame,
    grade_detail: pd.DataFrame,
    subject: str,
    grades: list[int] | None,
) -> pd.DataFrame:
    """Attach the chosen proficiency measure to every school.

    ``grades=None`` uses the published school total. Otherwise proficiency is
    rebuilt as summed proficient / summed tested across the selected grades.
    """
    out = schools.copy()

    if grades is None:
        out["proficient"] = out[f"total__{subject}__proficient"]
        out["tested"] = out[f"total__{subject}__tested"]
    else:
        subset = grade_detail[
            (grade_detail["subject"] == subject) & (grade_detail["grade"].isin(grades))
        ]
        agg = (
            subset.groupby(["corp_id", "school_id"], as_index=False)[
                ["proficient", "tested"]
            ]
            .sum()
        )
        out = out.merge(agg, on=["corp_id", "school_id"], how="left")

    out["proficiency_pct"] = np.where(
        out["tested"].to_numpy() > 0,
        out["proficient"].to_numpy() / out["tested"].to_numpy() * 100,
        np.nan,
    )
    return out
