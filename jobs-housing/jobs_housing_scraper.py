import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CENSUS_API_KEY")

OUTPUT_FILE = "county_jobs_housing.parquet"

STATES = ["18", "17", "26", "39", "21"]

QWI_URL = "https://api.census.gov/data/timeseries/qwi/se"
ACS_URL = "https://api.census.gov/data"


# ----------------------
# Helper
# ----------------------
def census_get(url, params):
    r = requests.get(url, params=params)

    if r.status_code != 200:
        print(f"[ERROR] {r.status_code}: {r.text[:200]}")
        return None

    try:
        return r.json()
    except Exception:
        print(f"[ERROR] Non-JSON response: {r.text[:200]}")
        return None


# ----------------------
# Load existing
# ----------------------
def load_existing():
    if os.path.exists(OUTPUT_FILE):
        return pd.read_parquet(OUTPUT_FILE)
    return pd.DataFrame()


# ----------------------
# QWI periods
# ----------------------
def discover_qwi_periods():
    params = {
        "get": "Emp",
        "for": "state:18",
        "time": "from 2018-Q1 to 2030-Q4",
        "key": API_KEY
    }

    data = census_get(QWI_URL, params)
    if not data:
        return []

    headers = data[0]
    time_idx = headers.index("time")

    return sorted(set(row[time_idx] for row in data[1:]))


def get_target_periods(existing_df, all_periods):
    if existing_df.empty:
        return all_periods[-3:]

    existing = set(existing_df["period"].unique())
    return [p for p in all_periods if p not in existing]


def get_acs_year_from_qwi(periods):
    latest = max(int(p.split("-Q")[0]) for p in periods)
    return latest - 1


# ----------------------
# QWI (jobs)
# ----------------------
def fetch_qwi(periods):
    rows = []

    for period in periods:
        year, q = period.split("-Q")

        for state in STATES:
            params = {
                "get": "Emp",
                "for": "county:*",
                "in": f"state:{state}",
                "time": period,
                "key": API_KEY
            }

            data = census_get(QWI_URL, params)
            if not data or len(data) <= 1:
                continue

            df = pd.DataFrame(data[1:], columns=data[0])
            df["period"] = period
            df["year"] = int(year)
            df["quarter"] = int(q)

            rows.append(df)

    qwi = pd.concat(rows, ignore_index=True)
    qwi["Emp"] = pd.to_numeric(qwi["Emp"], errors="coerce")

    return qwi


# ----------------------
# ACS (housing + demographics)
# ----------------------
AGE_COLS = [
    "B01001_001E",  # total pop
    "B01001_003E","B01001_004E","B01001_005E","B01001_006E","B01001_007E","B01001_008E",
    "B01001_009E","B01001_010E","B01001_011E","B01001_012E","B01001_013E","B01001_014E",
    "B01001_015E","B01001_016E","B01001_017E","B01001_018E","B01001_019E","B01001_020E",
    "B01001_021E","B01001_022E","B01001_023E",
    "B01001_027E","B01001_028E","B01001_029E","B01001_030E","B01001_031E","B01001_032E",
    "B01001_033E","B01001_034E","B01001_035E","B01001_036E","B01001_037E","B01001_038E",
    "B01001_039E","B01001_040E","B01001_041E","B01001_042E","B01001_043E","B01001_044E",
    "B01001_045E","B01001_046E","B01001_047E",
    "B25001_001E"  # housing units
]


def fetch_acs(years):
    rows = []

    for year in years:
        for state in STATES:

            url = f"{ACS_URL}/{year}/acs/acs5"

            params = {
                "get": ",".join(AGE_COLS),
                "for": "county:*",
                "in": f"state:{state}",
                "key": API_KEY
            }

            data = census_get(url, params)
            if not data:
                continue

            df = pd.DataFrame(data[1:], columns=data[0])
            df["year"] = int(year)

            for c in df.columns:
                if c.startswith("B"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # ----------------------
            # DEMOGRAPHIC FEATURES
            # ----------------------
            total_pop = df["B01001_001E"]

            under18 = (
                df["B01001_003E"] + df["B01001_004E"] + df["B01001_005E"] +
                df["B01001_006E"] + df["B01001_027E"] + df["B01001_028E"] +
                df["B01001_029E"] + df["B01001_030E"]
            )

            over65 = (
                df["B01001_020E"] + df["B01001_021E"] + df["B01001_022E"] +
                df["B01001_023E"] + df["B01001_044E"] + df["B01001_045E"] +
                df["B01001_046E"] + df["B01001_047E"]
            )

            # --- NEW AGE BANDS ---
            # 18-22ish (College age)
            # Includes 18-19, 20, 21
            age_18_22 = (
                df["B01001_007E"] + df["B01001_008E"] + df["B01001_009E"] +
                df["B01001_031E"] + df["B01001_032E"] + df["B01001_033E"]
            )
            # 23-34ish (Young Professional age)
            # Includes 22-24, 25-29, 30-34
            age_23_34 = (
                df["B01001_010E"] + df["B01001_011E"] + df["B01001_012E"] +
                df["B01001_034E"] + df["B01001_035E"] + df["B01001_036E"]
            )
            # 35-49ish (Mid Career)
            age_35_49 = (
                df["B01001_013E"] + df["B01001_014E"] + df["B01001_015E"] +
                df["B01001_037E"] + df["B01001_038E"] + df["B01001_039E"]
            )
            # 50-64ish (Pre Retirement)
            age_50_64 = (
                df["B01001_016E"] + df["B01001_017E"] + df["B01001_018E"] +
                df["B01001_019E"] + df["B01001_040E"] + df["B01001_041E"] +
                df["B01001_042E"] + df["B01001_043E"]
            )

            working_age = total_pop - under18 - over65

            df["pct_under18"] = under18 / total_pop
            df["pct_over65"] = over65 / total_pop
            df["pct_18_22"] = age_18_22 / total_pop
            df["pct_23_34"] = age_23_34 / total_pop
            df["pct_35_49"] = age_35_49 / total_pop
            df["pct_50_64"] = age_50_64 / total_pop
            df["pct_working_age"] = working_age / total_pop
            
            df["count_under18"] = under18
            df["count_18_22"] = age_18_22
            df["count_23_34"] = age_23_34
            df["count_35_49"] = age_35_49
            df["count_50_64"] = age_50_64
            df["count_over65"] = over65
            df["count_working_age"] = working_age

            df["pct_non_working"] = 1 - df["pct_working_age"]

            # ----------------------
            # AVG AGE (weighted approximation)
            # ----------------------
            age_midpoints = {
                "B01001_003E": 2, "B01001_004E": 7, "B01001_005E": 12,
                "B01001_006E": 16, "B01001_007E": 18.5, "B01001_008E": 22,
                "B01001_009E": 27, "B01001_010E": 32, "B01001_011E": 37,
                "B01001_012E": 42, "B01001_013E": 47, "B01001_014E": 52,
                "B01001_015E": 57, "B01001_016E": 60.5, "B01001_017E": 63,
                "B01001_018E": 65.5, "B01001_019E": 68, "B01001_020E": 72,
                "B01001_021E": 77, "B01001_022E": 82, "B01001_023E": 90,
                "B01001_027E": 2, "B01001_028E": 7, "B01001_029E": 12,
                "B01001_030E": 16, "B01001_031E": 18.5, "B01001_032E": 22,
                "B01001_033E": 27, "B01001_034E": 32, "B01001_035E": 37,
                "B01001_036E": 42, "B01001_037E": 47, "B01001_038E": 52,
                "B01001_039E": 57, "B01001_040E": 60.5, "B01001_041E": 63,
                "B01001_042E": 65.5, "B01001_043E": 68, "B01001_044E": 72,
                "B01001_045E": 77, "B01001_046E": 82, "B01001_047E": 90
            }

            weighted_age = 0
            pop_age = 0

            for col, mid in age_midpoints.items():
                if col in df.columns:
                    weighted_age += df[col] * mid
                    pop_age += df[col]

            df["avg_age"] = weighted_age / pop_age

            rows.append(df)

    return pd.concat(rows, ignore_index=True)


# ----------------------
# Names and Helpers
# ----------------------
STATE_FIPS_MAP = {
    "17": "Illinois",
    "18": "Indiana",
    "21": "Kentucky",
    "26": "Michigan",
    "39": "Ohio"
}

def classify_balance(x):
    if pd.isna(x): return None
    if x < 0.7:
        return "🔴 Strong shortage (tight housing stock relative to jobs)"
    elif x < 0.9:
        return "🟡 Mild shortage pressure"
    elif x <= 1.3:
        return "🟢 Within structural balance range"
    elif x <= 1.6:
        return "🟡 Mild surplus (housing exceeds jobs)"
    else:
        return "🔴 Strong surplus / commuter-exporting area"

def fetch_county_names():
    rows = []
    # Use 2022 as a stable year for names
    for state in STATES:
        url = f"{ACS_URL}/2022/acs/acs5"
        params = {"get": "NAME", "for": "county:*", "in": f"state:{state}", "key": API_KEY}
        data = census_get(url, params)
        if data:
            df = pd.DataFrame(data[1:], columns=data[0])
            rows.append(df)
    
    if not rows:
        return pd.DataFrame()
    
    names_df = pd.concat(rows, ignore_index=True)
    # "Adams County, Indiana" -> "Adams"
    names_df["county_name"] = names_df["NAME"].str.split(",").str[0].str.replace(" County", "")
    names_df["fips"] = names_df["state"].str.zfill(2) + names_df["county"].str.zfill(3)
    return names_df[["fips", "county_name"]]

# ----------------------
# MAIN
# ----------------------
def main():
    existing = load_existing()

    periods = discover_qwi_periods()
    target = get_target_periods(existing, periods)

    if not target:
        print("No new data.")
        return

    print("Fetching:", target)

    acs_year = get_acs_year_from_qwi(target)

    qwi = fetch_qwi(target)
    acs = fetch_acs([acs_year])
    
    # Get names
    names = fetch_county_names()

    merged = pd.merge(qwi, acs, on=["state", "county", "year"], how="left")

    merged = merged.rename(columns={
        "Emp": "jobs",
        "B25001_001E": "housing_units"
    })

    merged["fips"] = merged["state"].str.zfill(2) + merged["county"].str.zfill(3)
    
    # Add Names
    if not names.empty:
        merged = pd.merge(merged, names, on="fips", how="left")
    
    merged["state_name"] = merged["state"].map(STATE_FIPS_MAP)
    merged["full_name"] = merged["county_name"] + ", " + merged["state_name"]
    
    # Pre-calculate metrics
    merged["housing_per_job"] = merged["housing_units"] / merged["jobs"]
    merged["balance_flag"] = merged["housing_per_job"].apply(classify_balance)

    final = pd.concat([existing, merged], ignore_index=True) if not existing.empty else merged

    # Ensure existing records also have these fields if they were missing
    if not existing.empty:
        # If we added columns, some existing rows might have NaNs
        # We can re-apply the mapping and calculations to the whole thing if needed,
        # but for now let's just ensure the new columns are present.
        if "state_name" not in final.columns:
            final["state_name"] = final["state"].map(STATE_FIPS_MAP)
        if "full_name" not in final.columns:
             # This might be tricky if county_name is missing for old rows
             pass 

    final = final.drop_duplicates(subset=["fips", "period"])
    final = final.sort_values(["fips", "year", "quarter"])

    final.to_parquet(OUTPUT_FILE, index=False)

    print("Parquet updated with full demographic structure and names.")


if __name__ == "__main__":
    main()