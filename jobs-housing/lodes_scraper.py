import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CENSUS_API_KEY")

OUTPUT_FILE = "lodes_commuting.parquet"

# State mapping for URL construction and FIPS identification
STATE_CONFIG = {
    "IN": {"fips": "18", "name": "Indiana"},
    "IL": {"fips": "17", "name": "Illinois"},
    "KY": {"fips": "21", "name": "Kentucky"},
    "MI": {"fips": "26", "name": "Michigan"},
    "OH": {"fips": "39", "name": "Ohio"}
}

STATE_FIPS = {k: v["fips"] for k, v in STATE_CONFIG.items()}
STATE_FIPS_TO_NAME = {v["fips"]: v["name"] for k, v in STATE_CONFIG.items()}

STATES = list(STATE_CONFIG.keys())
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


# -----------------------
# Load existing
# -----------------------
def load_existing():
    if os.path.exists(OUTPUT_FILE):
        return pd.read_parquet(OUTPUT_FILE)
    return pd.DataFrame()


# -----------------------
# SAFE LODES LOADER
# -----------------------
def load_lodes_od(state_abbr: str, year: int):
    # state_abbr is used directly in the URL (e.g., 'in', 'il')
    url_state = state_abbr.lower()

    url = (
        f"https://lehd.ces.census.gov/data/lodes/LODES8/"
        f"{url_state}/od/{url_state}_od_main_JT00_{year}.csv.gz"
    )

    try:
        df = pd.read_csv(url, compression="gzip")
        return df
    except Exception as e:
        print(f"[LODES SKIP] {state_abbr}-{year}: {e}")
        return None


# -----------------------
# Convert OD → county flows
# -----------------------
def lodes_to_county(df, state_abbr, year):
    # This DF contains everyone working in 'state_abbr'
    df = df.copy()
    df["jobs"] = df["S000"]
    df["home_fips"] = df["h_geocode"].astype(str).str.zfill(15).str[:5]
    df["work_fips"] = df["w_geocode"].astype(str).str.zfill(15).str[:5]

    # 1. TOTAL JOBS: Count by Workplace (all people working in these counties)
    total_jobs_in_county = df.groupby("work_fips")["jobs"].sum()

    # 2. IN-COMMUTERS: Work in county, live elsewhere (anywhere in US)
    in_mask = df["work_fips"] != df["home_fips"]
    in_commuters = df[in_mask].groupby("work_fips")["jobs"].sum()

    # 3. INTERNAL: Live and work in the same county
    internal_mask = df["work_fips"] == df["home_fips"]
    internal_workers = df[internal_mask].groupby("work_fips")["jobs"].sum()

    # 4. POTENTIAL OUT-COMMUTERS: Residents of these counties working in THIS state
    # We will aggregate this across all state files to get the true total.
    out_commuters = df[in_mask].groupby("home_fips")["jobs"].sum()

    result = pd.DataFrame({
        "lodes_total_jobs": total_jobs_in_county,
        "in_commuters": in_commuters,
        "internal_workers": internal_workers,
        "out_commuters": out_commuters
    }).fillna(0)

    result = result.reset_index().rename(columns={"index": "fips"})
    result["year"] = year
    return result

# -----------------------
# MAIN
# -----------------------
def main():
    existing = load_existing()
    todo = get_missing(existing)
    if not todo: return

    print(f"Fetching {len(todo)} combinations")
    all_raw_flows = []
    
    for state, year in todo:
        print(f"Fetching {state}-{year}")
        df = load_lodes_od(state, year)
        if df is None or df.empty: continue
        
        # Get flows from this state's perspective
        out = lodes_to_county(df, state, year)
        all_raw_flows.append(out)

    if not all_raw_flows: return

    # 1. Combine everything
    new_df = pd.concat(all_raw_flows, ignore_index=True)
    
    # 2. Aggregate by FIPS and Year
    # For a county in Indiana:
    # 'lodes_total_jobs' will be captured correctly in the IN file (where it is work_fips).
    # 'in_commuters' will be captured correctly in the IN file (where it is work_fips).
    # 'out_commuters' will be summed across IN, IL, KY, MI, OH files (where it is home_fips).
    print("Aggregating regional flows...")
    agg_df = new_df.groupby(["fips", "year"], as_index=False).agg({
        "lodes_total_jobs": "max", # Total jobs in the county is specific to that county's workplace records
        "in_commuters": "max",     # In-commuters to a county is specific to that county's workplace records
        "internal_workers": "max", # Internal is specific
        "out_commuters": "sum"     # SUM: A resident of Marion, IN might work in Hamilton, IN OR Cook, IL.
    })
    
    agg_df["net_commute"] = agg_df["in_commuters"] - agg_df["out_commuters"]
    
    # 3. Filter to only include counties that belong to our 5 states
    # (Otherwise the 'sum' captures out-commuters to counties we don't care about)
    INV_STATE_FIPS = {v: k for k, v in STATE_FIPS.items()}
    agg_df["state_abbr"] = agg_df["fips"].str[:2].map(INV_STATE_FIPS)
    agg_df = agg_df.dropna(subset=["state_abbr"])

    print("Fetching and joining names...")
    names = fetch_county_names()
    if not names.empty:
        agg_df = pd.merge(agg_df, names, on="fips", how="left")
        agg_df["state_name"] = agg_df["state"].map(STATE_FIPS_TO_NAME)
        agg_df["full_name"] = agg_df["county_name"] + ", " + agg_df["state_name"]

    final = pd.concat([existing, agg_df], ignore_index=True) if not existing.empty else agg_df
    final = final.drop_duplicates(subset=["fips", "year"])
    final.to_parquet(OUTPUT_FILE, index=False)
    print("LODES updated with true cross-state out-commuter sums.")


if __name__ == "__main__":
    main()