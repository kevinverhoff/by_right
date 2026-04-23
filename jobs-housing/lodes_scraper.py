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
    df = df.copy()
    df["jobs"] = df["S000"]

    # Ensure FIPS are 5 digits
    df["home_fips"] = df["h_geocode"].astype(str).str.zfill(15).str[:5]
    df["work_fips"] = df["w_geocode"].astype(str).str.zfill(15).str[:5]

    # 1. Total Jobs located in the county (Denominator for job share)
    total_jobs_in_county = df.groupby("work_fips")["jobs"].sum()

    # 2. In-commuters: Work in county, but live ELSEWHERE
    in_mask = df["work_fips"] != df["home_fips"]
    in_commuters = df[in_mask].groupby("work_fips")["jobs"].sum()

    # 3. Out-commuters: Live in county, but work ELSEWHERE
    out_commuters = df[in_mask].groupby("home_fips")["jobs"].sum()

    # 4. Internal Workers: Live and Work in the same county
    internal_mask = df["work_fips"] == df["home_fips"]
    internal_workers = df[internal_mask].groupby("work_fips")["jobs"].sum()

    # Combine
    result = pd.DataFrame({
        "lodes_total_jobs": total_jobs_in_county,
        "in_commuters": in_commuters,
        "out_commuters": out_commuters,
        "internal_workers": internal_workers
    }).fillna(0)

    result["net_commute"] = result["in_commuters"] - result["out_commuters"]
    result = result.reset_index().rename(columns={"index": "fips"})

    result["state_abbr"] = state_abbr
    result["year"] = year

    return result

# -----------------------
# Names
# -----------------------
def fetch_county_names():
    rows = []
    # Use 2022 as a stable year for names
    for abbr, config in STATE_CONFIG.items():
        fips = config["fips"]
        url = f"https://api.census.gov/data/2022/acs/acs5"
        params = {"get": "NAME", "for": "county:*", "in": f"state:{fips}", "key": API_KEY}
        r = requests.get(url, params=params)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            rows.append(df)
    
    if not rows:
        return pd.DataFrame()
    
    names_df = pd.concat(rows, ignore_index=True)
    # "Adams County, Indiana" -> "Adams"
    names_df["county_name"] = names_df["NAME"].str.split(",").str[0].str.replace(" County", "")
    names_df["fips"] = names_df["state"].str.zfill(2) + names_df["county"].str.zfill(3)
    return names_df[["fips", "county_name", "state"]]


# -----------------------
# Missing logic
# -----------------------
def get_missing(existing):
    if existing.empty:
        return [(s, y) for s in STATES for y in YEARS]

    done = set(zip(existing["state_abbr"], existing["year"]))
    return [(s, y) for s in STATES for y in YEARS if (s, y) not in done]


# -----------------------
# MAIN
# -----------------------
def main():
    existing = load_existing()
    todo = get_missing(existing)

    print(f"Fetching {len(todo)} combinations")

    all_new = []
    
    # Get names once to apply to new data
    names = fetch_county_names()

    for state, year in todo:
        print(f"Fetching {state}-{year}")

        df = load_lodes_od(state, year)

        if df is None or df.empty:
            continue

        out = lodes_to_county(df, state, year)
        
        # Merge names and state info
        if not names.empty:
            out = pd.merge(out, names, on="fips", how="left")
            out["state_name"] = out["state"].map(STATE_FIPS_TO_NAME)
            out["full_name"] = out["county_name"] + ", " + out["state_name"]
            
        all_new.append(out)

    if not all_new:
        print("No new data.")
        return

    new_df = pd.concat(all_new, ignore_index=True)

    final = (
        pd.concat([existing, new_df], ignore_index=True)
        if not existing.empty else new_df
    )

    final = final.drop_duplicates(subset=["fips", "state_abbr", "year"])
    final = final.sort_values(["state_abbr", "year", "fips"])

    final.to_parquet(OUTPUT_FILE, index=False)

    print("LODES parquet updated with names and state info.")


if __name__ == "__main__":
    main()