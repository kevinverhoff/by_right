import pandas as pd

df = pd.read_parquet("lodes_commuting.parquet")

cols = ["in_commuters", "out_commuters", "net_commute"]

available = [c for c in cols if c in df.columns]

print(df[available].head(20))
print("\nColumns in dataset:")
print(df.columns.tolist())