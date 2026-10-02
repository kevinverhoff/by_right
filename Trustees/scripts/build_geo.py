"""Build a simplified township GeoJSON keyed to the CSV's county/township names.

Downloads the Census cartographic boundary file for Indiana county subdivisions,
simplifies each polygon, and writes data/indiana_townships.geojson. Run once --
the app reads the output and never touches the network.

    .venv/bin/pip install pyshp
    .venv/bin/python scripts/build_geo.py
"""
import io
import json
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import shapefile

URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_18_cousub_500k.zip"
STEM = "cb_2023_18_cousub_500k"
ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "indiana_townships_2025.csv"
OUT = ROOT / "data" / "indiana_townships.geojson"

# CSV township name -> the Census NAMELSAD pieces it covers. The CSV follows the
# SBOA's naming; the Census follows its own, and Johnson County reports three
# legally merged townships as one unit.
ALIASES = {
    "hamilton|washington": ["westfieldwashington"],
    "wayne|greene": ["green"],
    "johnson|franklinunionneedham": ["franklin", "union", "needham"],
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def key(county, township):
    c = norm(re.sub(r"(?i)\s+county$", "", str(county).strip()))
    t = norm(re.sub(r"(?i)\s+township$", "", str(township).strip()))
    return f"{c}|{t}"


def simplify(ring, tol):
    """Radial-distance thinning: drop points within tol degrees of the last kept one."""
    if len(ring) < 5:
        return ring
    out = [ring[0]]
    for x, y in ring[1:-1]:
        px, py = out[-1]
        if abs(x - px) > tol or abs(y - py) > tol:
            out.append((x, y))
    out.append(ring[0])
    return out if len(out) >= 4 else ring


def rings_of(shp, tol=0.004):
    pts = [tuple(p) for p in shp.points]
    parts = list(shp.parts) + [len(pts)]
    out = [simplify(pts[parts[i]:parts[i + 1]], tol) for i in range(len(parts) - 1)]
    return [[list(p) for p in r] for r in out if len(r) >= 4]


tmp = tempfile.mkdtemp()
print(f"downloading {URL}")
with urllib.request.urlopen(URL, timeout=120) as r:
    zipfile.ZipFile(io.BytesIO(r.read())).extractall(tmp)
shp_stem = Path(tmp) / STEM

df = pd.read_csv(CSV)

# Expand each wanted key into the set of census keys that make it up.
wanted = {}
for r in df.itertuples():
    k = key(r.county, r.township)
    county = k.split("|")[0]
    pieces = ALIASES.get(k, [k.split("|")[1]])
    for p in pieces:
        wanted.setdefault(f"{county}|{p}", []).append(k)

reader = shapefile.Reader(str(shp_stem))
fields = [f[0] for f in reader.fields[1:]]
collected, matched = {}, set()

for rec, shp in zip(reader.records(), reader.shapes()):
    d = dict(zip(fields, rec))
    ck = key(d["NAMELSADCO"], d["NAMELSAD"])
    for target in wanted.get(ck, []):
        matched.add(ck)
        collected.setdefault(target, []).extend(rings_of(shp))

features = []
for k, rings in collected.items():
    geom = ({"type": "Polygon", "coordinates": [rings[0]]} if len(rings) == 1
            else {"type": "MultiPolygon", "coordinates": [[r] for r in rings]})
    features.append({"type": "Feature", "id": k, "properties": {"key": k}, "geometry": geom})

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, separators=(",", ":"))

csv_keys = {key(r.county, r.township) for r in df.itertuples()}
print(f"wrote {OUT}")
print(f"features: {len(features)}   csv rows: {len(df)}   distinct csv keys: {len(csv_keys)}")
print("csv keys with no geometry:", sorted(csv_keys - set(collected)))
print("census subdivisions not used:", len(set(wanted) - matched), sorted(set(wanted) - matched)[:10])
