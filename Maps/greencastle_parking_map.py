"""Build an interactive map of downtown Greencastle, IN public parking lots.

Lot facts (names, space counts, accessible spaces, time restrictions) are
transcribed from the City of Greencastle's Downtown Parking page:
https://www.cityofgreencastle.com/188/Downtown-Parking

That page publishes no coordinates, so each lot is located by querying
OpenStreetMap via the Overpass API: we compute the intersection of the two
streets the city uses to describe the lot, then claim the nearest
``amenity=parking`` polygon within a match radius. Lots that OSM does not
map fall back to a hand-placed coordinate recorded in ``LOT_SPECS``.

On-street parking comes from OSM alone -- every ``parking=street_side`` bay
downtown, with its surveyed ``capacity``. The city publishes no counts for
it, so a bay tagged without a capacity contributes zero and is reported.

Output is a single self-contained HTML file. Clicking anywhere on the map
names the nearest lot, draws the walking route to it, and reports how many
spaces -- lot and on-street, counted separately -- lie within a 1, 2, and
5 minute walk. Distances come from a pedestrian router; walk time is
derived from ``WALK_SPEED_MPS``.

Usage:
    python greencastle_parking_map.py
    python greencastle_parking_map.py --refresh     # re-query OpenStreetMap
    python greencastle_parking_map.py --open        # open the map when done
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

# --- Configuration ----------------------------------------------------------

# Downtown Greencastle, generous enough to include every lot plus context.
BBOX = (39.6390, -86.8720, 39.6500, -86.8570)  # south, west, north, east

# Ordered by observed freshness. The main instance is the only one that has
# reliably been current for Putnam County -- the mirrors have been seen
# serving snapshots months behind, which silently hides recent edits. Hence
# MAX_DATA_AGE_DAYS below: a stale mirror is worse than a failed request,
# because it returns a plausible answer that is quietly wrong.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
MAX_DATA_AGE_DAYS = 7
USER_AGENT = "greencastle-parking-map/1.0 (https://www.cityofgreencastle.com)"

# FOSSGIS's routed-foot instance is a genuine pedestrian OSRM build: it
# ignores one-way restrictions and uses footpaths, and its durations imply a
# real walking speed (~1.25 m/s). Do NOT substitute router.project-osrm.org --
# that demo server advertises a /foot/ profile but silently ignores it and
# returns car routing (~7.6 m/s implied), which inflates downtown distances by
# up to 2.4x because it detours around the one-way courthouse square.
ROUTER_BASE = "https://routing.openstreetmap.de/routed-foot"
ROUTER_PROFILE = "foot"

# Walk time is derived from routed distance at this fixed pace rather than
# taken from the router, so that "a 2-minute walk" stays a defined, tunable
# quantity that does not shift if the routing service changes its cost model.
WALK_SPEED_MPS = 1.4  # ~3.1 mph, a normal adult walking pace

# Walk radii reported for every click, in minutes.
WALK_THRESHOLDS_MIN = (1, 2, 5)

# Reference walks at Walmart Supercenter #902, 1750 Indianapolis Road,
# measured from OSM geometry: the store (way/669146738), its ~893-space lot
# (way/419654159, 26,779 m2) and its two main entrance nodes.
#
# These are straight-line across the lot. The lot has no mapped aisles -- OSM
# carries zero highway ways inside it -- so routing there is not possible;
# asking the pedestrian router returns 75 m for a 79 m straight line, which
# means it snapped both ends onto a distant road. A shopper crosses a lot
# roughly directly anyway, so straight line is both the only option and a
# generous one to Walmart. Downtown distances use the real pedestrian
# network, so every comparison below understates downtown's advantage.
WALMART_BENCHMARKS = (
    ("Mid-lot to the door", 258),
    ("Worst space to the door", 547),
    ("Mid-lot to the back of the store", 661),
)
FEET_PER_METER = 3.28084

# How far from a street intersection we will still accept a parking polygon.
MATCH_RADIUS_M = 120.0

# ``parking=*`` values that mean on-street. These are kerbside bays, not the
# city's off-street lots, and they sit right at the intersections we anchor
# on -- so without this filter a bay will out-compete the lot it is next to.
ON_STREET_PARKING = frozenset({"street_side", "lane", "on_street"})

# A surface lot needs roughly 30-40 m2 per space once aisles are counted.
# Anything far below that is not the lot we are looking for, whatever its
# tags say -- a backstop against future mis-matches.
MIN_M2_PER_SPACE = 15.0

MAP_TITLE = "Downtown Greencastle Public Parking"


@dataclass(frozen=True)
class LotSpec:
    """A parking lot as described on the city's website.

    ``street``/``cross_street`` are OSM street names whose intersection anchors
    the lot. ``side`` narrows the search to one side of that intersection
    ("N" or "S"), which is what separates the two Jackson Street lots.
    ``fallback`` is a hand-placed (lat, lon) used only when OSM maps no
    parking polygon nearby.
    """

    name: str
    spaces: int
    accessible_spaces: int
    restrictions: str
    hours: str
    location_text: str
    street: str
    cross_street: str
    side: str | None = None
    notes: str = ""
    fallback: tuple[float, float] | None = None


# Transcribed from the city's Downtown Parking page.
LOT_SPECS: tuple[LotSpec, ...] = (
    LotSpec(
        name="Vine Street Lot",
        spaces=28,
        accessible_spaces=3,
        restrictions="2 hours, 8am-6pm Mon-Fri",
        hours="Open 24/7",
        location_text="South Vine Street, just south of Washington Street",
        street="South Vine Street",
        cross_street="East Walnut Street",
        side="N",
        notes="Accessible spaces: 1 northeast, 2 southwest.",
    ),
    LotSpec(
        name="Columbia Street Lot",
        spaces=40,
        accessible_spaces=2,
        restrictions="No time limit",
        hours="Open 24/7",
        location_text="North Indiana Street, just north of the square",
        street="North Indiana Street",
        cross_street="East Columbia Street",
        notes="Accessible spaces at the west end.",
    ),
    LotSpec(
        name="Jackson Street Lot",
        spaces=50,
        accessible_spaces=3,
        restrictions="No time limit",
        hours="Open 24/7",
        location_text="South Jackson Street, just south of Walnut Street",
        street="South Jackson Street",
        cross_street="West Walnut Street",
        side="S",
        notes="Some spaces reserved for Crown Equipment after hours and weekends.",
    ),
    LotSpec(
        name="Market Street Lot",
        spaces=37,
        accessible_spaces=2,
        restrictions="No time limit",
        hours="Open 24/7",
        location_text="Entrances on Washington Street (one block west) and Franklin Street",
        street="North Market Street",
        cross_street="West Washington Street",
        side="N",
        notes="37 lot spaces plus 8 on-street. Accessible spaces at the northeast corner.",
    ),
    LotSpec(
        name="North Jackson Street Lot",
        spaces=30,
        accessible_spaces=1,
        restrictions="No time limit",
        hours="Open 24/7",
        location_text="North Jackson Street, just north of the square",
        street="North Jackson Street",
        cross_street="West Franklin Street",
        side="N",
        notes="3 spaces reserved for county judges. Accessible space on the south side.",
    ),
    LotSpec(
        name="City Hall Lot",
        spaces=19,
        accessible_spaces=2,
        restrictions="City Hall visitors; open to all after hours and weekends",
        hours="Open 24/7",
        location_text="North Locust Street, just north of Washington Street",
        street="North Locust Street",
        cross_street="East Washington Street",
        side="N",
        notes="Accessible spaces at the northwest corner.",
        # OSM now maps this lot (way/1555164097) and the match succeeds, so
        # this coordinate is unused. Kept only as a safety net for the one
        # lot that was missing from OSM the longest.
        fallback=(39.64447, -86.86063),
    ),
)


# --- Geometry helpers -------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in meters between two (lat, lon) points."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    d_lat = lat2 - lat1
    d_lon = math.radians(b[1] - a[1])
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def way_points(way: dict) -> list[tuple[float, float]]:
    return [(p["lat"], p["lon"]) for p in way["geometry"]]


def polygon_centroid(way: dict) -> tuple[float, float]:
    pts = way_points(way)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def polygon_area_m2(way: dict) -> float:
    """Shoelace area, using a local equirectangular projection.

    Accurate to well under a percent at the scale of a city block, which is
    all we need -- the area is a sanity check on the name-to-polygon match,
    not a survey measurement.
    """
    pts = way_points(way)
    lat0 = math.radians(sum(p[0] for p in pts) / len(pts))
    xy = [
        (math.radians(lon) * EARTH_RADIUS_M * math.cos(lat0), math.radians(lat) * EARTH_RADIUS_M)
        for lat, lon in pts
    ]
    total = sum(
        xy[i][0] * xy[(i + 1) % len(xy)][1] - xy[(i + 1) % len(xy)][0] * xy[i][1]
        for i in range(len(xy))
    )
    return abs(total) / 2


# --- OpenStreetMap ----------------------------------------------------------

def build_overpass_query() -> str:
    south, west, north, east = BBOX
    bbox = f"{south},{west},{north},{east}"
    return f"""[out:json][timeout:120];
(
  way["highway"]["name"]({bbox});
  way["amenity"="parking"]({bbox});
);
out body geom;"""


def fetch_osm(cache_path: Path, refresh: bool) -> dict:
    """Return the Overpass response, reading the on-disk cache when possible."""
    if cache_path.exists() and not refresh:
        print(f"Using cached OpenStreetMap data: {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    query = build_overpass_query()
    stale: list[tuple[float, str, str]] = []  # (age_days, mirror, raw body)

    for mirror in OVERPASS_MIRRORS:
        print(f"Querying Overpass: {mirror}")
        body = post_overpass(mirror, query)
        if body is None:
            continue

        age = data_age_days(body)
        if age is None:
            print("  -> response carried no data timestamp, skipping")
            continue
        if age > MAX_DATA_AGE_DAYS:
            print(f"  -> data is {age:.0f} days old, trying a fresher mirror")
            stale.append((age, mirror, body))
            continue

        print(f"  -> data is current ({age * 24:.1f} hours old)")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(body, encoding="utf-8")
        return json.loads(body)

    if stale:
        age, mirror, body = min(stale)
        print(f"\nWARNING: every mirror is behind. Using {mirror}, whose data is "
              f"{age:.0f} days old.\n         Edits made to OpenStreetMap since "
              f"then will NOT appear on this map.\n")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(body, encoding="utf-8")
        return json.loads(body)

    raise SystemExit(
        "Every Overpass mirror failed. Retry in a few minutes (the main "
        "instance rate-limits repeated queries), or run without --refresh "
        f"if {cache_path.name} already exists."
    )


def post_overpass(mirror: str, query: str) -> str | None:
    """POST a query, retrying once past a rate-limit. Returns the raw body.

    Overpass answers rate limits and outages with an HTML error page rather
    than JSON, so the content type is checked before the body is trusted.
    """
    for attempt in (1, 2):
        # A dropped connection or DNS failure must fall through to the next
        # mirror, not abort the run -- this is a third-party network boundary,
        # so the exception is caught here rather than left to propagate.
        try:
            response = requests.post(
                mirror, data={"data": query}, timeout=240,
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException as error:
            print(f"  -> {type(error).__name__}: {error}")
            return None

        if response.status_code == 429 and attempt == 1:
            print("  -> rate limited, waiting 10s")
            time.sleep(10)
            continue
        if response.status_code != 200:
            print(f"  -> HTTP {response.status_code}")
            return None
        if "json" not in response.headers.get("content-type", "").lower():
            print("  -> non-JSON response (likely an error page)")
            return None
        return response.text
    return None


def data_age_days(body: str) -> float | None:
    """How far behind live OSM this Overpass response is, in days."""
    payload = json.loads(body)
    stamp = payload.get("osm3s", {}).get("timestamp_osm_base")
    if not stamp:
        return None
    base = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - base).total_seconds() / 86400


def index_osm(payload: dict) -> tuple[dict[str, list[dict]], list[dict]]:
    """Split the Overpass elements into named streets and parking polygons."""
    streets: dict[str, list[dict]] = {}
    parking: list[dict] = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        if "geometry" not in element:
            continue
        if tags.get("amenity") == "parking":
            parking.append(element)
        elif "highway" in tags and "name" in tags:
            streets.setdefault(tags["name"], []).append(element)
    return streets, parking


def street_intersection(
    streets: dict[str, list[dict]], name_a: str, name_b: str
) -> tuple[float, float] | None:
    """Find where two named streets cross, via a shared OSM node.

    Streets are usually split into several ways, so every way of one street is
    checked against every way of the other.
    """
    for way_a in streets.get(name_a, []):
        node_coords = dict(zip(way_a["nodes"], way_a["geometry"]))
        for way_b in streets.get(name_b, []):
            shared = set(node_coords) & set(way_b["nodes"])
            if shared:
                node = node_coords[next(iter(shared))]
                return (node["lat"], node["lon"])
    return None


def match_parking_polygon(
    spec: LotSpec, anchor: tuple[float, float], parking: list[dict]
) -> dict | None:
    """Nearest plausible off-street polygon to ``anchor``.

    Honors the lot's side hint, ignores on-street bays, and rejects any
    polygon too small to hold the lot's published space count.
    """
    candidates = []
    for way in parking:
        if way.get("tags", {}).get("parking") in ON_STREET_PARKING:
            continue
        centroid = polygon_centroid(way)
        if spec.side == "N" and centroid[0] < anchor[0]:
            continue
        if spec.side == "S" and centroid[0] > anchor[0]:
            continue
        distance = haversine_m(anchor, centroid)
        if distance <= MATCH_RADIUS_M:
            candidates.append((distance, way))

    candidates.sort(key=lambda pair: pair[0])
    for distance, way in candidates:
        density = polygon_area_m2(way) / spec.spaces
        if density >= MIN_M2_PER_SPACE:
            return way
        print(f"    (rejected osm:way/{way['id']}: {density:.1f} m2 per stated "
              f"space is too dense for a {spec.spaces}-space lot)")
    return None


def resolve_lots(streets: dict[str, list[dict]], parking: list[dict]) -> list[dict]:
    """Turn each LotSpec into a located, JSON-serializable lot record."""
    claimed: set[int] = set()
    resolved: list[dict] = []

    for spec in LOT_SPECS:
        record = {
            "name": spec.name,
            "spaces": spec.spaces,
            "accessible_spaces": spec.accessible_spaces,
            "restrictions": spec.restrictions,
            "hours": spec.hours,
            "location_text": spec.location_text,
            "notes": spec.notes,
            "time_limited": "hour" in spec.restrictions.lower(),
        }

        anchor = street_intersection(streets, spec.street, spec.cross_street)
        way = None
        if anchor is None:
            print(f"  {spec.name}: no OSM intersection for "
                  f"{spec.street} x {spec.cross_street}")
        else:
            way = match_parking_polygon(spec, anchor, parking)
            if way is not None and way["id"] in claimed:
                print(f"  {spec.name}: nearest polygon already claimed, skipping")
                way = None

        if way is not None:
            claimed.add(way["id"])
            centroid = polygon_centroid(way)
            area = polygon_area_m2(way)
            record.update(
                lat=round(centroid[0], 6),
                lon=round(centroid[1], 6),
                polygon=[[round(lat, 6), round(lon, 6)] for lat, lon in way_points(way)],
                source=f"osm:way/{way['id']}",
                approximate=False,
                area_m2=round(area),
                m2_per_space=round(area / spec.spaces, 1) if spec.spaces else None,
            )
            print(f"  {spec.name}: matched osm:way/{way['id']} "
                  f"({area:.0f} m2, {area / spec.spaces:.1f} m2/space)")
        elif spec.fallback is not None:
            record.update(
                lat=spec.fallback[0],
                lon=spec.fallback[1],
                polygon=None,
                source="manual",
                approximate=True,
                area_m2=None,
                m2_per_space=None,
            )
            print(f"  {spec.name}: not in OSM, using hand-placed coordinate")
        else:
            print(f"  {spec.name}: UNLOCATED -- omitted from the map")
            continue

        resolved.append(record)

    return resolved


def nearest_street_name(
    streets: dict[str, list[dict]], point: tuple[float, float]
) -> str:
    """Name of the street whose centerline passes closest to ``point``."""
    best_name = ""
    best_distance = float("inf")
    for name, ways in streets.items():
        for way in ways:
            for vertex in way_points(way):
                distance = haversine_m(point, vertex)
                if distance < best_distance:
                    best_distance = distance
                    best_name = name
    return best_name


def resolve_street_bays(
    streets: dict[str, list[dict]], parking: list[dict]
) -> list[dict]:
    """Collect the on-street parking bays surveyed in OpenStreetMap.

    Unlike the lots -- whose facts come from the city's website and whose
    only OSM contribution is a footprint -- a bay is described entirely by
    its OSM tags. A bay with no ``capacity`` counts as zero spaces and is
    flagged, so an untagged bay never silently inflates or deflates a total.
    """
    bays: list[dict] = []
    for way in parking:
        tags = way.get("tags", {})
        if tags.get("parking") not in ON_STREET_PARKING:
            continue

        centroid = polygon_centroid(way)
        capacity = tags.get("capacity", "")
        accessible = tags.get("capacity:disabled", "")
        bays.append({
            "source": f"osm:way/{way['id']}",
            "lat": round(centroid[0], 6),
            "lon": round(centroid[1], 6),
            "polygon": [[round(lat, 6), round(lon, 6)] for lat, lon in way_points(way)],
            "spaces": int(capacity) if capacity.isdigit() else 0,
            "capacity_known": capacity.isdigit(),
            "accessible_spaces": int(accessible) if accessible.isdigit() else 0,
            "orientation": tags.get("orientation", ""),
            "maxstay": tags.get("maxstay", ""),
            "street": nearest_street_name(streets, centroid),
        })

    bays.sort(key=lambda bay: (bay["street"], -bay["spaces"]))
    untagged = [b["source"] for b in bays if not b["capacity_known"]]
    print(f"Street bays: {len(bays)} covering "
          f"{sum(b['spaces'] for b in bays)} spaces")
    if untagged:
        print(f"  {len(untagged)} bay(s) have no capacity tag and count as 0: "
              f"{', '.join(untagged)}")
    return bays


# --- HTML rendering ---------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha384-sHL9NAb7lN7rfvG5lfHpm643Xkcjzp4jFvuavGOndn6pjVqS6ny56CAt3nsEVT4H"
      crossorigin="anonymous">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha384-cxOPjt7s7Iz04uaHJceBmS+qpjv2JkIHNVcuOrM+YHwZOmJGBXI00mdUXEq65HTH"
        crossorigin="anonymous"></script>
<style>
  :root {
    --ink: #17202a;
    --muted: #5b6770;
    --line: #d8dee3;
    --panel: #ffffff;
    /* One stack for the whole map -- panel, popups, and the numbered map
       badges alike -- so nothing renders in a mismatched face. */
    --font: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
            Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    font: 14px/1.5 var(--font);
    color: var(--ink);
    display: flex;
    -webkit-font-smoothing: antialiased;
  }
  #map { flex: 1 1 auto; height: 100vh; }
  #panel {
    flex: 0 0 340px;
    height: 100vh;
    overflow-y: auto;
    background: var(--panel);
    border-left: 1px solid var(--line);
    padding: 18px 18px 28px;
  }
  h1 { font-size: 17px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 12.5px; margin: 0 0 16px; }
  .hint {
    background: #f1f6fd; border: 1px solid #cfe0f7; border-radius: 8px;
    padding: 11px 13px; font-size: 13px; color: #1c3d63;
  }
  .status { font-size: 12px; color: var(--muted); margin: 10px 0 4px; min-height: 16px; }
  .stat {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 12px 14px; margin: 12px 0 4px; background: #f8fafc;
  }
  .stat .big {
    font-size: 26px; font-weight: 700; letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums; line-height: 1.1;
  }
  .stat .cap { font-size: 12.5px; color: var(--muted); margin-top: 3px; }
  .stat .note { font-size: 11.5px; color: #94a3b8; margin-top: 7px; }
  .radii { margin: 12px 0 4px; }
  .radii h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
    color: var(--muted); margin: 0 0 7px; font-weight: 700;
  }
  .radius {
    display: grid; grid-template-columns: 54px 1fr auto; align-items: baseline;
    gap: 10px; padding: 8px 11px; border: 1px solid var(--line);
    border-radius: 8px; margin-bottom: 6px;
  }
  .radius .when { font-size: 12px; font-weight: 600; color: var(--muted); }
  .radius .total {
    font-size: 19px; font-weight: 700; font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
  }
  .radius .split { font-size: 11.5px; color: var(--muted); text-align: right; }
  .radius.empty .total { color: #9aa5b1; font-weight: 600; font-size: 15px; }
  .bench { margin: 16px 0 4px; }
  .bench h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
    color: var(--muted); margin: 0 0 3px; font-weight: 700;
  }
  .bench .lede { font-size: 11.5px; color: var(--muted); margin: 0 0 8px; }
  .bench .row {
    display: grid; grid-template-columns: 1fr auto; gap: 8px;
    align-items: baseline; padding: 7px 0;
    border-bottom: 1px dashed var(--line);
  }
  .bench .row:last-of-type { border-bottom: none; }
  .bench .what { font-size: 12px; }
  .bench .ft { color: var(--muted); font-variant-numeric: tabular-nums; }
  .bench .beat {
    font-size: 16px; font-weight: 700; font-variant-numeric: tabular-nums;
    color: #0f766e;
  }
  .bench .beat span { font-size: 11px; font-weight: 600; color: var(--muted); }
  ol.results { list-style: none; margin: 4px 0 0; padding: 0; }
  ol.results li {
    border: 1px solid var(--line); border-radius: 8px;
    padding: 10px 12px; margin-bottom: 8px; cursor: pointer;
    display: grid; grid-template-columns: 26px 1fr; gap: 10px;
    transition: border-color .12s, background .12s;
  }
  ol.results li:hover { border-color: #9db6cf; background: #fafcff; }
  ol.results li.nearest { border-color: #1f6feb; background: #f4f9ff; }
  ol.results li.out { opacity: .48; }
  .rank {
    width: 24px; height: 24px; border-radius: 50%;
    background: #eef2f6; color: var(--muted);
    font-size: 12px; font-weight: 700;
    display: grid; place-items: center;
    font-variant-numeric: tabular-nums;
  }
  li.nearest .rank { background: #1f6feb; color: #fff; }
  .lot-name { font-weight: 600; }
  /* Tabular figures keep the distance column from jittering as the router
     result replaces the straight-line estimate. */
  .dist { font-size: 13px; margin-top: 1px; font-variant-numeric: tabular-nums; }
  /* The numbered badges pinned on the map. */
  .lot-badge {
    width: 22px; height: 22px; border-radius: 50%;
    display: grid; place-items: center;
    font: 700 11px/1 var(--font); color: #fff;
    border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0, 0, 0, .4);
  }
  .dist .walk { color: var(--muted); }
  .meta { font-size: 12px; color: var(--muted); margin-top: 3px; }
  .tag {
    display: inline-block; font-size: 11px; padding: 1px 7px;
    border-radius: 999px; margin-top: 5px; font-weight: 600;
  }
  .tag.limited { background: #fef3c7; color: #92400e; }
  .tag.open { background: #dbeafe; color: #1e40af; }
  .tag.approx { background: #f3f4f6; color: #4b5563; }
  .legend {
    margin-top: 20px; padding-top: 14px; border-top: 1px solid var(--line);
    font-size: 12px; color: var(--muted);
  }
  .legend div { margin-bottom: 5px; }
  .swatch {
    display: inline-block; width: 11px; height: 11px; border-radius: 2px;
    margin-right: 7px; vertical-align: -1px;
  }
  .credit { margin-top: 14px; font-size: 11px; color: #94a3b8; }
  .credit a { color: #94a3b8; }
  .leaflet-popup-content {
    font: 13px/1.5 var(--font); margin: 12px 14px;
  }
  .leaflet-popup-content b { font-size: 14px; }
  @media (max-width: 780px) {
    body { flex-direction: column; }
    #map { height: 58vh; flex: none; }
    #panel { flex: none; height: auto; border-left: none; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<div id="map"></div>
<aside id="panel">
  <h1>__TITLE__</h1>
  <p class="sub">__SUBTITLE__</p>
  <div id="results">
    <div class="hint">Click anywhere on the map to see how far each public lot
      is. Drag the pin to move it.</div>
  </div>
  <div class="legend">
    <div><span class="swatch" style="background:#1f6feb"></span>Lot, no time limit</div>
    <div><span class="swatch" style="background:#b45309"></span>Lot, time-limited</div>
    <div><span class="swatch" style="background:#0f766e"></span>On-street bay</div>
    <div><span class="swatch" style="background:#6b7280"></span>Approximate location</div>
  </div>
  <p class="credit">
    Lot details from the
    <a href="https://www.cityofgreencastle.com/188/Downtown-Parking" target="_blank"
       rel="noopener">City of Greencastle</a>.
    Lot outlines and routing &copy; OpenStreetMap contributors.
  </p>
</aside>

<script>
const LOTS = __LOTS_JSON__;
const BAYS = __BAYS_JSON__;
const THRESHOLDS = __THRESHOLDS__;
const BENCHMARKS = __BENCHMARKS__;
const FEET_PER_METER = __FEET_PER_METER__;
const TOTAL_SPACES = __TOTAL_SPACES__;
const ROUTER = "__ROUTER_BASE__";
const PROFILE = "__ROUTER_PROFILE__";
const WALK_SPEED = __WALK_SPEED__;
const CENTER = __CENTER__;

const map = L.map("map").setView(CENTER, 16);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Lot text is authored in the build script, but every bay field comes from an
// OpenStreetMap tag, which anyone can edit. Escape before it reaches innerHTML.
function esc(value) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(value == null ? "" : value).replace(/[&<>"']/g, function (c) {
    return map[c];
  });
}

function lotColor(lot) {
  if (lot.approximate) return "#6b7280";
  return lot.time_limited ? "#b45309" : "#1f6feb";
}

function lotPopup(lot) {
  const acc = lot.accessible_spaces
    ? " &middot; " + lot.accessible_spaces + " accessible" : "";
  const approx = lot.approximate
    ? '<div style="color:#6b7280;margin-top:6px">Approximate location &mdash; this lot'
      + ' is not mapped in OpenStreetMap.</div>' : "";
  const notes = lot.notes
    ? '<div style="color:#5b6770;margin-top:6px">' + esc(lot.notes) + "</div>" : "";
  return "<b>" + esc(lot.name) + "</b><br>"
    + '<span style="color:#5b6770">' + esc(lot.location_text) + "</span><br><br>"
    + "<b>" + lot.spaces + "</b> spaces" + acc + "<br>"
    + esc(lot.restrictions) + "<br>" + esc(lot.hours) + notes + approx;
}

function bayPopup(bay) {
  const acc = bay.accessible_spaces
    ? " &middot; " + bay.accessible_spaces + " accessible" : "";
  const orient = bay.orientation ? esc(bay.orientation) + " parking<br>" : "";
  const unknown = bay.capacity_known ? ""
    : '<div style="color:#b45309;margin-top:6px">No capacity recorded in '
      + "OpenStreetMap &mdash; counted as zero.</div>";
  return "<b>On-street parking</b><br>"
    + '<span style="color:#5b6770">' + esc(bay.street) + "</span><br><br>"
    + "<b>" + bay.spaces + "</b> spaces" + acc + "<br>" + orient
    + (bay.maxstay ? "Max stay: " + esc(bay.maxstay) : "No time limit recorded")
    + unknown;
}

// Lots: real footprint where OSM has one, a dashed circle where it does not.
LOTS.forEach(function (lot, i) {
  const color = lotColor(lot);
  const style = { color: color, weight: 2, fillColor: color, fillOpacity: 0.28 };
  lot.shape = lot.polygon
    ? L.polygon(lot.polygon, style)
    : L.circle([lot.lat, lot.lon],
        Object.assign({}, style, { radius: 26, dashArray: "4 4" }));
  lot.shape.addTo(map).bindPopup(lotPopup(lot));

  lot.marker = L.marker([lot.lat, lot.lon], {
    icon: L.divIcon({
      className: "",
      html: '<div class="lot-badge" style="background:' + color + '">'
        + (i + 1) + "</div>",
      iconSize: [22, 22], iconAnchor: [11, 11]
    })
  }).addTo(map).bindPopup(lotPopup(lot));
});

// On-street bays: drawn thinner so they read as kerbside strips, not lots.
BAYS.forEach(function (bay) {
  bay.shape = L.polygon(bay.polygon, {
    color: "#0f766e", weight: 1.5, fillColor: "#0f766e", fillOpacity: 0.35
  }).addTo(map).bindPopup(bayPopup(bay));
});

map.fitBounds(L.featureGroup(LOTS.map(function (l) { return l.shape; }))
  .getBounds().pad(0.18));

// --- Click to measure ------------------------------------------------------

function haversine(a, b) {
  const R = 6371000;
  const p1 = a[0] * Math.PI / 180, p2 = b[0] * Math.PI / 180;
  const dp = p2 - p1, dl = (b[1] - a[1]) * Math.PI / 180;
  const h = Math.sin(dp / 2) * Math.sin(dp / 2)
    + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
  return 2 * R * Math.asin(Math.sqrt(h));
}

function walkMinutes(meters) { return meters / WALK_SPEED / 60; }

// Round UP, so the time shown and the radius a place falls into can never
// disagree: ceil(x) <= N is true exactly when x <= N.
function formatWalk(meters) {
  const m = Math.ceil(walkMinutes(meters));
  return m <= 1 ? "1 min walk" : m + " min walk";
}

function formatDistance(meters) {
  const feet = meters * 3.28084;
  if (feet < 1000) return Math.round(feet / 10) * 10 + " ft";
  return (meters / 1609.344).toFixed(2) + " mi";
}

const resultsEl = document.getElementById("results");
let youMarker = null;
let routeLine = null;
// Bumped on every click/drag so a slow router response for an old origin
// cannot overwrite the results for a newer one.
let requestSeq = 0;

// Every destination we measure. Lots come first so a table response maps back
// cleanly: indices 0..LOTS.length-1 are lots, everything after is a bay.
const TARGETS = LOTS.concat(BAYS);

function nearestLotIndex(dists) {
  let best = -1;
  for (let i = 0; i < LOTS.length; i++) {
    if (best < 0 || dists[i] < dists[best]) best = i;
  }
  return best;
}

function radiiHtml(dists) {
  const rows = THRESHOLDS.map(function (t) {
    let lotSpaces = 0, baySpaces = 0;
    TARGETS.forEach(function (place, i) {
      if (walkMinutes(dists[i]) > t) return;
      if (i < LOTS.length) lotSpaces += place.spaces;
      else baySpaces += place.spaces;
    });
    const total = lotSpaces + baySpaces;
    const split = total
      ? lotSpaces + " lot &middot; " + baySpaces + " street" : "&nbsp;";
    return '<div class="radius' + (total ? "" : " empty") + '">'
      + '<span class="when">' + t + " min</span>"
      + '<span class="total">' + (total ? total + " spaces" : "none") + "</span>"
      + '<span class="split">' + split + "</span></div>";
  }).join("");
  return '<div class="radii"><h2>Spaces within a walk of</h2>' + rows + "</div>";
}

// How much downtown parking beats a walk you would not think twice about at
// a big-box store. Walmart figures are straight-line across its lot; these
// downtown ones follow the pedestrian network, so the gap is understated.
function benchmarkHtml(dists) {
  const rows = BENCHMARKS.map(function (b) {
    const meters = b.feet / FEET_PER_METER;
    let spaces = 0;
    TARGETS.forEach(function (place, i) {
      if (dists[i] <= meters) spaces += place.spaces;
    });
    const pct = TOTAL_SPACES ? Math.round(spaces / TOTAL_SPACES * 100) : 0;
    return '<div class="row"><div><div class="what">' + esc(b.label) + "</div>"
      + '<div class="ft">' + b.feet + " ft</div></div>"
      + '<div class="beat">' + spaces
      + ' <span>spaces &middot; ' + pct + "%</span></div></div>";
  }).join("");
  return '<div class="bench"><h2>Beats the walk at Walmart</h2>'
    + '<p class="lede">Downtown spaces closer than these walks at '
    + "Walmart Supercenter #902.</p>" + rows + "</div>";
}

function nearestLotHtml(dists) {
  const lot = LOTS[nearestLotIndex(dists)];
  const d = dists[nearestLotIndex(dists)];
  const acc = lot.accessible_spaces
    ? " &middot; " + lot.accessible_spaces + " accessible" : "";
  return '<div class="stat"><div class="cap">Nearest lot</div>'
    + '<div class="big" style="font-size:19px">' + esc(lot.name) + "</div>"
    + '<div class="cap">' + formatDistance(d) + " &middot; " + formatWalk(d)
    + " &middot; " + lot.spaces + " spaces" + acc + "</div></div>";
}

let lastDists = null;
let lastRouted = false;
let lastStatus = "";

function render(dists, statusText, routed) {
  lastDists = dists;
  lastStatus = statusText;
  lastRouted = !!routed;
  paint();
}

function paint() {
  if (!lastDists) return;
  const dists = lastDists;
  const widest = THRESHOLDS[THRESHOLDS.length - 1];

  const order = LOTS
    .map(function (lot, i) { return { lot: lot, meters: dists[i] }; })
    .sort(function (a, b) { return a.meters - b.meters; });

  const items = order.map(function (row, i) {
    const lot = row.lot;
    const within = walkMinutes(row.meters) <= widest;
    const acc = lot.accessible_spaces
      ? " &middot; " + lot.accessible_spaces + " accessible" : "";
    let tag;
    if (lot.approximate) {
      tag = '<span class="tag approx">Approximate location</span>';
    } else if (lot.time_limited) {
      tag = '<span class="tag limited">' + esc(lot.restrictions) + "</span>";
    } else {
      tag = '<span class="tag open">No time limit</span>';
    }
    const cls = (i === 0 ? "nearest " : "") + (within ? "" : "out");
    return '<li class="' + cls.trim() + '" data-idx="' + LOTS.indexOf(lot) + '">'
      + '<span class="rank">' + (i + 1) + "</span><div>"
      + '<div class="lot-name">' + esc(lot.name) + "</div>"
      + '<div class="dist">' + formatDistance(row.meters)
      + ' <span class="walk">&middot; ' + formatWalk(row.meters) + "</span></div>"
      + '<div class="meta">' + lot.spaces + " spaces" + acc + "</div>"
      + tag + "</div></li>";
  }).join("");

  const note = lastRouted
    ? "Pedestrian routing along streets and footpaths, at 3.1 mph."
    : "Estimated from straight-line distance &mdash; routing unavailable.";
  const status = lastStatus
    ? '<div class="status">' + lastStatus + "</div>" : "";

  resultsEl.innerHTML = nearestLotHtml(dists) + radiiHtml(dists)
    + benchmarkHtml(dists) + status
    + '<div class="status">' + note + "</div>"
    + '<ol class="results">' + items + "</ol>";

  resultsEl.querySelectorAll("li").forEach(function (li) {
    li.addEventListener("click", function () {
      const lot = LOTS[+li.dataset.idx];
      map.panTo([lot.lat, lot.lon]);
      lot.marker.openPopup();
    });
  });

  // Fade whatever falls outside the widest radius.
  TARGETS.forEach(function (place, i) {
    const within = walkMinutes(dists[i]) <= widest;
    const base = i < LOTS.length ? 0.28 : 0.35;
    place.shape.setStyle({
      fillOpacity: within ? base : 0.05,
      opacity: within ? 1 : 0.25
    });
    if (place.marker) place.marker.setOpacity(within ? 1 : 0.4);
  });
}

function getJson(url) {
  return fetch(url)
    .then(function (r) { return r.ok ? r.json() : null; })
    .catch(function () { return null; });
}

function drawRouteTo(origin, lot, seq) {
  return getJson(ROUTER + "/route/v1/" + PROFILE + "/"
    + origin[1] + "," + origin[0] + ";" + lot.lon + "," + lot.lat
    + "?overview=full&geometries=geojson").then(function (route) {
    if (seq !== requestSeq) return;
    if (route && route.code === "Ok" && route.routes.length) {
      if (routeLine) map.removeLayer(routeLine);
      routeLine = L.polyline(
        route.routes[0].geometry.coordinates.map(function (c) {
          return [c[1], c[0]];
        }),
        { color: "#c2410c", weight: 4, opacity: 0.85,
          dashArray: "1 7", lineCap: "round" }
      ).addTo(map);
    } else if (routeLine) {
      map.removeLayer(routeLine);
      routeLine = null;
    }
  });
}

function measure(origin) {
  const seq = ++requestSeq;

  // Straight-line first so the panel is never empty while the router responds.
  const straight = TARGETS.map(function (p) {
    return haversine(origin, [p.lat, p.lon]);
  });
  render(straight, "Finding walking routes&hellip;", false);

  const coords = [origin[1] + "," + origin[0]].concat(TARGETS.map(function (p) {
    return p.lon + "," + p.lat;
  })).join(";");

  getJson(ROUTER + "/table/v1/" + PROFILE + "/" + coords
          + "?sources=0&annotations=distance").then(function (table) {
    if (seq !== requestSeq) return null;  // a newer origin has been set
    let dists = straight;
    let routed = false;

    if (table && table.code === "Ok" && table.distances && table.distances[0]) {
      const walking = table.distances[0].slice(1);
      const usable = walking.length === TARGETS.length
        && walking.every(function (d) { return typeof d === "number"; });
      if (usable) { dists = walking; routed = true; }
    }
    render(dists, "", routed);
    return drawRouteTo(origin, LOTS[nearestLotIndex(dists)], seq);
  });
}

function setOrigin(latlng) {
  if (youMarker) {
    youMarker.setLatLng(latlng);
  } else {
    youMarker = L.marker(latlng, { draggable: true, autoPan: true }).addTo(map);
    youMarker.bindTooltip("Drag to move");
    youMarker.on("dragend", function (e) { setOrigin(e.target.getLatLng()); });
  }
  measure([latlng.lat, latlng.lng]);
}

map.on("click", function (e) { setOrigin(e.latlng); });
</script>
</body>
</html>
"""


def render_html(lots: list[dict], bays: list[dict], title: str, subtitle: str) -> str:
    center = [
        round(sum(lot["lat"] for lot in lots) / len(lots), 6),
        round(sum(lot["lon"] for lot in lots) / len(lots), 6),
    ]
    replacements = {
        "__TITLE__": title,
        "__SUBTITLE__": subtitle,
        "__LOTS_JSON__": json.dumps(lots, separators=(",", ":")),
        "__BAYS_JSON__": json.dumps(bays, separators=(",", ":")),
        "__THRESHOLDS__": json.dumps(WALK_THRESHOLDS_MIN),
        "__BENCHMARKS__": json.dumps(
            [{"label": label, "feet": feet} for label, feet in WALMART_BENCHMARKS]
        ),
        "__FEET_PER_METER__": str(FEET_PER_METER),
        "__TOTAL_SPACES__": str(
            sum(lot["spaces"] for lot in lots) + sum(bay["spaces"] for bay in bays)
        ),
        "__ROUTER_BASE__": ROUTER_BASE,
        "__ROUTER_PROFILE__": ROUTER_PROFILE,
        "__WALK_SPEED__": str(WALK_SPEED_MPS),
        "__CENTER__": json.dumps(center),
    }
    html = HTML_TEMPLATE
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    return html


# --- Entry point ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent,
        help="Where to write the map, lot data, and OSM cache "
             "(default: the script's directory).",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-query OpenStreetMap and re-derive lot locations, discarding "
             "any hand edits in parking_lots.json.",
    )
    parser.add_argument(
        "--open", action="store_true", dest="open_map",
        help="Open the finished map in the default browser.",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lots_path = out_dir / "parking_lots.json"
    bays_path = out_dir / "street_parking.json"
    cache_path = out_dir / "osm_cache.json"
    html_path = out_dir / "greencastle_parking_map.html"

    # Hand edits to either data file survive re-runs unless --refresh is given.
    reuse = lots_path.exists() and bays_path.exists() and not args.refresh
    if reuse:
        print(f"Using existing data: {lots_path.name}, {bays_path.name}"
              "  (--refresh to re-derive)")
        lots = json.loads(lots_path.read_text(encoding="utf-8"))
        bays = json.loads(bays_path.read_text(encoding="utf-8"))
    else:
        streets, parking = index_osm(fetch_osm(cache_path, args.refresh))
        print(f"OSM: {len(streets)} named streets, {len(parking)} parking polygons")
        print("Locating lots:")
        lots = resolve_lots(streets, parking)
        if not lots:
            print("No lots could be located.", file=sys.stderr)
            return 1
        bays = resolve_street_bays(streets, parking)
        lots_path.write_text(json.dumps(lots, indent=2) + "\n", encoding="utf-8")
        bays_path.write_text(json.dumps(bays, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {lots_path}\nWrote {bays_path}")

    lot_spaces = sum(lot["spaces"] for lot in lots)
    bay_spaces = sum(bay["spaces"] for bay in bays)
    subtitle = (f"{lot_spaces + bay_spaces} spaces &middot; "
                f"{len(lots)} lots ({lot_spaces}) &middot; "
                f"{len(bays)} street bays ({bay_spaces})")
    html_path.write_text(render_html(lots, bays, MAP_TITLE, subtitle), encoding="utf-8")
    print(f"Wrote {html_path}")

    if args.open_map:
        import webbrowser

        webbrowser.open(html_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
