"""Build an interactive map of downtown Greencastle, IN public parking lots.

Lot facts (names, space counts, accessible spaces, time restrictions) are
transcribed from the City of Greencastle's Downtown Parking page:
https://www.cityofgreencastle.com/188/Downtown-Parking

That page publishes no coordinates, so each lot is located by querying
OpenStreetMap via the Overpass API: we compute the intersection of the two
streets the city uses to describe the lot, then claim the nearest
``amenity=parking`` polygon within a match radius. Lots that OSM does not
map fall back to a hand-placed coordinate recorded in ``LOT_SPECS``.

Output is a single self-contained HTML file. Clicking anywhere on the map
drops a pin and ranks every lot by walking distance, drawing the route to
the closest one.

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
from pathlib import Path

import requests

# --- Configuration ----------------------------------------------------------

# Downtown Greencastle, generous enough to include every lot plus context.
BBOX = (39.6390, -86.8720, 39.6500, -86.8570)  # south, west, north, east

OVERPASS_MIRRORS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "greencastle-parking-map/1.0 (https://www.cityofgreencastle.com)"

# The public OSRM demo server advertises a /foot/ profile but ignores it --
# it returns car routing (measured ~7.6 m/s implied speed). We therefore use
# the router only for street-following distance and geometry, and derive walk
# time ourselves from WALK_SPEED_MPS. Point ROUTER_BASE at an OSRM instance
# actually built with the foot profile and the durations become trustworthy.
ROUTER_BASE = "https://router.project-osrm.org"
ROUTER_PROFILE = "foot"
WALK_SPEED_MPS = 1.4  # ~3.1 mph, a normal adult walking pace

# How far from a street intersection we will still accept a parking polygon.
MATCH_RADIUS_M = 120.0

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
        # OpenStreetMap does not map this lot. Placed beside the tagged City
        # Hall building (OSM way/1355501091, 1 North Locust Street).
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
    for mirror in OVERPASS_MIRRORS:
        print(f"Querying Overpass: {mirror}")
        response = requests.post(
            mirror,
            data={"data": query},
            timeout=240,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code == 200:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(response.text, encoding="utf-8")
            return response.json()
        print(f"  -> HTTP {response.status_code}, trying next mirror")
        time.sleep(2)

    raise SystemExit(
        "Every Overpass mirror failed. Retry later, or run without --refresh "
        f"if {cache_path.name} already exists."
    )


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
    """Nearest parking polygon to ``anchor``, honoring the lot's side hint."""
    candidates = []
    for way in parking:
        centroid = polygon_centroid(way)
        if spec.side == "N" and centroid[0] < anchor[0]:
            continue
        if spec.side == "S" and centroid[0] > anchor[0]:
            continue
        distance = haversine_m(anchor, centroid)
        if distance <= MATCH_RADIUS_M:
            candidates.append((distance, way))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


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
  ol.results { list-style: none; margin: 4px 0 0; padding: 0; }
  ol.results li {
    border: 1px solid var(--line); border-radius: 8px;
    padding: 10px 12px; margin-bottom: 8px; cursor: pointer;
    display: grid; grid-template-columns: 26px 1fr; gap: 10px;
    transition: border-color .12s, background .12s;
  }
  ol.results li:hover { border-color: #9db6cf; background: #fafcff; }
  ol.results li.nearest { border-color: #1f6feb; background: #f4f9ff; }
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
    <div><span class="swatch" style="background:#1f6feb"></span>No time limit</div>
    <div><span class="swatch" style="background:#b45309"></span>Time-limited</div>
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
const ROUTER = "__ROUTER_BASE__";
const PROFILE = "__ROUTER_PROFILE__";
const WALK_SPEED = __WALK_SPEED__;
const CENTER = __CENTER__;

const map = L.map("map").setView(CENTER, 16);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

function lotColor(lot) {
  if (lot.approximate) return "#6b7280";
  return lot.time_limited ? "#b45309" : "#1f6feb";
}

function popupHtml(lot) {
  const acc = lot.accessible_spaces
    ? " &middot; " + lot.accessible_spaces + " accessible" : "";
  const approx = lot.approximate
    ? '<div style="color:#6b7280;margin-top:6px">Approximate location &mdash; this lot'
      + ' is not mapped in OpenStreetMap.</div>' : "";
  const notes = lot.notes
    ? '<div style="color:#5b6770;margin-top:6px">' + lot.notes + "</div>" : "";
  return "<b>" + lot.name + "</b><br>"
    + '<span style="color:#5b6770">' + lot.location_text + "</span><br><br>"
    + "<b>" + lot.spaces + "</b> spaces" + acc + "<br>"
    + lot.restrictions + "<br>" + lot.hours + notes + approx;
}

// Draw every lot: its real footprint where OSM has one, a dashed circle where
// it does not.
LOTS.forEach(function (lot, i) {
  const color = lotColor(lot);
  const style = { color: color, weight: 2, fillColor: color, fillOpacity: 0.28 };
  lot.shape = lot.polygon
    ? L.polygon(lot.polygon, style)
    : L.circle([lot.lat, lot.lon],
        Object.assign({}, style, { radius: 26, dashArray: "4 4" }));
  lot.shape.addTo(map).bindPopup(popupHtml(lot));

  lot.marker = L.marker([lot.lat, lot.lon], {
    icon: L.divIcon({
      className: "",
      html: '<div class="lot-badge" style="background:' + color + '">'
        + (i + 1) + "</div>",
      iconSize: [22, 22], iconAnchor: [11, 11]
    })
  }).addTo(map).bindPopup(popupHtml(lot));
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

function formatDistance(meters) {
  const feet = meters * 3.28084;
  if (feet < 1000) return Math.round(feet / 10) * 10 + " ft";
  return (meters / 1609.344).toFixed(2) + " mi";
}

function formatWalk(meters) {
  const minutes = Math.round(meters / WALK_SPEED / 60);
  return minutes < 1 ? "under a minute" : minutes + " min walk";
}

const resultsEl = document.getElementById("results");
let youMarker = null;
let routeLine = null;
// Bumped on every click/drag so a slow router response for an old origin
// cannot overwrite the results for a newer one.
let requestSeq = 0;

function render(rows, statusText) {
  const items = rows.map(function (row, i) {
    const lot = row.lot;
    const acc = lot.accessible_spaces
      ? " &middot; " + lot.accessible_spaces + " accessible" : "";
    let tag;
    if (lot.approximate) {
      tag = '<span class="tag approx">Approximate location</span>';
    } else if (lot.time_limited) {
      tag = '<span class="tag limited">' + lot.restrictions + "</span>";
    } else {
      tag = '<span class="tag open">No time limit</span>';
    }
    return '<li class="' + (i === 0 ? "nearest" : "") + '" data-idx="'
      + LOTS.indexOf(lot) + '">'
      + '<span class="rank">' + (i + 1) + "</span><div>"
      + '<div class="lot-name">' + lot.name + "</div>"
      + '<div class="dist">' + formatDistance(row.meters)
      + ' <span class="walk">&middot; ' + formatWalk(row.meters) + "</span></div>"
      + '<div class="meta">' + lot.spaces + " spaces" + acc + "</div>"
      + tag + "</div></li>";
  }).join("");

  resultsEl.innerHTML = '<div class="status">' + statusText + "</div>"
    + '<ol class="results">' + items + "</ol>";

  resultsEl.querySelectorAll("li").forEach(function (li) {
    li.addEventListener("click", function () {
      const lot = LOTS[+li.dataset.idx];
      map.panTo([lot.lat, lot.lon]);
      lot.marker.openPopup();
    });
  });
}

function drawRoute(coords) {
  if (routeLine) map.removeLayer(routeLine);
  routeLine = L.polyline(coords.map(function (c) { return [c[1], c[0]]; }), {
    color: "#c2410c", weight: 4, opacity: 0.85, dashArray: "1 7", lineCap: "round"
  }).addTo(map);
}

function getJson(url) {
  return fetch(url)
    .then(function (r) { return r.ok ? r.json() : null; })
    .catch(function () { return null; });
}

function measure(origin) {
  const seq = ++requestSeq;

  // Show straight-line distances immediately so the panel is never empty
  // while the router responds.
  const straight = LOTS.map(function (lot) {
    return { lot: lot, meters: haversine(origin, [lot.lat, lot.lon]) };
  }).sort(function (a, b) { return a.meters - b.meters; });
  render(straight, "Straight-line distance &mdash; finding walking routes&hellip;");

  const coords = [origin[1] + "," + origin[0]].concat(LOTS.map(function (l) {
    return l.lon + "," + l.lat;
  })).join(";");

  const tableUrl = ROUTER + "/table/v1/" + PROFILE + "/" + coords
    + "?sources=0&annotations=distance";

  getJson(tableUrl).then(function (table) {
    if (seq !== requestSeq) return null;  // a newer origin has been set
    let rows = straight;
    let status = "Straight-line distance (routing unavailable).";

    if (table && table.code === "Ok" && table.distances && table.distances[0]) {
      const walking = table.distances[0].slice(1);
      const usable = walking.length === LOTS.length
        && walking.every(function (d) { return typeof d === "number"; });
      if (usable) {
        rows = LOTS.map(function (lot, i) {
          return { lot: lot, meters: walking[i] };
        }).sort(function (a, b) { return a.meters - b.meters; });
        status = "Walking distance along streets and paths.";
      }
    }
    render(rows, status);

    // Draw the actual path to the closest lot.
    const nearest = rows[0].lot;
    const routeUrl = ROUTER + "/route/v1/" + PROFILE + "/"
      + origin[1] + "," + origin[0] + ";" + nearest.lon + "," + nearest.lat
      + "?overview=full&geometries=geojson";

    return getJson(routeUrl).then(function (route) {
      if (seq !== requestSeq) return;  // a newer origin has been set
      if (route && route.code === "Ok" && route.routes.length) {
        drawRoute(route.routes[0].geometry.coordinates);
      } else if (routeLine) {
        map.removeLayer(routeLine);
        routeLine = null;
      }
    });
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


def render_html(lots: list[dict], title: str, subtitle: str) -> str:
    center = [
        round(sum(lot["lat"] for lot in lots) / len(lots), 6),
        round(sum(lot["lon"] for lot in lots) / len(lots), 6),
    ]
    replacements = {
        "__TITLE__": title,
        "__SUBTITLE__": subtitle,
        "__LOTS_JSON__": json.dumps(lots, separators=(",", ":")),
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
    cache_path = out_dir / "osm_cache.json"
    html_path = out_dir / "greencastle_parking_map.html"

    # Hand edits to parking_lots.json survive re-runs unless --refresh is given.
    if lots_path.exists() and not args.refresh:
        print(f"Using existing lot data: {lots_path}  (--refresh to re-derive)")
        lots = json.loads(lots_path.read_text(encoding="utf-8"))
    else:
        streets, parking = index_osm(fetch_osm(cache_path, args.refresh))
        print(f"OSM: {len(streets)} named streets, {len(parking)} parking polygons")
        print("Locating lots:")
        lots = resolve_lots(streets, parking)
        if not lots:
            print("No lots could be located.", file=sys.stderr)
            return 1
        lots_path.write_text(json.dumps(lots, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {lots_path}")

    subtitle = (f"{len(lots)} public lots &middot; "
                f"{sum(lot['spaces'] for lot in lots)} spaces")
    html_path.write_text(render_html(lots, MAP_TITLE, subtitle), encoding="utf-8")
    print(f"Wrote {html_path}")

    if args.open_map:
        import webbrowser

        webbrowser.open(html_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
