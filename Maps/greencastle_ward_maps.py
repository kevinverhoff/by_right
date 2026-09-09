"""
Greencastle, Indiana city council ward map.

Builds an interactive Leaflet map (via folium) that answers one question:
"which city council ward is this address / point in?"

Data source
-----------
OpenStreetMap. The four Greencastle city council wards are tagged
boundary=administrative + admin_level=10 and are stored as closed WAYS
(not relations). The city limit is relation 127791 (admin_level=8).

Geometry is read from the live OSM API, not Overpass. The ward ways are new
enough that Overpass mirrors were briefly serving a snapshot without them,
and mirror availability is unreliable in general. Overpass is used only to
*discover* ward way IDs, so renumbered or added wards get picked up
automatically; the pinned IDs below are the fallback when it is unavailable.

Caveat: the wards do not share ways with the city boundary, so their edges
do not line up exactly. check_topology() reports the resulting gaps and
overlaps, and the map discloses them rather than guessing.

Usage
-----
    python greencastle_ward_maps.py            # use cached geojson if present
    python greencastle_ward_maps.py --refresh  # re-fetch from OSM
"""

import argparse
import json
import os
import re
import sys
import time

import folium
import geopandas as gpd
import requests
from branca.element import MacroElement
from jinja2 import Template
from shapely.geometry import Polygon
from shapely.ops import unary_union

# --- Configuration -----------------------------------------------------------

OSM_API = "https://api.openstreetmap.org/api/0.6"
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
# OSM asks for a descriptive User-Agent on API and Overpass calls.
HEADERS = {"User-Agent": "PCD-greencastle-ward-map/1.0"}

CITY_RELATION_ID = 127791  # Greencastle, IN city limits, admin_level=8

# Fallback ward way IDs, verified against the live OSM API. Only used when
# Overpass discovery comes back empty (see module docstring).
PINNED_WARD_WAY_IDS = [1557663809, 1557663810, 1557663811, 1557663812]

# Bounding box used for Overpass discovery: (south, west, north, east)
GREENCASTLE_BBOX = (39.60, -86.92, 39.68, -86.81)

GEOJSON_PATH = "greencastle_wards.geojson"
HTML_PATH = "Greencastle_Ward_Map.html"

INITIAL_LAT, INITIAL_LNG = 39.64449, -86.86473  # Greencastle courthouse square
INITIAL_ZOOM = 13

# One color per ward. Chosen to stay distinguishable at 45% fill opacity.
WARD_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#76b7b2"]


# --- OSM fetching ------------------------------------------------------------


def discover_ward_way_ids():
    """Find admin_level=10 boundary ways around Greencastle.

    Tries each Overpass mirror once (no retry storm). Returns the pinned IDs
    if Overpass is down or its data is stale.
    """
    south, west, north, east = GREENCASTLE_BBOX
    query = (
        "[out:json][timeout:60];"
        'way["boundary"="administrative"]["admin_level"="10"]'
        f"({south},{west},{north},{east});"
        "out ids;"
    )
    for mirror in OVERPASS_MIRRORS:
        try:
            resp = requests.post(
                mirror, data={"data": query}, headers=HEADERS, timeout=60
            )
            if resp.status_code != 200:
                print(f"  Overpass {mirror} -> HTTP {resp.status_code}")
                continue
            ids = [el["id"] for el in resp.json().get("elements", [])]
            if ids:
                print(f"  Overpass discovery found {len(ids)} ward way(s)")
                return sorted(ids)
            print(f"  Overpass {mirror} returned 0 wards (stale snapshot?)")
        except requests.RequestException as exc:
            print(f"  Overpass {mirror} -> {type(exc).__name__}")
    print(f"  Falling back to pinned ward way IDs: {PINNED_WARD_WAY_IDS}")
    return list(PINNED_WARD_WAY_IDS)


def osm_get(path, tries=4):
    """GET a live OSM API endpoint, retrying transient timeouts and 5xx."""
    url = f"{OSM_API}/{path}"
    for attempt in range(tries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=90)
            if resp.status_code == 200:
                return resp.json()
            print(f"  OSM API {path} -> HTTP {resp.status_code}")
            if resp.status_code < 500:
                resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  OSM API {path} -> {type(exc).__name__}")
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"OSM API request failed after {tries} tries: {url}")


def fetch_way_polygon(way_id):
    """Return (tags, shapely Polygon) for a closed OSM way."""
    elements = osm_get(f"way/{way_id}/full.json")["elements"]

    nodes = {
        el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"
    }
    way = next(el for el in elements if el["type"] == "way" and el["id"] == way_id)

    ring = [nodes[node_id] for node_id in way["nodes"]]
    if ring[0] != ring[-1]:
        raise ValueError(f"OSM way {way_id} is not a closed ring")
    if len(ring) < 4:
        raise ValueError(f"OSM way {way_id} has too few points for a polygon")

    return way.get("tags", {}), Polygon(ring)


def fetch_city_polygon():
    """Return (tags, polygon) for the Greencastle city-limit boundary relation."""
    relation = osm_get(f"relation/{CITY_RELATION_ID}.json")["elements"][0]

    outer_way_ids = [m["ref"] for m in relation["members"] if m["role"] == "outer"]
    if len(outer_way_ids) != 1:
        # Stitching multiple outer ways would need ring assembly; flag it rather
        # than silently produce a wrong outline.
        raise ValueError(
            f"Expected 1 outer way on relation {CITY_RELATION_ID}, got "
            f"{len(outer_way_ids)}. The boundary structure changed in OSM."
        )

    _, polygon = fetch_way_polygon(outer_way_ids[0])
    return relation.get("tags", {}), polygon


def ward_label(osm_name, way_id):
    """Normalize an OSM ward name to 'Ward N'.

    OSM currently misspells ward 3 as 'Greecnastle City Council Ward 3', so we
    key off the trailing number rather than the full string.
    """
    match = re.search(r"Ward\s+(\d+)", osm_name or "", flags=re.IGNORECASE)
    if match:
        return f"Ward {int(match.group(1))}"
    print(f"  WARNING: no ward number found in '{osm_name}' (OSM way {way_id})")
    return osm_name or f"Way {way_id}"


def build_geodataframes():
    """Fetch wards and city limits from OSM. Returns (wards_gdf, city_gdf)."""
    print("Discovering ward boundaries...")
    way_ids = discover_ward_way_ids()

    records = []
    for way_id in way_ids:
        tags, polygon = fetch_way_polygon(way_id)
        osm_name = tags.get("name", "")
        label = ward_label(osm_name, way_id)
        if osm_name and label not in osm_name:
            print(f"  Note: OSM name '{osm_name}' will display as '{label}'")
        records.append(
            {
                "ward": label,
                "osm_name": osm_name,
                "osm_way_id": way_id,
                "geometry": polygon,
            }
        )
        print(f"  Loaded {label} (OSM way {way_id})")

    wards = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    # Sort by ward number so colors and legend order stay stable across runs.
    wards["_sort"] = wards["ward"].str.extract(r"(\d+)").astype(float)
    wards = wards.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    print("Fetching city limits...")
    city_tags, city_polygon = fetch_city_polygon()
    city = gpd.GeoDataFrame(
        [{"name": city_tags.get("name", "Greencastle"), "geometry": city_polygon}],
        geometry="geometry",
        crs="EPSG:4326",
    )

    # Area is only meaningful in a projected CRS. EPSG:2966 is Indiana West
    # (US survey feet); 27,878,400 sq ft = 1 sq mi.
    wards["area_sq_mi"] = (wards.to_crs(epsg=2966).area / 27_878_400).round(2)

    return wards, city


# --- Topology checks ---------------------------------------------------------

SQFT_PER_SQMI = 27_878_400
# The browser reports ANY overlap as ambiguous, so this threshold only decides
# what is loud enough to call out on the page. 5,000 sq ft is roughly the
# footprint of a small lot: below that an overlap cannot really contain an
# address, above it a resident could plausibly live inside the contested strip.
DISCLOSE_SQFT = 5_000


def check_topology(wards, city):
    """Report where the wards fail to cleanly partition the city.

    The wards are drawn as independent ways rather than sharing the city
    boundary's ways, so their edges do not line up exactly. Every overlap is
    printed; the ones big enough to matter to a resident are returned so the
    map can disclose them.
    """
    wards_proj = wards.to_crs(epsg=2966)
    city_proj = city.to_crs(epsg=2966)
    union = unary_union(wards_proj.geometry.values)
    city_geom = city_proj.geometry.iloc[0]

    gap = city_geom.difference(union).area
    overhang = union.difference(city_geom).area
    print("Topology check:")
    print(
        f"  ward union {union.area / SQFT_PER_SQMI:.3f} sq mi vs "
        f"city {city_geom.area / SQFT_PER_SQMI:.3f} sq mi"
    )
    print(
        f"  in city / no ward: {gap / SQFT_PER_SQMI:.4f} sq mi | "
        f"in ward / outside city: {overhang / SQFT_PER_SQMI:.4f} sq mi"
    )

    overlaps = []
    found_any = False
    for i in range(len(wards_proj)):
        for j in range(i + 1, len(wards_proj)):
            area = wards_proj.geometry.iloc[i].intersection(
                wards_proj.geometry.iloc[j]
            ).area
            if area <= 0:
                continue
            found_any = True
            pair = (wards["ward"].iloc[i], wards["ward"].iloc[j])
            if area >= DISCLOSE_SQFT:
                overlaps.append(
                    {"wards": list(pair), "acres": round(area / 43_560, 2)}
                )
                print(
                    f"  WARNING: {pair[0]} and {pair[1]} overlap by "
                    f"{area:,.0f} sq ft ({area / 43_560:.2f} acres) -- "
                    "an address could fall inside this"
                )
            else:
                print(
                    f"  minor: {pair[0]} / {pair[1]} overlap {area:,.0f} sq ft "
                    "(edge sliver; still reported as ambiguous if clicked)"
                )
    if not found_any:
        print("  wards are mutually exclusive")
    return overlaps


# --- Caching -----------------------------------------------------------------


def save_cache(wards, city, path=GEOJSON_PATH):
    """Write wards and city limits to one FeatureCollection for reuse."""
    ward_features = json.loads(wards.to_json())["features"]
    for feature in ward_features:
        feature["properties"]["layer"] = "ward"

    city_features = json.loads(city.to_json())["features"]
    for feature in city_features:
        feature["properties"]["layer"] = "city"

    payload = {
        "type": "FeatureCollection",
        "features": ward_features + city_features,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(f"Cached geometry -> {path}")


def load_cache(path=GEOJSON_PATH):
    """Read back a cache written by save_cache. Returns (wards_gdf, city_gdf)."""
    gdf = gpd.read_file(path)
    wards = gdf[gdf["layer"] == "ward"].reset_index(drop=True)
    city = gdf[gdf["layer"] == "city"].reset_index(drop=True)
    return wards, city


# --- Map assembly ------------------------------------------------------------


class WardLocator(MacroElement):
    """Injects the address search box and the click-to-identify behavior.

    The ward identification runs entirely client-side: the ward rings are
    embedded in the page and tested with a ray-casting point-in-polygon, so
    naming a ward needs no server round-trip. Only the address lookup calls
    out, to OSM's Nominatim geocoder.
    """

    _template = Template(
        """
{% macro script(this, kwargs) %}
(function () {
  var map = {{ this._parent.get_name() }};
  var WARDS = {{ this.wards_json }};
  var CITY = {{ this.city_json }};
  var HOME = [{{ this.home_lat }}, {{ this.home_lng }}, {{ this.home_zoom }}];
  var VIEWBOX = "{{ this.viewbox }}";
  var marker = null;
  // Bumped by every new search and by reset, so a geocode that resolves late
  // cannot drop a marker the user has already cleared or superseded.
  var requestSeq = 0;

  // Ray-casting test for one linear ring. Ring is [[lng, lat], ...].
  function inRing(lng, lat, ring) {
    var inside = false;
    for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      var xi = ring[i][0], yi = ring[i][1];
      var xj = ring[j][0], yj = ring[j][1];
      var crosses = ((yi > lat) !== (yj > lat)) &&
                    (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi);
      if (crosses) { inside = !inside; }
    }
    return inside;
  }

  // A polygon is [outerRing, hole1, hole2, ...] in GeoJSON order.
  function inPolygon(lng, lat, rings) {
    if (!inRing(lng, lat, rings[0])) { return false; }
    for (var h = 1; h < rings.length; h++) {
      if (inRing(lng, lat, rings[h])) { return false; }
    }
    return true;
  }

  // Returns every ward containing the point. Normally one, but the OSM ward
  // polygons overlap in places, so two is possible and must not be hidden.
  function wardsAt(lat, lng) {
    var hits = [];
    for (var w = 0; w < WARDS.length; w++) {
      var polys = WARDS[w].polygons;
      for (var p = 0; p < polys.length; p++) {
        if (inPolygon(lng, lat, polys[p])) { hits.push(WARDS[w]); break; }
      }
    }
    return hits;
  }

  function inCity(lat, lng) {
    for (var p = 0; p < CITY.length; p++) {
      if (inPolygon(lng, lat, CITY[p])) { return true; }
    }
    return false;
  }

  function popupHtml(label, lat, lng) {
    var hits = wardsAt(lat, lng);
    var city = inCity(lat, lng);
    var coords = '<div class="gc-coords">' +
      lat.toFixed(5) + ', ' + lng.toFixed(5) + '</div>';
    var head = label ? '<div class="gc-addr">' + label + '</div>' : '';
    var body;

    if (hits.length === 1) {
      body = '<div class="gc-ward" style="border-left-color:' +
        hits[0].color + '">' + hits[0].ward + '</div>';
      if (!city) {
        body += '<div class="gc-note">Note: this point falls just outside ' +
          'the mapped city limit line. Boundaries here are approximate.</div>';
      }
    } else if (hits.length > 1) {
      // The ward polygons disagree here; say so rather than picking one.
      var names = hits.map(function (h) { return h.ward; }).join(' or ');
      body = '<div class="gc-ward gc-warn">' + names + '</div>' +
        '<div class="gc-note">The ward boundaries overlap at this location, ' +
        'so the ward cannot be determined from the map. Confirm with the ' +
        'Greencastle City Clerk or the Putnam County Clerk.</div>';
    } else if (city) {
      body = '<div class="gc-ward gc-warn">Ward undetermined</div>' +
        '<div class="gc-note">This point is inside the city limits but is ' +
        'not covered by any mapped ward &mdash; a gap in the boundary data. ' +
        'Confirm with the Greencastle City Clerk.</div>';
    } else {
      body = '<div class="gc-ward gc-out">Outside Greencastle city limits' +
        '</div><div class="gc-note">No city council ward &mdash; this point ' +
        'is not inside the city.</div>';
    }
    return head + body + coords;
  }

  function locate(lat, lng, label, zoom) {
    if (marker) { map.removeLayer(marker); }
    // Move first, without animation, then open the popup. An animated setView
    // (or the popup's autopan) stays in flight and would land after a reset
    // clicked moments later, snapping the view back to the search result.
    map.stop();
    map.setView([lat, lng], zoom || Math.max(map.getZoom(), 15),
                {animate: false});
    marker = L.marker([lat, lng]).addTo(map);
    marker.bindPopup(popupHtml(label, lat, lng), {maxWidth: 280}).openPopup();
  }

  map.on('click', function (e) { locate(e.latlng.lat, e.latlng.lng, null); });

  // --- Address search control ---
  var Search = L.Control.extend({
    options: {position: 'topleft'},
    onAdd: function () {
      var box = L.DomUtil.create('div', 'gc-search leaflet-bar');
      box.innerHTML =
        '<form class="gc-form">' +
        '<input class="gc-input" type="text" placeholder="Enter an address..." ' +
        'aria-label="Address search" autocomplete="off">' +
        '<button class="gc-go" type="submit">Find ward</button>' +
        '</form><div class="gc-status" role="status"></div>';
      L.DomEvent.disableClickPropagation(box);
      L.DomEvent.disableScrollPropagation(box);

      var form = box.querySelector('.gc-form');
      var input = box.querySelector('.gc-input');
      var status = box.querySelector('.gc-status');

      function say(msg, isError) {
        status.textContent = msg || '';
        status.className = 'gc-status' + (isError ? ' gc-error' : '');
      }

      function geocode(query, retried) {
        var url = 'https://nominatim.openstreetmap.org/search?format=jsonv2' +
          '&limit=1&countrycodes=us&viewbox=' + VIEWBOX +
          '&q=' + encodeURIComponent(query);
        return fetch(url, {headers: {'Accept': 'application/json'}})
          .then(function (r) {
            if (!r.ok) { throw new Error('Lookup failed (' + r.status + ')'); }
            return r.json();
          })
          .then(function (hits) {
            if (hits.length) { return hits[0]; }
            // Retry once with the city appended, for bare street addresses.
            if (!retried && !/greencastle/i.test(query)) {
              return geocode(query + ', Greencastle, Indiana', true);
            }
            return null;
          });
      }

      form.addEventListener('submit', function (e) {
        e.preventDefault();
        var query = input.value.trim();
        if (!query) { return; }
        var seq = ++requestSeq;
        say('Searching...');
        geocode(query, false).then(function (hit) {
          if (seq !== requestSeq) { return; }  // superseded or reset
          if (!hit) {
            say('No match found. Try adding the street number or ZIP.', true);
            return;
          }
          say('');
          locate(parseFloat(hit.lat), parseFloat(hit.lon), hit.display_name, 16);
        }).catch(function (err) {
          if (seq !== requestSeq) { return; }
          say(err.message || 'Address lookup unavailable.', true);
        });
      });
      return box;
    }
  });
  map.addControl(new Search());

  // --- Reset control ---
  var Reset = L.Control.extend({
    options: {position: 'topleft'},
    onAdd: function () {
      var bar = L.DomUtil.create('div', 'leaflet-bar gc-reset');
      var link = L.DomUtil.create('a', '', bar);
      link.href = '#';
      link.title = 'Reset map';
      link.innerHTML = '&#8962;';
      L.DomEvent.disableClickPropagation(bar);
      L.DomEvent.on(link, 'click', function (e) {
        L.DomEvent.preventDefault(e);
        requestSeq++;  // cancel any in-flight address lookup
        if (marker) { map.removeLayer(marker); marker = null; }
        var box = document.querySelector('.gc-input');
        if (box) { box.value = ''; }
        var st = document.querySelector('.gc-status');
        if (st) { st.textContent = ''; st.className = 'gc-status'; }
        // Cancel any pan/zoom still animating from a search, otherwise that
        // animation finishes after this and the view snaps back.
        map.stop();
        map.setView([HOME[0], HOME[1]], HOME[2], {animate: false});
      });
      return bar;
    }
  });
  map.addControl(new Reset());
})();
{% endmacro %}
"""
    )

    def __init__(self, wards_json, city_json, home, viewbox):
        super().__init__()
        self._name = "WardLocator"
        self.wards_json = wards_json
        self.city_json = city_json
        self.home_lat, self.home_lng, self.home_zoom = home
        self.viewbox = viewbox


PAGE_CSS = """
<style>
.gc-search {
  background: #fff; padding: 8px; border-radius: 4px;
  font-family: Verdana, Calibri, sans-serif; font-size: 12px;
  box-shadow: 0 1px 5px rgba(0,0,0,.4);
}
.gc-form { display: flex; gap: 6px; }
.gc-input {
  width: 210px; padding: 6px 8px; font-size: 13px;
  border: 1px solid #bbb; border-radius: 3px; font-family: inherit;
}
.gc-go {
  padding: 6px 10px; font-size: 12px; font-family: inherit; cursor: pointer;
  border: 0; border-radius: 3px; background: #2c5f8a; color: #fff;
}
.gc-go:hover { background: #1e4463; }
.gc-status { margin-top: 5px; color: #555; }
.gc-status:empty { display: none; }
.gc-error { color: #b3261e; }
.gc-reset a { font-size: 20px; line-height: 26px; text-align: center; }
.gc-addr { font-size: 12px; color: #444; margin-bottom: 6px; }
.gc-ward {
  font-size: 17px; font-weight: 700; padding: 2px 0 2px 9px;
  border-left: 5px solid #999; margin-bottom: 5px;
}
.gc-out { font-size: 14px; font-weight: 600; border-left-color: #b3261e; }
.gc-warn { font-size: 14px; font-weight: 600; border-left-color: #d18b00; }
.gc-note { font-size: 11px; color: #555; margin-bottom: 5px; }
.gc-coords { font-size: 11px; color: #888; }
.gc-legend {
  position: fixed; bottom: 24px; right: 12px; z-index: 900;
  background: rgba(255,255,255,.94); padding: 10px 12px; border-radius: 4px;
  font-family: Verdana, Calibri, sans-serif; font-size: 12px;
  box-shadow: 0 1px 5px rgba(0,0,0,.35);
}
.gc-legend h4 { margin: 0 0 7px; font-size: 12px; }
.gc-legend div { margin-bottom: 4px; }
.gc-swatch {
  display: inline-block; width: 14px; height: 14px; margin-right: 7px;
  vertical-align: -2px; border: 1px solid #555;
}
.gc-help {
  position: fixed; bottom: 24px; left: 12px; z-index: 900; max-width: 250px;
  background: rgba(255,255,255,.94); padding: 10px 12px; border-radius: 4px;
  font-family: Verdana, Calibri, sans-serif; font-size: 11px; color: #333;
  box-shadow: 0 1px 5px rgba(0,0,0,.35);
}
.leaflet-popup-content { font-family: Verdana, Calibri, sans-serif; }
</style>
"""


def rings_of(geom):
    """Flatten a (Multi)Polygon to the nested coordinate lists GeoJSON uses.

    Result is [polygon, ...] where each polygon is [outerRing, hole, ...] and
    each ring is [[lng, lat], ...] -- the shape the browser-side
    point-in-polygon test expects.
    """
    polygons = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    return [
        [[list(c) for c in poly.exterior.coords]]
        + [[list(c) for c in ring.coords] for ring in poly.interiors]
        for poly in polygons
    ]


def build_map(wards, city, overlaps=()):
    """Assemble the folium map and return it.

    `overlaps` comes from check_topology and is disclosed on the page so the
    map does not quietly imply the wards are a clean partition.
    """
    colors = {
        row["ward"]: WARD_COLORS[i % len(WARD_COLORS)] for i, row in wards.iterrows()
    }

    fmap = folium.Map(
        location=[INITIAL_LAT, INITIAL_LNG],
        zoom_start=INITIAL_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # City limits: outline only, drawn under the wards and not clickable so it
    # never swallows the map click that identifies a ward.
    folium.GeoJson(
        city,
        name="Greencastle city limits",
        style_function=lambda _: {
            "color": "#1a1a1a",
            "weight": 3,
            "fillOpacity": 0,
            "dashArray": "6,4",
        },
        interactive=False,
    ).add_to(fmap)

    ward_layer = folium.FeatureGroup(name="City council wards")
    folium.GeoJson(
        wards,
        style_function=lambda feature: {
            "fillColor": colors.get(feature["properties"]["ward"], "#cccccc"),
            "color": "#333333",
            "weight": 1.5,
            "fillOpacity": 0.45,
        },
        highlight_function=lambda _: {"weight": 3.5, "fillOpacity": 0.6},
        tooltip=folium.GeoJsonTooltip(
            fields=["ward", "area_sq_mi"],
            aliases=["", "Area (sq mi):"],
            sticky=True,
        ),
    ).add_to(ward_layer)
    ward_layer.add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)

    legend_rows = "".join(
        f'<div><span class="gc-swatch" style="background:{colors[w]}"></span>{w}</div>'
        for w in wards["ward"]
    )
    caveat = ""
    if overlaps:
        pairs = "; ".join(
            f"{o['wards'][0]} and {o['wards'][1]} ({o['acres']} acres)"
            for o in overlaps
        )
        caveat = (
            '<br><br><b style="color:#8a5a00">Known boundary conflict:</b> '
            f"the OpenStreetMap ward polygons overlap at {pairs}. "
            "Locations inside an overlap are reported as ambiguous."
        )

    fmap.get_root().header.add_child(folium.Element(PAGE_CSS))
    fmap.get_root().html.add_child(
        folium.Element(
            f'<div class="gc-legend"><h4>Greencastle City Council</h4>{legend_rows}'
            '<div><span class="gc-swatch" style="background:transparent;'
            'border:2px dashed #1a1a1a"></span>City limits</div></div>'
            '<div class="gc-help"><b>How to use:</b> type an address in the box '
            "at top left, or click anywhere on the map, to see which city "
            "council ward the location falls in.<br><br>"
            "Ward boundaries from OpenStreetMap. Confirm with the Putnam "
            "County Clerk before relying on this for official purposes."
            f"{caveat}</div>"
        )
    )

    # Embed the rings the browser needs for point-in-polygon.
    ward_payload = [
        {
            "ward": row["ward"],
            "color": colors[row["ward"]],
            "polygons": rings_of(row["geometry"]),
        }
        for _, row in wards.iterrows()
    ]
    city_payload = rings_of(city.geometry.iloc[0])

    # Nominatim viewbox is west,north,east,south -- used to bias, not restrict.
    west, south, east, north = city.total_bounds
    viewbox = f"{west:.4f},{north:.4f},{east:.4f},{south:.4f}"

    # Attach to the Map, not to get_root(): the template resolves the Leaflet
    # map variable via `this._parent.get_name()`, and the root is the Figure,
    # whose name is not a declared JS variable.
    fmap.add_child(
        WardLocator(
            wards_json=json.dumps(ward_payload),
            city_json=json.dumps(city_payload),
            home=(INITIAL_LAT, INITIAL_LNG, INITIAL_ZOOM),
            viewbox=viewbox,
        )
    )
    return fmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch boundaries from OSM instead of using the cached GeoJSON",
    )
    args = parser.parse_args()

    if args.refresh or not os.path.exists(GEOJSON_PATH):
        wards, city = build_geodataframes()
        save_cache(wards, city)
    else:
        # Loudly date the cache: an OSM edit will not show up here until
        # --refresh is run, and a stale cache looks exactly like a live result.
        age_hours = (time.time() - os.path.getmtime(GEOJSON_PATH)) / 3600
        stamp = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(GEOJSON_PATH))
        )
        print(
            f"Using CACHED geometry from {GEOJSON_PATH}, fetched {stamp} "
            f"({age_hours:.1f}h ago)."
        )
        print("  Any OSM edits made since then are NOT reflected. "
              "Re-run with --refresh to pull current data.")
        wards, city = load_cache()

    if wards.empty:
        sys.exit("No ward boundaries loaded; nothing to map.")

    overlaps = check_topology(wards, city)

    print(f"Building map for {len(wards)} wards: {', '.join(wards['ward'])}")
    fmap = build_map(wards, city, overlaps)
    fmap.save(HTML_PATH)
    print(f"Interactive map saved to {HTML_PATH}")


if __name__ == "__main__":
    main()
