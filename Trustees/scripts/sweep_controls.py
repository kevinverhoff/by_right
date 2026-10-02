"""Run the app headlessly and sweep every selector value, looking for exceptions."""
import os, sys
APP = str(__import__('pathlib').Path(__file__).resolve().parent.parent)
sys.path.insert(0, APP); os.chdir(APP)
from streamlit.testing.v1 import AppTest
import metrics as mx

LABELS = [m.label for m in mx.CATALOG]
fails = []

def fresh():
    a = AppTest.from_file(f"{APP}/app.py", default_timeout=180)
    a.run()
    return a

def check(a, tag):
    if a.exception:
        fails.append(f"{tag}: {[e.value for e in a.exception]}")
        return False
    return True

base = fresh()
if not check(base, "BASELINE"):
    print("\n".join(fails)); sys.exit(1)
print(f"baseline OK  selectboxes={[s.label for s in base.selectbox]}")

def sweep(key, values, tag):
    for v in values:
        a = fresh()
        a.selectbox(key=key).set_value(v).run()
        check(a, f"{tag}={v}")
    print(f"{tag}: swept {len(values)} values | failures so far {len(fails)}")

sweep("y_metric", [l for l in LABELS if l != "Population (2025 est.)"], "Y")
sweep("size_metric", ["None (uniform)"] + [m.label for m in mx.CATALOG if m.positive], "SIZE")
sweep("color_metric", ["Population tier", "None"] + LABELS, "COLOR")
sweep("map_metric", LABELS, "MAP")
sweep("rank_metric", LABELS, "RANK")

# Toggles and edge slices
# The theme now follows st.context.theme, which AppTest does not drive;
# dark mode is verified in the browser instead.
a = fresh()
a.selectbox(key="map_metric").set_value("Net surplus (receipts − disbursements)").run()
check(a, "diverging-map")

a = fresh(); a.selectbox(key="y_metric").set_value("Net surplus (receipts − disbursements)").run()
check(a, "LOGY+negatives (checkbox disabled)")

a = fresh(); a.checkbox(key="log_x").set_value(False).run(); check(a, "LINEAR-X")
a = fresh(); a.checkbox(key="only_complete").set_value(True).run(); check(a, "ONLY_COMPLETE")
for mode in ["By rank", "By value", "By value, clipped"]:
    a = fresh(); a.radio(key="map_scale").set_value(mode).run(); check(a, f"MAP_SCALE={mode}")
    # and the same mode against a signed measure
    a = fresh(); a.radio(key="map_scale").set_value(mode).run()
    a.selectbox(key="map_metric").set_value("Net surplus (receipts − disbursements)").run()
    check(a, f"MAP_SCALE={mode}+signed")

for county in ["Ohio", "Marion", "Switzerland"]:
    a = fresh(); a.multiselect[0].set_value([county]).run(); check(a, f"COUNTY={county}")

# Narrow population band -> smallest townships only
a = fresh(); a.select_slider[0].set_value((0, 500)).run(); check(a, "POP 0-500")
a = fresh(); a.select_slider[0].set_value((100000, 155258)).run(); check(a, "POP 100k+")

# Rankings "Lowest" on a metric with negatives
a = fresh()
a.selectbox(key="rank_metric").set_value("Net surplus (receipts − disbursements)").run()
a.radio(key="rank_order").set_value("Lowest").run(); check(a, "RANK lowest negative")

# --- highlight ---
for county in ["Putnam", "Marion", "Ohio"]:
    a = fresh(); a.selectbox(key="hl_county").set_value(county).run()
    check(a, f"HIGHLIGHT={county}")

# highlight crossed with every colour mode, since it rides on top of them
for cm in ["Population tier", "None", "Reserve (years of spending held)"]:
    a = fresh(); a.selectbox(key="hl_county").set_value("Putnam").run()
    a.selectbox(key="color_metric").set_value(cm).run()
    check(a, f"HIGHLIGHT+COLOR={cm}")

# highlight on every tab's own controls
a = fresh(); a.selectbox(key="hl_county").set_value("Putnam").run()
a.radio(key="map_scale").set_value("By value").run(); check(a, "HIGHLIGHT+map by value")
a = fresh(); a.selectbox(key="hl_county").set_value("Putnam").run()
a.selectbox(key="rank_metric").set_value("Reserve (years of spending held)").run()
check(a, "HIGHLIGHT+rank")
# a county with none of the top rows must fall back to reporting its best rank
for order in ["Highest", "Lowest"]:
    a = fresh(); a.selectbox(key="hl_county").set_value("Putnam").run()
    a.selectbox(key="rank_metric").set_value("Trustee pay per resident").run()
    a.radio(key="rank_order").set_value(order).run()
    check(a, f"HIGHLIGHT+rank absent, {order}")

# highlight a county the filter has excluded -> warning, not a crash
a = fresh()
a.selectbox(key="hl_county").set_value("Putnam").run()
a.multiselect[0].set_value(["Marion"]).run()
check(a, "HIGHLIGHT excluded by filter")
if not a.warning:
    fails.append("HIGHLIGHT excluded by filter: expected a warning, got none")

# --- trend line ---
for ylab in ["Disbursements per resident", "Net surplus (receipts − disbursements)",
             "Population change, 2020→2025", "Reserve (years of spending held)"]:
    a = fresh(); a.selectbox(key="y_metric").set_value(ylab).run()
    check(a, f"TREND y={ylab}")
    a = fresh(); a.selectbox(key="y_metric").set_value(ylab).run()
    a.checkbox(key="log_x").set_value(False).run()
    check(a, f"TREND linear-x y={ylab}")

a = fresh(); a.checkbox(key="show_fit").set_value(False).run(); check(a, "TREND off")
# a two-point field cannot be fitted; must not raise
a = fresh(); a.multiselect[0].set_value(["Ohio"]).run(); check(a, "TREND tiny field")
print(f"highlight + trend checks done | failures so far {len(fails)}")

print("\n=== RESULT ===")
if fails:
    print(f"{len(fails)} FAILURES")
    for f in fails[:40]: print(" -", f)
    sys.exit(1)
print("all sweeps passed, no exceptions")
