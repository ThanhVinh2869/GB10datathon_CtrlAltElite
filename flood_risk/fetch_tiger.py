"""Fast, offline alternative to fetch_osm.py using US Census TIGER/Line 2024.

Per county it reads ROADS (named streets) and LINEARWATER (creeks, bayous,
channels). For every response-image footprint it lists the named streets
inside it, and finds every point where a road crosses a waterway: these are
the candidate low-water crossings.

Output: flood_risk/osm_features.json (same format build_risk.py reads).

Download (once, ~28 MB) into flood_risk/tiger/:
  https://www2.census.gov/geo/tiger/TIGER2024/ROADS/tl_2024_<county>_roads.zip
  https://www2.census.gov/geo/tiger/TIGER2024/LINEARWATER/tl_2024_<county>_linearwater.zip
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import shapefile  # pyshp
from shapely.geometry import LineString, MultiLineString, Point, box, shape
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TIGER = HERE / "tiger"
OUT = HERE / "osm_features.json"

ROAD_TYPES = {"S1100": "highway", "S1200": "arterial", "S1400": "local road", "S1500": "4WD road",
              "S1640": "service road", "S1730": "alley", "S1740": "private road"}
MINOR = {"S1400", "S1500", "S1640", "S1730", "S1740"}  # low-water crossings are typically on minor roads


def load(path: Path, keep) -> tuple[list, list]:
    geoms, attrs = [], []
    with shapefile.Reader(str(path)) as r:
        fields = [f[0] for f in r.fields[1:]]
        for sr in r.iterShapeRecords():
            rec = dict(zip(fields, sr.record))
            if not keep(rec):
                continue
            geoms.append(shape(sr.shape.__geo_interface__))
            attrs.append(rec)
    return geoms, attrs


def points_of(g) -> list[Point]:
    if g.is_empty:
        return []
    if g.geom_type == "Point":
        return [g]
    if hasattr(g, "geoms"):
        return [p for part in g.geoms for p in points_of(part)]
    if isinstance(g, (LineString, MultiLineString)):  # collinear overlap: use midpoint
        return [g.interpolate(0.5, normalized=True)]
    return [g.centroid]


def main() -> None:
    df = pd.read_csv(ROOT / "image_metadata.csv", low_memory=False, dtype={"county_geoid": str})
    tiles = df[(df.image_role == "response") & df.south.notna()].copy()
    # A footprint may straddle counties: use every county it intersects.
    tiles["counties"] = tiles.counties_intersecting_footprint.fillna(tiles.county_geoid).astype(str).str.split(";")
    out = {}
    for county in sorted({c for cs in tiles.counties for c in cs}):
        rz, wz = TIGER / f"tl_2024_{county}_roads.zip", TIGER / f"tl_2024_{county}_linearwater.zip"
        if not rz.exists():
            print(f"missing {rz.name}, skipping county {county}")
            continue
        roads, rattr = load(rz, lambda r: r.get("FULLNAME") and r.get("MTFCC") in ROAD_TYPES)
        water, wattr = load(wz, lambda r: True) if wz.exists() else ([], [])
        rtree, wtree = STRtree(roads), STRtree(water) if water else None
        sub = tiles[tiles.counties.map(lambda cs: county in cs)]
        print(f"county {county}: {len(roads)} named roads, {len(water)} waterways, {len(sub)} tiles", flush=True)
        for t in sub.itertuples():
            bb = box(t.west, t.south, t.east, t.north)
            rec = out.setdefault(t.image_id, {"streets": {}, "crossings": {}, "source": "tiger"})
            ridx = [i for i in rtree.query(bb) if roads[i].intersects(bb)]
            for i in ridx:
                rec["streets"].setdefault(rattr[i]["FULLNAME"], ROAD_TYPES[rattr[i]["MTFCC"]])
            if wtree is None:
                continue
            for j in wtree.query(bb):
                w = water[j].intersection(bb)
                if w.is_empty:
                    continue
                creek = wattr[j].get("FULLNAME") or "unnamed channel"
                for i in ridx:
                    for p in points_of(roads[i].intersection(w)):
                        name = f"{rattr[i]['FULLNAME']} at {creek}"
                        kind = "road-creek crossing (minor road)" if rattr[i]["MTFCC"] in MINOR else "road-creek crossing (major road)"
                        rec["crossings"].setdefault(name, {"kind": kind, "name": name, "lat": round(p.y, 6), "lon": round(p.x, 6)})
    final = {k: {"streets": [{"name": n, "type": ty} for n, ty in v["streets"].items()],
                 "crossings": list(v["crossings"].values()), "source": "tiger"} for k, v in out.items()}
    OUT.write_text(json.dumps(final))
    n_st = sum(len(v["streets"]) for v in final.values())
    n_cx = sum(len(v["crossings"]) for v in final.values())
    print(f"wrote {len(final)} tiles, {n_st} tile-street links, {n_cx} road-waterway crossings -> {OUT.name}")


if __name__ == "__main__":
    main()
