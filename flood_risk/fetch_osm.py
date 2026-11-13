"""Fetch named streets, fords, bridges and culverts inside every georeferenced
response-image footprint from OpenStreetMap (Overpass API).

Output: flood_risk/osm_features.json  {image_id: {"streets": [...], "crossings": [...]}}
Cached: tiles already present are skipped on re-run.

    python flood_risk/fetch_osm.py              # Overpass: all named streets + fords/culverts/bridges per tile
    python flood_risk/fetch_osm.py --nominatim  # fallback when Overpass is down: nearest street at tile center
                                                # (1 req/s per Nominatim policy; Overpass re-run upgrades these)
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "osm_features.json"
URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"]


def query(bboxes: list[tuple]) -> list[dict]:
    parts = []
    for s, w, n, e in bboxes:
        b = f"({s},{w},{n},{e})"
        parts += [
            f'way[highway~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street)$"][name]{b};',
            f"way[waterway][tunnel=culvert]{b};",
            f"node[ford=yes]{b};",
            f"way[ford=yes]{b};",
        ]
    q = "[out:json][timeout:90];(" + "".join(parts) + ");out tags geom;"
    data = urllib.parse.urlencode({"data": q}).encode()
    for attempt in range(6):
        url = URLS[attempt % len(URLS)]
        req = urllib.request.Request(url, data=data, headers={"User-Agent": "flood-risk-dashboard/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["elements"]
        except Exception as e:  # noqa: BLE001
            print("  retry", attempt, url, e, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Overpass failed")


def points(el: dict) -> list[tuple[float, float]]:
    if el["type"] == "node":
        return [(el["lat"], el["lon"])]
    return [(p["lat"], p["lon"]) for p in el.get("geometry", []) if p]


def main() -> None:
    df = pd.read_csv(ROOT / "image_metadata.csv", low_memory=False)
    tiles = df[(df.image_role == "response") & df.georeferenced.astype(bool) & df.south.notna()]
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args_nominatim():
        return nominatim(tiles, cache)
    upgradable = {k for k, v in cache.items() if v.get("source") == "nominatim"}
    todo = tiles[~tiles.image_id.isin(set(cache) - upgradable)].copy()
    print(f"{len(cache)} cached, {len(todo)} to fetch", flush=True)

    # One Overpass query per spatial block (tiles cluster tightly), then intersect each
    # tile footprint locally. Far fewer requests than per-tile queries.
    todo = todo.assign(block=todo.spatial_block_005deg.fillna(todo.locality))
    blocks = list(todo.groupby("block"))
    for bi, (block, chunk) in enumerate(blocks, 1):
        bbox = (chunk.south.min(), chunk.west.min(), chunk.north.max(), chunk.east.max())
        try:
            els = query([bbox])
        except RuntimeError as e:
            print(f"  skipping block {block}: {e} (re-run to retry)", flush=True)
            continue
        for r in chunk.itertuples():
            streets: dict[str, str] = {}
            crossings = []
            for el in els:
                tags = el.get("tags", {})
                pts = [p for p in points(el) if r.south <= p[0] <= r.north and r.west <= p[1] <= r.east]
                if not pts:
                    continue
                lat, lon = pts[len(pts) // 2]
                if tags.get("ford") == "yes":
                    crossings.append({"kind": "ford (low-water crossing)", "name": tags.get("name", "unnamed ford"), "lat": lat, "lon": lon})
                elif tags.get("tunnel") == "culvert":
                    crossings.append({"kind": "creek culvert", "name": tags.get("name", tags.get("waterway", "culvert")), "lat": lat, "lon": lon})
                elif "highway" in tags:
                    name = tags["name"]
                    if tags.get("bridge") == "yes":
                        crossings.append({"kind": "road bridge", "name": name, "lat": lat, "lon": lon})
                    streets.setdefault(name, tags["highway"])
            seen, uniq = set(), []
            for c in crossings:
                k = (c["kind"], c["name"])
                if k not in seen:
                    seen.add(k)
                    uniq.append(c)
            cache[r.image_id] = {"streets": [{"name": n, "type": t} for n, t in streets.items()], "crossings": uniq,
                                 "suburb": cache.get(r.image_id, {}).get("suburb"), "source": "overpass"}
        OUT.write_text(json.dumps(cache))
        print(f"  block {bi}/{len(blocks)} {block}: {len(chunk)} tiles, {len(els)} OSM elements", flush=True)
        time.sleep(1)


def args_nominatim() -> bool:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominatim", action="store_true")
    return ap.parse_args().nominatim


def nominatim(tiles: pd.DataFrame, cache: dict) -> None:
    todo = tiles[~tiles.image_id.isin(cache)]
    print(f"Nominatim fallback: {len(todo)} tiles (~{len(todo) * 1.1 / 60:.0f} min)", flush=True)
    for i, r in enumerate(todo.itertuples(), 1):
        q = urllib.parse.urlencode({"format": "jsonv2", "lat": r.latitude, "lon": r.longitude, "zoom": 17, "addressdetails": 1})
        req = urllib.request.Request(f"https://nominatim.openstreetmap.org/reverse?{q}",
                                     headers={"User-Agent": "flood-risk-dashboard/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.load(resp)
            a = d.get("address", {})
            road = a.get("road")
            cache[r.image_id] = {
                "streets": [{"name": road, "type": d.get("type", "road")}] if road else [],
                "crossings": [],
                "suburb": a.get("suburb") or a.get("neighbourhood") or a.get("village") or a.get("town") or a.get("city"),
                "source": "nominatim",
            }
        except Exception as e:  # noqa: BLE001
            print(f"  {r.image_id}: {e}", flush=True)
        if i % 25 == 0:
            OUT.write_text(json.dumps(cache))
            print(f"  {i}/{len(todo)}", flush=True)
        time.sleep(1.1)
    OUT.write_text(json.dumps(cache))
    print("done", flush=True)


if __name__ == "__main__":
    main()
