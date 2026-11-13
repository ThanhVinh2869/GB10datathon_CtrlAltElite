"""Combine Qwen3-VL flood scores, image metadata and OSM features into a
per-tile flood-risk score, a predicted susceptibility model, and neighborhood
rankings. Writes dashboard/data.js and before/after thumbnails.

Risk (0-100) = 100 * (0.50 * observed impact      <- Qwen3-VL before/after comparison
                    + 0.30 * predicted susceptibility <- model on static terrain/hydro features
                    + 0.20 * exposure & vulnerability) <- roads, buildings, county ACS

The susceptibility model uses ONLY pre-event features (elevation relative to
neighbors, slope, FEMA flood zone, waterways, high-water-mark proximity) to
predict where the VLM saw flooding. Probabilities are out-of-fold from spatial
group CV (groups = 0.05 deg blocks), so each tile's prediction comes from a
model that never saw its neighborhood block - this is what "flood-prone"
means for places with no imagery yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DASH = ROOT / "dashboard"
THUMBS = DASH / "thumbs"
IMG_ROOT = ROOT / "UTSA-flood-images"

ROAD_LVL = {"none": 0.0, "some": 0.5, "many": 1.0}
BLDG_LVL = {"none": 0.0, "few": 0.5, "many": 1.0}
SHRINK_K = 10  # pseudo-tiles of prior weight in neighborhood scores


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0


def tier(r: float) -> str:
    return "critical" if r >= 70 else "high" if r >= 50 else "moderate" if r >= 30 else "low"


def main() -> None:
    meta = pd.read_csv(ROOT / "image_metadata.csv", low_memory=False)
    vlm = pd.read_json(HERE / "vlm_scores.jsonl", lines=True).drop_duplicates("image_id", keep="last")
    osm = json.loads((HERE / "osm_features.json").read_text()) if (HERE / "osm_features.json").exists() else {}

    df = meta[meta.image_role == "response"].merge(vlm, on="image_id", how="inner")
    df = df[df.latitude.notna()].copy()
    print(f"{len(df)} georeferenced, VLM-scored tiles")

    # ---------- observed impact (VLM) ----------
    sev = pd.to_numeric(df.severity, errors="coerce").fillna(0).clip(0, 5) / 5
    neww = pd.to_numeric(df.new_water_vs_before_pct, errors="coerce").fillna(0).clip(0, 100) / 100
    stand = pd.to_numeric(df.standing_water_pct, errors="coerce").fillna(0).clip(0, 100) / 100
    water = np.where(df.before_id.notna(), neww, stand)  # single images have no baseline
    roads = df.roads_flooded.map(ROAD_LVL).fillna(0)
    bldg = df.buildings_affected.map(BLDG_LVL).fillna(0)
    df["observed"] = 0.45 * sev + 0.25 * water + 0.20 * roads + 0.10 * bldg
    df["flooded"] = (pd.to_numeric(df.severity, errors="coerce").fillna(0) >= 3).astype(int)

    # ---------- predicted susceptibility (static features only) ----------
    elev = df.terrain_elevation_mean_m.fillna(df.elevation_m)
    df["rel_elev_m"] = elev - elev.groupby(df.locality).transform("median")  # negative = lower than neighbors
    df["elev_pct_in_locality"] = elev.groupby(df.locality).rank(pct=True)
    zones = df.fema_flood_zones.fillna("[]")
    feats = pd.DataFrame({
        "rel_elev_m": df.rel_elev_m,
        "elev_pct_in_locality": df.elev_pct_in_locality,
        "relief_m": df.terrain_elevation_max_m - df.terrain_elevation_min_m,
        "slope_mean": df.terrain_slope_mean_degrees,
        "slope_max": df.terrain_slope_max_degrees,
        "fema_sfha": (df.fema_sfha_tf == "T").astype(int),
        # A blank FEMA field is "unknown", not "outside the floodplain" (dataset README),
        # so give the model an explicit indicator instead of silently treating it as 0.
        "fema_missing": df.fema_sfha_tf.isna().astype(int),
        "fema_AE_A": zones.str.contains(r'"A[EO]?"').astype(int),
        "fema_floodway": df.fema_zone_subtypes.fillna("").str.contains("FLOODWAY", case=False).astype(int),
        "waterways": df.osm_waterway_count,
        "water_areas": df.osm_water_area_count,
        "hwm_dist_km": df.nearest_high_water_mark_distance_km,
        "hwm_within_25km": df.nearby_marks_within_25km,
    }).astype(float)
    feats = feats.fillna(feats.median())
    groups = df.spatial_block_005deg.fillna(df.locality)
    oof = np.zeros(len(df))
    n_splits = min(5, groups.nunique())
    for tr, te in GroupKFold(n_splits).split(feats, df.flooded, groups):
        m = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
        m.fit(feats.iloc[tr], df.flooded.iloc[tr])
        oof[te] = m.predict_proba(feats.iloc[te])[:, 1]
    auc = roc_auc_score(df.flooded, oof) if df.flooded.nunique() > 1 else float("nan")
    full = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    full.fit(feats, df.flooded)
    importances = sorted(zip(feats.columns, full.feature_importances_), key=lambda x: -x[1])
    df["predicted"] = oof
    print(f"susceptibility model spatial-CV AUC = {auc:.3f}")
    print("top features:", [(k, round(v, 3)) for k, v in importances[:6]])

    # ---------- exposure & vulnerability ----------
    svi_cols = ["county_poverty_percent", "county_no_vehicle_households_percent",
                "county_age65plus_percent", "county_disability_percent"]
    svi = sum(minmax(df[c].fillna(df[c].median())) for c in svi_cols) / len(svi_cols)
    df["exposure"] = (0.4 * df.osm_road_or_path_count.rank(pct=True)
                      + 0.4 * df.osm_building_count.rank(pct=True) + 0.2 * svi)

    df["risk"] = (100 * (0.5 * df.observed + 0.3 * df.predicted + 0.2 * df.exposure)).round(1)
    df["tier"] = df.risk.map(tier)
    # Hill Country 2025 imagery was flown July 11-22, a week+ after the July 4 peak, so its
    # observed-impact scores reflect aftermath, not peak flooding. Expose a within-event
    # percentile so tiles are compared against their own event rather than across events.
    df["event_pct"] = df.groupby("event_name").risk.rank(pct=True)
    df["aftermath"] = df.image_view_type.eq("aftermath")

    # ---------- streets & low-water crossings ----------
    tiles, streets, crossings = [], [], []
    for r in df.itertuples():
        o = osm.get(r.image_id, {"streets": [], "crossings": []})
        roads_hit = r.roads_flooded in ("some", "many")
        vlm_lwc = str(r.low_water_crossing).lower() == "yes"
        water_frac = float(water[df.index.get_loc(r.Index)])
        lwc_reasons = []
        if vlm_lwc:
            lwc_reasons.append("VLM saw road overtopped at a channel")
        osm_cx = [c for c in o["crossings"] if "major road" not in c["kind"] and c["kind"] != "road bridge"]  # highways over bayous are bridges
        if osm_cx and (roads_hit or water_frac >= 0.2 or r.predicted >= 0.6):
            lwc_reasons.append(f"{len(osm_cx)} mapped crossing(s) in flooded/flood-prone tile")
        is_lwc = bool(lwc_reasons)
        tiles.append({
            "id": r.image_id, "before": r.before_id if isinstance(r.before_id, str) else None,
            "lat": round(r.latitude, 6), "lon": round(r.longitude, 6),
            "b": [round(r.south, 6), round(r.west, 6), round(r.north, 6), round(r.east, 6)],
            "loc": r.locality, "event": r.event_name, "date": r.acquisition_date, "county": r.county_name,
            "ws": ", ".join(json.loads(r.watershed_names)) if isinstance(r.watershed_names, str) else "",
            "risk": r.risk, "tier": r.tier, "obs": round(r.observed, 3), "pred": round(r.predicted, 3),
            "exp": round(r.exposure, 3), "sev": int(pd.to_numeric(r.severity, errors="coerce") or 0),
            "water": round(water_frac * 100), "roads": r.roads_flooded, "bldg": r.buildings_affected,
            "road_desc": r.flooded_road_desc or "", "notes": r.notes or "", "conf": r.confidence,
            "fema": r.fema_flood_zones, "fema_known": pd.notna(r.fema_sfha_tf),
            "event_pct": round(float(r.event_pct) * 100), "aftermath": bool(r.aftermath), "elev": round(float(r.terrain_elevation_mean_m), 1) if pd.notna(r.terrain_elevation_mean_m) else None,
            "rel_elev": round(float(r.rel_elev_m), 2) if pd.notna(r.rel_elev_m) else None,
            "lwc": is_lwc, "lwc_why": lwc_reasons,
            "streets": [s["name"] for s in o["streets"]][:12], "suburb": o.get("suburb") or "",
        })
        # Every street in every tile is listed; "flagged" marks the flood-prone ones.
        for s in o["streets"]:
            streets.append({"name": s["name"], "type": s["type"], "tile": r.image_id, "loc": r.locality, "suburb": o.get("suburb") or "",
                            "risk": r.risk, "observed": roads_hit, "predicted": r.predicted >= 0.6, "pred": round(r.predicted, 3),
                            "lat": r.latitude, "lon": r.longitude})
        # Every mapped minor crossing is listed; "flagged" marks low-water crossings in flooded/flood-prone tiles.
        pts = list(osm_cx)
        if vlm_lwc and not pts:
            desc = r.flooded_road_desc if isinstance(r.flooded_road_desc, str) else "unnamed"
            pts = [{"kind": "VLM-detected crossing", "name": desc[:60], "lat": r.latitude, "lon": r.longitude}]
        for c in pts:
            crossings.append({**c, "tile": r.image_id, "loc": r.locality, "risk": r.risk,
                              "vlm": vlm_lwc, "flagged": is_lwc, "why": "; ".join(lwc_reasons)})

    # Aggregate streets across tiles: a street's risk = max tile risk, count of flagged tiles.
    st = pd.DataFrame(streets)
    street_rank, n_flagged_streets = [], 0
    if len(st):
        g = st.groupby(["name", "loc"])
        street_rank = (g.agg(risk=("risk", "max"), tiles=("tile", "nunique"), observed=("observed", "any"),
                             predicted=("predicted", "any"), pred=("pred", "max"), type=("type", "first"), suburb=("suburb", "first"), lat=("lat", "mean"), lon=("lon", "mean"),
                             tile=("tile", "first"))
                       .reset_index().sort_values(["risk", "tiles"], ascending=False))
        street_rank["basis"] = np.where(street_rank.observed, "Observed", np.where(street_rank.predicted, "Predicted", "Not flagged"))
        street_rank["risk"] = street_rank.risk.round(1)
        n_flagged_streets = int((street_rank.basis != "Not flagged").sum())
        street_rank = street_rank.to_dict("records")
    cx = pd.DataFrame(crossings)
    if len(cx):
        cx = cx.sort_values("risk", ascending=False).drop_duplicates(["kind", "name", "loc"])
    crossings = cx.to_dict("records") if len(cx) else []

    # ---------- neighborhood ranking ----------
    nb = (df.groupby("locality")
          .agg(event=("event_name", "first"), county=("county_name", "first"), tiles=("image_id", "count"),
               risk_mean=("risk", "mean"), risk_p90=("risk", lambda s: s.quantile(0.9)),
               flooded_share=("flooded", "mean"), observed=("observed", "mean"), predicted=("predicted", "mean"),
               exposure=("exposure", "mean"), lat=("latitude", "mean"), lon=("longitude", "mean"))
          .reset_index())
    nb["critical_tiles"] = df[df.tier == "critical"].groupby("locality").size().reindex(nb.locality).fillna(0).astype(int).values
    lwc_by = pd.Series([t["loc"] for t in tiles if t["lwc"]]).value_counts()
    nb["lwc_tiles"] = nb.locality.map(lwc_by).fillna(0).astype(int)
    # Neighborhood score: blend of mean and 90th-percentile tile risk so small hot spots still count.
    # Shrink toward the all-tile mean by sample size (K pseudo-tiles) so 7-tile areas don't outrank 250-tile ones on noise.
    raw = 0.6 * nb.risk_mean + 0.4 * nb.risk_p90
    nb["raw_score"] = raw.round(1)
    ev_mean = df.groupby("event_name").risk.mean()
    nb["score"] = ((nb.tiles * raw + SHRINK_K * nb.event.map(ev_mean)) / (nb.tiles + SHRINK_K)).round(1)
    nb = nb.sort_values("score", ascending=False).reset_index(drop=True)
    nb["rank"] = nb.index + 1
    for c in ["risk_mean", "risk_p90", "flooded_share", "observed", "predicted", "exposure"]:
        nb[c] = nb[c].round(3)

    # Sub-neighborhood ranking by watershed (HUC12) within locality
    df["ws_name"] = df.watershed_names.fillna("[]").map(lambda s: ", ".join(json.loads(s)) or "unknown")
    ws = (df.groupby(["ws_name", "locality"])
          .agg(tiles=("image_id", "count"), risk_mean=("risk", "mean"), risk_p90=("risk", lambda s: s.quantile(0.9)),
               flooded_share=("flooded", "mean"))
          .reset_index())
    ws_prior = ws.locality.map(nb.set_index("locality").event).map(ev_mean)
    ws["score"] = ((ws.tiles * (0.6 * ws.risk_mean + 0.4 * ws.risk_p90) + SHRINK_K * ws_prior) / (ws.tiles + SHRINK_K)).round(1)
    ws = ws.sort_values("score", ascending=False)
    ws[["risk_mean", "risk_p90", "flooded_share"]] = ws[["risk_mean", "risk_p90", "flooded_share"]].round(3)

    # ---------- thumbnails ----------
    THUMBS.mkdir(parents=True, exist_ok=True)
    paths = meta.set_index("image_id").image_path
    for t in tiles:
        for iid in filter(None, [t["id"], t["before"]]):
            out = THUMBS / f"{iid}.jpg"
            if not out.exists():
                im = Image.open(IMG_ROOT / paths[iid]).convert("RGB")
                im.thumbnail((480, 480))
                im.save(out, quality=78)

    summary = {
        "tiles": len(df), "flooded_tiles": int(df.flooded.sum()), "auc": round(auc, 3),
        "critical": int((df.tier == "critical").sum()), "high": int((df.tier == "high").sum()),
        "lwc": int(sum(t["lwc"] for t in tiles)), "streets": n_flagged_streets,
        "importances": [[k, round(float(v), 3)] for k, v in importances],
        "weights": {"observed": 0.5, "predicted": 0.3, "exposure": 0.2}, "shrink_k": SHRINK_K, "event_mean": {k: round(float(v), 2) for k, v in df.groupby("event_name").risk.mean().items()},
    }
    payload = {"summary": summary, "tiles": tiles, "neighborhoods": nb.to_dict("records"),
               "watersheds": ws.to_dict("records"), "streets": street_rank, "crossings": crossings}
    DASH.mkdir(exist_ok=True)
    (DASH / "data.js").write_text("window.FLOOD_DATA = " + json.dumps(payload, default=str) + ";\n", encoding="utf-8")
    df.drop(columns=["source_tile_urls_json"], errors="ignore")[
        ["image_id", "event_name", "image_view_type", "locality", "latitude", "longitude", "risk", "event_pct", "tier", "observed", "predicted", "exposure",
         "severity", "roads_flooded", "low_water_crossing", "flooded_road_desc", "notes"]
    ].to_csv(HERE / "tile_risk.csv", index=False)
    nb.to_csv(HERE / "neighborhood_rank.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "importances"}))
    print(nb[["rank", "locality", "score", "raw_score", "tiles", "flooded_share", "critical_tiles", "lwc_tiles"]].to_string(index=False))


if __name__ == "__main__":
    main()
