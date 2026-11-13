# HydroAlert: flash flood risk dashboard

Interactive map that flags high-risk areas, predicts flood-prone streets and
low-water crossings, and ranks neighborhoods, built from the UTSA before/after
flood imagery and `image_metadata.csv`.

Open `dashboard/index.html` in a browser (no server needed). Append
`#tile=<image_id>` to the URL to open a specific tile.

## Pipeline

```bash
python flood_risk/score_images.py           # 1. Qwen3-VL scores every before/after pair  -> vlm_scores.jsonl
python flood_risk/fetch_tiger.py            # 2. Streets + road-creek crossings per tile from Census TIGER/Line (~7 s, offline)
                                            #    (needs flood_risk/tiger/*.zip, see the script's docstring;
                                            #     fetch_osm.py is a slower OpenStreetMap/Overpass alternative)
python flood_risk/build_risk.py             # 3. risk model + rankings -> dashboard/data.js, thumbs/, CSVs
```

All steps cache their results and are safe to re-run. Requires `pandas`,
`openai`, `pillow`, `scikit-learn`, and network access to the GB10 endpoint
(`http://10.246.100.230/v1`, model `qwen3-vl-8b-instruct`).

## How risk is scored

Each georeferenced response tile (~530 m square) gets a 0–100 score:

| Component | Weight | Source |
|---|---|---|
| Observed impact | 0.5 | Qwen3-VL compares BEFORE (2016 NAIP) vs AFTER (2017 NOAA) and returns severity 0–5, % new water, flooded roads, buildings affected |
| Predicted susceptibility | 0.3 | Gradient-boosted model on static features only (elevation relative to neighborhood, relief, slope, FEMA zone, waterways, high-water marks). Out-of-fold predictions from spatial GroupKFold over 0.05° blocks, so each prediction comes from a model that never saw that block. |
| Exposure & vulnerability | 0.2 | OSM road and building density, plus county ACS poverty, no-vehicle, 65+, and disability rates |

Tiers: critical ≥70, high ≥50, moderate ≥30, low <30.

- **Flood-prone streets**: named OSM streets in tiles where Qwen saw flooded
  roads (*observed*) or the terrain model gives ≥60% flood probability
  (*predicted*).
- **Low-water crossings**: OSM fords, creek culverts, and road bridges inside
  flooded or flood-prone tiles, plus crossings Qwen reported overtopped. VLM-only
  detections are noisier and shown in a separate color.
- **Neighborhood rank**: 0.6 × mean + 0.4 × 90th-percentile tile risk, shrunk
  toward the event mean by 10 pseudo-tiles so small samples don't dominate.
  Also ranked by HUC-12 watershed.

## Caveats

- The VLM labels are model-generated screening, not verified ground truth. The
  dataset's `flood_visible` human labels are blank.
- 93% of Harvey response tiles show some flooding, so risk mostly separates
  degree and location rather than flood vs. no flood.
- `hwm_within_25km` is the top predictor. It partly encodes regional event
  intensity, so treat the susceptibility model as event-conditional.
- Hill Country 2025 tiles have no baseline pair and are scored from a single
  image. They were flown July 11–22, after the July 4 peak, so they show
  aftermath and score lower partly for that reason. The dashboard flags them and
  shows each tile's percentile within its own event (`event_pct`); compare tiles
  within an event, not across events. The 4 USCG frames have no coordinates and
  are not mapped.
- A blank FEMA field means "unknown", not "outside the floodplain". The model
  gets an explicit `fema_missing` indicator (26 Harvey tiles) rather than
  treating blanks as zone X.
- FEMA, OSM and TIGER layers are current snapshots, not maps in effect on the
  flood dates. County ACS rates describe whole counties, not households or
  neighborhoods.

Outputs: `tile_risk.csv` (per tile) and `neighborhood_rank.csv`.
