TEXAS FLOOD STUDENT DATA
Consolidated November 10, 2026

START HERE
This folder contains everything from the collected Texas imagery and its
weather, river, hazard, terrain, infrastructure, community and historical data
in a small number of files. There is no original FloodNet data in this package.

Texas_Flood_Student_Data/
  images/                  1,204 original images; one flat folder
  image_metadata.csv       One row per image, including matched context
  observations.sqlite      Full tables, historical records, maps and source assets
  README.txt               This guide and short Python examples
  DATA_DICTIONARY.txt       Field meanings, units, table catalog and join guidance

Start with the CSV. Use the database when you want time series, detailed
geography, historical comparisons or original source records. SQLite support
is built into Python; it needs no database server or account.

The CSV has 273 columns to retain provenance and useful context. You only need
a few for a first project: image_id, image_path, paired_image_id, acquisition_date,
locality, source_agency, latitude, longitude and precipitation_mm.

IMAGES
All 1,204 images are from Texas: 625 NOAA, 575 USDA NAIP and 4 USCG/DVIDS.
There are 575 Harvey before/after pairs (1,150 images), 50 unpaired Hill Country
NOAA aftermath images and 4 unpaired USCG views of visible flood conditions.
All image bytes are unchanged. Filenames are now image_id plus the original
extension. Paths in image_metadata.csv are relative to this folder.

Acquisition dates cover USDA baselines in 2016, Harvey response in 2017, and
Hill Country flooding in July 2025. These are aircraft images, not satellite
images. NOAA/USDA image bounds are WGS84 (EPSG:4326); their image grids are
Web Mercator (EPSG:3857). PNGs do not contain embedded georeferencing.

1,200 images have precise geographic footprints. The four USCG views have
caption-level locations near Kerrville only; their coordinates and bounds
are blank. Three frames come from the same rescue video and have the same
dependence_group. Video offsets are not capture clock times.

Hill Country NOAA images were captured July 11, 12 and 22, after the July 4
flood peak. The USDA images have 1 m native resolution even when the matching
output pixels are about 0.52 m apart. Upsampling adds no ground detail.

PAIRS AND ANNOTATIONS
Use paired_image_id to find a companion image, or group by pair_key. Both
images share a geographic grid; fine registration and appearance can differ.
Keep pairs, nearby footprints and frames from the same video together when
making training/test splits. image_id is unique in this release; image_key
preserves the original collection/image_id key used by the database tables.

flood_visible, flood_aftermath_visible, annotator and annotation_notes are blank
fields for your annotations. Suggested labels are yes, no and uncertain.
image_role, image_view_type and candidate_condition are collection roles or
screening suggestions, not verified labels. Existing annotation templates were
blank. Preserve uncertainty when the pixels do not establish an answer.

WHAT IS IN THE DATABASE?
80 original prepared CSV tables retain 5,929,801 rows and every original cell.
These include observations, derived summaries, links, catalogs, dictionaries
and annotation templates; this is not a count of independent observations.
Event-window and historical tables overlap. Do not add them together without
checking source IDs, station, date, parameter, series and reporting period.

  weather_*   Event-window daily/surface weather, station links and storm reports
  extras_*    River observations, high-water marks, terrain, infrastructure,
              county statistics, coastal levels and storm-track context
  history_*   Long-term weather, hydrology, storm reports and hurricane tracks

Prepared historical weather covers 1980-2025; daily river observations cover
2000-2025. Official peak records extend back to 1869, and Texas storm reports
cover 1996-2025. Coverage varies by station and field; inspect coverage tables.
Original station files retain older/later records, including weather as early
as 1878, in binary_assets. Original source archives can cover broader areas
than the prepared Texas-focused tables. This package consolidates the existing
downloads; it does not claim exhaustive records for every Texas location.

Support tables:
  dataset_catalog       Table names, row counts, source paths, coverage and joins
  schema_fields         Column definitions, units and source documentation
  image_metadata        Initial snapshot of the supplied CSV for SQL joins
  image_source_metadata Original image manifest and download records as JSON
  spatial_datasets      10 original GeoJSON datasets and collection metadata
  spatial_features      183,091 features with GeoJSON geometry/properties
  binary_assets         Terrain rasters, original downloads and source provenance
  source_documents      Original README/documentation text
  package_info          Consolidation counts and validation summary

The database retains 26 original DEM/slope GeoTIFFs and original weather,
hydrology and hazard downloads inside binary_assets. You do not need to extract
them for ordinary table queries. This keeps maps and source files available
without adding more student-facing folders.

CSV and SQLite image_metadata are the same initial snapshot. Editing the CSV
does not automatically edit the database. Use the CSV for new annotations.

DATA TYPES AND MISSING VALUES
Imported database cells are TEXT to preserve station codes, leading zeros,
quality flags, decimal strings and original date labels exactly. A source blank
remains the empty string '', not zero or SQL NULL. Convert numeric columns
explicitly in pandas with pd.to_numeric(..., errors="coerce"). In SQL use
CAST(NULLIF(value_column, '') AS REAL), rather than casting blanks to zero.

KEEP THE MEANING OF THE DATA
- Nearby station readings are context, not measurements at each image pixel.
  Four unlocated USCG images use explicitly regional Kerrville weather only.
- Daily weather dates are station reporting-day labels, not exact UTC image
  times. Surface timestamps retain their documented UTC/local meanings.
- Retain quality flags and coverage. Strict preceding-rainfall totals exclude
  image day and remain blank if any required day is invalid/missing. The
  original image-feature and historical quality rules differ; see the source
  dictionaries. Do not re-scale already metric GHCN API values.
- The database preserves flagged values as well as valid ones. Do not treat
  every numeric observation as a quality-approved reading. Never sum rolling
  or overlapping precipitation accumulation intervals without resolving them.
- River stage, NOAA coastal levels, terrain and surveyed high-water elevations
  use different datums. Subtracting them directly does not give flood depth.
- OSM, FEMA and terrain layers include current snapshots, not necessarily maps
  available on the image date. County ACS estimates describe counties, not
  individual households. Release dates and margins of error are retained.
- A missing FEMA polygon does not establish low flood risk. Source boundaries,
  station distances, timestamps and reporting coverage matter for comparisons.

PYTHON: OPEN AN IMAGE
Run from the parent directory containing Texas_Flood_Student_Data, or set data
to its full path. Python packages for these examples: pandas and pillow.

    from pathlib import Path
    import pandas as pd
    from PIL import Image

    data = Path("Texas_Flood_Student_Data")
    images = pd.read_csv(data / "image_metadata.csv", dtype="string",
                         keep_default_na=False)
    row = images.loc[images["image_id"] == "TX001_2017_NOAA"].iloc[0]
    display(Image.open(data / row["image_path"]))
    pair = images.loc[images["image_id"] == row["paired_image_id"]]

If you are already inside the data folder, use data = Path(".").

PYTHON: FIND AND LOAD TABLES

    import sqlite3
    connection = sqlite3.connect(
        (data / "observations.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    catalog = pd.read_sql_query(
        "SELECT table_name, row_count, description FROM dataset_catalog",
        connection)
    display(catalog)

    rain = pd.read_sql_query(
        "SELECT DATE, PRCP, PRCP_ATTRIBUTES FROM weather_daily_weather "
        "WHERE window_id = ? AND STATION = ? ORDER BY DATE",
        connection,
        params=(row["weather_window_id"], row["precipitation_station_id"]))
    rain["PRCP"] = pd.to_numeric(rain["PRCP"], errors="coerce")
    # Inspect PRCP_ATTRIBUTES before computing rainfall statistics.

PYTHON: LOAD HISTORICAL WEATHER
The historical reference station can differ from the event-window station.
Use historical_weather_station_id from image_metadata.csv for this query.

    history = pd.read_sql_query(
        "SELECT date, PRCP, PRCP_FLAGS FROM history_weather_historical_daily_weather "
        "WHERE station_id = ? AND date BETWEEN ? AND ? ORDER BY date",
        connection,
        params=(row["historical_weather_station_id"], "2010-01-01", "2017-12-31"))
    history["PRCP"] = pd.to_numeric(history["PRCP"], errors="coerce")
    # Check PRCP_FLAGS and missing values before analysis.

PYTHON: FIND MAPS OR ORIGINAL SOURCE FILES

    maps = pd.read_sql_query(
        "SELECT source_path, feature_count FROM spatial_datasets", connection)
    assets = pd.read_sql_query(
        "SELECT source_path, asset_kind, source_bytes FROM binary_assets "
        "WHERE mime_type = 'image/tiff'", connection)
    display(maps)
    display(assets)
    connection.close()

Original source_path values are lookup keys in the database, not extra folders
you need to download. Source CSV references may be relative to their original
domain directory; prepend the source table's package/domain path from the
catalog to locate an asset. Use image_path to open an included image.

USING THE TEACHING NOTEBOOKS
The updated Flood_Hackathon_Notebooks helper can read this layout directly.
Place that notebook folder beside Texas_Flood_Student_Data. Notebook helpers
use this CSV for images and the SQLite daily weather table for rainfall.
The notebooks remain a separate download so the data folder stays minimal.

SOURCES AND REUSE
Image credits and source URLs are retained per row and in image_source_metadata.
NOAA imagery: NOAA National Geodetic Survey, CC0-1.0 as recorded in the source
metadata, with original use notes retained. USDA NAIP: original FGDC records
state no use constraints and request attribution; Microsoft Planetary Computer
hosts the public assets. USCG/DVIDS items are marked public domain; retain the
recorded photographer/video credit. Government imagery does not imply endorsement.

OSM-derived infrastructure data: copyright OpenStreetMap contributors, ODbL1.0.
Retain attribution and applicable share-alike terms for derived databases.
https://www.openstreetmap.org/copyright
https://opendatacommons.org/licenses/odbl/1-0/
Other records retain their NOAA, USGS, FEMA, Census and source-provider credits,
URLs, source dates and documentation inside the database. This combined package
does not impose a blanket license that overrides those source terms.

CHECKS PERFORMED
All image files match their original SHA-256 checksums and decode correctly.
No duplicate image bytes or decoded RGB images were found. All 575 pair links
are reciprocal with matching bounds. Every imported CSV cell, map feature and
binary asset was checked against its source. The consolidated CSV was checked
cell-by-cell after export, and SQLite integrity checks passed. No flood labels
or missing observations were invented during consolidation.
