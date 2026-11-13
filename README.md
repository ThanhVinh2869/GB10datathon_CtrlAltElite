# HydroAlert: User Guide

HydroAlert maps flash-flood risk across Texas neighborhoods hit by **Hurricane Harvey (2017)** and the **July 2025 Hill Country (Guadalupe River) flood**. An AI vision model compared aerial photos taken before and after each flood. Its results were combined with terrain, FEMA flood zones, road maps and community data to score every area from 0 to 100. Use the dashboard to find the neighborhoods, streets and low-water crossings most likely to flood.

---

## 1. Opening the dashboard

1. Open the `dashboard` folder.
2. Double-click **`index.html`**. It opens in your web browser. No install or server is needed.
3. Stay connected to the internet so the base map and map tools can load.

To open a specific area directly, add `#tile=<image_id>` to the end of the address, for example `index.html#tile=TX001_2017_NOAA`.

---

## 2. Screen layout

| Area | What it shows |
|---|---|
| **Filters** (top bar) | Choose which event, neighborhood and risk levels to show |
| **Summary cards** | Quick counts for everything currently shown |
| **Map** (left) | Risk heatmap, scored areas ("tiles") and low-water crossings |
| **Tabs** (right) | Rankings of neighborhoods, watersheds, streets and crossings, plus details for a selected tile |

Everything is linked. Changing a filter updates the map, the summary cards and every list at once.

---

## 3. Risk levels

Each **tile** is a square of land about 530 m (a third of a mile) on a side, covered by one aerial photo. Each tile gets a risk score from 0 to 100:

| Level | Score | Color |
|---|---|---|
| **Critical** | 70–100 | Red |
| **High** | 50–69 | Orange |
| **Moderate** | 30–49 | Yellow |
| **Low** | 0–29 | Green |

The score combines three things:

- **Observed impact (50%)**: the flooding the AI saw in the photos, such as water, flooded roads and affected buildings.
- **Predicted susceptibility (30%)**: how flood-prone the land is, based on elevation, slope, FEMA zones, nearby waterways and past high-water marks.
- **Exposure (20%)**: how many roads, buildings and vulnerable residents are in the area. Vulnerable here means county rates of poverty, households without a car, residents 65 and older, and disability.

---

## 4. Filters

| Filter | What it does |
|---|---|
| **Event** | Show Harvey only, Hill Country only, or both |
| **Neighborhood** | Zoom to and show a single neighborhood |
| **Tier chips** (Critical / High / Moderate / Low) | Click to hide or show each risk level |
| **Min risk** slider | Hide tiles that score below the chosen value |
| **Heatmap** | Turn the risk heat layer on or off |
| **Tile footprints** | Turn the tile outlines on or off |
| **Low-water crossings** | Turn the crossing markers on or off |

**Tip:** To find the worst areas quickly, turn off Moderate and Low, or move Min risk up to 70.

---

## 5. Using the map

- **Hover** over a tile to see its risk score, level, AI severity rating, percent new water and road flooding.
- **Click** a tile to open it in the **Tile detail** tab.
- **Crossing markers:**
  - **Blue:** a road crossing a creek or bayou, taken from Census road maps.
  - **Purple:** a crossing the AI saw flooded in the photo. These are less reliable, so check the imagery.
- **Base map:** use the layer button at the top right to switch between Light gray, Dark gray, Streets and Satellite.
- The map stays within the flood areas. Choosing an event zooms to that region.

---

## 6. The tabs

### Neighborhoods
Neighborhoods ranked by overall risk.
- **Score** blends the average tile risk with the neighborhood's worst tiles. Neighborhoods with only a few tiles are pulled toward their event's average, so a handful of tiles can't top the list by chance.
- **Flooded** is the share of tiles where the AI saw significant flooding.
- **Critical** and **Crossings** count critical tiles and low-water-crossing tiles.
- **Click a row** to filter the whole dashboard to that neighborhood.

### Watersheds
The same ranking, broken down by creek basin (USGS watershed). Use it to see which creek or bayou drives the risk inside a neighborhood. Click a row to zoom to it.

### Streets
Every named street in the scored tiles, ranked by the highest risk of any tile it passes through. The **Basis** column says why each street is listed:
- **Observed:** the AI saw flooded roads in that tile.
- **Predicted:** the terrain model gives the tile at least a 60% chance of flooding.
- **Not flagged:** neither, but the street is in a scored tile.

Click a row to open the tile.

### Crossings
Places where minor roads cross creeks, bayous or channels. Drivers are most at risk at these crossings during flash floods. **Why flagged** explains the reason: the AI saw the road overtopped, or a mapped crossing sits in a flooded or flood-prone tile. Unflagged crossings are listed but not shown on the map. Click a row to zoom in.

### Tile detail
Shown after you click a tile or any row. It includes:
- **Before and after photos:** for Harvey, a 2016 photo and a 2017 post-flood photo of the same spot. Hill Country tiles have only an after photo.
- **Risk score, level, and rank within its own event.**
- **Score components:** bars for observed impact, predicted susceptibility and exposure.
- **AI findings:** severity (0–5), percent new water, flooded roads, affected buildings, a low-water-crossing note, and the AI's written evidence.
- **Area facts:** elevation compared with nearby tiles, FEMA flood zone, watershed, county and the streets in the tile.

---

## 7. Common tasks

| I want to… | Do this |
|---|---|
| Find the most dangerous neighborhoods | Open the **Neighborhoods** tab. The list is already sorted. |
| See which roads to close or barricade | Open **Crossings** and look at the Critical and High rows |
| Check a specific street | Open **Streets** and scroll or search with Ctrl+F |
| Compare before and after imagery | Click any tile and open **Tile detail** |
| Focus on the 2025 Hill Country flood | Set **Event** to "July 2025 Texas Hill Country flooding" |
| Show only the worst areas | Move **Min risk** to 70 |

---

## 8. Read the results carefully

- **AI ratings are a screening tool, not confirmed fact.** Check important findings against the photos, which are in the Tile detail tab.
- **Hill Country tiles were photographed July 11–22, after the July 4, 2025 peak.** They show aftermath, so their scores are lower partly for that reason. Compare them with other Hill Country tiles, not with Harvey. These tiles show an amber note.
- **"Unknown" FEMA zone does not mean safe.** It means FEMA data is missing for that area.
- **Map layers are current**, not from the flood dates. Roads, FEMA zones and waterways may have changed.
- **Community vulnerability is measured per county**, not per neighborhood or household.
- **No score means no imagery, not no risk.** Only 625 tiles were photographed.
- HydroAlert is a planning and research tool. **Always follow official warnings** from the National Weather Service and local emergency management. Turn around, don't drown.

---

## 9. Data credits

- Aerial imagery: NOAA National Geodetic Survey (2017, 2025); USDA NAIP (2016); U.S. Coast Guard / DVIDS
- Streets and waterways: U.S. Census TIGER/Line 2024
- Terrain: USGS 3DEP · Flood zones: FEMA · River gauges and high-water marks: USGS
- Community data: U.S. Census American Community Survey
- Base maps: © OpenStreetMap contributors, Esri
- AI model: Qwen3-VL-8B, run locally on an NVIDIA GB10

For how the scores are calculated, see [`flood_risk/README.md`](../flood_risk/README.md).
