"""Small reusable helpers for the three Texas flood teaching notebooks."""
import base64
import json
import mimetypes
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from PIL import Image

from vlm_config import API_KEY, BASE_URL, MODEL

DATA_ROOT = Path(__file__).resolve().parent.parent
COLLECTION = "Texas_Flood_Annotation"


def load_texas_images(image_ids):
    """Return the requested Texas images and their metadata, in the given order."""
    folder = DATA_ROOT / "Texas_Flood_Student_Data"
    if (folder / "image_metadata.csv").is_file():
        images = pd.read_csv(folder / "image_metadata.csv", keep_default_na=False)
        path_column = "image_path"
    else:
        folder = DATA_ROOT / COLLECTION
        images = pd.DataFrame(json.loads((folder / "manifest.json").read_text()))
        path_column = "filename"
    images = images.loc[images["state"] == "Texas"].set_index("image_id", verify_integrity=True)
    selected = images.loc[list(image_ids)].reset_index()
    selected["path"] = selected[path_column].map(lambda filename: folder / filename)
    for path in selected["path"]:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
    return selected


def show_images(images):
    """Display a small image gallery, with dates and optional retrieval scores."""
    if images.empty:
        print("No images matched the query words. Try a different query.")
        return
    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, images.iterrows()):
        with Image.open(row["path"]) as image:
            ax.imshow(image)
        label = row.get("label", row["image_id"])
        title = f"{label} · {row['acquisition_date']}\n{row['locality']}"
        if "score" in row:
            title += f"\nSimilarity: {row['score']:.3f}"
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def image_data_url(path):
    """Encode a local PNG/JPEG/WebP as an image URL the VLM API can accept."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def ask_images(image_paths, question):
    """Ask Qwen one question about one or more images, in the supplied order."""
    paths = list(image_paths)
    if not paths:
        raise ValueError("Supply at least one image.")
    content = [{"type": "text", "text": question}]
    for path in paths:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
    with OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120, max_retries=0) as client:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            top_p=0.8,
            presence_penalty=0,
            max_tokens=300,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {"role": "system", "content": "Use visible image evidence. Be concise and state uncertainty."},
                {"role": "user", "content": content},
            ],
        )
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise ValueError("The model returned no text.")
    return answer


def ask_image(image_path, question):
    """Ask Qwen one question about a single image."""
    return ask_images([image_path], question)


def load_rainfall(image_id, days_before=14):
    """Load nearby daily rain before a Texas image date; preserve missing days."""
    folder = DATA_ROOT / "Texas_Flood_Student_Data"
    if (folder / "image_metadata.csv").is_file():
        image = load_texas_images([image_id]).iloc[0]
        image_date = pd.Timestamp(image["acquisition_date"])
    else:
        context = pd.read_csv(DATA_ROOT / "Texas_Flood_Hackathon_Extras/image_context.csv")
        image_key = f"{COLLECTION}/{image_id}"
        image = context.set_index("image_key", verify_integrity=True).loc[image_key]
        image_date = pd.Timestamp(image["image_date"])
    days = pd.date_range(end=image_date - pd.Timedelta(days=1), periods=days_before)
    if (folder / "image_metadata.csv").is_file():
        database = (folder / "observations.sqlite").as_uri() + "?mode=ro"
        with sqlite3.connect(database, uri=True, timeout=30) as connection:
            rain = pd.read_sql_query(
                'SELECT DATE, PRCP, PRCP_ATTRIBUTES FROM weather_daily_weather '
                'WHERE window_id = ? AND STATION = ? AND DATE >= ? AND DATE < ?',
                connection,
                params=[image["weather_window_id"], image["precipitation_station_id"],
                        str(days.min().date()), str(image_date.date())],
            )
    else:
        weather = pd.read_csv(
            DATA_ROOT / "Texas_Flood_Weather/daily_weather.csv",
            usecols=["window_id", "STATION", "DATE", "PRCP", "PRCP_ATTRIBUTES"],
            keep_default_na=False,
        )
        rain = weather.loc[
            (weather["window_id"] == image["weather_window_id"])
            & (weather["STATION"] == image["precipitation_station_id"])
        ].copy()
    rain["date"] = pd.to_datetime(rain["DATE"])
    flags = rain["PRCP_ATTRIBUTES"].fillna("").str.split(",")
    measurement_flag, quality_flag = flags.str[0].fillna(""), flags.str[1].fillna("")
    # Reject failed quality checks and source-presumed/unreported zeros.
    accepted = quality_flag.eq("") & ~measurement_flag.isin(["P", "I"])
    rain["rain_mm"] = pd.to_numeric(rain["PRCP"], errors="coerce").where(accepted)
    rain["is_trace"] = measurement_flag.eq("T") & rain["rain_mm"].notna()
    if rain["date"].duplicated().any():
        raise ValueError("More than one rainfall record for a station/date.")
    rain = rain.set_index("date")[["rain_mm", "is_trace"]].reindex(days)
    rain.index.name = "date"
    rain.attrs = {"station_id": image["precipitation_station_id"],
                  "distance_km": pd.to_numeric(image["precipitation_station_distance_km"], errors="coerce"),
                  "image_date": image_date}
    return rain


def plot_rainfall(rain):
    """Plot daily rain in mm; mark missing values and trace reports separately."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(rain.index, rain["rain_mm"], color="#276a9b")
    for mask, marker, color, label in [
        (rain["rain_mm"].isna(), "x", "red", "Missing / rejected"),
        (rain["is_trace"].eq(True), "o", "orange", "Trace"),
    ]:
        if mask.any():
            ax.scatter(rain.index[mask], [0] * mask.sum(), marker=marker, color=color, label=label)
    ax.set_ylabel("Daily rainfall (mm)")
    ax.set_title(f"Rain before {rain.attrs['image_date'].date()} · station {rain.attrs['station_id']} "
                 f"({rain.attrs['distance_km']:.1f} km away)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
