"""Score before/after flood imagery with Qwen3-VL and cache results as JSONL.

Each paired location (before = 2016 NAIP baseline, after = 2017 NOAA response)
is sent to the VLM together; unpaired response images (Hill Country 2025,
USCG frames) are scored alone. Output: flood_risk/vlm_scores.jsonl, one row
per *response* image. Re-running skips rows already scored.

Usage:
    python flood_risk/score_images.py              # score everything
    python flood_risk/score_images.py --limit 10   # quick smoke test
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "UTSA-flood-images" / "images"
OUT = Path(__file__).resolve().parent / "vlm_scores.jsonl"

client = OpenAI(base_url="http://10.246.100.230/v1", api_key="usaadatathon")
MODEL = "qwen3-vl-8b-instruct"

SCHEMA_TEXT = """Return ONLY a JSON object with exactly these keys:
{
  "flood_visible": "yes" | "no" | "uncertain",
  "standing_water_pct": integer 0-100,      // % of AFTER image covered by flood water (not normal rivers/ponds)
  "new_water_vs_before_pct": integer 0-100, // % of area that is water in AFTER but dry in BEFORE (0 if no BEFORE)
  "roads_flooded": "none" | "some" | "many",
  "flooded_road_desc": string,               // short: which roads/streets/intersections look submerged, e.g. "residential cul-de-sacs in NE quadrant"
  "low_water_crossing": "yes" | "no" | "uncertain", // a road crossing a creek/bayou/channel that is overtopped or at risk
  "buildings_affected": "none" | "few" | "many",
  "debris_or_scour": "yes" | "no",
  "severity": integer 0-5,                   // 0 no flooding .. 5 catastrophic, widespread inundation
  "confidence": number 0-1,
  "notes": string                            // one sentence of evidence
}"""

PAIR_PROMPT = (
    "You are a flood damage analyst reviewing orthorectified aerial imagery of the same "
    "ground footprint. Image A is BEFORE (dry baseline). Image B is AFTER a flash-flood event. "
    "Compare them. Brown/muddy water covering streets, yards, fields or around buildings in B "
    "that is not present in A indicates flooding. Do not count permanent ponds, pools or rivers "
    "that also appear in A.\n" + SCHEMA_TEXT
)
SINGLE_PROMPT = (
    "You are a flood damage analyst reviewing a single aerial image taken during or after a "
    "flash flood (no baseline image is available). Assess visible inundation, scoured banks, "
    "debris and damage to roads/crossings. Set new_water_vs_before_pct to 0.\n" + SCHEMA_TEXT
)


def encode(path: Path, max_side: int = 896) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    raw = re.sub(r"//[^\n]*", "", m.group(0))  # strip any echoed comments
    return json.loads(raw)


def ask(paths: list[Path], prompt: str, retries: int = 3) -> dict:
    content = [{"type": "text", "text": prompt}]
    for label, p in zip("AB", paths) if len(paths) == 2 else [("", paths[0])]:
        if label:
            content.append({"type": "text", "text": f"Image {label}:"})
        content.append({"type": "image_url", "image_url": {"url": encode(p)}})
    last = None
    for _ in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=400,
                temperature=0.1,
            )
            return parse_json(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001 - retry any API/parse failure
            last = e
    raise RuntimeError(f"failed after {retries} tries: {last}")


def build_jobs(df: pd.DataFrame) -> list[dict]:
    by_id = df.set_index("image_id")
    jobs = []
    for _, r in df[df.image_role == "response"].iterrows():
        after = ROOT / "UTSA-flood-images" / r.image_path
        before_id = r.paired_image_id if isinstance(r.paired_image_id, str) else None
        if before_id and before_id in by_id.index:
            before = ROOT / "UTSA-flood-images" / by_id.loc[before_id, "image_path"]
            jobs.append({"image_id": r.image_id, "before_id": before_id, "paths": [before, after]})
        else:
            jobs.append({"image_id": r.image_id, "before_id": None, "paths": [after]})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "image_metadata.csv", low_memory=False)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["image_id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    jobs = [j for j in build_jobs(df) if j["image_id"] not in done][: args.limit]
    print(f"{len(done)} already scored, {len(jobs)} to go", flush=True)

    lock = threading.Lock()

    def run(job):
        prompt = PAIR_PROMPT if job["before_id"] else SINGLE_PROMPT
        res = ask(job["paths"], prompt)
        row = {"image_id": job["image_id"], "before_id": job["before_id"], **res}
        with lock, OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    n_ok = n_err = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                row = fut.result()
                n_ok += 1
                if n_ok % 25 == 0 or n_ok <= 3:
                    print(f"[{n_ok}/{len(jobs)}] {row['image_id']} sev={row.get('severity')} "
                          f"water={row.get('standing_water_pct')}% roads={row.get('roads_flooded')}", flush=True)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"ERROR {futs[fut]['image_id']}: {e}", file=sys.stderr, flush=True)
    print(f"done: {n_ok} scored, {n_err} errors")


if __name__ == "__main__":
    main()
