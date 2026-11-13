import base64
import io
from typing import Literal, Optional, Tuple
from PIL import Image, ImageOps

def preprocess_image_to_data_url(
    path: str,
    *,
    max_side: int = 1024,
    format: Literal["JPEG", "PNG"] = "JPEG",
    jpeg_quality: int = 90,
    pad_to_square: bool = False,
    square_fill: Tuple[int, int, int] = (0, 0, 0),
) -> str:
    """
    Load an image from disk, normalize it (EXIF orientation, RGB),
    resize so the long side <= max_side, optionally pad to square,
    encode as JPEG/PNG bytes, and return a data: URL.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # fix phone rotations, etc.
    img = img.convert("RGB")

    # Resize (keep aspect ratio)
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_side:
        scale = max_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Optional: pad to square
    if pad_to_square:
        w, h = img.size
        side = max(w, h)
        padded = Image.new("RGB", (side, side), square_fill)
        left = (side - w) // 2
        top = (side - h) // 2
        padded.paste(img, (left, top))
        img = padded

    # Encode
    buf = io.BytesIO()
    save_kwargs = {}
    if format == "JPEG":
        save_kwargs.update({"quality": jpeg_quality, "optimize": True})
    img.save(buf, format=format, **save_kwargs)

    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = "image/jpeg" if format == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"
