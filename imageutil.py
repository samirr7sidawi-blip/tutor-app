"""Prepare uploaded photos for the vision models.

Downscales large phone photos and re-encodes them as JPEG base64, in one neutral
shape each provider converts to its own SDK format:
``{"media_type": "image/jpeg", "data": "<base64>"}``.
"""
import base64
import io

from PIL import Image

MAX_EDGE = 1568  # long-edge cap that keeps images within the vision API limits


def prepare_image(file) -> dict:
    """Read a Streamlit UploadedFile, downscale if huge, return base64 JPEG dict."""
    file.seek(0)
    img = Image.open(file).convert("RGB")

    w, h = img.size
    scale = MAX_EDGE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {"media_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode("ascii")}
