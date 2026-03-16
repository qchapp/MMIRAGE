from pathlib import Path
import base64, json
from typing import List, Optional, Tuple

import tqdm


def encode_image_to_base64(path: Path) -> str:
    """
    Read an image from disk and return base64-encoded string.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_media_type(path: Path) -> str:
    """
    Get the media type (MIME type) of a file based on its extension.
    """
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


