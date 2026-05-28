from pathlib import Path
from typing import Any

from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image and convert it to RGB."""

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = Image.open(path).convert("RGB")
    return image


def get_image_info(image: Image.Image, image_path: str | Path) -> dict[str, Any]:
    """Return basic image metadata."""

    width, height = image.size

    return {
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "mode": image.mode,
    }
