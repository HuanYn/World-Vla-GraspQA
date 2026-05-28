from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from world_vla_graspqa.utils.io import ensure_dir


def draw_scene_annotation(
    image: Image.Image,
    annotation: dict[str, Any],
) -> Image.Image:
    """Draw object bounding boxes and names on a scene image."""

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for obj in annotation.get("objects", []):
        bbox = obj.get("bbox")
        name = obj.get("name", "unknown")

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 16)), name, fill=(255, 0, 0))

    return annotated_image


def save_annotated_scene(
    image: Image.Image,
    annotation: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save a scene image with annotation boxes drawn on it."""

    path = Path(output_path)
    ensure_dir(path.parent)

    annotated_image = draw_scene_annotation(image, annotation)
    annotated_image.save(path)
