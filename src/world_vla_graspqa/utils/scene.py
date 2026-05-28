import json
from pathlib import Path
from typing import Any


def load_scene_annotation(annotation_path: str | Path) -> dict[str, Any]:
    """Load a scene annotation JSON file."""

    path = Path(annotation_path)

    if not path.exists():
        raise FileNotFoundError(f"Scene annotation file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    return annotation


def validate_bbox(bbox: Any) -> None:
    """Validate a bounding box in [x1, y1, x2, y2] format."""

    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("Object bbox must be a list with four values.")

    if not all(isinstance(value, int | float) for value in bbox):
        raise ValueError("Object bbox values must be numeric.")

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Object bbox must satisfy x2 > x1 and y2 > y1.")


def normalize_scene_object(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize a scene object into the flat format used by the dummy pipeline."""

    attributes = obj.get("attributes", {})

    normalized = {
        "object_id": obj.get("object_id", ""),
        "name": obj["name"],
        "bbox": obj.get("bbox"),
        "color": obj.get("color", attributes.get("color")),
        "shape": obj.get("shape", attributes.get("shape")),
        "graspable": obj.get("graspable", attributes.get("graspable", False)),
    }

    if normalized["bbox"] is not None:
        validate_bbox(normalized["bbox"])

    return normalized


def get_scene_objects(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized object annotations from a scene annotation."""

    objects = annotation.get("objects", [])

    if not isinstance(objects, list):
        raise ValueError("Scene annotation field 'objects' must be a list.")

    return [normalize_scene_object(obj) for obj in objects]
