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


def get_scene_objects(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return object annotations from a scene annotation."""

    objects = annotation.get("objects", [])

    if not isinstance(objects, list):
        raise ValueError("Scene annotation field 'objects' must be a list.")

    return objects
