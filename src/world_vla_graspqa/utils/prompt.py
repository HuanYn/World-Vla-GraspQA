from typing import Any

from world_vla_graspqa.utils.scene import get_scene_objects


def format_object_description(obj: dict[str, Any]) -> str:
    """Format one object annotation as a text description."""

    object_id = obj.get("object_id", "")
    name = obj.get("name", "unknown object")
    color = obj.get("color", "unknown")
    shape = obj.get("shape", "unknown")
    graspable = obj.get("graspable", False)
    bbox = obj.get("bbox", None)

    prefix = f"{object_id}: " if object_id else ""

    return (
        f"{prefix}{name}, "
        f"color={color}, "
        f"shape={shape}, "
        f"graspable={graspable}, "
        f"bbox={bbox}"
    )


def build_scene_description(annotation: dict[str, Any]) -> str:
    """Build a text description from a scene annotation."""

    scene_id = annotation.get("scene_id", "unknown_scene")
    description = annotation.get("description", "")
    objects = get_scene_objects(annotation)

    lines = [
        f"Scene {scene_id} contains {len(objects)} objects.",
    ]

    if description:
        lines.append(f"Scene description: {description}")

    lines.append("Objects:")

    for obj in objects:
        lines.append(f"- {format_object_description(obj)}")

    relations = annotation.get("relations", [])
    if relations:
        lines.append("Relations:")
        for relation in relations:
            lines.append(f"- {relation}")

    return "\n".join(lines)
