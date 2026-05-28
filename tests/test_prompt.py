from world_vla_graspqa.utils.prompt import (
    build_scene_description,
    format_object_description,
)


def test_format_object_description_includes_key_fields():
    obj = {
        "object_id": "obj_001",
        "name": "red cube",
        "color": "red",
        "shape": "cube",
        "graspable": True,
        "bbox": [10, 20, 30, 40],
    }

    text = format_object_description(obj)

    assert "obj_001" in text
    assert "red cube" in text
    assert "color=red" in text
    assert "shape=cube" in text
    assert "graspable=True" in text
    assert "bbox=[10, 20, 30, 40]" in text


def test_build_scene_description_includes_objects():
    annotation = {
        "scene_id": "test_scene",
        "description": "A test tabletop scene.",
        "objects": [
            {
                "object_id": "obj_001",
                "name": "red cube",
                "bbox": [10, 20, 30, 40],
                "attributes": {
                    "color": "red",
                    "shape": "cube",
                    "graspable": True,
                },
            },
            {
                "object_id": "obj_002",
                "name": "blue bowl",
                "bbox": [40, 50, 80, 90],
                "attributes": {
                    "color": "blue",
                    "shape": "bowl",
                    "graspable": False,
                },
            },
        ],
        "relations": [],
    }

    text = build_scene_description(annotation)

    assert "Scene test_scene contains 2 objects." in text
    assert "Scene description: A test tabletop scene." in text
    assert "obj_001: red cube" in text
    assert "obj_002: blue bowl" in text
    assert "graspable=True" in text
    assert "graspable=False" in text
