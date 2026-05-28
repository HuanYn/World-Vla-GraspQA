import json

import pytest

from world_vla_graspqa.utils.scene import (
    get_scene_objects,
    load_scene_annotation,
    normalize_scene_object,
    validate_bbox,
)


def test_load_scene_annotation_reads_json(tmp_path):
    annotation_path = tmp_path / "scene.json"
    data = {
        "scene_id": "test_scene",
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
            }
        ],
    }

    annotation_path.write_text(json.dumps(data), encoding="utf-8")

    annotation = load_scene_annotation(annotation_path)

    assert annotation["scene_id"] == "test_scene"
    assert annotation["objects"][0]["name"] == "red cube"


def test_load_scene_annotation_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_scene_annotation(missing_path)


def test_normalize_scene_object_flattens_attributes():
    obj = {
        "object_id": "obj_003",
        "name": "yellow banana",
        "bbox": [10, 20, 60, 80],
        "attributes": {
            "color": "yellow",
            "shape": "banana",
            "graspable": True,
        },
    }

    normalized = normalize_scene_object(obj)

    assert normalized["object_id"] == "obj_003"
    assert normalized["name"] == "yellow banana"
    assert normalized["color"] == "yellow"
    assert normalized["shape"] == "banana"
    assert normalized["graspable"] is True
    assert normalized["bbox"] == [10, 20, 60, 80]


def test_get_scene_objects_returns_normalized_objects():
    annotation = {
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
            }
        ]
    }

    objects = get_scene_objects(annotation)

    assert len(objects) == 1
    assert objects[0]["name"] == "red cube"
    assert objects[0]["color"] == "red"
    assert objects[0]["graspable"] is True


def test_get_scene_objects_raises_for_invalid_objects():
    annotation = {
        "objects": "not-a-list",
    }

    with pytest.raises(ValueError):
        get_scene_objects(annotation)


def test_validate_bbox_raises_for_invalid_bbox():
    with pytest.raises(ValueError):
        validate_bbox([10, 20, 5, 40])
