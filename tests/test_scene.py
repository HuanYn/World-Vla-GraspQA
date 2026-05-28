import json

import pytest

from world_vla_graspqa.utils.scene import get_scene_objects, load_scene_annotation


def test_load_scene_annotation_reads_json(tmp_path):
    annotation_path = tmp_path / "scene.json"
    data = {
        "scene_id": "test_scene",
        "objects": [
            {
                "name": "red cube",
                "graspable": True,
                "bbox": [10, 20, 30, 40],
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


def test_get_scene_objects_returns_objects():
    annotation = {
        "objects": [
            {
                "name": "yellow banana",
                "graspable": True,
            }
        ]
    }

    objects = get_scene_objects(annotation)

    assert len(objects) == 1
    assert objects[0]["name"] == "yellow banana"


def test_get_scene_objects_raises_for_invalid_objects():
    annotation = {
        "objects": "not-a-list",
    }

    with pytest.raises(ValueError):
        get_scene_objects(annotation)
