import json

import yaml

from world_vla_graspqa.utils.io import (
    create_run_dir,
    ensure_dir,
    save_json,
    save_yaml,
)


def test_ensure_dir_creates_directory(tmp_path):
    output_dir = tmp_path / "new_outputs"

    created_dir = ensure_dir(output_dir)

    assert created_dir.exists()
    assert created_dir.is_dir()


def test_create_run_dir_creates_timestamped_directory(tmp_path):
    run_dir = create_run_dir(tmp_path, run_name="dummy")

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    assert run_dir.name.endswith("_dummy")


def test_save_json_writes_file_and_creates_parent_dir(tmp_path):
    output_path = tmp_path / "nested" / "result.json"
    data = {
        "target_object": "red cube",
        "best_action": {
            "name": "grasp(red cube)",
            "gripper_pose": "top_down",
            "predicted_success": 0.85,
        },
    }

    save_json(data, output_path)

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    assert loaded_data == data


def test_save_yaml_writes_file_and_creates_parent_dir(tmp_path):
    output_path = tmp_path / "nested" / "config.yaml"
    data = {
        "project": {
            "name": "World-VLA-GraspQA",
            "stage": "dummy_pipeline",
        }
    }

    save_yaml(data, output_path)

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        loaded_data = yaml.safe_load(f)

    assert loaded_data == data
