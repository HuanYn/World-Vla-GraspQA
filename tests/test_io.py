import json

from world_vla_graspqa.utils.io import ensure_dir, save_json


def test_ensure_dir_creates_directory(tmp_path):
    output_dir = tmp_path / "new_outputs"

    created_dir = ensure_dir(output_dir)

    assert created_dir.exists()
    assert created_dir.is_dir()


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
