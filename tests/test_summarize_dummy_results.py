import csv
import json
from pathlib import Path

from world_vla_graspqa.utils.summary import (
    collect_result_paths,
    summarize_result,
    write_summary,
)


def create_dummy_result(path: Path, run_name: str, target_object: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "config_path": f"configs/{run_name}.yaml",
        "run_name": run_name,
        "target_object": target_object,
        "graspqa_result": {
            "mode": "dummy_vlm",
            "raw_response": target_object,
            "parse_success": True,
            "model_name": "dummy-vlm",
        },
        "world_model_info": {
            "mode": "empirical",
            "dataset_path": "data/action_outcomes/dummy_action_outcomes.json",
            "default_success": 0.5,
        },
        "best_action": {
            "name": f"grasp({target_object})",
            "gripper_pose": "top_down",
            "predicted_success": 0.85,
            "world_model_type": "empirical",
        },
        "closed_loop_result": {
            "final_success": True,
            "num_attempts": 1,
            "final_action": {
                "name": f"grasp({target_object})",
                "gripper_pose": "top_down",
            },
        },
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def test_collect_result_paths_finds_result_json_files(tmp_path):
    result_root = tmp_path / "outputs" / "dummy_pipeline"

    create_dummy_result(
        result_root / "20260527_000001_cube" / "result.json",
        run_name="cube",
        target_object="red cube",
    )
    create_dummy_result(
        result_root / "20260527_000002_banana" / "result.json",
        run_name="banana",
        target_object="yellow banana",
    )

    result_paths = collect_result_paths(result_root)

    assert len(result_paths) == 2
    assert all(path.name == "result.json" for path in result_paths)


def test_collect_result_paths_returns_empty_list_for_missing_root(tmp_path):
    missing_root = tmp_path / "missing_outputs"

    result_paths = collect_result_paths(missing_root)

    assert result_paths == []


def test_summarize_result_extracts_key_fields(tmp_path):
    project_root = tmp_path
    result_path = (
        project_root / "outputs" / "dummy_pipeline" / "run_001" / "result.json"
    )
    create_dummy_result(
        result_path,
        run_name="banana",
        target_object="yellow banana",
    )

    row = summarize_result(result_path, project_root)

    assert row["run_dir"] == "outputs/dummy_pipeline/run_001"
    assert row["run_name"] == "banana"
    assert row["target_object"] == "yellow banana"
    assert row["best_action"] == "grasp(yellow banana)"
    assert row["gripper_pose"] == "top_down"
    assert row["predicted_success"] == 0.85
    assert row["best_action_world_model_type"] == "empirical"
    assert row["world_model_mode"] == "empirical"
    assert (
        row["world_model_dataset_path"]
        == "data/action_outcomes/dummy_action_outcomes.json"
    )
    assert row["graspqa_mode"] == "dummy_vlm"
    assert row["vlm_model_name"] == "dummy-vlm"
    assert row["vlm_raw_response"] == "yellow banana"
    assert row["vlm_parse_success"] is True
    assert row["closed_loop_final_success"] is True
    assert row["closed_loop_num_attempts"] == 1
    assert row["closed_loop_final_pose"] == "top_down"
    assert row["config_path"] == "configs/banana.yaml"


def test_write_summary_creates_csv_file(tmp_path):
    output_path = tmp_path / "summary" / "dummy_pipeline_summary.csv"
    rows = [
        {
            "run_dir": "outputs/dummy_pipeline/run_001",
            "run_name": "cube",
            "target_object": "red cube",
            "best_action": "grasp(red cube)",
            "gripper_pose": "top_down",
            "predicted_success": 0.85,
            "best_action_world_model_type": "empirical",
            "world_model_mode": "empirical",
            "world_model_dataset_path": "data/action_outcomes/dummy_action_outcomes.json",
            "graspqa_mode": "dummy_vlm",
            "vlm_model_name": "dummy-vlm",
            "vlm_raw_response": "red cube",
            "vlm_parse_success": True,
            "closed_loop_final_success": True,
            "closed_loop_num_attempts": 1,
            "closed_loop_final_pose": "top_down",
            "config_path": "configs/dummy_pipeline.yaml",
        }
    ]

    write_summary(rows, output_path)

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        loaded_rows = list(reader)

    assert len(loaded_rows) == 1
    assert loaded_rows[0]["run_name"] == "cube"
    assert loaded_rows[0]["target_object"] == "red cube"
    assert loaded_rows[0]["best_action"] == "grasp(red cube)"
    assert loaded_rows[0]["best_action_world_model_type"] == "empirical"
    assert loaded_rows[0]["world_model_mode"] == "empirical"
    assert loaded_rows[0]["graspqa_mode"] == "dummy_vlm"
    assert loaded_rows[0]["vlm_model_name"] == "dummy-vlm"
    assert loaded_rows[0]["vlm_raw_response"] == "red cube"
    assert loaded_rows[0]["vlm_parse_success"] == "True"
    assert loaded_rows[0]["closed_loop_final_success"] == "True"
    assert loaded_rows[0]["closed_loop_num_attempts"] == "1"
    assert loaded_rows[0]["closed_loop_final_pose"] == "top_down"
