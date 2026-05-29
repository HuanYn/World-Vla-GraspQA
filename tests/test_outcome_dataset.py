import json

import pytest

from world_vla_graspqa.world_model.outcome_dataset import (
    compute_pose_success_rates,
    filter_records_by_target,
    get_outcome_records,
    load_outcome_dataset,
)


def test_load_outcome_dataset_reads_json(tmp_path):
    dataset_path = tmp_path / "outcomes.json"
    data = {
        "dataset_name": "test_outcomes",
        "records": [
            {
                "target_object": "red cube",
                "gripper_pose": "top_down",
                "success": True,
            }
        ],
    }

    dataset_path.write_text(json.dumps(data), encoding="utf-8")

    dataset = load_outcome_dataset(dataset_path)

    assert dataset["dataset_name"] == "test_outcomes"
    assert dataset["records"][0]["target_object"] == "red cube"


def test_load_outcome_dataset_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_outcome_dataset(missing_path)


def test_get_outcome_records_returns_records():
    dataset = {
        "records": [
            {
                "target_object": "red cube",
                "gripper_pose": "top_down",
                "success": True,
            }
        ]
    }

    records = get_outcome_records(dataset)

    assert len(records) == 1
    assert records[0]["gripper_pose"] == "top_down"


def test_get_outcome_records_raises_for_invalid_records():
    dataset = {
        "records": "not-a-list",
    }

    with pytest.raises(ValueError):
        get_outcome_records(dataset)


def test_filter_records_by_target():
    records = [
        {
            "target_object": "red cube",
            "gripper_pose": "top_down",
            "success": True,
        },
        {
            "target_object": "yellow banana",
            "gripper_pose": "left_side",
            "success": False,
        },
    ]

    filtered = filter_records_by_target(records, target_object="red cube")

    assert len(filtered) == 1
    assert filtered[0]["target_object"] == "red cube"


def test_compute_pose_success_rates():
    records = [
        {
            "gripper_pose": "top_down",
            "success": True,
        },
        {
            "gripper_pose": "top_down",
            "success": False,
        },
        {
            "gripper_pose": "left_side",
            "success": True,
        },
    ]

    success_rates = compute_pose_success_rates(records)

    assert success_rates["top_down"] == 0.5
    assert success_rates["left_side"] == 1.0
