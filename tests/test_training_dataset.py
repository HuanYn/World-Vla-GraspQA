import json

from world_vla_graspqa.world_model.training_dataset import (
    build_features,
    build_training_dataset_from_paths,
    build_training_samples,
    load_feedback_records,
    outcome_record_to_training_sample,
    write_training_samples_jsonl,
)


def test_build_features_from_record():
    record = {
        "target_object": "red cube",
        "gripper_pose": "top_down",
    }

    features = build_features(record)

    assert features["target_contains_cube"] is True
    assert features["target_contains_banana"] is False
    assert features["pose_is_top_down"] is True
    assert features["pose_is_left_side"] is False


def test_outcome_record_to_training_sample():
    record = {
        "scene_id": "tabletop_dummy_scene",
        "target_object": "yellow banana",
        "gripper_pose": "right_side",
        "action_name": "grasp(yellow banana)",
        "success": True,
        "outcome": "object_grasped",
    }

    sample = outcome_record_to_training_sample(
        record=record,
        sample_id="sample_000001",
        source="static_outcome",
    )

    assert sample["sample_id"] == "sample_000001"
    assert sample["source"] == "static_outcome"
    assert sample["target_object"] == "yellow banana"
    assert sample["gripper_pose"] == "right_side"
    assert sample["success"] is True
    assert sample["features"]["target_contains_banana"] is True
    assert sample["features"]["pose_is_right_side"] is True


def test_build_training_samples_combines_sources():
    outcome_records = [
        {
            "target_object": "red cube",
            "gripper_pose": "top_down",
            "success": True,
        }
    ]
    feedback_records = [
        {
            "target_object": "red cube",
            "gripper_pose": "left_side",
            "success": False,
        }
    ]

    samples = build_training_samples(
        outcome_records=outcome_records,
        feedback_records=feedback_records,
    )

    assert len(samples) == 2
    assert samples[0]["sample_id"] == "sample_000001"
    assert samples[0]["source"] == "static_outcome"
    assert samples[1]["sample_id"] == "sample_000002"
    assert samples[1]["source"] == "closed_loop_feedback"


def test_load_feedback_records_reads_jsonl(tmp_path):
    feedback_path = tmp_path / "feedback_records.jsonl"
    records = [
        {"target_object": "red cube", "success": True},
        {"target_object": "yellow banana", "success": False},
    ]

    with feedback_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    loaded_records = load_feedback_records(feedback_path)

    assert loaded_records == records


def test_load_feedback_records_returns_empty_for_missing_file(tmp_path):
    feedback_path = tmp_path / "missing.jsonl"

    loaded_records = load_feedback_records(feedback_path)

    assert loaded_records == []


def test_write_training_samples_jsonl(tmp_path):
    output_path = tmp_path / "training" / "samples.jsonl"
    samples = [
        {
            "sample_id": "sample_000001",
            "target_object": "red cube",
            "success": True,
        }
    ]

    write_training_samples_jsonl(samples, output_path)

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        loaded_samples = [json.loads(line) for line in f]

    assert loaded_samples == samples


def test_build_training_dataset_from_paths(tmp_path):
    outcome_dataset_path = tmp_path / "outcomes.json"
    feedback_path = tmp_path / "feedback.jsonl"
    output_path = tmp_path / "training_samples.jsonl"

    outcome_dataset = {
        "records": [
            {
                "scene_id": "scene_001",
                "target_object": "red cube",
                "gripper_pose": "top_down",
                "action_name": "grasp(red cube)",
                "success": True,
                "outcome": "object_grasped",
            }
        ]
    }
    outcome_dataset_path.write_text(json.dumps(outcome_dataset), encoding="utf-8")

    feedback_record = {
        "scene_id": "scene_001",
        "target_object": "red cube",
        "gripper_pose": "right_side",
        "action_name": "grasp(red cube)",
        "success": False,
        "outcome": "unstable_grasp",
    }
    feedback_path.write_text(json.dumps(feedback_record) + "\n", encoding="utf-8")

    samples = build_training_dataset_from_paths(
        outcome_dataset_path=outcome_dataset_path,
        feedback_path=feedback_path,
        output_path=output_path,
    )

    assert len(samples) == 2
    assert output_path.exists()
    assert samples[0]["source"] == "static_outcome"
    assert samples[1]["source"] == "closed_loop_feedback"
