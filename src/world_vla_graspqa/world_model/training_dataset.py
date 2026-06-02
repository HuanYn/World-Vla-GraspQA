import json
from pathlib import Path
from typing import Any

from world_vla_graspqa.utils.io import ensure_dir
from world_vla_graspqa.world_model.outcome_dataset import (
    get_outcome_records,
    load_outcome_dataset,
)


def load_feedback_records(feedback_path: str | Path) -> list[dict[str, Any]]:
    """Load feedback records from a JSONL file."""

    path = Path(feedback_path)

    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def build_features(record: dict[str, Any]) -> dict[str, Any]:
    """Build simple symbolic features from an action-outcome record."""

    target_object = record.get("target_object", "")
    gripper_pose = record.get("gripper_pose", "")

    return {
        "target_contains_cube": "cube" in target_object,
        "target_contains_banana": "banana" in target_object,
        "target_contains_bowl": "bowl" in target_object,
        "pose_is_top_down": gripper_pose == "top_down",
        "pose_is_left_side": gripper_pose == "left_side",
        "pose_is_right_side": gripper_pose == "right_side",
    }


def outcome_record_to_training_sample(
    record: dict[str, Any],
    sample_id: str,
    source: str,
) -> dict[str, Any]:
    """Convert one action-outcome record into a training sample."""

    return {
        "sample_id": sample_id,
        "source": source,
        "scene_id": record.get("scene_id", ""),
        "target_object": record.get("target_object", ""),
        "gripper_pose": record.get("gripper_pose", ""),
        "action_name": record.get("action_name", ""),
        "success": bool(record.get("success", False)),
        "outcome": record.get("outcome", ""),
        "features": build_features(record),
    }


def build_training_samples(
    outcome_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build training samples from static outcome records and feedback records."""

    combined_records = [("static_outcome", record) for record in outcome_records] + [
        ("closed_loop_feedback", record) for record in feedback_records
    ]

    samples = []
    for index, (source, record) in enumerate(combined_records, start=1):
        samples.append(
            outcome_record_to_training_sample(
                record=record,
                sample_id=f"sample_{index:06d}",
                source=source,
            )
        )

    return samples


def write_training_samples_jsonl(
    samples: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write training samples to a JSONL file."""

    path = Path(output_path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def build_training_dataset_from_paths(
    outcome_dataset_path: str | Path,
    feedback_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Build and save a world-model training dataset from file paths."""

    outcome_dataset = load_outcome_dataset(outcome_dataset_path)
    outcome_records = get_outcome_records(outcome_dataset)
    feedback_records = load_feedback_records(feedback_path)

    samples = build_training_samples(
        outcome_records=outcome_records,
        feedback_records=feedback_records,
    )
    write_training_samples_jsonl(samples, output_path)

    return samples
