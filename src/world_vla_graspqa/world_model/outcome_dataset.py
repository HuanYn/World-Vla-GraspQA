import json
from pathlib import Path
from typing import Any


def load_outcome_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """Load an action-outcome dataset JSON file."""

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Outcome dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    return dataset


def get_outcome_records(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Return action-outcome records from a dataset."""

    records = dataset.get("records", [])

    if not isinstance(records, list):
        raise ValueError("Outcome dataset field 'records' must be a list.")

    return records


def filter_records_by_target(
    records: list[dict[str, Any]],
    target_object: str,
) -> list[dict[str, Any]]:
    """Filter outcome records by target object."""

    return [
        record for record in records if record.get("target_object") == target_object
    ]


def compute_pose_success_rates(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute empirical success rate for each gripper pose."""

    pose_stats: dict[str, dict[str, int]] = {}

    for record in records:
        pose = record["gripper_pose"]
        success = bool(record["success"])

        if pose not in pose_stats:
            pose_stats[pose] = {
                "success_count": 0,
                "total_count": 0,
            }

        pose_stats[pose]["total_count"] += 1
        if success:
            pose_stats[pose]["success_count"] += 1

    success_rates = {}
    for pose, stats in pose_stats.items():
        success_rates[pose] = stats["success_count"] / stats["total_count"]

    return success_rates
