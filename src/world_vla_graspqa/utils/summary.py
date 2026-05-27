import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_result(result_path: Path, project_root: Path) -> dict[str, Any]:
    """Convert one result.json file into one CSV row."""

    result = load_json(result_path)
    best_action = result["best_action"]

    return {
        "run_dir": str(result_path.parent.relative_to(project_root)),
        "run_name": result.get("run_name", ""),
        "target_object": result.get("target_object", ""),
        "best_action": best_action.get("name", ""),
        "gripper_pose": best_action.get("gripper_pose", ""),
        "predicted_success": best_action.get("predicted_success", ""),
        "config_path": result.get("config_path", ""),
    }


def collect_result_paths(result_root: Path) -> list[Path]:
    """Collect all result.json files under the result root."""

    if not result_root.exists():
        return []

    return sorted(result_root.glob("*/result.json"))


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write summary rows into a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_dir",
        "run_name",
        "target_object",
        "best_action",
        "gripper_pose",
        "predicted_success",
        "config_path",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
