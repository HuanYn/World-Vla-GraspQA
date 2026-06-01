import json
from pathlib import Path
from typing import Any

from world_vla_graspqa.utils.io import ensure_dir


def trace_step_to_outcome_record(
    trace_step: dict[str, Any],
    scene_id: str,
) -> dict[str, Any]:
    """Convert one closed-loop trace step into an action-outcome record."""

    selected_action = trace_step["selected_action"]
    execution_result = trace_step["execution_result"]
    critic_result = trace_step["critic_result"]

    return {
        "scene_id": scene_id,
        "target_object": selected_action.get("target", ""),
        "gripper_pose": selected_action.get("gripper_pose", ""),
        "action_name": selected_action.get("name", ""),
        "success": bool(critic_result.get("critic_success", False)),
        "outcome": critic_result.get("critic_reason", ""),
        "predicted_success": selected_action.get("predicted_success", ""),
        "observed_result": execution_result.get("observed_result", ""),
        "failure_reason": execution_result.get("failure_reason"),
    }


def closed_loop_trace_to_outcome_records(
    closed_loop_trace: list[dict[str, Any]],
    scene_id: str,
) -> list[dict[str, Any]]:
    """Convert a closed-loop trace into action-outcome records."""

    return [
        trace_step_to_outcome_record(
            trace_step=trace_step,
            scene_id=scene_id,
        )
        for trace_step in closed_loop_trace
    ]


def append_jsonl_records(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Append records to a JSONL file."""

    path = Path(output_path)
    ensure_dir(path.parent)

    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
