import json

from world_vla_graspqa.world_model.feedback import (
    append_jsonl_records,
    closed_loop_trace_to_outcome_records,
    trace_step_to_outcome_record,
)


def make_trace_step():
    return {
        "step": 1,
        "selected_action": {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "top_down",
            "predicted_success": 1.0,
        },
        "execution_result": {
            "observed_result": "object_grasped",
            "failure_reason": None,
        },
        "critic_result": {
            "critic_success": True,
            "critic_reason": "object_grasped",
        },
    }


def test_trace_step_to_outcome_record():
    record = trace_step_to_outcome_record(
        trace_step=make_trace_step(),
        scene_id="tabletop_dummy_scene",
    )

    assert record["scene_id"] == "tabletop_dummy_scene"
    assert record["target_object"] == "red cube"
    assert record["gripper_pose"] == "top_down"
    assert record["action_name"] == "grasp(red cube)"
    assert record["success"] is True
    assert record["outcome"] == "object_grasped"
    assert record["predicted_success"] == 1.0
    assert record["observed_result"] == "object_grasped"
    assert record["failure_reason"] is None


def test_closed_loop_trace_to_outcome_records():
    records = closed_loop_trace_to_outcome_records(
        closed_loop_trace=[make_trace_step()],
        scene_id="tabletop_dummy_scene",
    )

    assert len(records) == 1
    assert records[0]["target_object"] == "red cube"


def test_append_jsonl_records_appends_records(tmp_path):
    output_path = tmp_path / "feedback" / "records.jsonl"
    records = [
        {
            "scene_id": "scene_001",
            "target_object": "red cube",
            "success": True,
        },
        {
            "scene_id": "scene_001",
            "target_object": "red cube",
            "success": False,
        },
    ]

    append_jsonl_records(records, output_path)

    with output_path.open("r", encoding="utf-8") as f:
        loaded_records = [json.loads(line) for line in f]

    assert len(loaded_records) == 2
    assert loaded_records[0]["success"] is True
    assert loaded_records[1]["success"] is False
