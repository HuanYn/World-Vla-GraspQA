import json

from world_vla_graspqa.world_model.empirical_world_model import EmpiricalWorldModel


def test_empirical_world_model_predicts_success_from_records():
    records = [
        {
            "target_object": "red cube",
            "gripper_pose": "top_down",
            "success": True,
        },
        {
            "target_object": "red cube",
            "gripper_pose": "top_down",
            "success": True,
        },
        {
            "target_object": "red cube",
            "gripper_pose": "left_side",
            "success": False,
        },
    ]
    model = EmpiricalWorldModel(records=records)

    actions = [
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "top_down",
        },
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "left_side",
        },
    ]

    scored_actions = model.predict(actions)

    assert scored_actions[0]["predicted_success"] == 1.0
    assert scored_actions[0]["predicted_result"] == "object_grasped"
    assert scored_actions[0]["world_model_type"] == "empirical"

    assert scored_actions[1]["predicted_success"] == 0.0
    assert scored_actions[1]["predicted_result"] == "grasp_failed"


def test_empirical_world_model_uses_default_for_unseen_pose():
    records = [
        {
            "target_object": "red cube",
            "gripper_pose": "top_down",
            "success": True,
        }
    ]
    model = EmpiricalWorldModel(records=records, default_success=0.25)

    actions = [
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "right_side",
        }
    ]

    scored_actions = model.predict(actions)

    assert scored_actions[0]["predicted_success"] == 0.25
    assert scored_actions[0]["predicted_result"] == "grasp_failed"


def test_empirical_world_model_loads_from_dataset(tmp_path):
    dataset_path = tmp_path / "outcomes.json"
    dataset = {
        "records": [
            {
                "target_object": "yellow banana",
                "gripper_pose": "right_side",
                "success": True,
            }
        ]
    }
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    model = EmpiricalWorldModel.from_dataset(dataset_path)

    actions = [
        {
            "name": "grasp(yellow banana)",
            "target": "yellow banana",
            "gripper_pose": "right_side",
        }
    ]

    scored_actions = model.predict(actions)

    assert scored_actions[0]["predicted_success"] == 1.0
