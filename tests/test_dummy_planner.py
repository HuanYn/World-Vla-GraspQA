import pytest

from world_vla_graspqa.planner.dummy_planner import DummyPlanner


def test_dummy_planner_selects_highest_success_action():
    scored_actions = [
        {
            "name": "grasp(red cube)",
            "gripper_pose": "left_side",
            "predicted_success": 0.65,
        },
        {
            "name": "grasp(red cube)",
            "gripper_pose": "top_down",
            "predicted_success": 0.85,
        },
    ]

    planner = DummyPlanner()
    best_action = planner.select_action(scored_actions)

    assert best_action["gripper_pose"] == "top_down"
    assert best_action["predicted_success"] == 0.85


def test_dummy_planner_raises_error_on_empty_actions():
    planner = DummyPlanner()

    with pytest.raises(ValueError):
        planner.select_action([])
