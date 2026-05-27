from world_vla_graspqa.world_model.dummy_world_model import DummyWorldModel


def test_dummy_world_model_scores_actions():
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

    world_model = DummyWorldModel()
    scored_actions = world_model.predict(actions)

    assert len(scored_actions) == 2
    assert scored_actions[0]["predicted_success"] == 0.85
    assert scored_actions[1]["predicted_success"] == 0.65
    assert scored_actions[0]["predicted_result"] == "object_grasped"
