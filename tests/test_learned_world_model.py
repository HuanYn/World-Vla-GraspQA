from world_vla_graspqa.world_model.learned_world_model import LearnedWorldModel
from world_vla_graspqa.world_model.train_logistic_model import (
    save_model_checkpoint,
    train_logistic_world_model,
)


def make_training_samples():
    return [
        {
            "success": True,
            "features": {
                "target_contains_cube": True,
                "target_contains_banana": False,
                "target_contains_bowl": False,
                "pose_is_top_down": True,
                "pose_is_left_side": False,
                "pose_is_right_side": False,
            },
        },
        {
            "success": False,
            "features": {
                "target_contains_cube": True,
                "target_contains_banana": False,
                "target_contains_bowl": False,
                "pose_is_top_down": False,
                "pose_is_left_side": False,
                "pose_is_right_side": True,
            },
        },
        {
            "success": True,
            "features": {
                "target_contains_cube": False,
                "target_contains_banana": True,
                "target_contains_bowl": False,
                "pose_is_top_down": True,
                "pose_is_left_side": False,
                "pose_is_right_side": False,
            },
        },
        {
            "success": False,
            "features": {
                "target_contains_cube": False,
                "target_contains_banana": True,
                "target_contains_bowl": False,
                "pose_is_top_down": False,
                "pose_is_left_side": True,
                "pose_is_right_side": False,
            },
        },
    ]


def test_learned_world_model_predicts_actions():
    model = train_logistic_world_model(make_training_samples())
    learned_world_model = LearnedWorldModel(
        checkpoint={
            "model_type": "logistic_regression",
            "feature_names": [
                "target_contains_cube",
                "target_contains_banana",
                "target_contains_bowl",
                "pose_is_top_down",
                "pose_is_left_side",
                "pose_is_right_side",
            ],
            "model": model,
        }
    )

    actions = [
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "top_down",
        },
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "right_side",
        },
    ]

    scored_actions = learned_world_model.predict(actions)

    assert len(scored_actions) == 2
    assert "predicted_success" in scored_actions[0]
    assert scored_actions[0]["world_model_type"] == "learned_logistic_regression"
    assert scored_actions[0]["predicted_result"] in ["object_grasped", "grasp_failed"]


def test_learned_world_model_loads_from_checkpoint(tmp_path):
    model = train_logistic_world_model(make_training_samples())
    checkpoint_path = tmp_path / "logistic_world_model.pkl"
    save_model_checkpoint(model, checkpoint_path)

    learned_world_model = LearnedWorldModel.from_checkpoint(checkpoint_path)

    actions = [
        {
            "name": "grasp(yellow banana)",
            "target": "yellow banana",
            "gripper_pose": "top_down",
        }
    ]

    scored_actions = learned_world_model.predict(actions)

    assert len(scored_actions) == 1
    assert scored_actions[0]["target"] == "yellow banana"
    assert "predicted_success" in scored_actions[0]
