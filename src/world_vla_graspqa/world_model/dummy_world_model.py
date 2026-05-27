from typing import Any

from world_vla_graspqa.utils.logger import log_step


class DummyWorldModel:
    """A dummy world model that predicts action success probabilities."""

    def predict(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored_actions = []

        pose_scores = {
            "top_down": 0.85,
            "left_side": 0.65,
            "right_side": 0.60,
        }

        for action in actions:
            pose = action["gripper_pose"]
            scored_action = {
                **action,
                "predicted_success": pose_scores.get(pose, 0.5),
                "predicted_result": "object_grasped",
            }
            scored_actions.append(scored_action)

        log_step("WorldModel", "Predicted outcomes for candidate actions.")
        return scored_actions
