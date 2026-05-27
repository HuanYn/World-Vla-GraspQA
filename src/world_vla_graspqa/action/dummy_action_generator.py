from typing import Any

from world_vla_graspqa.utils.logger import log_step


class DummyActionGenerator:
    """Generate candidate grasping actions."""

    def generate(self, target_object: str) -> list[dict[str, Any]]:
        actions = [
            {
                "name": f"grasp({target_object})",
                "target": target_object,
                "gripper_pose": "top_down",
            },
            {
                "name": f"grasp({target_object})",
                "target": target_object,
                "gripper_pose": "left_side",
            },
            {
                "name": f"grasp({target_object})",
                "target": target_object,
                "gripper_pose": "right_side",
            },
        ]

        log_step("Action", f"Generated {len(actions)} candidate actions.")
        return actions
