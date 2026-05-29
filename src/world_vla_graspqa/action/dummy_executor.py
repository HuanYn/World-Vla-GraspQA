from typing import Any

from world_vla_graspqa.utils.logger import log_step


class DummyExecutor:
    """A dummy action executor for simulating robot action execution."""

    def __init__(self, success_threshold: float = 0.5) -> None:
        self.success_threshold = success_threshold

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute an action and return an observed execution result."""

        predicted_success = float(action.get("predicted_success", 0.0))
        execution_success = predicted_success >= self.success_threshold

        if execution_success:
            observed_result = "object_grasped"
            failure_reason = None
        else:
            observed_result = "grasp_failed"
            failure_reason = self._infer_failure_reason(action)

        result = {
            "action_name": action.get("name", ""),
            "target": action.get("target", ""),
            "gripper_pose": action.get("gripper_pose", ""),
            "predicted_success": predicted_success,
            "execution_success": execution_success,
            "observed_result": observed_result,
            "failure_reason": failure_reason,
        }

        log_step(
            "Executor",
            (
                f"Executed {result['action_name']} with pose={result['gripper_pose']}. "
                f"success={result['execution_success']}"
            ),
        )

        return result

    def _infer_failure_reason(self, action: dict[str, Any]) -> str:
        """Infer a dummy failure reason from the action pose."""

        gripper_pose = action.get("gripper_pose", "")

        if gripper_pose == "left_side":
            return "slipped"

        if gripper_pose == "right_side":
            return "unstable_grasp"

        return "low_predicted_success"
