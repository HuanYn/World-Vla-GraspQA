from typing import Any

from world_vla_graspqa.planner.closed_loop_runner import ClosedLoopRunner


class FailOnceExecutor:
    """Test executor that fails the first execution and succeeds afterwards."""

    def __init__(self) -> None:
        self.call_count = 0

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        execution_success = self.call_count > 1

        return {
            "action_name": action.get("name", ""),
            "target": action.get("target", ""),
            "gripper_pose": action.get("gripper_pose", ""),
            "predicted_success": action.get("predicted_success", 0.0),
            "execution_success": execution_success,
            "observed_result": (
                "object_grasped" if execution_success else "grasp_failed"
            ),
            "failure_reason": None if execution_success else "simulated_failure",
        }


class AlwaysFailExecutor:
    """Test executor that always fails."""

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        return {
            "action_name": action.get("name", ""),
            "target": action.get("target", ""),
            "gripper_pose": action.get("gripper_pose", ""),
            "predicted_success": action.get("predicted_success", 0.0),
            "execution_success": False,
            "observed_result": "grasp_failed",
            "failure_reason": "simulated_failure",
        }


def test_closed_loop_runner_succeeds_on_first_attempt():
    runner = ClosedLoopRunner()
    scored_actions = [
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "top_down",
            "predicted_success": 0.9,
        },
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "right_side",
            "predicted_success": 0.2,
        },
    ]

    result = runner.run(scored_actions)

    assert result["final_success"] is True
    assert result["num_attempts"] == 1
    assert result["final_action"]["gripper_pose"] == "top_down"
    assert len(result["closed_loop_trace"]) == 1
    assert result["closed_loop_trace"][0]["critic_result"]["critic_success"] is True


def test_closed_loop_runner_retries_after_failed_action():
    runner = ClosedLoopRunner(
        executor=FailOnceExecutor(),
        max_attempts=2,
    )
    scored_actions = [
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "top_down",
            "predicted_success": 0.9,
        },
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "left_side",
            "predicted_success": 0.7,
        },
    ]

    result = runner.run(scored_actions)

    assert result["final_success"] is True
    assert result["num_attempts"] == 2
    assert (
        result["closed_loop_trace"][0]["selected_action"]["gripper_pose"] == "top_down"
    )
    assert result["closed_loop_trace"][0]["critic_result"]["critic_success"] is False
    assert (
        result["closed_loop_trace"][1]["selected_action"]["gripper_pose"] == "left_side"
    )
    assert result["closed_loop_trace"][1]["critic_result"]["critic_success"] is True


def test_closed_loop_runner_fails_when_all_attempts_fail():
    runner = ClosedLoopRunner(
        executor=AlwaysFailExecutor(),
        max_attempts=2,
    )
    scored_actions = [
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "left_side",
            "predicted_success": 0.7,
        },
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "right_side",
            "predicted_success": 0.6,
        },
    ]

    result = runner.run(scored_actions)

    assert result["final_success"] is False
    assert result["num_attempts"] == 2
    assert len(result["closed_loop_trace"]) == 2
    assert result["closed_loop_trace"][0]["critic_result"]["critic_success"] is False
    assert result["closed_loop_trace"][1]["critic_result"]["critic_success"] is False
