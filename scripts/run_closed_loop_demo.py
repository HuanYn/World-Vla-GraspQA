from world_vla_graspqa.action.dummy_executor import DummyExecutor
from world_vla_graspqa.planner.closed_loop_runner import ClosedLoopRunner


def main() -> None:
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
            "gripper_pose": "top_down",
            "predicted_success": 0.9,
        },
        {
            "name": "grasp(red cube)",
            "target": "red cube",
            "gripper_pose": "right_side",
            "predicted_success": 0.4,
        },
    ]

    runner = ClosedLoopRunner(
        executor=DummyExecutor(success_threshold=0.8),
        max_attempts=3,
    )
    result = runner.run(scored_actions)

    print("[ClosedLoopDemo] Final success:", result["final_success"])
    print("[ClosedLoopDemo] Num attempts:", result["num_attempts"])
    print("[ClosedLoopDemo] Final action:", result["final_action"])

    for step in result["closed_loop_trace"]:
        print("[ClosedLoopDemo] Trace step:", step)


if __name__ == "__main__":
    main()
