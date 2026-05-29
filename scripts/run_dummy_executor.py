from world_vla_graspqa.action.dummy_executor import DummyExecutor


def main() -> None:
    actions = [
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

    executor = DummyExecutor()

    for action in actions:
        result = executor.execute(action)
        print(result)


if __name__ == "__main__":
    main()
