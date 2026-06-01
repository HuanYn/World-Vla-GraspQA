from world_vla_graspqa.critic.dummy_critic import DummyCritic


def main() -> None:
    critic = DummyCritic()

    execution_results = [
        {
            "execution_success": True,
            "observed_result": "object_grasped",
            "failure_reason": None,
        },
        {
            "execution_success": False,
            "observed_result": "grasp_failed",
            "failure_reason": "unstable_grasp",
        },
    ]

    for execution_result in execution_results:
        critic_result = critic.evaluate(execution_result)
        print(critic_result)


if __name__ == "__main__":
    main()
