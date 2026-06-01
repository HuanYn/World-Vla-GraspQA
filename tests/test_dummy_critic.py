from world_vla_graspqa.critic.dummy_critic import DummyCritic


def test_dummy_critic_returns_success_for_successful_execution():
    critic = DummyCritic()
    execution_result = {
        "execution_success": True,
        "observed_result": "object_grasped",
        "failure_reason": None,
    }

    result = critic.evaluate(execution_result)

    assert result["critic_success"] is True
    assert result["critic_reason"] == "object_grasped"
    assert result["observed_result"] == "object_grasped"
    assert result["failure_reason"] is None


def test_dummy_critic_returns_failure_reason_for_failed_execution():
    critic = DummyCritic()
    execution_result = {
        "execution_success": False,
        "observed_result": "grasp_failed",
        "failure_reason": "unstable_grasp",
    }

    result = critic.evaluate(execution_result)

    assert result["critic_success"] is False
    assert result["critic_reason"] == "unstable_grasp"
    assert result["observed_result"] == "grasp_failed"
    assert result["failure_reason"] == "unstable_grasp"


def test_dummy_critic_handles_missing_failure_reason():
    critic = DummyCritic()
    execution_result = {
        "execution_success": False,
        "observed_result": "grasp_failed",
    }

    result = critic.evaluate(execution_result)

    assert result["critic_success"] is False
    assert result["critic_reason"] == "grasp_failed"
