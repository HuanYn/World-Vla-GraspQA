from world_vla_graspqa.action.dummy_executor import DummyExecutor


def test_dummy_executor_returns_success_for_high_predicted_success():
    executor = DummyExecutor(success_threshold=0.5)
    action = {
        "name": "grasp(red cube)",
        "target": "red cube",
        "gripper_pose": "top_down",
        "predicted_success": 0.9,
    }

    result = executor.execute(action)

    assert result["action_name"] == "grasp(red cube)"
    assert result["target"] == "red cube"
    assert result["gripper_pose"] == "top_down"
    assert result["predicted_success"] == 0.9
    assert result["execution_success"] is True
    assert result["observed_result"] == "object_grasped"
    assert result["failure_reason"] is None


def test_dummy_executor_returns_failure_for_low_predicted_success():
    executor = DummyExecutor(success_threshold=0.5)
    action = {
        "name": "grasp(red cube)",
        "target": "red cube",
        "gripper_pose": "right_side",
        "predicted_success": 0.2,
    }

    result = executor.execute(action)

    assert result["execution_success"] is False
    assert result["observed_result"] == "grasp_failed"
    assert result["failure_reason"] == "unstable_grasp"


def test_dummy_executor_uses_custom_threshold():
    executor = DummyExecutor(success_threshold=0.8)
    action = {
        "name": "grasp(yellow banana)",
        "target": "yellow banana",
        "gripper_pose": "top_down",
        "predicted_success": 0.7,
    }

    result = executor.execute(action)

    assert result["execution_success"] is False
    assert result["failure_reason"] == "low_predicted_success"
