from typing import Any

from world_vla_graspqa.utils.logger import log_step


class DummyCritic:
    """A dummy critic for judging whether execution succeeded."""

    def evaluate(self, execution_result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate an execution result."""

        execution_success = bool(execution_result.get("execution_success", False))
        observed_result = execution_result.get("observed_result", "")
        failure_reason = execution_result.get("failure_reason")

        if execution_success:
            critic_success = True
            critic_reason = observed_result or "object_grasped"
        else:
            critic_success = False
            critic_reason = failure_reason or observed_result or "unknown_failure"

        result = {
            "critic_success": critic_success,
            "critic_reason": critic_reason,
            "observed_result": observed_result,
            "failure_reason": failure_reason,
        }

        log_step(
            "Critic",
            f"Execution success={critic_success}, reason={critic_reason}",
        )

        return result
