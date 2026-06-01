from typing import Any

from world_vla_graspqa.action.dummy_executor import DummyExecutor
from world_vla_graspqa.critic.dummy_critic import DummyCritic
from world_vla_graspqa.planner.dummy_planner import DummyPlanner
from world_vla_graspqa.utils.logger import log_step


class ClosedLoopRunner:
    """Run a simple closed-loop action selection and execution process."""

    def __init__(
        self,
        planner: DummyPlanner | None = None,
        executor: DummyExecutor | None = None,
        critic: DummyCritic | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.planner = planner or DummyPlanner()
        self.executor = executor or DummyExecutor()
        self.critic = critic or DummyCritic()
        self.max_attempts = max_attempts

    def run(self, scored_actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Run closed-loop planning, execution, and critique."""

        remaining_actions = list(scored_actions)
        closed_loop_trace = []

        for step in range(1, self.max_attempts + 1):
            if not remaining_actions:
                break

            selected_action = self.planner.select_action(remaining_actions)
            execution_result = self.executor.execute(selected_action)
            critic_result = self.critic.evaluate(execution_result)

            trace_step = {
                "step": step,
                "selected_action": selected_action,
                "execution_result": execution_result,
                "critic_result": critic_result,
            }
            closed_loop_trace.append(trace_step)

            log_step(
                "ClosedLoop",
                (
                    f"Step {step}: action={selected_action['name']}, "
                    f"pose={selected_action['gripper_pose']}, "
                    f"success={critic_result['critic_success']}"
                ),
            )

            if critic_result["critic_success"]:
                return {
                    "final_success": True,
                    "final_action": selected_action,
                    "closed_loop_trace": closed_loop_trace,
                    "num_attempts": step,
                }

            remaining_actions = [
                action
                for action in remaining_actions
                if not self._same_action(action, selected_action)
            ]

        return {
            "final_success": False,
            "final_action": (
                closed_loop_trace[-1]["selected_action"] if closed_loop_trace else None
            ),
            "closed_loop_trace": closed_loop_trace,
            "num_attempts": len(closed_loop_trace),
        }

    def _same_action(
        self,
        action_a: dict[str, Any],
        action_b: dict[str, Any],
    ) -> bool:
        """Check whether two action dictionaries represent the same action."""

        return (
            action_a.get("name") == action_b.get("name")
            and action_a.get("target") == action_b.get("target")
            and action_a.get("gripper_pose") == action_b.get("gripper_pose")
        )
