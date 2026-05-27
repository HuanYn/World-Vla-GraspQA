from typing import Any

from world_vla_graspqa.utils.logger import log_step


class DummyPlanner:
    """Select the best action according to predicted success."""

    def select_action(self, scored_actions: list[dict[str, Any]]) -> dict[str, Any]:
        if not scored_actions:
            raise ValueError("No candidate actions were provided.")

        best_action = max(
            scored_actions,
            key=lambda action: action["predicted_success"],
        )

        log_step("Planner", f"Selected best action: {best_action['name']}")
        return best_action
