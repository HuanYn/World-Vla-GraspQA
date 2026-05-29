from pathlib import Path
from typing import Any

from world_vla_graspqa.world_model.outcome_dataset import (
    compute_pose_success_rates,
    filter_records_by_target,
    get_outcome_records,
    load_outcome_dataset,
)


class EmpiricalWorldModel:
    """A simple world model based on empirical action-outcome statistics."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        default_success: float = 0.5,
    ) -> None:
        self.records = records
        self.default_success = default_success

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        default_success: float = 0.5,
    ) -> "EmpiricalWorldModel":
        """Build an empirical world model from an action-outcome dataset."""

        dataset = load_outcome_dataset(dataset_path)
        records = get_outcome_records(dataset)

        return cls(
            records=records,
            default_success=default_success,
        )

    def predict(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict action outcomes using empirical pose success rates."""

        scored_actions = []

        for action in actions:
            target_object = action["target"]
            gripper_pose = action["gripper_pose"]

            target_records = filter_records_by_target(
                records=self.records,
                target_object=target_object,
            )
            pose_success_rates = compute_pose_success_rates(target_records)

            predicted_success = pose_success_rates.get(
                gripper_pose,
                self.default_success,
            )

            scored_action = {
                **action,
                "predicted_success": predicted_success,
                "predicted_result": self._predict_result(predicted_success),
                "world_model_type": "empirical",
            }
            scored_actions.append(scored_action)

        return scored_actions

    def _predict_result(self, predicted_success: float) -> str:
        """Convert a success probability into a coarse predicted result label."""

        if predicted_success >= 0.5:
            return "object_grasped"

        return "grasp_failed"
