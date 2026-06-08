from pathlib import Path
from typing import Any

from world_vla_graspqa.world_model.train_logistic_model import load_model_checkpoint
from world_vla_graspqa.world_model.training_dataset import build_features


class LearnedWorldModel:
    """A learned mini world model based on a trained classifier checkpoint."""

    def __init__(
        self,
        checkpoint: dict[str, Any],
        success_threshold: float = 0.5,
    ) -> None:
        self.model = checkpoint["model"]
        self.feature_names = checkpoint["feature_names"]
        self.model_type = checkpoint.get("model_type", "unknown")
        self.success_threshold = success_threshold

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        success_threshold: float = 0.5,
    ) -> "LearnedWorldModel":
        """Load a learned world model from a checkpoint file."""

        checkpoint = load_model_checkpoint(checkpoint_path)

        return cls(
            checkpoint=checkpoint,
            success_threshold=success_threshold,
        )

    def predict(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict action outcomes using the learned model."""

        scored_actions = []

        for action in actions:
            features = self._action_to_feature_row(action)
            predicted_success = float(self.model.predict_proba([features])[0][1])

            scored_actions.append(
                {
                    **action,
                    "predicted_success": predicted_success,
                    "predicted_result": self._predict_result(predicted_success),
                    "world_model_type": f"learned_{self.model_type}",
                }
            )

        return scored_actions

    def _action_to_feature_row(self, action: dict[str, Any]) -> list[float]:
        """Convert an action dictionary into a feature row."""

        record = {
            "target_object": action.get("target", ""),
            "gripper_pose": action.get("gripper_pose", ""),
        }
        feature_dict = build_features(record)

        return [
            float(bool(feature_dict.get(feature_name, False)))
            for feature_name in self.feature_names
        ]

    def _predict_result(self, predicted_success: float) -> str:
        """Convert success probability into a predicted result label."""

        if predicted_success >= self.success_threshold:
            return "object_grasped"

        return "grasp_failed"
