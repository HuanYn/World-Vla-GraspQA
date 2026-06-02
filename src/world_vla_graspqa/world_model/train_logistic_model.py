import json
import pickle
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression

from world_vla_graspqa.utils.io import ensure_dir

FEATURE_NAMES = [
    "target_contains_cube",
    "target_contains_banana",
    "target_contains_bowl",
    "pose_is_top_down",
    "pose_is_left_side",
    "pose_is_right_side",
]


def load_training_samples(path: str | Path) -> list[dict[str, Any]]:
    """Load world-model training samples from JSONL."""

    sample_path = Path(path)

    if not sample_path.exists():
        raise FileNotFoundError(f"Training sample file not found: {sample_path}")

    samples = []
    with sample_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    return samples


def samples_to_features_and_labels(
    samples: list[dict[str, Any]],
) -> tuple[list[list[float]], list[int]]:
    """Convert training samples into feature matrix and binary labels."""

    features = []
    labels = []

    for sample in samples:
        sample_features = sample.get("features", {})
        row = [
            float(bool(sample_features.get(feature_name, False)))
            for feature_name in FEATURE_NAMES
        ]
        features.append(row)
        labels.append(int(bool(sample.get("success", False))))

    return features, labels


def train_logistic_world_model(
    samples: list[dict[str, Any]],
) -> LogisticRegression:
    """Train a logistic regression mini world model."""

    features, labels = samples_to_features_and_labels(samples)

    if len(set(labels)) < 2:
        raise ValueError("Training data must contain both success and failure samples.")

    model = LogisticRegression(random_state=0)
    model.fit(features, labels)

    return model


def save_model_checkpoint(
    model: LogisticRegression,
    output_path: str | Path,
) -> None:
    """Save a trained model checkpoint."""

    path = Path(output_path)
    ensure_dir(path.parent)

    checkpoint = {
        "model_type": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "model": model,
    }

    with path.open("wb") as f:
        pickle.dump(checkpoint, f)


def load_model_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trained model checkpoint."""

    checkpoint_path = Path(path)

    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)

    return checkpoint
