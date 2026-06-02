import argparse
from pathlib import Path

from world_vla_graspqa.world_model.train_logistic_model import (
    load_training_samples,
    save_model_checkpoint,
    samples_to_features_and_labels,
    train_logistic_world_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a logistic mini world model.")
    parser.add_argument(
        "--training-samples",
        type=str,
        default="outputs/world_model_training/training_samples.jsonl",
        help="Path to world model training samples.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/world_model_training/logistic_world_model.pkl",
        help="Path to save model checkpoint.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    training_samples_path = resolve_project_path(args.training_samples)
    output_path = resolve_project_path(args.output)

    samples = load_training_samples(training_samples_path)
    features, labels = samples_to_features_and_labels(samples)
    model = train_logistic_world_model(samples)
    train_accuracy = model.score(features, labels)

    save_model_checkpoint(model, output_path)

    print("[LogisticWorldModel] Training samples:", training_samples_path)
    print("[LogisticWorldModel] Num samples:", len(samples))
    print("[LogisticWorldModel] Train accuracy:", train_accuracy)
    print("[LogisticWorldModel] Saved checkpoint to:", output_path)


if __name__ == "__main__":
    main()
