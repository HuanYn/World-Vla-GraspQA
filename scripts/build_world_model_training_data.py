import argparse
from pathlib import Path

from world_vla_graspqa.world_model.training_dataset import (
    build_training_dataset_from_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build world model training samples from outcome records."
    )
    parser.add_argument(
        "--outcome-dataset",
        type=str,
        default="data/action_outcomes/dummy_action_outcomes.json",
        help="Path to the static action-outcome dataset.",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default="outputs/action_outcomes/feedback_records.jsonl",
        help="Path to closed-loop feedback JSONL records.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/world_model_training/training_samples.jsonl",
        help="Path to save world model training samples.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    outcome_dataset_path = resolve_project_path(args.outcome_dataset)
    feedback_path = resolve_project_path(args.feedback)
    output_path = resolve_project_path(args.output)

    samples = build_training_dataset_from_paths(
        outcome_dataset_path=outcome_dataset_path,
        feedback_path=feedback_path,
        output_path=output_path,
    )

    print("[WorldModelTrainingData] Outcome dataset:", outcome_dataset_path)
    print("[WorldModelTrainingData] Feedback records:", feedback_path)
    print("[WorldModelTrainingData] Saved samples to:", output_path)
    print("[WorldModelTrainingData] Num samples:", len(samples))


if __name__ == "__main__":
    main()
