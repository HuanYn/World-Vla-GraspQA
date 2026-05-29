import argparse
from pathlib import Path

from world_vla_graspqa.action.dummy_action_generator import DummyActionGenerator
from world_vla_graspqa.world_model.empirical_world_model import EmpiricalWorldModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run empirical world model demo.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/action_outcomes/dummy_action_outcomes.json",
        help="Path to the action-outcome dataset.",
    )
    parser.add_argument(
        "--target-object",
        type=str,
        default="red cube",
        help="Target object to generate candidate actions for.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    dataset_path = resolve_project_path(args.dataset)

    action_generator = DummyActionGenerator()
    actions = action_generator.generate(target_object=args.target_object)

    world_model = EmpiricalWorldModel.from_dataset(dataset_path)
    scored_actions = world_model.predict(actions)

    print(f"[EmpiricalWorldModel] Dataset: {dataset_path}")
    print(f"[EmpiricalWorldModel] Target object: {args.target_object}")

    for action in scored_actions:
        print(
            "[EmpiricalWorldModel] "
            f"{action['name']}, pose={action['gripper_pose']}, "
            f"predicted_success={action['predicted_success']}, "
            f"predicted_result={action['predicted_result']}"
        )


if __name__ == "__main__":
    main()
