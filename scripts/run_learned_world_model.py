import argparse
from pathlib import Path

from world_vla_graspqa.action.dummy_action_generator import DummyActionGenerator
from world_vla_graspqa.world_model.learned_world_model import LearnedWorldModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned world model demo.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/world_model_training/logistic_world_model.pkl",
        help="Path to learned world model checkpoint.",
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

    checkpoint_path = resolve_project_path(args.checkpoint)

    action_generator = DummyActionGenerator()
    actions = action_generator.generate(target_object=args.target_object)

    world_model = LearnedWorldModel.from_checkpoint(checkpoint_path)
    scored_actions = world_model.predict(actions)

    print(f"[LearnedWorldModel] Checkpoint: {checkpoint_path}")
    print(f"[LearnedWorldModel] Target object: {args.target_object}")

    for action in scored_actions:
        print(
            "[LearnedWorldModel] "
            f"{action['name']}, pose={action['gripper_pose']}, "
            f"predicted_success={action['predicted_success']:.4f}, "
            f"predicted_result={action['predicted_result']}"
        )


if __name__ == "__main__":
    main()
