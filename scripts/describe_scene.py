import argparse
from pathlib import Path

from world_vla_graspqa.utils.scene import load_scene_annotation
from world_vla_graspqa.utils.prompt import build_scene_description

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe a scene annotation.")
    parser.add_argument(
        "--annotation",
        type=str,
        default="data/annotations/sample_scenes/tabletop_dummy_scene.json",
        help="Path to the scene annotation JSON.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    annotation_path = resolve_project_path(args.annotation)
    annotation = load_scene_annotation(annotation_path)
    scene_description = build_scene_description(annotation)

    print(scene_description)


if __name__ == "__main__":
    main()
