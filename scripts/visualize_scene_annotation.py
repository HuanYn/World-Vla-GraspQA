import argparse
from pathlib import Path

from world_vla_graspqa.utils.image import load_image
from world_vla_graspqa.utils.scene import load_scene_annotation
from world_vla_graspqa.utils.visualization import save_annotated_scene

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize scene annotation boxes.")
    parser.add_argument(
        "--image",
        type=str,
        default="data/raw/sample_scenes/tabletop_dummy_scene.png",
        help="Path to the scene image.",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="data/annotations/sample_scenes/tabletop_dummy_scene.json",
        help="Path to the scene annotation JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/visualizations/tabletop_dummy_scene_annotated.png",
        help="Path to save the annotated image.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    image_path = resolve_project_path(args.image)
    annotation_path = resolve_project_path(args.annotation)
    output_path = resolve_project_path(args.output)

    image = load_image(image_path)
    annotation = load_scene_annotation(annotation_path)

    save_annotated_scene(image, annotation, output_path)

    print(f"[Visualization] Loaded image from {image_path}")
    print(f"[Visualization] Loaded annotation from {annotation_path}")
    print(f"[Visualization] Saved annotated image to {output_path}")


if __name__ == "__main__":
    main()
