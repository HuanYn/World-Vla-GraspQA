import argparse
from pathlib import Path

from world_vla_graspqa.graspqa.vlm_graspqa import VLMGraspQA
from world_vla_graspqa.utils.config import load_yaml_config
from world_vla_graspqa.utils.scene import load_scene_annotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GraspQA only.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dummy_pipeline.yaml",
        help="Path to the pipeline config file.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    config_path = resolve_project_path(args.config)
    config = load_yaml_config(config_path)

    annotation_path = resolve_project_path(config["scene"]["annotation_path"])
    image_path = resolve_project_path(config["scene"]["image_path"])

    instruction = config["scene"]["instruction"]
    question = config["graspqa"]["question"]
    scene_annotation = load_scene_annotation(annotation_path)

    graspqa = VLMGraspQA()
    result = graspqa.answer(
        instruction=instruction,
        question=question,
        scene_annotation=scene_annotation,
        image_path=image_path,
    )

    print("[GraspQA] Config:", config_path)
    print("[GraspQA] Image:", image_path)
    print("[GraspQA] Annotation:", annotation_path)
    print("[GraspQA] Target object:", result.target_object)
    print("[GraspQA] Raw response:", result.raw_response)
    print("[GraspQA] Parse success:", result.parse_success)
    print("[GraspQA] Model:", result.model_name)


if __name__ == "__main__":
    main()
