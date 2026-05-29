import argparse
from pathlib import Path

from world_vla_graspqa.graspqa.prompt_builder import build_graspqa_prompt
from world_vla_graspqa.utils.config import load_yaml_config
from world_vla_graspqa.utils.prompt import build_scene_description
from world_vla_graspqa.utils.scene import get_scene_objects, load_scene_annotation
from world_vla_graspqa.vlm.dummy_client import DummyVLMClient
from world_vla_graspqa.vlm.response_parser import parse_graspqa_response

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dummy VLM GraspQA with parser.")
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

    image_path = resolve_project_path(config["scene"]["image_path"])
    annotation_path = resolve_project_path(config["scene"]["annotation_path"])

    annotation = load_scene_annotation(annotation_path)
    objects = get_scene_objects(annotation)
    candidate_objects = [obj["name"] for obj in objects]

    scene_description = build_scene_description(annotation)
    instruction = config["scene"]["instruction"]
    question = config["graspqa"]["question"]

    prompt = build_graspqa_prompt(
        scene_description=scene_description,
        instruction=instruction,
        question=question,
    )

    client = DummyVLMClient()
    response = client.generate(
        prompt=prompt.to_text(),
        image_path=image_path,
    )
    parsed_response = parse_graspqa_response(
        raw_response=response.text,
        candidate_objects=candidate_objects,
    )

    print("[DummyVLM] Model:", response.model_name)
    print("[DummyVLM] Raw answer:", response.text)
    print("[Parser] Target object:", parsed_response.target_object)
    print("[Parser] Parse success:", parsed_response.parse_success)


if __name__ == "__main__":
    main()
