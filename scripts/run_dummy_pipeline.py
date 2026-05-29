import argparse
from pathlib import Path

from world_vla_graspqa.action.dummy_action_generator import DummyActionGenerator
from world_vla_graspqa.graspqa.dummy_graspqa import DummyGraspQA
from world_vla_graspqa.graspqa.vlm_graspqa import VLMGraspQA
from world_vla_graspqa.perception.dummy_perception import DummyPerception
from world_vla_graspqa.planner.dummy_planner import DummyPlanner
from world_vla_graspqa.utils.config import load_yaml_config
from world_vla_graspqa.utils.image import get_image_info, load_image
from world_vla_graspqa.utils.io import create_run_dir, save_json, save_yaml
from world_vla_graspqa.utils.logger import log_step
from world_vla_graspqa.utils.scene import get_scene_objects, load_scene_annotation
from world_vla_graspqa.world_model.dummy_world_model import DummyWorldModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the dummy World-VLA-GraspQA pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dummy_pipeline.yaml",
        help="Path to the pipeline config file.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="dummy",
        help="Name suffix for the output run directory.",
    )
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if needed."""

    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def answer_graspqa(
    mode: str,
    instruction: str,
    question: str,
    detected_objects: list[dict],
    scene_annotation: dict,
    image_path: Path,
) -> tuple[str, dict]:
    """Answer the GraspQA question with either rule-based or VLM-style backend."""

    if mode == "rule":
        graspqa = DummyGraspQA()
        target_object = graspqa.answer(
            question=question,
            objects=detected_objects,
            instruction=instruction,
        )
        return target_object, {
            "mode": mode,
            "raw_response": target_object,
            "parse_success": True,
        }

    if mode == "dummy_vlm":
        graspqa = VLMGraspQA()
        parsed_response = graspqa.answer(
            instruction=instruction,
            question=question,
            scene_annotation=scene_annotation,
            image_path=image_path,
        )
        return parsed_response.target_object, {
            "mode": mode,
            "raw_response": parsed_response.raw_response,
            "parse_success": parsed_response.parse_success,
        }

    raise ValueError(f"Unsupported GraspQA mode: {mode}")


def main() -> None:
    args = parse_args()

    config_path = resolve_project_path(args.config)
    config = load_yaml_config(config_path)

    instruction = config["scene"]["instruction"]
    image_path = resolve_project_path(config["scene"]["image_path"])
    annotation_path = resolve_project_path(config["scene"]["annotation_path"])
    image = load_image(image_path)
    image_info = get_image_info(image, image_path)
    scene_annotation = load_scene_annotation(annotation_path)
    objects = get_scene_objects(scene_annotation)
    question = config["graspqa"]["question"]
    graspqa_mode = config["graspqa"].get("mode", "rule")

    run_dir = create_run_dir(
        output_root=PROJECT_ROOT / "outputs" / "dummy_pipeline",
        run_name=args.run_name,
    )
    save_yaml(config, run_dir / "config.yaml")

    log_step("Config", f"Loaded config from {config_path}")
    log_step("Observation", f"Loaded image from {image_path}")
    log_step("Observation", f"Loaded annotation from {annotation_path}")
    log_step(
        "Observation",
        (
            f"Image size: {image_info['width']}x{image_info['height']}, "
            f"mode={image_info['mode']}"
        ),
    )
    log_step("Instruction", instruction)
    log_step("GraspQA", f"Using mode: {graspqa_mode}")

    perception = DummyPerception(objects=objects)
    detected_objects = perception.detect_objects()

    target_object, graspqa_result = answer_graspqa(
        mode=graspqa_mode,
        instruction=instruction,
        question=question,
        detected_objects=detected_objects,
        scene_annotation=scene_annotation,
        image_path=image_path,
    )
    log_step("GraspQA", f"Answer: {target_object}")

    action_generator = DummyActionGenerator()
    candidate_actions = action_generator.generate(target_object=target_object)

    world_model = DummyWorldModel()
    scored_actions = world_model.predict(candidate_actions)

    planner = DummyPlanner()
    best_action = planner.select_action(scored_actions)

    result = {
        "config_path": str(config_path),
        "run_name": args.run_name,
        "instruction": instruction,
        "image_info": image_info,
        "scene_annotation": scene_annotation,
        "graspqa_result": graspqa_result,
        "question": question,
        "detected_objects": detected_objects,
        "target_object": target_object,
        "candidate_actions": candidate_actions,
        "scored_actions": scored_actions,
        "best_action": best_action,
    }

    result_path = run_dir / "result.json"
    save_json(result, result_path)

    log_step(
        "Result",
        (
            "Dummy pipeline finished successfully. "
            f"Best action={best_action['name']}, "
            f"pose={best_action['gripper_pose']}, "
            f"predicted_success={best_action['predicted_success']}"
        ),
    )
    log_step("Output", f"Saved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
