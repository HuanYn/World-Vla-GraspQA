from pathlib import Path

from world_vla_graspqa.action.dummy_action_generator import DummyActionGenerator
from world_vla_graspqa.graspqa.dummy_graspqa import DummyGraspQA
from world_vla_graspqa.perception.dummy_perception import DummyPerception
from world_vla_graspqa.planner.dummy_planner import DummyPlanner
from world_vla_graspqa.utils.config import load_yaml_config
from world_vla_graspqa.utils.io import save_json
from world_vla_graspqa.utils.logger import log_step
from world_vla_graspqa.world_model.dummy_world_model import DummyWorldModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "dummy_pipeline.yaml"
    config = load_yaml_config(config_path)

    instruction = config["scene"]["instruction"]
    objects = config["perception"]["objects"]
    question = config["graspqa"]["question"]

    log_step("Observation", "Loaded dummy scene.")
    log_step("Instruction", instruction)

    perception = DummyPerception(objects=objects)
    detected_objects = perception.detect_objects()

    graspqa = DummyGraspQA()
    target_object = graspqa.answer(
        question=question,
        objects=detected_objects,
        instruction=instruction,
    )

    action_generator = DummyActionGenerator()
    candidate_actions = action_generator.generate(target_object=target_object)

    world_model = DummyWorldModel()
    scored_actions = world_model.predict(candidate_actions)

    planner = DummyPlanner()
    best_action = planner.select_action(scored_actions)

    result = {
        "instruction": instruction,
        "question": question,
        "detected_objects": detected_objects,
        "target_object": target_object,
        "candidate_actions": candidate_actions,
        "scored_actions": scored_actions,
        "best_action": best_action,
    }

    output_path = PROJECT_ROOT / "outputs" / "dummy_pipeline_result.json"
    save_json(result, output_path)

    log_step(
        "Result",
        (
            "Dummy pipeline finished successfully. "
            f"Best action={best_action['name']}, "
            f"pose={best_action['gripper_pose']}, "
            f"predicted_success={best_action['predicted_success']}"
        ),
    )
    log_step("Output", f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
