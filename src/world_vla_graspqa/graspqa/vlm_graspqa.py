from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_vla_graspqa.graspqa.prompt_builder import build_graspqa_prompt
from world_vla_graspqa.utils.prompt import build_scene_description
from world_vla_graspqa.utils.scene import get_scene_objects
from world_vla_graspqa.vlm.dummy_client import DummyVLMClient
from world_vla_graspqa.vlm.response_parser import (
    ParsedGraspQAResponse,
    parse_graspqa_response,
)


@dataclass(frozen=True)
class VLMGraspQAResult:
    """Full result returned by VLM-based GraspQA."""

    target_object: str
    raw_response: str
    parse_success: bool
    model_name: str
    system_prompt: str
    user_prompt: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""

        return {
            "target_object": self.target_object,
            "raw_response": self.raw_response,
            "parse_success": self.parse_success,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }


class VLMGraspQA:
    """A GraspQA module backed by a VLM-style client."""

    def __init__(self, client: DummyVLMClient | None = None) -> None:
        self.client = client or DummyVLMClient()

    def answer(
        self,
        instruction: str,
        question: str,
        scene_annotation: dict[str, Any],
        image_path: str | Path | None = None,
    ) -> VLMGraspQAResult:
        """Answer a grasping question using a VLM-style prompt and parser."""

        objects = get_scene_objects(scene_annotation)
        candidate_objects = [obj["name"] for obj in objects]
        scene_description = build_scene_description(scene_annotation)

        prompt = build_graspqa_prompt(
            scene_description=scene_description,
            instruction=instruction,
            question=question,
        )

        response = self.client.generate(
            prompt=prompt.to_text(),
            image_path=image_path,
        )

        parsed_response: ParsedGraspQAResponse = parse_graspqa_response(
            raw_response=response.text,
            candidate_objects=candidate_objects,
        )

        return VLMGraspQAResult(
            target_object=parsed_response.target_object,
            raw_response=parsed_response.raw_response,
            parse_success=parsed_response.parse_success,
            model_name=response.model_name,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )
