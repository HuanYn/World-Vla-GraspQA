from pathlib import Path

from world_vla_graspqa.vlm.base_client import BaseVLMClient, VLMResponse


class MockRealVLMClient(BaseVLMClient):
    """A mock real VLM client that returns natural-language responses.

    This client does not call an external model. It simulates the response style
    of a real VLM so that parser and pipeline logic can be tested before adding
    an actual backend such as OpenAI, Qwen-VL, or InternVL.
    """

    def __init__(self, model_name: str = "mock-real-vlm") -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> VLMResponse:
        """Generate a natural-language mock response."""

        prompt_lower = prompt.lower()

        if (
            "yellow banana" in prompt_lower
            and "pick up the yellow banana" in prompt_lower
        ):
            answer = "The robot should grasp the yellow banana."
        elif "red cube" in prompt_lower and "pick up the red cube" in prompt_lower:
            answer = "The robot should grasp the red cube."
        else:
            answer = "I am not sure which object the robot should grasp."

        return VLMResponse(
            text=answer,
            model_name=self.model_name,
        )
