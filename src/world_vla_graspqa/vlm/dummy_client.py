from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VLMResponse:
    """A simple response object returned by a VLM client."""

    text: str
    model_name: str


class DummyVLMClient:
    """A dummy VLM client for testing the GraspQA interface."""

    def __init__(self, model_name: str = "dummy-vlm") -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> VLMResponse:
        """Generate a dummy response based on prompt text."""

        prompt_lower = prompt.lower()

        if (
            "yellow banana" in prompt_lower
            and "pick up the yellow banana" in prompt_lower
        ):
            answer = "yellow banana"
        elif "red cube" in prompt_lower and "pick up the red cube" in prompt_lower:
            answer = "red cube"
        else:
            answer = "unknown"

        return VLMResponse(
            text=answer,
            model_name=self.model_name,
        )
