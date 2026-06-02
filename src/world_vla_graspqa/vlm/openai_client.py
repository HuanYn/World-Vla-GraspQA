from pathlib import Path

from world_vla_graspqa.vlm.base_client import BaseVLMClient, VLMResponse
from world_vla_graspqa.vlm.errors import VLMClientNotEnabledError


class OpenAIVLMClient(BaseVLMClient):
    """OpenAI VLM client placeholder.

    This class defines the interface for a future real OpenAI VLM backend.
    It does not call the OpenAI API yet. The goal is to keep the project
    pipeline stable before adding external API dependencies.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        enabled: bool = False,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> VLMResponse:
        """Generate a response from a prompt and optional image path."""

        if not self.enabled:
            raise VLMClientNotEnabledError(
                "OpenAIVLMClient is configured but not enabled. "
                "This is currently a placeholder for future real VLM integration. "
                "Use provider=dummy or provider=mock_real for local testing."
            )

        raise NotImplementedError("OpenAI API call is not implemented yet.")
