from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VLMResponse:
    """A simple response object returned by a VLM client."""

    text: str
    model_name: str


class BaseVLMClient(ABC):
    """Base interface for all VLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> VLMResponse:
        """Generate a text response from a prompt and optional image path."""
