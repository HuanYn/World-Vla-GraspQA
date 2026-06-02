from typing import Any

from world_vla_graspqa.vlm.base_client import BaseVLMClient
from world_vla_graspqa.vlm.dummy_client import DummyVLMClient
from world_vla_graspqa.vlm.mock_real_client import MockRealVLMClient
from world_vla_graspqa.vlm.openai_client import OpenAIVLMClient


def build_vlm_client(config: dict[str, Any] | None = None) -> BaseVLMClient:
    """Build a VLM client from config."""

    config = config or {}
    provider = config.get("provider", "dummy")
    model_name = config.get("model_name")

    if provider == "dummy":
        return DummyVLMClient(model_name=model_name or "dummy-vlm")

    if provider == "mock_real":
        return MockRealVLMClient(model_name=model_name or "mock-real-vlm")

    if provider == "openai":
        return OpenAIVLMClient(
            model_name=model_name or "gpt-4o-mini",
            enabled=bool(config.get("enabled", False)),
        )

    raise ValueError(f"Unsupported VLM provider: {provider}")
