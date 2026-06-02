import pytest

from world_vla_graspqa.vlm.base_client import BaseVLMClient
from world_vla_graspqa.vlm.errors import VLMClientNotEnabledError
from world_vla_graspqa.vlm.openai_client import OpenAIVLMClient


def test_openai_vlm_client_is_base_vlm_client():
    client = OpenAIVLMClient()

    assert isinstance(client, BaseVLMClient)


def test_openai_vlm_client_stores_model_name():
    client = OpenAIVLMClient(model_name="test-openai-vlm")

    assert client.model_name == "test-openai-vlm"


def test_openai_vlm_client_raises_when_not_enabled():
    client = OpenAIVLMClient(enabled=False)

    with pytest.raises(VLMClientNotEnabledError):
        client.generate(prompt="Which object should the robot grasp?")
