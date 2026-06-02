import pytest

from world_vla_graspqa.vlm.dummy_client import DummyVLMClient
from world_vla_graspqa.vlm.factory import build_vlm_client
from world_vla_graspqa.vlm.mock_real_client import MockRealVLMClient
from world_vla_graspqa.vlm.openai_client import OpenAIVLMClient


def test_build_vlm_client_defaults_to_dummy_client():
    client = build_vlm_client()

    assert isinstance(client, DummyVLMClient)
    assert client.model_name == "dummy-vlm"


def test_build_vlm_client_builds_dummy_client_from_config():
    client = build_vlm_client(
        {
            "provider": "dummy",
            "model_name": "test-dummy-vlm",
        }
    )

    assert isinstance(client, DummyVLMClient)
    assert client.model_name == "test-dummy-vlm"


def test_build_vlm_client_builds_mock_real_client_from_config():
    client = build_vlm_client(
        {
            "provider": "mock_real",
            "model_name": "test-mock-real-vlm",
        }
    )

    assert isinstance(client, MockRealVLMClient)
    assert client.model_name == "test-mock-real-vlm"


def test_build_vlm_client_builds_openai_client_from_config():
    client = build_vlm_client(
        {
            "provider": "openai",
            "model_name": "test-openai-vlm",
            "enabled": False,
        }
    )

    assert isinstance(client, OpenAIVLMClient)
    assert client.model_name == "test-openai-vlm"
    assert client.enabled is False


def test_build_vlm_client_raises_for_unsupported_provider():
    with pytest.raises(ValueError):
        build_vlm_client(
            {
                "provider": "unknown",
                "model_name": "unknown-model",
            }
        )
