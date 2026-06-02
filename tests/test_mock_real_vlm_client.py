from world_vla_graspqa.vlm.base_client import BaseVLMClient, VLMResponse
from world_vla_graspqa.vlm.mock_real_client import MockRealVLMClient


def test_mock_real_vlm_client_is_base_vlm_client():
    client = MockRealVLMClient()

    assert isinstance(client, BaseVLMClient)


def test_mock_real_vlm_client_returns_sentence_for_red_cube_prompt():
    client = MockRealVLMClient()
    prompt = "Instruction: Pick up the red cube and place it into the blue bowl."

    response = client.generate(prompt=prompt)

    assert isinstance(response, VLMResponse)
    assert response.text == "The robot should grasp the red cube."
    assert response.model_name == "mock-real-vlm"


def test_mock_real_vlm_client_returns_sentence_for_banana_prompt():
    client = MockRealVLMClient()
    prompt = "Instruction: Pick up the yellow banana and place it into the blue bowl."

    response = client.generate(prompt=prompt)

    assert response.text == "The robot should grasp the yellow banana."


def test_mock_real_vlm_client_returns_uncertain_sentence_for_unknown_prompt():
    client = MockRealVLMClient()
    prompt = "Instruction: Move the object."

    response = client.generate(prompt=prompt)

    assert "not sure" in response.text
