from world_vla_graspqa.vlm.dummy_client import DummyVLMClient, VLMResponse


def test_dummy_vlm_client_returns_red_cube_for_red_cube_prompt():
    client = DummyVLMClient()
    prompt = "Instruction: Pick up the red cube and place it into the blue bowl."

    response = client.generate(prompt=prompt)

    assert isinstance(response, VLMResponse)
    assert response.text == "red cube"
    assert response.model_name == "dummy-vlm"


def test_dummy_vlm_client_returns_yellow_banana_for_banana_prompt():
    client = DummyVLMClient()
    prompt = "Instruction: Pick up the yellow banana and place it into the blue bowl."

    response = client.generate(prompt=prompt)

    assert response.text == "yellow banana"


def test_dummy_vlm_client_returns_unknown_for_unsupported_prompt():
    client = DummyVLMClient()
    prompt = "Instruction: Move the object."

    response = client.generate(prompt=prompt)

    assert response.text == "unknown"
