from world_vla_graspqa.graspqa.prompt_builder import (
    GraspQAPrompt,
    build_graspqa_prompt,
)


def test_graspqa_prompt_to_text_combines_system_and_user_prompt():
    prompt = GraspQAPrompt(
        system_prompt="System message.",
        user_prompt="User message.",
    )

    text = prompt.to_text()

    assert "System message." in text
    assert "User message." in text


def test_build_graspqa_prompt_includes_scene_instruction_and_question():
    scene_description = (
        "Scene tabletop_dummy_scene contains 3 objects.\n"
        "- obj_001: red cube, color=red, shape=cube, graspable=True"
    )
    instruction = "Pick up the red cube and place it into the blue bowl."
    question = "Which object should the robot grasp?"

    prompt = build_graspqa_prompt(
        scene_description=scene_description,
        instruction=instruction,
        question=question,
    )

    text = prompt.to_text()

    assert "robotic grasping assistant" in prompt.system_prompt
    assert "Scene:" in text
    assert "red cube" in text
    assert instruction in text
    assert question in text
    assert "Answer with the target object name only." in text
