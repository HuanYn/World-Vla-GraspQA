from world_vla_graspqa.graspqa.dummy_graspqa import DummyGraspQA


def test_dummy_graspqa_selects_graspable_object_from_instruction():
    objects = [
        {
            "name": "red cube",
            "graspable": True,
        },
        {
            "name": "blue bowl",
            "graspable": False,
        },
    ]

    instruction = "Pick up the red cube and place it into the blue bowl."
    question = "Which object should the robot grasp?"

    graspqa = DummyGraspQA()
    answer = graspqa.answer(
        question=question,
        objects=objects,
        instruction=instruction,
    )

    assert answer == "red cube"
