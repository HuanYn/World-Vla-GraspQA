from world_vla_graspqa.graspqa.vlm_graspqa import VLMGraspQA, VLMGraspQAResult


def make_scene_annotation():
    return {
        "scene_id": "test_scene",
        "objects": [
            {
                "object_id": "obj_001",
                "name": "red cube",
                "bbox": [10, 20, 30, 40],
                "attributes": {
                    "color": "red",
                    "shape": "cube",
                    "graspable": True,
                },
            },
            {
                "object_id": "obj_002",
                "name": "yellow banana",
                "bbox": [40, 50, 80, 90],
                "attributes": {
                    "color": "yellow",
                    "shape": "banana",
                    "graspable": True,
                },
            },
        ],
    }


def test_vlm_graspqa_answers_red_cube():
    graspqa = VLMGraspQA()

    result = graspqa.answer(
        instruction="Pick up the red cube and place it into the blue bowl.",
        question="Which object should the robot grasp?",
        scene_annotation=make_scene_annotation(),
    )

    assert isinstance(result, VLMGraspQAResult)
    assert result.target_object == "red cube"
    assert result.parse_success is True
    assert result.model_name == "dummy-vlm"
    assert "Scene:" in result.user_prompt
    assert "red cube" in result.user_prompt


def test_vlm_graspqa_answers_yellow_banana():
    graspqa = VLMGraspQA()

    result = graspqa.answer(
        instruction="Pick up the yellow banana and place it into the blue bowl.",
        question="Which object should the robot grasp?",
        scene_annotation=make_scene_annotation(),
    )

    assert result.target_object == "yellow banana"
    assert result.parse_success is True


def test_vlm_graspqa_result_to_dict():
    graspqa = VLMGraspQA()

    result = graspqa.answer(
        instruction="Pick up the red cube and place it into the blue bowl.",
        question="Which object should the robot grasp?",
        scene_annotation=make_scene_annotation(),
    )

    result_dict = result.to_dict()

    assert result_dict["target_object"] == "red cube"
    assert result_dict["raw_response"] == "red cube"
    assert result_dict["parse_success"] is True
    assert result_dict["model_name"] == "dummy-vlm"
    assert "system_prompt" in result_dict
    assert "user_prompt" in result_dict
