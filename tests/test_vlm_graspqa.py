from world_vla_graspqa.graspqa.vlm_graspqa import VLMGraspQA


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

    assert result.target_object == "red cube"
    assert result.parse_success is True


def test_vlm_graspqa_answers_yellow_banana():
    graspqa = VLMGraspQA()

    result = graspqa.answer(
        instruction="Pick up the yellow banana and place it into the blue bowl.",
        question="Which object should the robot grasp?",
        scene_annotation=make_scene_annotation(),
    )

    assert result.target_object == "yellow banana"
    assert result.parse_success is True
