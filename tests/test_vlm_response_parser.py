from world_vla_graspqa.vlm.response_parser import (
    ParsedGraspQAResponse,
    parse_graspqa_response,
)


def test_parse_exact_object_name():
    result = parse_graspqa_response(
        raw_response="red cube",
        candidate_objects=["red cube", "yellow banana"],
    )

    assert isinstance(result, ParsedGraspQAResponse)
    assert result.target_object == "red cube"
    assert result.parse_success is True


def test_parse_sentence_response():
    result = parse_graspqa_response(
        raw_response="The robot should grasp the yellow banana.",
        candidate_objects=["red cube", "yellow banana"],
    )

    assert result.target_object == "yellow banana"
    assert result.parse_success is True


def test_parse_json_response():
    result = parse_graspqa_response(
        raw_response='{"target_object": "red cube"}',
        candidate_objects=["red cube", "yellow banana"],
    )

    assert result.target_object == "red cube"
    assert result.parse_success is True


def test_parse_is_case_insensitive():
    result = parse_graspqa_response(
        raw_response="RED CUBE",
        candidate_objects=["red cube", "yellow banana"],
    )

    assert result.target_object == "red cube"
    assert result.parse_success is True


def test_parse_unknown_response():
    result = parse_graspqa_response(
        raw_response="I am not sure.",
        candidate_objects=["red cube", "yellow banana"],
    )

    assert result.target_object == "unknown"
    assert result.parse_success is False
