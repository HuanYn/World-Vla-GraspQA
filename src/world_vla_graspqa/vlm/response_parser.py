import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedGraspQAResponse:
    """Parsed result from a GraspQA model response."""

    target_object: str
    raw_response: str
    parse_success: bool


def parse_graspqa_response(
    raw_response: str, candidate_objects: list[str]
) -> ParsedGraspQAResponse:
    """Parse a raw VLM response into a target object name."""

    text = raw_response.strip()
    candidate_objects_lower = {name.lower(): name for name in candidate_objects}

    json_target = _try_parse_json_target(text)
    if json_target is not None:
        normalized = _match_candidate(json_target, candidate_objects_lower)
        if normalized is not None:
            return ParsedGraspQAResponse(
                target_object=normalized,
                raw_response=raw_response,
                parse_success=True,
            )

    normalized = _match_candidate(text, candidate_objects_lower)
    if normalized is not None:
        return ParsedGraspQAResponse(
            target_object=normalized,
            raw_response=raw_response,
            parse_success=True,
        )

    return ParsedGraspQAResponse(
        target_object="unknown",
        raw_response=raw_response,
        parse_success=False,
    )


def _try_parse_json_target(text: str) -> str | None:
    """Try to extract target_object from a JSON response."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    target = data.get("target_object")
    if not isinstance(target, str):
        return None

    return target


def _match_candidate(
    text: str,
    candidate_objects_lower: dict[str, str],
) -> str | None:
    """Match candidate object names from response text."""

    text_lower = text.lower()

    for candidate_lower, original_name in candidate_objects_lower.items():
        if candidate_lower == text_lower:
            return original_name

    for candidate_lower, original_name in candidate_objects_lower.items():
        if candidate_lower in text_lower:
            return original_name

    return None
