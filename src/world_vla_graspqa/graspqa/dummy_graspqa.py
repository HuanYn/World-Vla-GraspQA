from typing import Any

from world_vla_graspqa.utils.logger import log_step


class DummyGraspQA:
    """A dummy grasping question-answering module."""

    def answer(
        self, question: str, objects: list[dict[str, Any]], instruction: str
    ) -> str:
        log_step("GraspQA", f"Question: {question}")

        for obj in objects:
            if obj.get("graspable") and obj["name"] in instruction:
                answer = obj["name"]
                log_step("GraspQA", f"Answer: {answer}")
                return answer

        fallback = objects[0]["name"] if objects else "none"
        log_step("GraspQA", f"Answer: {fallback}")
        return fallback
