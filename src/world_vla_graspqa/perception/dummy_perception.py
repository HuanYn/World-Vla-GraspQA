from typing import Any

from world_vla_graspqa.utils.logger import log_step


class DummyPerception:
    """A dummy perception module."""

    def __init__(self, objects: list[dict[str, Any]]) -> None:
        self.objects = objects

    def detect_objects(self) -> list[dict[str, Any]]:
        names = [obj["name"] for obj in self.objects]
        log_step("Perception", f"Detected objects: {', '.join(names)}")
        return self.objects
