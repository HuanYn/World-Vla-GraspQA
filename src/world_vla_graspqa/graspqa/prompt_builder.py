from dataclasses import dataclass


@dataclass(frozen=True)
class GraspQAPrompt:
    """A structured prompt for grasping question answering."""

    system_prompt: str
    user_prompt: str

    def to_text(self) -> str:
        """Convert the structured prompt into one plain text prompt."""

        return f"{self.system_prompt}\n\n{self.user_prompt}"


def build_graspqa_prompt(
    scene_description: str,
    instruction: str,
    question: str,
) -> GraspQAPrompt:
    """Build a prompt for grasping question answering."""

    system_prompt = (
        "You are a robotic grasping assistant. "
        "You answer grasping questions based on the scene and instruction."
    )

    user_prompt = "\n".join(
        [
            "Your task is to identify the object that the robot should grasp.",
            "",
            "Scene:",
            scene_description,
            "",
            "Instruction:",
            instruction,
            "",
            "Question:",
            question,
            "",
            "Answer with the target object name only.",
        ]
    )

    return GraspQAPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
