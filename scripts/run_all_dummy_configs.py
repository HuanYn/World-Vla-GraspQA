import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DUMMY_RUNS = [
    {
        "config": "configs/dummy_pipeline.yaml",
        "run_name": "cube_batch",
    },
    {
        "config": "configs/dummy_pipeline_banana.yaml",
        "run_name": "banana_batch",
    },
]


def run_command(command: list[str]) -> None:
    """Run a shell command and fail fast if it fails."""

    print(f"[Batch] Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    for run in DUMMY_RUNS:
        command = [
            sys.executable,
            "scripts/run_dummy_pipeline.py",
            "--config",
            run["config"],
            "--run-name",
            run["run_name"],
        ]
        run_command(command)

    print("[Batch] All dummy configs finished successfully.")


if __name__ == "__main__":
    main()
