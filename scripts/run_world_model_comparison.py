import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "config": "configs/dummy_pipeline.yaml",
        "run_name": "cube_empirical_wm_batch",
    },
    {
        "config": "configs/dummy_pipeline_cube_dummywm.yaml",
        "run_name": "cube_dummy_wm_batch",
    },
    {
        "config": "configs/dummy_pipeline_banana.yaml",
        "run_name": "banana_empirical_wm_batch",
    },
    {
        "config": "configs/dummy_pipeline_banana_dummywm.yaml",
        "run_name": "banana_dummy_wm_batch",
    },
]


def run_command(command: list[str]) -> None:
    """Run a command from the project root."""

    print(f"[WorldModelComparison] Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    for experiment in EXPERIMENTS:
        run_command(
            [
                sys.executable,
                "scripts/run_dummy_pipeline.py",
                "--config",
                experiment["config"],
                "--run-name",
                experiment["run_name"],
            ]
        )

    run_command(
        [
            sys.executable,
            "scripts/summarize_dummy_results.py",
        ]
    )

    summary_path = PROJECT_ROOT / "outputs" / "dummy_pipeline_summary.csv"
    print(f"[WorldModelComparison] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
