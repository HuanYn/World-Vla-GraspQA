import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "config": "configs/dummy_pipeline_cube_dummywm.yaml",
        "run_name": "cube_dummy_wm_three_way",
    },
    {
        "config": "configs/dummy_pipeline.yaml",
        "run_name": "cube_empirical_wm_three_way",
    },
    {
        "config": "configs/dummy_pipeline_learned_wm.yaml",
        "run_name": "cube_learned_wm_three_way",
    },
]


def run_command(command: list[str]) -> None:
    """Run a command from the project root."""

    print(f"[WorldModelThreeWayComparison] Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    run_command(
        [
            sys.executable,
            "scripts/build_world_model_training_data.py",
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/train_logistic_world_model.py",
        ]
    )

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
    print(f"[WorldModelThreeWayComparison] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
