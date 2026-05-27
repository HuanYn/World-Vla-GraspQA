from pathlib import Path

from world_vla_graspqa.utils.summary import (
    collect_result_paths,
    summarize_result,
    write_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "outputs" / "dummy_pipeline"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "dummy_pipeline_summary.csv"


def main() -> None:
    result_paths = collect_result_paths(RESULT_ROOT)
    rows = [summarize_result(path, PROJECT_ROOT) for path in result_paths]

    write_summary(rows, SUMMARY_PATH)

    print(f"[Summary] Found {len(rows)} result files.")
    print(f"[Summary] Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
