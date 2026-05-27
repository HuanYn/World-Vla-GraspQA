import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_run_dir(output_root: str | Path, run_name: str | None = None) -> Path:
    """Create a timestamped directory for one experiment run."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_name:
        directory_name = f"{timestamp}_{run_name}"
    else:
        directory_name = timestamp

    run_dir = Path(output_root) / directory_name
    ensure_dir(run_dir)

    return run_dir


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save a dictionary as a JSON file."""

    path = Path(output_path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_yaml(data: dict[str, Any], output_path: str | Path) -> None:
    """Save a dictionary as a YAML file."""

    path = Path(output_path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
