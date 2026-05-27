from pathlib import Path

from world_vla_graspqa.utils.config import load_yaml_config


def test_load_dummy_pipeline_config():
    config_path = Path("configs/dummy_pipeline.yaml")
    config = load_yaml_config(config_path)

    assert config["project"]["name"] == "World-VLA-GraspQA"
    assert config["project"]["stage"] == "dummy_pipeline"
    assert config["scene"]["instruction"]
    assert len(config["perception"]["objects"]) == 3
