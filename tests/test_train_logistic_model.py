import pytest

from world_vla_graspqa.world_model.train_logistic_model import (
    FEATURE_NAMES,
    load_model_checkpoint,
    load_training_samples,
    samples_to_features_and_labels,
    save_model_checkpoint,
    train_logistic_world_model,
)


def make_samples():
    return [
        {
            "sample_id": "sample_000001",
            "success": True,
            "features": {
                "target_contains_cube": True,
                "target_contains_banana": False,
                "target_contains_bowl": False,
                "pose_is_top_down": True,
                "pose_is_left_side": False,
                "pose_is_right_side": False,
            },
        },
        {
            "sample_id": "sample_000002",
            "success": False,
            "features": {
                "target_contains_cube": True,
                "target_contains_banana": False,
                "target_contains_bowl": False,
                "pose_is_top_down": False,
                "pose_is_left_side": False,
                "pose_is_right_side": True,
            },
        },
    ]


def test_samples_to_features_and_labels():
    features, labels = samples_to_features_and_labels(make_samples())

    assert len(features) == 2
    assert len(features[0]) == len(FEATURE_NAMES)
    assert labels == [1, 0]


def test_train_logistic_world_model():
    model = train_logistic_world_model(make_samples())

    prediction = model.predict([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

    assert prediction[0] in [0, 1]


def test_train_logistic_world_model_requires_two_classes():
    samples = [
        {
            "success": True,
            "features": {
                "target_contains_cube": True,
                "pose_is_top_down": True,
            },
        }
    ]

    with pytest.raises(ValueError):
        train_logistic_world_model(samples)


def test_save_and_load_model_checkpoint(tmp_path):
    model = train_logistic_world_model(make_samples())
    output_path = tmp_path / "checkpoints" / "logistic_world_model.pkl"

    save_model_checkpoint(model, output_path)
    checkpoint = load_model_checkpoint(output_path)

    assert checkpoint["model_type"] == "logistic_regression"
    assert checkpoint["feature_names"] == FEATURE_NAMES
    assert checkpoint["model"] is not None


def test_load_training_samples_reads_jsonl(tmp_path):
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text(
        '{"sample_id": "sample_000001", "success": true, "features": {}}\n',
        encoding="utf-8",
    )

    samples = load_training_samples(sample_path)

    assert len(samples) == 1
    assert samples[0]["sample_id"] == "sample_000001"
